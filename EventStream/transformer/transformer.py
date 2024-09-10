"""The internal transformer module code.

TODO(mmcdermott): Can use `transformers.apply_chunking_to_forward` to save memory.

Based on
https://raw.githubusercontent.com/huggingface/transformers/\
e3cc4487fe66e03ec85970ea2db8e5fb34c455f4/src/transformers/models/gpt_neo/modeling_gpt_neo.py
"""  # noqa E501

import math
import os

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

from ..data.data_embedding_layer import DataEmbeddingLayer, MeasIndexGroupOptions
from ..data.types import PytorchBatch
from .config import StructuredEventProcessingMode, StructuredTransformerConfig
from .model_output import TransformerOutputWithPast
from .structured_attention import StructuredAttention
from ..data.vocabulary import VocabularyConfig

from typing import Union, Dict

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def expand_mask(mask: torch.BoolTensor, dtype: torch.dtype) -> torch.Tensor:
    """Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, 1, seq_len]` and converts to float.

    This enables broadcasting to [bsz, num_heads, from_seq_len, to_seq_len] by converting the size [bsz,
    seq_len] to [bsz, 1, 1, seq_len] and converts from a boolean form to an attention weights masking form,
    which has 0 where the original mask was True and the minimum possible floating point expressible value
    where it was False.

    Args:
        mask: The event presence/absence mask of shape `[bsz, seq_len]`.
        dtype: The target dtype of the attention mask.

    Returns:
        The passed event indicator mask reshaped and type converted, unless mask is `None` in which case
        returns `None`.

    Examples:
        >>> import torch
        >>> assert expand_mask(None, None) is None
        >>> mask = torch.BoolTensor([
        ...     [True, True, False, False],
        ...     [True, True, True, False],
        ... ])
        >>> dtype = torch.float16
        >>> print(expand_mask(mask, dtype))
        tensor([[[[    -0.,     -0., -65504., -65504.]]],
        <BLANKLINE>
        <BLANKLINE>
                [[[    -0.,     -0.,     -0., -65504.]]]], dtype=torch.float16)
    """
    if mask is None:
        return None

    # We create a 3D attention mask from a 2D tensor mask.
    # Sizes are [batch_size, 1, 1, to_seq_length]
    # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
    # this attention mask is more simple than the triangular masking of causal attention
    # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
    attention_mask = mask[:, None, None, :]

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and the dtype's smallest value for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    attention_mask = attention_mask.to(dtype=dtype)  # fp16 compatibility
    attention_mask = (1.0 - attention_mask) * torch.finfo(dtype).min

    return attention_mask


class InnerSelfAttention(nn.Module):
    """This class implements the inner self-attention mechanism.

    This involves
    performing the self-attention operation and returning the result along with
    some optional additional outputs. The constructor of this class accepts three arguments, which determine
    the configuration of the self-attention mechanism.

    Args:
        config: An instance of StructuredTransformerConfig which contains various
            configuration parameters.
        attention_type: A string indicating the type of attention to be applied.
            Currently, only "local" is implemented.
        window_size: An integer specifying the size of the attention window.

    Raises:
        ValueError: If the product of `num_heads` and `head_dim` from the config
            does not match `embed_dim`.
    """

    def __init__(
        self,
        config: StructuredTransformerConfig,
        attention_type: str,
        window_size: int,
    ):
        super().__init__()
        self.config = config  # Store the config as an attribute
        self.attention_type = attention_type
        self.window_size = window_size
        self.causal = attention_type in ["local", "global"]  # Set causal based on attention type
        self.use_flash_attention = config.use_flash_attention

        max_seq_len = config.max_seq_len
        self.window_size = window_size
        bias = torch.tril(torch.ones((max_seq_len, max_seq_len), dtype=torch.uint8)).view(
            1, 1, max_seq_len, max_seq_len
        )

        # local causal self attention is a sliding window where each token can only attend to the previous
        # window_size tokens. This is implemented by updating the causal mask such that for each token
        # all other tokens are masked except the previous window_size tokens.
        if attention_type == "local":
            bias = torch.bitwise_xor(bias, torch.tril(bias, -window_size))

        self.register_buffer("bias", bias)
        self.register_buffer("masked_bias", torch.tensor(-1e9))

        self.attn_dropout = nn.Dropout(float(config.attention_dropout))
        self.resid_dropout = nn.Dropout(float(config.resid_dropout))

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and "
                f"`num_heads`: {self.num_heads})."
            )

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def _split_heads(self, tensor, num_heads, attn_head_size):
        if tensor.dim() == 3:
            # Handle 3D tensors (batch_size, seq_length, hidden_size)
            new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
            tensor = tensor.view(new_shape)
            return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        elif tensor.dim() == 4:
            # Handle 4D tensors (batch_size, seq_length, dep_graph_len, hidden_size)
            batch_size, seq_len, dep_graph_len, hidden_size = tensor.shape
            tensor = tensor.view(batch_size, seq_len * dep_graph_len, num_heads, attn_head_size)
            return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length * dep_graph_len, head_features)
        else:
            raise ValueError(f"Unexpected tensor shape: {tensor.shape}")

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        if tensor.dim() == 4:
            # Handle 4D tensors (batch, head, seq_length, head_features)
            tensor = tensor.permute(0, 2, 1, 3).contiguous()
            new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
            return tensor.view(new_shape)
        else:
            raise ValueError(f"Unexpected tensor shape: {tensor.shape}")

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        """Performs the attention operation.

        Args:
            query: The query tensor.
            key: The key tensor.
            value: The value tensor.
            attention_mask: A mask to be applied on the attention weights.
            head_mask: A mask to be applied on the attention heads.

        Returns:
            A tuple containing the output of the attention operation and the attention weights.
        """

        if self.use_flash_attention:
            # Reshape inputs for Flash Attention
            qkv = torch.stack([query, key, value], dim=2)
            qkv = qkv.transpose(0, 1).contiguous()  # [seqlen, bsz, 3, num_heads, head_dim]
            
            # Call Flash Attention
            context = flash_attn_qkvpacked_func(
                qkv,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                softmax_scale=None,  # Let the function use default scale
                causal=self.causal
            )
            
            # Reshape output
            context = context.transpose(0, 1)  # [bsz, seqlen, num_heads, head_dim]
            return context, None  # Flash Attention doesn't return attention weights
        else:
            # Keep the attention weights computation in fp32 to avoid overflow issues
            query = query.to(torch.float32)
            key = key.to(torch.float32)

            # query, key, and value are all of shape (batch, head, seq_length, head_features)

            attn_weights = torch.matmul(query, key.transpose(-1, -2))
            # attn_weights is of shape batch, head, query_seq_length, key_seq_length

            # Move the tensors to the appropriate device
            query = query.to(attn_weights.device)
            key = key.to(attn_weights.device)
            value = value.to(attn_weights.device)

            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].to(torch.bool)
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error:
            # `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)

            if attention_mask is not None:
                # Apply the attention mask
                attn_weights = attn_weights + attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1)
            attn_weights = attn_weights.to(value.dtype)
            attn_weights = self.attn_dropout(attn_weights)

            # Mask heads if we want to
            if head_mask is not None:
                attn_weights = attn_weights * head_mask

            attn_output = torch.matmul(attn_weights, value)

            return attn_output, attn_weights

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_past=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
        static_kv_first: bool = False,
    ):
        original_shape = hidden_states.shape
        if hidden_states.dim() == 4:
            # For 4D input, reshape to 3D for compatibility with linear layers
            batch_size, seq_len, dep_graph_len, hidden_size = hidden_states.shape
            hidden_states = hidden_states.view(-1, hidden_size)

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Reshape back to 4D if necessary
        if len(original_shape) == 4:
            query = query.view(*original_shape)
            key = key.view(*original_shape)
            value = value.view(*original_shape)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if static_kv_first:
            query = query[:, :, 1:, :]

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        present = (key, value) if use_cache else None

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        # Reshape output back to 4D if necessary
        if len(original_shape) == 4:
            attn_output = attn_output.view(*original_shape)

        outputs = {"present_key_value": present}
        if output_attentions:
            outputs["attn_weights"] = attn_weights

        return attn_output, outputs

class CustomLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        try:
            logger.debug(f"CustomLayerNorm input shape: {x.shape}")
            logger.debug(f"CustomLayerNorm input dtype: {x.dtype}")
            logger.debug(f"CustomLayerNorm input device: {x.device}")
            logger.debug(f"CustomLayerNorm normalized_shape: {self.normalized_shape}")

            original_shape = x.shape
            original_dtype = x.dtype

            if x.dim() == 4:
                # For 4D input, reshape to 3D
                batch_size, seq_len, dep_graph_len, hidden_size = x.shape
                x = x.contiguous().view(-1, hidden_size)
            elif x.dim() == 3:
                # For 3D input, reshape to 2D
                batch_size, seq_len, hidden_size = x.shape
                x = x.contiguous().view(-1, hidden_size)
            elif x.dim() != 2:
                raise ValueError(f"Expected 2D, 3D or 4D input, got {x.dim()}D")

            # Check for NaN or Inf values
            if torch.isnan(x).any():
                nan_indices = torch.nonzero(torch.isnan(x))
                logger.error(f"NaN values found at indices: {nan_indices}")
                raise ValueError("Input tensor contains NaN values")
            if torch.isinf(x).any():
                inf_indices = torch.nonzero(torch.isinf(x))
                logger.error(f"Inf values found at indices: {inf_indices}")
                raise ValueError("Input tensor contains Inf values")

            # Ensure the last dimension matches the normalized shape
            if x.size(-1) != self.normalized_shape[-1]:
                raise ValueError(f"Last dimension of input ({x.size(-1)}) doesn't match normalized_shape ({self.normalized_shape[-1]})")

            # Compute mean and variance
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, unbiased=False, keepdim=True)

            logger.debug(f"Mean shape: {mean.shape}, Var shape: {var.shape}")
            logger.debug(f"Mean min: {mean.min().item()}, max: {mean.max().item()}")
            logger.debug(f"Var min: {var.min().item()}, max: {var.max().item()}")

            # Normalize
            x = (x - mean) / torch.sqrt(var + self.eps)

            # Apply weight and bias
            if self.weight.dim() == 1:
                x = x * self.weight + self.bias
            else:
                x = x * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)

            # Reshape back to original shape
            x = x.view(original_shape)

            # Ensure output dtype matches input dtype
            x = x.to(original_dtype)

            logger.debug(f"CustomLayerNorm output shape: {x.shape}")
            logger.debug(f"CustomLayerNorm output min: {x.min().item()}, max: {x.max().item()}")

            return x
        except Exception as e:
            logger.exception(f"Error in CustomLayerNorm: {str(e)}")
            raise

class InnerAttention(nn.Module):
    def __init__(self, config: StructuredTransformerConfig, layer_id: int = 0, is_seq: bool = True):
        super().__init__()
        self.layer_id = layer_id
        self.is_seq = is_seq
        self.attention_layers = config.seq_attention_layers if is_seq else config.dep_graph_attention_layers
        self.attention_type = self.attention_layers[layer_id]
        if self.attention_type == "local":
            self.window_size = config.seq_window_size if is_seq else config.dep_graph_window_size
        else:
            self.window_size = None

        self.layer_norm = CustomLayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attention = InnerSelfAttention(
            config, attention_type=self.attention_type, window_size=self.window_size
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_past=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
        static_kv_first: bool = False,
    ):
        logger.debug(f"InnerAttention input hidden_states shape: {hidden_states.shape}")
        logger.debug(f"InnerAttention input hidden_states device: {hidden_states.device}")
        logger.debug(f"InnerAttention layer_norm weight device: {self.layer_norm.weight.device}")

        # Ensure hidden_states is on the correct device
        hidden_states = hidden_states.to(self.layer_norm.weight.device)

        # Apply layer norm
        normalized_hidden_states = self.layer_norm(hidden_states)

        logger.debug(f"InnerAttention normalized_hidden_states shape: {normalized_hidden_states.shape}")
        logger.debug(f"InnerAttention normalized_hidden_states device: {normalized_hidden_states.device}")

        # Adjust attention_mask if necessary
        if attention_mask is not None and attention_mask.shape[1] != normalized_hidden_states.shape[1]:
            logger.warning(f"Attention mask shape mismatch: {attention_mask.shape} vs {normalized_hidden_states.shape}")
            attention_mask = F.pad(attention_mask, (0, normalized_hidden_states.shape[1] - attention_mask.shape[1]))

        try:
            attn_output, outputs = self.attention(
                normalized_hidden_states,
                attention_mask=attention_mask,
                layer_past=layer_past,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                static_kv_first=static_kv_first,
            )
        except Exception as e:
            logger.error(f"Error during attention computation: {str(e)}")
            logger.error(f"normalized_hidden_states shape: {normalized_hidden_states.shape}")
            logger.error(f"attention_mask shape: {attention_mask.shape if attention_mask is not None else None}")
            raise

        logger.debug(f"InnerAttention attn_output shape: {attn_output.shape}")
        logger.debug(f"InnerAttention attn_output device: {attn_output.device}")

        return attn_output, outputs

    def _set_static_graph(self):
        if hasattr(self.attention, '_set_static_graph'):
            self.attention._set_static_graph()

class InnerMLP(nn.Module):
    """Applies a multilayer perceptron (MLP) to the `hidden_states`.

    Args:
        config: Configuration parameters for the structured transformer.
    """

    def __init__(self, config: StructuredTransformerConfig):
        super().__init__()
        embed_dim = config.hidden_size
        inner_dim = config.intermediate_size if config.intermediate_size is not None else 4 * embed_dim

        self.c_fc = nn.Linear(embed_dim, inner_dim)
        self.c_proj = nn.Linear(inner_dim, embed_dim)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(float(config.resid_dropout))

    def forward(self, hidden_states):
        """Conducts forward pass for the MLP.

        Args:
            hidden_states: Input tensor.

        Returns:
            Modified hidden states after applying MLP.
        """
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

    def _set_static_graph(self):
        pass  # No submodules to set static graph for
        
class InnerBlock(nn.Module):
    def __init__(self, config: StructuredTransformerConfig, layer_id: int, is_seq: bool, device=None):
        super().__init__()
        self.attn = InnerAttention(config, layer_id, is_seq)
        self.layer_norm1 = CustomLayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.layer_norm2 = CustomLayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = InnerMLP(config)

        if device is not None:
            self.to(device)

    def forward(self, hidden_states, attention_mask=None, layer_past=None, head_mask=None, use_cache=False, output_attentions=False, static_kv_first: bool = False):
        try:
            logger.debug(f"InnerBlock input hidden_states shape: {hidden_states.shape}")
            logger.debug(f"InnerBlock input hidden_states dtype: {hidden_states.dtype}")
            logger.debug(f"InnerBlock input hidden_states device: {hidden_states.device}")

            residual = hidden_states
            hidden_states = self.layer_norm1(hidden_states)
            logger.debug(f"After layer_norm1 shape: {hidden_states.shape}")
            
            attn_outputs = self.attn(
                hidden_states,
                attention_mask=attention_mask,
                layer_past=layer_past,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                static_kv_first=static_kv_first,
            )
            attn_output, outputs = attn_outputs
            logger.debug(f"After attention shape: {attn_output.shape}")
            
            hidden_states = attn_output + residual

            residual = hidden_states
            hidden_states = self.layer_norm2(hidden_states)
            logger.debug(f"After layer_norm2 shape: {hidden_states.shape}")
            
            feed_forward_hidden_states = self.mlp(hidden_states)
            logger.debug(f"After MLP shape: {feed_forward_hidden_states.shape}")
            
            hidden_states = residual + feed_forward_hidden_states

            if not use_cache:
                outputs.pop("present_key_value", None)
            
            logger.debug(f"InnerBlock output hidden_states shape: {hidden_states.shape}")
            return hidden_states, outputs
        except Exception as e:
            logger.exception(f"Error in InnerBlock forward pass: {str(e)}")
            raise

    def _set_static_graph(self):
        self.attn._set_static_graph()
        if hasattr(self.mlp, '_set_static_graph'):
            self.mlp._set_static_graph()

class StructuredTransformerBlock(nn.Module):
    """A block for structured attention with both sequential and dependency graph modules.

    Args:
        config: Configuration parameters for the structured transformer.
        layer_id: Unique identifier (depth index) for the layer.
    """

    def __init__(self, config: StructuredTransformerConfig, layer_id: int):
        super().__init__()

        if config.do_full_block_in_seq_attention:
            seq_block = InnerBlock(config, layer_id, is_seq=True)
        else:
            seq_block = InnerAttention(config, layer_id, is_seq=True)

        if config.do_full_block_in_dep_graph_attention:
            dep_graph_block = InnerBlock(config, layer_id, is_seq=False)
        else:
            dep_graph_block = InnerAttention(config, layer_id, is_seq=False)

        self.block = StructuredAttention(
            seq_module=seq_block,
            dep_graph_module=dep_graph_block,
        )
        self.layer_norm.weight.data = self.layer_norm.weight.data.to(config.device)
        self.layer_norm.bias.data = self.layer_norm.bias.data.to(config.device)

    def forward(
        self, *args, **kwargs
    ) -> tuple[torch.Tensor, dict[str, dict[str, torch.Tensor | None] | None]]:
        """Conducts the forward pass for the structured transformer block.

        Args:
            args: Variable length argument list.
            kwargs: Arbitrary keyword arguments.

        Returns:
            tuple: Modified input tensor and a dictionary containing present key-value pair and
            attention weights.
        """

        return self.block(*args, **kwargs)


class StructuredTransformerPreTrainedModel(PreTrainedModel):
    """The base pre-trained model class for Transformer models."""
    config_class = StructuredTransformerConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["StructuredTransformerBlock"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
            
    def _set_static_graph(self):
        for module in self.children():
            if hasattr(module, '_set_static_graph'):
                module._set_static_graph()    

def time_from_deltas(batch: PytorchBatch) -> torch.Tensor:
    """Given a batch of time deltas, compute the relative time-since-start for each event.

    Args:
        batch: The input batch

    Examples:
        >>> batch = PytorchBatch(
        ...     event_mask=torch.BoolTensor([
        ...         [True, True, True], [True, True, False]
        ...     ]),
        ...     time_delta=torch.Tensor([[1.0, 3.2, 0.0], [1.4, 0.0, 1.0]])
        ... )
        >>> print(time_from_deltas(batch))
        tensor([[0.0000, 1.0000, 4.2000],
                [0.0000, 1.4000, 1.4000]])
    """
    if not isinstance(batch, PytorchBatch):
        raise TypeError("Input 'batch' should be a PytorchBatch object.")

    t_deltas = batch["time_delta"]

    if batch.event_mask is not None:
        t_deltas = torch.where(batch.event_mask, t_deltas, torch.zeros_like(t_deltas))

    return torch.hstack([torch.zeros_like(t_deltas[:, :1]), t_deltas.cumsum(-1)[:, :-1]])


class LearnableFrequencySinusoidalTemporalPositionEncoding(torch.nn.Module):
    """A module for applying time-based position encodings to a PytorchBatch.

    Adapted from :footcite:t:`wang2021on` (`link`_).

    .. _link: https://openreview.net/pdf?id=onxoVA9FxMw

    .. footbibliography::

    Args:
        embedding_dim: The desired size of the output embedding. Unlike many position embedding
            implementations, this does not need to be even.
    """

    def __init__(
        self,
        embedding_dim: int,
        max_timepoint: float = 10000.0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        # div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(max_timepoint) / embedding_dim))

        size = math.ceil(embedding_dim / 2)
        div_term = torch.empty(
            size,
        )
        torch.nn.init.normal_(div_term)

        # We still want this to work for odd embedding dimensions, so we'll lop off the end of the cos
        # embedding. This is not a principled decision, but enabling odd embedding dimensions helps avoid edge
        # cases during hyperparameter tuning when searching over possible embedding spaces.
        if self.embedding_dim % 2 == 0:
            self.sin_div_term = torch.nn.Parameter(div_term, requires_grad=True)
            self.cos_div_term = torch.nn.Parameter(div_term, requires_grad=True)
        else:
            self.sin_div_term = torch.nn.Parameter(div_term, requires_grad=True)
            self.cos_div_term = torch.nn.Parameter(div_term[:-1], requires_grad=True)

    def forward(self, batch: PytorchBatch) -> torch.Tensor:
        """Forward pass.

        Args:
            batch: The input batch to process.

        Returns:
            The temporal position embeddings tensor of shape (bsz, seq_len)
        """

        t = time_from_deltas(batch) if batch.get("time", None) is None else batch["time"]
        bsz, seq_len = t.shape
        device = t.device

        # First, we go from deltas to time values and unsqueeze it for broadcasting through the hidden dim.
        t = t.unsqueeze(-1)

        # temporal_embeddings will be our output container.
        # It should have shape (batch size, sequence length, embedding dim), and be on the same device as the
        # timepoints.
        temporal_embeddings = torch.zeros(bsz, seq_len, self.embedding_dim, device=device)

        temporal_embeddings[:, :, 0::2] = torch.sin(t * self.sin_div_term.unsqueeze(0).unsqueeze(0))
        temporal_embeddings[:, :, 1::2] = torch.cos(t * self.cos_div_term.unsqueeze(0).unsqueeze(0))

        return temporal_embeddings


class TemporalPositionEncoding(torch.nn.Module):
    """A module for applying time-based position encodings to a PytorchBatch.

    Adapted from https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    Args:
        embedding_dim: The desired size of the output embedding. Unlike many position embedding
            implementations, this does not need to be even.
        max_timepoint: The maximum observed timepoint, used to initialize the frequency space.
    """

    def __init__(
        self,
        embedding_dim: int,
        max_timepoint: float = 10000.0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(max_timepoint) / embedding_dim))

        # We still want this to work for odd embedding dimensions, so we'll lop off the end of the cos
        # embedding. This is not a principled decision, but enabling odd embedding dimensions helps avoid edge
        # cases during hyperparameter tuning when searching over possible embedding spaces.
        if self.embedding_dim % 2 == 0:
            self.sin_div_term = torch.nn.Parameter(div_term, requires_grad=False)
            self.cos_div_term = torch.nn.Parameter(div_term, requires_grad=False)
        else:
            self.sin_div_term = torch.nn.Parameter(div_term, requires_grad=False)
            self.cos_div_term = torch.nn.Parameter(div_term[:-1], requires_grad=False)

    def forward(self, batch: PytorchBatch) -> torch.Tensor:
        """Forward pass.

        Args:
            batch: The input batch to process.

        Returns:
            The temporal position embeddings tensor of shape (bsz, seq_len)
        """

        t = time_from_deltas(batch) if batch.get("time", None) is None else batch["time"]
        bsz, seq_len = t.shape
        device = t.device

        # First, we go from deltas to time values and unsqueeze it for broadcasting through the hidden dim.
        t = t.unsqueeze(-1)

        # temporal_embeddings will be our output container.
        # It should have shape (batch size, sequence length, embedding dim), and be on the same device as the
        # timepoints.
        temporal_embeddings = torch.zeros(bsz, seq_len, self.embedding_dim, device=device)

        temporal_embeddings[:, :, 0::2] = torch.sin(t * self.sin_div_term.unsqueeze(0).unsqueeze(0))
        temporal_embeddings[:, :, 1::2] = torch.cos(t * self.cos_div_term.unsqueeze(0).unsqueeze(0))

        return temporal_embeddings

class ConditionallyIndependentPointProcessInputLayer(nn.Module):
    def __init__(self, config: StructuredTransformerConfig, vocab_sizes_by_measurement: dict[str, int], oov_index: int):
        super().__init__()
        self.config = config
        self.oov_index = oov_index
        self.use_addition_for_static = config.use_addition_for_static

        self.data_embedding_layer = DataEmbeddingLayer(
            n_total_embeddings=max(vocab_sizes_by_measurement.values()) + 125,
            out_dim=config.hidden_size,
            categorical_embedding_dim=config.categorical_embedding_dim,
            numerical_embedding_dim=config.numerical_embedding_dim,
            static_embedding_mode=config.static_embedding_mode,
            split_by_measurement_indices=None,
            do_normalize_by_measurement_index=config.do_normalize_by_measurement_index,
            static_weight=config.static_embedding_weight,
            dynamic_weight=config.dynamic_embedding_weight,
            categorical_weight=config.categorical_embedding_weight,
            numerical_weight=config.numerical_embedding_weight,
            oov_index=oov_index,
        )

        self.embedding_dropout = nn.Dropout(p=config.input_dropout)
        self.dynamic_values_encoder = nn.Linear(1, config.hidden_size)
        self.dynamic_values_norm = nn.BatchNorm1d(1)
        self.static_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.time_embedding = nn.Linear(1, config.hidden_size)

    def forward(self, batch: dict | torch.Tensor) -> torch.Tensor:
        if isinstance(batch, dict):
            dynamic_indices = batch['dynamic_indices']
            dynamic_values = batch.get('dynamic_values')
            dynamic_measurement_indices = batch.get('dynamic_measurement_indices')
            static_indices = batch.get('static_indices')
            time = batch.get('time')

            data_embed: torch.Tensor = self.data_embedding_layer({
                'dynamic_indices': dynamic_indices,
                'dynamic_values': dynamic_values,
                'dynamic_measurement_indices': dynamic_measurement_indices
            })
                
            # Ensure data_embed is 3D: [batch_size, seq_len, hidden_size]
            if len(data_embed.shape) == 2:
                batch_size, total_elements = data_embed.shape
                hidden_size = self.config.hidden_size
                seq_len = total_elements // hidden_size
                data_embed = data_embed.view(batch_size, seq_len, hidden_size)

            max_seq_len = data_embed.shape[1]

            # Process dynamic values
            if dynamic_values is not None:
                dynamic_values = dynamic_values[:, :max_seq_len]
                mask = ~torch.isnan(dynamic_values)
                valid_values = dynamic_values[mask]

                if valid_values.numel() > 0:
                    valid_values_normalized = self.dynamic_values_norm(valid_values.unsqueeze(1)).squeeze(1)
                    valid_values_embed = self.dynamic_values_encoder(valid_values_normalized.unsqueeze(-1))
                    valid_values_embed = valid_values_embed.to(dtype=data_embed.dtype)

                    batch_size, seq_len = dynamic_values.shape
                    hidden_size = self.config.hidden_size

                    reshaped_valid_values_embed = torch.zeros(batch_size, seq_len, hidden_size, device=valid_values_embed.device, dtype=valid_values_embed.dtype)
                    reshaped_valid_values_embed[mask] = valid_values_embed.squeeze(1)

                    data_embed = torch.where(mask.unsqueeze(-1), reshaped_valid_values_embed, data_embed)

            # Process static indices
            if static_indices is not None:
                static_embed = self.static_embedding(static_indices)
                if self.use_addition_for_static:
                    data_embed += static_embed.mean(dim=1).unsqueeze(1)
                else:
                    static_embed = static_embed.unsqueeze(1).expand(-1, max_seq_len, -1)
                    data_embed = torch.cat([data_embed, static_embed], dim=-1)

            # Process time
            if time is not None:
                time = time[:, :max_seq_len]
                time_embed = self.time_embedding(time.unsqueeze(-1))
                data_embed = data_embed + time_embed

            # Ensure final data_embed has the correct shape
            if data_embed.size(-1) != self.config.hidden_size:
                logger.warning(f"Mismatch in embedding dimension: {data_embed.size(-1)} vs {self.config.hidden_size}")
                data_embed = data_embed[..., :self.config.hidden_size]

            return self.embedding_dropout(data_embed)
        elif isinstance(batch, torch.Tensor):
            return self.data_embedding_layer(batch)
        else:
            raise TypeError("Input 'batch' should be a dictionary or a Tensor.")

class ConditionallyIndependentPointProcessTransformer(StructuredTransformerPreTrainedModel):
    def __init__(self, config: StructuredTransformerConfig, vocabulary_config: VocabularyConfig, oov_index: int, save_dir: str = "./model_outputs"):
        super().__init__(config)
        
        self.config = config
        self.vocabulary_config = vocabulary_config
        self.oov_index = oov_index
        self.save_dir = save_dir

        self.embed_dim = config.hidden_size
        
        self.input_layer = ConditionallyIndependentPointProcessInputLayer(
            config,
            vocabulary_config.vocab_sizes_by_measurement,
            oov_index=self.oov_index,
        )
        
        self.h = torch.nn.ModuleList([
            torch.nn.utils.skip_init(InnerBlock, config, layer_id=i, is_seq=True, device=config.device)
            for i in range(config.num_hidden_layers)
        ])
        
        self.ln_f = CustomLayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        if config.use_batch_norm:
            self.bn_f = nn.BatchNorm1d(self.embed_dim)

        self.gradient_checkpointing = config.use_gradient_checkpointing

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        dynamic_indices: Union[torch.Tensor, Dict[str, torch.Tensor]],
        dynamic_values: torch.Tensor | None = None,
        static_indices: torch.Tensor | None = None,
        static_measurement_indices: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        past: tuple[torch.FloatTensor] | None = None,
        seq_attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple[torch.Tensor] | TransformerOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if past is None:
            past = tuple([None] * len(self.h))

        if input_embeds is None:
            input_embeds = self.input_layer({
                'dynamic_indices': dynamic_indices,
                'dynamic_values': dynamic_values,
                'static_indices': static_indices,
                'static_measurement_indices': static_measurement_indices,
                'time': time
            })

        hidden_states = input_embeds
        original_shape = hidden_states.shape

        if seq_attention_mask is None:
            if isinstance(dynamic_indices, dict):
                dynamic_indices_tensor = dynamic_indices['dynamic_indices']
            else:
                dynamic_indices_tensor = dynamic_indices
            seq_attention_mask = torch.ones(dynamic_indices_tensor.shape[:2], dtype=torch.bool, device=dynamic_indices_tensor.device)
        seq_attention_mask = expand_mask(seq_attention_mask, input_embeds.dtype)

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    seq_attention_mask,
                    layer_past,
                    head_mask[i],
                    use_cache,
                    output_attentions,
                    use_reentrant=False
                )
            else:
                outputs = block(
                    hidden_states,
                    attention_mask=seq_attention_mask,
                    layer_past=layer_past,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states, extra_return_info = outputs

            if use_cache is True:
                presents = presents + (extra_return_info["present_key_value"],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (extra_return_info["attn_weights"],)

        hidden_states = self.ln_f(hidden_states)
        if hasattr(self, 'bn_f'):
            if hidden_states.dim() == 4:
                # For 4D input, apply batch norm to the last two dimensions
                hidden_states = hidden_states.permute(0, 3, 1, 2)
                hidden_states = self.bn_f(hidden_states.reshape(hidden_states.size(0), -1, hidden_states.size(-1)))
                hidden_states = hidden_states.view(*hidden_states.shape[:2], -1, hidden_states.size(-1)).permute(0, 2, 3, 1)
            else:
                hidden_states = self.bn_f(hidden_states.transpose(1, 2)).transpose(1, 2)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return TransformerOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions
        )

class NestedAttentionPointProcessInputLayer(torch.nn.Module):
    """Processes input batch and produces input dependency graph element embeddings.

    This layer accepts a batch from an event-stream PyTorch dataset and returns input embeddings from it. This
    is designed for nested attention models, as it splits the input embeddings into different components
    corresponding to different dependency graph positions. Combines time and data embeddings.

    Args:
        config: Configuration parameters for the structured transformer.
    """

    def __init__(
        self,
        config: StructuredTransformerConfig,
        oov_index: int,  # Add this parameter
    ):
        super().__init__()

        self.config = config

        # We need to translate from measurement name to index here via config.measurements_idxmap
        split_by_measurement_indices = []
        for measurement_list in config.measurements_per_dep_graph_level:
            out_list = []
            for measurement in measurement_list:
                match measurement:
                    case str():
                        out_list.append(config.measurements_idxmap[measurement])
                    case [str() as measurement, str() | MeasIndexGroupOptions() as group_mode]:
                        out_list.append((config.measurements_idxmap[measurement], group_mode))
                    case _:
                        raise ValueError(
                            f"Unexpected measurement {type(measurement)}: {measurement}\n"
                            f"{config.measurements_per_dep_graph_level}"
                        )
            split_by_measurement_indices.append(out_list)

        self.data_embedding_layer = DataEmbeddingLayer(
            n_total_embeddings=max(config.vocab_sizes_by_measurement.values()) + 125,
            out_dim=config.hidden_size,
            categorical_embedding_dim=config.categorical_embedding_dim,
            numerical_embedding_dim=config.numerical_embedding_dim,
            static_embedding_mode=config.static_embedding_mode,
            split_by_measurement_indices=split_by_measurement_indices,
            do_normalize_by_measurement_index=config.do_normalize_by_measurement_index,
            static_weight=config.static_embedding_weight,
            dynamic_weight=config.dynamic_embedding_weight,
            categorical_weight=config.categorical_embedding_weight,
            numerical_weight=config.numerical_embedding_weight,
            oov_index=oov_index,  # Pass the oov_index
        )
        self.data_embedding_layer.to(config.device)

        if config.do_use_sinusoidal:
            self.time_embedding_layer = LearnableFrequencySinusoidalTemporalPositionEncoding(
                embedding_dim=config.hidden_size
            )
        else:
            self.time_embedding_layer = TemporalPositionEncoding(embedding_dim=config.hidden_size)
        self.embedding_dropout = torch.nn.Dropout(p=config.input_dropout)

    def forward(self, batch: PytorchBatch, dep_graph_el_generation_target: int | None = None) -> torch.Tensor:
        """Returns input dependency graph element embeddings for the provided batch.

        Args:
            batch: A PytorchBatch instance containing input data.
        """

        embed = self.data_embedding_layer(batch)
        # `data_embed` is of shape (batch_size, sequence_length, dep_graph_len, config.hidden_size).

        if not isinstance(batch, PytorchBatch):
            raise TypeError("Input 'batch' should be a PytorchBatch object.")

        time_embed = self.time_embedding_layer(batch)
        # `time_embed` is of shape (batch_size, sequence_length, config.hidden_size).

        # In this model, the first entry of the dependency graph *always* contains all the and only the time
        # dependent measures, so we combine the time_embedding in at this position as well.
        embed[:, :, 0] += time_embed

        # We perform a cumsum so that even in the first layer, our final embedding of the dep graph reflects
        # the entire event.
        embed = embed.cumsum(dim=2)

        if dep_graph_el_generation_target is not None:
            # This is used in generation to take advantage of the cache, where we only want to process a
            # single, new dependency graph element at a time.
            embed = embed[:, :, dep_graph_el_generation_target - 1].unsqueeze(2)

        if batch.event_mask is not None:
            embed = torch.where(
                batch.event_mask.unsqueeze(-1).unsqueeze(-1).expand_as(embed),
                embed,
                torch.zeros_like(embed),
            )

        return self.embedding_dropout(embed)


class NestedAttentionPointProcessTransformer(StructuredTransformerPreTrainedModel):
    """A transformer model specifically for nested attention point processes.

    This model uses an input layer to generate embeddings from an event-stream PyTorch dataset, and
    an InnerBlock layer for non-structured processing. As a nested attention model, event covariates are
    predicted in the sequence of the dependency graph elements, specified in the config's
    `measurements_per_dep_graph_level` parameter, depending on both the historical event embeddings and the
    prior dependency graph elements.

    Args:
        config: Configuration parameters for the structured transformer.

    Raises:
        ValueError: If the provided configuration indicates a conditionally independent model.
    """

    def __init__(self, config: StructuredTransformerConfig):
        super().__init__(config)

        if config.structured_event_processing_mode != StructuredEventProcessingMode.NESTED_ATTENTION:
            raise ValueError(f"{config.structured_event_processing_mode} invalid for this model!")

        self.embed_dim = config.hidden_size
        self.input_layer = NestedAttentionPointProcessInputLayer(
            config,
            oov_index=max(config.vocab_sizes_by_measurement.values()) + 1,
        )
        self.structured_event_processing_mode = config.structured_event_processing_mode

        self.h = nn.ModuleList(
            [StructuredTransformerBlock(config, layer_id=i) for i in range(config.num_hidden_layers)]
        )

        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False

        self.attention_dir = os.path.join(config.save_dir, "attention_weights")
        os.makedirs(self.attention_dir, exist_ok=True)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        batch: PytorchBatch | None = None,
        input_embeds: torch.Tensor | None = None,
        past: tuple[torch.FloatTensor] | None = None,
        seq_attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        dep_graph_past: tuple[torch.FloatTensor] | None = None,
        dep_graph_el_generation_target: int | None = None,
    ) -> tuple[torch.Tensor] | TransformerOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_embeds is None:
            assert batch is not None
            input_embeds = self.input_layer(
                batch, dep_graph_el_generation_target=dep_graph_el_generation_target
            )
            event_mask = batch["event_mask"]
        else:
            assert batch is None, "Can't specify both input_embeds and batch."
            event_mask = None

        if seq_attention_mask is None and batch is not None and batch.get("event_mask", None) is not None:
            seq_attention_mask = expand_mask(batch["event_mask"], input_embeds.dtype)

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        hidden_states = input_embeds
        bsz, seq_len, dep_graph_len, hidden_size = hidden_states.shape

        presents = {"seq_past": (), "dep_graph_past": ()} if use_cache else None
        all_self_attentions = {"seq_attentions": (), "dep_graph_attentions": ()} if output_attentions else None

        update_seq_cache = False
        update_dep_graph_cache = False
        re_set_dep_graph_cache = False
        prepend_graph_with_history_embeddings = True
        update_last_graph_el_to_history_embedding = True

        if use_cache:
            match dep_graph_el_generation_target:
                case int() if dep_graph_el_generation_target > 0:
                    update_dep_graph_cache = True
                    if dep_graph_past is None:
                        raise ValueError(
                            "dep_graph_past should not be None if dep_graph_el_generation_target is "
                            f"{dep_graph_el_generation_target}."
                        )
                    prepend_graph_with_history_embeddings = False
                    update_last_graph_el_to_history_embedding = False
                case int() if dep_graph_el_generation_target == 0:
                    update_seq_cache = True
                    update_dep_graph_cache = True
                    re_set_dep_graph_cache = True
                    prepend_graph_with_history_embeddings = False
                    update_last_graph_el_to_history_embedding = True
                case None:
                    if dep_graph_past is not None:
                        raise ValueError(
                            f"dep_graph_past should be None if gen target is None; got {dep_graph_past}"
                        )
                    update_seq_cache = True
                    update_dep_graph_cache = True
                    re_set_dep_graph_cache = True
                    prepend_graph_with_history_embeddings = True
                    update_last_graph_el_to_history_embedding = True
                case _:
                    raise ValueError(
                        "While use_cache=True, dep_graph generation target must be a non-negative int; got "
                        f"{dep_graph_el_generation_target}."
                    )

        compute_contextualized_history_embeddings = (
            prepend_graph_with_history_embeddings or update_last_graph_el_to_history_embedding
        )

        past = tuple([None] * len(self.h)) if past is None else past
        dep_graph_past = tuple([None] * len(self.h)) if dep_graph_past is None else dep_graph_past

        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past, dep_graph_layer_past) in enumerate(zip(self.h, past, dep_graph_past)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. "
                        "Setting `use_cache=False`..."
                    )
                    use_cache = False
                    prepend_graph_with_history_embeddings = True
                    update_last_graph_el_to_history_embedding = True
                    update_seq_cache = False
                    update_dep_graph_cache = False
                    re_set_dep_graph_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                args = (
                    hidden_states,
                    seq_attention_mask,
                    dict(
                        layer_past=layer_past,
                        head_mask=head_mask[i],
                        use_cache=False,
                        output_attentions=output_attentions,
                    ),
                    dict(
                        layer_past=dep_graph_layer_past,
                        use_cache=False,
                        output_attentions=output_attentions,
                    ),
                )

                outputs = torch.utils.checkpoint.checkpoint(create_custom_forward(block), *args, use_reentrant=False)
            else:
                kwargs = dict(
                    hidden_states=hidden_states,
                    seq_attention_mask=seq_attention_mask,
                    event_mask=event_mask,
                    prepend_graph_with_history_embeddings=prepend_graph_with_history_embeddings,
                    update_last_graph_el_to_history_embedding=update_last_graph_el_to_history_embedding,
                    seq_module_kwargs=dict(
                        layer_past=layer_past,
                        head_mask=head_mask[i],
                        use_cache=update_seq_cache,
                        output_attentions=output_attentions,
                    ),
                    dep_graph_module_kwargs=dict(
                        layer_past=dep_graph_layer_past,
                        use_cache=update_dep_graph_cache,
                        output_attentions=output_attentions,
                    ),
                )
                outputs = block(**kwargs)

            hidden_states, extra_return_info = outputs

            if update_seq_cache:
                presents["seq_past"] = presents["seq_past"] + (
                    extra_return_info["seq_module"]["present_key_value"],
                )
            if update_dep_graph_cache:
                presents["dep_graph_past"] = presents["dep_graph_past"] + (
                    extra_return_info["dep_graph_module"]["present_key_value"],
                )

            if output_attentions:
                if compute_contextualized_history_embeddings:
                    all_self_attentions["seq_attentions"] = all_self_attentions["seq_attentions"] + (
                        extra_return_info["seq_module"]["attn_weights"],
                    )
                all_self_attentions["dep_graph_attentions"] = all_self_attentions["dep_graph_attentions"] + (
                    extra_return_info["dep_graph_module"]["attn_weights"],
                )

        # Apply layer normalization if configured
        if self.config.use_layer_norm:
            hidden_states = self.ln_f(hidden_states)
        
        # Apply batch normalization if configured
        if self.config.use_batch_norm:
            hidden_states = hidden_states.permute(0, 3, 1, 2)  # [bsz, hidden_size, seq_len, dep_graph_len]
            hidden_states = self.bn_f(hidden_states)
            hidden_states = hidden_states.permute(0, 2, 3, 1)  # [bsz, seq_len, dep_graph_len, hidden_size]

        hidden_states = hidden_states.view(input_embeds.size())
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if use_cache:
            if not update_seq_cache:
                presents["seq_past"] = past
            if re_set_dep_graph_cache:
                def reshape_to_last_dep_graph_el(t: torch.FloatTensor) -> torch.FloatTensor:
                    want_shape = (
                        bsz * seq_len,
                        self.config.num_attention_heads,
                        "?",
                        self.config.head_dim,
                    )
                    err_str = f"Shape malformed! Want {want_shape}, Got {t.shape}"

                    torch._assert(t.shape[0] == (bsz * seq_len), err_str)
                    torch._assert(t.shape[1] == self.config.num_attention_heads, err_str)
                    torch._assert(t.shape[3] == self.config.head_dim, err_str)

                    t = t.reshape(bsz, seq_len, self.config.num_attention_heads, -1, self.config.head_dim)
                    return t[:, -1, :, -1, :].unsqueeze(2)

                presents["dep_graph_past"] = tuple(
                    tuple(reshape_to_last_dep_graph_el(e) for e in kv) for kv in presents["dep_graph_past"]
                )

        if not return_dict:
            return tuple(
                v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None
            )

        return TransformerOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )