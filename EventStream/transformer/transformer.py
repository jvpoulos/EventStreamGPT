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

logger = logging.get_logger(__name__)


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
        """Splits the last dimension of a tensor into `num_heads` and `attn_head_size`.

        Args:
            tensor: The input tensor.
            num_heads: Number of attention heads.
            attn_head_size: The size of each attention head.

        Returns:
            The re-shaped tensor.
        """
        if tensor.dim() == 2:
            # Handle 2D tensors (batch_size * seq_length, hidden_size)
            batch_seq, hidden_size = tensor.size()
            seq_length = hidden_size // (num_heads * attn_head_size)
            new_shape = (batch_seq // seq_length, seq_length, num_heads, attn_head_size)
            tensor = tensor.view(new_shape)
            return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        elif tensor.dim() == 3:
            # Handle 3D tensors (batch_size, seq_length, hidden_size)
            new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
            tensor = tensor.view(new_shape)
            return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        elif tensor.dim() == 4:
            # Handle 4D tensors (batch_size, seq_length, dep_graph_len, hidden_size)
            new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
            tensor = tensor.view(new_shape)
            return tensor.permute(0, 3, 1, 2, 4)  # (batch, head, seq_length, dep_graph_len, head_features)
        else:
            raise ValueError(f"Unexpected tensor shape: {tensor.shape}")

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """Merges the attention heads back into a single last dimension.

        Args:
            tensor: The input tensor.
            num_heads: Number of attention heads.
            attn_head_size: The size of each attention head.

        Returns:
            The re-shaped tensor.
        """
        if tensor.dim() == 4:
            # Handle 4D tensors (batch, head, seq_length, head_features)
            tensor = tensor.permute(0, 2, 1, 3).contiguous()
            batch_size, seq_length, _, _ = tensor.size()
            new_shape = (batch_size, seq_length, num_heads * attn_head_size)
            return tensor.view(new_shape)
        elif tensor.dim() == 5:
            # Handle 5D tensors (batch, head, seq_length, dep_graph_len, head_features)
            tensor = tensor.permute(0, 2, 3, 1, 4).contiguous()
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
        """Applies the attention mechanism to the input hidden states.

        Args:
            hidden_states: The input hidden states.
            attention_mask: A mask to be applied on the attention weights.
            layer_past: The past layer states.
            head_mask: A mask to be applied on the attention heads.
            use_cache: A flag indicating whether to cache the layer's past states.
            output_attentions: A flag indicating whether to output the attention weights.
            static_kv_first: In the case of attention over the dependency graph, the history embedding is
                dropped after processing, so we want to only use it as a KV, not as a query.

        Returns:
            A tuple containing the output of the attention mechanism and a dictionary of optional outputs.
        """

        # TODO(mmd): Flash attention
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # query, key, and value are all of shape (batch, head, seq_length, head_features)

        if static_kv_first:
            # In this case, we are only interested in performing the attention update over the non-static
            # queries.
            query = query[:, :, 1:, :]

        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = attn_output.to(hidden_states.device)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = {"present_key_value": present}
        if output_attentions:
            outputs["attn_weights"] = attn_weights

        return attn_output, outputs  # a, {present, (attentions)}


class InnerAttention(nn.Module):
    """The inner attention module used by the GPTs in this codebase.

    This module largely just selects what kind of attention computation should be used in this layer, and
    offloads computation therein.

    Args:
        config: The model configuration object.
        layer_id: Which layer is this attention computation in (by integer index)?
        is_seq: Is this a sequence or dependency-graph attention layer?

    Raises:
        ValueError: If an invalid attention type is provided.
    """

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

        if self.attention_type in ["global", "local"]:
            self.attention = InnerSelfAttention(
                config, attention_type=self.attention_type, window_size=self.window_size
            )
        else:
            raise ValueError(
                "Only attn layer types 'global' and 'local' exist, but got `self.attention_layers`: "
                f"{self.attention_layers}. Select attn layer types from ['global', 'local'] only."
            )

        # We put the layer norm in here as sometimes the attention layer is used independently of the full
        # block setup but we still want the layer norm to happen.
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

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
        """Forward pass.

        This returns the pre-selected attention calculation over the inputs (run through a layer norm).

        Args:
            hidden_states: The input hidden states.
            attention_mask: A mask to be applied on the attention weights.
            layer_past: The past layer states.
            head_mask: A mask to be applied on the attention heads.
            use_cache: A flag indicating whether to cache the layer's past states.
            output_attentions: A flag indicating whether to output the attention weights.
            static_kv_first: In the case of attention over the dependency graph, the history embedding is
                dropped after processing, so we want to only use it as a KV, not as a query.
        """

        return self.attention(
            self.layer_norm(hidden_states),
            attention_mask=attention_mask,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            static_kv_first=static_kv_first,
        )

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
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        # Move the weight and bias parameters of the LayerNorm module to the same device
        self.layer_norm.weight.data = self.layer_norm.weight.data.to(config.device)
        self.layer_norm.bias.data = self.layer_norm.bias.data.to(config.device)

        self.mlp = InnerMLP(config)

        if device is not None:
            self.to(device)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_past=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
        static_kv_first: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Conducts the forward pass for the inner block.

        Args:
            hidden_states: Input tensor.
            attention_mask: Mask to avoid attending to padded token positions.
            layer_past: Cache of past hidden states for more efficient decoding.
            head_mask: Mask to nullify selected heads of the self-attention module.
            use_cache: Whether to use caching.
            output_attentions: Whether to return attention probabilities in the output.
            static_kv_first: Whether the static key-value pair comes first.

        Returns:
            tuple: Modified hidden states and a dictionary containing present key-value pair and
            attention weights (if `output_attentions=True`).
        """

        # If we have a static kv entry first, we don't want to process it in the rest of the block, so we drop
        # it from the residual.
        residual = hidden_states if not static_kv_first else hidden_states[:, 1:, :]

        attn_outputs = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            static_kv_first=static_kv_first,
        )
        attn_output, outputs = attn_outputs  # output_attn: a, {present, (attentions)}

        # residual connection
        hidden_states = attn_output + residual
        
        # Move hidden_states to the same device as the LayerNorm weights
        hidden_states = hidden_states.to(self.layer_norm.weight.device)
        
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)

        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if not use_cache:
            outputs.pop("present_key_value")
        return hidden_states, outputs

    def _set_static_graph(self):
        if hasattr(self.attn, '_set_static_graph'):
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


class ConditionallyIndependentPointProcessInputLayer(torch.nn.Module):
    def __init__(self, config: StructuredTransformerConfig, vocab_sizes_by_measurement: dict[str, int], oov_index: int, do_use_sinusoidal: bool):
        super().__init__()

        self.config = config

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
        self.data_embedding_layer.to(config.device)

        if do_use_sinusoidal:
            self.time_embedding_layer = TemporalPositionEncoding(embedding_dim=config.hidden_size)
        else:
            self.time_embedding_layer = LearnableFrequencySinusoidalTemporalPositionEncoding(
                embedding_dim=config.hidden_size
            )

        self.embedding_dropout = torch.nn.Dropout(p=config.input_dropout)
        
        self.dynamic_values_encoder = nn.Linear(1, config.hidden_size)
        self.missing_value_embedding = nn.Parameter(torch.randn(config.hidden_size))
        self.dynamic_values_norm = nn.BatchNorm1d(1)

    def forward(self, batch: dict | torch.Tensor) -> torch.Tensor:
        if isinstance(batch, torch.Tensor):
            dynamic_indices = batch
            dynamic_values = None
        elif isinstance(batch, dict):
            dynamic_indices = batch['dynamic_indices']
            dynamic_values = batch.get('dynamic_values')
        else:
            raise TypeError("Input 'batch' should be a dictionary or a Tensor.")

        # Replace NaN values in dynamic_indices with the OOV index
        dynamic_indices = torch.where(torch.isnan(dynamic_indices), torch.full_like(dynamic_indices, self.config.vocab_size), dynamic_indices)
        
        data_embed: torch.Tensor = self.data_embedding_layer(dynamic_indices)
        
        if dynamic_values is not None:
            # Create a mask for non-missing values
            mask = ~torch.isnan(dynamic_values)
            
            # Process only non-missing values
            valid_values = dynamic_values[mask]
            if valid_values.numel() > 0:
                valid_values_normalized = self.dynamic_values_norm(valid_values.unsqueeze(1)).squeeze(1)
                valid_values_embed = self.dynamic_values_encoder(valid_values_normalized.unsqueeze(-1))
                
                # Use scatter to place the encoded values back
                dynamic_values_embed = self.missing_value_embedding.expand(*dynamic_values.shape, self.config.hidden_size)
                dynamic_values_embed = dynamic_values_embed.to(data_embed.dtype)  # Ensure same dtype as data_embed
                dynamic_values_embed[mask] = valid_values_embed.squeeze(1).to(data_embed.dtype)  # Ensure same dtype
            else:
                dynamic_values_embed = self.missing_value_embedding.expand(*dynamic_values.shape, self.config.hidden_size)
                dynamic_values_embed = dynamic_values_embed.to(data_embed.dtype)  # Ensure same dtype as data_embed
            
            # Combine with data embeddings
            data_embed = data_embed + dynamic_values_embed
        
        data_embed = self.embedding_dropout(data_embed)

        return data_embed
            
class ConditionallyIndependentPointProcessTransformer(StructuredTransformerPreTrainedModel):
    def __init__(self, config: StructuredTransformerConfig, vocabulary_config: VocabularyConfig, oov_index: int):
        super().__init__(config)
        
        self.config = config
        self.vocabulary_config = vocabulary_config
        self.oov_index = oov_index

        print(f"ConditionallyIndependentPointProcessTransformer: oov_index = {self.oov_index}")

        self.embed_dim = config.hidden_size
        
        self.input_layer = ConditionallyIndependentPointProcessInputLayer(
            config,
            vocabulary_config.vocab_sizes_by_measurement,
            oov_index=self.oov_index,
            do_use_sinusoidal=config.do_use_sinusoidal,
        )
        
        self.h = torch.nn.ModuleList([
            torch.nn.utils.skip_init(InnerBlock, config, layer_id=i, is_seq=True, device=config.device)
            for i in range(config.num_hidden_layers)
        ])

        if config.use_layer_norm:
            self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        if config.use_batch_norm:
            self.bn_f = nn.BatchNorm1d(self.embed_dim)

        self.gradient_checkpointing = config.use_gradient_checkpointing

        self.attention_dir = os.path.join(config.save_dir, "attention_weights")
        os.makedirs(self.attention_dir, exist_ok=True)

        # Initialize weights and apply final processing
        self.post_init()

        self._current_epoch = 0

    def _set_static_graph(self):
        for module in self.children():
            if hasattr(module, '_set_static_graph'):
                module._set_static_graph()
        
        # Explicitly set static graph for all InnerBlock modules
        for block in self.h:
            if hasattr(block, '_set_static_graph'):
                block._set_static_graph()

    @property
    def current_epoch(self):
        return self._current_epoch

    @current_epoch.setter
    def current_epoch(self, value):
        self._current_epoch = value
        
    def forward(
        self,
        batch: dict | torch.Tensor,
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
            if batch is None:
                raise ValueError("Either batch or input_embeds must be provided")
            input_embeds = self.input_layer(batch)

        # Add an extra dimension if input is 2D
        if input_embeds.dim() == 2:
            input_embeds = input_embeds.unsqueeze(1)

        # Handle NaNs more efficiently
        nan_mask = torch.isnan(input_embeds)
        if nan_mask.any():
            logger.warning(f"{nan_mask.sum()} NaNs detected in input_embeds. Replacing with zeros.")
            input_embeds = input_embeds.clone()
            input_embeds[nan_mask] = 0.0

        if batch is not None and hasattr(batch, 'event_mask') and batch.event_mask is not None:
            seq_attention_mask = expand_mask(batch.event_mask, input_embeds.dtype)

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        hidden_states = input_embeds

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
                kwargs = dict(
                    hidden_states=hidden_states,
                    attention_mask=seq_attention_mask,
                    layer_past=layer_past,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
                outputs = block(**kwargs)

            hidden_states, extra_return_info = outputs

            if batch is not None and hasattr(batch, 'event_mask') and batch.event_mask is not None:
                hidden_states = torch.where(
                    batch.event_mask.unsqueeze(-1).expand_as(hidden_states),
                    hidden_states,
                    torch.zeros_like(hidden_states),
                )

            if use_cache is True:
                presents = presents + (extra_return_info["present_key_value"],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (extra_return_info["attn_weights"],)

        hidden_states = self.ln_f(hidden_states)

        # Reshape hidden states to match input_embeds
        hidden_states = hidden_states.view(*input_embeds.shape)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if output_attentions:
            # Save attention weights
            attention_path = os.path.join(self.config.save_dir, f"attention_weights_epoch_{self.current_epoch}.pt")
            torch.save(all_self_attentions, attention_path)

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

        if config.do_use_learnable_sinusoidal_ATE:
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