"""The internal transformer module code.

TODO(mmcdermott): Can use `transformers.apply_chunking_to_forward` to save memory.

Based on
https://raw.githubusercontent.com/huggingface/transformers/\
e3cc4487fe66e03ec85970ea2db8e5fb34c455f4/src/transformers/models/gpt_neo/modeling_gpt_neo.py
"""  # noqa E501

import math
import os
import numpy as np

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
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Union, Dict

import wandb
from pytorch_lightning.loggers import WandbLogger

import logging

def adjust_tensor_shape(tensor, target_shape, name):
    if tensor is None:
        return None
    if tensor.shape != target_shape:
        logger.warning(f"Adjusting {name} shape from {tensor.shape} to {target_shape}.")
        if len(tensor.shape) < len(target_shape):
            # Add dimensions if needed
            tensor = tensor.view(*tensor.shape, *([1] * (len(target_shape) - len(tensor.shape))))
        # Expand or slice as needed
        tensor = tensor.expand(*target_shape)
    return tensor

class WandbLogger(logging.Logger):
    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)

    def _log_to_wandb(self, level, msg, *args, **kwargs):
        if wandb.run is not None:
            formatted_msg = msg % args if args else msg
            try:
                wandb.log({f"log/{self.name}/{level}": formatted_msg})
            except TypeError as e:
                # If serialization fails, convert to string
                wandb.log({f"log/{self.name}/{level}": str(formatted_msg)})

    def debug(self, msg, *args, **kwargs):
        super().debug(msg, *args, **kwargs)
        self._log_to_wandb("DEBUG", msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        super().info(msg, *args, **kwargs)
        self._log_to_wandb("INFO", msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        super().warning(msg, *args, **kwargs)
        self._log_to_wandb("WARNING", msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        super().error(msg, *args, **kwargs)
        self._log_to_wandb("ERROR", msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        super().critical(msg, *args, **kwargs)
        self._log_to_wandb("CRITICAL", msg, *args, **kwargs)

# Set up the custom logger
logging.setLoggerClass(WandbLogger)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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
    attention_mask = attention_mask.to(dtype=torch.float16)  # Change to float16
    attention_mask = (1.0 - attention_mask) * torch.finfo(torch.float16).min

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
        layer_id: int
    ):
        super().__init__()
        self.config = config  # Store the config as an attribute
        self.attention_type = attention_type
        self.window_size = window_size
        self.causal = attention_type in ["local", "global"]  # Set causal based on attention type
        self.use_flash_attention = config.use_flash_attention
        self.layer_id = layer_id

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
        assert self.head_dim * self.num_heads == self.embed_dim, f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."

        # Initialize linear layers with the same dtype as the config
        self.dtype = getattr(torch, config.dtype) if hasattr(config, 'dtype') else torch.float32
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False).to(self.dtype)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False).to(self.dtype)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False).to(self.dtype)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.embed_dim, bias=True).to(self.dtype)

    def _split_heads(self, tensor, num_heads, attn_head_size):
        if tensor.dim() == 2:
            batch_size, dim = tensor.size()
            seq_length = 1
        elif tensor.dim() == 3:
            batch_size, seq_length, dim = tensor.size()
        else:
            raise ValueError(f"Unexpected tensor dimensions: {tensor.dim()}")
        
        tensor = tensor.view(batch_size, seq_length, num_heads, attn_head_size)
        return tensor.permute(0, 2, 1, 3).contiguous()

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        if tensor.dim() == 3:
            batch_size, seq_length, _ = tensor.size()
            return tensor.view(batch_size, seq_length, num_heads * attn_head_size)
        elif tensor.dim() == 4:
            batch_size, num_heads, seq_length, head_dim = tensor.size()
            return tensor.transpose(1, 2).contiguous().view(batch_size, seq_length, num_heads * head_dim)
        else:
            raise ValueError(f"Unexpected tensor dimensions: {tensor.dim()}")

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        logger.debug(f"Query shape: {query.shape}")
        logger.debug(f"Key shape: {key.shape}")
        logger.debug(f"Value shape: {value.shape}")
        if attention_mask is not None:
            logger.debug(f"Original attention mask shape: {attention_mask.shape}")
        
        bsz, num_heads, seq_len, head_dim = query.shape

        if self.use_flash_attention and self.training:
            try:
                # Reshape for Flash Attention
                qkv = torch.stack([query, key, value], dim=2)
                qkv = qkv.permute(0, 3, 2, 1, 4).contiguous()  # [bsz, seq_len, 3, num_heads, head_dim]
                
                logger.debug(f"QKV shape for Flash Attention: {qkv.shape}")
                
                attn_output = flash_attn_qkvpacked_func(
                    qkv,
                    dropout_p=self.attn_dropout.p if self.training else 0.0,
                    softmax_scale=None,
                    causal=self.causal
                )
                
                # Reshape back
                attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # [bsz, num_heads, seq_len, head_dim]
                logger.debug(f"attn_output shape after Flash Attention: {attn_output.shape}")
                return attn_output, None  # Flash Attention doesn't return attention weights
            except Exception as e:
                logger.warning(f"Flash Attention failed: {str(e)}. Falling back to standard attention.")

        # Standard attention implementation
        query = query.to(key.dtype)
        attn_weights = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        logger.debug(f"attn_weights shape after matmul: {attn_weights.shape}")

        if self.causal:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=query.device), diagonal=1)
            causal_mask = causal_mask.view(1, 1, seq_len, seq_len).expand(bsz, num_heads, -1, -1)
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

        if attention_mask is not None:
            # Ensure attention_mask is on the same device as query
            attention_mask = attention_mask.to(query.device)
            # Reshape attention_mask to match the expected dimensions
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            
            # Adjust the sequence length dimension of the attention mask
            if attention_mask.size(-1) != seq_len:
                if attention_mask.size(-1) < seq_len:
                    attention_mask = F.pad(attention_mask, (0, seq_len - attention_mask.size(-1)), value=float('-inf'))
                else:
                    attention_mask = attention_mask[..., :seq_len]
            
            # Expand attention_mask to match attn_weights shape
            attention_mask = attention_mask.expand(bsz, num_heads, seq_len, seq_len)
            
            attn_weights = attn_weights + attention_mask

        # Clip extreme values to avoid overflow in softmax
        attn_weights = torch.clamp(attn_weights, min=-1e4, max=1e4)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        logger.debug(f"attn_weights shape after softmax: {attn_weights.shape}")
        
        attn_weights = self.attn_dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        logger.debug(f"Final attn_weights shape: {attn_weights.shape}, value shape: {value.shape}")
        attn_output = torch.matmul(attn_weights, value)
        logger.debug(f"attn_output shape: {attn_output.shape}")

        # Check for NaN values
        if torch.isnan(attn_output).any():
            logger.warning("NaN values detected in attention output. Replacing with zeros.")
            attn_output = torch.where(torch.isnan(attn_output), torch.zeros_like(attn_output), attn_output)

        return attn_output, attn_weights

    def _adjust_attention_mask(self, attention_mask, bsz, num_heads, seq_len):
        logger.debug(f"Input attention mask shape: {attention_mask.shape}")

        if attention_mask is None:
            return None

        if attention_mask.dim() == 2:  # [bsz, seq_len]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:  # [bsz, 1, seq_len]
            attention_mask = attention_mask.unsqueeze(1)

        # Adjust sequence length if necessary
        if attention_mask.size(-1) != seq_len:
            logger.warning(f"Adjusting attention mask sequence length from {attention_mask.size(-1)} to {seq_len}.")
            if attention_mask.size(-1) < seq_len:
                attention_mask = F.pad(attention_mask, (0, seq_len - attention_mask.size(-1)), value=0)
            else:
                attention_mask = attention_mask[..., :seq_len]

        # Expand for multiple heads without using repeat
        if attention_mask.size(1) != num_heads:
            attention_mask = attention_mask.expand(-1, num_heads, -1, -1)

        # Convert boolean mask to float mask
        attention_mask = (1.0 - attention_mask.to(torch.float32)) * torch.finfo(torch.float32).min

        logger.debug(f"Adjusted attention mask shape: {attention_mask.shape}")
        return attention_mask
    
    def forward(self, hidden_states, attention_mask=None, layer_past=None, head_mask=None, use_cache=False, output_attentions=False, static_kv_first: bool = False):
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(1)  # Add sequence dimension
        
        bsz, seq_len, _ = hidden_states.shape
        logger.debug(f"Input hidden_states shape: {hidden_states.shape}")
        
        # Ensure consistent dtype for query, key, and value
        hidden_states = hidden_states.to(self.dtype)
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if attention_mask is not None:
            attention_mask = self._adjust_attention_mask(attention_mask, bsz, self.num_heads, seq_len)

        if self.use_flash_attention and self.training:
            try:
                # Reshape for Flash Attention
                qkv = torch.stack([query, key, value], dim=2)
                qkv = qkv.permute(0, 3, 2, 1, 4).contiguous()  # [bsz, seq_len, 3, num_heads, head_dim]
                
                logger.debug(f"QKV shape for Flash Attention: {qkv.shape}")
                
                class FlashAttentionFunction(torch.autograd.Function):
                    @staticmethod
                    def forward(ctx, qkv, dropout_p, softmax_scale, causal):
                        output = flash_attn_qkvpacked_func(
                            qkv,
                            dropout_p=dropout_p,
                            softmax_scale=softmax_scale,
                            causal=causal
                        )
                        ctx.save_for_backward(qkv)
                        ctx.dropout_p = dropout_p
                        ctx.softmax_scale = softmax_scale
                        ctx.causal = causal
                        return output

                    @staticmethod
                    def backward(ctx, grad_output):
                        qkv, = ctx.saved_tensors
                        try:
                            if hasattr(flash_attn_qkvpacked_func, 'backward'):
                                grad_qkv, = flash_attn_qkvpacked_func.backward(
                                    grad_output,
                                    qkv,
                                    dropout_p=ctx.dropout_p,
                                    softmax_scale=ctx.softmax_scale,
                                    causal=ctx.causal
                                )
                            else:
                                # Fallback to autograd if backward is not available
                                grad_qkv = torch.autograd.grad(
                                    flash_attn_qkvpacked_func(
                                        qkv,
                                        dropout_p=ctx.dropout_p,
                                        softmax_scale=ctx.softmax_scale,
                                        causal=ctx.causal
                                    ),
                                    qkv,
                                    grad_outputs=grad_output,
                                    allow_unused=True
                                )[0]
                            
                            if grad_qkv is None:
                                raise RuntimeError("Gradient computation failed")
                            
                        except Exception as e:
                            logger.warning(f"Flash Attention backward pass failed: {str(e)}. Falling back to standard attention.")
                            return None, None, None, None
                        return grad_qkv, None, None, None

                attn_output = FlashAttentionFunction.apply(
                    qkv,
                    self.attn_dropout.p if self.training else 0.0,
                    None,
                    self.causal
                )
                
                # Reshape back
                attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # [bsz, num_heads, seq_len, head_dim]
                logger.debug(f"attn_output shape after Flash Attention: {attn_output.shape}")
                attn_weights = None  # Flash Attention doesn't return attention weights
            except Exception as e:
                logger.warning(f"Flash Attention failed: {str(e)}. Falling back to standard attention.")
                attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)

        # Ensure attn_output has the same shape as the input
        if attn_output.shape != hidden_states.shape:
            logger.warning(f"Reshaping attn_output from {attn_output.shape} to {hidden_states.shape}")
            attn_output = attn_output.view(*hidden_states.shape)

        attn_output = self.resid_dropout(attn_output)

        outputs = {"attn_output": attn_output}
        if output_attentions:
            outputs["attn_weights"] = attn_weights
        if use_cache:
            outputs["present"] = torch.stack((key, value))

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
        logger.debug(f"CustomLayerNorm input shape: {x.shape}")
        logger.debug(f"CustomLayerNorm input dtype: {x.dtype}")
        logger.debug(f"CustomLayerNorm input device: {x.device}")
        logger.debug(f"CustomLayerNorm normalized_shape: {self.normalized_shape}")

        original_shape = x.shape
        original_dtype = x.dtype

        # Ensure computations are done in float32 for stability
        x = x.float()

        # Compute mean and variance with better numerical stability
        # Compute along the last dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)

        # Add a small epsilon to avoid division by zero
        std = (var + self.eps).sqrt()
        
        # Normalize
        x = (x - mean) / std

        # Check for NaN or Inf values after normalization
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.warning("NaN or Inf values detected after normalization. Clipping values.")
            x = torch.clamp(x, min=-10, max=10)

        # Apply weight and bias
        if self.weight.dim() == 1:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)

        # Convert back to original dtype
        x = x.to(original_dtype)

        logger.debug(f"CustomLayerNorm output shape: {x.shape}")
        return x

    def _check_gradients(self, grad, name):
        if torch.isnan(grad).any() or torch.isinf(grad).any():
            logger.warning(f"NaN or Inf gradients detected in CustomLayerNorm {name}")

class InnerAttention(nn.Module):
    def __init__(self, config: StructuredTransformerConfig, layer_id: int, is_seq: bool):
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
            config, attention_type=self.attention_type, window_size=self.window_size, layer_id=self.layer_id
        )
        self.seq_window_size = config.seq_window_size

    def forward(self, hidden_states, attention_mask=None, layer_past=None, head_mask=None, use_cache=False, output_attentions=False, static_kv_first: bool = False):
        # Ensure hidden_states is 3D
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(1)

        logger.debug(f"InnerAttention input hidden_states shape: {hidden_states.shape}")

        # Ensure hidden_states is on the correct device
        hidden_states = hidden_states.to(self.layer_norm.weight.device)

        # Apply layer norm
        normalized_hidden_states = self.layer_norm(hidden_states)

        logger.debug(f"InnerAttention normalized_hidden_states shape: {normalized_hidden_states.shape}")

        # Use the _adjust_attention_mask method from InnerSelfAttention
        if attention_mask is not None:
            attention_mask = self.attention._adjust_attention_mask(
                attention_mask, 
                normalized_hidden_states.shape[0], 
                self.attention.num_heads, 
                normalized_hidden_states.shape[1]
            )

        # Ensure the sequence length matches seq_window_size
        seq_window_size = self.seq_window_size
        if normalized_hidden_states.shape[1] > seq_window_size:
            normalized_hidden_states = normalized_hidden_states[:, :seq_window_size, :]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :, :, :seq_window_size]
        elif normalized_hidden_states.shape[1] < seq_window_size:
            pad_length = seq_window_size - normalized_hidden_states.shape[1]
            normalized_hidden_states = F.pad(normalized_hidden_states, (0, 0, 0, pad_length), mode='constant', value=0)
            if attention_mask is not None:
                attention_mask = F.pad(attention_mask, (0, pad_length), value=float('-inf'))

        attn_output, outputs = self.attention(
            normalized_hidden_states,
            attention_mask=attention_mask,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            static_kv_first=static_kv_first,
        )

        logger.debug(f"InnerAttention attn_output shape: {attn_output.shape}")

        return attn_output, outputs

    def _set_static_graph(self):
        if hasattr(self.attention, '_set_static_graph'):
            self.attention._set_static_graph()
            
class InnerBlock(nn.Module):
    def __init__(self, config: StructuredTransformerConfig, layer_id: int, is_seq: bool, device=None):
        super().__init__()
        self.attn = InnerAttention(config, layer_id, is_seq)
        self.layer_norm1 = CustomLayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.layer_norm2 = CustomLayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = InnerMLP(config)
        self.seq_window_size = config.seq_window_size

        if device is not None:
            self.to(device)

    def forward(self, hidden_states, attention_mask=None, layer_past=None, head_mask=None, use_cache=False, output_attentions=False, static_kv_first: bool = False):
        logger.debug(f"InnerBlock input hidden_states shape: {hidden_states.shape}")
        
        # Ensure hidden_states is 3D
        if hidden_states.dim() == 2:
            logger.warning(f"Received 2D hidden_states, reshaping to 3D. Shape before: {hidden_states.shape}")
            hidden_states = hidden_states.unsqueeze(1)  # Add sequence dimension
            logger.warning(f"Shape after reshape: {hidden_states.shape}")
        elif hidden_states.dim() != 3:
            raise ValueError(f"Expected 2D or 3D hidden_states, got {hidden_states.dim()}D")
        
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        
        logger.debug(f"InnerBlock hidden_states shape after layer_norm1: {hidden_states.shape}")

        # Ensure hidden_states is still 3D after layer_norm1
        if hidden_states.dim() == 2:
            logger.warning(f"hidden_states became 2D after layer_norm1. Reshaping to 3D. Shape before: {hidden_states.shape}")
            hidden_states = hidden_states.unsqueeze(1)  # Add sequence dimension
            logger.warning(f"Shape after reshape: {hidden_states.shape}")

        # Adjust sequence length if necessary
        seq_window_size = self.seq_window_size
        current_seq_len = hidden_states.size(1)
        
        if current_seq_len > seq_window_size:
            logger.warning(f"Truncating sequence length from {current_seq_len} to {seq_window_size}.")
            hidden_states = hidden_states[:, :seq_window_size, :]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :, :, :seq_window_size]
        elif current_seq_len < seq_window_size:
            logger.warning(f"Padding sequence length from {current_seq_len} to {seq_window_size}.")
            pad_length = seq_window_size - current_seq_len
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_length))
            if attention_mask is not None:
                attention_mask = F.pad(attention_mask, (0, pad_length), value=float('-inf'))

        attn_output, attn_outputs = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            static_kv_first=static_kv_first,
        )

        # Ensure attn_output has the same shape as residual
        if attn_output.shape != residual.shape:
            logger.warning(f"Reshaping attn_output from {attn_output.shape} to {residual.shape}")
            try:
                attn_output = attn_output.view(*residual.shape)
            except RuntimeError:
                logger.error(f"Failed to reshape attn_output. Input size: {attn_output.numel()}, output shape: {residual.shape}")
                # Fallback: Repeat or truncate to match the required shape
                attn_output = attn_output.view(-1).repeat(residual.numel() // attn_output.numel() + 1)[:residual.numel()].view(*residual.shape)

        hidden_states = attn_output + residual
        
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)

        # Ensure feed_forward_hidden_states has the same shape as residual
        if feed_forward_hidden_states.shape != residual.shape:
            logger.warning(f"Reshaping feed_forward_hidden_states from {feed_forward_hidden_states.shape} to {residual.shape}")
            feed_forward_hidden_states = feed_forward_hidden_states.view(residual.shape)

        hidden_states = residual + feed_forward_hidden_states

        outputs = {
            "present": attn_outputs.get("present"),
            "attn_weights": attn_outputs.get("attn_weights"),
        }

        logger.debug(f"InnerBlock output hidden_states shape: {hidden_states.shape}")
        return hidden_states, outputs

    def _set_static_graph(self):
        self.attn._set_static_graph()
        if hasattr(self.mlp, '_set_static_graph'):
            self.mlp._set_static_graph()
            
class InnerMLP(nn.Module):
    def __init__(self, config: StructuredTransformerConfig):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = nn.Linear(embed_dim, config.intermediate_size)
        self.c_proj = nn.Linear(config.intermediate_size, embed_dim)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(float(config.resid_dropout))

    def forward(self, hidden_states):
        original_shape = hidden_states.shape
        if hidden_states.dim() == 3:
            batch_size, seq_len, hidden_size = hidden_states.shape
            hidden_states = hidden_states.reshape(-1, hidden_size)
        
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        if len(original_shape) == 3:
            hidden_states = hidden_states.view(original_shape)
        
        return hidden_states

    def _set_static_graph(self):
        pass  # No submodules to set static graph for

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
        
        # Adjust the input size of the feature_combiner
        self.feature_combiner = nn.Linear(config.hidden_size * 3, config.hidden_size)

    def forward(self, input_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        logger.debug("Entering ConditionallyIndependentPointProcessInputLayer.forward")

        # Log shapes of input features
        for key, value in input_features.items():
            if value is not None:
                logger.debug(f"{key} shape: {value.shape}")
            else:
                logger.debug(f"{key}: None")
        
        dynamic_indices = input_features.get('dynamic_indices')
        dynamic_values = input_features.get('dynamic_values')
        dynamic_measurement_indices = input_features.get('dynamic_measurement_indices')
        static_indices = input_features.get('static_indices')
        static_measurement_indices = input_features.get('static_measurement_indices')
        time = input_features.get('time')

        # Convert indices to LongTensor
        if dynamic_indices is not None:
            dynamic_indices = dynamic_indices.long()
        if dynamic_measurement_indices is not None:
            dynamic_measurement_indices = dynamic_measurement_indices.long()
        if static_indices is not None:
            static_indices = static_indices.long()
        if static_measurement_indices is not None:
            static_measurement_indices = static_measurement_indices.long()

        logger.debug(f"After conversion - dynamic_indices dtype: {dynamic_indices.dtype if dynamic_indices is not None else None}")
        logger.debug(f"After conversion - dynamic_measurement_indices dtype: {dynamic_measurement_indices.dtype if dynamic_measurement_indices is not None else None}")

        # Process dynamic indices and measurement indices
        data_embed = self.data_embedding_layer({
            'dynamic_indices': dynamic_indices,
            'dynamic_values': dynamic_values,
            'dynamic_measurement_indices': dynamic_measurement_indices
        })

        logger.debug(f"data_embed shape: {data_embed.shape}")

        # Handle different possible shapes of data_embed
        if len(data_embed.shape) == 3:
            batch_size, seq_len, hidden_size = data_embed.shape
        elif len(data_embed.shape) == 2:
            batch_size, hidden_size = data_embed.shape
            seq_len = 1
            data_embed = data_embed.unsqueeze(1)
        else:
            raise ValueError(f"Unexpected shape for data_embed: {data_embed.shape}")

        # Process dynamic values
        if dynamic_values is not None:
            dynamic_values = dynamic_values[:, :seq_len]  # Ensure same sequence length
            mask = ~torch.isnan(dynamic_values)
            valid_values = dynamic_values[mask]

            if valid_values.numel() > 0:
                valid_values_normalized = self.dynamic_values_norm(valid_values.unsqueeze(1)).squeeze(1)
                valid_values_embed = self.dynamic_values_encoder(valid_values_normalized.unsqueeze(-1))
                valid_values_embed = valid_values_embed.to(dtype=data_embed.dtype)

                # Reshape valid_values_embed to match data_embed shape
                reshaped_valid_values_embed = torch.zeros_like(data_embed)
                reshaped_valid_values_embed[mask.unsqueeze(-1).expand_as(reshaped_valid_values_embed)] = valid_values_embed.view(-1)

                data_embed = torch.where(mask.unsqueeze(-1), reshaped_valid_values_embed, data_embed)

        # Process static indices
        if static_indices is not None:
            static_embed = self.static_embedding(static_indices)
            if self.use_addition_for_static:
                static_embed = static_embed.mean(dim=1).unsqueeze(1).expand(-1, seq_len, -1)
            else:
                static_embed = static_embed.unsqueeze(1).expand(-1, seq_len, -1)
        else:
            static_embed = torch.zeros_like(data_embed)

        # Process time
        if time is not None:
            time = time[:, :seq_len]  # Ensure same sequence length
            time_embed = self.time_embedding(time.unsqueeze(-1))
        else:
            time_embed = torch.zeros_like(data_embed)

        # Combine all features
        combined_features = torch.cat([data_embed, static_embed, time_embed], dim=-1)
        logger.debug(f"Combined features shape: {combined_features.shape}")

        # Pass through the feature combiner
        combined_embed = self.feature_combiner(combined_features)
        logger.debug(f"Combined embed shape after feature combiner: {combined_embed.shape}")

        logger.debug(f"Dynamic indices shape: {dynamic_indices.shape}")
        logger.debug(f"Dynamic values shape: {dynamic_values.shape if dynamic_values is not None else None}")
        logger.debug(f"Dynamic measurement indices shape: {dynamic_measurement_indices.shape if dynamic_measurement_indices is not None else None}")
        logger.debug(f"Static indices shape: {static_indices.shape if static_indices is not None else None}")
        logger.debug(f"Static measurement indices shape: {static_measurement_indices.shape if static_measurement_indices is not None else None}")
        logger.debug(f"Time shape: {time.shape if time is not None else None}")
        logger.debug(f"Combined features shape: {combined_features.shape}")
        logger.debug(f"Combined embed shape: {combined_embed.shape}")

        # Ensure the sequence length matches seq_window_size
        seq_window_size = self.config.seq_window_size
        if combined_embed.shape[1] < seq_window_size:
            pad_length = seq_window_size - combined_embed.shape[1]
            combined_embed = F.pad(combined_embed, (0, 0, 0, pad_length), mode='constant', value=0)
            logger.debug(f"Padded combined_embed shape: {combined_embed.shape}")
        elif combined_embed.shape[1] > seq_window_size:
            combined_embed = combined_embed[:, :seq_window_size, :]
            logger.debug(f"Truncated combined_embed shape: {combined_embed.shape}")

        logger.debug(f"Final combined_embed shape: {combined_embed.shape}")
        return self.embedding_dropout(combined_embed)

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

        self.forward_step = 0
        self.fp16 = torch.cuda.is_available()  # Use fp16 if CUDA is available

        self.use_grad_value_clipping = config.optimization_config.get('use_grad_value_clipping', True)
        self.clip_grad_value = config.optimization_config.get('clip_grad_value', 1.0)

        # Instead of setting self.dtype, use a property
        self._dtype = torch.float32

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        self._dtype = value
     
    def forward(
        self,
        dynamic_indices: Union[torch.Tensor, Dict[str, torch.Tensor]],
        dynamic_values: torch.Tensor | None = None,
        dynamic_measurement_indices: torch.Tensor | None = None,
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
        self.forward_step += 1
        logger.debug(f"Forward pass step: {self.forward_step}")
        # Log input shapes and dtypes
        logger.debug(f"dynamic_indices shape: {dynamic_indices.shape if isinstance(dynamic_indices, torch.Tensor) else type(dynamic_indices)}, dtype: {dynamic_indices.dtype if isinstance(dynamic_indices, torch.Tensor) else None}")
        logger.debug(f"dynamic_values shape: {dynamic_values.shape if dynamic_values is not None else None}, dtype: {dynamic_values.dtype if dynamic_values is not None else None}")
        logger.debug(f"dynamic_measurement_indices shape: {dynamic_measurement_indices.shape if dynamic_measurement_indices is not None else None}, dtype: {dynamic_measurement_indices.dtype if dynamic_measurement_indices is not None else None}")
        logger.debug(f"static_indices shape: {static_indices.shape if static_indices is not None else None}, dtype: {static_indices.dtype if static_indices is not None else None}")
        logger.debug(f"static_measurement_indices shape: {static_measurement_indices.shape if static_measurement_indices is not None else None}, dtype: {static_measurement_indices.dtype if static_measurement_indices is not None else None}")
        logger.debug(f"time shape: {time.shape if time is not None else None}, dtype: {time.dtype if time is not None else None}")
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if past is None:
            past = tuple([None] * len(self.h))

        if isinstance(dynamic_indices, dict):
            dynamic_indices = dynamic_indices.get('dynamic_indices')
        
        if dynamic_indices is None:
            raise ValueError("dynamic_indices cannot be None")

        # Get the local batch size for this GPU
        local_batch_size = dynamic_indices.size(0)

        # Adjust batch sizes for all input tensors
        dynamic_values = self._adjust_tensor_shape(dynamic_values, local_batch_size, "dynamic_values")
        dynamic_measurement_indices = self._adjust_tensor_shape(dynamic_measurement_indices, local_batch_size, "dynamic_measurement_indices")
        static_indices = self._adjust_tensor_shape(static_indices, local_batch_size, "static_indices")
        static_measurement_indices = self._adjust_tensor_shape(static_measurement_indices, local_batch_size, "static_measurement_indices")
        time = self._adjust_tensor_shape(time, local_batch_size, "time")

        if input_embeds is None:
            # Convert indices to LongTensor
            if isinstance(dynamic_indices, torch.Tensor):
                dynamic_indices = dynamic_indices.long()
            if dynamic_measurement_indices is not None:
                dynamic_measurement_indices = dynamic_measurement_indices.long()
            if static_indices is not None:
                static_indices = static_indices.long()
            if static_measurement_indices is not None:
                static_measurement_indices = static_measurement_indices.long()

            logger.debug(f"Before input_layer - dynamic_indices dtype: {dynamic_indices.dtype if isinstance(dynamic_indices, torch.Tensor) else None}")
            logger.debug(f"Before input_layer - dynamic_measurement_indices dtype: {dynamic_measurement_indices.dtype if dynamic_measurement_indices is not None else None}")

            input_embeds = self.input_layer({
                'dynamic_indices': dynamic_indices,
                'dynamic_values': dynamic_values,
                'dynamic_measurement_indices': dynamic_measurement_indices,
                'static_indices': static_indices,
                'static_measurement_indices': static_measurement_indices,
                'time': time
            })
            
        if input_embeds is not None:
            input_embeds = input_embeds.to(self.dtype)
        
        hidden_states = input_embeds if input_embeds is not None else self.input_layer(...)
        
        # Ensure consistent dtype for hidden_states
        hidden_states = hidden_states.to(self.dtype)
        
        logger.debug(f"hidden_states shape after input layer: {hidden_states.shape}")

        # Ensure hidden_states is 3D
        if hidden_states.dim() == 2:
            logger.warning(f"hidden_states is 2D with shape {hidden_states.shape}. Unsqueezing to 3D.")
            hidden_states = hidden_states.unsqueeze(1)
        elif hidden_states.dim() != 3:
            raise ValueError(f"Expected hidden_states to be 2D or 3D, but got {hidden_states.dim()}D with shape {hidden_states.shape}")
        
        logger.debug(f"hidden_states shape before entering transformer blocks: {hidden_states.shape}")


        if seq_attention_mask is None:
            seq_attention_mask = torch.ones(hidden_states.size(0), hidden_states.size(1), device=hidden_states.device)
        
        # Expand and adjust the mask to match expected dimensions
        seq_attention_mask = expand_mask(seq_attention_mask, input_embeds.dtype)

        # Truncate the sequence length to seq_window_size
        seq_window_size = self.config.seq_window_size
        if hidden_states.size(1) > seq_window_size:
            logger.warning(f"Truncating sequence length from {hidden_states.size(1)} to {seq_window_size}")
            hidden_states = hidden_states[:, :seq_window_size, :]
            if seq_attention_mask is not None:
                seq_attention_mask = seq_attention_mask[:, :, :, :seq_window_size]

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        if self.fp16:
            hidden_states = hidden_states.half()  # Convert to fp16

        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            logger.debug(f"Before block {i}: hidden_states shape: {hidden_states.shape}")
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # Adjust attention mask for each layer
            layer_attention_mask = self.h[i].attn.attention._adjust_attention_mask(
                seq_attention_mask, 
                hidden_states.size(0), 
                self.config.num_attention_heads, 
                hidden_states.size(1)
            )

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    layer_attention_mask,
                    layer_past,
                    head_mask[i],
                    use_cache,
                    output_attentions,
                    use_reentrant=False
                )
                hidden_states = outputs[0]
                extra_return_info = outputs[1] if len(outputs) > 1 else {}
            else:
                try:
                    outputs = block(
                        hidden_states,
                        attention_mask=layer_attention_mask,
                        layer_past=layer_past,
                        head_mask=head_mask[i],
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                    )

                    hidden_states, extra_return_info = outputs
                    logger.debug(f"After block {i}: hidden_states shape: {hidden_states.shape}")

                    # Ensure hidden_states has the correct shape
                    if hidden_states.shape != (local_batch_size, self.config.seq_window_size, self.config.hidden_size):
                        logger.warning(f"Reshaping hidden_states from {hidden_states.shape} to ({local_batch_size}, {self.config.seq_window_size}, {self.config.hidden_size})")
                        hidden_states = hidden_states.view(local_batch_size, self.config.seq_window_size, self.config.hidden_size)

                except Exception as e:
                    logger.error(f"Error in block {i}: {str(e)}")
                    logger.warning("Falling back to standard attention for this block.")
                    # Fallback to standard attention
                    self.h[i].attn.attention.use_flash_attention = False
                    outputs = block(
                        hidden_states,
                        attention_mask=layer_attention_mask,
                        layer_past=layer_past,
                        head_mask=head_mask[i],
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                    )
                    hidden_states, extra_return_info = outputs
                logger.debug(f"After block {i}: hidden_states shape: {hidden_states.shape}")

                # Ensure hidden_states has the correct shape
                if hidden_states.shape != (local_batch_size, self.config.seq_window_size, self.config.hidden_size):
                    logger.warning(f"Reshaping hidden_states from {hidden_states.shape} to ({local_batch_size}, {self.config.seq_window_size}, {self.config.hidden_size})")
                    hidden_states = hidden_states.view(local_batch_size, self.config.seq_window_size, self.config.hidden_size)
                logger.debug(f"After block {i}: hidden_states shape: {hidden_states.shape}")

                # Apply gradient clipping if enabled
                if self.training and self.use_grad_value_clipping:
                    # Check if any parameters have gradients
                    params_with_grad = [p for p in self.parameters() if p.grad is not None]
                    
                    if params_with_grad:
                        try:
                            torch.nn.utils.clip_grad_value_(params_with_grad, self.clip_grad_value)
                        except RuntimeError as e:
                            logger.warning(f"Error during gradient clipping: {str(e)}")
                            logger.info("Skipping gradient clipping for this step.")
                    else:
                        logger.warning("No gradients found. Skipping gradient clipping.")
                    
                # Ensure hidden_states has the correct batch size
                if hidden_states.size(0) != local_batch_size:
                    logger.warning(f"Adjusting hidden_states batch size from {hidden_states.size(0)} to {local_batch_size}")
                    hidden_states = hidden_states[:local_batch_size]

            logger.debug(f"After block {i}: hidden_states shape: {hidden_states.shape}")
            if use_cache is True:
                presents = presents + (extra_return_info.get("present"),)

            if output_attentions:
                all_self_attentions = all_self_attentions + (extra_return_info.get("attn_weights"),)

        hidden_states = self.ln_f(hidden_states)
        if hasattr(self, 'bn_f'):
            logger.debug(f"Applying batch normalization. hidden_states shape: {hidden_states.shape}")
            if hidden_states.dim() == 3:
                hidden_states = self.bn_f(hidden_states.transpose(1, 2)).transpose(1, 2)
            elif hidden_states.dim() == 2:
                hidden_states = self.bn_f(hidden_states.unsqueeze(-1)).squeeze(-1)
            else:
                logger.warning(f"Unexpected hidden_states dimensions: {hidden_states.dim()}. Skipping batch normalization.")
            logger.debug(f"hidden_states shape after batch norm: {hidden_states.shape}")

        # Ensure hidden_states is 3D before returning
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(1)
            logger.debug(f"Expanded hidden_states shape before return: {hidden_states.shape}")

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        logger.debug(f"Input embeds shape: {input_embeds.shape}")
        logger.debug(f"Hidden states shape after each layer: {hidden_states.shape}")
        return TransformerOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions if output_attentions else None
        )

    def _adjust_tensor_shape(self, tensor, target_batch_size, name):
        if tensor is None:
            return None
        if tensor.size(0) != target_batch_size:
            logger.warning(f"Adjusting {name} batch size from {tensor.size(0)} to {target_batch_size}.")
            if tensor.size(0) < target_batch_size:
                # Repeat the tensor to match the target batch size
                repeat_factor = (target_batch_size + tensor.size(0) - 1) // tensor.size(0)
                tensor = tensor.repeat(repeat_factor, *([1] * (tensor.dim() - 1)))
            return tensor[:target_batch_size]
        return tensor
    
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