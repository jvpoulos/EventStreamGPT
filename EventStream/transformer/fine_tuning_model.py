"""A model for fine-tuning on classification tasks."""
import torch
import math
from torch import nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from torchmetrics.classification import BinaryAUROC
from pytorch_lightning.callbacks import EarlyStopping

from ..data.types import PytorchBatch
from .config import StructuredEventProcessingMode, StructuredTransformerConfig, OptimizationConfig
from .model_output import StreamClassificationModelOutput
from .transformer import (
    ConditionallyIndependentPointProcessTransformer,
    NestedAttentionPointProcessTransformer,
    StructuredTransformerPreTrainedModel,
    ConditionallyIndependentPointProcessInputLayer
)
from .utils import safe_masked_max, safe_weighted_avg

from ..data.vocabulary import VocabularyConfig

import logging

# Configure the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class CustomConditionallyIndependentPointProcessTransformer(ConditionallyIndependentPointProcessTransformer):
    def __init__(self, config: StructuredTransformerConfig, vocabulary_config: VocabularyConfig):
        oov_index = max(vocabulary_config.vocab_sizes_by_measurement.values()) + 1
        print(f"CustomConditionallyIndependentPointProcessTransformer: oov_index = {oov_index}")
        super().__init__(config, vocabulary_config, oov_index=oov_index)

        if config.use_layer_norm:
            self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        else:
            self.ln_f = nn.Identity()  # Use Identity if layer norm is not required

        # Initialize weights
        self.initialize_weights()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            embedding_dim = module.embedding_dim
            std = math.sqrt(1.0 / embedding_dim)
            nn.init.normal_(module.weight, mean=0, std=std)

    def _init_input_embeddings(self):
        if hasattr(self.input_layer, 'data_embedding_layer'):
            data_embedding_layer = self.input_layer.data_embedding_layer
            if hasattr(data_embedding_layer, 'embedding'):
                embedding = data_embedding_layer.embedding
                embedding_dim = embedding.embedding_dim
                std = math.sqrt(1.0 / embedding_dim)
                nn.init.normal_(embedding.weight, mean=0, std=std)
            elif hasattr(data_embedding_layer, 'categorical_embedding'):
                embedding = data_embedding_layer.categorical_embedding
                embedding_dim = embedding.embedding_dim
                std = math.sqrt(1.0 / embedding_dim)
                nn.init.normal_(embedding.weight, mean=0, std=std)

        if hasattr(self.input_layer, 'time_embedding_layer'):
            if isinstance(self.input_layer.time_embedding_layer, nn.Embedding):
                embedding_dim = self.input_layer.time_embedding_layer.embedding_dim
                std = math.sqrt(1.0 / embedding_dim)
                nn.init.normal_(self.input_layer.time_embedding_layer.weight, mean=0, std=std)
            elif hasattr(self.input_layer.time_embedding_layer, 'sin_div_term') and \
                 hasattr(self.input_layer.time_embedding_layer, 'cos_div_term'):
                # For learnable sinusoidal embeddings
                nn.init.normal_(self.input_layer.time_embedding_layer.sin_div_term)
                nn.init.normal_(self.input_layer.time_embedding_layer.cos_div_term)

    def initialize_weights(self):
        # Apply Xavier initialization to all parameters except embeddings
        self.apply(self._init_weights)
        
        # Apply Gaussian initialization to input embeddings
        self._init_input_embeddings()

        # Initialize other components if needed
        if hasattr(self, 'encoder'):
            if hasattr(self.encoder, 'initialize_weights'):
                self.encoder.initialize_weights()
            else:
                self.encoder.apply(self._init_weights)

        if hasattr(self, 'decoder'):
            if hasattr(self.decoder, 'initialize_weights'):
                self.decoder.initialize_weights()
            else:
                self.decoder.apply(self._init_weights)

class ESTForStreamClassification(StructuredTransformerPreTrainedModel):
    def __init__(self, config: StructuredTransformerConfig, vocabulary_config: VocabularyConfig, optimization_config: OptimizationConfig):
        super().__init__(config)
        self.config = config
        self.vocabulary_config = vocabulary_config
        self.optimization_config = optimization_config

        if self._uses_dep_graph:
            self.encoder = NestedAttentionPointProcessTransformer(config)
        else:
            self.encoder = CustomConditionallyIndependentPointProcessTransformer(config, vocabulary_config)
        
        self.pooling_method = config.task_specific_params["pooling_method"]
        
        self.logit_layer = torch.nn.Linear(config.hidden_size, 1)
        self.criteria = torch.nn.BCEWithLogitsLoss()
        self.dropout = torch.nn.Dropout(config.intermediate_dropout)
        self.static_encoder = nn.Linear(8, config.hidden_size)
        
        # Add embeddings for categorical variables
        self.categorical_embeddings = nn.ModuleDict({
            'Female': nn.Embedding(2, config.hidden_size),
            'Married': nn.Embedding(2, config.hidden_size),
            'GovIns': nn.Embedding(2, config.hidden_size),
            'English': nn.Embedding(2, config.hidden_size),
            'Veteran': nn.Embedding(2, config.hidden_size)
        })
        
        # Linear layer for continuous variables
        self.continuous_encoder = nn.Linear(3, config.hidden_size)

        # Add a linear layer for dynamic_values
        self.dynamic_values_encoder = nn.Linear(1, config.hidden_size)

        # Create normalization layers
        self.layer_norm = nn.LayerNorm(config.hidden_size) if config.use_layer_norm else nn.Identity()
        self.batch_norm = nn.BatchNorm1d(config.hidden_size) if config.use_batch_norm else nn.Identity()

        self.intermediate = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            self.layer_norm,
            self.batch_norm,
            self.dropout
        )

        self.static_weight = nn.Parameter(torch.tensor(config.static_embedding_weight))
        self.dynamic_weight = nn.Parameter(torch.tensor(config.dynamic_embedding_weight))

        self.apply(self._init_weights)

        # Initialize AUROC
        self.auroc = BinaryAUROC()

        self.max_grad_norm = getattr(config, 'max_grad_norm', 1.0)
        self.use_grad_value_clipping = getattr(optimization_config, 'use_grad_value_clipping', False)
        self.clip_grad_value = getattr(optimization_config, 'clip_grad_value', 1.0)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            embedding_dim = module.embedding_dim
            std = math.sqrt(1.0 / embedding_dim)
            nn.init.normal_(module.weight, mean=0, std=std)

    def _init_input_embeddings(self):
        for embedding in self.categorical_embeddings.values():
            embedding_dim = embedding.embedding_dim
            std = math.sqrt(1.0 / embedding_dim)
            nn.init.normal_(embedding.weight, mean=0, std=std)

        # Initialize the continuous encoder (which is essentially an embedding for numerical values)
        nn.init.normal_(self.continuous_encoder.weight, mean=0, std=math.sqrt(1.0 / self.config.numerical_embedding_dim))
        if self.continuous_encoder.bias is not None:
            nn.init.zeros_(self.continuous_encoder.bias)

    def initialize_weights(self):
        # Apply Xavier initialization to all parameters except embeddings
        self.apply(self._init_weights)
        
        # Apply Gaussian initialization to input embeddings
        self._init_input_embeddings()

        # Initialize the encoder separately
        if hasattr(self.encoder, 'initialize_weights'):
            self.encoder.initialize_weights()

    @classmethod
    def from_pretrained(cls, pretrained_weights_fp, config, vocabulary_config, optimization_config):
        model = cls(config, vocabulary_config, optimization_config)
        # Load pretrained weights here
        return model

    def forward(self, batch: dict, labels=None):
        device = self.logit_layer.weight.device
        
        # Process dynamic data
        dynamic_indices = batch['dynamic_indices'].to(device)
        
        # Check if dynamic_values is present in the batch
        if 'dynamic_values' in batch:
            dynamic_values = batch['dynamic_values'].to(device)
        else:
            # If not present, create a tensor of zeros with the same shape as dynamic_indices
            dynamic_values = torch.zeros_like(dynamic_indices, dtype=torch.float32).to(device)
        
        # Validate vocab size
        max_index = dynamic_indices.max().item()
        if max_index >= self.config.vocab_size:
            raise ValueError(f"Max index in dynamic_indices ({max_index}) is >= vocab_size ({self.config.vocab_size})")

        # Process static data
        static_categorical = {k: batch[k].to(device) for k in ['Female', 'Married', 'GovIns', 'English', 'Veteran']}
        static_continuous = torch.stack([
            batch['InitialA1c'].to(device),
            batch['AgeYears'].to(device), 
            batch['SDI_score'].to(device)
        ], dim=1)
        
        # Encode dynamic data
        pytorch_batch = PytorchBatch(dynamic_indices=dynamic_indices)
        encoded = self.encoder(pytorch_batch).last_hidden_state
        
        # Extract relevant encoded information
        event_encoded = encoded[:, :, -1, :] if self._uses_dep_graph else encoded
        
        # Apply pooling to dynamic data
        if self.pooling_method == "mean":
            pooled_dynamic = event_encoded.mean(dim=1)
        elif self.pooling_method == "max":
            pooled_dynamic = event_encoded.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")
        
        # Encode static data
        static_cat_embeds = [self.categorical_embeddings[k](static_categorical[k]) for k in static_categorical]
        static_cat_embed = torch.stack(static_cat_embeds, dim=1).mean(dim=1)
        static_num_embed = self.continuous_encoder(static_continuous)

        # Encode dynamic_values
        dynamic_values_embed = self.dynamic_values_encoder(dynamic_values.unsqueeze(-1))
        pooled_dynamic_values = dynamic_values_embed.mean(dim=1)  # or use max pooling if preferred

        # Combine static embeddings
        static_embed = static_cat_embed + static_num_embed

        # Combine dynamic and static embeddings
        combined_embed = self.static_weight * static_embed + self.dynamic_weight * (pooled_dynamic + pooled_dynamic_values)
        
        # Apply intermediate layers and dropout
        intermediate = self.intermediate(combined_embed)
        
        # Get logits and probabilities
        logits = self.logit_layer(intermediate).squeeze(-1)  # Ensure logits are squeezed
        probs = torch.sigmoid(logits)
        
        # Compute loss and metrics if labels are provided
        loss, accuracy, auc = None, None, None
        if labels is not None:
            labels = labels.to(logits.device).float()
            
            # Ensure labels have the same shape as logits
            if labels.dim() == 1 and logits.dim() == 1:
                pass  # Both are 1D, no need to adjust
            elif labels.dim() == 0 and logits.dim() == 1:
                labels = labels.unsqueeze(0)  # Make labels 1D to match logits
            elif labels.dim() == 1 and logits.dim() == 0:
                logits = logits.unsqueeze(0)  # Make logits 1D to match labels
            
            loss = self.criteria(logits, labels)
            
            with torch.no_grad():
                self.auroc.update(probs, labels.int())
                auc = self.auroc.compute()
                accuracy = ((probs > 0.5) == labels).float().mean()
        
        # Collect debugging information
        debug_info = {
            "encoded_mean": encoded.mean().item(),
            "encoded_std": encoded.std().item(),
            "combined_embed_mean": combined_embed.mean().item(),
            "combined_embed_std": combined_embed.std().item(),
            "logits_mean": logits.mean().item(),
            "logits_std": logits.std().item(),
        }
        
        return StreamClassificationModelOutput(
            loss=loss,
            preds=logits,
            labels=labels,
            accuracy=accuracy,
            auc=auc,
            debug_info=debug_info
        )

    @property
    def _uses_dep_graph(self):
        return self.config.structured_event_processing_mode == StructuredEventProcessingMode.NESTED_ATTENTION