"""A model for fine-tuning on classification tasks."""
import torch
import math
from torch import nn
import torch.utils.checkpoint
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

class ESTForStreamClassification(StructuredTransformerPreTrainedModel):
    def __init__(self, config: StructuredTransformerConfig, vocabulary_config: VocabularyConfig, optimization_config: OptimizationConfig, oov_index: int):
        super().__init__(config)
        self.config = config
        self.vocabulary_config = vocabulary_config
        self.optimization_config = optimization_config
        self.oov_index = oov_index

        self._current_epoch = 0  # Use a private attribute for current_epoch

        # Add a default save_dir if it's not present in the config
        if not hasattr(self.config, 'save_dir'):
            self.config.save_dir = './model_outputs'  # You can change this to any default path you prefer

        # Set OOV index based on vocabulary_config
        dynamic_indices_vocab_size = vocabulary_config.vocab_sizes_by_measurement.get("dynamic_indices", 0)
        self.oov_index = dynamic_indices_vocab_size + 1

        if self._uses_dep_graph:
            self.encoder = NestedAttentionPointProcessTransformer(self.config)
        else:
            self.encoder = ConditionallyIndependentPointProcessTransformer(self.config, vocabulary_config, oov_index=self.oov_index)
       
        # Convert encoder to use SyncBatchNorm
        self.encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)

        self.pooling_method = config.task_specific_params["pooling_method"]
        
        self.logit_layer = torch.nn.Linear(config.hidden_size, 1)
        self.criteria = torch.nn.BCEWithLogitsLoss()
        self.dropout = torch.nn.Dropout(config.intermediate_dropout)
        
        # Add embeddings for categorical variables
        self.categorical_embeddings = nn.ModuleDict({
            'Female': nn.Embedding(2, config.hidden_size),
            'Married': nn.Embedding(2, config.hidden_size),
            'GovIns': nn.Embedding(2, config.hidden_size),
            'English': nn.Embedding(2, config.hidden_size),
            'Veteran': nn.Embedding(2, config.hidden_size)
        })
        
        # Linear layer for continuous variables (including SDI_score)
        self.continuous_encoder = nn.Linear(3, config.hidden_size)

        # Add a linear layer for dynamic_values
        self.dynamic_values_encoder = nn.Linear(1, config.hidden_size)

        # Create normalization layers
        self.layer_norm = nn.LayerNorm(config.hidden_size) if config.use_layer_norm else nn.Identity()
        self.batch_norm = nn.BatchNorm1d(config.hidden_size) if config.use_batch_norm else nn.Identity()
        self.dynamic_values_norm = nn.BatchNorm1d(1)

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

        # Add a mask for dynamic_values to handle missing values
        self.dynamic_values_mask = nn.Parameter(torch.ones(1), requires_grad=False)

        # Add a learnable embedding for missing values
        self.missing_value_embedding = nn.Parameter(torch.randn(config.hidden_size))
        
        # Enable gradient checkpointing for the encoder
        self.encoder.gradient_checkpointing = True

    def _set_static_graph(self):
        for module in self.children():
            if hasattr(module, '_set_static_graph'):
                module._set_static_graph()

        # Explicitly set static graph for all InnerBlock modules
        if hasattr(self.encoder, 'h'):
            for block in self.encoder.h:
                if hasattr(block, '_set_static_graph'):
                    block._set_static_graph()
                
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

    @classmethod
    def from_pretrained(cls, pretrained_weights_fp, config, vocabulary_config, optimization_config, oov_index):
        model = cls(config, vocabulary_config, optimization_config, oov_index)
        # Load pretrained weights here
        return model

    @property
    def current_epoch(self):
        return self._current_epoch

    @current_epoch.setter
    def current_epoch(self, value):
        self._current_epoch = value
        if hasattr(self, 'encoder') and hasattr(self.encoder, 'current_epoch'):
            self.encoder.current_epoch = value
            
    def on_train_epoch_start(self):
        self.current_epoch += 1  # Increment the epoch counter
        if hasattr(self, 'encoder') and hasattr(self.encoder, 'current_epoch'):
            self.encoder.current_epoch = self.current_epoch
            
    def forward(self, batch: dict, labels=None):
        device = self.logit_layer.weight.device
        
        # Process dynamic data
        dynamic_indices = batch['dynamic_indices'].to(device)
        dynamic_values = batch.get('dynamic_values')
        if dynamic_values is not None:
            dynamic_values = dynamic_values.to(device)
        
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
        pytorch_batch = {
            'dynamic_indices': dynamic_indices,
            'dynamic_values': dynamic_values
        }
        
        if self.training and self.encoder.gradient_checkpointing:
            encoded = torch.utils.checkpoint.checkpoint(self.encoder, pytorch_batch).last_hidden_state
        else:
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

        # Encode and normalize dynamic_values
        if dynamic_values is not None:
            # Create a mask for non-missing values
            mask = ~torch.isnan(dynamic_values)
            
            # Normalize non-missing values
            dynamic_values_normalized = torch.where(mask, self.dynamic_values_norm(dynamic_values.unsqueeze(1)).squeeze(1), dynamic_values)
            
            # Replace missing values with learnable embedding
            dynamic_values_embed = torch.where(mask.unsqueeze(-1), 
                                               self.dynamic_values_encoder(dynamic_values_normalized.unsqueeze(-1)), 
                                               self.missing_value_embedding.unsqueeze(0).unsqueeze(0))
            
            pooled_dynamic_values = dynamic_values_embed.mean(dim=1)  # or use max pooling if preferred
        else:
            pooled_dynamic_values = self.missing_value_embedding.unsqueeze(0).expand(pooled_dynamic.size(0), -1)

        # Combine static embeddings
        static_embed = static_cat_embed + static_num_embed

        # Combine dynamic and static embeddings
        combined_embed = self.static_weight * static_embed + self.dynamic_weight * (pooled_dynamic + pooled_dynamic_values)
        
        # Apply intermediate layers and dropout
        intermediate = self.intermediate(combined_embed)
        
        # Get logits and probabilities
        logits = self.logit_layer(intermediate).squeeze(-1)
        probs = torch.sigmoid(logits)
        
        # Compute loss and metrics if labels are provided
        loss, accuracy, auc = None, None, None
        if labels is not None:
            labels = labels.to(logits.device).float()
            loss = self.criteria(logits, labels)
            
            with torch.no_grad():
                self.auroc.update(probs, labels.int())
                auc = self.auroc.compute()
                accuracy = ((probs > 0.5) == labels).float().mean()
        
        return StreamClassificationModelOutput(
            loss=loss,
            preds=logits,
            labels=labels,
            accuracy=accuracy,
            auc=auc
        )
        
    @property
    def _uses_dep_graph(self):
        return self.config.structured_event_processing_mode == StructuredEventProcessingMode.NESTED_ATTENTION