"""A model for fine-tuning on classification tasks."""
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from torchmetrics.classification import BinaryAUROC
from pytorch_lightning.callbacks import EarlyStopping

from ..data.types import PytorchBatch
from .config import StructuredEventProcessingMode, StructuredTransformerConfig
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

        self.input_layer = ConditionallyIndependentPointProcessInputLayer(
            config,
            vocabulary_config.vocab_sizes_by_measurement,
            oov_index=oov_index,
            do_use_sinusoidal=config.do_use_sinusoidal  # Pass the attribute here
        )

class ESTForStreamClassification(StructuredTransformerPreTrainedModel):
    def __init__(self, config: StructuredTransformerConfig, vocabulary_config: VocabularyConfig):
        super().__init__(config)
        if self._uses_dep_graph:
            self.encoder = NestedAttentionPointProcessTransformer(config)
        else:
            self.encoder = CustomConditionallyIndependentPointProcessTransformer(config, vocabulary_config)
        
        self.pooling_method = config.task_specific_params["pooling_method"]
        
        # Output layer for binary classification
        self.logit_layer = torch.nn.Linear(config.hidden_size, 1)
        
        # Loss function
        self.criteria = torch.nn.BCEWithLogitsLoss()
        
        # Initialize weights and apply final processing
        self.post_init()

        # Initialize AUROC
        self.auroc = BinaryAUROC()

        # Initialize gradient clipping
        self.max_grad_norm = getattr(config, 'max_grad_norm', 1.0)

        # Initialize dropout
        self.dropout = torch.nn.Dropout(config.intermediate_dropout)

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
        
        # Adjust the input size of the first layer in self.intermediate
        self.intermediate = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            LayerNorm(config.hidden_size),
            self.dropout
        )

        self.static_weight = nn.Parameter(torch.tensor(config.static_embedding_weight))
        self.dynamic_weight = nn.Parameter(torch.tensor(config.dynamic_embedding_weight))
        
        def forward(self, batch: dict, labels=None):
            device = self.logit_layer.weight.device
            
            # Process dynamic data
            dynamic_indices = batch['dynamic_indices'].to(device)
            dynamic_counts = batch['dynamic_counts'].to(device)
            
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
            pytorch_batch = PytorchBatch(dynamic_indices=dynamic_indices, dynamic_counts=dynamic_counts)
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

            # Combine static embeddings
            static_embed = static_cat_embed + static_num_embed

            # Combine dynamic and static embeddings
            combined_embed = self.static_weight * static_embed + self.dynamic_weight * pooled_dynamic
            
            # Apply intermediate layers and dropout
            intermediate = self.dropout(self.intermediate(combined_embed))
            
            # Get logits and probabilities
            logits = self.logit_layer(intermediate).squeeze(-1)
            probs = torch.sigmoid(logits)
            
            # Compute loss and metrics if labels are provided
            loss, accuracy, auc = None, None, None
            if labels is not None:
                labels = labels.to(device).float()
                loss = self.criteria(logits, labels)
                with torch.no_grad():
                    self.auroc.update(probs, labels.int())
                    auc = self.auroc.compute()
                    accuracy = ((probs > 0.5) == labels).float().mean()
            
            # Apply gradient clipping if in training mode
            if self.training:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
            else:
                grad_norm = torch.tensor(0.0)
            
            # Collect debugging information
            debug_info = {
                "encoded_mean": encoded.mean().item(),
                "encoded_std": encoded.std().item(),
                "combined_embed_mean": combined_embed.mean().item(),
                "combined_embed_std": combined_embed.std().item(),
                "logits_mean": logits.mean().item(),
                "logits_std": logits.std().item(),
                "gradient_norm": grad_norm.item(),
                "max_dynamic_index": dynamic_indices.max().item(),
            }
            
            logger.debug(f"Debug info: {debug_info}")
            
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