"""A model for fine-tuning on classification tasks."""
import torch
import torch.nn.functional as F
from torch.nn import LayerNorm
from torchmetrics.classification import BinaryAUROC

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
        
        # Add intermediate layers
        self.intermediate = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, config.hidden_size),
            torch.nn.ReLU(),
            LayerNorm(config.hidden_size),
            torch.nn.Dropout(config.resid_dropout)
        )
        
        # Output layer for binary classification
        self.logit_layer = torch.nn.Linear(config.hidden_size, 1)
        
        # Loss function
        self.criteria = torch.nn.BCEWithLogitsLoss()
        
        # Initialize weights and apply final processing
        self.post_init()

        # Initialize AUROC
        self.auroc = BinaryAUROC()

    def forward(self, batch: dict, labels=None):
        device = self.logit_layer.weight.device
        
        # Move batch data to the correct device
        dynamic_indices = batch['dynamic_indices'].to(device)
        dynamic_counts = batch['dynamic_counts'].to(device)
        
        # Create PytorchBatch object
        pytorch_batch = PytorchBatch(
            dynamic_indices=dynamic_indices,
            dynamic_counts=dynamic_counts
        )
        
        try:
            # Get encoded representation from the encoder
            encoded = self.encoder(pytorch_batch).last_hidden_state
        except AssertionError as e:
            logger.error(f"AssertionError in encoder: {str(e)}")
            logger.error(f"dynamic_indices shape: {dynamic_indices.shape}")
            logger.error(f"dynamic_counts shape: {dynamic_counts.shape}")
            raise
        
        # Extract relevant encoded information
        if self._uses_dep_graph:
            event_encoded = encoded[:, :, -1, :]
        else:
            event_encoded = encoded
        
        # Apply pooling
        if self.pooling_method == "mean":
            pooled = event_encoded.mean(dim=1)
        elif self.pooling_method == "max":
            pooled = event_encoded.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")
        
        # Apply intermediate layers
        intermediate = self.intermediate(pooled)
        
        # Get logits
        logits = self.logit_layer(intermediate).squeeze(-1)
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        
        # Compute loss if labels are provided
        if labels is not None:
            labels = labels.to(logits.device).float()
            loss = self.criteria(logits, labels)
            
            # Compute additional metrics
            with torch.no_grad():
                self.auroc.update(probs, labels.int())
                auc = self.auroc.compute()
                accuracy = ((probs > 0.5) == labels).float().mean()
        else:
            loss = None
            auc = None
            accuracy = None
        
        # Collect debugging information
        debug_info = {
            "encoded_mean": encoded.mean().item(),
            "encoded_std": encoded.std().item(),
            "pooled_mean": pooled.mean().item(),
            "pooled_std": pooled.std().item(),
            "logits_mean": logits.mean().item(),
            "logits_std": logits.std().item(),
        }
        
        return StreamClassificationModelOutput(
            loss=loss,
            preds=probs,  # Return probabilities instead of logits
            labels=labels,
            auc=auc,
            accuracy=accuracy,
            debug_info=debug_info
        )
        
    @property
    def _uses_dep_graph(self):
        return self.config.structured_event_processing_mode == StructuredEventProcessingMode.NESTED_ATTENTION