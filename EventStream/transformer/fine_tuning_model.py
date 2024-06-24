"""A model for fine-tuning on classification tasks."""
import torch

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

logger = logging.getLogger(__name__)

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

        is_binary = config.id2label == {0: False, 1: True}
        if is_binary:
            assert config.num_labels == 2
            self.logit_layer = torch.nn.Linear(config.hidden_size, 1).to(config.device)
            self.criteria = torch.nn.BCEWithLogitsLoss()
        else:
            self.logit_layer = torch.nn.Linear(config.hidden_size, config.num_labels).to(config.device)
            self.criteria = torch.nn.CrossEntropyLoss()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, batch: dict, labels=None):
        device = self.logit_layer.weight.device
        
        dynamic_indices = batch['dynamic_indices'].to(device)
        dynamic_counts = batch['dynamic_counts'].to(device)
        
        pytorch_batch = PytorchBatch(
            dynamic_indices=dynamic_indices,
            dynamic_counts=dynamic_counts
        )

        encoded = self.encoder(batch=pytorch_batch).last_hidden_state

        event_encoded = encoded[:, :, -1, :] if self._uses_dep_graph else encoded

        logits = self.logit_layer(event_encoded.mean(dim=1))

        if labels is not None:
            labels = labels.to(logits.device)
            loss = self.criteria(logits, labels)
        else:
            loss = None

        return StreamClassificationModelOutput(
            loss=loss,
            preds=logits,
            labels=labels,
        )
        
    @property
    def _uses_dep_graph(self):
        return self.config.structured_event_processing_mode == StructuredEventProcessingMode.NESTED_ATTENTION