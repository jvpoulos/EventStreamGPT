"""A model for fine-tuning on classification tasks."""
import torch

from ..data.types import PytorchBatch
from .config import StructuredEventProcessingMode, StructuredTransformerConfig
from .model_output import StreamClassificationModelOutput
from .transformer import (
    ConditionallyIndependentPointProcessTransformer,
    NestedAttentionPointProcessTransformer,
    StructuredTransformerPreTrainedModel,
)
from .utils import safe_masked_max, safe_weighted_avg


from ..data.vocabulary import VocabularyConfig

class ESTForStreamClassification(StructuredTransformerPreTrainedModel):
    """A model for fine-tuning on classification tasks."""

    def __init__(
        self,
        config: StructuredTransformerConfig,
        vocabulary_config: VocabularyConfig,
    ):
        super().__init__(config)

        self.task = config.finetuning_task
        print(f"self.task type: {type(self.task)}")
        print(f"self.task value: {self.task}")

        if self._uses_dep_graph:
            self.encoder = NestedAttentionPointProcessTransformer(config)
        else:
            self.encoder = ConditionallyIndependentPointProcessTransformer(
                config, vocabulary_config=vocabulary_config  # Pass the vocabulary_config argument
            )

        self.pooling_method = config.task_specific_params["pooling_method"]

        is_binary = config.id2label == {0: False, 1: True}
        if is_binary:
            assert config.num_labels == 2
            self.logit_layer = torch.nn.Linear(config.hidden_size, 1)
            self.criteria = torch.nn.BCEWithLogitsLoss()
        else:
            self.logit_layer = torch.nn.Linear(config.hidden_size, config.num_labels)
            self.criteria = torch.nn.CrossEntropyLoss()

        # Initialize weights and apply final processing
        self.post_init()

    @property
    def _uses_dep_graph(self):
        return self.config.structured_event_processing_mode == StructuredEventProcessingMode.NESTED_ATTENTION

    def forward(self, batch: PytorchBatch, **kwargs) -> StreamClassificationModelOutput:
        """Runs the forward pass through the fine-tuning label prediction.

        Args:
            batch: The batch of data to model.

        Returns:
            A `StreamClassificationModelOutput` object capturing loss, predictions, and labels for the
            fine-tuning task in question.
        """
        encoded = self.encoder(batch, **kwargs).last_hidden_state
        event_encoded = encoded[:, :, -1, :] if self._uses_dep_graph else encoded

        # `event_encoded` is of shape [batch X seq X hidden_dim]. For pooling, I want to put the sequence
        # dimension as last, so we'll transpose.
        event_encoded = event_encoded.transpose(1, 2)

        match self.pooling_method:
            case "cls":
                stream_encoded = event_encoded[:, :, 0]
            case "last":
                stream_encoded = event_encoded[:, :, -1]
            case "max":
                stream_encoded = safe_masked_max(event_encoded, batch["event_mask"])
            case "mean":
                stream_encoded, _ = safe_weighted_avg(event_encoded, batch["event_mask"])
            case _:
                raise ValueError(f"{self.pooling_method} is not a supported pooling method.")

        logits = self.logit_layer(stream_encoded).squeeze(-1)

        if "stream_labels" not in batch or batch["stream_labels"] is None:
            # Labels are missing
            return StreamClassificationModelOutput(
                loss=None,
                preds=logits,
                labels=None,
            )
        
        if isinstance(self.task, str):
            labels = batch["stream_labels"].get(self.task)
        else:
            labels = batch["stream_labels"]

        print("Labels in the forward pass:")
        print(batch["stream_labels"])

        if labels is None:
            # Labels for the specific task are missing
            return StreamClassificationModelOutput(
                loss=None,
                preds=logits,
                labels=None,
            )

        loss = self.criteria(logits, labels)

        return StreamClassificationModelOutput(
            loss=loss,
            preds=logits,
            labels=labels,
        )