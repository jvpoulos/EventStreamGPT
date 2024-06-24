"""The conditionally independent event stream GPT model."""
from typing import Any

import torch

from ..data.types import DataModality, PytorchBatch
from ..data.vocabulary import VocabularyConfig  # Add this import statement
from .config import StructuredEventProcessingMode, StructuredTransformerConfig
from .generation.generation_utils import StructuredGenerationMixin
from .model_output import (
    GenerativeOutputLayerBase,
    GenerativeSequenceModelLabels,
    GenerativeSequenceModelLosses,
    GenerativeSequenceModelOutput,
    GenerativeSequenceModelPredictions,
)
from .transformer import (
    ConditionallyIndependentPointProcessTransformer,
    StructuredTransformerPreTrainedModel,
    expand_mask,
    time_from_deltas,
)

class ConditionallyIndependentGenerativeOutputLayer(GenerativeOutputLayerBase):
    def __init__(
        self,
        config: StructuredTransformerConfig,
        vocab_offsets_by_measurement: dict[str, int],
        vocab_sizes_by_measurement: dict[str, int],
    ):
        super().__init__(config)
        if config.structured_event_processing_mode != StructuredEventProcessingMode.CONDITIONALLY_INDEPENDENT:
            raise ValueError(f"{config.structured_event_processing_mode} invalid!")

        self.vocab_offsets_by_measurement = vocab_offsets_by_measurement
        self.vocab_sizes_by_measurement = vocab_sizes_by_measurement

    def forward(
        self,
        batch: PytorchBatch,
        encoded: torch.FloatTensor,
        is_generation: bool = False,
    ) -> GenerativeSequenceModelOutput:
        """Returns the overall model output for the input batch.

        It takes the final hidden states from the encoder and runs them through various output layers to
        predict subsequent event timing and contents. It's difference from a nested attention variant is
        largely in that it predicts everything simultaneously.

        Args:
            batch: The batch of data to process.
            encoded: The encoded representation of the input data.
            is_generation: Whether or not we are in generation mode. If so, the output predictions are for the
                next event for both time and event contents; if not, then we shift the event contents
                predictoin back by one event in order to align with the labels.
        """

        # These are the containers we'll use to process the outputs
        classification_dists_by_measurement = {}
        classification_losses_by_measurement = None if is_generation else {}
        classification_labels_by_measurement = None if is_generation else {}
        regression_dists = {}
        regression_loss_values = None if is_generation else {}
        regression_labels = None if is_generation else {}
        regression_indices = None if is_generation else {}

        classification_measurements = set(self.classification_mode_per_measurement.keys())
        regression_measurements = set(
            self.config.measurements_for(DataModality.MULTIVARIATE_REGRESSION)
            + self.config.measurements_for(DataModality.UNIVARIATE_REGRESSION)
        )

        # encoded is of shape: (batch size, sequence length, config.hidden_size)
        bsz, seq_len, _ = encoded.shape
        whole_event_encoded = encoded

        # In this case, the whole_event_encoded representation actually is used to predict the next event's
        # contents, so it is what we want if we are in generative mode, but if we are not in generative mode
        # then to make it align with the labels we need to shift it to be in the right form. In particular, we
        # prepend a vector of zeros to be used to predict the contents of the first event (excluding the TTE
        # of the first event which is guaranteed to be zero) and we _don't_ predict the contents of the event
        # after the end of this sequence (as we have no way to judge them).

        if is_generation:
            for_event_contents_prediction = whole_event_encoded
        else:
            for_event_contents_prediction = torch.cat(
                (
                    torch.zeros_like(whole_event_encoded[:, 0, :]).unsqueeze(1),
                    whole_event_encoded[:, :-1, :],
                ),
                dim=1,
            )

        classification_out = self.get_classification_outputs(
            batch,
            for_event_contents_prediction,
            classification_measurements,
        )

        print("Shape of for_event_contents_prediction:", for_event_contents_prediction.shape)
        print("Shape of self.ClassificationLayer(for_event_contents_prediction):", self.ClassificationLayer(for_event_contents_prediction).shape)

        if not is_generation and "stream_labels" in batch and batch["stream_labels"] is not None:
            if "A1cGreaterThan7" in batch["stream_labels"]:
                a1c_greater_than_7_labels = batch["stream_labels"]["A1cGreaterThan7"]
                if a1c_greater_than_7_labels is not None:
                    classification_labels_by_measurement["A1cGreaterThan7"] = a1c_greater_than_7_labels
            else:
                print("Warning: 'A1cGreaterThan7' labels not found in the batch['stream_labels'].")

        regression_out = self.get_regression_outputs(
            batch,
            for_event_contents_prediction,
            regression_measurements,
            is_generation=is_generation,
        )
        regression_dists.update(regression_out[1])
        if not is_generation:
            regression_loss_values.update(regression_out[0])
            regression_labels.update(regression_out[2])
            regression_indices.update(regression_out[3])

        TTE_LL_overall, TTE_dist, TTE_true = self.get_TTE_outputs(
            batch,
            whole_event_encoded,
            is_generation=is_generation,
        )

        return GenerativeSequenceModelOutput(
            **{
                "loss": (
                    sum(classification_losses_by_measurement.values())
                    + sum(regression_loss_values.values())
                    - TTE_LL_overall
                )
                if not is_generation
                else None,
                "losses": GenerativeSequenceModelLosses(
                    **{
                        "classification": classification_losses_by_measurement,
                        "regression": regression_loss_values,
                        "time_to_event": None if is_generation else -TTE_LL_overall,
                    }
                ),
                "preds": GenerativeSequenceModelPredictions(
                    classification=classification_dists_by_measurement,
                    regression=regression_dists,
                    regression_indices=regression_indices,
                    time_to_event=TTE_dist,
                ),
                "labels": GenerativeSequenceModelLabels(
                    classification=classification_labels_by_measurement,
                    regression=regression_labels,
                    regression_indices=regression_indices,
                    time_to_event=None if is_generation else TTE_true,
                ),
                "event_mask": batch["event_mask"],
                "dynamic_values_mask": batch["dynamic_values_mask"],
            }
        )


class CIPPTForGenerativeSequenceModeling(StructuredGenerationMixin, StructuredTransformerPreTrainedModel):
    def __init__(
        self,
        config: StructuredTransformerConfig,
        vocabulary_config: VocabularyConfig,
    ):
        super().__init__(config)

        if config.structured_event_processing_mode != StructuredEventProcessingMode.CONDITIONALLY_INDEPENDENT:
            raise ValueError(f"{config.structured_event_processing_mode} invalid!")

        self.encoder = ConditionallyIndependentPointProcessTransformer(config, vocabulary_config)
        self.output_layer = ConditionallyIndependentGenerativeOutputLayer(
            config,
            vocab_offsets_by_measurement=vocabulary_config.vocab_offsets_by_measurement,
            vocab_sizes_by_measurement=vocabulary_config.vocab_sizes_by_measurement,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def prepare_inputs_for_generation(
        self, batch: PytorchBatch, past: tuple | None = None, **kwargs
    ) -> dict[str, Any]:
        """Returns model keyword arguments that have been modified for generation purposes.

        Args:
            batch: The batch of data to be transformed.
            past: The past state of the model, if any. If specified, it must be a tuple containing the past
                values over prior layers and heads.

            **kwargs: Additional keyword arguments. If "use_cache" is set in the kwargs to False, then the
                past state is ignored. If not, then the past state is passed through the model to accelerate
                generation, if past is not None then the batch is trimmed to the last element in the sequence,
                and the sequential attention mask is pre-computed.

        Raises:
            ValueError: If the past state is malformed or if there is a dep_graph_el_generation_target in the
                kwargs that is not None.
        """
        # only last sequence element in the batch if past is defined in kwargs
        batch.time = time_from_deltas(batch)

        use_cache = kwargs.get("use_cache", False)
        if not use_cache:
            return {**kwargs, "batch": batch}

        seq_attention_mask = expand_mask(batch.event_mask, batch.time_delta.dtype)

        dep_graph_el_generation_target = kwargs.get("dep_graph_el_generation_target", None)
        if dep_graph_el_generation_target is not None:
            raise ValueError(
                f"Can't use dep_graph_el_generation_target ({dep_graph_el_generation_target}) "
                "in a conditionally independent model."
            )

        match past:
            case None:
                pass

            case tuple():
                batch = batch.last_sequence_element_unsqueezed()

            case _:
                raise ValueError(f"{past} malformed!")

        return {
            **kwargs,
            "seq_attention_mask": seq_attention_mask,
            "batch": batch,
            "past": past,
        }
        
def forward(
    self,
    dynamic_indices: torch.Tensor,
    dynamic_counts: torch.Tensor | None = None,
    batch: PytorchBatch | None = None,
    input_embeds: torch.Tensor | None = None,
    past: tuple[torch.FloatTensor] | None = None,
    seq_attention_mask: torch.Tensor | None = None,
    head_mask: torch.Tensor | None = None,
    use_cache: bool | None = None,
    output_attentions: bool | None = None,
    output_hidden_states: bool | None = None,
    return_dict: bool | None = None,
    is_generation: bool = False,
) -> GenerativeSequenceModelOutput:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if batch is None:
        batch = PytorchBatch(
            dynamic_indices=dynamic_indices,
            dynamic_counts=dynamic_counts,
        )

    if input_embeds is None:
        input_embeds = self.encoder.input_layer(batch)

    encoded = self.encoder(
        batch=batch,
        input_embeds=input_embeds,
        past=past,
        seq_attention_mask=seq_attention_mask,
        head_mask=head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
    )

    output = self.output_layer(batch, encoded.last_hidden_state, is_generation=is_generation)

    if use_cache:
        output['past_key_values'] = encoded.past_key_values

    if output_attentions:
        output['attentions'] = encoded.attentions

    if output_hidden_states:
        output['hidden_states'] = encoded.hidden_states

    return output