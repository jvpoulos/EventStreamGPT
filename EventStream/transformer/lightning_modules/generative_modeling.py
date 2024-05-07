import dataclasses
import json
import os
from pathlib import Path
from typing import Any
import pathlib
from EventStream.utils import JSONableMixin

import lightning as L
import omegaconf
import torch
import torch.multiprocessing
import torchmetrics
from torchmetrics import AUROC
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MultilabelAccuracy,
    MultilabelAUROC,
    MultilabelAveragePrecision,
)
from transformers import get_polynomial_decay_schedule_with_warmup

from ...data.config import PytorchDatasetConfig
from ...data.pytorch_dataset import PytorchDataset
from ...data.types import DataModality, PytorchBatch
from ...data.vocabulary import VocabularyConfig  # Add this import statement
from ...utils import hydra_dataclass, task_wrapper
from ..conditionally_independent_model import CIPPTForGenerativeSequenceModeling
from ..config import (
    Averaging,
    MetricCategories,
    Metrics,
    MetricsConfig,
    OptimizationConfig,
    Split,
    StructuredEventProcessingMode,
    StructuredTransformerConfig,
)
from ..model_output import GenerativeSequenceModelOutput
from ..nested_attention_model import NAPPTForGenerativeSequenceModeling
from ..utils import expand_indexed_regression, str_summary

from dataclasses import dataclass
from omegaconf import MISSING

from dataclasses import asdict, is_dataclass, fields 
import collections

from ..config import Split

def make_json_serializable(item, seen=None):
    if seen is None:
        seen = set()

    if id(item) in seen:
        return str(item)

    seen.add(id(item))

    if is_dataclass(item):
        result = {}
        for field in fields(item):
            value = getattr(item, field.name)
            result[field.name] = make_json_serializable(value, seen)
        return result
    elif isinstance(item, dict):
        return {k: make_json_serializable(v, seen) for k, v in item.items()}
    elif isinstance(item, (list, tuple, set)):
        return [make_json_serializable(i, seen) for i in item]
    elif isinstance(item, (str, int, float, bool, type(None))):
        return item
    elif hasattr(item, '__dict__'):
        # Convert objects with __dict__ (that are not dataclasses) into dicts
        # Handle tuple keys by converting them to strings
        return {str(k): make_json_serializable(v, seen) for k, v in item.__dict__.items()}
    else:
        return str(item)

def sanitize_config(config):
    """
    Recursively sanitize a configuration dictionary to be JSON serializable.
    """
    return make_json_serializable(config)


class ESTForGenerativeSequenceModelingLM(L.LightningModule):
    def __init__(
        self,
        config: StructuredTransformerConfig | dict[str, Any],
        optimization_config: OptimizationConfig | dict[str, Any],
        metrics_config: MetricsConfig | dict[str, Any],
        pretraining_metrics_config: MetricsConfig | dict[str, Any],
        final_validation_metrics_config: MetricsConfig | dict[str, Any],
        vocabulary_config: VocabularyConfig,
        pretrained_weights_fp: Path | None = None,
        CLASSIFICATION = {DataModality.SINGLE_LABEL_CLASSIFICATION, DataModality.MULTI_LABEL_CLASSIFICATION},
    ):
        super().__init__()

        # If the configurations are dictionaries, convert them to class objects.
        if isinstance(config, dict):
            config = StructuredTransformerConfig(**config)
        if isinstance(optimization_config, dict):
            optimization_config = OptimizationConfig(**optimization_config)
        if isinstance(metrics_config, dict):
            metrics_config = MetricsConfig(**metrics_config)
        if isinstance(pretraining_metrics_config, dict):
            pretraining_metrics_config = MetricsConfig(**pretraining_metrics_config)
        if isinstance(final_validation_metrics_config, dict):
            final_validation_metrics_config = MetricsConfig(**final_validation_metrics_config)

        self.pretraining_metrics_config = pretraining_metrics_config
        self.final_validation_metrics_config = final_validation_metrics_config

        self.config = config
        self.optimization_config = optimization_config
        self.metrics_config = metrics_config

        if isinstance(config, StructuredTransformerConfig):
            config_dict = config.to_dict()
        else:
            config_dict = config

        self.save_hyperparameters(
            {
                "config": config_dict,
                "optimization_config": optimization_config,  # Pass the object directly
            }
        )
        self.build_metrics()

        match config.structured_event_processing_mode:
            case StructuredEventProcessingMode.NESTED_ATTENTION:
                model_cls = NAPPTForGenerativeSequenceModeling
            case StructuredEventProcessingMode.CONDITIONALLY_INDEPENDENT:
                model_cls = CIPPTForGenerativeSequenceModeling
            case _:
                raise ValueError(
                    f"Unsupported structured event processing mode: {config.structured_event_processing_mode}"
                )

        if pretrained_weights_fp is None:
            self.model = model_cls(config, vocabulary_config)  # Pass vocabulary_config
        else:
            self.model = model_cls.from_pretrained(pretrained_weights_fp, config=config)

    def log_classification_metrics(self, results: GenerativeSequenceModelOutput, split: Split, metrics_config: MetricsConfig):
        for measurement, metrics_dict in self.metrics.items():
            mask = results["event_mask"]

            if not mask.any():
                continue

            for task_type, metrics in metrics_dict.items():
                if task_type in self.CLASSIFICATION and metrics_config.include_metrics.get(split.value, {}).get(MetricCategories.CLASSIFICATION.value, False):
                    _, sample_dist = results["preds"]["classification"][measurement]
                    preds = sample_dist.logits
                    labels = results["labels"]["classification"][measurement]

                    preds = preds[mask]
                    labels = labels[mask].long()

                    auroc_metric = metrics.get("WEIGHTED_AUROC")
                    if auroc_metric is not None:
                        auroc_metric.update(preds, labels)
                        self.log(f"{split}_{measurement}_WEIGHTED_AUROC", auroc_metric, on_step=(split == Split.TRAIN), on_epoch=True)

                    self._log_metric_dict(
                        preds=preds,
                        labels=labels,
                        metrics=metrics,
                        measurement=measurement,
                        split=split,
                        cat=MetricCategories.CLASSIFICATION,
                        metrics_config=metrics_config,
                    )

    def do_log_any(self, split: Split, metric_name=None):
        """Returns True if `metric_name` should be tracked for `split` and any other split."""
        for s in Split.values():
            if self.do_log(s, metric_name):
                return True
        return False

    def do_log(self, split: Split, metric_name=None):
        """Returns True if `metric_name` should be tracked for `split`."""
        if self.metrics_config.do_log_only_loss(split):
            return False

        split_config = self.metrics_config.include_metrics.get(split.value, {})
        if not split_config:
            return False

        if metric_name is None or split_config is True:
            return True

        has_averaging = "_" in metric_name.replace("explained_variance", "")
        if not has_averaging:
            return metric_name in split_config

        parts = metric_name.split("_")
        averaging = parts[0]
        metric = "_".join(parts[1:])

        permissible_averagings = split_config.get(metric, [])
        if permissible_averagings is True or averaging in permissible_averagings:
            return True
        else:
            return False

    def log_loss(self, out, split: Split):
        loss = out.loss
        self.log(f"{split}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        for loss_name, loss_value in out.loss_parts.items():
            self.log(f"{split}_{loss_name}", loss_value, on_step=False, on_epoch=True) 

    def save_pretrained(self, model_dir: str):
        fp = Path(model_dir) / "pretrained_weights"
        self.model.save_pretrained(fp)

    def build_metrics(self):
        """Build the various torchmetrics we'll use to track performance."""

        # For judging our ability to predict time-to-event, we'll use the following scores:
        #   1. Explained Variance
        #   2. Mean Squared Error
        #   3. Mean Squared Log Error
        self.tte_metrics = torch.nn.ModuleDict(
            {
                "MSE": torchmetrics.MeanSquaredError(),
                "MSLE": torchmetrics.MeanSquaredLogError(),
                "explained_variance": torchmetrics.ExplainedVariance(),
            }
        )

        self.metrics = torch.nn.ModuleDict()
        for task_type, measurements in self.config.measurements_per_generative_mode.items():
            for measurement in measurements:
                vocab_size = self.config.vocab_sizes_by_measurement[measurement]

                if measurement not in self.metrics:
                    self.metrics[measurement] = torch.nn.ModuleDict()
                if task_type not in self.metrics[measurement]:
                    self.metrics[measurement][task_type] = torch.nn.ModuleDict()

                match task_type:
                    case DataModality.SINGLE_LABEL_CLASSIFICATION:
                        cat = MetricCategories.CLASSIFICATION
                        metrics = {
                            Metrics.ACCURACY: (
                                MulticlassAccuracy,
                                [Averaging.MACRO, Averaging.WEIGHTED, Averaging.MICRO],
                            ),
                            Metrics.AUROC: (
                                MulticlassAUROC,
                                [Averaging.MACRO, Averaging.WEIGHTED],
                            ),
                            Metrics.AUPRC: (
                                MulticlassAveragePrecision,
                                [Averaging.MACRO, Averaging.WEIGHTED],
                            ),
                        }
                        metric_kwargs = {
                            "num_classes": vocab_size,
                            "ignore_index": 0,
                            "validate_args": self.metrics_config.do_validate_args,
                        }
                    case DataModality.MULTI_LABEL_CLASSIFICATION:
                        cat = MetricCategories.CLASSIFICATION
                        metrics = {
                            Metrics.ACCURACY: (
                                MultilabelAccuracy,
                                [Averaging.MACRO, Averaging.WEIGHTED, Averaging.MICRO],
                            ),
                            Metrics.AUROC: (
                                MultilabelAUROC,
                                [Averaging.MACRO, Averaging.WEIGHTED, Averaging.MICRO],
                            ),
                            Metrics.AUPRC: (
                                MultilabelAveragePrecision,
                                [Averaging.MACRO, Averaging.WEIGHTED, Averaging.MICRO],
                            ),
                        }
                        metric_kwargs = {
                            "num_labels": vocab_size,
                            "validate_args": self.metrics_config.do_validate_args,
                        }
                    case DataModality.UNIVARIATE_REGRESSION:
                        cat = MetricCategories.REGRESSION
                        metrics = {
                            Metrics.MSE: (torchmetrics.MeanSquaredError, [None]),
                            Metrics.EXPLAINED_VARIANCE: (torchmetrics.ExplainedVariance, [None]),
                        }
                        metric_kwargs = {}
                    case DataModality.MULTIVARIATE_REGRESSION:
                        cat = MetricCategories.REGRESSION
                        metrics = {
                            Metrics.MSE: (torchmetrics.MeanSquaredError, [None]),
                            Metrics.EXPLAINED_VARIANCE: (
                                torchmetrics.ExplainedVariance,
                                [Averaging.MACRO, Averaging.WEIGHTED],
                            ),
                        }
                        metric_kwargs = {}
                    case _:
                        raise ValueError(f"Unrecognized modality {task_type}!")

                auc_kwargs = {
                    **metric_kwargs,
                    "thresholds": self.metrics_config.n_auc_thresholds,
                    "compute_on_cpu": True,
                }
                for metric, (metric_cls, averagings) in metrics.items():
                    if metric in (Metrics.AUROC, Metrics.AUPRC):
                        metric_cls_kwargs = {**auc_kwargs}
                    else:
                        metric_cls_kwargs = {**metric_kwargs}

                    for averaging in averagings:
                        if averaging is None:
                            metric_name = str(metric)
                            averaging_kwargs = {}
                        else:
                            metric_name = f"{averaging}_{metric}"
                            if metric == Metrics.EXPLAINED_VARIANCE:
                                if averaging == Averaging.MACRO:
                                    avg_str = "uniform_average"
                                elif averaging == Averaging.WEIGHTED:
                                    avg_str = "variance_weighted"
                                else:
                                    raise ValueError(f"{averaging} not supported for explained variance.")

                                averaging_kwargs = {"multioutput": avg_str}
                            else:
                                averaging_kwargs = {"average": averaging}

                        if self.metrics_config.do_log_any(cat, metric_name):
                            self.metrics[measurement][task_type][metric_name] = metric_cls(
                                **metric_cls_kwargs, **averaging_kwargs
                            )

    def _log_metric_dict(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        metrics: dict[str, torchmetrics.Metric],
        split: Split,
        measurement: str,
        cat: MetricCategories,
        metrics_config: MetricsConfig,
    ):
        """This helper function logs the set of named metrics for the predictions `preds` and labels `labels`."""
        for metric_name, metric in metrics.items():
            if metric_name not in metrics_config.include_metrics.get(split.value, {}).get(cat.value, []):
                continue

            try:
                if split != Split.TRAIN:
                    metric.update(preds, labels)
                else:
                    metric(preds, labels)

                self.log(
                    f"{split}_{measurement}_{metric_name}",
                    metric,
                    batch_size=self.optimization_config.batch_size,
                    sync_dist=True,
                    rank_zero_only=True,
                )
            except (ValueError, IndexError) as e:
                print(
                    f"Failed to compute {metric_name} for {measurement} "
                    f"with preds ({str_summary(preds)}) and labels ({str_summary(labels)}): {e}."
                )

    def log_tte_metrics(self, results: GenerativeSequenceModelOutput, split: Split, metrics_config: MetricsConfig):
        # The output of the model for time-to-event (and for regression targets as well) are pytorch
        # distribution objects, not scalars. So, for some evaluation metrics, we need to sample values from
        # those distributions to assess the metric.
        # TODO(mmd): We should likely be able to control how many samples are used, to minimize variance of
        # these results.
        tte_dist = results["preds"]["time_to_event"]
        tte_preds = tte_dist.sample()

        # After sampling, we also need to slice this down to just those intra-event-times that are actually
        # observed. This means we should drop the last sequence element (slice to `[:, :-1]` (as our observed
        # intra-event-times will only exist for the interior of our sequence), then further filter down to
        # just elements of the prediction for which the next sequence element was not masked
        # (mask via `results['event_mask'][:, 1:]`). We also need to filter the observed labels down to also
        # only be present for sequence elements where the next sequence element was truly observed.
        tte_preds = tte_preds[:, :-1][results["event_mask"][:, 1:]]
        tte_labels = results["labels"]["time_to_event"][results["event_mask"][:, 1:]]

        # Finally, we can log all relevant TTE metrics given these predictions and labels.
        self._log_metric_dict(
            preds=tte_preds,
            labels=tte_labels,
            metrics=self.tte_metrics,
            measurement="TTE",
            split=split,
            cat=MetricCategories.TTE,
            metrics_config=metrics_config,
        )

    def log_regression_metrics(self, results: GenerativeSequenceModelOutput, split: Split, metrics_config: MetricsConfig):
        for measurement, metrics_dict in self.metrics.items():
            mask = results["event_mask"]

            if not mask.any():
                continue

            for task_type, metrics in metrics_dict.items():
                if task_type == DataModality.MULTIVARIATE_REGRESSION and metrics_config.include_metrics.get(split.value, {}).get(MetricCategories.REGRESSION.value, False):
                    vocab_size = self.config.vocab_sizes_by_measurement[measurement]

                    # Here, like for TTE, we need to sample from the returned distribution before we can use
                    # it directly. Here we also need to limit to just those events that are actually observed.
                    # Like above, the assumption here is that preds and labels correspond to predictions for
                    # and labels of the events at their indexed position; not for the subsequent event. So we
                    # don't need to shift `results['event_mask']` here to account for that.
                    _, dist = results["preds"]["regression"][measurement]
                    preds = dist.sample()[mask]
                    labels = results["labels"]["regression"][measurement][mask]

                    # However, as our regression output is actually indexed only to the group keys that are
                    # actually measured (tracked in `results['preds']['regression_indices']`, we need to
                    # expand our predictions and labels to be in the full vocabulary space for the metrics to
                    # work naturally.
                    preds_indices = results["preds"]["regression_indices"][measurement][mask]
                    labels_indices = results["labels"]["regression_indices"][measurement][mask]

                    # We also need to reflect just those data elements for which values were observed:
                    data_el_mask = results["dynamic_values_mask"][mask]

                    preds = preds[data_el_mask]
                    labels = labels[data_el_mask]
                    preds_indices = preds_indices[data_el_mask]
                    labels_indices = labels_indices[data_el_mask]

                    preds_expanded = expand_indexed_regression(preds, preds_indices, vocab_size)
                    labels_expanded = expand_indexed_regression(labels, labels_indices, vocab_size)

                    self._log_metric_dict(
                        preds=preds_expanded,
                        labels=labels_expanded,
                        metrics=metrics,
                        measurement=measurement,
                        split=split,
                        cat=MetricCategories.REGRESSION,
                        metrics_config=metrics_config,
                    )
                elif task_type == DataModality.UNIVARIATE_REGRESSION and metrics_config.include_metrics.get(split.value, {}).get(MetricCategories.REGRESSION.value, False):
                    # Here, like for TTE, we need to sample from the returned distribution before we can use
                    # it directly. Here we also need to limit to just those events that are actually observed.
                    # Like above, the assumption here is that preds and labels correspond to predictions for
                    # and labels of the events at their indexed position; not for the subsequent event. So we
                    # don't need to shift `results['event_mask']` here to account for that.
                    # We ignore the is observed distribution here.
                    _, dist = results["preds"]["regression"][measurement]
                    preds = dist.sample()[mask]
                    labels = results["labels"]["regression"][measurement][mask]

                    self._log_metric_dict(
                        preds=preds,
                        labels=labels,
                        metrics=metrics,
                        measurement=measurement,
                        split=split,
                        cat=MetricCategories.REGRESSION,
                    )

    def log_metrics(self, results: GenerativeSequenceModelOutput, split: Split):
        """Logs metric results for a given output result."""
        log_kwargs = {"batch_size": self.optimization_config.batch_size, "sync_dist": True}

        # Log the loss separately
        self.log(f"{split}_loss", results["loss"], **log_kwargs, rank_zero_only=True)

        if split == Split.TRAIN or split == Split.TUNING:
            metrics_config = self.pretraining_metrics_config
        else:
            metrics_config = self.final_validation_metrics_config

        if MetricCategories.LOSS_PARTS.value in metrics_config.include_metrics.get(split.value, {}):
            # Log other losses
            for loss_name, loss_value in results["losses"].items():
                self.log(f"{split}_{loss_name}", loss_value, **log_kwargs)

        # Log other metrics
        self.log_classification_metrics(results, split, metrics_config)
        self.log_regression_metrics(results, split, metrics_config)
        self.log_tte_metrics(results, split, metrics_config)

        # Per data type
        for measurement, metrics_dict in self.metrics.items():
            mask = results["event_mask"]

            if not mask.any():
                continue

            for task_type, metrics in metrics_dict.items():
                if task_type in self.CLASSIFICATION and self.metrics_config.do_log(
                    split, MetricCategories.CLASSIFICATION
                ):
                    # For now, we ignore the is_observed distribution (the first element of the below tuple).
                    _, sample_dist = results["preds"]["classification"][measurement]
                    preds = sample_dist.logits
                    labels = results["labels"]["classification"][measurement]

                    # We need to filter these down to just those corresponding to observed events. Note that
                    # unlike TTE, the assumption here is that preds and labels correspond to predictions for
                    # and labels of the events at their indexed position; not for the subsequent event. So we
                    # don't need to shift `results['event_mask']` here to account for that.

                    preds = preds[mask]
                    labels = labels[mask].long()

                    self._log_metric_dict(
                        preds=preds,
                        labels=labels,
                        metrics=metrics,
                        measurement=measurement,
                        split=split,
                        cat=MetricCategories.CLASSIFICATION,
                    )

                elif task_type == DataModality.MULTIVARIATE_REGRESSION and self.metrics_config.do_log(
                    split, MetricCategories.REGRESSION
                ):
                    vocab_size = self.config.vocab_sizes_by_measurement[measurement]

                    # Here, like for TTE, we need to sample from the returned distribution before we can use
                    # it directly. Here we also need to limit to just those events that are actually observed.
                    # Like above, the assumption here is that preds and labels correspond to predictions for
                    # and labels of the events at their indexed position; not for the subsequent event. So we
                    # don't need to shift `results['event_mask']` here to account for that.
                    _, dist = results["preds"]["regression"][measurement]
                    preds = dist.sample()[mask]
                    labels = results["labels"]["regression"][measurement][mask]

                    # However, as our regression output is actually indexed only to the group keys that are
                    # actually measured (tracked in `results['preds']['regression_indices']`, we need to
                    # expand our predictions and labels to be in the full vocabulary space for the metrics to
                    # work naturally.
                    preds_indices = results["preds"]["regression_indices"][measurement][mask]
                    labels_indices = results["labels"]["regression_indices"][measurement][mask]

                    # We also need to reflect just those data elements for which values were observed:
                    data_el_mask = results["dynamic_values_mask"][mask]

                    preds = preds[data_el_mask]
                    labels = labels[data_el_mask]
                    preds_indices = preds_indices[data_el_mask]
                    labels_indices = labels_indices[data_el_mask]

                    preds_expanded = expand_indexed_regression(preds, preds_indices, vocab_size)
                    labels_expanded = expand_indexed_regression(labels, labels_indices, vocab_size)

                    self._log_metric_dict(
                        preds=preds_expanded,
                        labels=labels_expanded,
                        metrics=metrics,
                        measurement=measurement,
                        split=split,
                        cat=MetricCategories.REGRESSION,
                    )
                elif task_type == DataModality.UNIVARIATE_REGRESSION and self.metrics_config.do_log(
                    split, MetricCategories.REGRESSION
                ):
                    # Here, like for TTE, we need to sample from the returned distribution before we can use
                    # it directly. Here we also need to limit to just those events that are actually observed.
                    # Like above, the assumption here is that preds and labels correspond to predictions for
                    # and labels of the events at their indexed position; not for the subsequent event. So we
                    # don't need to shift `results['event_mask']` here to account for that.
                    # We ignore the is observed distribution here.
                    _, dist = results["preds"]["regression"][measurement]
                    preds = dist.sample()[mask]
                    labels = results["labels"]["regression"][measurement][mask]

                    self._log_metric_dict(
                        preds=preds,
                        labels=labels,
                        metrics=metrics,
                        measurement=measurement,
                        split=split,
                        cat=MetricCategories.REGRESSION,
                    )

    def training_step(self, batch: PytorchBatch, batch_idx: int) -> torch.Tensor:
        """Training step.

        Skips logging all AUROC, AUPRC, and per_class metric to save compute.
        """
        out = self.model(batch)
        self.log_metrics(out, split=Split.TRAIN)

        # Calculate and log AUROC
        for measurement, metrics_dict in self.metrics.items():
            for task_type, metrics in metrics_dict.items():
                if task_type in self.CLASSIFICATION:
                    _, sample_dist = out["preds"]["classification"][measurement]
                    preds = sample_dist.logits
                    labels = out["labels"]["classification"][measurement].long()
                    mask = out["event_mask"]
                    preds = preds[mask]
                    labels = labels[mask]

                    auroc_metric = metrics.get("WEIGHTED_AUROC")
                    if auroc_metric is not None:
                        auroc_value = auroc_metric(preds, labels)
                        self.log(f"{split}_{measurement}_WEIGHTED_AUROC", auroc_value, on_step=(split == Split.TRAIN), on_epoch=True)

        return out["loss"]

    def validation_step(self, batch: PytorchBatch, batch_idx: int):
        """Validation step.

        Differs from training only in that it does not skip metrics.
        """
        out = self.model(batch)
        self.log_metrics(out, split=Split.TUNING)

        # Calculate and log AUROC
        for measurement, metrics_dict in self.metrics.items():
            for task_type, metrics in metrics_dict.items():
                if task_type in self.CLASSIFICATION:
                    _, sample_dist = out["preds"]["classification"][measurement]
                    preds = sample_dist.logits
                    labels = out["labels"]["classification"][measurement].long()
                    mask = out["event_mask"]
                    preds = preds[mask]
                    labels = labels[mask]

                    auroc_metric = metrics.get("WEIGHTED_AUROC")
                    if auroc_metric is not None:
                        auroc_value = auroc_metric(preds, labels)
                        self.log(f"{split}_{measurement}_WEIGHTED_AUROC", auroc_value, on_step=(split == Split.TRAIN), on_epoch=True)

    def test_step(self, batch: PytorchBatch, batch_idx: int):
        """Test step.

        Differs from training only in that it does not skip metrics.
        """
        out = self.model(batch)
        self.log_metrics(out, split=Split.HELD_OUT)

        # Calculate and log AUROC
        for measurement, metrics_dict in self.metrics.items():
            for task_type, metrics in metrics_dict.items():
                if task_type in self.CLASSIFICATION:
                    _, sample_dist = out["preds"]["classification"][measurement]
                    preds = sample_dist.logits
                    labels = out["labels"]["classification"][measurement].long()
                    mask = out["event_mask"]
                    preds = preds[mask]
                    labels = labels[mask]

                    auroc_metric = metrics.get("WEIGHTED_AUROC")
                    if auroc_metric is not None:
                        auroc_value = auroc_metric(preds, labels)
                        self.log(f"{split}_{measurement}_WEIGHTED_AUROC", auroc_value, on_step=(split == Split.TRAIN), on_epoch=True)

    def configure_optimizers(self):
        """Configures optimizer and learning rate scheduler.

        Currently this module uses the AdamW optimizer, with configurable weight_decay, with a learning rate
        warming up from 0 on a per-step manner to the configurable `self.optimization_config.init_lr`, then
        undergoes polynomial decay as specified via `self.optimization_config`.
        """
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.optimization_config.init_lr,
            weight_decay=self.optimization_config.weight_decay,
        )
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer=opt,
            num_warmup_steps=self.optimization_config.lr_num_warmup_steps,
            num_training_steps=self.optimization_config.max_training_steps,
            power=self.optimization_config.lr_decay_power,
            lr_end=self.optimization_config.end_lr,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


SKIP_CFG_PARAMS = {"seq_attention_layers", "dep_graph_attention_layers", "hidden_size"}


@dataclass
class PretrainConfig:
    from EventStream.data.config import PytorchDatasetConfig
    do_overwrite: bool = False
    seed: int = 42

    config: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "_target_": "EventStream.transformer.config.StructuredTransformerConfig",
            **{
                k: v
                for k, v in StructuredTransformerConfig(measurements_per_dep_graph_level=[]).to_dict().items()
                if k not in SKIP_CFG_PARAMS
            },
        }
    )
    optimization_config: OptimizationConfig = dataclasses.field(default_factory=OptimizationConfig)
    dataset_path: Path = MISSING  # Remove this line
    data_config: PytorchDatasetConfig = dataclasses.field(default_factory=PytorchDatasetConfig)
    pretraining_metrics_config: MetricsConfig = dataclasses.field(
        default_factory=lambda: MetricsConfig(
            include_metrics={Split.TRAIN: {MetricCategories.LOSS_PARTS: True}},
        )
    )
    final_validation_metrics_config: MetricsConfig = dataclasses.field(
        default_factory=lambda: MetricsConfig(do_skip_all_metrics=False)
    )

    trainer_config: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "accelerator": "auto",
            "devices": "auto",
            "detect_anomaly": False,
            "default_root_dir": "${save_dir}/model_checkpoints",
            "log_every_n_steps": 10,
        }
    )

    experiment_dir: str = MISSING
    save_dir: str = MISSING
    wandb_logger_kwargs: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "name": "generative_event_stream_transformer",
            "project": None,
            "team": None,
            "log_model": True,
            "do_log_graph": True,
        }
    )
    wandb_experiment_config_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    do_final_validation_on_metrics: bool = True
    do_use_filesystem_sharing: bool = True

    # compile: bool = True

    def __post_init__(self):
        if "max_epochs" in self.trainer_config:
            raise ValueError("Max epochs is set in the optimization_config, not the trainer config!")
        if "callbacks" in self.trainer_config:
            raise ValueError("Callbacks are built internally, not set via trainer_config!")


@task_wrapper
def train(cfg: PretrainConfig, train_pyd: PytorchDataset, tuning_pyd: PytorchDataset):
    """Runs the end to end training procedure for the pre-training model.

    Args:
        cfg: The pre-training config defining the generative modeling task.
    """
    import wandb
    L.seed_everything(cfg.seed)
    if cfg.do_use_filesystem_sharing:
        torch.multiprocessing.set_sharing_strategy("file_system")

    config = cfg.config
    optimization_config = cfg.optimization_config
    data_config = cfg.data_config

    if os.environ.get("LOCAL_RANK", "0") == "0":
        pathlib.Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

        print("Saving config files...")
        config_fp = pathlib.Path(cfg.save_dir) / "config.json"
        if config_fp.exists() and not cfg.do_overwrite:
            raise FileExistsError(f"{config_fp} already exists!")
        else:
            print(f"Writing to {config_fp}")
            config_fp = pathlib.Path(cfg.save_dir) / "config.json"
            config_dict = omegaconf.OmegaConf.to_container(config, resolve=True)
            if not cfg.do_overwrite and config_fp.exists():
                raise FileExistsError(f"{config_fp} already exists and do_overwrite is False.")

            with open(config_fp, 'w') as f:
                json.dump(config_dict, f)

        data_config_dict = omegaconf.OmegaConf.to_container(data_config, resolve=True)
        with open(pathlib.Path(cfg.save_dir) / "data_config.json", 'w') as f:
            json.dump(data_config_dict, f)
        optimization_config_dict = omegaconf.OmegaConf.to_container(optimization_config, resolve=True)
        with open(pathlib.Path(cfg.save_dir) / "optimization_config.json", 'w') as f:
            json.dump(optimization_config_dict, f)

        pretraining_metrics_config_dict = omegaconf.OmegaConf.to_container(cfg.pretraining_metrics_config, resolve=True)
        with open(pathlib.Path(cfg.save_dir) / "pretraining_metrics_config.json", 'w') as f:
            json.dump(pretraining_metrics_config_dict, f)

        final_validation_metrics_config_dict = omegaconf.OmegaConf.to_container(cfg.final_validation_metrics_config, resolve=True)
        with open(pathlib.Path(cfg.save_dir) / "final_validation_metrics_config.json", 'w') as f:
            json.dump(final_validation_metrics_config_dict, f)

    # Model
    config_dict = omegaconf.OmegaConf.to_container(config, resolve=True)
    LM = ESTForGenerativeSequenceModelingLM(
        config=config_dict,
        optimization_config=optimization_config,
        metrics_config=MetricsConfig(**cfg.pretraining_metrics_config),
        pretraining_metrics_config=cfg.pretraining_metrics_config,
        final_validation_metrics_config=cfg.final_validation_metrics_config,
        vocabulary_config=train_pyd.vocabulary_config,
    )

    # TODO(mmd): Get this working!
    # if cfg.compile:
    #     print("Compiling model!")
    #     LM = torch.compile(LM)

    # Setting up torch dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_pyd,
        batch_size=optimization_config.batch_size,
        num_workers=optimization_config.num_dataloader_workers,
        collate_fn=train_pyd.collate,
        pin_memory=False,
        shuffle=True,
    )
    tuning_dataloader = torch.utils.data.DataLoader(
        tuning_pyd,
        batch_size=optimization_config.validation_batch_size,
        num_workers=0, # avoid CUDA error
        collate_fn=tuning_pyd.collate,
        pin_memory=False,
        shuffle=False,
    )

    # Setting up model configurations
    # This will track the learning rate value as it updates through warmup and decay.
    callbacks = [LearningRateMonitor(logging_interval="step")]
    if optimization_config.patience is not None:
        callbacks.append(
            EarlyStopping(monitor="train_loss", mode="min", patience=optimization_config.patience)  # Change the monitor to 'train_loss'
        )

    trainer_kwargs = dict(
        **cfg.trainer_config,
        max_epochs=optimization_config.max_epochs,
        callbacks=callbacks,
    )
    if cfg.wandb_logger_kwargs.get("name", None):
        if "do_log_graph" in cfg.wandb_logger_kwargs:
            do_log_graph = cfg.wandb_logger_kwargs.get("do_log_graph", False)
        else:
            do_log_graph = False

        serializable_config = {
            k: v for k, v in cfg.wandb_experiment_config_kwargs.items()
            if isinstance(v, (str, int, float, bool, type(None)))
        }
        wandb_logger = WandbLogger(
            **{k: v for k, v in cfg.wandb_logger_kwargs.items() if v is not None},
            save_dir=str(cfg.save_dir),  # Convert to string
            config=serializable_config,
        )

        if os.environ.get("LOCAL_RANK", "0") == "0":
            if do_log_graph:
                # Watching the model naturally tracks parameter values and gradients.
                wandb_logger.watch(LM, log="all", log_graph=True)

        trainer_kwargs["logger"] = wandb_logger

    if (optimization_config.gradient_accumulation is not None) and (
        optimization_config.gradient_accumulation > 1
    ):
        trainer_kwargs["accumulate_grad_batches"] = optimization_config.gradient_accumulation

    # Fitting model
    trainer = L.Trainer(**trainer_kwargs, strategy='ddp_find_unused_parameters_true')
    trainer.fit(model=LM, train_dataloaders=train_dataloader, val_dataloaders=tuning_dataloader)

    LM.save_pretrained(cfg.save_dir)

    if cfg.do_final_validation_on_metrics:
        held_out_pyd = PytorchDataset(cfg.data_config, split="held_out")
        held_out_dataloader = torch.utils.data.DataLoader(
            held_out_pyd,
            batch_size=optimization_config.validation_batch_size,
            num_workers=optimization_config.num_dataloader_workers,
            collate_fn=held_out_pyd.collate,
            shuffle=False,
        )

        LM.metrics_config = cfg.final_validation_metrics_config
        LM.build_metrics()

        tuning_metrics = trainer.validate(model=LM, dataloaders=tuning_dataloader)
        held_out_metrics = trainer.test(model=LM, dataloaders=held_out_dataloader)

        if os.environ.get("LOCAL_RANK", "0") == "0":
            print("Saving final metrics...")

            with open(Path(cfg.save_dir) / "tuning_metrics.json", mode="w") as f:
                json.dump(tuning_metrics, f)
            with open(Path(cfg.save_dir) / "held_out_metrics.json", mode="w") as f:
                json.dump(held_out_metrics, f)

        return tuning_metrics[0]["tuning_loss"], tuning_metrics, held_out_metrics

    return None