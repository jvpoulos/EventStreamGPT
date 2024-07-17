import dataclasses
from dataclasses import is_dataclass, asdict
import json
import os
import random
from collections.abc import Sequence
from pathlib import Path
import pathlib
from typing import Dict, Any, Union, Optional
import wandb
from torch.optim.lr_scheduler import LambdaLR
import math
import lightning as L
import omegaconf
from omegaconf import DictConfig
import torch
from pytorch_lightning import LightningModule
import torchmetrics
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from omegaconf import OmegaConf
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MultilabelAccuracy,
    MultilabelAUROC,
    MultilabelAveragePrecision,
)
from transformers import get_polynomial_decay_schedule_with_warmup

from ...data.config import (
    PytorchDatasetConfig,
    SeqPaddingSide,
    SubsequenceSamplingStrategy,
)
from ...data.pytorch_dataset import PytorchDataset
from ...utils import hydra_dataclass, task_wrapper
from ..config import OptimizationConfig, StructuredTransformerConfig
from ..fine_tuning_model import ESTForStreamClassification
from ..model_output import StreamClassificationModelOutput
from ..utils import str_summary

from ...data.vocabulary import VocabularyConfig
from ...data.types import PytorchBatch
from pytorch_lightning.loggers import WandbLogger

import time
from torch.utils.data import DataLoader

import logging

import numpy as np
from torch.cuda.amp import GradScaler

from dataclasses import dataclass, field

# Configure the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_config_value(config, key, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)

class TimeoutDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        self.timeout = kwargs.pop('timeout', 600)  # Default timeout of 600 seconds (10 minutes)
        super().__init__(*args, **kwargs)

    def __iter__(self):
        return TimeoutDataLoaderIter(super().__iter__(), self.timeout)

class TimeoutDataLoaderIter:
    def __init__(self, iterator, timeout):
        self.iterator = iterator
        self.timeout = timeout
        self.start_time = time.time()

    def __next__(self):
        if self.timeout <= 0:  # If timeout is 0 or negative, don't apply timeout
            return next(self.iterator)
        
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.timeout:
            raise TimeoutError(f"DataLoader timed out after {elapsed_time:.2f} seconds")
        
        try:
            return next(self.iterator)
        except StopIteration:
            raise
        except Exception as e:
            raise TimeoutError(f"DataLoader encountered an error after {elapsed_time:.2f} seconds: {str(e)}")

class ESTForStreamClassificationLM(L.LightningModule):
    def __init__(
        self,
        config: StructuredTransformerConfig | dict[str, Any],
        optimization_config: OptimizationConfig | dict[str, Any],
        cfg,
        pretrained_weights_fp: Path | str | None = None,
        do_debug_mode: bool = True,
        **model_params
    ):
        super().__init__()
        if isinstance(config, dict):
            config = StructuredTransformerConfig(**config)
        self.config = config
        if isinstance(optimization_config, dict):
            self.optimization_config = OptimizationConfig(**optimization_config)
        else:
            self.optimization_config = optimization_config
        self.cfg = cfg
        self.do_debug_mode = do_debug_mode
        
        self.gradient_norm_changes = []
        self.max_outliers_to_log = 10

        # Initialize metrics dictionaries
        self.train_metrics = {}
        self.val_metrics = {}
        self.test_metrics = {}

        self.learning_rate = self.optimization_config.init_lr
        self.num_training_steps = self.optimization_config.max_training_steps
        self.weight_decay = self.optimization_config.weight_decay
        self.use_lr_scheduler = self.optimization_config.use_lr_scheduler
        self.lr_scheduler_type = self.optimization_config.lr_scheduler_type
        self.end_lr = self.optimization_config.end_lr
        self.end_lr_frac_of_init_lr = self.optimization_config.end_lr_frac_of_init_lr
        self.use_grad_value_clipping = self.optimization_config.use_grad_value_clipping
        self.clip_grad_value = self.optimization_config.clip_grad_value
        self.max_grad_norm = self.config.max_grad_norm

        self.grad_norm_before = None
        self.grad_norm_after = None

        # Load the vocabulary_config from a file
        vocabulary_config_path = "data/vocabulary_config.json"
        with open(vocabulary_config_path, "r") as f:
            vocabulary_config_dict = json.load(f)
        vocabulary_config = VocabularyConfig.from_dict(vocabulary_config_dict)

        if pretrained_weights_fp is None or pretrained_weights_fp == "skip":
            self.model = ESTForStreamClassification(config, vocabulary_config, self.optimization_config)
        else:
            self.model = ESTForStreamClassification.from_pretrained(pretrained_weights_fp, config=config, vocabulary_config=vocabulary_config, optimization_config=self.optimization_config)

        self.save_hyperparameters(
            {
                "config": config.to_dict(),
                "optimization_config": asdict(self.optimization_config) if is_dataclass(self.optimization_config) else self.optimization_config,
            }
        )

        self.build_metrics()

        # Load the vocabulary_config from a file
        vocabulary_config_path = "data/vocabulary_config.json"
        with open(vocabulary_config_path, "r") as f:
            vocabulary_config_dict = json.load(f)
        vocabulary_config = VocabularyConfig.from_dict(vocabulary_config_dict)

        if pretrained_weights_fp is None or pretrained_weights_fp == "skip":
            self.model = ESTForStreamClassification(config, vocabulary_config, self.optimization_config)
        else:
            self.model = ESTForStreamClassification.from_pretrained(pretrained_weights_fp, config=config, vocabulary_config=vocabulary_config, optimization_config=self.optimization_config)

    def on_before_optimizer_step(self, optimizer):
        self.grad_norm_before = self.get_gradient_norm()
        self.log('grad_norm_before', self.grad_norm_before, on_step=True, on_epoch=False, prog_bar=False, logger=True)

    def on_after_optimizer_step(self, optimizer):
        self.grad_norm_after = self.get_gradient_norm()
        self.log('grad_norm_after', self.grad_norm_after, on_step=True, on_epoch=False, prog_bar=False, logger=True)

    def get_gradient_norm(self):
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
        
    def __reduce__(self):
        return (self.__class__, (self.config, self.optimization_config, self.cfg))

    def save_pretrained(self, model_dir: Path):
        fp = model_dir / "pretrained_weights"
        self.model.save_pretrained(fp)

    def build_metrics(self):
        """Build the various torchmetrics we'll use to track performance."""

        if (self.config.problem_type == "single_label_classification") and (self.config.num_labels > 2):
            metric_kwargs = {"num_classes": self.config.num_labels}
            if not self.do_debug_mode:
                metric_kwargs["validate_args"] = False

            self.metrics = torch.nn.ModuleDict(
                {
                    "macro_AUROC": MulticlassAUROC(**metric_kwargs, average="macro"),
                    "weighted_AUROC": MulticlassAUROC(**metric_kwargs, average="weighted"),
                    "macro_accuracy": MulticlassAccuracy(**metric_kwargs, average="macro"),
                    "weighted_accuracy": MulticlassAccuracy(**metric_kwargs, average="weighted"),
                    "micro_accuracy": MulticlassAccuracy(**metric_kwargs, average="micro"),
                    "macro_AUPRC": MulticlassAveragePrecision(**metric_kwargs, average="macro"),
                    "weighted_AUPRC": MulticlassAveragePrecision(**metric_kwargs, average="weighted"),
                }
            )
        elif (self.config.problem_type == "single_label_classification") and (self.config.num_labels == 2):
            metric_kwargs = {}
            if not self.do_debug_mode:
                metric_kwargs["validate_args"] = False

            self.metrics = torch.nn.ModuleDict(
                {
                    "AUROC": BinaryAUROC(**metric_kwargs),
                    "accuracy": BinaryAccuracy(**metric_kwargs),
                    "AUPRC": BinaryAveragePrecision(**metric_kwargs),
                }
            )
        elif self.config.problem_type == "multi_label_classification":
            metric_kwargs = {"num_labels": self.config.num_labels}
            if not self.do_debug_mode:
                metric_kwargs["validate_args"] = False

            self.metrics = torch.nn.ModuleDict(
                {
                    "macro_AUROC": MultilabelAUROC(**metric_kwargs, average="macro"),
                    "weighted_AUROC": MultilabelAUROC(**metric_kwargs, average="weighted"),
                    "micro_AUROC": MultilabelAUROC(**metric_kwargs, average="micro"),
                    "macro_accuracy": MultilabelAccuracy(**metric_kwargs, average="macro"),
                    "weighted_accuracy": MultilabelAccuracy(**metric_kwargs, average="weighted"),
                    "micro_accuracy": MultilabelAccuracy(**metric_kwargs, average="micro"),
                    "macro_AUPRC": MultilabelAveragePrecision(**metric_kwargs, average="macro"),
                    "weighted_AUPRC": MultilabelAveragePrecision(**metric_kwargs, average="weighted"),
                    "micro_AUPRC": MultilabelAveragePrecision(**metric_kwargs, average="micro"),
                }
            )
        else:
            raise ValueError(f"{self.config.problem_type} not valid")

    def _log_metric_dict(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        metrics: dict[str, torchmetrics.Metric],
        skip_metrics: Sequence[str],
        prefix: str,
    ):
        """This helper function logs the set of named metrics for the predictions `preds` and labels `labels`.

        Args:
            `preds` (`torch.Tensor`): The predictions for this metric calculation.
            `labels` (`torch.Tensor`): The labels for this metric calculation.
            `metrics` (`Dict[str, torchmetrics.Metric]`): The metrics to log, by name.
            `skip_metrics` (`Sequence[str]`):
                A list of metrics to skip. Entries are not full metric names, but rather are partial names and
                any metric whose name contains an element of `skip_metrics` will be skipped.
                For example, if `skip_metrics = ['AUROC', 'AUPRC']`, then a metric with name `'macro_AUROC'`
                or `'micro_AUPRC'` would be skipped, whereas a metric named `'weighted_accuracy'` would not.
            `prefix` (`str`):
                The prefix that should be used when logging metric results. Will likely be 'train', 'tuning',
                or 'held_out', for example.
        """
        for metric_name, metric in metrics.items():
            if any(to_skip in metric_name for to_skip in skip_metrics):
                continue

            try:
                metric(preds, labels.long())
                self.log(f"{prefix}_{metric_name}", metric)
            except (ValueError, IndexError) as e:
                print(
                    f"Failed to compute {metric_name} "
                    f"with preds ({str_summary(preds)}) and labels ({str_summary(labels)}): {e}."
                )

    def log_metrics(self, results: StreamClassificationModelOutput, skip_metrics: Sequence[str], prefix: str):
        """Logs metric results for a given output result."""

        if results.labels is None:
            if results.loss is not None:
                self.log(f"{prefix}_loss", results.loss)
            return

        self._log_metric_dict(
            preds=results.preds,
            labels=results.labels,
            metrics=self.metrics,
            skip_metrics=skip_metrics,
            prefix=prefix,
        )

        if results.loss is not None:
            self.log(f"{prefix}_loss", results.loss)

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch, labels=batch['labels'])
        loss = outputs.loss

        # Check for NaN or Inf values
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"NaN or Inf loss detected: {loss}")
            return None

        # Calculate gradient norm (this will be after potential clipping by PyTorch Lightning)
        grad_norm = self.get_gradient_norm()

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_accuracy', outputs.accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_auc', outputs.auc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('grad_norm', grad_norm, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        
        log_message = f"Step {batch_idx}: train_loss={loss.item():.4f}, train_accuracy={outputs.accuracy:.4f}, grad_norm={grad_norm:.4f}"
        logger.info(log_message)

        # Accumulate metrics
        self._accumulate_metrics('train', {
            'loss': loss.item(),
            'accuracy': outputs.accuracy if outputs.accuracy is not None else 0.0,
            'auc': outputs.auc if outputs.auc is not None else 0.0,
            'grad_norm': grad_norm,
        })

        # Store gradient norm change
        if len(self.gradient_norm_changes) > 0:
            prev_norm = self.gradient_norm_changes[-1][1]
            change = grad_norm - prev_norm
            self.gradient_norm_changes.append((batch_idx, change))
        else:
            self.gradient_norm_changes.append((batch_idx, grad_norm))
        
        return loss

    def get_gradient_norm(self):
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def validation_step(self, batch, batch_idx):
        if self.do_debug_mode:
            logger.debug(f"Validation step - Batch keys: {batch.keys() if batch is not None else 'None'}")
        outputs = self.model(batch, labels=batch['labels'])
        loss = outputs.loss

        if self.do_debug_mode:
            # Add learning rate to debug info
            if outputs.debug_info is not None:
                outputs.debug_info["learning_rate"] = self.trainer.optimizers[0].param_groups[0]['lr']
            
            logger.debug(f"Model outputs: {outputs}")
            logger.debug(f"Loss: {loss}")
        
        if loss is not None:
            if self.do_debug_mode:
                logger.debug(f"Validation step - Logits shape: {outputs.preds.shape}")
                logger.debug(f"Validation step - Labels shape: {outputs.labels.shape}")
                logger.debug(f"Validation step - Loss: {loss.item()}")

            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log('val_accuracy', outputs.accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log('val_auc', outputs.auc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

            if batch_idx % 100 == 0:
                logger.info(f"Validation step {batch_idx}: val_loss={loss.item():.4f}, val_accuracy={outputs.accuracy:.4f}")

            if outputs.accuracy is not None:
                self.log('val_accuracy', outputs.accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            
            # Accumulate metrics
            self._accumulate_metrics('val', {
                'loss': loss.item(),
                'accuracy': outputs.accuracy if outputs.accuracy is not None else 0.0,
                'auc': outputs.auc if outputs.auc is not None else 0.0,
            })
                
            # Log debugging information
            if outputs.debug_info:
                for key, value in outputs.debug_info.items():
                    self._accumulate_metrics('val', {f'debug_{key}': value})
        
        return loss

    def _accumulate_metrics(self, prefix, metrics):
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            if key not in self.__dict__[f'{prefix}_metrics']:
                self.__dict__[f'{prefix}_metrics'][key] = []
            self.__dict__[f'{prefix}_metrics'][key].append(value)

    def _log_epoch_metrics(self, prefix):
        metrics = self.__dict__[f'{prefix}_metrics']
        for key, values in metrics.items():
            if values:  # Check if the list is not empty
                avg_value = sum(values) / len(values)
                self.log(f'{prefix}_{key}_epoch', avg_value, on_step=False, on_epoch=True, logger=True)
        
        # Clear accumulated metrics
        self.__dict__[f'{prefix}_metrics'].clear()

    def on_train_epoch_end(self):
        self._log_epoch_metrics('train')
        if self.do_debug_mode:
            self.log_gradient_outliers()

    def log_gradient_outliers(self):
        if not self.gradient_norm_changes:
            return

        # Sort changes by absolute value, descending
        sorted_changes = sorted(self.gradient_norm_changes, key=lambda x: abs(x[1]), reverse=True)
        
        # Log top outliers
        logger.info(f"Top {self.max_outliers_to_log} gradient norm change outliers:")
        for batch_idx, change in sorted_changes[:self.max_outliers_to_log]:
            logger.info(f"Batch {batch_idx}: gradient norm change of {change:.4f}")

        # Calculate and log statistics
        changes = [change for _, change in self.gradient_norm_changes]
        mean_change = np.mean(changes)
        std_change = np.std(changes)
        logger.info(f"Gradient norm change statistics - Mean: {mean_change:.4f}, Std: {std_change:.4f}")

        # Log subjects with changes more than 3 standard deviations from the mean
        significant_outliers = [(batch_idx, change) for batch_idx, change in self.gradient_norm_changes if abs(change - mean_change) > 3 * std_change]
        logger.info(f"Batches with gradient norm changes > 3 std devs from mean:")
        for batch_idx, change in significant_outliers:
            logger.info(f"Batch {batch_idx}: gradient norm change of {change:.4f}")

        # Clear the list for the next epoch
        self.gradient_norm_changes.clear()

    def on_validation_epoch_end(self):
        self._log_epoch_metrics('val')

    def test_step(self, batch, batch_idx):
        if self.do_debug_mode:
            logger.debug(f"Test step - Batch keys: {batch.keys() if batch is not None else 'None'}")
        if batch is None:
            self.log('test_loss', None)
            return None
        
        outputs = self.model(batch, labels=batch['labels'])
        loss = outputs.loss

        if self.do_debug_mode:
            # Add learning rate to debug info
            if outputs.debug_info is not None:
                outputs.debug_info["learning_rate"] = self.trainer.optimizers[0].param_groups[0]['lr']
            
            logger.debug(f"Model outputs: {outputs}")
            logger.debug(f"Loss: {loss}")
        
            if loss is not None:
                # Accumulate metrics
                self._accumulate_metrics('test', {
                    'loss': loss.item(),
                    'accuracy': outputs.accuracy,
                    'auc': outputs.auc if outputs.auc is not None else BinaryAUROC()(outputs.preds, batch['labels']).item(),
                })
        
        # Log debugging information
        if outputs.debug_info:
            for key, value in outputs.debug_info.items():
                self._accumulate_metrics('test', {f'debug_{key}': value})
        
        return loss

    def _load_code_mapping(self):
        mapping_file = Path("data/code_mapping.json")
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError("Code mapping file not found. Please run preprocessing first.")


    def forward(self, batch, **kwargs):
        if not isinstance(batch, dict):
            raise TypeError("Input 'batch' should be a dictionary.")
        
        dynamic_indices = batch.get("dynamic_indices")
        dynamic_counts = batch.get("dynamic_counts")
        
        if dynamic_indices is None:
            raise ValueError("'dynamic_indices' must be provided in the batch.")

        if 'dynamic_indices' in batch:
            batch['dynamic_indices'] = batch['dynamic_indices'].long()
        outputs = self.model(
            dynamic_indices=dynamic_indices,
            dynamic_counts=dynamic_counts,
            **kwargs
        )
        
        return outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs

    def on_test_epoch_end(self):
        self._log_epoch_metrics('test')

    def configure_optimizers(self):
        # Group parameters by weight decay
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        # Initialize optimizer
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        if self.use_grad_value_clipping:
            for param_group in optimizer.param_groups:
                param_group['clip_value'] = self.clip_grad_value

        # Initialize scheduler
        if self.use_lr_scheduler:
            if self.lr_scheduler_type == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.num_training_steps,
                    eta_min=self.end_lr
                )
            elif self.lr_scheduler_type == "linear":
                scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=1.0,
                    end_factor=self.end_lr_frac_of_init_lr,
                    total_iters=self.num_training_steps
                )
            elif self.lr_scheduler_type == "one_cycle":
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=self.learning_rate,
                    total_steps=self.num_training_steps,
                    pct_start=0.3,
                    anneal_strategy='cos'
                )
            elif self.lr_scheduler_type == "reduce_on_plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=0.1,
                    patience=10,
                    verbose=True
                )
            else:
                raise ValueError(f"Unknown scheduler type: {self.lr_scheduler_type}")

            scheduler_config = {
                "scheduler": scheduler,
                "interval": "step" if self.lr_scheduler_type != "reduce_on_plateau" else "epoch",
                "frequency": 1,
                "monitor": "val_loss" if self.lr_scheduler_type == "reduce_on_plateau" else None,
            }
            return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
        else:
            return optimizer

@dataclass
class FinetuneConfig:
    experiment_dir: Optional[Union[str, Path]] = "${load_from_model_dir}/finetuning"
    load_from_model_dir: Optional[Union[str, Path]] = omegaconf.MISSING
    task_df_name: Optional[str] = omegaconf.MISSING
    optimization_config_path: str = omegaconf.MISSING
    vocabulary_config: Any = field(default_factory=VocabularyConfig)  # Changed to Any
    pretrain_config_path: Optional[str] = None
    dataset_path: Optional[str] = None
    pretraining_metrics_config: Optional[Dict[str, Any]] = None
    final_validation_metrics_config: Optional[Dict[str, Any]] = None
    do_final_validation_metrics_config: Optional[Dict[str, Any]] = None
    do_final_validation_on_metrics: bool = False
    pretrained_weights_fp: Path | str | None = "skip"

    save_dir: Optional[str] = (
        "${experiment_dir}/${task_df_name}/"
        "subset_size_${data_config.train_subset_size}/"
        "subset_seed_${data_config.train_subset_seed}/"
        "${now:%Y-%m-%d_%H-%M-%S}"
    )

    wandb_logger_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "name": "${task_df_name}_finetuning",
            "project": None,
            "log_model": True,
        }
    )

    wandb_experiment_config_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "save_dir": "${save_dir}",
        }
    )

    do_overwrite: bool = False
    seed: int = 1

    config: Dict[str, Any] = field(
        default_factory=lambda: {
            **{k: None for k in StructuredTransformerConfig().to_dict().keys()},
            "task_specific_params": {
                "pooling_method": "last",
                "num_samples": None,
            },
        }
    )

    optimization_config: Dict[str, Any] = field(
        default_factory=lambda: {
            'batch_size': 32,
            'validation_batch_size': 64,
            'num_dataloader_workers': 4,
            'max_epochs': 100,
            'gradient_accumulation': 1,
        }
    )

    data_config: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {
            **{k: None for k in PytorchDatasetConfig().to_dict().keys()},
            "subsequence_sampling_strategy": SubsequenceSamplingStrategy.TO_END,
            "seq_padding_side": SeqPaddingSide.RIGHT,
            "task_df_name": "${task_df_name}",
            "train_subset_size": "FULL",
            "train_subset_seed": 1,
            "dl_reps_dir": None,
        }
    )

    data_config_path: Optional[str] = None
    
    trainer_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "accelerator": "auto",
            "devices": "auto",
            "detect_anomaly": False,
            "default_root_dir": "${save_dir}/model_checkpoints",
            "log_every_n_steps": 10,
        }
    )
    do_use_filesystem_sharing: bool = True

    def __post_init__(self):
        if isinstance(self.save_dir, str):
            self.save_dir = Path(self.save_dir)
        if isinstance(self.load_from_model_dir, str):
            self.load_from_model_dir = Path(self.load_from_model_dir)
        if isinstance(self.pretrained_weights_fp, str) and self.pretrained_weights_fp != "skip":
            self.pretrained_weights_fp = Path(self.pretrained_weights_fp)

    def to_dict(self):
        return dataclasses.asdict(self)

    def __post_init__(self, data_config_path: str = None):
        match self.save_dir:
            case str():
                self.save_dir = Path(self.save_dir)
            case Path():
                pass
            case _:
                raise TypeError(
                    f"`save_dir` must be a str or path! Got {type(self.save_dir)}({self.save_dir})"
                )

        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True)
        elif not self.save_dir.is_dir():
            raise FileExistsError(f"{self.save_dir} is not a directory!")

        if self.load_from_model_dir in (omegaconf.MISSING, None, "skip"):
            self.config = StructuredTransformerConfig(
                **{k: v for k, v in self.config.items() if v is not None}
            )
            
            if 'dl_reps_dir' in self.data_config:
                del self.data_config['dl_reps_dir']
            
            self.data_config = PytorchDatasetConfig(**self.data_config)
            return

        match self.pretrained_weights_fp:
            case "skip" | None | Path():
                pass
            case str():
                self.pretrained_weights_fp = Path(self.pretrained_weights_fp)
            case _:
                raise TypeError(
                    "`pretrained_weights_fp` must be a str or path! Got "
                    f"{type(self.pretrained_weights_fp)}({self.pretrained_weights_fp})"
                )

        match self.load_from_model_dir:
            case str():
                self.load_from_model_dir = Path(self.load_from_model_dir)
            case Path():
                pass
            case _:
                raise TypeError(
                    "`load_from_model_dir` must be a str or path! Got "
                    f"{type(self.load_from_model_dir)}({self.load_from_model_dir})"
                )

        reloaded_data_config = None
        if data_config_path:
            data_config_fp = Path(data_config_path)
            print(f"Loading data_config from {data_config_fp}")
            reloaded_data_config = PytorchDatasetConfig.from_json_file(data_config_fp)
            reloaded_data_config.task_df_name = self.task_df_name
            self.data_config = reloaded_data_config
        else:
            if isinstance(self.data_config, dict):
                dl_reps_dir = self.data_config.pop('dl_reps_dir', None)
                self.data_config = PytorchDatasetConfig(**self.data_config)
                self.data_config.dl_reps_dir = dl_reps_dir
                reloaded_data_config = self.data_config

        config_fp = self.load_from_model_dir / "config.json"
        print(f"Loading config from {config_fp}")
        reloaded_config = StructuredTransformerConfig.from_json_file(config_fp)

        if isinstance(self.data_config, dict):
            dl_reps_dir = self.data_config.pop('dl_reps_dir', None)
            self.data_config = PytorchDatasetConfig(**self.data_config)
            self.data_config.dl_reps_dir = dl_reps_dir

        if isinstance(self.optimization_config, OptimizationConfig):
            self.optimization_config = dataclasses.asdict(self.optimization_config)        

        for param, val in self.config.items():
            if val is None:
                continue
            print(f"Overwriting {param} in config from {getattr(reloaded_config, param)} to {val}")
            setattr(reloaded_config, param, val)

        self.config = reloaded_config

        if self.pretrain_config_path:
            pretrain_config_fp = Path(self.pretrain_config_path)
            print(f"Loading pretrain_config from {pretrain_config_fp}")
            reloaded_pretrain_config = OmegaConf.load(pretrain_config_fp)
        else:
            reloaded_pretrain_config = OmegaConf.load(self.load_from_model_dir / "pretrain_config.yaml")

        if self.wandb_logger_kwargs.get("project", None) is None:
            print(f"Setting wandb project to {reloaded_pretrain_config.wandb_logger_kwargs.project}")
            self.wandb_logger_kwargs["project"] = reloaded_pretrain_config.wandb_logger_kwargs.project

class CollateFunction:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.logger = logging.getLogger(__name__)

    def __call__(self, batch):
        valid_items = []
        for i, item in enumerate(batch):
            if item is None:
                self.logger.warning(f"Item {i} in batch is None, skipping")
                continue
            
            if 'dynamic_indices' not in item or item['dynamic_indices'] is None:
                self.logger.warning(f"Item {i} has missing or None dynamic_indices, using default")
                item['dynamic_indices'] = torch.tensor([0], dtype=torch.long)  # Use 0 for padding
            
            if not isinstance(item['dynamic_indices'], torch.Tensor) or item['dynamic_indices'].numel() == 0:
                self.logger.warning(f"Invalid dynamic_indices in item {i}, using default")
                item['dynamic_indices'] = torch.tensor([0], dtype=torch.long)  # Use 0 for padding
            
            valid_items.append(item)

        if not valid_items:
            self.logger.error("No valid items found in batch")
            raise ValueError("No valid items in batch")

        max_seq_len = max(item['dynamic_indices'].size(0) for item in valid_items)

        dynamic_indices = torch.zeros((len(valid_items), max_seq_len), dtype=torch.long)
        dynamic_counts = torch.zeros((len(valid_items), max_seq_len), dtype=torch.float32)

        for i, item in enumerate(valid_items):
            seq_len = item['dynamic_indices'].size(0)
            dynamic_indices[i, :seq_len] = item['dynamic_indices']
            dynamic_counts[i, :seq_len] = item['dynamic_counts'][:seq_len]

        collated_batch = {
            'dynamic_indices': dynamic_indices,
            'dynamic_counts': dynamic_counts,
            'labels': torch.stack([self.safe_tensor_conversion(item.get('labels', 0), torch.float32) for item in valid_items]).squeeze(),
            'InitialA1c': torch.tensor([self.safe_float_conversion(item.get('InitialA1c', 0.0)) for item in valid_items], dtype=torch.float32),
            'Female': torch.tensor([self.safe_int_conversion(item.get('Female', 0)) for item in valid_items], dtype=torch.long),
            'Married': torch.tensor([self.safe_int_conversion(item.get('Married', 0)) for item in valid_items], dtype=torch.long),
            'GovIns': torch.tensor([self.safe_int_conversion(item.get('GovIns', 0)) for item in valid_items], dtype=torch.long),
            'English': torch.tensor([self.safe_int_conversion(item.get('English', 0)) for item in valid_items], dtype=torch.long),
            'AgeYears': torch.tensor([self.safe_float_conversion(item.get('AgeYears', 0.0)) for item in valid_items], dtype=torch.float32),
            'SDI_score': torch.tensor([self.safe_float_conversion(item.get('SDI_score', 0.0)) for item in valid_items], dtype=torch.float32),
            'Veteran': torch.tensor([self.safe_int_conversion(item.get('Veteran', 0)) for item in valid_items], dtype=torch.long),
        }

        return collated_batch

    def safe_tensor_conversion(self, value, dtype):
        if value is None:
            self.logger.warning(f"Received None value for tensor conversion, using 0")
            return torch.tensor([0], dtype=dtype)
        if isinstance(value, (int, float)):
            value = [value]
        try:
            return torch.tensor(value, dtype=dtype)
        except ValueError:
            self.logger.warning(f"Could not convert value to tensor: {value}, using 0")
            return torch.tensor([0], dtype=dtype)

    def safe_float_conversion(self, value):
        try:
            return float(value) if value is not None else 0.0
        except ValueError:
            self.logger.warning(f"Could not convert to float: {value}, using 0.0")
            return 0.0

    def safe_int_conversion(self, value):
        try:
            return int(value) if value is not None else 0
        except ValueError:
            self.logger.warning(f"Could not convert to int: {value}, using 0")
            return 0

    def safe_pad_sequence(self, sequences, max_val, dtype):
        try:
            sequences = [torch.tensor(s, dtype=dtype) if not isinstance(s, torch.Tensor) else s for s in sequences]
            sequences = [torch.clamp(s, max=max_val) for s in sequences]
            if all(s.numel() == 0 for s in sequences):
                raise ValueError("All sequences are empty")
            padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
            return padded
        except Exception as e:
            self.logger.error(f"Error in safe_pad_sequence: {str(e)}")
            raise

def worker_init_fn(worker_id):
    np.random.seed(worker_id)
    random.seed(worker_id)

@task_wrapper
def train(cfg: FinetuneConfig, train_pyd, tuning_pyd, held_out_pyd, wandb_logger: WandbLogger | None = None):
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    try:
        if wandb_logger is None:
            wandb.init(project=cfg.wandb_logger_kwargs.get('project', 'default_project'),
                       name=cfg.wandb_logger_kwargs.get('name', 'default_run'),
                       config=cfg.to_dict())
        else:
            wandb.init(config=cfg.to_dict())

        L.seed_everything(cfg.seed)

        config = cfg.config
        data_config = cfg.data_config
        optimization_config = cfg.optimization_config

        if not hasattr(config, 'problem_type') or config.problem_type is None:
            config.problem_type = "single_label_classification"

        model_params = dict()
        if cfg.pretrained_weights_fp is not None:
            model_params["pretrained_weights_fp"] = cfg.pretrained_weights_fp

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        logger.info("Initializing model")
        LM = ESTForStreamClassificationLM(config, optimization_config, cfg, **model_params).to(device)

        logger.info(f"Train dataset length: {len(train_pyd)}")
        logger.info(f"First few items from train dataset:")
        for i in range(min(5, len(train_pyd))):
            item = train_pyd[i]
            logger.info(f"Item {i}: {item}")

        logger.info("Creating data loaders")
        collate_fn = CollateFunction(config.vocab_size)
        
        def create_dataloader(dataset, batch_size, shuffle):
            return DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=collate_fn,
                shuffle=shuffle,
                num_workers=0,  
                pin_memory=True
            )

        train_dataloader = create_dataloader(train_pyd, optimization_config['batch_size'], True)
        tuning_dataloader = create_dataloader(tuning_pyd, optimization_config['validation_batch_size'], False)
        held_out_dataloader = create_dataloader(held_out_pyd, optimization_config['validation_batch_size'], False)

        callbacks = [
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                dirpath=cfg.save_dir / "checkpoints",
                filename="{epoch}-{val_loss:.2f}",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                save_weights_only=True,
                save_last=True,
                auto_insert_metric_name=False,
            ),
            EarlyStopping(
                monitor='val_auc_epoch',
                min_delta=0.00,
                patience=cfg.optimization_config['patience'],
                verbose=True,
                mode='max'
            )
        ]

        logger.info("Setting up trainer")
        trainer = L.Trainer(
            **cfg.trainer_config,
            callbacks=callbacks,
            logger=wandb_logger,
            accumulate_grad_batches=optimization_config.get('gradient_accumulation', 4),
            max_epochs=optimization_config.get('max_epochs', 100),
            gradient_clip_val=config.max_grad_norm if not optimization_config.get('use_grad_value_clipping', False) else optimization_config.get('clip_grad_value', None),
            gradient_clip_algorithm="norm" if not optimization_config.get('use_grad_value_clipping', False) else "value",
            enable_progress_bar=True,
        )

        logger.info("Starting training")
        trainer.fit(model=LM, train_dataloaders=train_dataloader, val_dataloaders=tuning_dataloader)

        logger.info("Training completed. Evaluating on validation and test sets.")
        
        tuning_metrics = trainer.validate(model=LM, dataloaders=tuning_dataloader, ckpt_path="best") if len(tuning_pyd) > 0 else None
        held_out_metrics = trainer.test(model=LM, dataloaders=held_out_dataloader, ckpt_path="best") if len(held_out_pyd) > 0 else None

        logger.info("Evaluation completed.")
        
        return None, tuning_metrics, held_out_metrics

    except Exception as e:
        logger.exception(f"An error occurred during training: {str(e)}")
        return None, None, None

__all__ = ['FinetuneConfig', 'train']