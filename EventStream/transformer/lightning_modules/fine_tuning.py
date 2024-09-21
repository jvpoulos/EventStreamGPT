import dataclasses
from dataclasses import is_dataclass, asdict
import json
import datetime
import os
import torch.distributed as dist
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
from collections.abc import Sequence
from collections import defaultdict
from pathlib import Path
import pathlib
from typing import Dict, Any, Union, Optional
import wandb
from torch.optim.lr_scheduler import LambdaLR
import math
import lightning as L
import omegaconf
import torch.nn.functional as F
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
from ..transformer import InnerSelfAttention
from ..model_output import StreamClassificationModelOutput
from ..utils import str_summary

from ...data.vocabulary import VocabularyConfig
from ...data.types import PytorchBatch
from pytorch_lightning.loggers import WandbLogger

import time
from torch.utils.data import DataLoader

import logging

import numpy as np
from torch.cuda.amp import autocast, GradScaler

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Configure the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ESTForStreamClassificationLM(L.LightningModule):
    def __init__(
        self,
        config: StructuredTransformerConfig | dict[str, Any],
        optimization_config: OptimizationConfig | dict[str, Any],
        cfg,
        vocabulary_config: VocabularyConfig,
        oov_index: int,
        pretrained_weights_fp: Path | str | None = None,
        do_debug_mode: bool = False,
        save_dir: str = "./model_outputs",
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
        self.oov_index = oov_index
        self.save_dir = save_dir
        
        self.gradient_norm_changes = []
        self.max_outliers_to_log = 10
        self.batch_indices = []  # Store batch indices instead of subject_ids
        self._epoch_counter = 0
        
        # Initialize metrics dictionaries
        self.train_metrics = {}
        self.val_metrics = {}
        self.test_metrics = {}

        # Initialize prediction storage
        self.val_predictions = []
        self.val_labels = []
        self.test_predictions = []
        self.test_labels = []
        
        # Set optimization parameters
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

        self.automatic_optimization = True

        self.scaler = GradScaler()

        self.train_dataset = None
        self.val_dataset = None

        self.use_static_features = cfg.use_static_features

        # Load the vocabulary_config from a file
        vocabulary_config_path = "/home/jvp/diabetes_pred/data/labs/vocabulary_config.json"
        with open(vocabulary_config_path, "r") as f:
            vocabulary_config_dict = json.load(f)
        vocabulary_config = VocabularyConfig.from_dict(vocabulary_config_dict)
        
        # Initialize the model
        if pretrained_weights_fp is None or pretrained_weights_fp == "skip":
            self.model = ESTForStreamClassification(config, vocabulary_config, self.optimization_config, oov_index=self.oov_index, save_dir=self.save_dir)
        else:
            self.model = ESTForStreamClassification.from_pretrained(
                pretrained_weights_fp,
                config=config,
                vocabulary_config=vocabulary_config,
                optimization_config=self.optimization_config,
                oov_index=self.oov_index,
                save_dir=self.save_dir
            )

        if self.config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self._set_static_graph()

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                print(f"Parameter {name} does not require gradients")
        
        # Ensure all parameters require gradients
        for param in self.model.parameters():
            param.requires_grad = True

        # Save hyperparameters
        self.save_hyperparameters(
            {
                "config": config.to_dict(),
                "optimization_config": asdict(self.optimization_config) if is_dataclass(self.optimization_config) else self.optimization_config,
            }
        )
        
        # Build metrics
        self.build_metrics()
        
        # Initialize gradient stats
        self.gradient_stats = {
            'before_norm': [],
            'after_norm': [],
            'after_clip': []
        }
        
        # Initialize metric accumulator
        self.metric_accumulator = defaultdict(list)

    def on_validation_start(self):
        self.model.eval()
        for module in self.model.modules():
            if isinstance(module, InnerSelfAttention):
                module.inference_mode = True

    def on_validation_end(self):
        for module in self.model.modules():
            if isinstance(module, InnerSelfAttention):
                module.inference_mode = False

    def on_test_start(self):
        self.model.eval()
        for module in self.model.modules():
            if isinstance(module, InnerSelfAttention):
                module.inference_mode = True

    def on_test_end(self):
        for module in self.model.modules():
            if isinstance(module, InnerSelfAttention):
                module.inference_mode = False
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.optimization_config.batch_size,
            shuffle=True,
            num_workers=self.optimization_config.num_dataloader_workers,
            pin_memory=True,
            collate_fn=self.custom_collate_fn,
            drop_last=True
        )

    def val_dataloader(self):
            return DataLoader(
                self.val_dataset,
                batch_size=self.optimization_config.validation_batch_size,
                shuffle=False,
                num_workers=self.optimization_config.num_dataloader_workers,
                pin_memory=True,
                collate_fn=self.custom_collate_fn,
                drop_last=True  # Add this line
            )

    def custom_collate_fn(self, batch):
        # Filter out None items
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None

        # Separate the items in the batch
        dynamic_indices = [item['dynamic_indices'] for item in batch if 'dynamic_indices' in item]
        dynamic_values = [item['dynamic_values'] for item in batch if 'dynamic_values' in item]
        dynamic_measurement_indices = [item['dynamic_measurement_indices'] for item in batch if 'dynamic_measurement_indices' in item]
        static_indices = [item['static_indices'] for item in batch if 'static_indices' in item]
        static_measurement_indices = [item['static_measurement_indices'] for item in batch if 'static_measurement_indices' in item]
        times = [item['time'] for item in batch if 'time' in item]
        labels = [item['labels'] for item in batch if 'labels' in item]

        # Pad sequences
        max_dynamic_length = max(tensor.size(0) for tensor in dynamic_indices)
        dynamic_indices_padded = torch.stack([self.pad_sequence(tensor, max_dynamic_length) for tensor in dynamic_indices])
        dynamic_values_padded = torch.stack([self.pad_sequence(tensor, max_dynamic_length, pad_value=0.0) for tensor in dynamic_values])
        dynamic_measurement_indices_padded = torch.stack([self.pad_sequence(tensor, max_dynamic_length) for tensor in dynamic_measurement_indices])
        times_padded = torch.stack([self.pad_sequence(tensor, max_dynamic_length, pad_value=0.0) for tensor in times])

        max_static_length = max(tensor.size(0) for tensor in static_indices + static_measurement_indices)
        static_indices_padded = torch.stack([self.pad_sequence(tensor, max_static_length) for tensor in static_indices])
        static_measurement_indices_padded = torch.stack([self.pad_sequence(tensor, max_static_length) for tensor in static_measurement_indices])

        return {
            'dynamic_indices': dynamic_indices_padded,
            'dynamic_values': dynamic_values_padded,
            'dynamic_measurement_indices': dynamic_measurement_indices_padded,
            'static_indices': static_indices_padded,
            'static_measurement_indices': static_measurement_indices_padded,
            'time': times_padded,
            'labels': torch.stack(labels) if labels else None
        }

    def pad_sequence(self, sequence, max_length, pad_value=0):
        if isinstance(sequence, list):
            return self.pad_sequence(torch.tensor(sequence), max_length, pad_value)
        if len(sequence) >= max_length:
            return sequence[:max_length]
        padding = torch.full((max_length - len(sequence),), pad_value, dtype=sequence.dtype, device=sequence.device)
        return torch.cat([sequence, padding])

    def _set_static_graph(self):
        def apply_static_graph(module):
            if hasattr(module, '_set_static_graph'):
                module._set_static_graph()
            for child in module.children():
                apply_static_graph(child)

        apply_static_graph(self.model)

    def configure_sharded_model(self):
        self.model._set_static_graph()

    def on_train_start(self):
        if self.config.use_gradient_checkpointing:
            self.model._set_static_graph()

    @property
    def is_gradient_checkpointing(self):
        return getattr(self.model, 'gradient_checkpointing', False)

    def gradient_checkpointing_enable(self):
        self.model.gradient_checkpointing_enable()
    
    def get_gradient_norm(self):
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def on_before_optimizer_step(self, optimizer):
        # Custom gradient clipping logic here
        if self.trainer.gradient_clip_val is not None:
            params = [p for group in optimizer.param_groups for p in group['params']]
            torch.nn.utils.clip_grad_norm_(params, self.trainer.gradient_clip_val)

    def on_after_optimizer_step(self, optimizer):
        self.grad_norm_after = self.get_gradient_norm()
        self.log('grad_norm_after', self.grad_norm_after, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        
        # Calculate and store gradient norm change
        grad_norm_change = self.grad_norm_after - self.grad_norm_before
        self.gradient_norm_changes.append((grad_norm_change, self.batch_indices[-1]))

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

    def log_metrics(self, prefix, outputs):
        metrics = {}
        
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            metrics[f'{prefix}_loss'] = outputs.loss
        
        if hasattr(outputs, 'accuracy') and outputs.accuracy is not None:
            metrics[f'{prefix}_accuracy'] = outputs.accuracy
        
        if hasattr(outputs, 'auc') and outputs.auc is not None:
            metrics[f'{prefix}_auc'] = outputs.auc

        if hasattr(outputs, 'auprc') and outputs.auprc is not None:
            metrics[f'{prefix}_auprc'] = outputs.auprc

        if hasattr(outputs, 'f1') and outputs.f1 is not None:
            metrics[f'{prefix}_f1'] = outputs.f1

        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            self.metric_accumulator[name].append(value)
        
        # Log all metrics at once
        self.log_dict(metrics, on_step=(prefix == 'train'), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # For validation, we only want to log the loss on_epoch
        if prefix == 'val':
            self.log(f'{prefix}_loss', metrics[f'{prefix}_loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def plot_visualizations(self, preds, labels, epoch, split):
        # Ensure preds and labels are on CPU and in numpy format
        preds_np = preds.cpu().numpy()
        labels_np = labels.cpu().numpy()

        # Confusion Matrix
        cm = confusion_matrix(labels_np, (preds_np > 0.5).astype(float))
        plt.figure(figsize=(10,7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix ({split.capitalize()} Set) - Epoch {epoch}')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        confusion_matrix_plot = wandb.Image(plt)
        plt.close()

        # ROC Curve
        fpr, tpr, _ = roc_curve(labels_np, preds_np)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve ({split.capitalize()} Set) - Epoch {epoch}')
        plt.legend(loc="lower right")
        roc_curve_plot = wandb.Image(plt)
        plt.close()

        # Log both plots to wandb
        wandb.log({
            f"{split}_confusion_matrix": confusion_matrix_plot,
            f"{split}_roc_curve": roc_curve_plot
        })

    def training_step(self, batch, batch_idx):
        if batch is None:
            return None

        labels = batch.pop('labels').to(self.device)  # Move labels to the correct device
        outputs = self.model(batch, labels=labels)
        loss = outputs.loss

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"NaN or Inf loss detected in training step {batch_idx}")
            loss = torch.where(torch.isnan(loss) | torch.isinf(loss), torch.full_like(loss, 1e-8), loss)

        self.log_metrics('train', outputs)
        
        return loss

    def validation_step(self, batch, batch_idx):
        if batch is None:
            logger.warning("Received None batch in validation_step")
            return None
        
        labels = batch.pop('labels').to(self.device)  # Move labels to the correct device

        # Log input data statistics
        logger.info(f"Validation step {batch_idx}: Input data stats:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"{key}: shape={value.shape}, dtype={value.dtype}, min={value.min().item()}, max={value.max().item()}, mean={value.float().mean().item()}")
        
        outputs = self.model(batch, labels=labels)
        
        # Log information about the outputs
        logger.info(f"Validation step {batch_idx}: loss = {outputs.loss}")
        if torch.isnan(outputs.loss) or torch.isinf(outputs.loss):
            logger.warning(f"NaN or Inf loss detected in validation step {batch_idx}")
            logger.info(f"Batch contents: {batch}")
            logger.info(f"Labels: {labels}")
            logger.info(f"Model outputs: {outputs}")
        
        # Check for NaN or Inf values in model parameters
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                logger.warning(f"NaN or Inf detected in model parameter: {name}")
        
        # Store predictions and labels
        self.val_predictions.append(outputs.preds.detach().cpu())
        self.val_labels.append(labels.detach().cpu())

        self.log_metrics('val', outputs)
        return outputs.loss

    def test_step(self, batch, batch_idx):
        if batch is None:
            logger.warning("Received None batch in test_step")
            return None
        
        labels = batch.pop('labels').to(self.device)  # Move labels to the correct device
        outputs = self.model(batch, labels=labels)
        loss = outputs.loss

        # Store predictions and labels
        self.test_predictions.append(outputs.preds.detach().cpu())
        self.test_labels.append(labels.detach().cpu())

        self.log_metrics('test', outputs)

        return loss

    def log_gradients(self, stage):
        total_norm = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.gradient_stats[stage].append(total_norm)

    def on_train_epoch_end(self):
        self.log_accumulated_metrics('train')
        # Log gradient statistics
        self.log_gradient_statistics()
        
        # Log subject_ids with highest gradient increases
        self.log_gradient_instability()
        self.gradient_stats = {k: [] for k in self.gradient_stats}

    def log_accumulated_metrics(self, prefix):
        for name, values in self.metric_accumulator.items():
            if name.startswith(prefix):
                self.log(f'{name}_epoch', sum(values) / len(values), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.metric_accumulator = defaultdict(list)

    def log_gradient_statistics(self):
        for stage, norms in self.gradient_stats.items():
            if norms:
                mean_norm = sum(norms) / len(norms)
                max_norm = max(norms)
                min_norm = min(norms)
                self.log(f'grad_norm_{stage}_mean', mean_norm, on_epoch=True, logger=True, sync_dist=True)
                self.log(f'grad_norm_{stage}_max', max_norm, on_epoch=True, logger=True, sync_dist=True)
                self.log(f'grad_norm_{stage}_min', min_norm, on_epoch=True, logger=True, sync_dist=True)

    def log_gradient_instability(self):
        # Sort gradient changes and get top outliers
        sorted_changes = sorted(self.gradient_norm_changes, key=lambda x: x[0], reverse=True)
        top_outliers = sorted_changes[:self.max_outliers_to_log]

        # Log batch indices of top outliers
        for i, (change, batch_idx) in enumerate(top_outliers):
            self.logger.experiment.log({f"gradient_instability/top_{i+1}": {
                "change": change,
                "batch_idx": batch_idx
            }})

        # Reset for next epoch
        self.gradient_norm_changes = []

    def on_validation_epoch_end(self):
        # Concatenate all predictions and labels
        all_preds = torch.cat(self.val_predictions)
        all_labels = torch.cat(self.val_labels)

        # Plot visualizations
        self.plot_visualizations(all_preds.cpu(), all_labels.cpu(), self.trainer.current_epoch, 'val')

        # Save predictions and labels for the entire validation set
        predictions_path = os.path.join(self.save_dir, "predictions", f"val_predictions_epoch_{self.trainer.current_epoch}.pt")
        labels_path = os.path.join(self.save_dir, "labels", f"val_labels_epoch_{self.trainer.current_epoch}.pt")
        os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
        os.makedirs(os.path.dirname(labels_path), exist_ok=True)
        torch.save(all_preds, predictions_path)
        torch.save(all_labels, labels_path)

        # Clear the lists for the next epoch
        self.val_predictions = []
        self.val_labels = []
        self.log_accumulated_metrics('val')

    def forward(self, batch, **kwargs):
        self.logger.info(f"Forward method input keys: {locals().keys()}")
        # Move batch to the correct device
        if isinstance(batch, dict):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor) and torch.isnan(value).any():
                    logger.warning(f"NaNs detected in batch['{key}']")
        else:
            raise TypeError("Input 'batch' should be a dictionary.")

        # Move kwargs tensors to the correct device
        kwargs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}

        if not self.cfg.use_static_features:
            batch.pop('static_indices', None)
            batch.pop('static_measurement_indices', None)

        outputs = self.model(
            dynamic_indices=batch.get("dynamic_indices"),
            dynamic_values=batch.get("dynamic_values"),
            dynamic_measurement_indices=batch.get("dynamic_measurement_indices"),
            static_indices=batch.get("static_indices"),
            static_measurement_indices=batch.get("static_measurement_indices"),
            time=batch.get("time"),
            labels=batch.get("labels"),
            **kwargs
        )

        if hasattr(outputs, 'loss') and (torch.isnan(outputs.loss).any() or torch.isinf(outputs.loss).any()):
            logger.warning("NaN or Inf detected in model outputs")
            logger.info(f"Inputs: {batch}")
            logger.info(f"Outputs: {outputs}")

        return outputs

    def on_test_epoch_end(self):
        # Concatenate all predictions and labels
        all_preds = torch.cat(self.test_predictions)
        all_labels = torch.cat(self.test_labels)

        # Plot visualizations
        self.plot_visualizations(all_preds.cpu(), all_labels.cpu(), self.trainer.current_epoch, 'test')

        # Save predictions and labels for the entire test set
        predictions_path = os.path.join(self.save_dir, "predictions", f"test_predictions_epoch_{self.trainer.current_epoch}.pt")
        labels_path = os.path.join(self.save_dir, "labels", f"test_labels_epoch_{self.trainer.current_epoch}.pt")
        os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
        os.makedirs(os.path.dirname(labels_path), exist_ok=True)
        torch.save(all_preds, predictions_path)
        torch.save(all_labels, labels_path)

        # Clear the lists
        self.test_predictions = []
        self.test_labels = []

        self.log_accumulated_metrics('test')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        if self.use_lr_scheduler and self.lr_scheduler_type is not None:
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
                    factor=0.5,
                    patience=5,
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
    vocabulary_config: Any = field(default_factory=VocabularyConfig)
    pretrain_config_path: Optional[str] = None
    dataset_path: Optional[str] = None
    pretraining_metrics_config: Optional[Dict[str, Any]] = None
    final_validation_metrics_config: Optional[Dict[str, Any]] = None
    do_final_validation_metrics_config: Optional[Dict[str, Any]] = None
    do_final_validation_on_metrics: bool = False
    pretrained_weights_fp: Path | str | None = "skip"
    sweep: bool = False
    use_labs: bool = False
    do_debug_mode: bool = False
    use_static_features: bool = True
    data_config: PytorchDatasetConfig = field(default_factory=PytorchDatasetConfig)

    save_dir: Optional[str] = (
        "${experiment_dir}/${task_df_name}/"
        "subset_size_${data_config.train_subset_size}/"
        "subset_seed_${data_config.train_subset_seed}/"
        "${now:%Y-%m-%d_%H-%M-%S}"
    )

    def get_data_directories(self):
        if self.use_labs:
            save_dir = "./data/labs"
            dl_reps_dir = "data/labs/DL_reps"
        else:
            save_dir = "./data"
            dl_reps_dir = "data/DL_reps"
        return save_dir, dl_reps_dir

    def update_data_config(self):
        save_dir, dl_reps_dir = self.get_data_directories()
        self.data_config = dataclasses.replace(self.data_config, save_dir=save_dir, dl_reps_dir=dl_reps_dir)

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
            'batch_size': 256,
            'validation_batch_size': 256,
            'num_dataloader_workers': 4,
            'max_epochs': 100,
            'patience': 10,
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
            "strategy": "ddp_find_unused_parameters_true",
            "detect_anomaly": False,
            "default_root_dir": "${save_dir}/model_checkpoints",
            "log_every_n_steps": 10,
        }
    )
    do_use_filesystem_sharing: bool = True

    def set_data_directory(self):
        if self.use_labs:
            self.data_config['save_dir'] = "./data/labs"
            self.data_config['dl_reps_dir'] = "data/labs/DL_reps"
        else:
            self.data_config['save_dir'] = "./data"
            self.data_config['dl_reps_dir'] = "data/DL_reps"

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
    def __init__(self, vocab_size, oov_index, include_labels=True, static_size=8, max_seq_len=None, use_static_features=True):
        self.vocab_size = vocab_size
        self.oov_index = oov_index
        self.include_labels = include_labels
        self.static_size = static_size
        self.max_seq_len = max_seq_len
        self.use_static_features = use_static_features
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"CollateFunction initialized with use_static_features: {self.use_static_features}")

    def __call__(self, batch):
        self.logger.info(f"CollateFunction use_static_features: {self.use_static_features}")
        try:
            # Filter out None items
            batch = [item for item in batch if item is not None]
            if len(batch) == 0:
                return None

            # Pad sequences
            collated_batch = {
                'dynamic_indices': torch.stack([self.pad_sequence(item['dynamic_indices'], self.max_seq_len) for item in batch]),
                'dynamic_values': torch.stack([self.pad_sequence(item['dynamic_values'], self.max_seq_len, pad_value=0.0) for item in batch]),
                'dynamic_measurement_indices': torch.stack([self.pad_sequence(item['dynamic_measurement_indices'], self.max_seq_len) for item in batch]),
                'time': torch.stack([self.pad_sequence(item['time'], self.max_seq_len, pad_value=0.0) for item in batch]),
            }

            if self.use_static_features:
                collated_batch['static_indices'] = torch.stack([self.pad_sequence(item['static_indices'], self.static_size) for item in batch])
                collated_batch['static_measurement_indices'] = torch.stack([self.pad_sequence(item['static_measurement_indices'], self.static_size) for item in batch])

            if self.include_labels:
                collated_batch['labels'] = torch.stack([item['labels'] for item in batch]).squeeze(1)

            self.logger.info(f"Collated batch keys: {collated_batch.keys()}")
            return collated_batch
        except Exception as e:
            self.logger.error(f"Error in collate function: {str(e)}")
            self.logger.error(f"Batch size: {len(batch)}")
            for i, item in enumerate(batch):
                self.logger.error(f"Item {i} keys: {item.keys()}")
            raise

    def pad_sequence(self, sequence, length, pad_value=0):
        if len(sequence) >= length:
            return sequence[:length]
        return F.pad(sequence, (0, length - len(sequence)), value=pad_value)

    def is_valid_item(self, item):
        required_keys = ['dynamic_indices', 'dynamic_values', 'static_indices', 'static_measurement_indices', 'dynamic_measurement_indices', 'labels']
        return all(k in item for k in required_keys) and all(torch.is_tensor(item[k]) for k in required_keys)

    def get_empty_batch(self):
        return {
            'dynamic_indices': torch.tensor([], dtype=torch.long),
            'dynamic_values': torch.tensor([], dtype=torch.float32),
            'dynamic_values_mask': torch.tensor([], dtype=torch.bool),
            'static_indices': torch.tensor([], dtype=torch.long),
            'static_measurement_indices': torch.tensor([], dtype=torch.long),
            'dynamic_measurement_indices': torch.tensor([], dtype=torch.long),
            'labels': torch.tensor([], dtype=torch.float32),
        }

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

    def safe_tensor_conversion(self, value, dtype):
        if value is None:
            return torch.tensor(0, dtype=dtype)
        return torch.tensor(value, dtype=dtype)

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

def worker_init_fn(worker_id):
    np.random.seed(worker_id)
    random.seed(worker_id)

@task_wrapper
def train(cfg: FinetuneConfig, train_pyd, tuning_pyd, held_out_pyd, vocabulary_config: VocabularyConfig, oov_index: int, wandb_logger: WandbLogger | None = None):

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    logger.info(f"Train dataset length: {len(train_pyd)}")
    logger.info(f"Tuning dataset length: {len(tuning_pyd)}")
    logger.info(f"Held-out dataset length: {len(held_out_pyd)}")

    logger.info(f"FinetuneConfig use_static_features: {cfg.use_static_features}")
    logger.info(f"Config use_static_features: {cfg.config.use_static_features}")

    try:
        # Update data config
        cfg.update_data_config()
        
        # Load the vocabulary_config from the correct file
        vocabulary_config_path = "/home/jvp/diabetes_pred/data/labs/vocabulary_config.json" if cfg.use_labs else "/home/jvp/diabetes_pred/data/vocabulary_config.json"
        with open(vocabulary_config_path, "r") as f:
            vocabulary_config_dict = json.load(f)
        vocabulary_config = VocabularyConfig.from_dict(vocabulary_config_dict)

        # Calculate OOV index
        dynamic_indices_vocab_size = vocabulary_config.vocab_sizes_by_measurement.get("dynamic_indices", 0)
        oov_index = cfg.config.vocab_size  # Set oov_index to vocab_size

        logger.info(f"Calculated OOV index: {oov_index}")
        
        # Log the data directories
        logger.info(f"Using data directory: {cfg.data_config.save_dir}")
        logger.info(f"Using DL reps directory: {cfg.data_config.dl_reps_dir}")
        
        # Always create a new WandbLogger
        wandb_logger = WandbLogger(project=cfg.wandb_logger_kwargs.get('project', 'default_project'),
                                   name=cfg.wandb_logger_kwargs.get('name', 'default_run'),
                                   config=cfg.to_dict())

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
        LM = ESTForStreamClassificationLM(
            config,
            optimization_config,
            cfg,
            vocabulary_config=vocabulary_config,
            oov_index=oov_index,
            pretrained_weights_fp=cfg.pretrained_weights_fp,
            do_debug_mode=cfg.do_debug_mode,
            device=device,
            **model_params
        ).to(device)

        if not cfg.use_static_features:
            # Remove static features from the batch
            for dataset in [train_pyd, tuning_pyd, held_out_pyd]:
                dataset.static_indices = None
                dataset.static_measurement_indices = None

        # Set the datasets
        LM.train_dataset = train_pyd
        LM.val_dataset = tuning_pyd

        logger.info(f"Train dataset length: {len(train_pyd)}")
        logger.info("Creating data loaders")
        logger.info(f"cfg.use_static_features: {cfg.use_static_features}")
        collate_fn = CollateFunction(
            vocab_size=config.vocab_size,
            oov_index=oov_index,
            include_labels=True,
            static_size=8,
            max_seq_len=cfg.config.max_seq_len,
            use_static_features=cfg.use_static_features
        )

        train_dataloader = DataLoader(
            train_pyd,
            batch_size=optimization_config['batch_size'],
            collate_fn=collate_fn,
            shuffle=True,
            num_workers=optimization_config['num_dataloader_workers'],
            pin_memory=True
        )

        tuning_dataloader = DataLoader(
            tuning_pyd,
            batch_size=optimization_config['validation_batch_size'],
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=optimization_config['num_dataloader_workers'],
            pin_memory=True
        )

        held_out_dataloader = DataLoader(
            held_out_pyd,
            batch_size=optimization_config['validation_batch_size'],
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=optimization_config['num_dataloader_workers'],
            pin_memory=True
        )

        callbacks = [
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                dirpath=cfg.save_dir / "checkpoints",
                filename="{epoch}-{val_auc_epoch:.2f}",
                monitor="val_auc_epoch", 
                mode="max",
                save_top_k=3,
                save_weights_only=False,
                save_last=True,
                auto_insert_metric_name=False,
            ),
            EarlyStopping(
                monitor='val_auc_epoch',
                patience=cfg.optimization_config['patience'],
                verbose=True,
                mode='max',
                check_finite=True
            )
        ]

        class NCCLErrorHandler(Callback):
            def on_exception(self, trainer, pl_module, exception):
                if isinstance(exception, RuntimeError) and "NCCL" in str(exception):
                    print("NCCL error detected. Attempting to recover...")
                    trainer.strategy.barrier()
                    return True
                return False

        # Add the callbacks
        callbacks.append(NCCLErrorHandler())

        logger.info("Setting up trainer")
        
        # Create a GradScaler for mixed precision training
        scaler = GradScaler()

        # Update trainer configuration
        cfg.trainer_config["precision"] = "16-mixed"

        logger.info("Creating trainer")
        trainer = L.Trainer(
            **cfg.trainer_config,
            callbacks=callbacks,
            logger=wandb_logger,
            max_epochs=optimization_config.get('max_epochs', 100),
            gradient_clip_val=config.max_grad_norm,
            gradient_clip_algorithm="norm",
            enable_progress_bar=True,
            deterministic=False,
        )
        logger.info("Trainer created")

        logger.info("Starting training")
        trainer.fit(model=LM, train_dataloaders=train_dataloader, val_dataloaders=tuning_dataloader)

        logger.info("Training completed. Evaluating on validation and test sets.")
        
        tuning_metrics = trainer.validate(model=LM, dataloaders=tuning_dataloader, ckpt_path="last") if len(tuning_pyd) > 0 else None
        held_out_metrics = trainer.test(model=LM, dataloaders=held_out_dataloader, ckpt_path="last") if len(held_out_pyd) > 0 else None

        logger.info("Evaluation completed.")

        return None, tuning_metrics, held_out_metrics

    except Exception as e:
        logger.exception(f"An error occurred during training: {str(e)}")
        return None, None, None
    finally:
        pass

__all__ = ['FinetuneConfig', 'train']