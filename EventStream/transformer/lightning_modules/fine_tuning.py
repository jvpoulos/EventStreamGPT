import dataclasses
from dataclasses import is_dataclass, asdict
import json
import datetime
import os
import time
from scipy import stats
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
import random
import numpy as np
from functools import lru_cache
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
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
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import omegaconf
import torch.nn.functional as F
from omegaconf import DictConfig
import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from datetime import timedelta
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
from transformers import get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup

from ...data.config import (
    PytorchDatasetConfig,
    SeqPaddingSide,
    SubsequenceSamplingStrategy,
    MeasurementConfig
)
from ...data.pytorch_dataset import PytorchDataset
from ...utils import hydra_dataclass, task_wrapper
from ..config import StructuredTransformerConfig, OptimizationConfig
from ..fine_tuning_model import ESTForStreamClassification
from ..transformer import InnerSelfAttention
from ..model_output import StreamClassificationModelOutput
from ..utils import str_summary

from ...data.vocabulary import VocabularyConfig
from ...data.types import PytorchBatch

from torch.utils.data import DataLoader, DistributedSampler

import numpy as np
from torch.cuda.amp import autocast, GradScaler

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import seaborn as sns

import logging
from pytorch_lightning.loggers import WandbLogger
from contextlib import contextmanager
import signal

# Set up the standard logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class NCCLErrorHandler(Callback):
    def on_exception(self, trainer, pl_module, exception):
        if isinstance(exception, RuntimeError) and "NCCL" in str(exception):
            logger.error(f"NCCL error detected: {str(exception)}")
            logger.info("Attempting to recover...")
            trainer.strategy.barrier()
            return True
        return False

class GradientCheckCallback(Callback):
    @staticmethod
    def check_and_log_gradients(model):
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm()
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    logger.warning(f"NaN or Inf gradient detected in {name}")
                else:
                    wandb.log({f"gradient_norm/{name}": grad_norm})

    def on_after_backward(self, trainer, model):
        self.check_and_log_gradients(model)

class AttentionMechanismSwitch(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if pl_module.config.use_flash_attention and batch_idx % 100 == 0:
            try:
                # Try to use Flash Attention on a small batch
                small_batch = {k: v[:4] for k, v in batch.items() if isinstance(v, torch.Tensor)}
                _ = pl_module(small_batch, output_attentions=True)
            except Exception as e:
                logger.warning(f"Flash Attention failed. Switching to standard attention.")
                pl_module.config.use_flash_attention = False

class WandbLoggerHandler(logging.Handler):
    def __init__(self, wandb_logger):
        super().__init__()
        self.wandb_logger = wandb_logger

    def emit(self, record):
        msg = self.format(record)
        self.wandb_logger.experiment.log({f"log/{record.levelname}": msg})

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
        self.max_grad_norm = self.config.max_grad_norm
        
        self.grad_norm_before = None
        self.grad_norm_after = None

        self.automatic_optimization = True

        self.scaler = GradScaler()

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.use_static_features = cfg.use_static_features

        # Load the vocabulary_config from a file
        vocabulary_config_path = "data/labs/vocabulary_config.json"
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
        
        # Move the model to the correct device
        self.model = self.model.to(self.device)

        # Save hyperparameters
        self.save_hyperparameters(
            {
                "config": config.to_dict(),
                "optimization_config": asdict(self.optimization_config) if is_dataclass(self.optimization_config) else self.optimization_config,
            }
        )
        
        # Build metrics
        self.build_metrics()
        
        # Initialize metric accumulator
        self.metric_accumulator = defaultdict(list)

        self.custom_logger = logging.getLogger(self.__class__.__name__)
        self.custom_logger.setLevel(logging.INFO)

        # Importance metrics cache
        self.feature_importance_cache = {}
        self.temporal_importance_cache = {}

        self.register_gradient_hooks()

    def check_grad_requirements(self):
        for name, param in self.named_parameters():
            if not param.requires_grad:
                logger.warning(f"Parameter {name} does not require gradients")
            else:
                logger.debug(f"Parameter {name} requires gradients")
            
    def register_gradient_hooks(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                param.register_hook(lambda grad, name=name: self.log(f'grad_norm/{name}', grad.norm(), on_step=True, on_epoch=False))

    def set_dtype(self, dtype):
        self.to(dtype)
        if hasattr(self, 'model'):
            self.model.to(dtype)

    def setup(self, stage=None):
        # This method is called by PyTorch Lightning before training/testing
        if stage == 'fit' or stage is None:
            self.train_dataset.set_epoch(0)  # Reset the dataset for each epoch
            self.val_dataset.set_epoch(0)

    def on_train_end(self):
        self.switch_to_inference_mode()

    def on_train_start(self):
        if self.config.use_gradient_checkpointing:
            self.model._set_static_graph()
        self.switch_to_training_mode()
        self.check_grad_requirements()

    def on_validation_start(self):
        self.model.eval()
        self.switch_to_inference_mode()

    def on_validation_end(self):
        self.switch_to_training_mode()

    def on_test_start(self):
        self.model.eval()
        self.switch_to_inference_mode()

    def on_test_end(self):
        self.switch_to_training_mode()

    def switch_to_inference_mode(self):
        for module in self.model.modules():
            if isinstance(module, InnerSelfAttention):
                module.use_flash_attention = False
                module.inference_mode = True

    def switch_to_training_mode(self):
        for module in self.model.modules():
            if isinstance(module, InnerSelfAttention):
                module.use_flash_attention = self.config.use_flash_attention
                module.inference_mode = False

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            world_size = self.trainer.num_devices if self.trainer.num_devices is not None else 1
            if world_size > 1:
                self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
                self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False)
            else:
                self.train_sampler = None
                self.val_sampler = None

    def train_dataloader(self):
        collate_fn = CollateFunction(
            vocab_size=self.config.vocab_size,
            oov_index=self.oov_index,
            include_labels=True,
            static_size=8,
            max_seq_len=self.config.max_seq_len,
            use_static_features=self.cfg.use_static_features
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.optimization_config.batch_size,
            sampler=self.train_sampler,
            shuffle=self.train_sampler is None,  # Only shuffle if we're not using a sampler
            num_workers=self.optimization_config.num_dataloader_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )

    def val_dataloader(self):
        collate_fn = CollateFunction(
            vocab_size=self.config.vocab_size,
            oov_index=self.oov_index,
            include_labels=True,
            static_size=8,
            max_seq_len=self.config.max_seq_len,
            use_static_features=self.cfg.use_static_features
        )
        return DataLoader(
            self.val_dataset,
            batch_size=self.optimization_config.validation_batch_size,
            sampler=self.val_sampler,
            shuffle=False,  # We don't shuffle the validation data
            num_workers=self.optimization_config.num_dataloader_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            raise ValueError("Test dataset has not been set. Please set it before calling test_dataloader().")
        
        collate_fn = CollateFunction(
            vocab_size=self.config.vocab_size,
            oov_index=self.oov_index,
            include_labels=True,
            static_size=8,
            max_seq_len=self.config.max_seq_len,
            use_static_features=self.cfg.use_static_features
        )
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.optimization_config.validation_batch_size,
            shuffle=False,
            num_workers=self.optimization_config.num_dataloader_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )

    def on_train_epoch_start(self):
        if isinstance(self.trainer.train_dataloader.sampler, DistributedSampler):
            self.trainer.train_dataloader.sampler.set_epoch(self.current_epoch)

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

    @property
    def is_gradient_checkpointing(self):
        return getattr(self.model, 'gradient_checkpointing', False)

    def gradient_checkpointing_enable(self):
        self.model.gradient_checkpointing_enable()
    
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

        auc_value = roc_auc_score(labels_np, preds_np)

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
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_value:.2f})')
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
            f"{split}_roc_curve": roc_curve_plot,
            f"{split}_auc": auc_value
        })

    def log_attention_weights(self, attentions, batch_idx, top_n=5):
        if self.trainer.is_global_zero and wandb.run is not None:
            if attentions is None:
                logger.warning("No attention weights to log")
                return

            num_layers = len(attentions)
            for layer, attn_dict in enumerate(attentions):
                if not isinstance(attn_dict, dict) or 'attn_weights' not in attn_dict:
                    logger.warning(f"Unexpected attention format in layer {layer}")
                    continue

                attn = attn_dict['attn_weights']
                if not isinstance(attn, torch.Tensor):
                    logger.warning(f"Attention weights are not a tensor in layer {layer}")
                    continue

                # Log the shape of the attention tensor
                logger.info(f"Attention tensor shape for layer {layer}: {attn.shape}")

                # Ensure the attention tensor is 4D: [batch_size, num_heads, seq_len, seq_len]
                if attn.dim() == 3:
                    attn = attn.unsqueeze(1)  # Add a dimension for num_heads if it's missing
                elif attn.dim() != 4:
                    logger.warning(f"Unexpected attention tensor shape in layer {layer}: {attn.shape}")
                    continue

                num_heads = attn.size(1)
                seq_len = attn.size(-1)

                fig, axes = plt.subplots(num_heads, top_n, figsize=(20, 4 * num_heads))
                fig.suptitle(f"Top {top_n} Attention Weights for Layer {layer}")

                for head in range(num_heads):
                    head_attn = attn[:, head].detach().cpu()
                    # Calculate the mean attention across all positions
                    mean_attn = head_attn.mean(dim=(-2, -1))
                    # Get the indices of the top N samples
                    top_indices = mean_attn.argsort(descending=True)[:top_n]

                    for i, idx in enumerate(top_indices):
                        ax = axes[head, i] if num_heads > 1 and top_n > 1 else axes[head] if num_heads > 1 else axes[i] if top_n > 1 else axes
                        sns.heatmap(head_attn[idx], ax=ax, cmap='viridis', vmin=0, vmax=1)
                        ax.set_title(f"Head {head}, Sample {idx}")
                        ax.set_ylabel('Query position')
                        ax.set_xlabel('Key position')

                plt.tight_layout()
                wandb.log({f"attention_weights/layer_{layer}": wandb.Image(fig)})
                plt.close(fig)

            # Log overall max attention
            if attentions:
                overall_max_attn = torch.max(torch.stack([attn_dict['attn_weights'].max(dim=1)[0] for attn_dict in attentions if isinstance(attn_dict, dict) and 'attn_weights' in attn_dict]), dim=0)[0]
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(overall_max_attn[0].detach().cpu().numpy(), ax=ax, cmap='viridis', vmin=0, vmax=1)
                ax.set_title("Overall Maximum Attention Weights")
                ax.set_ylabel('Query position')
                ax.set_xlabel('Key position')
                wandb.log({"attention_weights/overall_maximum": wandb.Image(fig)})
                plt.close(fig)

    def log_debug(self, message):
        """Log debug messages to both the standard logger and WandB."""
        logging.getLogger(__name__).debug(message)
        if wandb.run is not None:
            wandb.log({"debug": message})

    def check_gradients_and_weights(self):
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm()
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    logger.warning(f"NaN or Inf gradient detected in {name}")
                elif grad_norm == 0:
                    logger.warning(f"Zero gradient detected in {name}")
            else:
                logger.warning(f"No gradient for parameter: {name}")

            if torch.isnan(param).any() or torch.isinf(param).any():
                logger.warning(f"NaN or Inf values detected in parameter: {name}")
            elif (param == 0).all():
                logger.warning(f"All zero values detected in parameter: {name}")

        # Check attention weights
        for module in self.modules():
            if isinstance(module, InnerSelfAttention):
                if hasattr(module, 'last_attn_weights'):
                    attn_weights = module.last_attn_weights
                    if torch.isnan(attn_weights).any() or torch.isinf(attn_weights).any():
                        logger.warning(f"NaN or Inf values detected in attention weights")
                    elif (attn_weights == 0).all():
                        logger.warning(f"All zero values detected in attention weights")

    def training_step(self, batch, batch_idx):
        # Ensure all tensors in the batch are on the correct device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Create attention mask
        seq_lens = batch['seq_len']
        max_seq_len = batch['dynamic_indices'].size(1)
        attention_mask = torch.arange(max_seq_len, device=self.device)[None, :] < seq_lens[:, None]
        
        # Add attention mask to the batch
        batch['attention_mask'] = attention_mask

        # Ensure all tensors have the correct sequence length
        for key in ['dynamic_indices', 'dynamic_values', 'dynamic_measurement_indices', 'time']:
            if batch[key].size(1) > max_seq_len:
                batch[key] = batch[key][:, :max_seq_len]

        # Log shapes for debugging
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                self.log_debug(f"Training batch {k} shape: {v.shape}")

        # Forward pass
        outputs = self.model(batch, output_attentions=True)  # Pass the entire batch to the model
        loss = outputs.loss

        self.check_gradients_and_weights()

        if torch.isnan(loss) or torch.isinf(loss):
            self.log_debug(f"NaN or Inf loss detected in training step {batch_idx}")
            loss = torch.where(torch.isnan(loss) | torch.isinf(loss), torch.full_like(loss, 1e-8), loss)

        # Add more detailed gradient logging
        if self.trainer.is_global_zero and batch_idx % 100 == 0:
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_stats = f"min={param.grad.min().item()}, max={param.grad.max().item()}, mean={param.grad.mean().item()}"
                    self.logger.experiment.log({f"gradients/{name}": grad_stats}, step=self.global_step)

        # Log attention weights statistics
        if outputs.attentions is not None:
            for i, layer_attention in enumerate(outputs.attentions):
                if isinstance(layer_attention, dict) and 'attn_weights' in layer_attention:
                    attn_weights = layer_attention['attn_weights']
                    if attn_weights is not None:
                        self.log(f'train_attention_layer_{i}_min', attn_weights.min().item(), on_step=True, on_epoch=False)
                        self.log(f'train_attention_layer_{i}_max', attn_weights.max().item(), on_step=True, on_epoch=False)
                        self.log(f'train_attention_layer_{i}_mean', attn_weights.mean().item(), on_step=True, on_epoch=False)
                    else:
                        logger.warning(f"Attention weights are None for layer {i}")
                else:
                    logger.warning(f"Unexpected attention format in layer {i}")

        # Log metrics
        self.log_metrics('train', outputs)
        
        # Ensure the loss is properly connected to the computational graph
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # Ensure all tensors in the batch are on the correct device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Create attention mask
        seq_lens = batch['seq_len']
        max_seq_len = batch['dynamic_indices'].size(1)
        attention_mask = torch.arange(max_seq_len, device=self.device)[None, :] < seq_lens[:, None]
        
        # Add attention mask to the batch
        batch['attention_mask'] = attention_mask

        # Log shapes for debugging
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                self.log_debug(f"Validation batch {k} shape: {v.shape}")

        # Forward pass
        outputs = self.model(batch, output_attentions=True)  # Pass the entire batch to the model
        
        self.check_gradients_and_weights()

        if torch.isnan(outputs.loss) or torch.isinf(outputs.loss):
            self.log_debug(f"NaN or Inf loss detected in validation step {batch_idx}")
        
        # Store predictions and labels for visualizations later
        self.val_predictions.append(outputs.preds.detach().cpu())
        self.val_labels.append(batch['labels'].detach().cpu())

        # Log metrics
        self.log_metrics('val', outputs)

        # Log attention weights if available
        if outputs.attentions is not None:
            self.log_attention_weights(outputs.attentions, batch_idx)

        return outputs.loss

    def test_step(self, batch, batch_idx):
        # Ensure all tensors in the batch are on the correct device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Create attention mask
        seq_lens = batch['seq_len']
        max_seq_len = batch['dynamic_indices'].size(1)
        attention_mask = torch.arange(max_seq_len, device=self.device)[None, :] < seq_lens[:, None]
        
        # Add attention mask to the batch
        batch['attention_mask'] = attention_mask

        # Forward pass
        outputs = self.model(batch, output_attentions=True)  # Pass the entire batch to the model

        self.check_gradients_and_weights()

        # Store predictions and labels
        self.test_predictions.append(outputs.preds.detach().cpu())
        self.test_labels.append(batch['labels'].detach().cpu())

        # Log metrics
        self.log_metrics('test', outputs)

        # Log attention weights if available
        if outputs.attentions is not None:
            self.log_attention_weights(outputs.attentions, batch_idx)

        return outputs.loss

    def log_accumulated_metrics(self, prefix):
        """
        Logs the accumulated metrics for the given prefix (train, val, or test) over the epoch.
        Args:
            prefix (str): The phase of training ('train', 'val', 'test').
        """
        for name, values in self.metric_accumulator.items():
            # Ensure that only metrics for the current prefix are processed
            if name.startswith(prefix) and len(values) > 0:
                # Log the average of the accumulated metric values
                avg_value = sum(values) / len(values)
                self.log(f'{name}_epoch', avg_value, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # Reset the metric accumulator for the next epoch
        self.metric_accumulator = defaultdict(list)

    @lru_cache(maxsize=128)
    def _calculate_single_feature_importance(self, feature, feature_data_key, baseline_loss, max_permutations):
        feature_data = self.feature_importance_cache.get(feature_data_key)
        if feature_data is None:
            return None, None

        permuted_losses = []
        for _ in range(max_permutations):
            permuted_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in self.feature_importance_cache.items()}
            permuted_batch[feature] = permuted_batch[feature][torch.randperm(permuted_batch[feature].size(0))]
            
            with torch.no_grad():
                permuted_output = self(permuted_batch)
            permuted_loss = permuted_output.loss.item()
            permuted_losses.append(permuted_loss - baseline_loss)
        return np.mean(permuted_losses), np.std(permuted_losses)

    def calculate_feature_importance(self, num_samples=1000, initial_permutations=10, max_permutations=100, adaptive_threshold=0.01, early_stop_threshold=0.001, dataloader=None, vocab_sample_size=5, vocab_significance_threshold=0.05):
        if dataloader is None:
            dataloader = self.val_dataloader()

        samples = []
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
            samples.append(batch)

        if not samples:
            logger.warning("No samples collected from the dataloader. Cannot calculate feature importance.")
            return {}, {}, {}

        combined_batch = {
            key: torch.cat([sample[key] for sample in samples], dim=0)
            for key in samples[0].keys()
        }

        max_seq_len = self.config.max_seq_len
        for key in ['dynamic_indices', 'dynamic_values', 'dynamic_measurement_indices', 'time']:
            if key in combined_batch and combined_batch[key].size(1) > max_seq_len:
                combined_batch[key] = combined_batch[key][:, :max_seq_len]
        
        if 'seq_len' in combined_batch:
            combined_batch['seq_len'] = torch.clamp(combined_batch['seq_len'], max=max_seq_len)

        self.feature_importance_cache = combined_batch

        with torch.no_grad():
            baseline_output = self(combined_batch)
        baseline_loss = baseline_output.loss.item()

        feature_importance = {}
        feature_importance_std = {}
        feature_importance_p_value = {}

        # Load vocabularies and result crosswalk
        vocabularies = self.load_vocabularies()
        result_crosswalk = self.load_result_crosswalk()

        total_features = len(combined_batch.keys()) - 2  # Exclude 'labels' and 'seq_len'
        for i, feature in enumerate(combined_batch.keys()):
            if feature not in ['labels', 'seq_len']:
                self.process_feature(feature, combined_batch, baseline_loss, initial_permutations, max_permutations, 
                                     adaptive_threshold, vocabularies, result_crosswalk, vocab_sample_size, 
                                     vocab_significance_threshold, feature_importance, feature_importance_std, 
                                     feature_importance_p_value)

        # Process fake feature only if it's not already in the batch
        if getattr(self.config, 'use_fake_feature', True) and 'fake_temporal_feature' not in combined_batch:
            fake_importance, fake_std, fake_p_value = self.process_fake_feature(combined_batch, baseline_loss, initial_permutations)
            
            if abs(fake_importance) > adaptive_threshold:
                additional_permutations = min(max_permutations - initial_permutations, 
                                              int((abs(fake_importance) - adaptive_threshold) / adaptive_threshold * initial_permutations))
                
                if additional_permutations > 0:
                    new_importance, new_std, new_p_value = self.process_fake_feature(combined_batch, baseline_loss, additional_permutations)
                    
                    total_permutations = initial_permutations + additional_permutations
                    fake_importance = (fake_importance * initial_permutations + new_importance * additional_permutations) / total_permutations
                    fake_std = np.sqrt((fake_std**2 * initial_permutations + new_std**2 * additional_permutations) / total_permutations)
                    fake_p_value = new_p_value  # Use the p-value from the larger sample

            feature_importance['fake_temporal_feature'] = fake_importance
            feature_importance_std['fake_temporal_feature'] = fake_std
            feature_importance_p_value['fake_temporal_feature'] = fake_p_value

        total_importance = sum(abs(score) for score in feature_importance.values())
        normalized_importance = {k: abs(v) / total_importance for k, v in feature_importance.items()}

        # Log progress
        progress = (i + 1) / total_features
        wandb.log({"feature_importance_progress": progress})

        return feature_importance, feature_importance_std, feature_importance_p_value

    def load_vocabularies(self):
        vocabularies = {}
        vocab_files = {
            'static_indices': 'data/labs/static_indices_vocab.json',
            'dynamic_indices': 'data/labs/dynamic_indices_vocab.json',
            'dynamic_measurement_indices': 'data/labs/dynamic_measurement_indices_vocab.json',
            'static_measurement_indices': 'data/labs/static_measurement_indices_vocab.json',
        }
        for key, file_path in vocab_files.items():
            with open(file_path, 'r') as f:
                vocabularies[key] = json.load(f)
        return vocabularies

    def load_result_crosswalk(self):
        crosswalk = {}
        with open('data/labs/Labs_Result_Numeric_Crosswalk.txt', 'r') as f:
            next(f)  # Skip header
            for line in f:
                result_t, result_n = line.strip().split('|')
                crosswalk[result_t] = float(result_n)
        return crosswalk

    def process_feature(self, feature, combined_batch, baseline_loss, initial_permutations, max_permutations, 
                        adaptive_threshold, vocabularies, result_crosswalk, vocab_sample_size, 
                        vocab_significance_threshold, feature_importance, feature_importance_std, 
                        feature_importance_p_value):
        importance, std = self._calculate_single_feature_importance(
            feature, feature, baseline_loss, initial_permutations
        )

        if importance is not None:
            self.update_feature_importance(feature, importance, std, adaptive_threshold, max_permutations, 
                                           initial_permutations, feature_importance, feature_importance_std, 
                                           feature_importance_p_value)

        if feature in vocabularies:
            self.process_vocabulary_items(feature, vocabularies[feature], combined_batch, baseline_loss, 
                                          initial_permutations, max_permutations, adaptive_threshold, 
                                          result_crosswalk, vocab_sample_size, vocab_significance_threshold, 
                                          feature_importance, feature_importance_std, feature_importance_p_value)

    def update_feature_importance(self, feature, importance, std, adaptive_threshold, max_permutations, 
                                  initial_permutations, feature_importance, feature_importance_std, 
                                  feature_importance_p_value):
        feature_importance[feature] = importance
        feature_importance_std[feature] = std

        if abs(importance) > adaptive_threshold:
            additional_permutations = min(max_permutations - initial_permutations, 
                                          int((abs(importance) - adaptive_threshold) / adaptive_threshold * initial_permutations))
            
            if additional_permutations > 0:
                new_importance, new_std = self._calculate_single_feature_importance(
                    feature, feature, baseline_loss, additional_permutations
                )
                
                if new_importance is not None:
                    total_permutations = initial_permutations + additional_permutations
                    feature_importance[feature] = (feature_importance[feature] * initial_permutations + new_importance * additional_permutations) / total_permutations
                    feature_importance_std[feature] = np.sqrt((feature_importance_std[feature]**2 * initial_permutations + new_std**2 * additional_permutations) / total_permutations)

        t_statistic, p_value = stats.ttest_1samp([feature_importance[feature]], 0)
        feature_importance_p_value[feature] = p_value

        logger.info(f"Feature '{feature}' importance: {feature_importance[feature]:.4f} ± {feature_importance_std[feature]:.4f} (p-value: {p_value:.4f})")

    def process_vocabulary_items(self, feature, vocab, combined_batch, baseline_loss, initial_permutations, 
                                 max_permutations, adaptive_threshold, result_crosswalk, vocab_sample_size, 
                                 vocab_significance_threshold, feature_importance, feature_importance_std, 
                                 feature_importance_p_value):
        vocab_items = list(vocab.items())
        sampled_vocab = random.sample(vocab_items, min(vocab_sample_size, len(vocab_items)))
        
        significant_entries = 0
        for idx, word in sampled_vocab:
            self.process_vocab_item(feature, idx, word, combined_batch, baseline_loss, initial_permutations, 
                                    max_permutations, adaptive_threshold, result_crosswalk, 
                                    feature_importance, feature_importance_std, feature_importance_p_value)
            
            feature_key = f"{feature}_{word}"
            if feature_key in feature_importance_p_value and feature_importance_p_value[feature_key] < vocab_significance_threshold:
                significant_entries += 1
        
        if significant_entries > 0 and len(vocab_items) > vocab_sample_size:
            remaining_vocab = [item for item in vocab_items if item not in sampled_vocab]
            additional_item = random.choice(remaining_vocab)
            idx, word = additional_item
            self.process_vocab_item(feature, idx, word, combined_batch, baseline_loss, initial_permutations, 
                                    max_permutations, adaptive_threshold, result_crosswalk, 
                                    feature_importance, feature_importance_std, feature_importance_p_value)

    def process_vocab_item(self, feature, idx, word, combined_batch, baseline_loss, initial_permutations, 
                           max_permutations, adaptive_threshold, result_crosswalk, 
                           feature_importance, feature_importance_std, feature_importance_p_value):
        feature_key = f"{feature}_{word}"
        permuted_losses = []
        
        try:
            for _ in range(initial_permutations):
                permuted_batch = combined_batch.copy()
                mask = self.create_mask(feature, word, idx, permuted_batch, result_crosswalk)
                
                mask = mask.to(torch.bool)
                mask_sum = torch.sum(mask).item()
                if mask_sum == 0:
                    logger.warning(f"Empty mask for feature '{feature_key}'. Skipping permutation.")
                    continue
                
                permuted_values = permuted_batch[feature][mask][torch.randperm(mask_sum)]
                permuted_batch[feature][mask] = permuted_values
                
                with torch.no_grad():
                    permuted_output = self(permuted_batch)
                permuted_loss = permuted_output.loss.item()
                permuted_losses.append(permuted_loss - baseline_loss)

            if not permuted_losses:
                logger.warning(f"No valid permutations for feature '{feature_key}'. Setting importance to 0.")
                importance, importance_std, p_value = 0, 0, 1
            else:
                importance = np.mean(permuted_losses)
                importance_std = np.std(permuted_losses)
                t_statistic, p_value = stats.ttest_1samp(permuted_losses, 0)

        except Exception as e:
            logger.error(f"Error during processing for feature '{feature_key}': {str(e)}")
            importance, importance_std, p_value = 0, 0, 1

        feature_importance[feature_key] = importance
        feature_importance_std[feature_key] = importance_std
        feature_importance_p_value[feature_key] = p_value
        
        logger.info(f"Feature '{feature_key}' importance: {importance:.4f} ± {importance_std:.4f} (p-value: {p_value:.4f})")

    def create_mask(self, feature, word, idx, permuted_batch, result_crosswalk):
        if feature == 'dynamic_values':
            numeric_value = result_crosswalk.get(word, word)
            try:
                numeric_value = float(numeric_value)
                mask = torch.isclose(permuted_batch[feature], torch.tensor(numeric_value, device=permuted_batch[feature].device))
            except ValueError:
                mask = permuted_batch[feature] == word
        elif feature == 'dynamic_indices':
            mask = permuted_batch[feature] == word
        else:
            try:
                mask = permuted_batch[feature] == int(idx)
            except ValueError:
                mask = permuted_batch[feature] == idx
        return mask

    def process_fake_feature(self, combined_batch, baseline_loss, initial_permutations, 
                             feature_importance, feature_importance_std, feature_importance_p_value):
        fake_feature = 'fake_temporal_feature'
        permuted_losses = []
        
        fake_temporal = self.create_correlated_fake_feature(combined_batch['time'], combined_batch['labels'])
        logger.debug(f"Fake feature correlation: {torch.corrcoef(torch.stack([fake_temporal.flatten(), combined_batch['labels'].flatten()]))[0,1]}")
        
        for _ in range(initial_permutations):
            permuted_batch = combined_batch.copy()
            permuted_values = fake_temporal[torch.randperm(fake_temporal.size(0))]
            permuted_batch[fake_feature] = permuted_values.to(permuted_batch[fake_feature].device)
            
            try:
                with torch.no_grad():
                    permuted_output = self(permuted_batch)
                permuted_loss = permuted_output.loss.item()
                permuted_losses.append(permuted_loss - baseline_loss)
            except Exception as e:
                logger.error(f"Error during permutation for fake feature: {str(e)}")
                continue
        
        logger.debug(f"Permuted losses: {permuted_losses}")
        logger.debug(f"Baseline loss: {baseline_loss}")

        if not permuted_losses:
            logger.warning("No valid permutations for fake feature. Skipping importance calculation.")
            return

        importance = np.mean(permuted_losses)
        importance_std = np.std(permuted_losses)
        t_statistic, p_value = stats.ttest_1samp(permuted_losses, 0)
        
        # Calculate confidence interval
        confidence_interval = stats.t.interval(0.95, len(permuted_losses)-1, loc=importance, scale=importance_std/np.sqrt(len(permuted_losses)))
        
        feature_importance[fake_feature] = importance
        feature_importance_std[fake_feature] = importance_std
        feature_importance_p_value[fake_feature] = p_value
        
        logger.info(f"Fake feature importance: {importance:.4f} ± {importance_std:.4f} (95% CI: {confidence_interval})")
        logger.info(f"Fake feature p-value: {p_value:.4f}")
        
        logger.debug(f"Fake feature correlation: {fake_temporal.corrcoef(permuted_batch['labels'])}")
        logger.debug(f"Permuted losses: {permuted_losses}")
        logger.debug(f"Baseline loss: {baseline_loss}")

        logger.info(f"Fake temporal feature importance: {importance:.4f} ± {importance_std:.4f} (p-value: {p_value:.4f})")

    def create_correlated_fake_feature(self, time, labels):
        batch_size, seq_len = time.shape
        
        # Create a base temporal feature
        base_temporal = torch.linspace(0, 1, seq_len).unsqueeze(0).repeat(batch_size, 1).to(time.device)
        
        # Add noise to create variation
        noise = torch.randn_like(base_temporal) * 0.1
        
        # Combine base temporal feature with labels to create correlation
        fake_feature_correlation = getattr(self.config, 'fake_feature_correlation', 0.8)
        fake_feature = (1 - fake_feature_correlation) * (base_temporal + noise) + \
                       fake_feature_correlation * labels.unsqueeze(1).repeat(1, seq_len)
        
        # Normalize to [0, 1] range
        fake_feature = (fake_feature - fake_feature.min()) / (fake_feature.max() - fake_feature.min())
        
        return fake_feature

    def calculate_temporal_importance(self, num_samples=1000, initial_permutations=10, max_permutations=100, adaptive_threshold=0.01, early_stop_threshold=0.001, dataloader=None):
        if dataloader is None:
            dataloader = self.val_dataloader()

        samples = []
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
            samples.append(batch)

        if not samples:
            logger.warning("No samples collected from the dataloader. Cannot calculate temporal importance.")
            return {}, {}, {}

        combined_batch = {
            key: torch.cat([sample[key] for sample in samples], dim=0)
            for key in samples[0].keys()
        }

        max_seq_len = self.config.max_seq_len
        for key in ['dynamic_indices', 'dynamic_values', 'dynamic_measurement_indices', 'time']:
            if key in combined_batch and combined_batch[key].size(1) > max_seq_len:
                combined_batch[key] = combined_batch[key][:, :max_seq_len]
        
        if 'seq_len' in combined_batch:
            combined_batch['seq_len'] = torch.clamp(combined_batch['seq_len'], max=max_seq_len)

        combined_batch = {k: v.to(self.device) for k, v in combined_batch.items()}

        with torch.no_grad():
            baseline_output = self(combined_batch)
        baseline_loss = baseline_output.loss.item()

        temporal_importance = {}
        temporal_importance_std = {}
        temporal_importance_p_value = {}
        
        features_to_permute = ['dynamic_indices', 'dynamic_values']
        window_size = self.config.seq_window_size

        for feature in features_to_permute:
            if feature not in combined_batch:
                logger.warning(f"Feature '{feature}' is missing from the combined batch. Skipping temporal importance calculation.")
                continue
            
            feature_data = combined_batch[feature]
            logger.info(f"Processing feature '{feature}' with shape: {feature_data.shape}")

            feature_temporal_importance = []
            feature_temporal_importance_std = []
            feature_temporal_importance_p_value = []

            for start in range(0, max_seq_len - window_size + 1):
                end = start + window_size
                
                permuted_losses = []
                for _ in range(initial_permutations):
                    permuted_data = feature_data.clone()
                    
                    if feature == 'dynamic_values':
                        nan_mask = torch.isnan(permuted_data[:, start:end])
                        valid_mask = ~nan_mask
                        valid_data = permuted_data[:, start:end][valid_mask]
                        
                        if valid_data.numel() == 0:
                            logger.warning(f"No valid data for feature '{feature}' in window {start}-{end}. Skipping permutation.")
                            continue
                        
                        window_permutation = torch.randperm(valid_data.size(0))
                        permuted_valid_data = valid_data[window_permutation]
                        
                        permuted_data[:, start:end][valid_mask] = permuted_valid_data
                    else:
                        window_permutation = torch.randperm(end - start)
                        permuted_data[:, start:end] = permuted_data[:, start:end][:, window_permutation]
                    
                    permuted_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in combined_batch.items()}
                    permuted_batch[feature] = permuted_data

                    with torch.no_grad():
                        permuted_output = self(permuted_batch)
                    permuted_loss = permuted_output.loss.item()
                    permuted_losses.append(permuted_loss - baseline_loss)

                if not permuted_losses:
                    logger.warning(f"No valid permutations for feature '{feature}' in window {start}-{end}. Skipping importance calculation.")
                    continue

                importance = np.mean(permuted_losses)
                std = np.std(permuted_losses)

                if abs(importance) > adaptive_threshold:
                    additional_permutations = min(max_permutations - initial_permutations, 
                                                  int((abs(importance) - adaptive_threshold) / adaptive_threshold * initial_permutations))
                    
                    if additional_permutations > 0:
                        for _ in range(additional_permutations):
                            permuted_data = feature_data.clone()
                            
                            if feature == 'dynamic_values':
                                nan_mask = torch.isnan(permuted_data[:, start:end])
                                valid_mask = ~nan_mask
                                valid_data = permuted_data[:, start:end][valid_mask]
                                
                                if valid_data.numel() == 0:
                                    continue
                                
                                window_permutation = torch.randperm(valid_data.size(0))
                                permuted_valid_data = valid_data[window_permutation]
                                
                                permuted_data[:, start:end][valid_mask] = permuted_valid_data
                            else:
                                window_permutation = torch.randperm(end - start)
                                permuted_data[:, start:end] = permuted_data[:, start:end][:, window_permutation]
                            
                            permuted_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in combined_batch.items()}
                            permuted_batch[feature] = permuted_data

                            with torch.no_grad():
                                permuted_output = self(permuted_batch)
                            permuted_loss = permuted_output.loss.item()
                            permuted_losses.append(permuted_loss - baseline_loss)

                        importance = np.mean(permuted_losses)
                        std = np.std(permuted_losses)

                feature_temporal_importance.append(importance)
                feature_temporal_importance_std.append(std)

                t_statistic, p_value = stats.ttest_1samp(permuted_losses, 0)
                feature_temporal_importance_p_value.append(p_value)

                logger.info(f"Feature '{feature}' temporal importance at window {start}-{end}: {importance:.4f} ± {std:.4f} (p-value: {p_value:.4f})")

            temporal_importance[feature] = feature_temporal_importance
            temporal_importance_std[feature] = feature_temporal_importance_std
            temporal_importance_p_value[feature] = feature_temporal_importance_p_value

        return temporal_importance, temporal_importance_std, temporal_importance_p_value
    
    def on_train_epoch_end(self):
        # Log accumulated metrics for the training phase
        self.log_accumulated_metrics('train')

        # Ensure all processes are synchronized before proceeding
        if dist.is_initialized():
            dist.barrier()

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
        
    def calculate_and_log_importances(self):
        # Calculate feature importance
        feature_importance, feature_importance_std, feature_importance_p_value = self.calculate_feature_importance()
        
        # Log feature importance
        for feature, importance in feature_importance.items():
            self.log(f"feature_importance/{feature}", importance, on_epoch=True, sync_dist=True)
            if feature in feature_importance_std:
                self.log(f"feature_importance_std/{feature}", feature_importance_std[feature], on_epoch=True, sync_dist=True)
            if feature in feature_importance_p_value:
                self.log(f"feature_importance_p_value/{feature}", feature_importance_p_value[feature], on_epoch=True, sync_dist=True)

        # Calculate temporal importance
        temporal_importance, temporal_importance_std, temporal_importance_p_value = self.calculate_temporal_importance()
        
        # Log temporal importance
        for feature, importance_list in temporal_importance.items():
            for i, importance in enumerate(importance_list):
                self.log(f"temporal_importance/{feature}/window_{i}", importance, on_epoch=True, sync_dist=True)
                if feature in temporal_importance_std:
                    self.log(f"temporal_importance_std/{feature}/window_{i}", temporal_importance_std[feature][i], on_epoch=True, sync_dist=True)
                if feature in temporal_importance_p_value:
                    self.log(f"temporal_importance_p_value/{feature}/window_{i}", temporal_importance_p_value[feature][i], on_step=False, on_epoch=True, sync_dist=True)
    
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

        # Perform feature and temporal importance calculations
        self.calculate_and_log_test_importances()

    def calculate_and_log_test_importances(self):
        feature_importance, feature_importance_std, feature_importance_p_value = self.calculate_feature_importance(dataloader=self.test_dataloader())
        temporal_importance, temporal_importance_std, temporal_importance_p_value = self.calculate_temporal_importance(dataloader=self.test_dataloader())

        # Log feature importance
        for feature, importance in feature_importance.items():
            wandb.log({
                f"feature_importance/{feature}/score": importance,
                f"feature_importance/{feature}/std": feature_importance_std.get(feature, 0),
                f"feature_importance/{feature}/p_value": feature_importance_p_value.get(feature, 1),
            })
            
            # Calculate and log confidence interval
            ci = stats.t.interval(0.95, len(self.test_dataloader.dataset)-1, 
                                  loc=importance, 
                                  scale=feature_importance_std.get(feature, 0))
            wandb.log({
                f"feature_importance/{feature}/ci_lower": ci[0],
                f"feature_importance/{feature}/ci_upper": ci[1],
            })

        # Log temporal importance
        for feature, importance_list in temporal_importance.items():
            for i, importance in enumerate(importance_list):
                wandb.log({
                    f"temporal_importance/{feature}/window_{i}/score": importance,
                    f"temporal_importance/{feature}/window_{i}/std": temporal_importance_std[feature][i],
                    f"temporal_importance/{feature}/window_{i}/p_value": temporal_importance_p_value[feature][i],
                })
                
                # Calculate and log confidence interval
                ci = stats.t.interval(0.95, len(self.test_dataloader.dataset)-1, 
                                      loc=importance, 
                                      scale=temporal_importance_std[feature][i])
                wandb.log({
                    f"temporal_importance/{feature}/window_{i}/ci_lower": ci[0],
                    f"temporal_importance/{feature}/window_{i}/ci_upper": ci[1],
                })

        # Create and log charts
        self.log_importance_charts(feature_importance, feature_importance_std, feature_importance_p_value, 
                                   temporal_importance, temporal_importance_std, temporal_importance_p_value)

    def log_importance_charts(self, feature_importance, feature_importance_std, feature_importance_p_value,
                              temporal_importance, temporal_importance_std, temporal_importance_p_value):
        # Feature importance chart
        fig, ax = plt.subplots(figsize=(12, 6))
        features = list(feature_importance.keys())
        y_pos = np.arange(len(features))
        ax.barh(y_pos, list(feature_importance.values()), xerr=list(feature_importance_std.values()), align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance')
        wandb.log({"feature_importance_chart": wandb.Image(fig)})
        plt.close(fig)

        # Temporal importance chart
        for feature, importance_list in temporal_importance.items():
            fig, ax = plt.subplots(figsize=(12, 6))
            windows = list(range(len(importance_list)))
            ax.plot(windows, importance_list)
            ax.fill_between(windows, 
                            [i-s for i, s in zip(importance_list, temporal_importance_std[feature])],
                            [i+s for i, s in zip(importance_list, temporal_importance_std[feature])],
                            alpha=0.3)
            ax.set_xlabel('Window')
            ax.set_ylabel('Importance')
            ax.set_title(f'Temporal Importance: {feature}')
            wandb.log({f"temporal_importance_chart/{feature}": wandb.Image(fig)})
            plt.close(fig)

    def on_after_backward(self):
        # Log gradient norms for specific layers
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.log(f'grad_norm/{name}', param.grad.norm(), on_step=True, on_epoch=False)

    def forward(self, batch, **kwargs):
        if not isinstance(batch, dict):
            raise TypeError("Input 'batch' should be a dictionary.")

        required_keys = ['dynamic_indices', 'dynamic_values', 'dynamic_measurement_indices', 'time']
        for key in required_keys:
            if key not in batch or batch[key] is None:
                logger.warning(f"Required key '{key}' is missing or None in the input batch. Using dummy data.")
                batch[key] = torch.zeros((batch['dynamic_values'].shape[0], batch['dynamic_values'].shape[1]), device=self.device)
            
            if isinstance(batch[key], torch.Tensor):
                logger.debug(f"Batch[{key}] shape: {batch[key].shape}, dtype: {batch[key].dtype}")
                if key in ['dynamic_indices', 'dynamic_measurement_indices']:
                    batch[key] = batch[key].long()
                if key == 'dynamic_values':
                    nan_mask = torch.isnan(batch[key])
                    if nan_mask.any():
                        logger.debug(f"NaN values found in {key}: {nan_mask.sum().item()} / {nan_mask.numel()}")
                        # Replace NaN values with a special value or mask
                        batch[key] = torch.where(nan_mask, torch.tensor(0.0, device=batch[key].device), batch[key])
            else:
                logger.warning(f"Unexpected type for key '{key}': {type(batch[key])}")

        if getattr(self.config, 'use_fake_feature', True):
            if 'fake_temporal_feature' not in batch:
                logger.warning("Fake temporal feature is not in the batch!")
            else:
                logger.debug(f"Fake temporal feature shape: {batch['fake_temporal_feature'].shape}")
    
        # Move batch to the correct device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Check for NaNs after moving to device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and torch.isnan(value).any():
                logger.warning(f"NaNs detected in batch['{key}']")

        # Move kwargs tensors to the correct device
        kwargs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}

        if not self.cfg.use_static_features:
            batch.pop('static_indices', None)
            batch.pop('static_measurement_indices', None)

        logger.debug("Entering model forward pass")
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
        logger.debug("Exiting model forward pass")

        # Ensure loss is not detached
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            outputs.loss = outputs.loss.requires_grad_(True)

        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            eps=1e-8
        )
        
        logger.info(f"Configured optimizer: AdamW with lr={self.learning_rate}, weight_decay={self.weight_decay}")
        
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
            elif self.lr_scheduler_type == "linear_warmup":
                warmup_steps = int(self.num_training_steps * 0.05)  # 5% of total steps for warmup
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=self.num_training_steps
                )
            elif self.lr_scheduler_type == "one_cycle":
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=self.learning_rate,
                    total_steps=self.num_training_steps,
                    pct_start=0.05,  # 5% of training for warmup
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
            
            logger.info(f"Configured learning rate scheduler: {self.lr_scheduler_type}")
            return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
        else:
            logger.info("No learning rate scheduler configured")
            return optimizer

        # Log the gradient clipping value
        logger.info(f"Gradient clipping value: {self.trainer.gradient_clip_val}")

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
            "log_model": False,
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
            self.save_dir.mkdir(parents=True, exist_ok=True)
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
    def __init__(self, vocab_size, oov_index, include_labels=True, static_size=8, max_seq_len=168, use_static_features=True, model_dtype=torch.float32):
        self.vocab_size = vocab_size
        self.oov_index = oov_index
        self.include_labels = include_labels
        self.static_size = static_size
        self.max_seq_len = max_seq_len
        self.use_static_features = use_static_features
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"CollateFunction initialized with use_static_features: {self.use_static_features}")
        self.logger.info(f"CollateFunction initialized with max_seq_len: {self.max_seq_len}")

        self.use_fake_feature = use_fake_feature
        self.fake_feature_correlation = fake_feature_correlation
        self.logger.info(f"CollateFunction initialized with use_fake_feature: {self.use_fake_feature}")
        self.logger.info(f"CollateFunction initialized with fake_feature_correlation: {self.fake_feature_correlation}")

        self.dtype = model_dtype

    def __call__(self, batch):
        self.logger.info(f"CollateFunction use_static_features: {self.use_static_features}")
        try:
            # Filter out None items
            batch = [item for item in batch if item is not None]
            if len(batch) == 0:
                return None

            # Use the fixed max_seq_len
            max_seq_len = self.max_seq_len

            # Pad sequences
            collated_batch = {
                'dynamic_indices': self.pad_and_stack([item['dynamic_indices'] for item in batch], max_seq_len),
                'dynamic_values': self.pad_and_stack([item['dynamic_values'] for item in batch], max_seq_len, pad_value=0.0),
                'dynamic_measurement_indices': self.pad_and_stack([item['dynamic_measurement_indices'] for item in batch], max_seq_len),
                'time': self.pad_and_stack([item['time'] for item in batch], max_seq_len, pad_value=0.0),
                'seq_len': torch.tensor([min(len(item['dynamic_indices']), max_seq_len) for item in batch], dtype=torch.long),
            }

            if self.use_static_features:
                collated_batch['static_indices'] = self.pad_and_stack([item['static_indices'] for item in batch], self.static_size)
                collated_batch['static_measurement_indices'] = self.pad_and_stack([item['static_measurement_indices'] for item in batch], self.static_size)

            if self.include_labels:
                collated_batch['labels'] = torch.stack([item['labels'] for item in batch]).squeeze(1)

            if self.use_fake_feature:
                fake_feature = self.create_correlated_fake_feature(collated_batch['time'], collated_batch['labels'])
                collated_batch['fake_temporal_feature'] = fake_feature.to(self.dtype)
                self.logger.info(f"Added fake_temporal_feature with shape: {fake_feature.shape}")

            self.logger.info(f"Collated batch keys: {collated_batch.keys()}")
            self.logger.info(f"Collated batch sizes: {[(k, v.shape) for k, v in collated_batch.items()]}")

            # Convert all float tensors to the specified dtype
            for key, value in collated_batch.items():
                if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
                    collated_batch[key] = value.to(self.dtype)

            return collated_batch
        except Exception as e:
            self.logger.error(f"Error in collate function: {str(e)}")
            self.logger.error(f"Batch size: {len(batch)}")
            for i, item in enumerate(batch):
                self.logger.error(f"Item {i} keys: {item.keys()}")
                for k, v in item.items():
                    self.logger.error(f"Item {i} {k} shape: {v.shape if isinstance(v, torch.Tensor) else 'N/A'}")
            raise

    def pad_and_stack(self, sequences, max_length, pad_value=0):
        # Convert to list of tensors if not already
        sequences = [torch.as_tensor(seq) for seq in sequences]
        
        # Truncate or pad sequences to max_length
        sequences = [seq[:max_length] if len(seq) > max_length else seq for seq in sequences]
        
        # Create output tensor
        out_dims = (len(sequences), max_length) + sequences[0].shape[1:]
        out_tensor = sequences[0].new_full(out_dims, pad_value)
        
        # Copy data from sequences into output tensor
        for i, seq in enumerate(sequences):
            out_tensor[i, :len(seq)] = seq
        
        # Convert the output tensor to the specified dtype if it's a floating point tensor
        if out_tensor.dtype.is_floating_point:
            out_tensor = out_tensor.to(self.dtype)
        
        return out_tensor

def setup_distributed():
    if dist.is_initialized():
        logger.info("Process group already initialized.")
        return dist.get_world_size(), dist.get_rank()
    
    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        dist.init_process_group(backend='nccl', init_method='env://')
    else:
        world_size = 1
        rank = 0
    return world_size, rank

def initialize_wandb(cfg, rank):
    if rank == 0:
        if wandb.run is None:
            wandb.init(project=cfg.wandb_logger_kwargs.get("project", "default_project"),
                       name=cfg.wandb_logger_kwargs.get("name", "default_run"),
                       config=cfg.to_dict())
            logger.info("Initialized new wandb run")
        else:
            logger.info("Using existing wandb run")
    return wandb.run is not None

@task_wrapper
def train(cfg: FinetuneConfig, train_pyd, tuning_pyd, held_out_pyd, vocabulary_config: VocabularyConfig, oov_index: int, wandb_logger: WandbLogger | None = None):
    # Set up the standard logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    start_time = time.time()
    logger.info(f"Train function started at {start_time}")

    # Initialize rank
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # Initialize Wandb only for the main process
    wandb_initialized = initialize_wandb(cfg, rank)

    logger.info(f"Entering train function")

    logger.info(f"Train dataset length: {len(train_pyd)}")
    logger.info(f"Tuning dataset length: {len(tuning_pyd)}")
    logger.info(f"Held-out dataset length: {len(held_out_pyd)}")

    logger.info(f"FinetuneConfig use_static_features: {cfg.use_static_features}")
    logger.info(f"Config use_static_features: {cfg.config.use_static_features}")

    # Set up distributed environment
    world_size, rank = setup_distributed()
    logger.info(f"Distributed setup complete. Rank: {rank}, World Size: {world_size} after {time.time() - start_time:.2f} seconds")

    # Update data config
    cfg.update_data_config()
    logger.info(f"Data config updated after {time.time() - start_time:.2f} seconds")
    
    # Load the vocabulary_config from the correct file
    vocabulary_config_path = "data/labs/vocabulary_config.json" if cfg.use_labs else "data/vocabulary_config.json"
    with open(vocabulary_config_path, "r") as f:
        vocabulary_config_dict = json.load(f)
    vocabulary_config = VocabularyConfig.from_dict(vocabulary_config_dict)
    logger.info(f"Vocabulary config loaded after {time.time() - start_time:.2f} seconds")

    # Calculate OOV index
    dynamic_indices_vocab_size = vocabulary_config.vocab_sizes_by_measurement.get("dynamic_indices", 0)
    oov_index = cfg.config.vocab_size  # Set oov_index to vocab_size
    logger.info(f"Calculated OOV index: {oov_index} after {time.time() - start_time:.2f} seconds")
    
    # Log the data directories
    logger.info(f"Using data directory: {cfg.data_config.save_dir}")
    logger.info(f"Using DL reps directory: {cfg.data_config.dl_reps_dir}")

    # Check if wandb is already initialized
    if wandb.run is None:
        if wandb_logger is None:
            # If no WandbLogger is provided and wandb is not initialized, create a new one
            wandb_kwargs = cfg.wandb_logger_kwargs.copy()
            project = wandb_kwargs.pop('project', 'default_project')
            name = wandb_kwargs.pop('name', 'default_run')
            
            # Remove unsupported parameters
            wandb_kwargs.pop('team', None)
            wandb_kwargs.pop('do_log_graph', None)
            
            wandb_logger = WandbLogger(
                project=project,
                name=name,
                save_dir=cfg.save_dir,
                **wandb_kwargs
            )
            logger.info("Initialized new WandbLogger in train function")
        else:
            logger.info("Using provided WandbLogger")
    else:
        if wandb_logger is None:
            # If wandb is already initialized but no logger is provided, create one with the existing run
            wandb_logger = WandbLogger(experiment=wandb.run)
            logger.info("Created WandbLogger from existing wandb run")
        else:
            logger.info("Using provided WandbLogger with existing wandb run")

    # Log to both standard logger and WandbLogger
    logger.info("Starting training process")
    if wandb.run is not None:
        wandb.log({"info": "Starting training process"})

    wandb_handler = WandbLoggerHandler(wandb_logger)
    logger.addHandler(wandb_handler)

    L.seed_everything(cfg.seed)

    config = cfg.config
    data_config = cfg.data_config
    optimization_config = cfg.optimization_config

    if not hasattr(config, 'problem_type') or config.problem_type is None:
        config.problem_type = "single_label_classification"

    model_params = dict()
    if cfg.pretrained_weights_fp is not None:
        model_params["pretrained_weights_fp"] = cfg.pretrained_weights_fp

    # Initialize model
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
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
        **model_params
    ).to(device)
    logger.info(f"Model initialized after {time.time() - start_time:.2f} seconds")

    # Set dtype for the entire model
    LM.set_dtype(torch.float32)

    # Check if all parameters are on the correct device
    on_cpu = any(p.device.type == "cpu" for p in LM.parameters())
    if on_cpu:
        logger.warning("Some parameters are still on CPU. Moving all to CUDA...")
        LM = LM.cuda()
    
    if not cfg.use_static_features:
        # Remove static features from the batch
        for dataset in [train_pyd, tuning_pyd, held_out_pyd]:
            dataset.static_indices = None
            dataset.static_measurement_indices = None

    # Ensure batch and layer normalization are synchronized across GPUs
    if torch.cuda.device_count() > 1:
        LM = torch.nn.SyncBatchNorm.convert_sync_batchnorm(LM)

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
        use_static_features=cfg.use_static_features,
        model_dtype=LM.dtype,
        use_fake_feature=cfg.config.use_fake_feature,
        fake_feature_correlation=cfg.config.fake_feature_correlation
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

    # Add the callbacks
    callbacks.append(NCCLErrorHandler())
    callbacks.extend([GradientCheckCallback(), AttentionMechanismSwitch()])

    logger.info("Setting up trainer")
    
    # Create a GradScaler for mixed precision training
    scaler = GradScaler()

    # Update trainer configuration
    cfg.trainer_config["precision"] = "16-mixed"

    logger.info("Setting up distributed environment")

    logger.info("Creating trainer")
    trainer = L.Trainer(
        **cfg.trainer_config,
        callbacks=callbacks,
        logger=wandb_logger,
        max_epochs=optimization_config.get('max_epochs', 100),  # Ensure this is correctly passed
        gradient_clip_val=config.max_grad_norm,
        gradient_clip_algorithm="norm",
        enable_progress_bar=True,
        deterministic=False,
        devices=torch.cuda.device_count(),  # Use all available GPUs
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_true",
    )
    logger.info(f"Trainer created after {time.time() - start_time:.2f} seconds")

    # Before wrapping with DistributedDataParallel
    param_sum = sum(p.sum() for p in LM.parameters())
    logger.info(f"Sum of parameters before DDP: {param_sum}")
    dist.barrier()
    logger.info("All processes passed barrier after trainer creation")

    logger.info(f"NCCL is available: {torch.distributed.is_nccl_available()}")
    logger.info(f"Gloo is available: {torch.distributed.is_gloo_available()}")

    # Log model parameters
    total_params = sum(p.numel() for p in LM.parameters())
    logger.info(f"Total parameters: {total_params}")
    logger.info(f"Model memory usage: {sum(p.numel() * p.element_size() for p in LM.parameters()) / 1024**2:.2f} MB")
    
    # Create distributed samplers for the dataloaders
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_pyd, shuffle=True, drop_last=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(tuning_pyd, shuffle=False, drop_last=True)
    test_sampler = torch.utils.data.distributed.DistributedSampler(held_out_pyd, shuffle=False, drop_last=True)

    # Create the dataloaders
    logger.info(f"Process {rank}/{world_size} creating dataloaders")
    train_dataloader = DataLoader(
        train_pyd,
        batch_size=optimization_config['batch_size'],
        sampler=train_sampler,
        num_workers=optimization_config['num_dataloader_workers'],
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    logger.info(f"Train dataloader created after {time.time() - start_time:.2f} seconds")

    tuning_dataloader = DataLoader(
        tuning_pyd,
        batch_size=optimization_config['validation_batch_size'],
        sampler=val_sampler,
        num_workers=optimization_config['num_dataloader_workers'],
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )

    held_out_dataloader = DataLoader(
        held_out_pyd,
        batch_size=optimization_config['validation_batch_size'],
        sampler=test_sampler,
        num_workers=optimization_config['num_dataloader_workers'],
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    logger.info(f"All dataloaders created after {time.time() - start_time:.2f} seconds")

    # Set the test dataset for the model
    LM.test_dataset = held_out_pyd

    # Start training
    logger.info(f"Process {rank}/{world_size} about to start training")
    torch.cuda.synchronize()
    
    trainer.fit(model=LM, train_dataloaders=train_dataloader, val_dataloaders=tuning_dataloader)

    torch.cuda.synchronize()

    logger.info(f"Process {rank}/{world_size} completed training after {time.time() - start_time:.2f} seconds")

    # Evaluation
    logger.info("Training completed. Evaluating on validation and test sets.")
    tuning_metrics = trainer.validate(model=LM, dataloaders=tuning_dataloader, ckpt_path="best") if len(tuning_pyd) > 0 else None
    held_out_metrics = trainer.test(model=LM, dataloaders=held_out_dataloader, ckpt_path="best") if len(held_out_pyd) > 0 else None

    logger.info(f"Evaluation completed after {time.time() - start_time:.2f} seconds")

    # Ensure all processes are synchronized before ending
    if dist.is_initialized():
        dist.barrier()

    # Only finish the Wandb run from the main process
    if rank == 0 and wandb.run is not None:
        wandb.finish()

    return None, tuning_metrics, held_out_metrics

__all__ = ['FinetuneConfig', 'train']