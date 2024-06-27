import dataclasses
import json
import os
import random
from collections.abc import Sequence
from pathlib import Path
import pathlib
from typing import Dict, Any
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
import torch.multiprocessing as mp

import logging

import numpy as np

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
        """Initializes the Lightning Module."""
        super().__init__()
        if isinstance(config, dict):
            config = StructuredTransformerConfig(**config)
        if isinstance(optimization_config, dict):
            optimization_config = OptimizationConfig(**optimization_config)
        
        self.config = config
        self.optimization_config = optimization_config
        self.cfg = cfg
        self.do_debug_mode = do_debug_mode
        
        # Use the optimization_config object directly since we've already converted it
        self.learning_rate = get_config_value(optimization_config, 'init_lr', 1e-3)
        self.num_training_steps = optimization_config.max_training_steps
        self.weight_decay = optimization_config.weight_decay
        self.use_lr_scheduler = getattr(optimization_config, 'use_lr_scheduler', False)
        self.lr_scheduler_type = getattr(optimization_config, 'lr_scheduler_type', 'cosine')
        self.end_lr = optimization_config.end_lr
        self.end_lr_frac_of_init_lr = optimization_config.end_lr_frac_of_init_lr
        self.train_metrics = {}
        self.val_metrics = {}
        self.test_metrics = {}

        self.save_hyperparameters(
            {
                "config": config.to_dict(),
                "optimization_config": dataclasses.asdict(optimization_config),
            }
        )
        self.build_metrics()

        # Load the vocabulary_config from a file
        vocabulary_config_path = "data/vocabulary_config.json"
        with open(vocabulary_config_path, "r") as f:
            vocabulary_config_dict = json.load(f)
        vocabulary_config = VocabularyConfig.from_dict(vocabulary_config_dict)

        if pretrained_weights_fp is None or pretrained_weights_fp == "skip":
            self.model = ESTForStreamClassification(config, vocabulary_config)
        else:
            self.model = ESTForStreamClassification.from_pretrained(pretrained_weights_fp, config=config, vocabulary_config=vocabulary_config)

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
        
        if loss is not None:
            self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.log('train_accuracy', outputs.accuracy, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            
            # Accumulate metrics
            self._accumulate_metrics('train', {
                'loss': loss.item(),
                'accuracy': outputs.accuracy,
                'auc': outputs.auc if outputs.auc is not None else 0.0,
            })
            
            # Debugging print statements
            if batch_idx % 100 == 0:
                logger.info(f"Step {batch_idx}: train_loss={loss.item():.4f}, train_accuracy={outputs.accuracy:.4f}")
            
            # Log gradient norm
            if self.global_step % 100 == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.log('grad_norm', grad_norm)
        
        return loss

    def validation_step(self, batch, batch_idx):
        logger.debug(f"Validation step - Batch keys: {batch.keys() if batch is not None else 'None'}")
        
        outputs = self.model(batch, labels=batch['labels'])
        loss = outputs.loss
        
        # Add learning rate to debug info
        if outputs.debug_info is not None:
            outputs.debug_info["learning_rate"] = self.trainer.optimizers[0].param_groups[0]['lr']
        
        logger.debug(f"Model outputs: {outputs}")
        logger.debug(f"Loss: {loss}")
        
        if loss is not None:
            if torch.isnan(loss):
                logger.error("NaN loss detected in validation step")
                logger.error(f"Batch that caused NaN loss: {batch}")
                self.log('val_loss', torch.tensor(float('inf')))
                return None
            
            # Accumulate metrics
            self._accumulate_metrics('val', {
                'loss': loss.item(),
                'accuracy': outputs.accuracy,
                'auc': outputs.auc if outputs.auc is not None else BinaryAUROC()(outputs.preds, batch['labels']).item(),
            })
     
        if loss is not None:
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log('val_loss_epoch', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            
        # Log debugging information
        if outputs.debug_info:
            for key, value in outputs.debug_info.items():
                self._accumulate_metrics('val', {f'debug_{key}': value})
        
        return loss

    def test_step(self, batch, batch_idx):
        logger.debug(f"Test step - Batch keys: {batch.keys() if batch is not None else 'None'}")
        if batch is None:
            self.log('test_loss', None)
            return None
        
        outputs = self.model(batch, labels=batch['labels'])
        loss = outputs.loss
        
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

    def forward(self, batch, **kwargs):
        if not isinstance(batch, dict):
            raise TypeError("Input 'batch' should be a dictionary.")
        
        dynamic_indices = batch.get("dynamic_indices")
        dynamic_counts = batch.get("dynamic_counts")
        
        if dynamic_indices is None:
            raise ValueError("'dynamic_indices' must be provided in the batch.")
        
        outputs = self.model(
            dynamic_indices=dynamic_indices,
            dynamic_counts=dynamic_counts,
            **kwargs
        )
        
        return outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs

    def _accumulate_metrics(self, prefix, metrics):
        for key, value in metrics.items():
            if key not in self.__dict__[f'{prefix}_metrics']:
                self.__dict__[f'{prefix}_metrics'][key] = []
            self.__dict__[f'{prefix}_metrics'][key].append(value)

    def on_train_epoch_end(self):
        self._log_epoch_metrics('train')

    def on_validation_epoch_end(self):
        self._log_epoch_metrics('val')

    def on_test_epoch_end(self):
        self._log_epoch_metrics('test')

    def _log_epoch_metrics(self, prefix):
        metrics = self.__dict__[f'{prefix}_metrics']
        for key, values in metrics.items():
            avg_value = sum(values) / len(values)
            self.log(f'{prefix}_{key}_epoch', avg_value, on_step=False, on_epoch=True, logger=True)
        
        # Log learning rate
        if prefix in ['train', 'val'] and self.trainer.optimizers:
            self.log(f'{prefix}_learning_rate_epoch', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=False, on_epoch=True, logger=True)
        
        # Clear accumulated metrics
        self.__dict__[f'{prefix}_metrics'].clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Clip gradients
        self.clip_gradients(optimizer, gradient_clip_val=1.0, gradient_clip_algorithm="norm")

        if self.use_lr_scheduler:
            if self.lr_scheduler_type == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,  # Changed from optimizer to self.optimizer
                    T_max=self.num_training_steps,
                    eta_min=self.end_lr
                )
            elif self.lr_scheduler_type == "linear":
                scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,  # Changed from optimizer to self.optimizer
                    start_factor=1.0,
                    end_factor=self.end_lr_frac_of_init_lr,
                    total_iters=self.num_training_steps
                )
            else:
                raise ValueError(f"Unknown scheduler type: {self.lr_scheduler_type}")

            scheduler = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
            return [optimizer], [scheduler]
        else:
            return optimizer

@hydra_dataclass
class FinetuneConfig:
    experiment_dir: str | Path | None = "${load_from_model_dir}/finetuning"
    load_from_model_dir: str | Path | None = omegaconf.MISSING
    task_df_name: str | None = omegaconf.MISSING
    optimization_config_path: str = omegaconf.MISSING
    pretrain_config_path: str | None = None
    dataset_path: str | None = None
    pretraining_metrics_config: dict[str, Any] | None = None
    final_validation_metrics_config: dict[str, Any] | None = None
    do_final_validation_metrics_config: dict[str, Any] | None = None
    do_final_validation_on_metrics: bool = False
    pretrained_weights_fp: Path | str | None = "skip"

    save_dir: str | None = (
        "${experiment_dir}/${task_df_name}/"
        "subset_size_${data_config.train_subset_size}/"
        "subset_seed_${data_config.train_subset_seed}/"
        "${now:%Y-%m-%d_%H-%M-%S}"
    )

    wandb_logger_kwargs: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "name": "${task_df_name}_finetuning",
            "project": None,
            "log_model": True,
        }
    )

    wandb_experiment_config_kwargs: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "save_dir": "${save_dir}",
        }
    )

    do_overwrite: bool = False
    seed: int = 1

    config: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            **{k: None for k in StructuredTransformerConfig().to_dict().keys()},
            "task_specific_params": {
                "pooling_method": "last",
                "num_samples": None,
            },
        }
    )

    optimization_config: Dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            'batch_size': 32,
            'validation_batch_size': 64,
            'num_dataloader_workers': 4,
            'max_epochs': 100,
            'gradient_accumulation': 1,
        }
    )

    data_config: dict[str, Any] | None = dataclasses.field(
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

    data_config_path: str | None = None
    
    trainer_config: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "accelerator": "auto",
            "devices": "auto",
            "detect_anomaly": False,
            "default_root_dir": "${save_dir}/model_checkpoints",
            "log_every_n_steps": 10,
        }
    )
    do_use_filesystem_sharing: bool = True

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

    def __call__(self, batch):
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            logger.warning("All items in batch are None")
            return None

        valid_items = [item for item in batch if all(k in item for k in ['dynamic_indices', 'dynamic_counts', 'labels'])]
        
        if not valid_items:
            logger.warning("No valid items found in batch")
            return None

        max_allowed_index = self.vocab_size - 1

        def safe_pad_sequence(sequences, max_val, dtype):
            try:
                sequences = [s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=dtype) for s in sequences]
                sequences = [torch.clamp(s, max=max_val) for s in sequences]
                padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
                return padded
            except Exception as e:
                logger.error(f"Error in safe_pad_sequence: {str(e)}")
                return None

        try:
            collated_batch = {
                'dynamic_indices': safe_pad_sequence([item['dynamic_indices'] for item in valid_items], max_allowed_index, torch.int32),
                'dynamic_counts': safe_pad_sequence([torch.nan_to_num(item['dynamic_counts'], nan=0.0) for item in valid_items], float('inf'), torch.float32),
                'labels': torch.stack([torch.tensor(item['labels']) for item in valid_items]).squeeze()
            }

            # Check if any of the tensors are None
            if any(value is None for value in collated_batch.values()):
                logger.error("One or more tensors in collated_batch are None")
                return None

            # Add shape and dtype information for debugging
            for key, value in collated_batch.items():
                logger.info(f"{key}: shape = {value.shape}, dtype = {value.dtype}, max = {value.max().item()}, min = {value.min().item()}")

            return collated_batch

        except Exception as e:
            logger.error(f"Error in collate_fn: {str(e)}")
            return None

def worker_init_fn(worker_id):
    np.random.seed(worker_id)
    random.seed(worker_id)

@task_wrapper
def train(cfg: FinetuneConfig, train_pyd, tuning_pyd, held_out_pyd, wandb_logger: WandbLogger | None = None):

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    if wandb_logger is None:
        wandb.init(project=cfg.wandb_logger_kwargs.get('project', 'default_project'),
                   name=cfg.wandb_logger_kwargs.get('name', 'default_run'),
                   config=cfg.to_dict())
    else:
        wandb.init(config=cfg.to_dict())

    try:
        L.seed_everything(cfg.seed)
        if cfg.do_use_filesystem_sharing:
            torch.multiprocessing.set_sharing_strategy("file_system")

        config = cfg.config
        data_config = cfg.data_config
        optimization_config = cfg.optimization_config

        # Ensure problem_type is set
        if not hasattr(config, 'problem_type') or config.problem_type is None:
            config.problem_type = "single_label_classification"

        model_params = dict()
        if cfg.pretrained_weights_fp is not None:
            model_params["pretrained_weights_fp"] = cfg.pretrained_weights_fp

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Initializing model")
        LM = ESTForStreamClassificationLM(config, optimization_config, cfg, **model_params).to(device)

        logger.info(f"Train dataset length: {len(train_pyd)}")
        logger.info(f"First few items from train dataset:")
        for i in range(min(5, len(train_pyd))):
            item = train_pyd[i]
            logger.info(f"Item {i}: {item}")

        logger.info("Creating data loaders")

        collate_fn = CollateFunction(config.vocab_size)
        train_dataloader = TimeoutDataLoader(
            train_pyd,
            batch_size=optimization_config['batch_size'],
            num_workers=optimization_config['num_dataloader_workers'],
            collate_fn=collate_fn,
            shuffle=True,
            persistent_workers=True,
            prefetch_factor=2,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            timeout=600  # 10 minutes timeout
        )

        tuning_dataloader = TimeoutDataLoader(
            tuning_pyd,
            batch_size=optimization_config['validation_batch_size'],
            num_workers=optimization_config['num_dataloader_workers'],
            collate_fn=collate_fn,
            shuffle=False,
            persistent_workers=True,
            prefetch_factor=2,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            timeout=120  # 2 minutes timeout
        )

        held_out_dataloader = TimeoutDataLoader(
            held_out_pyd,
            batch_size=optimization_config['validation_batch_size'],
            num_workers=optimization_config['num_dataloader_workers'],
            collate_fn=collate_fn,
            shuffle=False,
            persistent_workers=True,
            prefetch_factor=2,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            timeout=120  # 2 minutes timeout
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=cfg.save_dir / "checkpoints",
            filename="{epoch}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_weights_only=True,
            save_last=True,
            auto_insert_metric_name=False,
        )

        callbacks = [
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                dirpath=cfg.save_dir / "checkpoints",
                filename="{epoch}-{val_loss_epoch:.2f}",  # Changed from val_loss to val_loss_epoch
                monitor="val_loss_epoch",  # Changed from val_loss to val_loss_epoch
                mode="min",
                save_top_k=1,
                save_weights_only=True,
                save_last=True,
                auto_insert_metric_name=False,
            ),
            EarlyStopping(
                monitor='val_loss_epoch',  # Changed from val_loss to val_loss_epoch
                min_delta=0.00,
                patience=cfg.optimization_config['patience'],
                verbose=False,
                mode='min'
            )
        ]

        trainer_kwargs = dict(
            **cfg.trainer_config,
            callbacks=callbacks,
            logger=wandb_logger,
        )

        trainer_kwargs["max_epochs"] = optimization_config.get('max_epochs', 100)  # Add a default value

        if optimization_config.get('gradient_accumulation', 1) > 1:
            trainer_kwargs["accumulate_grad_batches"] = optimization_config['gradient_accumulation']

        logger.info("Setting up trainer")
        if cfg.trainer_config.get("strategy") == "ddp_find_unused_parameters_true":
            from lightning.pytorch.strategies import DDPStrategy
            strategy = DDPStrategy(find_unused_parameters=True)
            cfg.trainer_config["strategy"] = strategy

        trainer = L.Trainer(
            **cfg.trainer_config,
            callbacks=callbacks,
            logger=wandb_logger,
            precision="16-mixed",  # Enable mixed precision
            accumulate_grad_batches=optimization_config.get('gradient_accumulation', 1),
            max_epochs=optimization_config.get('max_epochs', 100),
        )

        logger.info("Starting training")
        trainer.fit(model=LM, train_dataloaders=train_dataloader, val_dataloaders=tuning_dataloader)

        if len(tuning_pyd) == 0:
            tuning_metrics = None
        else:
            tuning_metrics = trainer.validate(model=LM, dataloaders=tuning_dataloader, ckpt_path="best")

        if len(held_out_pyd) == 0:
            held_out_metrics = None
        else:
            held_out_metrics = trainer.test(model=LM, dataloaders=held_out_dataloader, ckpt_path="best")

        return None, tuning_metrics, held_out_metrics

    except Exception as e:
        logger.exception(f"An error occurred during training: {str(e)}")
        return None, None, None

__all__ = ['FinetuneConfig', 'train']