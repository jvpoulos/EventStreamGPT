import dataclasses
import json
import os
import random
from collections.abc import Sequence
from pathlib import Path
import pathlib
from typing import Any
import wandb
from torch.optim.lr_scheduler import LambdaLR

import lightning as L
import omegaconf
from omegaconf import DictConfig
import torch
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
import torch.multiprocessing as mp

import logging

import numpy as np

# Configure the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class TimeoutDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        self.timeout = kwargs.pop('timeout', 60)  # Default timeout of 60 seconds
        super().__init__(*args, **kwargs)

    def __iter__(self):
        return TimeoutDataLoaderIter(super().__iter__(), self.timeout)

class TimeoutDataLoaderIter:
    def __init__(self, iterator, timeout):
        self.iterator = iterator
        self.timeout = timeout

    def __next__(self):
        start = time.time()
        while True:
            try:
                return next(self.iterator)
            except StopIteration:
                raise
            except Exception as e:
                if time.time() - start > self.timeout:
                    raise TimeoutError(f"DataLoader timed out after {self.timeout} seconds") from e

class ESTForStreamClassificationLM(L.LightningModule):
    """A PyTorch Lightning Module for a `ESTForStreamClassification` model."""

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
        self.cfg = cfg  # Assign the passed `cfg` instance to the instance attribute
        self.do_debug_mode = do_debug_mode

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
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('train_accuracy', outputs.accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            
            # Log debugging information
            if outputs.debug_info:
                for key, value in outputs.debug_info.items():
                    self.log(f'train_{key}', value)
    
        return loss

    def validation_step(self, batch, batch_idx):
        logger.debug(f"Validation step - Batch keys: {batch.keys() if batch is not None else 'None'}")
        
        outputs = self.model(batch, labels=batch['labels'])
        loss = outputs.loss
        
        logger.debug(f"Model outputs: {outputs}")
        logger.debug(f"Loss: {loss}")
        
        if loss is not None:
            if torch.isnan(loss):
                logger.error("NaN loss detected in validation step")
                logger.error(f"Batch that caused NaN loss: {batch}")
                self.log('val_loss', torch.tensor(float('inf')))
                return None
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        else:
            self.log('val_loss', torch.tensor(float('inf')), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # Log accuracy and AUROC from model outputs
        if outputs.accuracy is not None:
            self.log('val_acc', outputs.accuracy)
        
        if outputs.auc is not None:
            self.log('val_auroc', outputs.auc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        else:
            # Compute AUROC using torchmetrics if not provided by the model
            auroc = BinaryAUROC()(outputs.preds, batch['labels'])
            self.log('val_auroc', auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # Log debugging information
        if outputs.debug_info:
            for key, value in outputs.debug_info.items():
                self.log(f'val_{key}', value)
        
        return loss

    def test_step(self, batch, batch_idx):
        logger.debug(f"Test step - Batch keys: {batch.keys() if batch is not None else 'None'}")
        if batch is None:
            self.log('test_loss', None)
            return None
        
        outputs = self.model(batch, labels=batch['labels'])
        loss = outputs.loss
        
        if loss is not None:
            self.log('test_loss', loss)
        else:
            self.log('test_loss', torch.tensor(float('inf')))
        
        # Log accuracy and AUROC
        if outputs.accuracy is not None:
            self.log('test_acc', outputs.accuracy)
        
        if outputs.auc is not None:
            self.log('test_auroc', outputs.auc)
        else:
            # Compute AUROC using torchmetrics if not provided by the model
            auroc = BinaryAUROC()(outputs.preds, batch['labels'])
            self.log('test_auroc', auroc)
        
        # Log debugging information
        if outputs.debug_info:
            for key, value in outputs.debug_info.items():
                self.log(f'test_{key}', value)
        
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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.optimization_config.init_lr,
            weight_decay=self.optimization_config.weight_decay,
        )
        
        num_warmup_steps = self.optimization_config.lr_num_warmup_steps
        num_training_steps = self.optimization_config.max_training_steps

        def get_custom_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
            # custom scheduler that combines linear warmup and cosine decay
            def lr_lambda(current_step):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
                return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

            return LambdaLR(optimizer, lr_lambda)

        scheduler = get_custom_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            min_lr_ratio=self.optimization_config.end_lr_frac_of_init_lr
        )
        
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
            "gradient_clip_val": 1.0,
            "gradient_clip_algorithm": "norm",
            "monitor": "val_loss",
        }

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
    optimization_config: OptimizationConfig = dataclasses.field(default_factory=lambda: OptimizationConfig())
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


def collate_fn(batch):
    if batch is None or len(batch) == 0:
        return None

    valid_items = [item for item in batch if item is not None and all(k in item for k in ['dynamic_indices', 'dynamic_counts', 'labels'])]

    if not valid_items:
        return None

    collated_batch = {
        'dynamic_indices': torch.nn.utils.rnn.pad_sequence([item['dynamic_indices'] for item in valid_items], batch_first=True),
        'dynamic_counts': torch.nn.utils.rnn.pad_sequence([item['dynamic_counts'] for item in valid_items], batch_first=True),
        'labels': torch.stack([item['labels'] for item in valid_items]).squeeze()
    }

    return collated_batch
    
def worker_init_fn(worker_id):
    np.random.seed(worker_id)
    random.seed(worker_id)

@task_wrapper
def train(cfg: FinetuneConfig, train_pyd, tuning_pyd, held_out_pyd, wandb_logger: WandbLogger | None = None):
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

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

        logger.info("Checking first batch from DataLoader:")
        for batch in train_dataloader:
            if batch is not None:
                logger.info(f"Batch keys: {batch.keys()}")
                logger.info(f"Batch shapes: {[batch[k].shape for k in batch.keys()]}")
            else:
                logger.info("Batch is None")
            break

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
            checkpoint_callback,
        ]

        if 'patience' in optimization_config and optimization_config['patience'] is not None:
            callbacks.append(
                EarlyStopping(monitor="val_loss", min_delta=0.00, mode="min", patience=optimization_config['patience'], verbose=True)
            )

        trainer_kwargs = dict(
            **cfg.trainer_config,
            callbacks=callbacks,
            logger=wandb_logger,
        )

        trainer_kwargs["max_epochs"] = optimization_config['max_epochs']

        if (optimization_config.get('gradient_accumulation') is not None) and (
            optimization_config['gradient_accumulation'] > 1
        ):
            trainer_kwargs["accumulate_grad_batches"] = optimization_config['gradient_accumulation']

        logger.info("Setting up trainer")
        if cfg.trainer_config.get("strategy") == "ddp_find_unused_parameters_true":
            from lightning.pytorch.strategies import DDPStrategy
            strategy = DDPStrategy(find_unused_parameters=True)
            cfg.trainer_config["strategy"] = strategy

        trainer = L.Trainer(**cfg.trainer_config, callbacks=callbacks, logger=wandb_logger)

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