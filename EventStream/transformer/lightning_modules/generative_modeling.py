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
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy, BinaryAveragePrecision
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
    ):
        super().__init__()

        # Define the CLASSIFICATION attribute
        self.CLASSIFICATION = {DataModality.SINGLE_LABEL_CLASSIFICATION, DataModality.MULTI_LABEL_CLASSIFICATION}   

        # If the configurations are dictionaries, convert them to class objects.
        if isinstance(config, dict):
            config = StructuredTransformerConfig(**config)
            # Set the problem_type if it's None
            if config.problem_type is None:
                config.problem_type = "single_label_classification"
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

        # Initialize the AUROC metric for the binary classification task
        self.binary_auroc = BinaryAUROC()

        # Initialize self.metrics and self.binary_metrics
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
        self.metrics = {
            "A1cGreaterThan7": torch.nn.ModuleDict({
                "AUROC": BinaryAUROC(),
            })
        }
    
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

    def log_metrics(self, results: GenerativeSequenceModelOutput, split: Split):
        """Logs metric results for a given output result."""
        log_kwargs = {"batch_size": self.optimization_config.batch_size, "sync_dist": True}

        # Log the loss
        self.log(f"{split}_loss", results["loss"], **log_kwargs, rank_zero_only=True, prog_bar=(split == Split.TRAIN))

        if split == Split.TRAIN or split == Split.TUNING:
            metrics_config = self.pretraining_metrics_config
        else:
            metrics_config = self.final_validation_metrics_config

        if MetricCategories.LOSS_PARTS.value in metrics_config.include_metrics.get(split.value, {}):
            # Log other losses
            for loss_name, loss_value in results["losses"].items():
                self.log(f"{split}_{loss_name}", loss_value, **log_kwargs)

        # Log A1cGreaterThan7 classification metrics
        if "classification" in results["preds"] and "A1cGreaterThan7" in results["preds"]["classification"]:
            _, sample_dist = results["preds"]["classification"]["A1cGreaterThan7"]
            if sample_dist is not None:
                preds = sample_dist.logits
                labels = results["labels"]["classification"].get("A1cGreaterThan7", None)

                if labels is not None:
                    self._log_metric_dict(
                        preds=preds,
                        labels=labels,
                        metrics=self.metrics["A1cGreaterThan7"],
                        measurement="A1cGreaterThan7",
                        split=split,
                        cat=MetricCategories.CLASSIFICATION,
                        metrics_config=metrics_config,
                    )
                else:
                    print("Warning: 'A1cGreaterThan7' labels are None. Skipping metric logging.")
            else:
                print("Warning: 'A1cGreaterThan7' sample distribution is None. Skipping metric logging.")
        else:
            print("Warning: 'A1cGreaterThan7' not found in the model's predictions. Skipping metric logging.")
            
    def calculate_and_log_auroc(self, out, split):
        if "A1cGreaterThan7" in out["preds"]["classification"]:
            _, preds_dist = out["preds"]["classification"]["A1cGreaterThan7"]
            
            if preds_dist is not None:
                preds = preds_dist.logits
                
                if "A1cGreaterThan7" in out["labels"]["classification"] and out["labels"]["classification"]["A1cGreaterThan7"] is not None:
                    labels = out["labels"]["classification"]["A1cGreaterThan7"]
                    self.binary_auroc.update(preds, labels)
                    self.log(f"{split}_A1cGreaterThan7_AUROC", self.binary_auroc, on_step=(split == "train"), on_epoch=True)
                else:
                    print(f"Warning: 'A1cGreaterThan7' labels are missing or None for split {split}. Skipping AUROC calculation.")
            else:
                print(f"Warning: 'A1cGreaterThan7' prediction distribution is None for split {split}. Skipping AUROC calculation.")
        else:
            print(f"Warning: 'A1cGreaterThan7' key not found in predictions for split {split}. Skipping AUROC calculation.")

    def training_step(self, batch: PytorchBatch, batch_idx: int) -> torch.Tensor:
        out = self.model(batch)
        self.log_metrics(out, Split.TRAIN)
        self.calculate_and_log_auroc(out, "train")
        loss = out["loss"]
        return loss

    def validation_step(self, batch: PytorchBatch, batch_idx: int):
        out = self.model(batch)
        self.log_metrics(out, Split.TUNING)
        self.calculate_and_log_auroc(out, "val")
        loss = out["loss"]
        self.log("val_loss", loss, on_step=False, on_epoch=True)

    def test_step(self, batch: PytorchBatch, batch_idx: int):
        out = self.model(batch)
        self.log_metrics(out, Split.HELD_OUT)
        self.calculate_and_log_auroc(out, "test")
        loss = out["loss"]
        self.log("test_loss", loss, on_step=False, on_epoch=True)

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
            "measurements_per_generative_mode": {
                DataModality.SINGLE_LABEL_CLASSIFICATION.value: ['A1cGreaterThan7'],
            },
            "measurements_per_dep_graph_level": [
                [('A1cGreaterThan7', MeasIndexGroupOptions.FLAT)],
            ],
            "seq_attention_types": ["global", "local"],  # Set the seq_attention_types directly
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

    def convert_to_python_object(self, obj):
        if isinstance(obj, dict):
            return {k: self.convert_to_python_object(v) for k, v in obj.items()}
        elif isinstance(obj, (list, OmegaConf.ListConfig)):
            return [self.convert_to_python_object(v) for v in obj]
        elif isinstance(obj, OmegaConf.StringNode):
            return obj.decode()
        else:
            return obj

    def __post_init__(self):
        self.seq_attention_types = self.convert_to_python_object(self.seq_attention_types)
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