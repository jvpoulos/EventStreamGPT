#!/usr/bin/env python
"""Launches a [weights and biases](https://wandb.ai/) hyperparameter tuning sweep."""

try:
    # This color-codes and prettifies error messages if the script fails.
    import stackprinter

    stackprinter.set_excepthook(style="darkbg2")
except ImportError:
    pass  # no need to fail because of missing dev dependency

from typing import Any
import sys
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
import json
import base64

# This is a (non-exhaustive) set of weights and biases sweep parameter keywords, which is used to indicate
# when a configuration dictionary contains actual parameter choices, rather than further nested parameter
# groups.
WANDB_SWEEP_KEYS: set[str] = {"value", "values", "min", "max", "distribution"}


def collapse_cfg(k: str, v: dict[str, Any]) -> dict[str, Any]:
    """Collapses a nested config into the hydra parameter override syntax.

    The weights and biases sweep configuration system leverages nested parameter groups, but they are
    represented via a different syntax than that which Hydra uses for overrides (dot separated). This function
    converts the former to the latter in the sweep config so that program runs work down the line. The
    dictionary `v` is collapsed to leaves, where leaf is defined by when the dictionary `v` contains any
    sentinel `WANDB_SWEEP_KEYS` keys. If the dictionary `v` contains just the value `None` (`{'value': None}`)
    then an empty dictionary is returned to remove that parameter from the configuration.

    Args:
        k: The string name of the containing parameter group.
        v: The dictionary value of the nested sub-parameter group to be collapsed.

    Returns:
        A single dictionary with nested key strings and all leaf values represented.

    Raises:
        TypeError: if `v` is not a dictionary.

    Examples:
        >>> collapse_cfg("foo", None)
        Traceback (most recent call last):
            ...
        TypeError: Misconfigured @ foo: None (<class 'NoneType'>) is not a dict!
        >>> collapse_cfg("bar", {"values": "vals"})
        {'bar': {'values': 'vals'}}
        >>> collapse_cfg("foo", {"bar": {"baz": {"values": "vals"}}, "biz": {"max": "MX"}})
        {'foo.bar.baz': {'values': 'vals'}, 'foo.biz': {'max': 'MX'}}
        >>> collapse_cfg("foo", {"bar": {"value": None}})
        {}
    """
    if type(v) is not dict:
        raise TypeError(f"Misconfigured @ {k}: {v} ({type(v)}) is not a dict!")
    if len(WANDB_SWEEP_KEYS.intersection(v.keys())) > 0:
        if set(v.keys()) == {"value"} and v["value"] is None:
            return {}
        else:
            return {k: v}

    out = {}
    if k:
        for kk, vv in v.items():
            out.update(collapse_cfg(f"{k}.{kk}", vv))
    else:
        for kk, vv in v.items():
            out.update(collapse_cfg(kk, vv))
    return out

def json_serializable_config(cfg):
    if isinstance(cfg, dict):
        return {k: json_serializable_config(v) for k, v in cfg.items()}
    elif isinstance(cfg, list):
        return [json_serializable_config(v) for v in cfg]
    elif callable(cfg) or isinstance(cfg, str) and cfg.startswith('lambda'):
        return str(cfg)
    else:
        return cfg

def calculate_end_lr_frac(trial):
    init_lr = trial.suggest_float("optimization_config.init_lr", 1e-5, 1e-2, log=True)
    end_lr = trial.suggest_float("optimization_config.end_lr", 1e-7, 1e-4, log=True)
    return end_lr / init_lr

def calculate_max_seq_len(trial):
    seq_window_size = trial.suggest_categorical("config.seq_window_size", [168, 336, 504])
    return max(seq_window_size * 2, 512)

def calculate_min_seq_len(trial):
    seq_window_size = trial.suggest_categorical("config.seq_window_size", [168, 336, 504])
    return min(seq_window_size // 2, 16)

def calculate_hidden_size(trial):
    head_dim = trial.suggest_categorical("config.head_dim", [16, 32, 64])
    num_attention_heads = trial.suggest_categorical("config.num_attention_heads", [4, 8, 12])
    return head_dim * num_attention_heads

@hydra.main(version_base=None, config_path="../configs", config_name="finetuning_hyperparameter_sweep_base")
def main(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    
    # Update configurations
    if 'config' in cfg['parameters']:
        # Remove the hidden_size parameter
        if 'hidden_size' in cfg['parameters']['config']:
            del cfg['parameters']['config']['hidden_size']
    
    if 'optimization_config' in cfg['parameters']:
        cfg['parameters']['optimization_config']['batch_size'] = {
            'values': [256, 512, 1024]
        }
        cfg['parameters']['optimization_config']['end_lr'] = {
            'distribution': 'log_uniform_values',
            'min': 1e-7,
            'max': 1e-4
        }
        cfg['parameters']['optimization_config']['lr_frac_warmup_steps'] = {
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 0.5
        }
        cfg['parameters']['optimization_config']['init_lr'] = {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-2
        }
        cfg['parameters']['optimization_config']['weight_decay'] = {
            'values': [0.0, 0.01, 0.03]
        }

    # Define dropout parameters
    dropout_params = ['intermediate_dropout', 'attention_dropout', 'input_dropout', 'resid_dropout']
    for param in dropout_params:
        cfg['parameters']['config'][param] = {
            'values': [0.1, 0.3, 0.5]
        }

    new_params = {}
    for k, v in cfg["parameters"].items():
        new_params.update(collapse_cfg(k, v))

    # After creating new_params
    problematic_params = ['config.hidden_size', 'data_config.max_seq_len', 'data_config.min_seq_len', 'optimization_config.end_lr_frac_of_init_lr', 'optimization_config.validation_batch_size']

    for param in problematic_params:
        if param in new_params:
            del new_params[param]

    # Add these parameters back with modified values
    new_params['config.hidden_size'] = {
        'value': "config.head_dim * config.num_attention_heads"
    }
    new_params['data_config.max_seq_len'] = {
        'value': "max(config.seq_window_size * 2, 512)"
    }
    new_params['data_config.min_seq_len'] = {
        'value': "min(config.seq_window_size // 2, 16)"
    }
    new_params['optimization_config.end_lr_frac_of_init_lr'] = {
        'value': "optimization_config.end_lr / optimization_config.init_lr"
    }
    new_params['optimization_config.validation_batch_size'] = {
        'value': "optimization_config.batch_size"
    }

    # Create the final sweep configuration
    sweep_config = {
        "method": cfg.get("method", "bayes"),
        "metric": cfg.get("metric", {"name": "val_auc_epoch", "goal": "maximize"}),
        "parameters": new_params,
        "name": cfg.get("name", "EST_FT_sweep"),
    }

    # After creating the sweep_config
    encoded_config = base64.b64encode(json.dumps(sweep_config).encode()).decode()

    # Write the encoded config to a file
    with open('encoded_config.json', 'w') as f:
        f.write(encoded_config)

    # Construct the command to run finetune.py
    sweep_config["command"] = [
        sys.executable,  # This will be the path to the Python interpreter
        "/home/jvp/diabetes_pred/src/finetune.py",  # Full path to finetune.py
        "sweep=true",  # Use Hydra's override syntax
    ]
    
    # Start the sweep
    sweep_id = wandb.sweep(sweep_config, project="your_project_name")
    wandb.agent(sweep_id, function=lambda: subprocess.call(sweep_config["command"]))

    sweep_kwargs = {}
    if "entity" in cfg:
        sweep_kwargs["entity"] = cfg["entity"]
    else:
        sweep_kwargs["entity"] = "jvpoulos"  # your W&B username

    if "project" in cfg:
        sweep_kwargs["project"] = cfg["project"]
    else:
        sweep_kwargs["project"] = "diabetes_sweep"  # default project name

    print("Sweep configuration:")
    print(json.dumps(json_serializable_config(sweep_config), indent=2))
    print("Sweep kwargs:")
    print(json.dumps(sweep_kwargs, indent=2))

    sweep_id = wandb.sweep(sweep=sweep_config, **sweep_kwargs)
    return sweep_id

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
