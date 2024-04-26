from pathlib import Path
from typing import Any

from EventStream.data.dataset_polars import Dataset
from omegaconf import OmegaConf

def load_dataset(path: Path) -> Dataset:
    """Custom OmegaConf resolver for loading a Dataset object from a file path."""
    return Dataset.load(path)

# Register the custom resolver with OmegaConf
OmegaConf.register_new_resolver("load_dataset", load_dataset)