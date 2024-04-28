"""Various configuration classes for EventStream data objects."""

from __future__ import annotations

import dataclasses
import enum
import random
from collections import OrderedDict, defaultdict
from collections.abc import Hashable, Sequence
from io import StringIO, TextIOBase
from pathlib import Path
from textwrap import shorten, wrap
from typing import Any, Union

import omegaconf
import pandas as pd
import polars as pl

from ..utils import (
    COUNT_OR_PROPORTION,
    PROPORTION,
    JSONableMixin,
    StrEnum,
    hydra_dataclass,
    num_initial_spaces,
)
from .time_dependent_functor import AgeFunctor, TimeDependentFunctor, TimeOfDayFunctor
from .types import DataModality, InputDataType, InputDFType, TemporalityType
from .vocabulary import Vocabulary, VocabularyConfig

import dataclasses
from omegaconf import MISSING

from EventStream.utils import hydra_dataclass
from .dataset_config import DatasetConfig
from .input_df_schema import InputDFSchema
from .measurement_config import MeasurementConfig
from .custom_resolvers import load_dataset

# Represents the type for a column name in a dataframe.
DF_COL = Union[str, Sequence[str]]

# Represents the type of an input column during pre-processing.
INPUT_COL_T = Union[InputDataType, tuple[InputDataType, str]]

# A unified type for a schema of an input dataframe.
DF_SCHEMA = Union[
    # For cases where you specify a list of columns of a constant type.
    tuple[list[DF_COL], INPUT_COL_T],
    # For specifying a single column and type.
    tuple[DF_COL, INPUT_COL_T],
    # For specifying a dictionary of columns to types.
    dict[DF_COL, INPUT_COL_T],
    # For specifying a dictionary of column in names to column out names and types.
    dict[DF_COL, tuple[str, INPUT_COL_T]],
    # For specifying a dictionary of column in names to out names, all of a constant type.
    tuple[dict[DF_COL, str], INPUT_COL_T],
]

class SeqPaddingSide(StrEnum):
    """Enumeration for the side of sequence padding during PyTorch Batch construction."""

    RIGHT = enum.auto()
    """Pad on the right side (at the end of the sequence).

    This is the default during normal training.
    """

    LEFT = enum.auto()
    """Pad on the left side (at the beginning of the sequence).

    This is the default during generation.
    """


class SubsequenceSamplingStrategy(StrEnum):
    """Enumeration for subsequence sampling strategies.

    When the maximum allowed sequence length for a PyTorchDataset is shorter than the sequence length of a
    subject's data, this enumeration dictates how we sample a subsequence to include.
    """

    TO_END = enum.auto()
    """Sample subsequences of the maximum length up to the end of the permitted window.

    This is the default during fine-tuning and with task dataframes.
    """

    FROM_START = enum.auto()
    """Sample subsequences of the maximum length from the start of the permitted window."""

    RANDOM = enum.auto()
    """Sample subsequences of the maximum length randomly within the permitted window.

    This is the default during pre-training.
    """

@hydra_dataclass
class DataConfig:
    pass


@hydra_dataclass
class PytorchDatasetConfig(DataConfig):
    """Configuration options for building a PyTorch dataset from a `Dataset`.

    This is the main configuration object for a `PytorchDataset`. The `PytorchDataset` class specializes the
    representation of the data in a base `Dataset` class for sequential deep learning. This dataclass is also
    an acceptable `Hydra Structured Config`_ object with the name "pytorch_dataset_config".

    .. _Hydra Structured Config: https://hydra.cc/docs/tutorials/structured_config/intro/

    Attributes:
        save_dir: Directory where the base dataset, including the deep learning representation outputs, is
            saved.
        max_seq_len: Maximum sequence length the dataset should output in any individual item.
        min_seq_len: Minimum sequence length required to include a subject in the dataset.
        seq_padding_side: Whether to pad smaller sequences on the right or the left.
        subsequence_sampling_strategy: Strategy for sampling subsequences when an individual item's total
            sequence length in the raw data exceeds the maximum allowed sequence length.
        train_subset_size: If the training data should be subsampled randomly, this specifies the size of the
            training subset. If `None` or "FULL", then the full training data is used.
        train_subset_seed: If the training data should be subsampled randomly, this specifies the seed for
            that random subsampling.
        task_df_name: If the raw dataset should be limited to a task dataframe view, this specifies the name
            of the task dataframe, and indirectly the path on disk from where that task dataframe will be
            read (save_dir / "task_dfs" / f"{task_df_name}.parquet").
        do_include_subject_id: Whether or not to include the subject ID of the individual for this batch.
        do_include_subsequence_indices: Whether or not to include the start and end indices of the sampled
            subsequence for the individual from their full dataset for this batch. This is sometimes used
            during generative-based evaluation.
        do_include_start_time_min: Whether or not to include the start time of the individual's sequence in
            minutes since the epoch (1/1/1970) in the output data. This is necessary during generation, and
            not used anywhere else currently.

    Raises:
        ValueError: If 'seq_padding_side' is not a valid value; If 'min_seq_len' is not a non-negative
            integer; If 'max_seq_len' is not an integer greater or equal to 'min_seq_len'; If
            'train_subset_seed' is not None when 'train_subset_size' is None or 'FULL'; If 'train_subset_size'
            is negative when it's an integer; If 'train_subset_size' is not within (0, 1) when it's a float.
        TypeError: If 'train_subset_size' is of unrecognized type.

    Examples:
        >>> config = PytorchDatasetConfig(
        ...     save_dir='./dataset',
        ...     max_seq_len=256,
        ...     min_seq_len=2,
        ...     seq_padding_side=SeqPaddingSide.RIGHT,
        ...     subsequence_sampling_strategy=SubsequenceSamplingStrategy.RANDOM,
        ...     train_subset_size="FULL",
        ...     train_subset_seed=None,
        ...     task_df_name=None,
        ...     do_include_start_time_min=False
        ... )
        >>> config_dict = config.to_dict()
        >>> new_config = PytorchDatasetConfig.from_dict(config_dict)
        >>> config == new_config
        True
        >>> config = PytorchDatasetConfig(train_subset_size=-1)
        Traceback (most recent call last):
            ...
        ValueError: If integral, train_subset_size must be positive! Got -1
        >>> config = PytorchDatasetConfig(train_subset_size=1.2)
        Traceback (most recent call last):
            ...
        ValueError: If float, train_subset_size must be in (0, 1)! Got 1.2
        >>> config = PytorchDatasetConfig(train_subset_size='200')
        Traceback (most recent call last):
            ...
        TypeError: train_subset_size is of unrecognized type <class 'str'>.
        >>> config = PytorchDatasetConfig(
        ...     save_dir='./dataset',
        ...     max_seq_len=256,
        ...     min_seq_len=2,
        ...     seq_padding_side='left',
        ...     subsequence_sampling_strategy=SubsequenceSamplingStrategy.RANDOM,
        ...     train_subset_size=100,
        ...     train_subset_seed=None,
        ...     task_df_name=None,
        ...     do_include_start_time_min=False
        ... )
        WARNING! train_subset_size is set, but train_subset_seed is not. Setting to...
        >>> assert config.train_subset_seed is not None
    """

    save_dir: Path = MISSING
    dataset = MISSING 
    dataset_path: Path = MISSING

    max_seq_len: int = 256
    min_seq_len: int = 2
    seq_padding_side: SeqPaddingSide = SeqPaddingSide.RIGHT
    subsequence_sampling_strategy: SubsequenceSamplingStrategy = SubsequenceSamplingStrategy.RANDOM

    train_subset_size: int | float | str = "FULL"
    train_subset_seed: int | None = None

    task_df_name: str | None = None

    do_include_subsequence_indices: bool = False
    do_include_subject_id: bool = False
    do_include_start_time_min: bool = False

    def __post_init__(self):
        if self.seq_padding_side not in SeqPaddingSide.values():
            raise ValueError(f"seq_padding_side invalid; must be in {', '.join(SeqPaddingSide.values())}")
        if type(self.min_seq_len) is not int or self.min_seq_len < 0:
            raise ValueError(f"min_seq_len must be a non-negative integer; got {self.min_seq_len}")
        if type(self.max_seq_len) is not int or self.max_seq_len < self.min_seq_len:
            raise ValueError(
                f"max_seq_len must be an integer at least equal to min_seq_len; got {self.max_seq_len} "
                f"(min_seq_len = {self.min_seq_len})"
            )

        if type(self.save_dir) is str and self.save_dir != omegaconf.MISSING:
            self.save_dir = Path(self.save_dir)

        if self.dataset_path != omegaconf.MISSING:
            self.dataset = OmegaConf.structured(OmegaConf.load_dataset(self.dataset_path))
            
        match self.train_subset_size:
            case int() as n if n < 0:
                raise ValueError(f"If integral, train_subset_size must be positive! Got {n}")
            case float() as frac if frac <= 0 or frac >= 1:
                raise ValueError(f"If float, train_subset_size must be in (0, 1)! Got {frac}")
            case int() | float() if (self.train_subset_seed is None):
                seed = int(random.randint(1, int(1e6)))
                print(f"WARNING! train_subset_size is set, but train_subset_seed is not. Setting to {seed}")
                self.train_subset_seed = seed
            case None | "FULL" | int() | float():
                pass
            case _:
                raise TypeError(f"train_subset_size is of unrecognized type {type(self.train_subset_size)}.")

    def to_dict(self) -> dict:
        """Represents this configuration object as a plain dictionary."""
        as_dict = dataclasses.asdict(self)
        as_dict["save_dir"] = str(as_dict["save_dir"])
        return as_dict

    @classmethod
    def from_dict(cls, as_dict: dict) -> PytorchDatasetConfig:
        """Creates a new instance of this class from a plain dictionary."""
        as_dict["save_dir"] = Path(as_dict["save_dir"])
        return cls(**as_dict)