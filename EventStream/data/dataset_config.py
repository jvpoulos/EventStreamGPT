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
from .vocabulary import Vocabulary

import dataclasses
from omegaconf import MISSING

from EventStream.utils import hydra_dataclass
from .dataset_schema import DatasetSchema
from .input_df_schema import InputDFSchema
from .measurement_config import MeasurementConfig
from typing import TYPE_CHECKING
    
@dataclasses.dataclass
class DatasetConfig(JSONableMixin):
    """Configuration options for a Dataset class.

    This is the core configuration object for Dataset objects. Contains configuration options for
    pre-processing a dataset already in the "Subject-Events-Measurements" data model or interpreting an
    existing dataset. This configures details such as

    1. Which measurements should be extracted and included in the raw dataset, via the `measurement_configs`
       arg.
    2. What filtering parameters should be applied to eliminate infrequently observed variables or columns.
    3. How/whether or not numerical values should be re-cast as categorical or integral types.
    4. Configuration options for outlier detector or normalization models.
    5. Time aggregation controls.
    6. The output save directory.

    These configuration options do not include options to extract the raw dataset from source. For options for
    raw dataset extraction, see `DatasetSchema` and `InputDFSchema`, and for options for the raw script
    builder, see `configs/dataset_base.yml`.

    Attributes:
        measurement_configs: The dataset configuration for this `Dataset`. Keys are measurement names, and
            values are `MeasurementConfig` objects detailing configuration parameters for that measure.
            Measurement names / dictionary keys are also used as source columns for the data of that measure,
            though in the case of `DataModality.MULTIVARIATE_REGRESSION` measures, this name will reference
            the categorical regression target index column and the config will also contain a reference to a
            values column name which points to the column containing the associated numerical values.
            Columns not referenced in any configs are not pre-processed. Measurement configs are checked for
            validity upon creation. Dictionary keys must match measurement config object names if such are
            specified; if measurement config object names are not specified, they will be set to their
            associated dictionary keys.

        min_valid_column_observations: The minimum number of column observations or proportion of possible
            events that contain a column that must be observed for the column to be included in the training
            set. If fewer than this many observations are observed, the entire column will be dropped. Can be
            either an integer count or a proportion (of total vocabulary size) in (0, 1). If `None`, no
            constraint is applied.

        min_valid_vocab_element_observations: The minimum number or proportion of observations of a particular
            metadata vocabulary element that must be observed for the element to be included in the training
            set vocabulary. If fewer than this many observations are observed, observed elements will be
            dropped. Can be either an integer count or a proportion (of total vocabulary size) in (0, 1). If
            `None`, no constraint is applied.

        min_true_float_frequency: The minimum proportion of true float values that must be observed in order
            for observations to be treated as true floating point numbers, not integers.

        min_unique_numerical_observations: The minimum number of unique values a numerical column must have in
            the training set to be treated as a numerical type (rather than an implied categorical or ordinal
            type). Numerical entries with fewer than this many observations will be converted to categorical
            or ordinal types. Can be either an integer count or a proportion (of total numerical observations)
            in (0, 1). If `None`, no constraint is applied.

        outlier_detector_config: Configuration options for outlier detection. If not `None`, must contain the
            key `'cls'`, which points to the class used outlier detection. All other keys and values are
            keyword arguments to be passed to the specified class. The API of these objects is expected to
            mirror scikit-learn outlier detection model APIs. If `None`, numerical outlier values are not
            removed.

        normalizer_config: Configuration options for normalization. If not `None`, must contain the key
            `'cls'`, which points to the class used normalization. All other keys and values are keyword
            arguments to be passed to the specified class. The API of these objects is expected to mirror
            scikit-learn normalization system APIs. If `None`, numerical values are not normalized.

        save_dir: The output save directory for this dataset. Will be converted to a `pathlib.Path` upon
            creation if it is not already one.

        agg_by_time_scale: Aggregate events into temporal buckets at this frequency. Uses the string language
            described here:
            https://pola-rs.github.io/polars/py-polars/html/reference/dataframe/api/polars.DataFrame.groupby_dynamic.html

    Raises:
        ValueError: If configuration parameters are invalid (e.g., proportion parameters being > 1, etc.).
        TypeError: If configuration parameters are of invalid types.

    Examples:
        >>> cfg = DatasetConfig(
        ...     measurement_configs={
        ...         "meas1": MeasurementConfig(
        ...             temporality=TemporalityType.DYNAMIC,
        ...             modality=DataModality.MULTI_LABEL_CLASSIFICATION,
        ...         ),
        ...     },
        ...     min_valid_column_observations=0.5,
        ...     save_dir="/path/to/save/dir",
        ... )
        >>> cfg.save_dir
        PosixPath('/path/to/save/dir')
        >>> cfg.to_dict() # doctest: +NORMALIZE_WHITESPACE
        {'measurement_configs':
            {'meas1':
                {'name': 'meas1',
                 'temporality': <TemporalityType.DYNAMIC: 'dynamic'>,
                 'modality': <DataModality.MULTI_LABEL_CLASSIFICATION: 'multi_label_classification'>,
                 'observation_rate_over_cases': None,
                 'observation_rate_per_case': None,
                 'functor': None,
                 'vocabulary': None,
                 'values_column': None,
                 '_measurement_metadata': None,
                 'modifiers': None}},
            'min_events_per_subject': None,
            'agg_by_time_scale': '1h',
            'min_valid_column_observations': 0.5,
            'min_valid_vocab_element_observations': None,
            'min_true_float_frequency': None,
            'min_unique_numerical_observations': None,
            'outlier_detector_config': None,
            'normalizer_config': None,
            'save_dir': '/path/to/save/dir'}
        >>> cfg2 = DatasetConfig.from_dict(cfg.to_dict())
        >>> assert cfg == cfg2
        >>> DatasetConfig(
        ...     measurement_configs={
        ...         "meas1": MeasurementConfig(
        ...             name="invalid_name",
        ...             temporality=TemporalityType.DYNAMIC,
        ...             modality=DataModality.MULTI_LABEL_CLASSIFICATION,
        ...         ),
        ...     },
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Measurement config meas1 has name invalid_name which differs from dict key!
        >>> DatasetConfig(
        ...     min_valid_column_observations="invalid type"
        ... )
        Traceback (most recent call last):
            ...
        TypeError: min_valid_column_observations must either be a fraction (float between 0 and 1) or count\
 (int > 1). Got <class 'str'> of invalid type
        >>> measurement_configs = {
        ...     "meas1": MeasurementConfig(
        ...         temporality=TemporalityType.DYNAMIC,
        ...         modality=DataModality.MULTI_LABEL_CLASSIFICATION,
        ...     ),
        ... }
        >>> # Make one of the measurements invalid to show that validitiy is re-checked...
        >>> measurement_configs["meas1"].temporality = None
        >>> DatasetConfig(
        ...     measurement_configs=measurement_configs,
        ...     min_valid_column_observations=0.5,
        ...     save_dir="/path/to/save/dir",
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Measurement config meas1 invalid!
    """
    
    measurement_configs: dict[str, MeasurementConfig] = dataclasses.field(default_factory=lambda: {})

    min_events_per_subject: int | None = None

    agg_by_time_scale: str | None = "1h"

    min_valid_column_observations: COUNT_OR_PROPORTION | None = None
    min_valid_vocab_element_observations: COUNT_OR_PROPORTION | None = None
    min_true_float_frequency: PROPORTION | None = None
    min_unique_numerical_observations: COUNT_OR_PROPORTION | None = None

    outlier_detector_config: dict[str, Any] | None = None
    normalizer_config: dict[str, Any] | None = None

    save_dir: Path | None = None

    def __post_init__(self):
        """Validates that parameters take on valid values."""
        for name, cfg in self.measurement_configs.items():
            if cfg.name is None:
                cfg.name = name
            elif cfg.name != name:
                raise ValueError(
                    f"Measurement config {name} has name {cfg.name} which differs from dict key!"
                )

        for var in (
            "min_valid_column_observations",
            "min_valid_vocab_element_observations",
            "min_unique_numerical_observations",
        ):
            val = getattr(self, var)
            match val:
                case None:
                    pass
                case float() if (0 < val) and (val < 1):
                    pass
                case int() if val > 1:
                    pass
                case float():
                    raise ValueError(f"{var} must be in (0, 1) if float; got {val}!")
                case int():
                    raise ValueError(f"{var} must be > 1 if integral; got {val}!")
                case _:
                    raise TypeError(
                        f"{var} must either be a fraction (float between 0 and 1) or count (int > 1). Got "
                        f"{type(val)} of {val}"
                    )

        for var in ("min_true_float_frequency",):
            val = getattr(self, var)
            match val:
                case None:
                    pass
                case float() if (0 < val) and (val < 1):
                    pass
                case float():
                    raise ValueError(f"{var} must be in (0, 1) if float; got {val}!")
                case _:
                    raise TypeError(
                        f"{var} must be a fraction (float between 0 and 1). Got {type(val)} of {val}"
                    )

        for var in ("outlier_detector_config", "normalizer_config"):
            val = getattr(self, var)
            if val is not None and (type(val) is not dict or "cls" not in val):
                raise ValueError(f"{var} must be either None or a dictionary with 'cls' as a key! Got {val}")

        for k, v in self.measurement_configs.items():
            try:
                v._validate()
            except Exception as e:
                raise ValueError(f"Measurement config {k} invalid!") from e

        if type(self.save_dir) is str:
            self.save_dir = Path(self.save_dir)

    def to_dict(self) -> dict:
        """Represents this configuration object as a plain dictionary.

        Returns:
            A plain dictionary representation of self (nested through measurement configs as well).
        """
        as_dict = dataclasses.asdict(self)
        print("Serializing DatasetConfig attributes:")
        for key, value in as_dict.items():
            print(f"Key: {key}, Type: {type(value)}")
        if self.save_dir is not None:
            as_dict["save_dir"] = str(self.save_dir.absolute())
        as_dict["measurement_configs"] = {k: v.to_dict() for k, v in self.measurement_configs.items()}
        return as_dict

    @classmethod
    def from_dict(cls, as_dict: dict) -> DatasetConfig:
        """Build a configuration object from a plain dictionary representation.

        Args:
            as_dict: The plain dictionary representation to be converted.

        Returns: A DatasetConfig instance containing the same data as `as_dict`.
        """
        as_dict["measurement_configs"] = {
            k: MeasurementConfig.from_dict(v) for k, v in as_dict["measurement_configs"].items()
        }
        if type(as_dict["save_dir"]) is str:
            as_dict["save_dir"] = Path(as_dict["save_dir"])
            
        return cls(**as_dict)

    def __eq__(self, other: DatasetConfig) -> bool:
        """Returns true if self and other are equal."""
        return self.to_dict() == other.to_dict()