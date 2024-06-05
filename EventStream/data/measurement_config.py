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
@dataclasses.dataclass
class MeasurementConfig(JSONableMixin):
    @staticmethod
    def get_smallest_valid_uint_type(num: int | float | pl.Expr) -> pl.DataType:
        from .dataset_utils import get_dataset
        Dataset = get_dataset()
    """The Configuration class for a measurement in the Dataset.

    A measurement is any observation in the dataset; be it static or dynamic, categorical or continuous. This
    class contains configuration options to define a measurement and dictate how it should be pre-processed,
    embedded, and generated in generative models.

    Attributes:
        name:
            Stores the name of this measurement; also the column in the appropriate internal dataframe
            (`subjects_df`, `events_df`, or `dynamic_measurements_df`) that will contain this measurement. All
            measurements will have this set.

            The 'column' linkage has slightly different meanings depending on `self.modality`:

            * If `modality == DataModality.UNIVARIATE_REGRESSION`, then this column stores the values
              associated with this continuous-valued measure.
            * If `modality == DataModality.MULTIVARIATE_REGRESSION`, then this column stores the keys that
              dictate the dimensions for which the associated `values_column` has the values.
            * Otherwise, this column stores the categorical values of this measure.

            Similarly, it has slightly different meanings depending on `self.temporality`:

            * If `temporality == TemporalityType.STATIC`, this is an existent column in the `subjects_df`
              dataframe.
            * If `temporality == TemporalityType.DYNAMIC`, this is an existent column in the
              `dynamic_measurements_df` dataframe.
            * Otherwise, (when `temporality == TemporalityType.FUNCTIONAL_TIME_DEPENDENT`), then this is
              the name the *output-to-be-created* column will take in the `events_df` dataframe.

        modality: The modality of this measurement. If `DataModality.UNIVARIATE_REGRESSION`, then this
            measurement takes on single-variate continuous values. If `DataModality.MULTIVARIATE_REGRESSION`,
            then this measurement consists of key-value pairs of categorical covariate identifiers and
            continuous values. Keys are stored in the column reflected in `self.name` and values in
            `self.values_column`.
        temporality: How this measure varies in time. If `TemporalityType.STATIC`, this is a static
            measurement. If `TemporalityType.FUNCTIONAL_TIME_DEPENDENT`, then this measurement is a
            time-dependent measure that varies with time and static data in an analytically computable manner
            (e.g., age). If `TemporalityType.DYNAMIC`, then this is a measurement that varies in time in a
            non-a-priori computable manner.
        observation_rate_over_cases: The fraction of valid "instances" in which this measure is observed at
            all. For example, for a static measurement, this is the fraction of subjects for which this
            measure is observed to take on a non-null value at least once. For a dynamic measurement, this is
            the fraction of events for which this measure is observed to take on a non-null value at least
            once. This is set dynamically during pre-procesisng, and not specified at construction.
        observation_rate_per_case: The number of times this measure is observed to take on a non-null value
            per possible valid "instance" where at least one measure is observed. For example, for a static
            measurement, this is the number of times this measure is observed per subject when
            this measure is observed at all. For a dynamic measurement, this is the number of times this
            measure is observed per event when this measure is observed at all. This is set dynamically during
            pre-procesisng, and not specified at construction.
        functor: If `temporality == TemporalityType.FUNCTIONAL_TIME_DEPENDENT`, then this will be set to the
            functor used to compute the value of a known-time-depedency measure. In this case, `functor` must
            be a subclass of `data.time_dependent_functor.TimeDependentFunctor`. If `temporality` is anything
            else, then this will be `None`.
        vocabulary: The vocabulary for this column, realized as a `Vocabulary` object. Begins with `'UNK'`.
            Not set on `modality==UNIVARIATE_REGRESSION` measurements.
        values_column: For `modality==MULTIVARIATE_REGRESSION` measurements, this will store the name of the
            column which will contain the numerical values corresponding to this measurement. Otherwise will
            be `None`.
        measurement_metadata: Stores metadata about the numerical values corresponding to this measurement.
            This can take one of two forms, depending on the measurement modality. If
            `modality==UNIVARIATE_REGRESSION`, then this will be a `pd.Series` whose index will contain the
            set of possible column headers listed below. If `modality==MULTIVARIATE_REGRESSION`, then this
            will be a `pd.DataFrame`, whose index will contain the possible regression covariate identifier
            keys and whose columns will contain the set of possible columns listed below.

            Metadata Columns:

            * drop_lower_bound: A lower bound such that values either below or at or below this level will
              be dropped (key presence will be retained for multivariate regression measures). Optional.
            * drop_lower_bound_inclusive: This must be set if `drop_lower_bound` is set. If this is true,
              then values will be dropped if they are $<=$ `drop_lower_bound`. If it is false, then values
              will be dropped if they are $<$ `drop_lower_bound`.
            * censor_lower_bound: A lower bound such that values either below or at or below this level,
              but above the level of `drop_lower_bound`, will be replaced with the value
              `censor_lower_bound`. Optional.
            * drop_upper_bound An upper bound such that values either above or at or above this level will
              be dropped (key presence will be retained for multivariate regression measures). Optional.
            * drop_upper_bound_inclusive: This must be set if `drop_upper_bound` is set. If this is true,
              then values will be dropped if they are $>=$ `drop_upper_bound`. If it is false, then values
              will be dropped if they are $>$ `drop_upper_bound`.
            * censor_upper_bound: An upper bound such that values either above or at or above this level,
              but below the level of `drop_upper_bound`, will be replaced with the value
              `censor_upper_bound`. Optional.
            * value_type: To which kind of value (e.g., integer, categorical, float) this key corresponds.
              Must be an element of the enum `NumericMetadataValueType`. Optional. If not pre-specified,
              will be inferred from the data.
            * outlier_model: The parameters (in dictionary form) for the fit outlier model. Optional. If
              not pre-specified, will be inferred from the data.
            * normalizer: The parameters (in dictionary form) for the fit normalizer model. Optional. If
              not pre-specified, will be inferred from the data.

        modifiers: Stores a list of additional column names that modify this measurement that should be
            tracked with this measurement record through the dataset.

    Raises:
        ValueError: If the configuration is not self consistent (e.g., a functor specified on a
            non-functional_time_dependent measure).
        NotImplementedError: If the configuration relies on a measurement configuration that is not yet
            supported, such as numeric, static measurements.


    Examples:
        >>> cfg = MeasurementConfig(
        ...     name='key',
        ...     modality='multi_label_classification',
        ...     temporality='dynamic',
        ...     vocabulary=Vocabulary(['foo', 'bar', 'baz'], [0.3, 0.4, 0.3]),
        ... )
        >>> cfg.is_numeric
        False
        >>> cfg.is_dropped
        False
        >>> cfg = MeasurementConfig(
        ...     name='key',
        ...     modality='univariate_regression',
        ...     temporality='dynamic',
        ...     _measurement_metadata=pd.Series([1, 0.2], index=['censor_upper_bound', 'censor_lower_bound']),
        ... )
        >>> cfg.is_numeric
        True
        >>> cfg.is_dropped
        False
        >>> cfg = MeasurementConfig(
        ...     name='key',
        ...     modality='multivariate_regression',
        ...     temporality='dynamic',
        ...     values_column='vals',
        ...     _measurement_metadata=pd.DataFrame(
        ...         {'censor_lower_bound': [1, 0.2, 0.1]},
        ...         index=pd.Index(['foo', 'bar', 'baz'], name='key'),
        ...     ),
        ...     vocabulary=Vocabulary(['foo', 'bar', 'baz'], [0.3, 0.4, 0.3]),
        ... )
        >>> cfg.is_numeric
        True
        >>> cfg.is_dropped
        False
        >>> cfg = MeasurementConfig(
        ...     name='key',
        ...     modality='multi_label_classification',
        ...     temporality='dynamic',
        ...     modifiers=['foo', 'bar'],
        ... )
        >>> cfg = MeasurementConfig(
        ...     name='key',
        ...     modality='multi_label_classification',
        ...     temporality='dynamic',
        ...     modifiers=[1, 2],
        ... )
        Traceback (most recent call last):
            ...
        ValueError: `self.modifiers` must be a list of strings; got element 1.
        >>> MeasurementConfig()
        Traceback (most recent call last):
            ...
        ValueError: `self.temporality = None` Invalid! Must be in static, dynamic, functional_time_dependent
        >>> MeasurementConfig(
        ...     temporality=TemporalityType.FUNCTIONAL_TIME_DEPENDENT,
        ...     functor=None,
        ... )
        Traceback (most recent call last):
            ...
        ValueError: functor must be set for functional_time_dependent measurements!
        >>> MeasurementConfig(
        ...     temporality=TemporalityType.STATIC,
        ...     functor=AgeFunctor(dob_col="date_of_birth"),
        ... )
        Traceback (most recent call last):
            ...
        ValueError: functor should be None for static measurements! Got ...
        >>> MeasurementConfig(
        ...     temporality=TemporalityType.DYNAMIC,
        ...     modality=DataModality.MULTIVARIATE_REGRESSION,
        ...     _measurement_metadata=pd.Series([1, 10], index=['censor_lower_bound', 'censor_upper_bound']),
        ...     values_column='vals',
        ... )
        Traceback (most recent call last):
            ...
        ValueError: If set, measurement_metadata must be a DataFrame on a multivariate_regression\
 MeasurementConfig. Got <class 'pandas.core.series.Series'>
        censor_lower_bound     1
        censor_upper_bound    10
        dtype: int64
    """

    FUNCTORS = {
        "AgeFunctor": AgeFunctor,
        "TimeOfDayFunctor": TimeOfDayFunctor,
    }

    PREPROCESSING_METADATA_COLUMNS = OrderedDict(
        {
            "value_type": str,
            "outlier_model": object,
            "normalizer": object,
        }
    )

    # Present in all measures
    name: str | None = None
    temporality: TemporalityType | None = None
    modality: DataModality | None = None
    observation_rate_over_cases: float | None = None
    observation_rate_per_case: float | None = None

    # Specific to time-dependent measures
    functor: TimeDependentFunctor | None = None

    # Specific to categorical or partially observed multivariate regression measures.
    vocabulary: Vocabulary | None = None

    # Specific to numeric measures
    values_column: str | None = None
    _measurement_metadata: pd.DataFrame | pd.Series | str | Path | None = None

    modifiers: list[str] | None = None

    def __post_init__(self):
        self._validate()

    def _validate(self):
        """Checks the internal state of `self` and ensures internal consistency and validity."""
        match self.temporality:
            case TemporalityType.STATIC:
                if self.functor is not None:
                    raise ValueError(
                        f"functor should be None for {self.temporality} measurements! Got {self.functor}"
                    )

                # if self.is_numeric:
                #     raise NotImplementedError(
                #         f"Numeric data modalities like {self.modality} not yet supported on static measures."
                #     )
            case TemporalityType.DYNAMIC:
                if self.functor is not None:
                    raise ValueError(
                        f"functor should be None for {self.temporality} measurements! Got {self.functor}"
                    )
                if self.modality == DataModality.SINGLE_LABEL_CLASSIFICATION:
                    raise ValueError(
                        f"{self.modality} on {self.temporality} measurements is not currently supported, as "
                        "event aggregation can turn single-label tasks into multi-label tasks in a manner "
                        "that is not currently automatically detected or compensated for."
                    )

            case TemporalityType.FUNCTIONAL_TIME_DEPENDENT:
                if self.functor is None:
                    raise ValueError(f"functor must be set for {self.temporality} measurements!")

                if self.modality is None:
                    self.modality = self.functor.OUTPUT_MODALITY
                elif self.modality not in (DataModality.DROPPED, self.functor.OUTPUT_MODALITY):
                    raise ValueError(
                        "self.modality must either be DataModality.DROPPED or "
                        f"{self.functor.OUTPUT_MODALITY} for {self.temporality} measures; got {self.modality}"
                    )
            case _:
                raise ValueError(
                    f"`self.temporality = {self.temporality}` Invalid! Must be in "
                    f"{', '.join(TemporalityType.values())}"
                )

        err_strings = []
        match self.modality:
            case DataModality.MULTIVARIATE_REGRESSION:
                if self.values_column is None:
                    err_strings.append(f"values_column must be set on a {self.modality} MeasurementConfig")
                if (self.measurement_metadata is not None) and not isinstance(
                    self.measurement_metadata, pd.DataFrame
                ):
                    err_strings.append(
                        f"If set, measurement_metadata must be a DataFrame on a {self.modality} "
                        f"MeasurementConfig. Got {type(self.measurement_metadata)}\n"
                        f"{self.measurement_metadata}"
                    )
            case DataModality.UNIVARIATE_REGRESSION:
                if self.values_column is not None:
                    err_strings.append(
                        f"values_column must be None on a {self.modality} MeasurementConfig. "
                        f"Got {self.values_column}"
                    )
                if (self.measurement_metadata is not None) and not isinstance(
                    self.measurement_metadata, pd.Series
                ):
                    err_strings.append(
                        f"If set, measurement_metadata must be a Series on a {self.modality} "
                        f"MeasurementConfig. Got {type(self.measurement_metadata)}\n"
                        f"{self.measurement_metadata}"
                    )
            case DataModality.SINGLE_LABEL_CLASSIFICATION | DataModality.MULTI_LABEL_CLASSIFICATION:
                if self.values_column is not None:
                    err_strings.append(
                        f"values_column must be None on a {self.modality} MeasurementConfig. "
                        f"Got {self.values_column}"
                    )
                if self._measurement_metadata is not None:
                    err_strings.append(
                        f"measurement_metadata must be None on a {self.modality} MeasurementConfig. "
                        f"Got {type(self.measurement_metadata)}\n{self.measurement_metadata}"
                    )
            case DataModality.DROPPED:
                if self.vocabulary is not None:
                    err_strings.append(
                        f"vocabulary must be None on a {self.modality} MeasurementConfig. "
                        f"Got {self.vocabulary}"
                    )
                if self._measurement_metadata is not None:
                    err_strings.append(
                        f"measurement_metadata must be None on a {self.modality} MeasurementConfig. "
                        f"Got {type(self.measurement_metadata)}\n{self.measurement_metadata}"
                    )
            case _:
                raise ValueError(f"`self.modality = {self.modality}` Invalid!")
        if err_strings:
            raise ValueError("\n".join(err_strings))

        if self.modifiers is not None:
            for mod in self.modifiers:
                if not isinstance(mod, str):
                    raise ValueError(f"`self.modifiers` must be a list of strings; got element {mod}.")

    def drop(self):
        """Sets the modality to DROPPED and does associated post-processing to ensure validity.

        Examples:
            >>> cfg = MeasurementConfig(
            ...     name='key',
            ...     modality='multivariate_regression',
            ...     temporality='dynamic',
            ...     values_column='vals',
            ...     _measurement_metadata=pd.DataFrame(
            ...         {'censor_lower_bound': [1, 0.2, 0.1]},
            ...         index=pd.Index(['foo', 'bar', 'baz'], name='key'),
            ...     ),
            ...     vocabulary=Vocabulary(['foo', 'bar', 'baz'], [0.3, 0.4, 0.3]),
            ... )
            >>> cfg.drop()
            >>> cfg.modality
            <DataModality.DROPPED: 'dropped'>
            >>> assert cfg._measurement_metadata is None
            >>> assert cfg.vocabulary is None
            >>> assert cfg.is_dropped
        """
        self.modality = DataModality.DROPPED
        self._measurement_metadata = None
        self.vocabulary = None

    @property
    def is_dropped(self) -> bool:
        return self.modality == DataModality.DROPPED

    @property
    def is_numeric(self) -> bool:
        return self.modality in (
            DataModality.MULTIVARIATE_REGRESSION,
            DataModality.UNIVARIATE_REGRESSION,
        )

    @property
    def measurement_metadata(self) -> pd.DataFrame | pd.Series | None:
        match self._measurement_metadata:
            case None | pd.DataFrame() | pd.Series():
                return self._measurement_metadata
            case [(Path() | str()) as base_dir, str() as fn]:
                fp = Path(base_dir) / fn
            case (Path() | str()) as fp:
                fp = Path(fp)
            case _:
                raise ValueError(f"_measurement_metadata is invalid! Got {type(self._measurement_metadata)}!")

        out = pd.read_csv(fp, index_col=0)

        if self.modality == DataModality.UNIVARIATE_REGRESSION:
            if out.shape[1] != 1:
                raise ValueError(
                    f"For {self.modality}, measurement metadata at {fp} should be a series, but "
                    f"it has shape {out.shape} (expecting out.shape[1] == 1)!"
                )
            out = out.iloc[:, 0]
            for col in ("outlier_model", "normalizer"):
                if col in out and type(out[col]) is str:
                    try:
                        out[col] = eval(out[col])
                    except (TypeError, ValueError) as e:
                        raise ValueError(
                            f"Failed to eval {col} for measure {self.name} with value {out[col]}"
                        ) from e
        elif self.modality != DataModality.MULTIVARIATE_REGRESSION:
            raise ValueError(
                "Only DataModality.UNIVARIATE_REGRESSION and DataModality.MULTIVARIATE_REGRESSION "
                f"measurements should have measurement metadata paths stored. Got {fp} on "
                f"{self.modality} measurement!"
            )
        else:
            for col in ("outlier_model", "normalizer"):
                if col in out:
                    try:
                        out[col] = out[col].apply(lambda x: eval(x) if type(x) is str else x)
                    except (TypeError, ValueError) as e:
                        raise ValueError(
                            f"Failed to eval {col} for measure {self.name} with values {list(out[col])[:5]}"
                        ) from e
        return out

    @measurement_metadata.setter
    def measurement_metadata(self, new_metadata: pd.DataFrame | pd.Series | None):
        if new_metadata is None:
            self._measurement_metadata = None
            return

        match self._measurement_metadata:
            case [Path() as base_dir, str() as fn]:
                new_metadata.to_csv(base_dir / fn)
            case Path() | str() as fp:
                new_metadata.to_csv(fp)
            case _:
                self._measurement_metadata = new_metadata

    def cache_measurement_metadata(self, base_dir: Path, fn: str):
        fp = base_dir / fn
        if isinstance(self._measurement_metadata, (str, Path)):
            if str(fp) != str(self._measurement_metadata):
                raise ValueError(f"Caching is already enabled at {self._measurement_metadata} != {fp}")
            return
        if self.measurement_metadata is None:
            return

        fp.parent.mkdir(exist_ok=True, parents=True)
        self.measurement_metadata.to_csv(fp)
        self._measurement_metadata = [str(base_dir.resolve()), fn]

    def uncache_measurement_metadata(self):
        if self._measurement_metadata is None:
            return

        match self._measurement_metadata:
            case [Path(), str()]:
                pass
            case Path() | str():
                pass
            case _:
                raise ValueError("Caching is not enabled, can't uncache!")

        self._measurement_metadata = self.measurement_metadata

    def add_empty_metadata(self):
        """Adds an empty `measurement_metadata` dataframe or series."""
        if self.measurement_metadata is not None:
            raise ValueError(f"Can't add empty metadata; already set to {self.measurement_metadata}")

        match self.modality:
            case DataModality.UNIVARIATE_REGRESSION:
                self._measurement_metadata = pd.Series(
                    [None] * len(self.PREPROCESSING_METADATA_COLUMNS),
                    index=self.PREPROCESSING_METADATA_COLUMNS,
                    dtype=object,
                )
            case DataModality.MULTIVARIATE_REGRESSION:
                self._measurement_metadata = pd.DataFrame(
                    {c: pd.Series([], dtype=t) for c, t in self.PREPROCESSING_METADATA_COLUMNS.items()},
                    index=pd.Index([], name=self.name),
                )
            case _:
                raise ValueError(f"Can't add metadata to a {self.modality} measure!")

    def add_missing_mandatory_metadata_cols(self):
        if not self.is_numeric:
            raise ValueError("Only numeric measurements can have measurement metadata")
        match self.measurement_metadata:
            case None:
                self.add_empty_metadata()

            case pd.DataFrame():
                for col, dtype in self.PREPROCESSING_METADATA_COLUMNS.items():
                    if col not in self.measurement_metadata.columns:
                        self.measurement_metadata[col] = pd.Series(
                            [None] * len(self.measurement_metadata), dtype=dtype
                        )
                if self.measurement_metadata.index.names == [None]:
                    self.measurement_metadata.index.names = [self.name]
            case pd.Series():
                for col, dtype in self.PREPROCESSING_METADATA_COLUMNS.items():
                    if col not in self.measurement_metadata.index:
                        self.measurement_metadata[col] = None

    def to_dict(self) -> dict:
        """Represents this configuration object as a plain dictionary."""
        as_dict = dataclasses.asdict(self)
        match self._measurement_metadata:
            case pd.DataFrame():
                as_dict["_measurement_metadata"] = self.measurement_metadata.to_dict(orient="tight")
            case pd.Series():
                as_dict["_measurement_metadata"] = self.measurement_metadata.to_dict(into=OrderedDict)
            case Path():
                as_dict["_measurement_metadata"] = str(self._measurement_metadata)
        if self.temporality == TemporalityType.FUNCTIONAL_TIME_DEPENDENT:
            as_dict["functor"] = self.functor.to_dict()
        if as_dict.get("vocabulary", None) is not None:
            as_dict["vocabulary"]["obs_frequencies"] = [
                float(x) for x in as_dict["vocabulary"]["obs_frequencies"]
            ]
        return as_dict

    @classmethod
    def from_dict(cls, as_dict: dict, base_dir: Path | None = None) -> MeasurementConfig:
        """Build a configuration object from a plain dictionary representation."""
        if as_dict["vocabulary"] is not None:
            as_dict["vocabulary"] = Vocabulary(**as_dict["vocabulary"])

        match as_dict["_measurement_metadata"], as_dict["modality"]:
            case None, _:
                pass
            case str() as full_path, _:
                full_path = Path(full_path)
                if full_path.parts[-2] == "inferred_measurement_metadata":
                    prior_base_dir = "/".join(full_path.parts[:-2])
                    relative_path = "/".join(full_path.parts[-2:])
                else:
                    raise ValueError(f"Can't process old path format of {full_path}")

                if base_dir is not None:
                    as_dict["_measurement_metadata"] = [base_dir, relative_path]
                else:
                    as_dict["_measurement_metadata"] = [str(prior_base_dir), relative_path]
            case [str() as prior_base_dir, str() as relative_path], _:
                if base_dir is not None:
                    as_dict["_measurement_metadata"] = [base_dir, relative_path]
                else:
                    as_dict["_measurement_metadata"] = [str(prior_base_dir), relative_path]
            case dict(), DataModality.MULTIVARIATE_REGRESSION:
                as_dict["_measurement_metadata"] = pd.DataFrame.from_dict(
                    as_dict["_measurement_metadata"], orient="tight"
                )
            case dict(), DataModality.UNIVARIATE_REGRESSION:
                as_dict["_measurement_metadata"] = pd.Series(as_dict["_measurement_metadata"])
            case _:
                raise ValueError(
                    f"{as_dict['_measurement_metadata']} and {as_dict['modality']} incompatible!"
                )

        # Handle the case where the 'functor' key is missing
        if "functor" not in as_dict:
            as_dict["functor"] = None
        elif as_dict["functor"] is not None:
            if as_dict["temporality"] != TemporalityType.FUNCTIONAL_TIME_DEPENDENT:
                raise ValueError(
                    "Only TemporalityType.FUNCTIONAL_TIME_DEPENDENT measures can have functors. Got "
                    f"{as_dict['temporality']}"
                )
            as_dict["functor"] = cls.FUNCTORS[as_dict["functor"]["class"]].from_dict(as_dict["functor"])

        return cls(**as_dict)

    def __eq__(self, other: MeasurementConfig) -> bool:
        return self.to_dict() == other.to_dict()

    def describe(
        self, line_width: int = 60, wrap_lines: bool = False, stream: TextIOBase | None = None
    ) -> int | None:
        """Provides a plain-text description of the measurement.

        Prints the following information about the MeasurementConfig object:

        1. The measurement's name, temporality, modality, and observation frequency.
        2. What value types (e.g., integral, float, etc.) it's values take on, if the measurement is a
           numerical modality whose values may take on distinct value types.
        3. Details about its internal `self.vocabulary` object, via `Vocabulary.describe`.

        Args:
            line_width: The maximum width of each line in the description.
            wrap_lines: Whether to wrap lines that exceed the `line_width`.
            stream: The stream to write the description to. If `None`, the description is printed to stdout.

        Returns:
            The number of characters written to the stream if a stream was provided, otherwise `None`.

        Raises:
            ValueError: if the calling object is misconfigured.

        Examples:
            >>> vocab = Vocabulary(
            ...     vocabulary=['apple', 'banana', 'pear', 'UNK'],
            ...     obs_frequencies=[3, 4, 1, 2],
            ... )
            >>> cfg = MeasurementConfig(
            ...     name="MVR",
            ...     values_column='bar',
            ...     temporality='dynamic',
            ...     modality='multivariate_regression',
            ...     observation_rate_over_cases=0.6816,
            ...     observation_rate_per_case=1.32,
            ...     _measurement_metadata=pd.DataFrame(
            ...         {'value_type': ['float', 'categorical', 'categorical']},
            ...         index=pd.Index(['apple', 'pear', 'banana'], name='MVR'),
            ...     ),
            ...     vocabulary=vocab,
            ... )
            >>> cfg.describe(line_width=100)
            MVR: dynamic, multivariate_regression observed 68.2%, 1.3/case on average
            Value Types:
              2 categorical
              1 float
            Vocabulary:
              4 elements, 20.0% UNKs
              Frequencies: █▆▁
              Elements:
                (40.0%) banana
                (30.0%) apple
                (10.0%) pear
            >>> cfg.modality = 'wrong'
            >>> cfg.describe()
            Traceback (most recent call last):
                ...
            ValueError: Can't describe wrong measure MVR!
        """
        lines = []
        lines.append(
            f"{self.name}: {self.temporality}, {self.modality} "
            f"observed {100*self.observation_rate_over_cases:.1f}%, "
            f"{self.observation_rate_per_case:.1f}/case on average"
        )

        match self.modality:
            case DataModality.UNIVARIATE_REGRESSION:
                lines.append(f"Value is a {self.measurement_metadata.value_type}")
            case DataModality.MULTIVARIATE_REGRESSION:
                lines.append("Value Types:")
                for t, cnt in self.measurement_metadata.value_type.value_counts().items():
                    lines.append(f"  {cnt} {t}")
            case DataModality.MULTI_LABEL_CLASSIFICATION:
                pass
            case DataModality.SINGLE_LABEL_CLASSIFICATION:
                pass
            case _:
                raise ValueError(f"Can't describe {self.modality} measure {self.name}!")

        if self.vocabulary is not None:
            SIO = StringIO()
            self.vocabulary.describe(line_width=line_width - 2, stream=SIO, wrap_lines=wrap_lines)
            lines.append("Vocabulary:")
            lines.extend(f"  {line}" for line in SIO.getvalue().split("\n"))

        line_indents = [num_initial_spaces(line) for line in lines]
        if wrap_lines:
            lines = [
                wrap(line, width=line_width, initial_indent="", subsequent_indent=(" " * ind))
                for line, ind in zip(lines, line_indents)
            ]
        else:
            lines = [
                shorten(line, width=line_width, initial_indent=(" " * ind))
                for line, ind in zip(lines, line_indents)
            ]

        desc = "\n".join(lines)
        if stream is None:
            print(desc)
            return
        return stream.write(desc)