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
class DatasetSchema(JSONableMixin):
    """Represents the schema of an input dataset, including static and dynamic data sources.

    Contains the information necessary for extracting and pulling input dataset elements during a
    pre-processing pipeline. Inputs can be represented in either structured (typed) or plain (dictionary)
    form. There can only be one static schema currently, but arbitrarily many dynamic measurement schemas.
    During pre-processing the model will read all these dynamic input datasets and combine their outputs into
    the appropriate format. This can be written to or read from JSON files via the `JSONableMixin` base class
    methods.

    Attributes:
        static: The schema for the input dataset containing static (per-subject) information, in either object
            or dict form.
        dynamic: A list of schemas for all dynamic dataset schemas, each in either object or dict form.

    Raises:
        ValueError: If the static schema is `None`, if there is not a subject ID column specified in the
            static schema, if the passed "static" schema is not typed as a static schema, or if any dynamic
            schema is typed as a static schema.

    Examples:
        >>> DatasetSchema(dynamic=[])
        Traceback (most recent call last):
            ...
        ValueError: Must specify a static schema!
        >>> DatasetSchema(
        ...     static=dict(type="event", event_type="foo", input_df="/path/to/df.csv", ts_col="col"),
        ...     dynamic=[]
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Must pass a static schema config for static.
        >>> DatasetSchema(
        ...     static=dict(type="static", input_df="/path/to/df.csv", subject_id_col="col"),
        ...     dynamic=[dict(type="static", input_df="/path/to/df.csv", subject_id_col="col")]
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Must pass dynamic schemas in self.dynamic!
        >>> DS = DatasetSchema(
        ...     static=dict(type="static", input_df="/path/to/df.csv", subject_id_col="col"),
        ...     dynamic=[
        ...         dict(type="event", event_type="foo", input_df="/path/to/foo.csv", ts_col="col"),
        ...         dict(type="event", event_type="bar", input_df="/path/to/bar.csv", ts_col="col"),
        ...         dict(type="event", event_type="bar2", input_df="/path/to/bar.csv", ts_col="col2"),
        ...     ],
        ... )
        >>> DS.dynamic_by_df # doctest: +NORMALIZE_WHITESPACE
        {'/path/to/foo.csv': [InputDFSchema(input_df='/path/to/foo.csv', type='event', event_type='foo',
        subject_id_col='col', ts_col='col')], '/path/to/bar.csv': [InputDFSchema(input_df='/path/to/bar.csv',
        type='event', event_type='bar', subject_id_col='col', ts_col='col'),
        InputDFSchema(input_df='/path/to/bar.csv', type='event', event_type='bar2', subject_id_col='col',
        ts_col='col2')]}
    """

    static: dict[str, Any] | InputDFSchema | None = None
    dynamic: list[InputDFSchema | dict[str, Any]] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        if self.static is None:
            raise ValueError("Must specify a static schema!")

        if isinstance(self.static, dict):
            self.static = InputDFSchema(**self.static)
            if not self.static.is_static:
                raise ValueError("Must pass a static schema config for static.")

        if self.dynamic is not None:
            new_dynamic = []
            for v in self.dynamic:
                if isinstance(v, dict):
                    v = InputDFSchema(**v)
                if isinstance(v.input_df, pl.DataFrame):
                    v.input_df = v.input_df.to_pandas()
                v.subject_id_col = self.static.subject_id_col

                new_dynamic.append(v)

                if v.is_static:
                    raise ValueError("Must pass dynamic schemas in self.dynamic!")
            self.dynamic = new_dynamic

        self.dynamic_by_df = defaultdict(list)
        for v in self.dynamic:
            # Convert polars DataFrame to pandas DataFrame
            if isinstance(v.input_df, pl.DataFrame):
                v.input_df = v.input_df.to_pandas()
            # Generate a unique identifier for the DataFrame
            df_id = str(hash(pd.util.hash_pandas_object(v.input_df).sum()))
            self.dynamic_by_df[df_id].append(v)
        self.dynamic_by_df = {k: v for k, v in self.dynamic_by_df.items()}

    def to_dict(self) -> dict:
        """Represents this configuration object as a plain dictionary."""
        as_dict = dataclasses.asdict(self)
        as_dict["static"] = self.static.to_dict()
        as_dict["dynamic"] = [schema.to_dict() for schema in self.dynamic]
        return as_dict

    @classmethod
    def from_dict(cls, as_dict: dict) -> DatasetSchema:
        """Build a configuration object from a plain dictionary representation."""
        as_dict["static"] = InputDFSchema.from_dict(as_dict["static"])
        as_dict["dynamic"] = [InputDFSchema.from_dict(schema) for schema in as_dict["dynamic"]]
        return cls(**as_dict)