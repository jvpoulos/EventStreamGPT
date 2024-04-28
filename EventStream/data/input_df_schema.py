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
class InputDFSchema(JSONableMixin):
    """The schema for one input DataFrame.

    Dataclass that defines the schema for an input DataFrame. It verifies the provided attributes
    during the post-initialization stage, and raises exceptions if mandatory attributes are missing or
    if any inconsistencies are found. It stores sufficient data to extract subject IDs;
    produce event or range timestamps; extract, rename, and convert columns; and filter data.

    Attributes:
        input_df: DataFrame input. This can take on many types, including an actual dataframe, a query to a
            database, or a path to a dataframe stored on disk. Mandatory attribute.
        type: Type of the input data. Possible values are InputDFType.STATIC, InputDFType.EVENT,
            or InputDFType.RANGE. Mandatory attribute.
        event_type: What categorical event_type should be assigned to events sourced from this input
            dataframe? For events, must be only a single string, or for ranges can either be a single string
            or a tuple of strings indicating event type names for start, start == stop, and stop events. If
            the string starts with "COL:" then the remaining portion of the string will be interpreted as a
            column name in the input from which the event type should be read. Otherwise it will be
            intrepreted as a literal event_type category name.
        subject_id_col: The name of the column containing the subject ID.
        ts_col: Column name containing timestamp for events.
        start_ts_col: Column name containing start timestamp for ranges.
        end_ts_col: Column name containing end timestamp for ranges.
        ts_format: String format of the timestamp in ts_col.
        start_ts_format: String format of the timestamp in start_ts_col.
        end_ts_format: String format of the timestamp in end_ts_col.
        data_schema: Schema of the input data.
        start_data_schema: Schema of the start data in a range. If unspecified for a range, will fall back on
            data_schema.
        end_data_schema: Schema of the end data in a range. If unspecified for a range, will fall back on
            data_schema.
        must_have: List of mandatory columns or filters to apply, as a mapping from column name to filter to
            apply. The filter can either be `True`, in which case the column simply must have a non-null
            value, or a list of options, in which case the column must take on one of those values for the row
            to be included.

    Raises:
        ValueError: If mandatory attributes (input_df, type) are not provided, or if inconsistencies
            are found in the attributes based on the input data type.
        TypeError: If attributes are of the wrong type.

    Examples:
        >>> S = InputDFSchema(
        ...     input_df="/path/to/df.csv",
        ...     type='static',
        ...     subject_id_col='subj_id',
        ...     must_have=['subj_id', ['foo', ['opt1', 'opt2']]],
        ... )
        >>> S.filter_on
        {'subj_id': True, 'foo': ['opt1', 'opt2']}
        >>> S.is_static
        True
        >>> S = InputDFSchema(
        ...     input_df="/path/to_df.parquet",
        ...     type='event',
        ...     ts_col='col',
        ...     event_type='bar',
        ... )
        >>> S.is_static
        False
        >>> S
        InputDFSchema(input_df='/path/to_df.parquet', type='event', event_type='bar', ts_col='col')
        >>> S = InputDFSchema(
        ...     input_df="/path/to_df.parquet",
        ...     type='range',
        ...     start_ts_col='start',
        ...     end_ts_col='end',
        ...     event_type=('bar_st_eq_end', 'bar_st', 'bar_end'),
        ... )
        >>> S.is_static
        False
        >>> InputDFSchema()
        Traceback (most recent call last):
            ...
        ValueError: Missing mandatory parameter input_df!
        >>> S = InputDFSchema(input_df="/path/to/df.csv")
        Traceback (most recent call last):
            ...
        ValueError: Missing mandatory parameter type!
        >>> S = InputDFSchema(
        ...     input_df="/path/to/df.csv",
        ...     type='static',
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Must set subject_id_col for static source!
        >>> S = InputDFSchema(
        ...     input_df="/path/to/df.csv",
        ...     type='static',
        ...     subject_id_col='subj_id',
        ...     must_have=[34]
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Malformed filter: 34
        >>> S = InputDFSchema(
        ...     input_df="/path/to/df.parquet",
        ...     type=InputDFType.RANGE,
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Missing mandatory range parameter event_type!
        >>> S = InputDFSchema(
        ...     input_df="/path/to/df.csv",
        ...     type='static',
        ...     subject_id_col='subj_id',
        ...     event_type='foo'
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Set invalid param event_type for static source!
        >>> S = InputDFSchema(
        ...     input_df="/path/to_df.parquet",
        ...     type='event',
        ...     event_type='bar',
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Missing mandatory event parameter ts_col!
        >>> S = InputDFSchema(
        ...     input_df="/path/to_df.parquet",
        ...     type='event',
        ...     ts_col='bar',
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Missing mandatory event parameter event_type!
        >>> S = InputDFSchema(
        ...     input_df="/path/to_df.parquet",
        ...     type='event',
        ...     ts_col='bar',
        ...     event_type='foo',
        ...     subject_id_col='subj',
        ... )
        Traceback (most recent call last):
            ...
        ValueError: subject_id_col should be None for non-static types!
        >>> S = InputDFSchema(
        ...     input_df="/path/to_df.parquet",
        ...     type='event',
        ...     ts_col='bar',
        ...     event_type=('foo', 'categorical'),
        ... )
        Traceback (most recent call last):
            ...
        TypeError: event_type must be a string for events. Got ('foo', 'categorical')
        >>> S = InputDFSchema(
        ...     input_df="/path/to_df.parquet",
        ...     type='event',
        ...     ts_col='bar',
        ...     event_type='foo',
        ...     start_ts_col='start',
        ... )
        Traceback (most recent call last):
            ...
        ValueError: start_ts_col should be None for event schema: Got start
        >>> S = InputDFSchema(
        ...     input_df="/path/to_df.parquet",
        ...     type='event',
        ...     ts_col='col',
        ...     event_type='bar',
        ...     data_schema=('foobar', 'categorical'),
        ... )
        >>> S.is_static
        False
        >>> S # doctest: +NORMALIZE_WHITESPACE
        InputDFSchema(input_df='/path/to_df.parquet',
                      type='event',
                      event_type='bar',
                      ts_col='col',
                      data_schema=[('foobar', 'categorical')])
        >>> S.unified_schema
        {'foobar': ('foobar', 'categorical')}
        >>> S.columns_to_load
        [('foobar', 'categorical'), ('col', <InputDataType.TIMESTAMP: 'timestamp'>)]
        >>> S = InputDFSchema(
        ...     input_df="/path/to_df.parquet",
        ...     type='range',
        ...     start_ts_col='start',
        ...     end_ts_col='end',
        ...     event_type='bar',
        ...     start_data_schema=[
        ...         {'buz': 'float'},
        ...         {'baz': ['timestamp', '%Y-%m']}
        ...     ],
        ...     end_data_schema={'foobar': InputDataType.FLOAT},
        ... )
        >>> for n, schema in zip(('EQ', 'ST', 'END'), S.unified_schema):
        ...     print(f"{n}:")
        ...     for k, v in sorted(schema.items()):
        ...         print(f"  {k}: {v}")
        EQ:
          baz: ('baz', ['timestamp', '%Y-%m'])
          buz: ('buz', 'float')
          foobar: ('foobar', <InputDataType.FLOAT: 'float'>)
        ST:
          baz: ('baz', ['timestamp', '%Y-%m'])
          buz: ('buz', 'float')
        END:
          foobar: ('foobar', <InputDataType.FLOAT: 'float'>)
        >>> S = InputDFSchema(
        ...     input_df="/path/to_df.parquet",
        ...     type='range',
        ...     start_ts_col='start',
        ...     end_ts_col='end',
        ...     ts_format='%Y-%m-%d',
        ...     event_type='bar',
        ...     start_data_schema={'foobar': ('foobar_renamed', ['timestamp', '%Y'])},
        ...     end_data_schema=[
        ...         ('buz', 'float'),
        ...         (['biz', 'whizz'], 'categorical'),
        ...     ],
        ... )
        >>> for n, schema in zip(('EQ', 'ST', 'END'), S.unified_schema):
        ...     print(f"{n}:")
        ...     for k, v in sorted(schema.items()):
        ...         print(f"  {k}: {v}")
        EQ:
          biz: ('biz', 'categorical')
          buz: ('buz', 'float')
          foobar: ('foobar_renamed', ['timestamp', '%Y'])
          whizz: ('whizz', 'categorical')
        ST:
          foobar: ('foobar_renamed', ['timestamp', '%Y'])
        END:
          biz: ('biz', 'categorical')
          buz: ('buz', 'float')
          whizz: ('whizz', 'categorical')
        >>> list(sorted(S.columns_to_load)) # doctest: +NORMALIZE_WHITESPACE
        [('biz', 'categorical'), ('buz', 'float'),
         ('end', (<InputDataType.TIMESTAMP: 'timestamp'>, '%Y-%m-%d')),
         ('foobar', ['timestamp', '%Y']),
         ('start', (<InputDataType.TIMESTAMP: 'timestamp'>, '%Y-%m-%d')),
         ('whizz', 'categorical')]
    """

    input_df: Any | None = None

    type: InputDFType | None = None
    event_type: str | tuple[str, str, str] | None = None

    subject_id_col: str | None = None
    ts_col: DF_COL | None = None
    start_ts_col: DF_COL | None = None
    end_ts_col: DF_COL | None = None
    ts_format: str | None = None
    start_ts_format: str | None = None
    end_ts_format: str | None = None

    data_schema: DF_SCHEMA | list[DF_SCHEMA] | None = None
    start_data_schema: DF_SCHEMA | list[DF_SCHEMA] | None = None
    end_data_schema: DF_SCHEMA | list[DF_SCHEMA] | None = None

    must_have: list[str | tuple[str, list[Any]]] = dataclasses.field(default_factory=list)

    @property
    def is_static(self):
        """Returns True if and only if the input data type is static."""
        return self.type == InputDFType.STATIC

    def __post_init__(self):
        if self.input_df is None:
            raise ValueError("Missing mandatory parameter input_df!")
        if self.type is None:
            raise ValueError("Missing mandatory parameter type!")
        if type(self.data_schema) is not list and self.data_schema is not None:
            self.data_schema = [self.data_schema]
        if type(self.start_data_schema) is not list and self.start_data_schema is not None:
            self.start_data_schema = [self.start_data_schema]
        if type(self.end_data_schema) is not list and self.end_data_schema is not None:
            self.end_data_schema = [self.end_data_schema]

        self.filter_on = {}
        for filter_col in self.must_have:
            match filter_col:
                case str():
                    self.filter_on[filter_col] = True
                case (str() as filter_col, list() as vals):
                    self.filter_on[filter_col] = vals
                case _:
                    raise ValueError(f"Malformed filter: {filter_col}")

        match self.type:
            case InputDFType.STATIC:
                if self.subject_id_col is None:
                    raise ValueError("Must set subject_id_col for static source!")

                for param in ("event_type", "ts_col", "start_ts_col", "end_ts_col"):
                    if getattr(self, param) is not None:
                        raise ValueError(f"Set invalid param {param} for static source!")

            case InputDFType.EVENT:
                if self.ts_col is None:
                    raise ValueError("Missing mandatory event parameter ts_col!")
                match self.event_type:
                    case None:
                        raise ValueError("Missing mandatory event parameter event_type!")
                    case str():
                        pass
                    case _:
                        raise TypeError(f"event_type must be a string for events. Got {self.event_type}")
                if self.subject_id_col is not None:
                    raise ValueError("subject_id_col should be None for non-static types!")
                for param in (
                    "start_ts_col",
                    "end_ts_col",
                    "start_ts_format",
                    "end_ts_format",
                    "start_data_schema",
                    "end_data_schema",
                ):
                    val = getattr(self, param)
                    if val is not None:
                        raise ValueError(f"{param} should be None for {self.type} schema: Got {val}")

            case InputDFType.RANGE:
                match self.event_type:
                    case None:
                        raise ValueError("Missing mandatory range parameter event_type!")
                    case (str(), str(), str()):
                        pass
                    case str():
                        self.event_type = (
                            self.event_type,
                            f"{self.event_type}_START",
                            f"{self.event_type}_END",
                        )
                    case _:
                        raise TypeError(
                            "event_type must be a string or a 3-element tuple (eq_type, st_type, end_type) "
                            f"for ranges. Got {self.event_type}."
                        )

                if self.data_schema is not None:
                    for param in ("start_data_schema", "end_data_schema"):
                        val = getattr(self, param)
                        if val is not None:
                            raise ValueError(
                                f"{param} can't be simultaneously set with `self.data_schema`! Got {val}"
                            )

                    self.start_data_schema = self.data_schema
                    self.end_data_schema = self.data_schema

                if self.start_ts_col is None:
                    raise ValueError("Missing mandatory range parameter start_ts_col!")
                if self.end_ts_col is None:
                    raise ValueError("Missing mandatory range parameter end_ts_col!")
                if self.ts_col is not None:
                    raise ValueError(f"ts_col should be `None` for {self.type} schemas! Got: {self.ts_col}.")
                if self.subject_id_col is not None:
                    raise ValueError("subject_id_col should be None for non-static types!")
                if self.start_ts_format is not None:
                    if self.end_ts_format is None:
                        raise ValueError(
                            "If start_ts_format is specified, end_ts_format must also be specified!"
                        )
                    if self.ts_format is not None:
                        raise ValueError("If start_ts_format is specified, ts_format must be `None`!")
                else:
                    if self.end_ts_format is not None:
                        raise ValueError(
                            "If end_ts_format is specified, start_ts_format must also be specified!"
                        )

                    self.start_ts_format = self.ts_format
                    self.end_ts_format = self.ts_format
                    self.ts_format = None

        # This checks validity.
        self.columns_to_load

    def __repr__(self) -> str:
        kwargs = {k: v for k, v in self.to_dict().items() if v}
        kwargs_str = [f"{k}={repr(v)}" for k, v in kwargs.items()]
        return f"InputDFSchema({', '.join(kwargs_str)})"

    @property
    def columns_to_load(self) -> list[tuple[str, InputDataType]]:
        """Computes the columns to be loaded based on the input data type and schema.

        Returns:
            A list of tuples of column names and desired types for the columns to be loaded from the input
            dataframe.

        Raises:
            ValueError: If any of the column definitions are invalid or repeated.
        """
        columns_to_load = {}

        match self.type:
            case InputDFType.EVENT | InputDFType.STATIC:
                for in_col, (out_col, dt) in self.unified_schema.items():
                    if in_col in columns_to_load:
                        raise ValueError(f"Duplicate column {in_col}!")
                    columns_to_load[in_col] = dt
            case InputDFType.RANGE:
                for unified_schema in self.unified_schema:
                    for in_col, (out_col, dt) in unified_schema.items():
                        if in_col in columns_to_load:
                            if dt != columns_to_load[in_col]:
                                raise ValueError(f"Duplicate column {in_col} with differing dts!")
                        else:
                            columns_to_load[in_col] = dt
            case _:
                raise ValueError(f"Unrecognized type {self.type}!")

        columns_to_load = list(columns_to_load.items())

        for param, fmt_param in [
            ("start_ts_col", "start_ts_format"),
            ("end_ts_col", "end_ts_format"),
            ("ts_col", "ts_format"),
        ]:
            val = getattr(self, param)
            fmt_param = getattr(self, fmt_param)
            if fmt_param is None:
                fmt = InputDataType.TIMESTAMP
            else:
                fmt = (InputDataType.TIMESTAMP, fmt_param)

            match val:
                case list():
                    columns_to_load.extend([(c, fmt) for c in val])
                case str():
                    columns_to_load.append((val, fmt))
                case None:
                    pass
                case _:
                    raise ValueError(f"Can't parse timestamp {param}, {fmt_param}, {val}")

        return columns_to_load

    @property
    def unified_schema(self) -> dict[str, tuple[str, InputDataType]]:
        """Computes the unified schema based on the input data type and data schema.

        Returns:
            A unified schema mapping from output column names to input column names and types.

        Raises:
            ValueError: If the type attribute of the calling object is invalid.
        """
        match self.type:
            case InputDFType.EVENT | InputDFType.STATIC:
                return self.unified_event_schema
            case InputDFType.RANGE:
                return [self.unified_eq_schema, self.unified_start_schema, self.unified_end_schema]
            case _:
                raise ValueError(f"Unrecognized type {self.type}!")

    @property
    def unified_event_schema(self) -> dict[str, tuple[str, InputDataType]]:
        return self._unify_schema(self.data_schema)

    @property
    def unified_start_schema(self) -> dict[str, tuple[str, InputDataType]]:
        if self.type != InputDFType.RANGE:
            raise ValueError(f"Start schema is invalid for {self.type}")

        if self.start_data_schema is None:
            return self._unify_schema(self.data_schema)
        return self._unify_schema(self.start_data_schema)

    @property
    def unified_end_schema(self) -> dict[str, tuple[str, InputDataType]]:
        if self.type != InputDFType.RANGE:
            raise ValueError(f"End schema is invalid for {self.type}")

        if self.end_data_schema is None:
            return self._unify_schema(self.data_schema)
        return self._unify_schema(self.end_data_schema)

    @property
    def unified_eq_schema(self) -> dict[str, tuple[str, InputDataType]]:
        if self.type != InputDFType.RANGE:
            raise ValueError(f"Start=End schema is invalid for {self.type}")

        if self.start_data_schema is None and self.end_data_schema is None:
            return self._unify_schema(self.data_schema)

        ds = []
        if self.start_data_schema is not None:
            if type(self.start_data_schema) is list:
                ds.extend(self.start_data_schema)
            else:
                ds.append(self.start_data_schema)

        if self.end_data_schema is not None:
            if type(self.end_data_schema) is list:
                ds.extend(self.end_data_schema)
            else:
                ds.append(self.end_data_schema)

        return self._unify_schema(ds)

    @classmethod
    def __add_to_schema(
        cls,
        container: dict[str, tuple[str, InputDataType]],
        in_col: str,
        dt: INPUT_COL_T,
        out_col: str | None = None,
    ):
        if out_col is None:
            out_col = in_col

        if type(in_col) is not str or type(out_col) is not str:
            raise ValueError(f"Column names must be strings! Got {in_col}, {out_col}")
        elif in_col in container and container[in_col] != (out_col, dt):
            raise ValueError(
                f"Column {in_col} is repeated in schema with different value!\n"
                f"Existing: {container[in_col]}\n"
                f"New: ({out_col}, {dt})"
            )
        container[in_col] = (out_col, dt)

    @classmethod
    def _unify_schema(
        cls, data_schema: DF_SCHEMA | list[DF_SCHEMA] | None
    ) -> dict[str, tuple[str, InputDataType]]:
        if data_schema is None:
            return {}

        unified_schema = {}
        for schema in data_schema:
            match schema:
                case str() as col, ((InputDataType() | str()) | [InputDataType.TIMESTAMP, str()]) as dt:
                    cls.__add_to_schema(unified_schema, in_col=col, dt=dt)
                case list() as cols, ((InputDataType() | str()) | [InputDataType.TIMESTAMP, str()]) as dt:
                    for c in cols:
                        cls.__add_to_schema(unified_schema, in_col=c, dt=dt)
                case dict():
                    for in_col, schema_info in schema.items():
                        match schema_info:
                            case str() as out_col, str() as dt if dt in InputDataType.values():
                                cls.__add_to_schema(unified_schema, in_col=in_col, dt=dt, out_col=out_col)
                            case str() as out_col, InputDataType() as dt:
                                cls.__add_to_schema(unified_schema, in_col=in_col, dt=dt, out_col=out_col)
                            case str() as out_col, [InputDataType.TIMESTAMP, str()] as dt:
                                cls.__add_to_schema(unified_schema, in_col=in_col, dt=dt, out_col=out_col)
                            case [InputDataType.TIMESTAMP, str()] as dt:
                                cls.__add_to_schema(unified_schema, in_col=in_col, dt=dt)
                            case str() | InputDataType() as dt if dt in InputDataType.values():
                                cls.__add_to_schema(unified_schema, in_col=in_col, dt=dt)
                            case _:
                                raise ValueError(f"Schema Unprocessable!\n{schema_info}")
                case dict() as col_names_map, (InputDataType() | [InputDataType.TIMESTAMP, str()]) as dt:
                    for in_col, out_col in col_names_map.items():
                        cls.__add_to_schema(unified_schema, in_col=in_col, dt=dt, out_col=out_col)
                case _:
                    raise ValueError(f"Schema Unprocessable!\n{schema}")

        return unified_schema

    def to_dict(self) -> dict:
        """Represents this configuration object as a plain dictionary."""
        as_dict = dataclasses.asdict(self)

        # Convert input_df to a JSON-serializable format
        if isinstance(self.input_df, pl.DataFrame):
            as_dict["input_df"] = self.input_df.to_pandas().to_dict(orient="records")
        elif isinstance(self.input_df, pd.DataFrame):
            as_dict["input_df"] = self.input_df.to_dict(orient="records")

        return as_dict

    @classmethod
    def from_dict(cls, as_dict: dict) -> InputDFSchema:
        """Build a configuration object from a plain dictionary representation."""
        # Convert input_df back to the original format
        if "input_df" in as_dict and isinstance(as_dict["input_df"], list):
            as_dict["input_df"] = pd.DataFrame(as_dict["input_df"])

        return cls(**as_dict)