"""The polars implementation of the Dataset class.

Attributes:
    INPUT_DF_T: The types of supported input dataframes, which includes paths, pandas dataframes, polars
        dataframes, or queries.
    DF_T: The types of supported dataframes, which include polars lazyframes, dataframes, expressions, or
        series.
"""

from .dataset_config import DatasetConfig
from .dataset_base import DatasetBase
def get_dataset_class():
    from EventStream.data.dataset_base import get_dataset_class
    return get_dataset_class

import dataclasses
import math
import multiprocessing
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
from mixins import TimeableMixin

from ..utils import lt_count_or_proportion
from .preprocessing import Preprocessor, StandardScaler, StddevCutoffOutlierDetector
from .types import (
    DataModality,
    InputDataType,
    NumericDataModalitySubtype,
    TemporalityType,
)
from .vocabulary import Vocabulary
from .measurement_config import MeasurementConfig

# We need to do this so that categorical columns can be reliably used via category names.
pl.enable_string_cache(True)

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

DF_T = Union[pl.LazyFrame, pl.DataFrame, pl.Expr, pl.Series]

@dataclasses.dataclass(frozen=True)
class Query:
    """A structure for database query based input dataframes.

    Args:
        connection_uri: The connection URI for the database. This is in the `connectorx`_ format.
        query: The query to be run over the database. It can be specified either as a direct string, a path to
            a file on disk containing the query in txt format, or a list of said options.
        partition_on: If the query should be partitioned, on what column should it be partitioned? See the
            `polars documentation`_ for more details.
        partition_num: If the query should be partitioned, into how many partitions should it be divided? See
            the `polars documentation`_ for more details.
        protocol: The `connectorx`_ backend protocol.

    .. connectorx_: https://github.com/sfu-db/connector-x
    .. polars documentation_: https://pola-rs.github.io/polars/py-polars/html/reference/api/polars.read_database.html
    """  # noqa E501

    connection_uri: str
    query: str | Path | list[str | Path]
    partition_on: str | None = None
    partition_num: int | None = None
    protocol: str = "binary"

    def __str__(self):
        return f'Query("{self.query}")'

INPUT_DF_T = Union[Path, pd.DataFrame, pl.DataFrame, Query]

class Dataset(DatasetBase):
    def __init__(self, config=None, subjects_df=None, events_df=None, dynamic_measurements_df=None, code_mapping=None, **kwargs):
        if config is None:
            config = DatasetConfig()
        elif isinstance(config, dict):
            config = DatasetConfig.from_dict(config)

        self.config = config
        self.code_mapping = code_mapping or self._create_code_mapping()
        self.inverse_mapping = {v: k for k, v in self.code_mapping.items()}

        # Initialize the necessary attributes
        self.subjects_df = subjects_df
        self.events_df = events_df
        self.dynamic_measurements_df = dynamic_measurements_df

        # Call the parent class constructor
        super().__init__(config, subjects_df, events_df, dynamic_measurements_df, **kwargs)

    """The polars specific implementation of the dataset.

    Args:
        config: Configuration object for this dataset.
        subjects_df: The dataframe containing all static, subject-level data. If this is specified,
            `events_df` and `dynamic_measurements_df` should also be specified. Otherwise, this will be built
            from source via the extraction pipeline defined in `input_schema`.
        events_df:  The dataframe containing all event timestamps, types, and subject IDs. If this is
            specified, `subjects_df` and `dynamic_measurements_df` should also be specified. Otherwise, this
            will be built from source via the extraction pipeline defined in `input_schema`.
        dynamic_measurements_df: The dataframe containing all time-varying measurement observations. If this
            is specified, `subjects_df` and `events_df` should also be specified. Otherwise, this will be
            built from source via the extraction pipeline defined in `input_schema`.
        input_schema: The schema configuration object to define the extraction pipeline for pulling raw data
            from source and produce the `subjects_df`, `events_df`, `dynamic_measurements_df` input view.
    """

    # Dictates what models can be fit on numerical metadata columns, for both outlier detection and
    # normalization.

    PREPROCESSORS: dict[str, Preprocessor] = {
        # Outlier Detectors
        "stddev_cutoff": StddevCutoffOutlierDetector,
        # Normalizers
        "standard_scaler": StandardScaler,
    }
    """A dictionary containing the valid pre-processors that can be used by this model class."""

    METADATA_SCHEMA = {
        "drop_upper_bound": pl.Float64,
        "drop_upper_bound_inclusive": pl.Boolean,
        "drop_lower_bound": pl.Float64,
        "drop_lower_bound_inclusive": pl.Boolean,
        "censor_upper_bound": pl.Float64,
        "censor_lower_bound": pl.Float64,
        "outlier_model": lambda outlier_params_schema: pl.Struct(outlier_params_schema),
        "normalizer": lambda normalizer_params_schema: pl.Struct(normalizer_params_schema),
        "value_type": pl.Categorical,
    }
    """The Polars schema of the numerical measurement metadata dataframes which track fit parameters."""

    WRITE_USE_PYARROW = False
    """Use C++ parquet implementation vs Rust parquet implementation for writing parquets."""
    STREAMING = True
    """Execute any lazy query in streaming mode."""

    @staticmethod
    def get_smallest_valid_uint_type(num: Union[int, float]) -> pl.DataType:
        if num <= 255:
            return pl.UInt8
        elif num <= 65535:
            return pl.UInt16
        elif num <= 4294967295:
            return pl.UInt32
        else:
            return pl.UInt64
            
    @classmethod
    def _load_input_df(
        cls,
        df: INPUT_DF_T,
        columns: list[tuple[str, InputDataType | tuple[InputDataType, str]]],
        subject_id_col: str | None = None,
        subject_ids_map: dict[Any, int] | None = None,
        subject_id_dtype: Any | None = None,
        filter_on: dict[str, bool | list[Any]] | None = None,
        subject_id_source_col: str | None = None,
    ) -> DF_T | tuple[DF_T, str]:
        if subject_id_col is None:
            if subject_ids_map is not None:
                raise ValueError("Must not set subject_ids_map if subject_id_col is not set")
            if subject_id_dtype is not None:
                raise ValueError("Must not set subject_id_dtype if subject_id_col is not set")
        else:
            if subject_ids_map is None:
                raise ValueError("Must set subject_ids_map if subject_id_col is set")
            if subject_id_dtype is None:
                raise ValueError("Must set subject_id_dtype if subject_id_col is set")

        match df:
            case (str() | Path()) as fp:
                if not isinstance(fp, Path):
                    fp = Path(fp)

                if fp.suffix == ".csv":
                    df = pl.scan_csv(df, null_values="")
                elif fp.suffix == ".parquet":
                    df = pl.scan_parquet(df)
                else:
                    raise ValueError(f"Can't read dataframe from file of suffix {fp.suffix}")
            case pd.DataFrame():
                df = pl.from_pandas(df, include_index=True).lazy()
            case pl.DataFrame():
                df = df.lazy()
            case pl.LazyFrame():
                pass
            case Query() as q:
                query = q.query
                if not isinstance(query, (list, tuple)):
                    query = [query]

                out_query = []
                for qq in query:
                    if type(qq) is Path:
                        with open(qq) as f:
                            qq = f.read()
                    elif type(qq) is not str:
                        raise ValueError(f"{type(qq)} is an invalid query.")
                    out_query.append(qq)

                if len(out_query) == 1:
                    partition_kwargs = {
                        "partition_on": subject_id_col if q.partition_on is None else q.partition_on,
                        "partition_num": (
                            multiprocessing.cpu_count() if q.partition_num is None else q.partition_num
                        ),
                    }
                elif q.partition_on is not None or q.partition_num is not None:
                    raise ValueError(
                        "Partitioning ({q.partition_on}, {q.partition_num}) not supported when "
                        "passing multiple queries ({out_query})"
                    )
                else:
                    partition_kwargs = {}

                df = pl.read_database(
                    query=out_query,
                    connection_uri=q.connection_uri,
                    protocol=q.protocol,
                    **partition_kwargs,
                ).lazy()
            case _:
                # If df is not a file path, DataFrame, or Query, assume it's a unique identifier
                df = pl.DataFrame({subject_id_col: [], **{col[0]: [] for col in columns}})


        col_exprs = []

        df = df.select(pl.all().shrink_dtype())

        if filter_on:
            df = cls._filter_col_inclusion(df, filter_on)

        if subject_id_source_col is not None:
            internal_subj_key = "subject_id"
            while internal_subj_key in df.columns:
                internal_subj_key = f"_{internal_subj_key}"
            df = df.with_row_count(internal_subj_key)
            col_exprs.append(internal_subj_key)
        else:
            assert subject_id_col is not None
            df = df.with_columns(pl.col(subject_id_col).cast(pl.Utf8).cast(pl.Categorical))
            df = cls._filter_col_inclusion(df, {subject_id_col: list(subject_ids_map.keys())})
            col_exprs.append(
                pl.col(subject_id_col).map_dict(subject_ids_map).cast(subject_id_dtype).alias("subject_id")
            )

        for in_col, out_dt in columns:
            match out_dt:
                case InputDataType.FLOAT:
                    col_exprs.append(pl.col(in_col).cast(pl.Float32, strict=False))
                case InputDataType.CATEGORICAL:
                    col_exprs.append(pl.col(in_col).cast(pl.Utf8).cast(pl.Categorical))
                case InputDataType.BOOLEAN:
                    col_exprs.append(pl.col(in_col).cast(pl.Boolean, strict=False))
                case InputDataType.TIMESTAMP:
                    col_exprs.append(pl.col(in_col).cast(pl.Datetime, strict=True))
                case (InputDataType.TIMESTAMP, str() as ts_format):
                    col_exprs.append(pl.col(in_col).str.strptime(pl.Datetime, ts_format, strict=False))
                case _:
                    raise ValueError(f"Invalid out data type {out_dt}!")

        if subject_id_source_col is not None:
            df = df.select(col_exprs).collect(streaming=cls.STREAMING)

            ID_map = {o: n for o, n in zip(df[subject_id_source_col], df[internal_subj_key])}
            df = df.with_columns(pl.col(internal_subj_key).alias("subject_id"))
            return df, ID_map
        else:
            return df.select(col_exprs)

    @classmethod
    def _rename_cols(cls, df: DF_T, to_rename: dict[str, str]) -> DF_T:
        """Renames the columns in df according to the {in_name: out_name}s specified in to_rename.

        Args:
            df: The dataframe whose columns should be renamed.
            to_rename: A mapping of in column names to out column names.

        Returns: The dataframe with columns renamed.

        Examples:
            >>> import polars as pl
            >>> df = pl.DataFrame({'a': [1, 2, 3], 'b': ['foo', None, 'bar'], 'c': [1., 2.0, float('inf')]})
            >>> Dataset._rename_cols(df, {'a': 'a', 'b': 'biz'})
            shape: (3, 3)
            ┌─────┬──────┬─────┐
            │ a   ┆ biz  ┆ c   │
            │ --- ┆ ---  ┆ --- │
            │ i64 ┆ str  ┆ f64 │
            ╞═════╪══════╪═════╡
            │ 1   ┆ foo  ┆ 1.0 │
            │ 2   ┆ null ┆ 2.0 │
            │ 3   ┆ bar  ┆ inf │
            └─────┴──────┴─────┘
        """

        return df.rename(to_rename)

    @classmethod
    def _resolve_ts_col(cls, df: DF_T, ts_col: str | list[str], out_name: str = "timestamp") -> DF_T:
        match ts_col:
            case list():
                ts_expr = pl.min(ts_col)
                ts_to_drop = [c for c in ts_col if c != out_name]
            case str():
                ts_expr = pl.col(ts_col)
                ts_to_drop = [ts_col] if ts_col != out_name else []

        return df.with_columns(ts_expr.alias(out_name)).drop(ts_to_drop)

    @classmethod
    def _process_events_and_measurements_df(
        cls,
        df: DF_T,
        event_type: str,
        columns_schema: dict[str, tuple[str, InputDataType]],
    ) -> tuple[DF_T, DF_T | None]:
        cols_select_exprs = [
            pl.col("Date").alias("timestamp"),
            pl.col("EMPI").alias("subject_id"),
        ]
        
        if event_type.startswith("COL:"):
            event_type_col = event_type[len("COL:") :]
            cols_select_exprs.append(pl.col(event_type_col).cast(pl.Categorical).alias("event_type"))
        else:
            cols_select_exprs.append(pl.lit(event_type).cast(pl.Categorical).alias("event_type"))

        for in_col, (out_col, _) in columns_schema.items():
            cols_select_exprs.append(pl.col(in_col).alias(out_col))

        df = (
            df.filter(pl.col("Date").is_not_null() & pl.col("EMPI").is_not_null())
            .select(cols_select_exprs)
            .unique()
            .with_row_count("event_id")
        )

        events_df = df.select("event_id", "subject_id", "timestamp", "event_type")
        dynamic_measurements_df = df.select("event_id", "subject_id", "timestamp", "CodeWithType")
        
        print(f"Before processing: {dynamic_measurements_df.filter(pl.col('CodeWithType').is_null()).shape[0]} null CodeWithType values")
        
        # Ensure CodeWithType is not null and is a string
        dynamic_measurements_df = dynamic_measurements_df.with_columns([
            pl.col("CodeWithType").cast(pl.Utf8).alias("dynamic_indices")
        ])
        
        print(f"After processing: {dynamic_measurements_df.filter(pl.col('dynamic_indices').is_null()).shape[0]} null dynamic_indices values")
        print(f"Sample of dynamic_measurements_df:\n{dynamic_measurements_df.head()}")

        return events_df, dynamic_measurements_df

    @classmethod
    def _split_range_events_df(cls, df: DF_T) -> tuple[DF_T, DF_T, DF_T]:
        """Performs the following steps:

        1. Produces unified start and end timestamp columns representing the minimum of the passed start and
           end timestamps, respectively.
        2. Filters out records where the end timestamp is earlier than the start timestamp.
        3. Splits the dataframe into 3 events dataframes, all with only a single timestamp column, named
           `'timestamp'`:
           (a) An "EQ" dataframe, where start_ts_col == end_ts_col,
           (b) A "start" dataframe, with start events, and
           (c) An "end" dataframe, with end events.
        """

        df = df.filter(pl.col("start_time") <= pl.col("end_time"))

        eq_df = df.filter(pl.col("start_time") == pl.col("end_time"))
        ne_df = df.filter(pl.col("start_time") != pl.col("end_time"))

        st_col, end_col = pl.col("start_time").alias("timestamp"), pl.col("end_time").alias("timestamp")
        drop_cols = ["start_time", "end_time"]
        return (
            eq_df.with_columns(st_col).drop(drop_cols),
            ne_df.with_columns(st_col).drop(drop_cols),
            ne_df.with_columns(end_col).drop(drop_cols),
        )

    @classmethod
    def _inc_df_col(cls, df: DF_T, col: str, inc_by: int) -> DF_T:
        """Increments the values in a column by a given amount and returns a dataframe with the incremented
        column."""
        return df.with_columns(pl.col(col) + inc_by).collect(streaming=cls.STREAMING)

    @classmethod
    def _concat_dfs(cls, dfs: list[DF_T]) -> DF_T:
        """Concatenates a list of dataframes into a single dataframe."""
        return pl.concat(dfs, how="diagonal")

    @classmethod
    def _read_df(cls, fp: Path, **kwargs) -> DF_T:
        return pl.read_parquet(fp)

    @classmethod
    def _write_df(cls, df: DF_T, fp: Path, **kwargs):
        do_overwrite = kwargs.get("do_overwrite", False)

        if not do_overwrite and fp.is_file():
            raise FileExistsError(f"{fp} exists and do_overwrite is {do_overwrite}!")

        fp.parent.mkdir(exist_ok=True, parents=True)

        if isinstance(df, pl.LazyFrame):
            df.collect().write_parquet(fp, use_pyarrow=cls.WRITE_USE_PYARROW)
        else:
            df.write_parquet(fp, use_pyarrow=cls.WRITE_USE_PYARROW)

    def get_metadata_schema(self, config: MeasurementConfig) -> dict[str, pl.DataType]:
        schema = {
            "value_type": self.METADATA_SCHEMA["value_type"],
        }

        if self.config.outlier_detector_config is not None:
            M = self._get_preprocessing_model(self.config.outlier_detector_config, for_fit=False)
            schema["outlier_model"] = self.METADATA_SCHEMA["outlier_model"](M.params_schema())
        if self.config.normalizer_config is not None:
            M = self._get_preprocessing_model(self.config.normalizer_config, for_fit=False)
            schema["normalizer"] = self.METADATA_SCHEMA["normalizer"](M.params_schema())

        metadata = config.measurement_metadata
        if metadata is None:
            return schema

        for col in (
            "drop_upper_bound",
            "drop_lower_bound",
            "censor_upper_bound",
            "censor_lower_bound",
            "drop_upper_bound_inclusive",
            "drop_lower_bound_inclusive",
        ):
            if col in metadata:
                schema[col] = self.METADATA_SCHEMA[col]

        return schema

    @staticmethod
    def drop_or_censor(
        col: pl.Expr,
        drop_lower_bound: pl.Expr | None = None,
        drop_lower_bound_inclusive: pl.Expr | None = None,
        drop_upper_bound: pl.Expr | None = None,
        drop_upper_bound_inclusive: pl.Expr | None = None,
        censor_lower_bound: pl.Expr | None = None,
        censor_upper_bound: pl.Expr | None = None,
        **ignored_kwargs,
    ) -> pl.Expr:
        """Appropriately either drops (returns np.NaN) or censors (returns the censor value) the value `val`
        based on the bounds in `row`.

        TODO(mmd): could move this code to an outlier model in Preprocessing and have it be one that is
        pre-set in metadata.

        Args:
            val: The value to drop, censor, or return unchanged.
            drop_lower_bound: A lower bound such that if `val` is either below or at or below this level,
                `np.NaN` will be returned. If `None` or `np.NaN`, no bound will be applied.
            drop_lower_bound_inclusive: If `True`, returns `np.NaN` if ``val <= row['drop_lower_bound']``.
                Else, returns `np.NaN` if ``val < row['drop_lower_bound']``.
            drop_upper_bound: An upper bound such that if `val` is either above or at or above this level,
                `np.NaN` will be returned. If `None` or `np.NaN`, no bound will be applied.
            drop_upper_bound_inclusive: If `True`, returns `np.NaN` if ``val >= row['drop_upper_bound']``.
                Else, returns `np.NaN` if ``val > row['drop_upper_bound']``.
            censor_lower_bound: A lower bound such that if `val` is below this level but above
                `drop_lower_bound`, `censor_lower_bound` will be returned. If `None` or `np.NaN`, no bound
                will be applied.
            censor_upper_bound: An upper bound such that if `val` is above this level but below
                `drop_upper_bound`, `censor_upper_bound` will be returned. If `None` or `np.NaN`, no bound
                will be applied.
        """

        conditions = []

        if drop_lower_bound is not None:
            conditions.append(
                (
                    (col < drop_lower_bound) | ((col == drop_lower_bound) & drop_lower_bound_inclusive),
                    np.NaN,
                )
            )

        if drop_upper_bound is not None:
            conditions.append(
                (
                    (col > drop_upper_bound) | ((col == drop_upper_bound) & drop_upper_bound_inclusive),
                    np.NaN,
                )
            )

        if censor_lower_bound is not None:
            conditions.append((col < censor_lower_bound, censor_lower_bound))
        if censor_upper_bound is not None:
            conditions.append((col > censor_upper_bound, censor_upper_bound))

        if not conditions:
            return col

        expr = pl.when(conditions[0][0]).then(conditions[0][1])
        for cond, val in conditions[1:]:
            expr = expr.when(cond).then(val)
        return expr.otherwise(col)

    def _validate_id_col(self, id_col: Union[pl.DataFrame, pl.LazyFrame]) -> tuple[pl.Series, pl.datatypes.DataTypeClass]:
        is_lazy = isinstance(id_col, pl.LazyFrame)
        col_name = id_col.columns[0]

        # Check uniqueness
        if is_lazy:
            unique_count = id_col.select(pl.col(col_name)).unique().count().collect()[0, 0]
            total_count = id_col.select(pl.col(col_name)).count().collect()[0, 0]
        else:
            unique_count = id_col[col_name].n_unique()
            total_count = id_col.shape[0]
        
        if unique_count != total_count:
            raise ValueError(f"ID column {col_name} is not unique!")

        # Check data type and non-negativity
        dtype = id_col.dtypes[0]
        if dtype in (pl.Float32, pl.Float64):
            check_expr = (pl.col(col_name) == pl.col(col_name).round(0)) & (pl.col(col_name) >= 0)
        elif dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64):
            check_expr = pl.col(col_name) >= 0
        elif dtype in (pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
            check_expr = pl.lit(True)
        else:
            raise ValueError(f"ID column {col_name} is not a non-negative integer type!")

        if is_lazy:
            is_valid = id_col.select(check_expr.all()).collect()[0, 0]
        else:
            is_valid = id_col.select(check_expr.all()).item()

        if not is_valid:
            raise ValueError(f"ID column {col_name} is not a non-negative integer type!")

        # Get max value and determine smallest valid uint type
        if is_lazy:
            max_val = id_col.select(pl.col(col_name).max()).collect()[0, 0]
        else:
            max_val = id_col[col_name].max()

        dt = self.get_smallest_valid_uint_type(max_val)

        # Cast to the appropriate type and return as a Series
        if is_lazy:
            return id_col.select(pl.col(col_name).cast(dt)).collect()[col_name], dt
        else:
            return id_col.select(pl.col(col_name).cast(dt))[col_name], dt

    def _validate_initial_df(
        self,
        source_df: pl.DataFrame | pl.LazyFrame | None,
        id_col_name: str,
        valid_temporality_type: str,
        linked_id_cols: dict[str, pl.datatypes.DataType] | None = None,
    ) -> tuple[pl.DataFrame | pl.LazyFrame | None, pl.datatypes.DataType]:
        if source_df is None:
            return None, None

        if linked_id_cols:
            for id_col, id_col_dt in linked_id_cols.items():
                if id_col not in source_df.columns:
                    raise ValueError(f"Missing mandatory linkage col {id_col}")
                source_df = source_df.with_columns(pl.col(id_col).cast(id_col_dt))

        if id_col_name not in source_df.columns:
            source_df = source_df.with_row_count(name=id_col_name)

        id_col, id_col_dt = self._validate_id_col(source_df.select(id_col_name))

        # Use the name of the Series directly
        if id_col_name != id_col.name:
            source_df = source_df.rename({id_col.name: id_col_name})

        for col, cfg in self.config.measurement_configs.items():
            match cfg.modality:
                case DataModality.DROPPED:
                    continue
                case DataModality.UNIVARIATE_REGRESSION:
                    cat_col, val_col = None, col
                case DataModality.MULTIVARIATE_REGRESSION:
                    cat_col, val_col = col, cfg.values_column
                case _:
                    cat_col, val_col = col, None

            if cat_col is not None and cat_col in source_df.columns:
                if cfg.temporality != valid_temporality_type and cat_col != 'dynamic_indices':
                    raise ValueError(f"Column {cat_col} found in dataframe of wrong temporality")

                source_df = source_df.with_columns(pl.col(cat_col).cast(pl.Utf8).cast(pl.Categorical))

            if val_col is not None and val_col in source_df.columns:
                if cfg.temporality != valid_temporality_type and val_col != 'dynamic_indices':
                    raise ValueError(f"Column {val_col} found in dataframe of wrong temporality")

                if val_col == "SDI_score":  # Special case for SDI_score column
                    source_df = source_df.with_columns(pl.col(val_col).cast(pl.Float64))
                else:
                    source_df = source_df.with_columns(pl.col(val_col).cast(pl.Float64))

        return source_df, id_col_dt

    def _validate_initial_dfs(
        self,
        subjects_df: pl.DataFrame | pl.LazyFrame | None,
        events_df: pl.DataFrame | pl.LazyFrame | None,
        dynamic_measurements_df: pl.DataFrame | pl.LazyFrame | None,
    ) -> tuple[pl.DataFrame | pl.LazyFrame | None, pl.DataFrame | pl.LazyFrame | None, pl.DataFrame | pl.LazyFrame | None]:
        subjects_df, subjects_id_type = self._validate_initial_df(
            subjects_df, "subject_id", TemporalityType.STATIC
        )
        events_df, event_id_type = self._validate_initial_df(
            events_df,
            "event_id",
            TemporalityType.FUNCTIONAL_TIME_DEPENDENT,
            {"subject_id": subjects_id_type} if subjects_df is not None else None,
        )
        if events_df is not None:
            if "event_type" not in events_df.columns:
                raise ValueError("Missing event_type column!")
            events_df = events_df.with_columns(pl.col("event_type").cast(pl.Categorical))

            if "timestamp" not in events_df.columns or events_df.select("timestamp").dtypes[0] != pl.Datetime:
                raise ValueError("Malformed timestamp column!")

        if dynamic_measurements_df is not None:
            linked_ids = {}
            if events_df is not None:
                linked_ids["event_id"] = event_id_type

            dynamic_measurements_df, dynamic_measurement_id_types = self._validate_initial_df(
                dynamic_measurements_df, "measurement_id", TemporalityType.DYNAMIC, linked_ids
            )

            # Ensure 'dynamic_indices' is in dynamic_measurements_df
            if 'dynamic_indices' not in dynamic_measurements_df.columns:
                raise ValueError("'dynamic_indices' column not found in dynamic_measurements_df")

        return subjects_df, events_df, dynamic_measurements_df

    @TimeableMixin.TimeAs
    def _sort_events(self):
        self.events_df = self.events_df.sort("subject_id", "timestamp", descending=False)
        
    @TimeableMixin.TimeAs
    def _agg_by_time(self):
        event_id_dt = self.events_df["event_id"].dtype

        if self.config.agg_by_time_scale is None:
            grouped = self.events_df.filter(pl.col("timestamp").is_not_null()).groupby(["subject_id", "timestamp"], maintain_order=True)
        else:
            self.events_df = self.events_df.filter(pl.col("timestamp").is_not_null())
            grouped = self.events_df.sort(["subject_id", "timestamp"], descending=False).groupby_dynamic(
                "timestamp",
                every=self.config.agg_by_time_scale,
                truncate=True,
                closed="left",
                start_by="datapoint",
                by="subject_id",
            )

        grouped = (
            grouped.agg(
                pl.col("event_type").unique().sort(),
                pl.col("event_id").unique().alias("old_event_id"),
            )
            .sort("subject_id", "timestamp", descending=False)
            .with_row_count("event_id")
            .with_columns(
                pl.col("event_id").cast(event_id_dt),
                pl.col("event_type")
                .list.eval(pl.col("").cast(pl.Utf8))
                .list.join("&")
                .cast(pl.Categorical)
                .alias("event_type"),
            )
        )

        new_to_old_set = grouped[["event_id", "old_event_id"]].explode("old_event_id")

        self.events_df = grouped.drop("old_event_id")

        self.dynamic_measurements_df = (
            self.dynamic_measurements_df.rename({"event_id": "old_event_id"})
            .join(new_to_old_set, on="old_event_id", how="left")
            .drop("old_event_id")
        )

    def _update_subject_event_properties(self):
        if self.events_df is not None:
            self.event_types = (
                self.events_df.get_column("event_type")
                .value_counts(sort=True)
                .get_column("event_type")
                .to_list()
            )

            n_events_pd = self.events_df.get_column("subject_id").value_counts(sort=False).to_pandas()
            n_events_pd.columns = ['subject_id', 'counts']  # Rename the columns
            self.n_events_per_subject = dict(zip(n_events_pd['subject_id'], n_events_pd['counts']))
            self.subject_ids = set(self.n_events_per_subject.keys())

        if self.subjects_df is not None:
            subjects_with_no_events = (
                set(self.subjects_df.get_column("subject_id").to_list()) - self.subject_ids
            )
            for sid in subjects_with_no_events:
                self.n_events_per_subject[sid] = 0
            self.subject_ids.update(subjects_with_no_events)

    @classmethod
    def _filter_col_inclusion(cls, df: DF_T, col_inclusion_targets: dict[str, bool | Sequence[Any]]) -> DF_T:
        filter_exprs = []
        for col, incl_targets in col_inclusion_targets.items():
            match incl_targets:
                case True:
                    filter_exprs.append(pl.col(col).is_not_null())
                case False:
                    filter_exprs.append(pl.col(col).is_null())
                case _:
                    filter_exprs.append(pl.col(col).is_in(list(incl_targets)))

        return df.filter(pl.all_horizontal(filter_exprs))

    @TimeableMixin.TimeAs
    def _add_time_dependent_measurements(self):
        exprs = []
        join_cols = set()
        for col, cfg in self.config.measurement_configs.items():
            if cfg.temporality != TemporalityType.FUNCTIONAL_TIME_DEPENDENT:
                continue
            fn = cfg.functor
            join_cols.update(fn.link_static_cols)
            exprs.append(fn.pl_expr().alias(col))

        join_cols = list(join_cols)

        if join_cols:
            self.events_df = (
                self.events_df.join(self.subjects_df.select("subject_id", *join_cols), on="subject_id")
                .with_columns(exprs)
                .drop(join_cols)
            )
        else:
            self.events_df = self.events_df.with_columns(exprs)

    @TimeableMixin.TimeAs
    def _prep_numerical_source(
        self, measure: str, config: MeasurementConfig, source_df: DF_T
    ) -> tuple[DF_T, str, str, str, pl.DataFrame]:
        metadata = config.measurement_metadata

        print(f"Measure: {measure}")
        print(f"Config: {config}")
        print(f"Metadata: {metadata}")

        metadata_schema = self.get_metadata_schema(config)

        match config.modality:
            case DataModality.UNIVARIATE_REGRESSION:
                key_col = "const_key"
                val_col = measure
                print(f"Univariate Regression - Key column: {key_col}, Value column: {val_col}")
                metadata_as_polars = pl.DataFrame(
                    {key_col: [measure], **{c: [v] for c, v in metadata.items()}}
                )
                source_df = source_df.with_columns(pl.lit(measure).cast(pl.Categorical).alias(key_col))
            case DataModality.MULTIVARIATE_REGRESSION:
                key_col = measure
                val_col = config.values_column
                print(f"Multivariate Regression - Key column: {key_col}, Value column: {val_col}")
                metadata_as_polars = pl.DataFrame(
                    {key_col: [measure], **{c: [v] for c, v in metadata.items() if c != key_col}}
                )
            case _:
                raise ValueError(f"Called _pre_numerical_source on {config.modality} measure {measure}!")

        print(f"Metadata as Polars DataFrame: {metadata_as_polars}")
        print(f"Source DataFrame schema: {source_df.schema}")

        if "outlier_model" in metadata_as_polars and len(metadata_as_polars.drop_nulls("outlier_model")) == 0:
            metadata_as_polars = metadata_as_polars.with_columns(pl.lit(None).alias("outlier_model"))
        if "normalizer" in metadata_as_polars and len(metadata_as_polars.drop_nulls("normalizer")) == 0:
            metadata_as_polars = metadata_as_polars.with_columns(pl.lit(None).alias("normalizer"))

        if val_col not in metadata_as_polars.columns:
            metadata_as_polars = metadata_as_polars.with_columns(pl.lit(None).alias(val_col))

        metadata_as_polars = metadata_as_polars.with_columns(
            pl.col(key_col).cast(pl.Categorical),
            pl.col(val_col).cast(pl.Float64).alias("value"),  # Rename the values column to 'value'
            **{k: pl.col(k).cast(v) for k, v in metadata_schema.items()},
        )

        source_df = source_df.join(metadata_as_polars, on=key_col, how="left")
        return source_df, key_col, val_col, f"{measure}_is_inlier", metadata_as_polars

    def _total_possible_and_observed(
        self, measure: str, config: MeasurementConfig, source_df: DF_T
    ) -> tuple[int, int, int]:
        agg_by_col = pl.col("event_id") if config.temporality == TemporalityType.DYNAMIC else None

        if agg_by_col is None:
            num_possible = len(source_df)
            num_non_null = len(source_df.drop_nulls(measure))
            num_total = num_non_null
        else:
            num_possible = source_df.select(pl.col("event_id").n_unique()).item()
            num_non_null = source_df.select(
                pl.col("event_id").filter(pl.col(measure).is_not_null()).n_unique()
            ).item()
            num_total = source_df.select(pl.col(measure).is_not_null().sum()).item()
        return num_possible, num_non_null, num_total

    @TimeableMixin.TimeAs
    def _add_inferred_val_types(
        self,
        measurement_metadata: DF_T,
        source_df: DF_T,
        vocab_keys_col: str,
        vals_col: str,
    ) -> DF_T:
        """Infers the appropriate type of the passed metadata column values. Performs the following
        steps:

        1. Determines if the column should be dropped for having too few measurements.
        2. Determines if the column actually contains integral, not floating point values.
        3. Determines if the column should be partially or fully re-categorized as a categorical column.

        Args:
            measurement_metadata: The metadata (pre-set or to-be-fit pre-processing parameters) for the
                numerical measure in question.
            source_df: The governing source dataframe for this measurement.
            vocab_keys_col: The column containing the "keys" for this measure. If it is a multivariate
                regression measure, this column will be the column that indicates to which covariate the value
                in the values column corresponds. If it is a univariate regression measure, this column will
                be an artificial column containing a constant key.
            vals_col: The column containing the numerical values to be assessed.


        Returns: The appropriate `NumericDataModalitySubtype` for the values.
        """

        vals_col_expr = pl.col(vals_col)

        if "value_type" in measurement_metadata:
            missing_val_types = measurement_metadata.filter(pl.col("value_type").is_null())[vocab_keys_col]
            for_val_type_inference = source_df.filter(
                (~pl.col(vocab_keys_col).is_in(measurement_metadata[vocab_keys_col]))
                | pl.col(vocab_keys_col).is_in(missing_val_types)
            )
        else:
            for_val_type_inference = source_df

          # a. Convert to integeres where appropriate.
        if self.config.min_true_float_frequency is not None:
            if source_df.select(vals_col_expr).dtypes[0] in (pl.Int8, pl.Int16, pl.Int32, pl.Int64):
                is_int_expr = pl.lit(True).alias("is_int")
                for_val_type_inference = for_val_type_inference.with_columns(vals_col_expr.cast(pl.Int64).alias(vals_col))
            else:
                is_int_expr = (
                    ((vals_col_expr == vals_col_expr.round(0)).mean() > (1 - self.config.min_true_float_frequency))
                    .cast(pl.Boolean)
                    .alias("is_int")
                )
            int_keys = for_val_type_inference.groupby(vocab_keys_col).agg(is_int_expr)

            # Use a different suffix to avoid column name conflict
            measurement_metadata = measurement_metadata.join(int_keys, on=vocab_keys_col, how="outer", suffix="_int")

            key_is_int = pl.col(vocab_keys_col).is_in(int_keys.filter("is_int")[vocab_keys_col])
            for_val_type_inference = for_val_type_inference.with_columns(
                pl.when(key_is_int & (source_df.select(vals_col_expr).dtypes[0] != pl.Int64)).then(vals_col_expr.cast(pl.Int64)).otherwise(vals_col_expr)
            )
        else:
            measurement_metadata = measurement_metadata.with_columns(pl.lit(False).alias("is_int"))

        # b. Drop if only has a single observed numerical value.
        dropped_keys = (
            for_val_type_inference.groupby(vocab_keys_col)
            .agg((vals_col_expr.n_unique() == 1).cast(pl.Boolean).alias("should_drop"))
            .filter("should_drop")
        )
        keep_key_expr = ~pl.col(vocab_keys_col).is_in(dropped_keys[vocab_keys_col])
        measurement_metadata = measurement_metadata.with_columns(
            pl.when(keep_key_expr)
            .then(pl.col("value_type"))
            .otherwise(pl.lit(NumericDataModalitySubtype.DROPPED))
            .alias("value_type")
        )
        for_val_type_inference = for_val_type_inference.filter(keep_key_expr)

        # c. Convert to categorical if too few unique observations are seen.
        if self.config.min_unique_numerical_observations is not None:
            is_cat_expr = (
                lt_count_or_proportion(
                    vals_col_expr.n_unique(),
                    self.config.min_unique_numerical_observations,
                    vals_col_expr.len(),
                )
                .cast(pl.Boolean)
                .alias("is_categorical")
            )

            categorical_keys = for_val_type_inference.groupby(vocab_keys_col).agg(is_cat_expr)

            # Use a different suffix to avoid column name conflict
            measurement_metadata = measurement_metadata.join(categorical_keys, on=vocab_keys_col, how="outer", suffix="_cat")
        else:
            measurement_metadata = measurement_metadata.with_columns(pl.lit(False).alias("is_categorical"))

        inferred_value_type = (
            pl.when(pl.col("is_int") & pl.col("is_categorical"))
            .then(pl.lit(NumericDataModalitySubtype.CATEGORICAL_INTEGER))
            .when(pl.col("is_categorical"))
            .then(pl.lit(NumericDataModalitySubtype.CATEGORICAL_FLOAT))
            .when(pl.col("is_int"))
            .then(pl.lit(NumericDataModalitySubtype.INTEGER))
            .otherwise(pl.lit(NumericDataModalitySubtype.FLOAT))
        )

        return measurement_metadata.with_columns(
            pl.coalesce(["value_type", inferred_value_type]).alias("value_type")
        ).drop(["is_int", "is_categorical"])
        
    @TimeableMixin.TimeAs
    def _fit_measurement_metadata(
        self, measure: str, config: MeasurementConfig, source_df: DF_T
    ) -> pd.DataFrame:
        orig_source_df = source_df.clone()

        source_df, vocab_keys_col, vals_col, _, measurement_metadata = self._prep_numerical_source(
            measure, config, source_df
        )

        # Calculate total possible and observed instances before dropping the 'Result' column
        if self.config.min_valid_vocab_element_observations is not None:
            if config.temporality == TemporalityType.DYNAMIC:
                num_possible = orig_source_df.select(pl.col("event_id").n_unique()).item()
                num_non_null = orig_source_df.select(
                    pl.col("event_id").filter(pl.col(vocab_keys_col).is_not_null()).n_unique()
                ).item()
            else:
                num_possible = len(orig_source_df)
                num_non_null = orig_source_df.select(pl.col(measure).drop_nulls().len()).item()

        # Drop the 'Result' column after calculating total possible and observed instances
        source_df = source_df.drop_nulls([vocab_keys_col, vals_col]).filter(pl.col(vals_col).is_not_nan())

        # Check if measurement metadata is missing for categorical variables
        if config.measurement_metadata is None:
            if config.modality in (DataModality.SINGLE_LABEL_CLASSIFICATION, DataModality.MULTI_LABEL_CLASSIFICATION):
                # Populate metadata for categorical variables
                config.measurement_metadata = pd.DataFrame({"value_type": [NumericDataModalitySubtype.CATEGORICAL]}, index=[measure])
            else:
                raise ValueError(f"Measurement metadata is missing for measure {measure} with modality {config.modality}")

        # 1. Determines which vocab elements should be dropped due to insufficient occurrences.
        if self.config.min_valid_vocab_element_observations is not None:
            should_drop_expr = pl.lit(lt_count_or_proportion(
                num_non_null, self.config.min_valid_vocab_element_observations, num_possible
            )).cast(pl.Boolean)

            dropped_keys = (
                source_df.groupby(vocab_keys_col)
                .agg(should_drop_expr.alias("should_drop"))
                .filter("should_drop")
                .with_columns(pl.lit(NumericDataModalitySubtype.DROPPED).alias("value_type"))
                .drop("should_drop")
            )

            measurement_metadata = (
                measurement_metadata.join(
                    dropped_keys,
                    on=vocab_keys_col,
                    how="outer",
                    suffix="_right",
                )
                .with_columns(pl.coalesce(["value_type", "value_type_right"]).alias("value_type"))
                .drop("value_type_right")
            )
            source_df = source_df.filter(~pl.col(vocab_keys_col).is_in(dropped_keys[vocab_keys_col]))

            if measure == 'SDI_score':
                source_df = source_df.fill_null(0)

            if len(source_df) == 0:
                measurement_metadata = measurement_metadata.to_pandas()
                measurement_metadata = measurement_metadata.set_index(vocab_keys_col)

                if config.modality == DataModality.UNIVARIATE_REGRESSION:
                    assert len(measurement_metadata) == 1
                    return measurement_metadata.loc[measure]
                else:
                    return measurement_metadata

        # 2. Eliminates hard outliers and performs censoring via specified config.
        bound_cols = {}
        for col in (
            "drop_upper_bound",
            "drop_upper_bound_inclusive",
            "drop_lower_bound",
            "drop_lower_bound_inclusive",
            "censor_lower_bound",
            "censor_upper_bound",
        ):
            if col in source_df:
                bound_cols[col] = pl.col(col)

        if bound_cols:
            source_df = source_df.with_columns(
                self.drop_or_censor(pl.col(vals_col), **bound_cols).alias(vals_col)
            )

        source_df = source_df.filter(pl.col(vals_col).is_not_nan())
        if len(source_df) == 0:
            return config.measurement_metadata

        # 3. Infer the value type and convert where necessary.
        measurement_metadata = self._add_inferred_val_types(
            measurement_metadata, source_df, vocab_keys_col, vals_col
        )

        source_df = (
            source_df.update(measurement_metadata.select(vocab_keys_col, "value_type"), on=vocab_keys_col)
            .with_columns(
                pl.when(pl.col("value_type") == NumericDataModalitySubtype.INTEGER)
                .then(pl.col(vals_col).cast(pl.Float64).round(0))  # Cast to float before rounding
                .when(pl.col("value_type") == NumericDataModalitySubtype.FLOAT)
                .then(pl.col(vals_col))
                .otherwise(None)
                .alias(vals_col)
            )
            .drop_nulls(vals_col)
            .filter(pl.col(vals_col).is_not_nan())
        )

        # 4. Infer outlier detector and normalizer parameters.
        if self.config.outlier_detector_config is not None:
            with self._time_as("fit_outlier_detector"):
                M = self._get_preprocessing_model(self.config.outlier_detector_config, for_fit=True)
                outlier_model_params = source_df.groupby(vocab_keys_col).agg(
                    M.fit_from_polars(pl.col(vals_col)).alias("outlier_model")
                )

                measurement_metadata = measurement_metadata.with_columns(
                    pl.col("outlier_model").cast(outlier_model_params["outlier_model"].dtype)
                )
                source_df = source_df.with_columns(
                    pl.col("outlier_model").cast(outlier_model_params["outlier_model"].dtype)
                )

                measurement_metadata = measurement_metadata.update(outlier_model_params, on=vocab_keys_col)
                source_df = source_df.update(
                    measurement_metadata.select(vocab_keys_col, "outlier_model"), on=vocab_keys_col
                )

                is_inlier = ~M.predict_from_polars(pl.col(vals_col), pl.col("outlier_model"))
                source_df = source_df.filter(is_inlier)

        # 5. Fit a normalizer model.
        if self.config.normalizer_config is not None:
            with self._time_as("fit_normalizer"):
                M = self._get_preprocessing_model(self.config.normalizer_config, for_fit=True)
                normalizer_params = source_df.groupby(vocab_keys_col).agg(
                    M.fit_from_polars(pl.col(vals_col)).alias("normalizer")
                )
                measurement_metadata = measurement_metadata.with_columns(
                    pl.col("normalizer").cast(normalizer_params["normalizer"].dtype)
                )
                measurement_metadata = measurement_metadata.update(normalizer_params, on=vocab_keys_col)

        # 6. Convert to the appropriate type and return.
        measurement_metadata = measurement_metadata.to_pandas()
        measurement_metadata = measurement_metadata.set_index(vocab_keys_col)

        if config.modality == DataModality.UNIVARIATE_REGRESSION:
            assert len(measurement_metadata) == 1
            return measurement_metadata.loc[measure]
        else:
            return measurement_metadata

    def _create_code_mapping(self):
        # Collect all unique codes from dynamic_measurements_df
        all_codes = set(self.dynamic_measurements_df['CodeWithType'].unique())
        # Create a mapping from codes to indices, starting from 1
        return {code: idx for idx, code in enumerate(sorted(all_codes), start=1)}

    def preprocess(self):
        print("Starting preprocessing...")
        self._filter_subjects()
        print("Finished filtering subjects.")
        self._add_time_dependent_measurements()
        print("Finished adding time-dependent measurements.")
        
        self._create_code_mapping()
        self._convert_dynamic_indices_to_indices()

        self.inverse_mapping = {idx: code for code, idx in self.code_mapping.items()}
        
        self.fit_measurements()
        print("Finished fitting measurements.")
        self.transform_measurements()
        print("Finished transforming measurements.")
        print("Preprocessing completed successfully.")

    def _convert_dynamic_indices_to_indices(self):
        self.dynamic_measurements_df = self.dynamic_measurements_df.with_columns([
            pl.col('CodeWithType').map_dict(self.code_mapping).alias('dynamic_indices')
        ])

    @TimeableMixin.TimeAs
    def _fit_vocabulary(self, measure: str, config: MeasurementConfig, source_df: DF_T) -> Vocabulary:
        if measure == 'dynamic_indices':
            print(f"Fitting vocabulary for {measure}")
            print(f"Source dataframe shape: {source_df.shape}")
            print(f"Source dataframe columns: {source_df.columns}")
            print(f"Sample of source dataframe:\n{source_df.head()}")
            
            # Ensure 'dynamic_indices' column exists
            if 'dynamic_indices' not in source_df.columns:
                raise ValueError("'dynamic_indices' column not found in source dataframe")
            
            # Convert to polars DataFrame if it's a LazyFrame
            if isinstance(source_df, pl.LazyFrame):
                source_df = source_df.collect()
            
            # Get unique codes from the dynamic_indices column
            unique_codes = source_df['dynamic_indices'].unique().sort()
            
            # Create vocab_elements and el_counts
            vocab_elements = unique_codes.to_list()
            el_counts = source_df.groupby('dynamic_indices').count().sort('dynamic_indices')['count'].to_list()
            
            print(f"Number of unique codes: {len(vocab_elements)}")
            print(f"Number of element counts: {len(el_counts)}")
            print(f"Sample of vocab_elements: {vocab_elements[:5]}")
            print(f"Sample of el_counts: {el_counts[:5]}")
            
            return Vocabulary(vocabulary=vocab_elements, obs_frequencies=el_counts)
        
        elif config.modality == DataModality.MULTI_LABEL_CLASSIFICATION:
            observations = source_df.get_column(measure).cast(pl.Utf8)
            observations = observations.apply(lambda s: s.split("|") if s is not None else [], return_dtype=pl.List(pl.Utf8))
            observations = observations.explode()
        else:
            observations = source_df.get_column(measure)

        observations = observations.drop_nulls()
        N = len(observations)
        if N == 0:
            return None

        try:
            value_counts = observations.value_counts().sort(by="count", descending=True)
            vocab_elements = value_counts[measure].to_list()
            el_counts = value_counts["count"].to_list()
            return Vocabulary(vocabulary=vocab_elements, obs_frequencies=el_counts)
        except AssertionError as e:
            raise AssertionError(f"Failed to build vocabulary for {measure}") from e
    
        if config.modality == DataModality.MULTIVARIATE_REGRESSION:
            val_types = pl.from_pandas(
                config.measurement_metadata[["value_type"]], include_index=True
            ).with_columns(
                pl.col("value_type").cast(pl.Categorical), pl.col(measure).cast(pl.Categorical)
            )
            observations = (
                source_df.join(val_types, on=measure)
                .with_columns(
                    pl.when(pl.col("value_type") == NumericDataModalitySubtype.CATEGORICAL_INTEGER)
                    .then(
                        pl.col(measure).cast(pl.Utf8)
                        + "__EQ_"
                        + pl.col(config.values_column).round(0).cast(int).cast(pl.Utf8)
                    )
                    .when(pl.col("value_type") == NumericDataModalitySubtype.CATEGORICAL_FLOAT)
                    .then(
                        pl.col(measure).cast(pl.Utf8)
                        + "__EQ_"
                        + pl.col(config.values_column).cast(pl.Utf8)
                    )
                    .otherwise(pl.col(measure))
                    .alias(measure)
                )
                .get_column(measure)
            )
        elif config.modality == DataModality.UNIVARIATE_REGRESSION:
            match config.measurement_metadata.value_type:
                case NumericDataModalitySubtype.CATEGORICAL_INTEGER:
                    observations = source_df.with_columns(
                        (f"{measure}__EQ_" + pl.col(measure).round(0).cast(int).cast(pl.Utf8)).alias(
                            measure
                        )
                    ).get_column(measure)
                case NumericDataModalitySubtype.CATEGORICAL_FLOAT:
                    observations = source_df.with_columns(
                        (f"{measure}__EQ_" + pl.col(measure).cast(pl.Utf8)).alias(measure)
                    ).get_column(measure)
                case _:
                    return
        elif config.modality == DataModality.MULTI_LABEL_CLASSIFICATION:
            observations = source_df.get_column(measure).cast(pl.Utf8)
            observations = observations.apply(lambda s: s.split("|") if s is not None else [], return_dtype=pl.List(pl.Utf8))
            observations = observations.explode()
        else:
            observations = source_df.get_column(measure)

        observations = observations.drop_nulls()
        N = len(observations)
        if N == 0:
            return

        try:
            unique_values = observations.unique().sort()
            vocab_elements = unique_values.to_list()
            
            # Use 'count' instead of 'counts' and handle potential column name differences
            value_counts = observations.value_counts().sort(by=measure)
            count_column = 'count' if 'count' in value_counts.columns else 'counts'
            if count_column not in value_counts.columns:
                print(f"Warning: Count column not found. Available columns: {value_counts.columns}")
                el_counts = [1] * len(vocab_elements)  # Fallback to assigning count of 1 to each element
            else:
                el_counts = value_counts[count_column].to_list()
            
            return Vocabulary(vocabulary=vocab_elements, obs_frequencies=el_counts)
        except AssertionError as e:
            raise AssertionError(f"Failed to build vocabulary for {measure}") from e

    @TimeableMixin.TimeAs
    def _transform_multi_label_classification(self, measure, config, source_df):
        if measure == 'dynamic_indices':
            # Ensure no null or empty values
            source_df = source_df.filter(
                (pl.col('dynamic_indices').is_not_null()) & (pl.col('dynamic_indices') != '')
            )
            
            # Create a list column of dynamic_indices
            source_df = source_df.with_columns(
                pl.col('dynamic_indices').apply(lambda x: [x]).alias('dynamic_indices_list')
            )
            
            # Count the occurrences
            source_df = source_df.with_columns(
                pl.col('dynamic_indices_list').apply(len).alias('dynamic_counts')
            )
            
            return source_df
               
            # Convert the dynamic_indices column to string type
            source_df = source_df.with_columns(pl.col(measure).cast(str))

            # Split the dynamic_indices column into a list
            source_df = source_df.with_columns(
                pl.col(measure).map_elements(lambda x: [x], return_dtype=pl.List(pl.Utf8)).alias(f"{measure}_list")
            )

            # Explode the dynamic_indices_list column and count the occurrences
            transformed_df = source_df.with_columns(
                pl.col(f"{measure}_list").explode().alias(measure)
            ).with_columns(
                pl.col(f"{measure}_list").map_elements(len, return_dtype=pl.UInt32).alias(f"{measure}_counts")
            )

            return transformed_df
     
    @TimeableMixin.TimeAs
    def _transform_numerical_measurement(
        self, measure: str, config: MeasurementConfig, source_df: DF_T
    ) -> DF_T:
        source_df, keys_col_name, vals_col_name, inliers_col_name, _ = self._prep_numerical_source(
            measure, config, source_df
        )

        keys_col = pl.col(keys_col_name)
        vals_col = pl.col(vals_col_name)

        cols_to_drop_at_end = []
        for col in config.measurement_metadata:
            if col != measure and col in source_df:
                cols_to_drop_at_end.append(col)

        bound_cols = {}
        for col in (
            "drop_upper_bound",
            "drop_upper_bound_inclusive",
            "drop_lower_bound",
            "drop_lower_bound_inclusive",
            "censor_lower_bound",
            "censor_upper_bound",
        ):
            if col in source_df:
                bound_cols[col] = pl.col(col)

        if bound_cols:
            vals_col = self.drop_or_censor(vals_col, **bound_cols)

        value_type = pl.col("value_type")
        keys_col = (
            pl.when(value_type == NumericDataModalitySubtype.DROPPED)
            .then(keys_col)
            .when(value_type == NumericDataModalitySubtype.CATEGORICAL_INTEGER)
            .then(keys_col + "__EQ_" + vals_col.fill_nan(0).cast(pl.Int64).cast(pl.Utf8))
            .when(value_type == NumericDataModalitySubtype.CATEGORICAL_FLOAT)
            .then(keys_col + "__EQ_" + vals_col.fill_nan(0).cast(pl.Utf8))
            .otherwise(keys_col)
            .alias(keys_col_name)
        )

        vals_col = (
            pl.when(
                value_type.is_in(
                    [
                        NumericDataModalitySubtype.DROPPED,
                        NumericDataModalitySubtype.CATEGORICAL_INTEGER,
                        NumericDataModalitySubtype.CATEGORICAL_FLOAT,
                    ]
                )
            )
            .then(pl.lit(None))
            .otherwise(vals_col.fill_nan(0))
            .alias(vals_col_name)
        )

        null_idx = keys_col.is_null() | vals_col.is_null()

        source_df = source_df.with_columns(
            pl.when(null_idx)
            .then(pl.lit(None))
            .otherwise(pl.struct([keys_col, vals_col]))
            .alias("temp")
        )

        source_df = source_df.select(
            pl.all().exclude([keys_col_name, vals_col_name, "temp"]),
            pl.col("temp").struct.field(keys_col_name).alias(keys_col_name),
            pl.col("temp").struct.field(vals_col_name).cast(pl.Float32).alias(vals_col_name),
        )

        null_source = source_df.filter(null_idx)
        present_source = source_df.filter(~null_idx)

        if len(present_source) == 0:
            if self.config.outlier_detector_config is not None:
                null_source = null_source.with_columns(pl.lit(None).cast(pl.Boolean).alias(inliers_col_name))
            return null_source.drop(cols_to_drop_at_end)

        if self.config.outlier_detector_config is not None:
            M = self._get_preprocessing_model(self.config.outlier_detector_config, for_fit=False)

            inliers_col = ~M.predict_from_polars(vals_col, pl.col("outlier_model")).alias(inliers_col_name)
            vals_col = pl.when(inliers_col).then(vals_col).otherwise(0)

            present_source = present_source.with_columns(inliers_col, vals_col)
            null_source = null_source.with_columns(pl.lit(None).cast(pl.Boolean).alias(inliers_col_name))

            new_nulls = present_source.filter(~pl.col(inliers_col_name))
            null_source = null_source.vstack(new_nulls)
            present_source = present_source.filter(inliers_col_name)

        if len(present_source) == 0:
            return null_source.drop(cols_to_drop_at_end)

        if self.config.normalizer_config is not None:
            M = self._get_preprocessing_model(self.config.normalizer_config, for_fit=False)

            vals_col = M.predict_from_polars(vals_col, pl.col("normalizer"))
            present_source = present_source.with_columns(vals_col)

        present_source = present_source.with_columns(vals_col.cast(pl.Float32))
        null_source = null_source.with_columns(vals_col.cast(pl.Float32))
        source_df = present_source.vstack(null_source)

        return source_df.drop(cols_to_drop_at_end)

    @TimeableMixin.TimeAs
    def _transform_categorical_measurement(
        self, measure: str, config: MeasurementConfig, source_df: DF_T
    ) -> DF_T:
        print(f"Transforming categorical measurement: {measure}")
        if measure not in source_df.columns:
            print(f"Warning: Measure {measure} not found in the source DataFrame.")
            return source_df
        if config.vocabulary is None:
            print(f"Warning: Vocabulary is None for measure {measure}. Skipping transformation.")
            return source_df

        vocab_el_col = pl.col(measure)
        vocab_lit = pl.Series(config.vocabulary.vocabulary).cast(pl.Categorical)

        if measure == 'dynamic_indices':
            # For dynamic_indices, we don't need to transform it as it's already in the correct format
            transform_expr = [vocab_el_col.alias(measure)]
        else:
            transform_expr = [
                pl.when(vocab_el_col.is_null())
                .then(None)
                .when(~vocab_el_col.is_in(vocab_lit))
                .then(None)
                .otherwise(vocab_el_col)
                .cast(pl.Categorical)
                .alias(measure)
            ]

        return source_df.with_columns(transform_expr)

    @TimeableMixin.TimeAs
    def _update_attr_df(self, attr: str, id_col: str, df: DF_T, cols_to_update: list[str]):
        old_df = getattr(self, attr)

        old_df = old_df.with_columns(**{c: pl.lit(None).cast(df[c].dtype) for c in cols_to_update})
        new_df = df.select(id_col, *cols_to_update)

        setattr(self, attr, old_df.update(new_df, on=id_col))

    def _melt_df(self, source_df: DF_T, id_cols: Sequence[str], measures: list[str]) -> pl.Expr:
        exprs = []
        value_var_names = []
        total_vocab_size = self.vocabulary_config.total_vocab_size
        idx_dt = self.get_smallest_valid_uint_type(total_vocab_size)

        for m in measures:
            if m not in source_df.columns:
                continue

            if m in self.measurement_vocabs:
                idx_present_expr = (pl.col(m).is_not_null()).alias(f"{m}_present")
                idx_value_expr = pl.col(m).map_dict(self.unified_vocabulary_idxmap[m], default=0).cast(idx_dt).alias(f"{m}_index")
            else:
                idx_present_expr = pl.col(m).is_not_null().alias(f"{m}_present")
                idx_value_expr = pl.lit(self.unified_vocabulary_idxmap[m][m]).cast(idx_dt).alias(f"{m}_index")

            exprs.extend([idx_present_expr, idx_value_expr])
            value_var_names.extend([f"{m}_present", f"{m}_index"])

        measurements_idx_dt = self.get_smallest_valid_uint_type(len(self.unified_measurements_idxmap))

        melted_df = (
            source_df.select(*id_cols, *value_var_names)
            .melt(
                id_vars=id_cols,
                value_vars=value_var_names,
                variable_name="variable",
                value_name="value",
            )
            .with_columns(
                pl.col("variable").str.replace_all(r"_present$|_index$", "").alias("measurement"),
                pl.col("variable").str.contains("_present$").alias("is_present"),
                pl.col("variable").str.contains("_index$").alias("is_index"),
            )
        )

        melted_df_present = (
            melted_df
            .filter(pl.col("is_present"))
            .with_columns(
                pl.col("measurement").map_dict(self.unified_measurements_idxmap).cast(measurements_idx_dt).alias("measurement_index"),
                pl.col("value").cast(pl.Boolean).alias("present"),
            )
        )

        melted_df_index = melted_df.filter(pl.col("is_index")).select(*id_cols, "measurement", pl.col("value").alias("index"))

        melted_df = (
            melted_df_present
            .join(melted_df_index, on=id_cols + ["measurement"], how="left")
            .filter(pl.col("present"))
            .select(*id_cols, "measurement_index", "index")
        )

        return melted_df

    def build_DL_cached_representation(
        self, subject_ids: list[int] | None = None, do_sort_outputs: bool = False
    ) -> DF_T:
        print("Starting build_DL_cached_representation")
        subject_measures, event_measures, dynamic_measures = [], ["event_type"], ["dynamic_indices"]
        for m in self.unified_measurements_vocab[1:]:
            temporality = self.measurement_configs[m].temporality
            match temporality:
                case TemporalityType.STATIC:
                    subject_measures.append(m)
                case TemporalityType.FUNCTIONAL_TIME_DEPENDENT:
                    event_measures.append(m)
                case TemporalityType.DYNAMIC:
                    dynamic_measures.append(m)
                case _:
                    raise ValueError(f"Unknown temporality type {temporality} for {m}")

        if subject_ids:
            subjects_df = self._filter_col_inclusion(self.subjects_df, {"subject_id": subject_ids})
            events_df = self._filter_col_inclusion(self.events_df, {"subject_id": subject_ids})
            dynamic_measurements_df = self._filter_col_inclusion(self.dynamic_measurements_df, {"subject_id": subject_ids})
        else:
            subjects_df = self.subjects_df
            events_df = self.events_df
            dynamic_measurements_df = self.dynamic_measurements_df

        print(f"Subjects DataFrame shape: {subjects_df.shape}")
        print(f"Events DataFrame shape: {events_df.shape}")
        print(f"Dynamic Measurements DataFrame shape: {dynamic_measurements_df.shape}")

        static_data = subjects_df.select(
            "subject_id",
            *[pl.col(m).alias(f"static_indices_{m}") for m in subject_measures],
            *[(pl.col(m).is_null().cast(pl.Int32) == 0).alias(f"static_counts_{m}") for m in subject_measures],
            # Add these new columns
            "InitialA1c", "Female", "Married", "GovIns", 
            "English", "AgeYears", "SDI_score", "Veteran"
        )

        print("Checking dynamic_measurements_df for null or invalid values")
        null_count = dynamic_measurements_df.filter(pl.col('dynamic_indices').is_null()).shape[0]
        invalid_count = dynamic_measurements_df.filter(pl.col('dynamic_indices') == '').shape[0]
        print(f"Null values in dynamic_indices: {null_count}")
        print(f"Empty string values in dynamic_indices: {invalid_count}")

        dynamic_data = dynamic_measurements_df.select(
            "event_id",
            pl.col("dynamic_indices"),
            pl.lit(1).alias("dynamic_counts")  # Set all counts to 1
        )

        event_data = events_df.select(
            "subject_id",
            "timestamp",
            "event_id",
            *[pl.col(m).alias(f"{m}_index") for m in event_measures]
        ).join(
            dynamic_data,
            on="event_id",
            how="left"
        )

        print("Checking event_data for null or invalid values")
        null_count = event_data.filter(pl.col('dynamic_indices').is_null()).shape[0]
        invalid_count = event_data.filter(pl.col('dynamic_indices') == '').shape[0]
        print(f"Null values in dynamic_indices: {null_count}")
        print(f"Empty string values in dynamic_indices: {invalid_count}")

        # Remove rows with null or empty dynamic_indices
        event_data = event_data.filter(
            pl.col("dynamic_indices").is_not_null() & (pl.col("dynamic_indices") != "")
        )

        if do_sort_outputs:
            event_data = event_data.sort("event_id")

        out = static_data.join(event_data, on="subject_id", how="outer")

        if do_sort_outputs:
            out = out.sort("subject_id")

        print("Final check for UNKNOWN_CODE")
        unknown_count = out.filter(pl.col('dynamic_indices') == 'UNKNOWN_CODE').shape[0]
        print(f"UNKNOWN_CODE count in final output: {unknown_count}")

        print(f"Final output shape: {out.shape}")
        print(f"Sample of final output:\n{out.head()}")

        return out

    @staticmethod
    def _parse_flat_feature_column(c: str) -> tuple[str, str, str, str]:
        parts = c.split("/")
        if len(parts) < 4:
            raise ValueError(f"Column {c} is not a valid flat feature column!")
        return (parts[0], parts[1], "/".join(parts[2:-1]), parts[-1])

    def _summarize_static_measurements(
        self,
        feature_columns: list[str],
        include_only_subjects: set[int] | None = None,
    ) -> pl.LazyFrame:
        if include_only_subjects is None:
            df = self.subjects_df
        else:
            df = self.subjects_df.filter(pl.col("subject_id").is_in(list(include_only_subjects)))

        valid_measures = {}
        for feat_col in feature_columns:
            temp, meas, feat, _ = self._parse_flat_feature_column(feat_col)

            if temp != "static":
                continue

            if meas not in valid_measures:
                valid_measures[meas] = set()
            valid_measures[meas].add(feat)

        out_dfs = {}
        for m, allowed_vocab in valid_measures.items():
            cfg = self.measurement_configs[m]

            if cfg.modality == "univariate_regression" and cfg.vocabulary is None:
                if allowed_vocab != {m}:
                    raise ValueError(
                        f"Encountered a measure {m} with no vocab but a pre-set feature vocab of "
                        f"{allowed_vocab}"
                    )
                out_dfs[m] = (
                    df.lazy()
                    .filter(pl.col(m).is_not_null())
                    .select("subject_id", pl.col(m).alias(f"static/{m}/{m}/value").cast(pl.Float32))
                )
                continue
            elif cfg.modality == "multivariate_regression":
                raise ValueError(f"{cfg.modality} is not supported for {cfg.temporality} measures.")

            ID_cols = ["subject_id"]
            pivoted_df = (
                df.select(*ID_cols, m)
                .filter(pl.col(m).is_in(list(allowed_vocab)))
                .with_columns(pl.lit(True).alias("__indicator"))
                .pivot(
                    index=ID_cols,
                    columns=m,
                    values="__indicator",
                    aggregate_function=None,
                )
            )

            remap_cols = [c for c in pivoted_df.columns if c not in ID_cols]
            out_dfs[m] = pivoted_df.lazy().select(
                *ID_cols, *[pl.col(c).alias(f"static/{m}/{c}/present").cast(pl.Boolean) for c in remap_cols]
            )

        return pl.concat(list(out_dfs.values()), how="align")

    def _summarize_time_dependent_measurements(
        self,
        feature_columns: list[str],
        include_only_subjects: set[int] | None = None,
    ) -> pl.LazyFrame:
        if include_only_subjects is None:
            df = self.events_df
        else:
            df = self.events_df.filter(pl.col("subject_id").is_in(list(include_only_subjects)))

        valid_measures = {}
        for feat_col in feature_columns:
            temp, meas, feat, _ = self._parse_flat_feature_column(feat_col)

            if temp != "functional_time_dependent":
                continue

            if meas not in valid_measures:
                valid_measures[meas] = set()
            valid_measures[meas].add(feat)

        out_dfs = {}
        for m, allowed_vocab in valid_measures.items():
            cfg = self.measurement_configs[m]
            if cfg.modality == "univariate_regression" and cfg.vocabulary is None:
                out_dfs[m] = (
                    df.lazy()
                    .filter(pl.col(m).is_not_null())
                    .select(
                        "event_id",
                        "subject_id",
                        "timestamp",
                        pl.col(m).cast(pl.Float32).alias(f"functional_time_dependent/{m}/{m}/value"),
                    )
                )
                continue
            elif cfg.modality == "multivariate_regression":
                raise ValueError(f"{cfg.modality} is not supported for {cfg.temporality} measures.")

            ID_cols = ["event_id", "subject_id", "timestamp"]
            pivoted_df = (
                df.select(*ID_cols, m)
                .filter(pl.col(m).is_in(allowed_vocab))
                .with_columns(pl.lit(True).alias("__indicator"))
                .pivot(
                    index=ID_cols,
                    columns=m,
                    values="__indicator",
                    aggregate_function=None,
                )
            )

            remap_cols = [c for c in pivoted_df.columns if c not in ID_cols]
            out_dfs[m] = pivoted_df.lazy().select(
                *ID_cols,
                *[
                    pl.col(c).cast(pl.Boolean).alias(f"functional_time_dependent/{m}/{c}/present")
                    for c in remap_cols
                ],
            )

        return pl.concat(list(out_dfs.values()), how="align")

    def _summarize_dynamic_measurements(
        self,
        feature_columns: list[str],
        include_only_subjects: set[int] | None = None,
    ) -> pl.LazyFrame:
        if include_only_subjects is None:
            df = self.dynamic_measurements_df
        else:
            df = self.dynamic_measurements_df.join(
                self.events_df.filter(pl.col("subject_id").is_in(list(include_only_subjects))).select(
                    "event_id"
                ),
                on="event_id",
                how="inner",
            )

        valid_measures = {}
        for feat_col in feature_columns:
            temp, meas, feat, _ = self._parse_flat_feature_column(feat_col)

            if temp != "dynamic":
                continue

            if meas not in valid_measures:
                valid_measures[meas] = set()
            valid_measures[meas].add(feat)

        out_dfs = {}
        for m, allowed_vocab in valid_measures.items():
            cfg = self.measurement_configs[m]

            total_observations = int(
                math.ceil(
                    cfg.observation_rate_per_case
                    * cfg.observation_rate_over_cases
                    * sum(self.n_events_per_subject.values())
                )
            )

            count_type = self.get_smallest_valid_uint_type(total_observations)

            if cfg.modality == "univariate_regression" and cfg.vocabulary is None:
                prefix = f"dynamic/{m}/{m}"

                key_col = pl.col(m)
                val_col = pl.col(m).drop_nans().cast(pl.Float32)

                out_dfs[m] = (
                    df.lazy()
                    .select("measurement_id", "event_id", m)
                    .filter(pl.col(m).is_not_null())
                    .groupby("event_id")
                    .agg(
                        pl.col(m).is_not_null().sum().cast(count_type).alias(f"{prefix}/count"),
                        (
                            (pl.col(m).is_not_nan() & pl.col(m).is_not_null())
                            .sum()
                            .cast(count_type)
                            .alias(f"{prefix}/has_values_count")
                        ),
                        val_col.sum().alias(f"{prefix}/sum"),
                        (val_col**2).sum().alias(f"{prefix}/sum_sqd"),
                        val_col.min().alias(f"{prefix}/min"),
                        val_col.max().alias(f"{prefix}/max"),
                    )
                )
                continue
            elif cfg.modality == "multivariate_regression":
                column_cols = [m, m]
                values_cols = [m, cfg.values_column]
                key_prefix = f"{m}_{m}_"
                val_prefix = f"{cfg.values_column}_{m}_"

                key_col = cs.starts_with(key_prefix)
                val_col = cs.starts_with(val_prefix).drop_nans().cast(pl.Float32)

                aggs = [
                    key_col.is_not_null()
                    .sum()
                    .cast(count_type)
                    .map_alias(lambda c: f"dynamic/{m}/{c.replace(key_prefix, '')}/count"),
                    (
                        (cs.starts_with(val_prefix).is_not_null() & cs.starts_with(val_prefix).is_not_nan())
                        .sum()
                        .map_alias(lambda c: f"dynamic/{m}/{c.replace(val_prefix, '')}/has_values_count")
                    ),
                    val_col.sum().map_alias(lambda c: f"dynamic/{m}/{c.replace(val_prefix, '')}/sum"),
                    (val_col**2)
                    .sum()
                    .map_alias(lambda c: f"dynamic/{m}/{c.replace(val_prefix, '')}/sum_sqd"),
                    val_col.min().map_alias(lambda c: f"dynamic/{m}/{c.replace(val_prefix, '')}/min"),
                    val_col.max().map_alias(lambda c: f"dynamic/{m}/{c.replace(val_prefix, '')}/max"),
                ]
            else:
                column_cols = [m]
                values_cols = [m]
                aggs = [
                    pl.all()
                    .is_not_null()
                    .sum()
                    .cast(count_type)
                    .map_alias(lambda c: f"dynamic/{m}/{c}/count")
                ]

            ID_cols = ["measurement_id", "event_id"]
            out_dfs[m] = (
                df.select(*ID_cols, *set(column_cols + values_cols))
                .filter(pl.col(m).is_in(allowed_vocab))
                .pivot(
                    index=ID_cols,
                    columns=column_cols,
                    values=values_cols,
                    aggregate_function=None,
                )
                .lazy()
                .drop("measurement_id")
                .groupby("event_id")
                .agg(*aggs)
            )

        return pl.concat(list(out_dfs.values()), how="align")

    def _get_flat_col_dtype(self, col: str) -> pl.DataType:
        """Gets the appropriate minimal dtype for the given flat representation column string."""

        parts = col.split("/")
        if len(parts) < 4:
            raise ValueError(f"Malformed column {col}. Should be temporal/measurement/feature/agg")

        temp, meas = parts[0], parts[1]
        agg = parts[-1]
        feature = "/".join(parts[2:-1])

        cfg = self.measurement_configs[meas]

        match agg:
            case "sum" | "sum_sqd" | "min" | "max" | "value":
                return pl.Float32
            case "present":
                return pl.Boolean
            case "count" | "has_values_count":
                # config.observation_rate_over_cases = total_observed / total_possible
                # config.observation_rate_per_case = raw_total_observed / total_observed

                match temp:
                    case TemporalityType.STATIC:
                        n_possible = len(self.subject_ids)
                    case str() | TemporalityType.DYNAMIC | TemporalityType.FUNCTIONAL_TIME_DEPENDENT:
                        n_possible = sum(self.n_events_per_subject.values())
                    case _:
                        raise ValueError(
                            f"Column name {col} malformed: Temporality {temp} not in "
                            f"{', '.join(TemporalityType.values())} nor is it a window size string"
                        )

                if cfg.vocabulary is None:
                    observation_frequency = cfg.observation_rate_per_case * cfg.observation_rate_over_cases
                else:
                    if feature not in cfg.vocabulary.idxmap:
                        raise ValueError(f"Column name {col} malformed: Feature {feature} not in {meas}!")
                    else:
                        observation_frequency = cfg.vocabulary.obs_frequencies[cfg.vocabulary[feature]]

                total_observations = int(math.ceil(observation_frequency * n_possible))

                return self.get_smallest_valid_uint_type(total_observations)
            case _:
                raise ValueError(f"Column name {col} malformed!")

    def _get_flat_static_rep(
        self,
        feature_columns: list[str],
        **kwargs,
    ) -> pl.LazyFrame:
        static_features = [c for c in feature_columns if c.startswith("static/")]
        return self._normalize_flat_rep_df_cols(
            self._summarize_static_measurements(static_features, **kwargs).collect().lazy(),
            static_features,
            set_count_0_to_null=False,
        )

    def _get_flat_ts_rep(
        self,
        feature_columns: list[str],
        **kwargs,
    ) -> pl.LazyFrame:
        return self._normalize_flat_rep_df_cols(
            self._summarize_time_dependent_measurements(feature_columns, **kwargs)
            .join(
                self._summarize_dynamic_measurements(feature_columns, **kwargs),
                on="event_id",
                how="inner",
            )
            .drop("event_id")
            .sort(by=["subject_id", "timestamp"])
            .collect()
            .lazy(),
            [c for c in feature_columns if not c.startswith("static/")],
        )
        # The above .collect().lazy() shouldn't be necessary but it appears to be for some reason...

    def _normalize_flat_rep_df_cols(
        self, flat_df: DF_T, feature_columns: list[str] | None = None, set_count_0_to_null: bool = False
    ) -> DF_T:
        if feature_columns is None:
            feature_columns = [x for x in flat_df.columns if x not in ("subject_id", "timestamp")]
            cols_to_add = set()
            cols_to_retype = set(feature_columns)
        else:
            cols_to_add = set(feature_columns) - set(flat_df.columns)
            cols_to_retype = set(feature_columns).intersection(set(flat_df.columns))

        cols_to_add = [(c, self._get_flat_col_dtype(c)) for c in cols_to_add]
        cols_to_retype = [(c, self._get_flat_col_dtype(c)) for c in cols_to_retype]

        if "timestamp" in flat_df.columns:
            key_cols = ["subject_id", "timestamp"]
        else:
            key_cols = ["subject_id"]

        flat_df = flat_df.with_columns(
            *[pl.lit(None, dtype=dt).alias(c) for c, dt in cols_to_add],
            *[pl.col(c).cast(dt).alias(c) for c, dt in cols_to_retype],
        ).select(*key_cols, *feature_columns)

        if not set_count_0_to_null:
            return flat_df

        flat_df = flat_df.collect()

        flat_df = flat_df.with_columns(
            pl.when(cs.ends_with("count") != 0).then(cs.ends_with("count")).keep_name()
        ).lazy()
        return flat_df

    def _summarize_over_window(self, df: DF_T, window_size: str) -> pl.LazyFrame:
        if isinstance(df, Path):
            df = pl.scan_parquet(df)

        # Handle null values in the "timestamp" column
        df = df.with_columns(
            pl.col("timestamp").fill_null(pl.datetime(1900, 1, 1))
        )

        def time_aggd_col_alias_fntr(new_agg: str | None = None) -> Callable[[str], str]:
            if new_agg is None:
                def f(c: str) -> str:
                    return "/".join([window_size] + c.split("/")[1:])
            else:
                def f(c: str) -> str:
                    return "/".join([window_size] + c.split("/")[1:-1] + [new_agg])
            return f

        # Columns to convert to counts:
        present_indicator_cols = cs.ends_with("/present")

        # Columns to convert to value aggregations:
        value_cols = cs.ends_with("/value")

        # Columns to aggregate via other operations
        cnt_cols = (cs.ends_with("/count") | cs.ends_with("/has_values_count")).fill_null(0)

        cols_to_sum = cs.ends_with("/sum") | cs.ends_with("/sum_sqd")
        cols_to_min = cs.ends_with("/min")
        cols_to_max = cs.ends_with("/max")

        if window_size == "FULL":
            df = df.groupby("subject_id").agg(
                "timestamp",
                # present to counts
                present_indicator_cols.cumsum().map_alias(time_aggd_col_alias_fntr("count")),
                # values to stats
                value_cols.is_not_null().cumsum().map_alias(time_aggd_col_alias_fntr("count")),
                (
                    (value_cols.is_not_null() & value_cols.is_not_nan())
                    .cumsum()
                    .map_alias(time_aggd_col_alias_fntr("has_values_count"))
                ),
                value_cols.cumsum().map_alias(time_aggd_col_alias_fntr("sum")),
                (value_cols**2).cumsum().map_alias(time_aggd_col_alias_fntr("sum_sqd")),
                value_cols.cummin().map_alias(time_aggd_col_alias_fntr("min")),
                value_cols.cummax().map_alias(time_aggd_col_alias_fntr("max")),
                # Raw aggregations
                cnt_cols.cumsum().map_alias(time_aggd_col_alias_fntr()),
                cols_to_sum.cumsum().map_alias(time_aggd_col_alias_fntr()),
                cols_to_min.cummin().map_alias(time_aggd_col_alias_fntr()),
                cols_to_max.cummax().map_alias(time_aggd_col_alias_fntr()),
            )
            df = df.explode(*[c for c in df.columns if c != "subject_id"])
        else:
            # Calculate the window start and end times
            window_start = df["timestamp"].dt.truncate(window_size)
            window_end = window_start + pl.durationDt.nanoseconds(pl.to_url(window_size))

            # Group by subject_id and the calculated window start/end times
            grouped_df = df.groupby(["subject_id", window_start, window_end]).agg(
                # present to counts
                present_indicator_cols.sum().map_alias(time_aggd_col_alias_fntr("count")),
                # values to stats
                value_cols.is_not_null().sum().map_alias(time_aggd_col_alias_fntr("count")),
                (
                    (value_cols.is_not_null() & value_cols.is_not_nan())
                    .sum()
                    .map_alias(time_aggd_col_alias_fntr("has_values_count"))
                ),
                value_cols.sum().map_alias(time_aggd_col_alias_fntr("sum")),
                (value_cols**2).sum().map_alias(time_aggd_col_alias_fntr("sum_sqd")),
                value_cols.min().map_alias(time_aggd_col_alias_fntr("min")),
                value_cols.max().map_alias(time_aggd_col_alias_fntr("max")),
                # Raw aggregations
                cnt_cols.sum().map_alias(time_aggd_col_alias_fntr()),
                cols_to_sum.sum().map_alias(time_aggd_col_alias_fntr()),
                cols_to_min.min().map_alias(time_aggd_col_alias_fntr()),
                cols_to_max.max().map_alias(time_aggd_col_alias_fntr()),
            )

            # Drop the window_end column and rename window_start to timestamp
            df = grouped_df.drop("window_end").rename({"window_start": "timestamp"})

        return self._normalize_flat_rep_df_cols(df, set_count_0_to_null=True)

    def _denormalize(self, events_df: DF_T, col: str) -> DF_T:
        if self.config.normalizer_config is None:
            return events_df
        elif self.config.normalizer_config["cls"] != "standard_scaler":
            raise ValueError(f"De-normalizing from {self.config.normalizer_config} not yet supported!")

        config = self.measurement_configs[col]
        if config.modality != DataModality.UNIVARIATE_REGRESSION:
            raise ValueError(f"De-normalizing {config.modality} is not currently supported.")

        normalizer_params = config.measurement_metadata.normalizer
        return events_df.with_columns(
            ((pl.col(col) * normalizer_params["std_"]) + normalizer_params["mean_"]).alias(col)
        )