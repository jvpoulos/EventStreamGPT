import json
from collections import defaultdict
from pathlib import Path
import pathlib
import traceback
import pandas as pd
import numpy as np
import polars as pl
import torch
from mixins import SaveableMixin, SeedableMixin, TimeableMixin
from EventStream.data.dataset_config import DatasetConfig

from datetime import datetime

from .config import (
    MeasurementConfig,
    PytorchDatasetConfig,
    SeqPaddingSide,
    SubsequenceSamplingStrategy,
    VocabularyConfig,
)
from .types import PytorchBatch

try:
    from pyo3 import PanicException
except ImportError:
    PanicException = Exception

DATA_ITEM_T = dict[str, list[float]]


def to_int_index(col: pl.Expr) -> pl.Expr:
    """Returns an integer index of the unique elements seen in this column.

    The returned index is into a vocabulary sorted lexographically.

    Args:
        col: The column containing the data to be converted into integer indices.

    Examples:
        >>> import polars as pl
        >>> X = pl.DataFrame({
        ...     'c': ['foo', 'bar', 'foo', 'bar', 'baz', None, 'bar', 'aba'],
        ...     'd': [1, 2, 3, 4, 5, 6, 7, 8]
        ... })
        >>> X.with_columns(to_int_index(pl.col('c')))
        shape: (8, 2)
        ┌──────┬─────┐
        │ c    ┆ d   │
        │ ---  ┆ --- │
        │ u32  ┆ i64 │
        ╞══════╪═════╡
        │ 4    ┆ 1   │
        │ 1    ┆ 2   │
        │ 4    ┆ 3   │
        │ 1    ┆ 4   │
        │ 2    ┆ 5   │
        │ null ┆ 6   │
        │ 1    ┆ 7   │
        │ 0    ┆ 8   │
        └──────┴─────┘
    """

    indices = col.unique(maintain_order=True).drop_nulls().search_sorted(col)
    return pl.when(col.is_null()).then(pl.lit(None)).otherwise(indices).alias(col.meta.output_name())


class PytorchDataset(SaveableMixin, SeedableMixin, TimeableMixin, torch.utils.data.Dataset):
    """A PyTorch Dataset class built on a pre-processed `DatasetBase` instance.

    This class enables accessing the deep-learning friendly representation produced by
    `Dataset.build_DL_cached_representation` in a PyTorch Dataset format. The `getitem` method of this class
    will return a dictionary containing a subject's data from this deep learning representation, with event
    sequences sliced to be within max sequence length according to configuration parameters, and the `collate`
    method of this class will collate those output dictionaries into a `PytorchBatch` object usable by
    downstream pipelines.

    Upon construction, this class will try to load a number of dataset files from disk. These files should be
    saved in accordance with the `Dataset.save` method; in particular,

    * There should be pre-cached deep-learning representation parquet dataframes stored in ``config.save_dir /
      'DL_reps' / f"{split}*.parquet"``
    * There should be a vocabulary config object in json form stored in ``config.save_dir /
      'vocabulary_config.json'``
    * There should be a set of inferred measurement configs stored in ``config.save_dir /
      'inferred_measurement_configs.json'``
    * If a task dataframe name is specified in the configuration object, then there should be either a
      pre-cached task-specifid DL representation dataframe in ``config.save_dir / 'DL_reps' / 'for_task' /
      config.task_df_name / f"{split}.parquet"``, or a "raw" task dataframe, containing subject IDs, start and
      end times, and labels, stored in ``config.save_dir / task_dfs / f"{config.task_df_name}.parquet"``. In
      the case that the latter is all that exists, then the former will be constructed by limiting the input
      cached dataframe down to the appropriate sequences and adding label columns. This newly constructed
      datafrmae will then be saved in the former filepath for future use. This construction process should
      happen first on the train split, so that inferred task vocabularies are shared across splits.

    Args:
        config: Configuration options for the dataset.
        split: The split of data which should be used in this dataset (e.g., ``'train'``, ``'tuning'``,
            ``'held_out'``). This will dictate where the system looks for pre-cached deep-learning
            representation files.
    """

    TYPE_CHECKERS = {
        "multi_class_classification": [
            (
                {pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Int8, pl.Int16, pl.Int32, pl.Int64},
                None,
            ),
            ({pl.Categorical}, to_int_index),
            ({pl.Utf8}, to_int_index),
        ],
        "binary_classification": [({pl.Boolean}, lambda Y: Y.cast(pl.Float32))],
        "regression": [({pl.Float32, pl.Float64}, None)],
    }
    """Type checker and conversion parameters for labeled datasets."""

    @classmethod
    def normalize_task(cls, col: pl.Expr, dtype: pl.DataType) -> tuple[str, pl.Expr]:
        """Normalizes the task labels in `col` of dtype `dtype` to a common format.

        Args:
            col: The column containing the task labels, in polars expression format.
            dtype: The polars data type of the task labels.

        Returns:
            The task type (a string key into the `TYPE_CHECKERS` dictionary) and the normalized column
            expression.

        Raises:
            TypeError: If the task labels are not of a supported type.
        """
        for task_type, checkers in cls.TYPE_CHECKERS.items():
            for valid_dtypes, normalize_fn in checkers:
                if dtype in valid_dtypes:
                    return task_type, (col if normalize_fn is None else normalize_fn(col))

        raise TypeError(f"Can't process label of {dtype} type!")

    def __init__(self, config: PytorchDatasetConfig, split: str, task_df: pl.DataFrame = None, dl_reps_dir: Path = None):
        super().__init__()
        self.split = split
        config.save_dir = Path(config.save_dir)
        self.dl_reps_dir = dl_reps_dir or config.save_dir / "DL_reps"
        self.config = config
        self.task_types = {}
        self.task_vocabs = {}

        if task_df is not None:
            self.task_df = task_df
        elif self.config.task_df_name is not None:
            if split == "train":
                self.task_df = pl.read_parquet(self.config.save_dir / "task_dfs" / f"{self.config.task_df_name}.parquet")
            elif split == "tuning":
                self.task_df = pl.read_parquet(self.config.save_dir / "task_dfs" / f"{self.config.task_df_name}_val.parquet")
            elif split == "held_out":
                self.task_df = pl.read_parquet(self.config.save_dir / "task_dfs" / f"{self.config.task_df_name}_test.parquet")
            else:
                raise ValueError(f"Invalid split: {split}")
        else:
            self.task_df = None

        self.cached_data_list = []
        self.load_cached_data()  # Load the cached data after initializing self.cached_data

        self.length = sum(df.shape[0] for df in self.cached_data_list)

        if self.cached_data is None or self.cached_data.is_empty():
            self.cached_data = pl.DataFrame()  # Initialize cached_data as an empty Polars DataFrame

        # Check if the required columns exist in the cached data
        if self.cached_data.is_empty():
            raise ValueError(f"Cached data is empty for split '{split}'")

        required_columns = ["subject_id", "dynamic_indices"]
        missing_columns = [col for col in required_columns if col not in self.cached_data.columns]
        if missing_columns:
            raise ValueError(f"Required columns {missing_columns} are missing in the cached data for split '{split}'")

        # Handle missing columns
        for col in missing_columns:
            if col == "subject_id":
                self.cached_data["subject_id"] = np.arange(len(self.cached_data))
            else:
                self.cached_data[col] = pd.Series([[]] * len(self.cached_data))

        self.vocabulary_config = VocabularyConfig.from_json_file(
            Path("data/labs") / "vocabulary_config.json"
        )
        inferred_measurement_config_fp = Path("data") / "inferred_measurement_configs.json"
        with open(inferred_measurement_config_fp) as f:
            inferred_measurement_configs = {
                k: MeasurementConfig.from_dict(v) for k, v in json.load(f).items()
            }
        self.measurement_configs = {k: v for k, v in inferred_measurement_configs.items() if not v.is_dropped}

        self.split = split
        config.save_dir = Path(config.save_dir)
        self.dl_reps_dir = Path("./data/DL_reps")
        print(f"dl_reps_dir: {self.dl_reps_dir}")

        parquet_files = list(self.dl_reps_dir.glob(f"{split}*.parquet"))
        if parquet_files:
            print(f"Loading parquet files for split '{split}': {parquet_files}")

            # Read the Parquet files directly into a Pandas DataFrame
            self.cached_data = pd.concat(
                [pd.read_parquet(file) for file in parquet_files], ignore_index=True
            )

            print(f"Initial cached data columns: {self.cached_data.columns}")

            # Convert the data types of the required columns to Int64
            int_cols = ['subject_id', 'subject_id_right', 'event_id', 'measurement_id']
            for col in int_cols:
                if col in self.cached_data.columns:
                    # Replace non-finite values with 0 before converting to int64
                    self.cached_data[col] = self.cached_data[col].fillna(0).astype('int64')

            print(f"Cached data shape after transformations: {self.cached_data.shape}")
            print(f"Cached data columns after transformations: {self.cached_data.columns}")
        else:
            print(f"No parquet files found for split '{split}', creating an empty DataFrame")
            self.cached_data = pd.DataFrame()
            
        print(f"Cached data shape: {self.cached_data.shape}")
        print(f"Cached data columns: {self.cached_data.columns}")

        self.do_produce_static_data = "static_indices" in self.cached_data.columns
        self.seq_padding_side = config.seq_padding_side
        self.max_seq_len = config.max_seq_len

        length_constraint = pl.col("dynamic_indices").apply(lambda x: len(x) if isinstance(x, list) else 0) >= config.min_seq_len
        self.cached_data = self.cached_data[self.cached_data["dynamic_indices"].apply(lambda x: len(x) if isinstance(x, list) else 0) >= config.min_seq_len]

        time_columns = ["start_time", "time", "time_delta"]
        if any(col not in self.cached_data.columns for col in time_columns):
            print("Warning: One or more time-related columns are missing in the cached data. Skipping time-related calculations.")
            self.mean_log_inter_event_time_min = None
            self.std_log_inter_event_time_min = None
        else:
            if "time_delta" not in self.cached_data.columns:
                if "time" in self.cached_data.columns:
                    self.cached_data["start_time"] = self.cached_data["start_time"] + pd.to_timedelta(self.cached_data["time"].apply(lambda x: x[0] if isinstance(x, list) else 0), unit='m')
                    self.cached_data["time_delta"] = self.cached_data["time"].apply(lambda x: [t - s for s, t in zip(x[:-1], x[1:])] + [1] if isinstance(x, list) else [])
                    self.cached_data = self.cached_data.drop("time", axis=1)
                else:
                    print("Warning: 'time_delta' and 'time' columns are missing in the cached data. Skipping time-related calculations.")
                    self.mean_log_inter_event_time_min = None
                    self.std_log_inter_event_time_min = None
            else:
                inter_event_times = self.cached_data["time_delta"].explode().dropna()
                min_inter_event_time = inter_event_times.min()

                if min_inter_event_time is not None and min_inter_event_time <= 0:
                    bad_inter_event_times = self.cached_data[self.cached_data["time_delta"].apply(lambda x: min(x) <= 0 if isinstance(x, list) else False)]
                    bad_subject_ids = bad_inter_event_times["subject_id"].astype(str).tolist()
                    warning_strs = [
                        f"WARNING: Observed inter-event times <= 0 for {len(bad_inter_event_times)} subjects!",
                        f"ESD Subject IDs: {', '.join(bad_subject_ids)}",
                        f"Global min: {min_inter_event_time}",
                    ]
                    if self.config.save_dir is not None:
                        fp = self.config.save_dir / f"malformed_data_{self.split}.parquet"
                        bad_inter_event_times.to_parquet(fp)
                        warning_strs.append(f"Wrote malformed data records to {fp}")
                    warning_strs.append("Removing malformed subjects")

                    print("\n".join(warning_strs))

                    self.cached_data = self.cached_data[self.cached_data["time_delta"].apply(lambda x: min(x) > 0 if isinstance(x, list) else True)]

                self.mean_log_inter_event_time_min = np.log(inter_event_times).mean()
                self.std_log_inter_event_time_min = np.log(inter_event_times).std()

        if self.config.train_subset_size not in (None, "FULL") and self.split == "train":
            if isinstance(self.config.train_subset_size, int) and self.config.train_subset_size > 0:
                kwargs = {"n": self.config.train_subset_size}
            elif isinstance(self.config.train_subset_size, float) and 0 < self.config.train_subset_size < 1:
                kwargs = {"frac": self.config.train_subset_size}
            else:
                raise TypeError(
                    f"Can't process subset size of {type(self.config.train_subset_size)}, "
                    f"{self.config.train_subset_size}"
                )

            self.cached_data = self.cached_data.sample(random_state=self.config.train_subset_seed, **kwargs)

        with self._time_as("convert_to_rows"):
            self.subject_ids = self.cached_data["subject_id"].tolist()
            self.cached_data = self.cached_data.drop("subject_id", axis=1)
            self.columns = self.cached_data.columns

            self.cached_data = self.cached_data.to_dict('records')

            # Initialize out_batch here
            self.out_batch = {}
            if "dynamic_indices" in self.columns:
                self.out_batch["dynamic_indices"] = torch.zeros(len(self.cached_data), len(self.columns), dtype=torch.long)
            else:
                if "event_mask" in self.columns:
                    self.out_batch["event_mask"] = torch.zeros(len(self.cached_data), dtype=torch.bool)
                    self.out_batch["dynamic_indices"] = torch.zeros_like(self.out_batch["event_mask"], dtype=torch.long)
                else:
                    self.out_batch["dynamic_indices"] = torch.zeros(len(self.cached_data), dtype=torch.long)
                
    def is_cached_data_empty(self):
        return self.cached_data.is_empty()

    def load_cached_data(self):
        self.logger.info(f"Loading cached data for split: {self.split}")
        
        if not self.dl_reps_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.dl_reps_dir}")

        parquet_files = list(self.dl_reps_dir.glob(f"{self.split}*.parquet"))
        self.logger.debug(f"Found {len(parquet_files)} Parquet files")

        if not parquet_files:
            raise FileNotFoundError(f"No Parquet files found for split '{self.split}' in directory '{self.dl_reps_dir}'")
        
        self.cached_data_list = []
        total_rows = 0
        for parquet_file in parquet_files:
            self.logger.debug(f"Reading file: {parquet_file}")
            try:
                df = pl.read_parquet(parquet_file)
                self.logger.debug(f"File {parquet_file} shape: {df.shape}")
                if self.task_df is not None:
                    df = df.filter(pl.col('subject_id').is_in(self.task_df['subject_id']))
                self.cached_data_list.append(df)
                total_rows += df.shape[0]
            except Exception as e:
                self.logger.error(f"Error reading Parquet file: {parquet_file}")
                self.logger.error(f"Error message: {str(e)}")
                continue

        if not self.cached_data_list:
            self.logger.error(f"No data loaded for split: {self.split}")
            raise ValueError(f"No data loaded for split: {self.split}")

        self.logger.info(f"Loaded {total_rows} rows of cached data")

        if total_rows == 0:
            raise ValueError(f"No matching data found for split: {self.split}")

        # Use Polars concat instead of Pandas
        self.cached_data = pl.concat(self.cached_data_list)

        self.logger.info(f"Cached data loaded successfully for split: {self.split}")
            
    @staticmethod
    def _build_task_cached_df(task_df: pl.LazyFrame, cached_data: pl.LazyFrame) -> pl.LazyFrame:
        """Restricts the data in a cached dataframe to only contain data for the passed task dataframe.

        Args:
            task_df: A polars LazyFrame, which must have columns ``subject_id``, ``start_time`` and
                ``end_time``. These three columns define the schema of the task (the inputs). The remaining
                columns in the task dataframe will be interpreted as labels.
            cached_data: A polars LazyFrame containing the data to be restricted to the task dataframe. Must
                have the columns ``subject_id``, ``start_time``, ``time`` or ``time_delta``,
                ``dynamic_indices``, ``dynamic_values``, and ``dynamic_measurement_indices``. These columns
                will all be restricted to just contain those events whose time values are in the specified
                task specific time range.

        Returns:
            The restricted cached dataframe, which will have the same columns as the input cached dataframe
            plus the task label columns, and will be limited to just those subjects and time-periods specified
            in the task dataframe.

        Examples:
            >>> import polars as pl
            >>> from datetime import datetime
            >>> cached_data = pl.DataFrame({
            ...     "subject_id": [0, 1, 2, 3],
            ...     "start_time": [
            ...         datetime(2020, 1, 1),
            ...         datetime(2020, 2, 1),
            ...         datetime(2020, 3, 1),
            ...         datetime(2020, 1, 2)
            ...     ],
            ...     "time": [
            ...         [0.0, 60*24.0, 2*60*24., 3*60*24., 4*60*24.],
            ...         [0.0, 7*60*24.0, 2*7*60*24., 3*7*60*24., 4*7*60*24.],
            ...         [0.0, 60*12.0, 2*60*12.],
            ...         [0.0, 60*24.0, 2*60*24., 3*60*24., 4*60*24.],
            ...     ],
            ...     "dynamic_measurement_indices": [
            ...         [[0, 1, 1], [0, 2], [0], [0, 3], [0]],
            ...         [[0, 1, 1], [0, 4], [0], [0, 1], [0]],
            ...         [[0, 1, 1], [0], [0, 4]],
            ...         [[0, 1, 1], [0, 4], [0], [0, 2], [0]],
            ...     ],
            ...     "dynamic_indices": [
            ...         [[6, 11, 12], [1, 40], [5], [1, 55], [5]],
            ...         [[2, 11, 13], [1, 84], [8], [1, 19], [5]],
            ...         [[1, 18, 21], [1], [5, 87]],
            ...         [[3, 20, 21], [1, 94], [8], [1, 33], [9]],
            ...     ],
            ...     "dynamic_values": [
            ...         [[None, 0.2, 1.0], [None, 0.0], [None], [None, None], [None]],
            ...         [[None, -0.1, 0.0], [None, None], [None], [None, -4.2], [None]],
            ...         [[None, 0.9, 1.2], [None], [None, None]],
            ...         [[None, 3.2, -1.0], [None, None], [None], [None, 0.5], [None]],
            ...     ],
            ... })
            >>> task_df = pl.DataFrame({
            ...     "subject_id": [0, 1, 2, 5],
            ...     "start_time": [
            ...         datetime(2020, 1, 1),
            ...         datetime(2020, 1, 11),
            ...         datetime(2020, 3, 1, 13),
            ...         datetime(2020, 1, 2)
            ...     ],
            ...     "end_time": [
            ...         datetime(2020, 1, 3),
            ...         datetime(2020, 1, 21),
            ...         datetime(2020, 3, 4),
            ...         datetime(2020, 1, 3)
            ...     ],
            ...     "label1": [0, 1, 0, 1],
            ...     "label2": [0, 1, 5, 1]
            ... })
            >>> pl.Config.set_tbl_width_chars(88)
            <class 'polars.config.Config'>
            >>> PytorchDataset._build_task_cached_df(task_df, cached_data)
            shape: (3, 8)
            ┌───────────┬───────────┬───────────┬──────────┬──────────┬──────────┬────────┬────────┐
            │ subject_i ┆ start_tim ┆ time      ┆ dynamic_ ┆ dynamic_ ┆ dynamic_ ┆ label1 ┆ label2 │
            │ d         ┆ e         ┆ ---       ┆ measurem ┆ indices  ┆ values   ┆ ---    ┆ ---    │
            │ ---       ┆ ---       ┆ list[f64] ┆ ent_indi ┆ ---      ┆ ---      ┆ i64    ┆ i64    │
            │ i64       ┆ datetime[ ┆           ┆ ces      ┆ list[lis ┆ list[lis ┆        ┆        │
            │           ┆ μs]       ┆           ┆ ---      ┆ t[i64]]  ┆ t[f64]]  ┆        ┆        │
            │           ┆           ┆           ┆ list[lis ┆          ┆          ┆        ┆        │
            │           ┆           ┆           ┆ t[i64]]  ┆          ┆          ┆        ┆        │
            ╞═══════════╪═══════════╪═══════════╪══════════╪══════════╪══════════╪════════╪════════╡
            │ 0         ┆ 2020-01-0 ┆ [0.0,     ┆ [[0, 1,  ┆ [[6, 11, ┆ [[null,  ┆ 0      ┆ 0      │
            │           ┆ 1         ┆ 1440.0]   ┆ 1], [0,  ┆ 12], [1, ┆ 0.2,     ┆        ┆        │
            │           ┆ 00:00:00  ┆           ┆ 2]]      ┆ 40]]     ┆ 1.0],    ┆        ┆        │
            │           ┆           ┆           ┆          ┆          ┆ [null,   ┆        ┆        │
            │           ┆           ┆           ┆          ┆          ┆ 0.0]]    ┆        ┆        │
            │ 1         ┆ 2020-02-0 ┆ []        ┆ []       ┆ []       ┆ []       ┆ 1      ┆ 1      │
            │           ┆ 1         ┆           ┆          ┆          ┆          ┆        ┆        │
            │           ┆ 00:00:00  ┆           ┆          ┆          ┆          ┆        ┆        │
            │ 2         ┆ 2020-03-0 ┆ [1440.0]  ┆ [[0, 4]] ┆ [[5,     ┆ [[null,  ┆ 0      ┆ 5      │
            │           ┆ 1         ┆           ┆          ┆ 87]]     ┆ null]]   ┆        ┆        │
            │           ┆ 00:00:00  ┆           ┆          ┆          ┆          ┆        ┆        │
            └───────────┴───────────┴───────────┴──────────┴──────────┴──────────┴────────┴────────┘
        """
        time_dep_cols = [c for c in ("time", "time_delta") if c in cached_data.columns]
        time_dep_cols.extend(["dynamic_indices", "dynamic_values", "dynamic_measurement_indices"])

        if self.cached_data.empty:
            self.out_batch["dynamic_indices"] = torch.zeros(0, dtype=torch.long)
        else:
            if "dynamic_indices" in self.columns:
                self.out_batch["dynamic_indices"] = torch.zeros(len(self.cached_data), len(self.columns), dtype=torch.long)
            else:
                if "event_mask" in self.columns:
                    self.out_batch["event_mask"] = torch.zeros(len(self.cached_data), dtype=torch.bool)
                    self.out_batch["dynamic_indices"] = torch.zeros_like(self.out_batch["event_mask"], dtype=torch.long)
                else:
                    self.out_batch["dynamic_indices"] = torch.zeros(len(self.cached_data), dtype=torch.long)

        if task_cached_data.collect().empty:
            raise ValueError(f"Task-specific cached data is empty for split '{split}'")

        # Handle missing columns
        missing_cols = [col for col in time_dep_cols if col not in cached_data.columns]
        for col in missing_cols:
            cached_data = cached_data.with_column(pl.lit(None).alias(col))

        if "time" in cached_data.columns:
            time_col_expr = pl.col("time")
        elif "time_delta" in cached_data.columns:
            time_col_expr = pl.col("time_delta").cumsum().over("subject_id")

        start_idx_expr = (
            time_col_expr.list.explode().search_sorted(pl.col("start_time_min")).over("subject_id")
        )
        end_idx_expr = time_col_expr.list.explode().search_sorted(pl.col("end_time_min")).over("subject_id")

        return (
            cached_data.join(task_df, on="subject_id", how="inner", suffix="_task")
            .with_columns(
                start_time_min=(pl.col("start_time_task") - pl.col("start_time")) / np.timedelta64(1, "m"),
                end_time_min=(pl.col("end_time") - pl.col("start_time")) / np.timedelta64(1, "m"),
            )
            .with_columns(
                **{
                    t: pl.col(t).list.slice(start_idx_expr, end_idx_expr - start_idx_expr)
                    for t in time_dep_cols
                },
            )
            .drop("start_time_task", "end_time_min", "start_time_min", "end_time")
        )

        return cached_data

    def __len__(self):
        return len(self.cached_data)

    def __getitem__(self, idx: int) -> dict[str, any]:
        try:
            full_subj_data = {}
            for col, v in self.cached_data[idx].items():
                if isinstance(v, str):
                    full_subj_data[col] = json.loads(v)
                else:
                    full_subj_data[col] = v
            
            # Convert dynamic_indices to tensor
            if 'dynamic_indices' in full_subj_data:
                full_subj_data['dynamic_indices'] = torch.tensor(full_subj_data['dynamic_indices'], dtype=torch.long)
            
            # Convert dynamic_values to tensor, keeping it as float
            if 'dynamic_values' in full_subj_data:
                full_subj_data['dynamic_values'] = torch.tensor(full_subj_data['dynamic_values'], dtype=torch.float)
            
            # Process static data
            static_fields = ['InitialA1c', 'A1cGreaterThan7', 'Female', 'Married', 'GovIns', 'English', 'AgeYears', 'SDI_score', 'Veteran']
            for field in static_fields:
                if field in full_subj_data:
                    full_subj_data[field] = torch.tensor(full_subj_data[field], 
                                                         dtype=torch.float32 if field in ['InitialA1c', 'A1cGreaterThan7', 'AgeYears', 'SDI_score'] else torch.long)

            for k in ["static_indices", "static_measurement_indices"]:
                if full_subj_data.get(k) is None:
                    full_subj_data[k] = []

            if self.config.do_include_subject_id:
                full_subj_data["subject_id"] = self.subject_ids[idx]

            if self.config.do_include_start_time_min:
                full_subj_data["start_time"] = full_subj_data["start_time"].timestamp() / 60.0
            else:
                full_subj_data.pop("start_time", None)

            # If "time_delta" is not present in the data, set it to an empty list
            if "time_delta" not in full_subj_data:
                full_subj_data["time_delta"] = []

            seq_len = len(full_subj_data["time_delta"])
            if seq_len > self.max_seq_len:
                with self._time_as("truncate_to_max_seq_len"):
                    match self.config.subsequence_sampling_strategy:
                        case SubsequenceSamplingStrategy.RANDOM:
                            start_idx = np.random.choice(seq_len - self.max_seq_len)
                        case SubsequenceSamplingStrategy.TO_END:
                            start_idx = seq_len - self.max_seq_len
                        case SubsequenceSamplingStrategy.FROM_START:
                            start_idx = 0
                        case _:
                            raise ValueError(
                                f"Invalid sampling strategy: {self.config.subsequence_sampling_strategy}!"
                            )

                if self.config.do_include_start_time_min:
                    full_subj_data["start_time"] += sum(full_subj_data["time_delta"][:start_idx])
                if self.config.do_include_subsequence_indices:
                    full_subj_data["start_idx"] = start_idx
                    full_subj_data["end_idx"] = start_idx + self.max_seq_len

                for k in (
                    "time_delta",
                    "dynamic_indices",
                    "dynamic_values",
                    "dynamic_measurement_indices",
                ):
                    full_subj_data[k] = full_subj_data[k][start_idx : start_idx + self.max_seq_len]
            elif self.config.do_include_subsequence_indices:
                full_subj_data["start_idx"] = 0
                full_subj_data["end_idx"] = seq_len

            return self._seeded_getitem(idx, full_subj_data)

        except (IndexError, KeyError, AttributeError) as e:
            print(f"Error accessing cached data at index {idx}: {str(e)}")
            print("Cached data length:", len(self.cached_data))
            print("Cached data columns:", self.cached_data.columns)
            raise e

    @SeedableMixin.WithSeed
    @TimeableMixin.TimeAs
    def _seeded_getitem(self, idx: int, full_subj_data: dict[str, list]) -> dict[str, list]:
        """Returns a dictionary corresponding to a single subject's data.

        This function is automatically seeded for robustness. See `__getitem__` for a description of the
        output format.
        """
        try:
            for k in ["static_indices", "static_measurement_indices"]:
                if full_subj_data.get(k) is None:
                    full_subj_data[k] = []
            if self.config.do_include_subject_id:
                full_subj_data["subject_id"] = self.subject_ids[idx]
            if self.config.do_include_start_time_min:
                # Note that this is using the python datetime module's `timestamp` function which differs from
                # some dataframe libraries' timestamp functions (e.g., polars).
                full_subj_data["start_time"] = full_subj_data["start_time"].timestamp() / 60.0
            else:
                full_subj_data.pop("start_time", None)

            # If "time_delta" is not present in the data, set it to an empty list
            if "time_delta" not in full_subj_data:
                full_subj_data["time_delta"] = []

            # If we need to truncate to `self.max_seq_len`, grab a random full-size span to capture that.
            # TODO(mmd): This will proportionally underweight the front and back ends of the subjects data
            # relative to the middle, as there are fewer full length sequences containing those elements.
            seq_len = len(full_subj_data["time_delta"])
            if seq_len > self.max_seq_len:
                with self._time_as("truncate_to_max_seq_len"):
                    match self.config.subsequence_sampling_strategy:
                        case SubsequenceSamplingStrategy.RANDOM:
                            start_idx = np.random.choice(seq_len - self.max_seq_len)
                        case SubsequenceSamplingStrategy.TO_END:
                            start_idx = seq_len - self.max_seq_len
                        case SubsequenceSamplingStrategy.FROM_START:
                            start_idx = 0
                        case _:
                            raise ValueError(
                                f"Invalid sampling strategy: {self.config.subsequence_sampling_strategy}!"
                            )

                    if self.config.do_include_start_time_min:
                        full_subj_data["start_time"] += sum(full_subj_data["time_delta"][:start_idx])
                    if self.config.do_include_subsequence_indices:
                        full_subj_data["start_idx"] = start_idx
                        full_subj_data["end_idx"] = start_idx + self.max_seq_len

                    for k in (
                        "time_delta",
                        "dynamic_indices",
                        "dynamic_values",
                        "dynamic_measurement_indices",
                    ):
                        if k in full_subj_data:
                            full_subj_data[k] = full_subj_data[k][start_idx : start_idx + self.max_seq_len]
            elif self.config.do_include_subsequence_indices:
                full_subj_data["start_idx"] = 0
                full_subj_data["end_idx"] = seq_len

            return full_subj_data

        except (IndexError, KeyError) as e:
            print(f"Error accessing cached data at index {idx}: {str(e)}")
            print("Cached data length:", len(self.cached_data))
            print("Cached data:", self.cached_data)
            raise e

    def __static_and_dynamic_collate(self, batch: list[DATA_ITEM_T], device: torch.device) -> dict:
        """An internal collate function for both static and dynamic data."""
        out_batch = self.__dynamic_only_collate(batch, device)

        # Get the maximum number of static elements in the batch.
        max_n_static = max(len(e.get("static_indices", [])) for e in batch)

        # Walk through the batch and pad the associated tensors in all requisite dimensions.
        self._register_start("collate_static_padding")
        out = defaultdict(list)
        for e in batch:
            if self.do_produce_static_data:
                n_static = len(e.get("static_indices", []))
                static_delta = max_n_static - n_static
                static_indices = torch.tensor(e.get("static_indices", []), device=device)
                static_measurement_indices = torch.tensor(e.get("static_measurement_indices", []), device=device)
                out["static_indices"].append(
                    torch.nn.functional.pad(static_indices, (0, static_delta), value=0)
                )
                out["static_measurement_indices"].append(
                    torch.nn.functional.pad(static_measurement_indices, (0, static_delta), value=0)
                )
        self._register_end("collate_static_padding")

        self._register_start("collate_static_post_padding")
        # Unsqueeze the padded tensors into the batch dimension and combine them.
        out = {
            k: torch.stack(Ts, dim=0)
            for k, Ts in out.items()
        }

        # Convert to the right types and add to the batch.
        out_batch["static_indices"] = out["static_indices"].long()
        out_batch["static_measurement_indices"] = out["static_measurement_indices"].long()
        self._register_end("collate_static_post_padding")

        return out_batch

    def __dynamic_only_collate(self, batch: list[DATA_ITEM_T], device: torch.device) -> dict:
        """An internal collate function for dynamic data alone."""
        # Get the local max sequence length and n_data elements for padding.
        max_seq_len = max(len(e.get("time_delta", [])) for e in batch)

        max_n_data = 0
        for e in batch:
            max_n_data = max(max_n_data, max(len(v) for v in e.get("dynamic_indices", [])))
        if max_n_data == 0:
            raise ValueError(f"Batch has no dynamic measurements! Got:\n{batch[0]}\n{batch[1]}\n...")

        # Walk through the batch and pad the associated tensors in all requisite dimensions.
        self._register_start("collate_dynamic_padding")
        out_numeric = defaultdict(list)
        out_datetime = defaultdict(list)
        out_object = defaultdict(list)

        for e in batch:
            seq_len = len(e.get("time_delta", []))
            seq_delta = max_seq_len - seq_len

            if self.seq_padding_side == SeqPaddingSide.RIGHT:
                time_delta = torch.tensor(e.get("time_delta", []), device=device)
                out_numeric["time_delta"].append(
                    torch.nn.functional.pad(time_delta, (0, seq_delta), value=0)
                )
            else:
                time_delta = torch.tensor(e.get("time_delta", []), device=device)
                out_numeric["time_delta"].append(
                    torch.nn.functional.pad(time_delta, (seq_delta, 0), value=0)
                )

            data_elements_numeric = defaultdict(list)
            data_elements_datetime = defaultdict(list)
            data_elements_object = defaultdict(list)

            # Handle dynamic_indices_event_type and dynamic_counts_event_type
            dynamic_indices_event_type = torch.tensor(e.get("dynamic_indices_event_type", []), device=device)
            dynamic_counts_event_type = torch.tensor(e.get("dynamic_counts_event_type", []), device=device)
            if dynamic_indices_event_type.numel() > 0:
                data_elements_object["dynamic_indices_event_type"].append(
                    torch.nn.functional.pad(dynamic_indices_event_type, (0, seq_delta), value=0)
                )
            if dynamic_counts_event_type.numel() > 0:
                data_elements_numeric["dynamic_counts_event_type"].append(
                    torch.nn.functional.pad(dynamic_counts_event_type, (0, seq_delta), value=0)
                )

            for k in ("dynamic_indices"):
                values = e.get(k, [])
                if not values:
                    continue

                values = [torch.tensor(value, device=device) for value in values]
                data_delta = max_n_data - max(len(v) for v in values)

                data_elements_numeric[k].extend(
                    [
                        torch.nn.functional.pad(value, (0, data_delta), value=0)
                        for value in values
                    ]
                )

            if self.seq_padding_side == SeqPaddingSide.RIGHT:
                for d_elem in (data_elements_numeric, data_elements_datetime, data_elements_object):
                    for k, values in d_elem.items():
                        if not values:
                            d_elem[k] = torch.tensor([], device=device)
                        else:
                            d_elem[k] = torch.stack(values, dim=1)
            else:
                for d_elem in (data_elements_numeric, data_elements_datetime, data_elements_object):
                    for k, values in d_elem.items():
                        if not values:
                            d_elem[k] = torch.tensor([], device=device)
                        else:
                            d_elem[k] = torch.stack(values, dim=2)

            out_numeric.update(data_elements_numeric)
            out_datetime.update(data_elements_datetime)
            out_object.update(data_elements_object)

        self._register_end("collate_dynamic_padding")

        self._register_start("collate_post_padding_processing")
        # Unsqueeze the padded tensors into the batch dimension and combine them.
        out_batch = {
            k: torch.stack(Ts, dim=0)
            for k, Ts in ((*out_numeric.items(), *out_datetime.items(), *out_object.items()))
        }

        # Add event and data masks on the basis of which elements are present, then convert the tensor
        # elements to the appropriate types.
        out_batch["event_mask"] = ~out_batch.get("time_delta", torch.zeros_like(out_batch["dynamic_indices"])).isnan()

        if "dynamic_values" in out_batch:
            out_batch["dynamic_values_mask"] = ~out_batch["dynamic_values"].isnan()
        else:
            # Create a dynamic_values_mask with the correct dimensions
            out_batch["dynamic_values_mask"] = torch.zeros(
                (out_batch["event_mask"].shape[0], out_batch["event_mask"].shape[1], max_n_data),
                dtype=torch.bool,
                device=device,
            )

        out_batch["time_delta"] = out_batch["time_delta"].nan_to_num(0).long()

        out_batch["dynamic_indices"] = out_batch["dynamic_indices"].nan_to_num(0).long()

        if "dynamic_measurement_indices" in out_batch:
            out_batch["dynamic_measurement_indices"] = out_batch["dynamic_measurement_indices"].nan_to_num(0).long()

        if "dynamic_values" in out_batch:
            out_batch["dynamic_values"] = out_batch["dynamic_values"].nan_to_num(0)

        if self.config.do_include_start_time_min:
            out_batch["start_time"] = torch.tensor([e.get("start_time", 0.0) for e in batch], device=device)

        if self.config.do_include_subsequence_indices:
            out_batch["start_idx"] = torch.tensor([e.get("start_idx", 0) for e in batch], device=device, dtype=torch.long)
            out_batch["end_idx"] = torch.tensor([e.get("end_idx", 0) for e in batch], device=device, dtype=torch.long)

        if self.config.do_include_subject_id:
            out_batch["subject_id"] = torch.tensor([e.get("subject_id", 0) for e in batch], device=device, dtype=torch.long)

        self._register_end("collate_post_padding_processing")

        return out_batch

@TimeableMixin.TimeAs
def collate(self, batch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Filter out items with None labels if we have a task
    if self.has_task:
        valid_items = [item for item in batch if item["labels"] is not None]
        if not valid_items:
            return None
    else:
        valid_items = batch
    
    # Get the maximum sequence length in the batch
    max_seq_len = max(len(item["dynamic_indices"]) for item in valid_items)
    
    # Initialize the tensors with the correct shape
    dynamic_indices = torch.zeros((len(valid_items), max_seq_len), dtype=torch.long, device=device)
    dynamic_values = torch.zeros((len(valid_items), max_seq_len), dtype=torch.float, device=device)
    
    # Populate the tensors with the data from valid items
    for i, item in enumerate(valid_items):
        seq_len = len(item["dynamic_indices"])
        dynamic_indices[i, :seq_len] = item["dynamic_indices"]
        dynamic_values[i, :seq_len] = item["dynamic_values"]
    
    out_batch = {
        "dynamic_indices": dynamic_indices,
        "dynamic_values": dynamic_values,
    }
    
    # Add static features
    static_features = ['InitialA1c', 'Female', 'Married', 'GovIns', 'English', 'AgeYears', 'SDI_score', 'Veteran', 'A1cGreaterThan7']
    for feature in static_features:
        if feature in valid_items[0]:
            out_batch[feature] = torch.stack([item[feature] for item in valid_items]).to(device)
    
    # Process task-specific labels
    if self.has_task:
        labels = torch.stack([torch.tensor(item["labels"], dtype=torch.float) for item in valid_items]).to(device)
        out_batch['labels'] = labels
        out_labels = {}
        for task in self.tasks:
            task_type = self.task_types[task]
            if task_type == "multi_class_classification":
                out_labels[task] = labels.long()
            elif task_type in ["binary_classification", "regression"]:
                out_labels[task] = labels.float()
            else:
                raise TypeError(f"Don't know how to tensorify task of type {task_type}!")
        out_batch['stream_labels'] = out_labels
    
    return out_batch