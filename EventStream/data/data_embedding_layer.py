import enum
from typing import Union

import torch

from ..utils import StrEnum
from .types import PytorchBatch


class EmbeddingMode(StrEnum):
    """The different ways that the data can be embedded."""

    JOINT = enum.auto()
    """Embed all data jointly via a single embedding layer, weighting observed measurement embdddings by
    values when present."""

    SPLIT_CATEGORICAL_NUMERICAL = enum.auto()
    """Embed the categorical observations of measurements separately from their numerical values, and combine
    the two via a specifiable strategy."""


class MeasIndexGroupOptions(StrEnum):
    """The different ways that the `split_by_measurement_indices` argument can be interpreted.

    If measurements are split, then the final embedding can be seen as a combination of
    ``emb_cat(measurement_indices)`` and ``emb_num(measurement_indices, measurement_values)``, where ``emb_*``
    are embedding layers with sum aggregations that take in indices to be embedded and possible values to use
    in the output sum. This enumeration controls how those two elements are combined for a given measurement
    feature.
    """

    CATEGORICAL_ONLY = enum.auto()
    """Only embed the categorical component of this measurement (``emb_cat(...)``)."""

    CATEGORICAL_AND_NUMERICAL = enum.auto()
    """Embed both the categorical features and the numerical features of this measurement."""

    NUMERICAL_ONLY = enum.auto()
    """Only embed the numerical component of this measurement (``emb_num(...)``)."""


MEAS_INDEX_GROUP_T = Union[int, tuple[int, MeasIndexGroupOptions]]


class StaticEmbeddingMode(StrEnum):
    """The different ways that static embeddings can be combined with the dynamic embeddings."""

    DROP = enum.auto()
    """Static embeddings are dropped, and only the dynamic embeddings are used."""

    SUM_ALL = enum.auto()
    """Static embeddings are summed with the dynamic embeddings per event."""


class DataEmbeddingLayer(torch.nn.Module):
    """This class efficiently embeds an `PytorchBatch` into a fixed-size embedding.

    This embeds the `PytorchBatch`'s dynamic and static indices into a fixed-size embedding via a PyTorch
    `EmbeddingBag` layer, weighted by the batch's ``dynamic_values`` (respecting ``dynamic_values_mask``).
    This layer assumes a padding index of 0, as that is how the `PytorchDataset` object is structured.
    layer, taking into account `dynamic_indices` (including an implicit padding index of 0), It *does not*
    take into account the time component of the events; that should be embedded separately.

    It has two possible embedding modes; a joint embedding mode, in which categorical data and numerical
    values are embedded jointly through a unified feature map, which effectively equates to a constant value
    imputation strategy with value 1 for missing numerical values, and a split embedding mode, in which
    categorical data and numerical values that are present are embedded through separate feature maps, which
    equates to an imputation strategy of zero imputation (equivalent to mean imputation given normalization)
    and indicator variables indicating present variables. This further follows (roughly) the embedding
    strategy of :footcite:t:`gorishniy2021revisiting` (`link`_) for joint embedding of categorical and
    multi-variate numerical features. In particular, given categorical indices and associated continuous
    values, it produces a categorical embedding of the indices first, then (with a separate embedding layer)
    re-embeds those categorical indices that have associated values observed, this time weighted by the
    associated numerical values, then outputs a weighted sum of the two embeddings. In the case that numerical
    and categorical output embeddings are distinct, both are projected into the output dimensionality through
    additional linear layers prior to the final summation.

    The model uses the joint embedding mode if categorical and numerical embedding dimensions are not
    specified; otherwise, it uses the split embedding mode.

    .. _link: https://openreview.net/pdf?id=i_Q1yrOegLY

    .. footbibliography::

    Args:
        n_total_embeddings: The total vocabulary size that needs to be embedded.
        out_dim: The output dimension of the embedding layer.
        static_embedding_mode: The way that static embeddings are combined with the dynamic embeddings.
        categorical_embedding_dim: The dimension of the categorical embeddings. If `None`, no separate
            categorical embeddings are used.
        numerical_embedding_dim: The dimension of the numerical embeddings. If `None`, no separate numerical
            embeddings are used.
        split_by_measurement_indices: If not `None`, then the `dynamic_indices` are split into multiple
            groups, and each group is embedded separately. The `split_by_measurement_indices` argument is a
            list of lists of indices. Each inner list is a group of indices that will be embedded separately.
            Each index can be an integer, in which case it is the index of the measurement to be embedded, or
            it can be a tuple of the form ``(index, meas_index_group_mode)``, in which case ``index`` is the
            index of the measurement to be embedded, and ``meas_index_group_mode`` indicates whether the group
            includes only the categorical index of the measurement, only the numerical value of the
            measurement, or both its categorical index and it's numerical values, as specified through the
            `MeasIndexGroupOptions` enum. Note that measurement index groups are assumed to only apply to the
            dynamic indices, not the static indices, as static indices are never generated and should be
            assumed to be causally linked to all elements of a given event. Furthermore, note that if
            specified, no measurement group **except for the first** can be empty. The first is allowed to be
            empty to account for settings where a model is built with a dependency graph with no
            `FUNCTIONAL_TIME_DEPENDENT` measures, as time is always assumed to be the first element of the
            dependency graph.
        do_normalize_by_measurement_index: If `True`, then the embeddings of each measurement are normalized
            by the number of measurements of that `measurement_index` in the batch.
        static_weight: The weight of the static embeddings. Only used if `static_embedding_mode` is not
            `StaticEmbeddingMode.DROP`.
        dynamic_weight: The weight of the dynamic embeddings. Only used if `static_embedding_mode` is not
            `StaticEmbeddingMode.DROP`.
        categorical_weight: The weight of the categorical embeddings. Only used if `categorical_embedding_dim`
            and `numerical_embedding_dim` are not `None`.
        numerical_weight: The weight of the numerical embeddings. Only used if `categorical_embedding_dim` and
            `numerical_embedding_dim` are not `None`.

    Raises:
        TypeError: If any of the arguments are of the wrong type.
        ValueError: If any of the arguments are not valid.

    Examples:
        >>> valid_layer = DataEmbeddingLayer(
        ...     n_total_embeddings=100,
        ...     out_dim=10,
        ...     static_embedding_mode=StaticEmbeddingMode.DROP,
        ... )
        >>> valid_layer.embedding_mode
        <EmbeddingMode.JOINT: 'joint'>
        >>> valid_layer = DataEmbeddingLayer(
        ...     n_total_embeddings=100,
        ...     out_dim=10,
        ...     static_embedding_mode=StaticEmbeddingMode.DROP,
        ...     categorical_embedding_dim=5,
        ...     numerical_embedding_dim=5,
        ...     split_by_measurement_indices=None,
        ...     do_normalize_by_measurement_index=False,
        ...     categorical_weight=1 / 2,
        ...     numerical_weight=1 / 2,
        ... )
        >>> valid_layer.embedding_mode
        <EmbeddingMode.SPLIT_CATEGORICAL_NUMERICAL: 'split_categorical_numerical'>
        >>> DataEmbeddingLayer(
        ...     n_total_embeddings=100,
        ...     out_dim="10",
        ...     static_embedding_mode=StaticEmbeddingMode.DROP,
        ... )
        Traceback (most recent call last):
            ...
        TypeError: `out_dim` must be an `int`.
        >>> DataEmbeddingLayer(
        ...     n_total_embeddings=100,
        ...     out_dim=-10,
        ...     static_embedding_mode=StaticEmbeddingMode.DROP,
        ... )
        Traceback (most recent call last):
            ...
        ValueError: `out_dim` must be positive.
        >>> DataEmbeddingLayer(
        ...     n_total_embeddings="100",
        ...     out_dim=10,
        ...     static_embedding_mode=StaticEmbeddingMode.DROP,
        ... )
        Traceback (most recent call last):
            ...
        TypeError: `n_total_embeddings` must be an `int`.
        >>> DataEmbeddingLayer(
        ...     n_total_embeddings=-100,
        ...     out_dim=10,
        ...     static_embedding_mode=StaticEmbeddingMode.DROP,
        ... )
        Traceback (most recent call last):
            ...
        ValueError: `n_total_embeddings` must be positive.
        >>> DataEmbeddingLayer(
        ...     n_total_embeddings=100,
        ...     out_dim=10,
        ...     static_embedding_mode=StaticEmbeddingMode.DROP,
        ...     categorical_embedding_dim=5,
        ...     numerical_embedding_dim=5,
        ...     split_by_measurement_indices=[4, (5, MeasIndexGroupOptions.CATEGORICAL_ONLY)],
        ... )
        Traceback (most recent call last):
            ...
        TypeError: `split_by_measurement_indices` must be a list of lists.
        >>> DataEmbeddingLayer(
        ...     n_total_embeddings=100,
        ...     out_dim=10,
        ...     static_embedding_mode=StaticEmbeddingMode.DROP,
        ...     categorical_embedding_dim=5,
        ...     numerical_embedding_dim=5,
        ...     split_by_measurement_indices=[[4, [5, MeasIndexGroupOptions.CATEGORICAL_ONLY]]],
        ... )
        Traceback (most recent call last):
            ...
        TypeError: `split_by_measurement_indices` must be a list of lists of ints and/or tuples.
    """

    def __init__(
        self,
        n_total_embeddings: int,
        out_dim: int,
        static_embedding_mode: StaticEmbeddingMode = StaticEmbeddingMode.DROP,
        categorical_embedding_dim: int | None = None,
        numerical_embedding_dim: int | None = None,
        split_by_measurement_indices: list[list[MEAS_INDEX_GROUP_T]] | None = None,
        do_normalize_by_measurement_index: bool = False,
        static_weight: float = 1 / 2,
        dynamic_weight: float = 1 / 2,
        categorical_weight: float = 1 / 2,
        numerical_weight: float = 1 / 2,
        oov_index: int | None = None,
    ):
        super().__init__()
        self.n_total_embeddings = n_total_embeddings
        self.embedding = torch.nn.Linear(n_total_embeddings, out_dim, bias=False)
        self.out_dim = out_dim
        self.static_embedding_mode = static_embedding_mode
        self.do_normalize_by_measurement_index = do_normalize_by_measurement_index
        self.oov_index = oov_index

        if categorical_embedding_dim is None and numerical_embedding_dim is None:
            self.embedding_mode = EmbeddingMode.JOINT
            self.data_embedding_layer = torch.nn.EmbeddingBag(
                num_embeddings=n_total_embeddings,
                embedding_dim=out_dim,
                mode="sum",
                padding_idx=0,
                sparse=True,
            )
            if self.oov_index is not None:
                self.data_embedding_layer.weight.data[self.oov_index] = 0
        else:
            self.embedding_mode = EmbeddingMode.SPLIT_CATEGORICAL_NUMERICAL
            self.categorical_embed_layer = torch.nn.EmbeddingBag(
                num_embeddings=n_total_embeddings,
                embedding_dim=categorical_embedding_dim,
                mode="sum",
                padding_idx=0,
                sparse=True,
            )
            self.numerical_embed_layer = torch.nn.EmbeddingBag(
                num_embeddings=n_total_embeddings,
                embedding_dim=numerical_embedding_dim,
                mode="sum",
                padding_idx=0,
                sparse=True,
            )
            if self.oov_index is not None:
                self.categorical_embed_layer.weight.data[self.oov_index] = 0
                self.numerical_embed_layer.weight.data[self.oov_index] = 0
            self.cat_proj = torch.nn.Linear(categorical_embedding_dim, out_dim)
            self.num_proj = torch.nn.Linear(numerical_embedding_dim, out_dim)

        self.static_weight = static_weight / (static_weight + dynamic_weight)
        self.dynamic_weight = dynamic_weight / (static_weight + dynamic_weight)
        self.categorical_weight = categorical_weight / (categorical_weight + numerical_weight)
        self.numerical_weight = numerical_weight / (categorical_weight + numerical_weight)

    def forward(self, indices):
        one_hot = F.one_hot(indices, num_classes=self.n_total_embeddings).float()
        return self.embedding(one_hot)

    def _dynamic_embedding(self, batch: PytorchBatch) -> torch.Tensor:
        dynamic_indices = batch.dynamic_indices
        dynamic_counts = batch.dynamic_counts
        measurement_indices = getattr(batch, 'dynamic_measurement_indices', None)
        values = getattr(batch, 'dynamic_values', None)
        values_mask = getattr(batch, 'dynamic_values_mask', None)

        return self._embed(dynamic_indices, measurement_indices, values, values_mask, None)

    def _embed(
        self,
        indices: torch.Tensor,
        measurement_indices: torch.Tensor | None,
        values: torch.Tensor | None,
        values_mask: torch.Tensor | None,
        cat_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.oov_index is not None:
            indices = torch.where(indices >= self.n_total_embeddings, self.oov_index, indices)

        if self.embedding_mode == EmbeddingMode.JOINT:
            return self._joint_embed(indices, measurement_indices, values, values_mask)
        else:
            return self._split_embed(indices, measurement_indices, values, values_mask, cat_mask)

    def _joint_embed(
        self,
        indices: torch.Tensor,
        measurement_indices: torch.Tensor | None,
        values: torch.Tensor | None,
        values_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if values is None:
            values = torch.ones_like(indices, dtype=torch.float32)
        else:
            values = torch.where(values_mask, values, torch.ones_like(values))

        if self.do_normalize_by_measurement_index and measurement_indices is not None:
            values *= self.get_measurement_index_normalziation(measurement_indices)

        return self.data_embedding_layer(indices, per_sample_weights=values)

    def _split_embed(
        self,
        indices: torch.Tensor,
        measurement_indices: torch.Tensor | None,
        values: torch.Tensor | None,
        values_mask: torch.Tensor | None,
        cat_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        cat_values = torch.ones_like(indices, dtype=torch.float32)
        if cat_mask is not None:
            cat_values = torch.where(cat_mask, cat_values, torch.zeros_like(cat_values))

        if self.do_normalize_by_measurement_index and measurement_indices is not None:
            meas_norm = self.get_measurement_index_normalziation(measurement_indices)
            cat_values *= meas_norm

        cat_embeds = self.cat_proj(self.categorical_embed_layer(indices, per_sample_weights=cat_values))

        if values is None:
            return cat_embeds

        num_values = torch.where(values_mask, values, torch.zeros_like(values))
        if self.do_normalize_by_measurement_index and measurement_indices is not None:
            num_values *= meas_norm

        num_embeds = self.num_proj(self.numerical_embed_layer(indices, per_sample_weights=num_values))

        return self.categorical_weight * cat_embeds + self.numerical_weight * num_embeds

    @staticmethod
    def get_measurement_index_normalziation(measurement_indices: torch.Tensor) -> torch.Tensor:
        one_hot = torch.nn.functional.one_hot(measurement_indices)
        normalization_vals = 1.0 / one_hot.sum(dim=-2)
        normalization_vals = torch.gather(normalization_vals, dim=-1, index=measurement_indices)
        normalization_vals = torch.where(measurement_indices == 0, 0, normalization_vals)
        normalization_vals_sum = normalization_vals.sum(dim=-1, keepdim=True)
        normalization_vals_sum = torch.where(normalization_vals_sum == 0, 1, normalization_vals_sum)
        return normalization_vals / normalization_vals_sum

    def _split_embed(
        self,
        indices: torch.Tensor,
        measurement_indices: torch.Tensor,
        values: torch.Tensor | None = None,
        values_mask: torch.Tensor | None = None,
        cat_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cat_values = torch.ones_like(indices, dtype=torch.float32)
        if cat_mask is not None:
            cat_values = torch.where(cat_mask, cat_values, 0)
        if self.do_normalize_by_measurement_index:
            meas_norm = self.get_measurement_index_normalziation(measurement_indices)
            cat_values *= meas_norm

        cat_embeds = self.cat_proj(self.categorical_embed_layer(indices, per_sample_weights=cat_values))

        if values is None:
            return cat_embeds

        num_values = torch.where(values_mask, values, 0)
        if self.do_normalize_by_measurement_index:
            num_values *= meas_norm

        # Check if num_values is empty
        if num_values.numel() == 0:
            num_embeds = torch.zeros_like(cat_embeds)
        else:
            num_values = num_values.view(indices.shape)
            num_embeds = self.num_proj(self.numerical_embed_layer(indices, per_sample_weights=num_values))

        return self.categorical_weight * cat_embeds + self.numerical_weight * num_embeds

    def _embed(
        self,
        indices: torch.Tensor,
        measurement_indices: torch.Tensor,
        values: torch.Tensor | None = None,
        values_mask: torch.Tensor | None = None,
        cat_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Map OOV indices to the special OOV token index
        if self.oov_index is not None:
            indices = torch.where(indices >= self.n_total_embeddings, self.oov_index, indices)
        
        torch._assert(
            indices.max() < self.n_total_embeddings,
            f"Invalid embedding! {indices.max()} >= {self.n_total_embeddings}",
        )

        match self.embedding_mode:
            case EmbeddingMode.JOINT:
                return self._joint_embed(indices, measurement_indices, values, values_mask)
            case EmbeddingMode.SPLIT_CATEGORICAL_NUMERICAL:
                return self._split_embed(indices, measurement_indices, values, values_mask, cat_mask)
            case _:
                raise ValueError(f"Invalid embedding mode: {self.embedding_mode}")

    def _static_embedding(self, batch: PytorchBatch) -> torch.Tensor:
        """Returns the embedding of the static features of the input batch.

        Args:
            batch: The input batch to be embedded.
        """
        return self._embed(batch["static_indices"], batch["static_measurement_indices"])

    def _split_batch_into_measurement_index_buckets(
        self, batch: PytorchBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Splits the batch into groups of measurement indices.

        Given a batch of data and the list of measurement index groups passed at construction, this function
        produces categorical and numerical values masks for each of the measurement groups. The measurement
        groups (except for the first, which is reserved for `FUNCTIONAL_TIME_DEPENDENT` measurements, which
        may not be present) must not be empty.

        Args:
            batch: A batch of data.

        Returns:
            A tuple of tensors that contain the categorical mask and values mask for each group.

        Raises:
            ValueError: if either there is an empty measurement group beyond the first or there is an invalid
                specified group mode.
        """
        batch_size, sequence_length, num_data_elements = batch["dynamic_measurement_indices"].shape

        categorical_masks = []
        numerical_masks = []
        for i, meas_index_group in enumerate(self.split_by_measurement_indices):
            if len(meas_index_group) == 0 and i > 0:
                raise ValueError(
                    f"Empty measurement index group: {meas_index_group} at index {i}! "
                    "Only the first (i=0) group can be empty (in cases where there are no "
                    "FUNCTIONAL_TIME_DEPENDENT measurements)."
                )

            # Create a mask that is True if each data element in the batch is in the measurement group.
            group_categorical_mask = torch.zeros_like(batch["dynamic_measurement_indices"], dtype=torch.bool)
            group_values_mask = torch.zeros_like(batch["dynamic_measurement_indices"], dtype=torch.bool)
            for meas_index in meas_index_group:
                if type(meas_index) is tuple:
                    meas_index, group_mode = meas_index
                else:
                    group_mode = MeasIndexGroupOptions.CATEGORICAL_AND_NUMERICAL

                new_index_mask = batch["dynamic_measurement_indices"] == meas_index
                match group_mode:
                    case MeasIndexGroupOptions.CATEGORICAL_AND_NUMERICAL:
                        group_categorical_mask |= new_index_mask
                        group_values_mask |= new_index_mask
                    case MeasIndexGroupOptions.CATEGORICAL_ONLY:
                        group_categorical_mask |= new_index_mask
                    case MeasIndexGroupOptions.NUMERICAL_ONLY:
                        group_values_mask |= new_index_mask
                    case _:
                        raise ValueError(f"Invalid group mode: {group_mode}")

            categorical_masks.append(group_categorical_mask)
            numerical_masks.append(group_values_mask)

        return torch.stack(categorical_masks, dim=-2), torch.stack(numerical_masks, dim=-2)

    def _dynamic_embedding(self, batch: PytorchBatch) -> torch.Tensor:
        dynamic_indices = batch.dynamic_indices
        dynamic_counts = batch.dynamic_counts
        measurement_indices = getattr(batch, 'dynamic_measurement_indices', None)
        values = getattr(batch, 'dynamic_values', None)
        values_mask = getattr(batch, 'dynamic_values_mask', None)

        return self._embed(dynamic_indices, measurement_indices, values, values_mask, None)

    def forward(self, batch: PytorchBatch | torch.Tensor) -> torch.Tensor:
        if isinstance(batch, torch.Tensor):
            return self._embed(batch, None, None, None, None)
        elif isinstance(batch, PytorchBatch):
            return self._dynamic_embedding(batch)
        else:
            raise TypeError("Input 'batch' should be a PytorchBatch object or a Tensor.")