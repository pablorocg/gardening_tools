from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class NumericalTokenizer(nn.Module):
    """
    Tokenizes numerical features in FT-Transformer style.

    For feature i with value x_i:
        token_i = name_embed[i] + W_i * x_i + b_i

    Missing values (NaN) are replaced by name_embed[i] + missing_token,
    so the field identity is preserved for the downstream transformer.
    """

    def __init__(self, d_model: int, feature_names: list[str]):
        super().__init__()
        self.feature_names = list(feature_names)
        n = len(self.feature_names)

        self.name_embed = nn.Embedding(n, d_model)
        self.weight = nn.Parameter(torch.randn(n, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n, d_model))
        self.missing_token = nn.Parameter(torch.zeros(d_model))

        self.register_buffer("feat_ids", torch.arange(n), persistent=False)

    @property
    def num_features(self) -> int:
        return len(self.feature_names)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, n_cont) float tensor, NaN where the value is missing.

        Returns:
            (B, n_cont, d_model) tokens.
        """
        is_missing = x.isnan()
        x_safe = x.nan_to_num(0.0)

        # value projection: (B, n) x (n, d) -> (B, n, d)
        value = x_safe.unsqueeze(-1) * self.weight + self.bias

        # name embeddings: (n, d), broadcast over batch
        name = self.name_embed(self.feat_ids)
        tokens = name + value

        # replace missing positions with name + missing_token (identity preserved)
        missing_tok = name + self.missing_token  # (n, d)
        tokens = torch.where(
            is_missing.unsqueeze(-1),
            missing_tok.unsqueeze(0).expand_as(tokens),
            tokens,
        )
        return tokens


class CategoricalTokenizer(nn.Module):
    """
    Tokenizes categorical features in FT-Transformer style.

    For feature i with category id c_i:
        token_i = name_embed[i] + value_embed[offset_i + c_i]

    All categorical values share a single embedding table indexed via per-feature
    offsets. Each feature reserves one extra slot for the missing token, so a -1
    in the input maps to that slot.
    """

    def __init__(self, d_model: int, cat_cardinalities: dict[str, int]):
        super().__init__()
        self.feature_names = list(cat_cardinalities.keys())
        cards = list(cat_cardinalities.values())
        n = len(cards)

        # one extra row per feature for the missing token
        sizes = [c + 1 for c in cards]
        cumsizes = torch.tensor([0, *sizes[:-1]], dtype=torch.long).cumsum(0)
        # offsets[i] is the starting row of feature i in the shared embedding
        offsets = cumsizes
        # missing row for feature i sits at offsets[i] + cards[i] (last slot)
        missing_idx = offsets + torch.tensor(cards, dtype=torch.long)

        self.name_embed = nn.Embedding(n, d_model)
        self.value_embed = nn.Embedding(sum(sizes), d_model)

        self.register_buffer("feat_ids", torch.arange(n), persistent=False)
        self.register_buffer("offsets", offsets, persistent=False)
        self.register_buffer("missing_idx", missing_idx, persistent=False)

    @property
    def num_features(self) -> int:
        return len(self.feature_names)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, n_cat) integer tensor, -1 where the value is missing.

        Returns:
            (B, n_cat, d_model) tokens.
        """
        is_missing = x == -1
        # add offsets to remap local ids to global ids in the shared table
        ids = x + self.offsets  # (B, n_cat)
        ids = torch.where(is_missing, self.missing_idx.expand_as(ids), ids)

        value = self.value_embed(ids)             # (B, n_cat, d)
        name = self.name_embed(self.feat_ids)     # (n_cat, d)
        return name + value


class MetadataTokenizer(nn.Module):
    """
    An FT-Transformer-based tokenizer for metadata fields.

    Converts a mixed batch of numerical and categorical metadata into a sequence
    of tokens. Each token is a sum of a field name embedding and a value
    representation:
        - Categorical: name_embed[i] + value_embed[i, c_i]
        - Numerical:   name_embed[i] + W_i * x_i + b_i

    Supports MAE-style masking by replacing selected positions with
    name_embed[i] + mask_token, so the model knows which field was masked.
    """

    def __init__(
        self,
        d_model: int,
        continuous_features: list[str],
        cat_cardinalities: dict[str, int],
    ):
        super().__init__()
        if not continuous_features and not cat_cardinalities:
            raise ValueError("MetadataTokenizer needs at least one feature.")

        self.d_model = d_model
        self.numerical = (
            NumericalTokenizer(d_model, continuous_features)
            if continuous_features
            else None
        )
        self.categorical = (
            CategoricalTokenizer(d_model, cat_cardinalities)
            if cat_cardinalities
            else None
        )
        self.mask_token = nn.Parameter(torch.zeros(d_model))

    @property
    def n_cont(self) -> int:
        return self.numerical.num_features if self.numerical is not None else 0

    @property
    def n_cat(self) -> int:
        return self.categorical.num_features if self.categorical is not None else 0

    @property
    def n_tokens(self) -> int:
        return self.n_cont + self.n_cat

    @property
    def feature_names(self) -> list[str]:
        names: list[str] = []
        if self.numerical is not None:
            names += self.numerical.feature_names
        if self.categorical is not None:
            names += self.categorical.feature_names
        return names

    def _all_name_embeddings(self) -> Tensor:
        """Concatenated name embeddings in the same order as the output tokens."""
        chunks = []
        if self.numerical is not None:
            chunks.append(self.numerical.name_embed(self.numerical.feat_ids))
        if self.categorical is not None:
            chunks.append(self.categorical.name_embed(self.categorical.feat_ids))
        return torch.cat(chunks, dim=0)  # (n_total, d_model)

    def forward(
        self,
        cont: Tensor | None = None,
        cat: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            cont: (B, n_cont) float tensor with NaN for missing values, or None
                  if no continuous features were registered.
            cat:  (B, n_cat) long tensor with -1 for missing values, or None.
            mask: (B, n_tokens) boolean tensor, True where the token should be
                  replaced by the mask token. Order is [continuous, categorical].

        Returns:
            (B, n_tokens, d_model) tensor of tokens.
        """
        chunks: list[Tensor] = []
        if self.numerical is not None:
            if cont is None:
                raise ValueError("cont tensor is required when continuous_features is non empty.")
            chunks.append(self.numerical(cont))
        if self.categorical is not None:
            if cat is None:
                raise ValueError("cat tensor is required when cat_cardinalities is non empty.")
            chunks.append(self.categorical(cat))

        tokens = torch.cat(chunks, dim=1)  # (B, n_tokens, d_model)

        if mask is not None:
            # masked token = field name + shared mask vector (preserves field identity)
            names = self._all_name_embeddings()           # (n_tokens, d)
            masked = names + self.mask_token              # (n_tokens, d)
            tokens = torch.where(
                mask.unsqueeze(-1),
                masked.unsqueeze(0).expand_as(tokens),
                tokens,
            )
        return tokens





