"""Autoregressive GRU language model for symbolic-music token sequences."""

from __future__ import annotations

from typing import overload

import torch
from torch import Tensor, nn


class ModelConfigurationError(ValueError):
    """Raised when a GRU model is constructed with an invalid configuration."""


class ModelInputError(ValueError):
    """Raised when model inputs do not satisfy the recurrent model contract."""


class GRUModel(nn.Module):
    """A small unidirectional GRU language model.

    The model consumes integer token identifiers with shape ``(batch, time)`` and
    emits one unnormalised logit vector per input position.  Callers should use
    these logits directly with :class:`torch.nn.CrossEntropyLoss`; applying a
    softmax in ``forward`` would make that loss both slower and less stable.

    ``PAD`` receives a dedicated ``padding_idx`` in the embedding.  This keeps
    the padding row fixed at zero, while masking padded targets remains the
    responsibility of the training loss.
    """

    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        *,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self._validate_configuration(
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = float(dropout)

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_token_id,
        )
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=False,
        )
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

    @staticmethod
    def _validate_configuration(
        *,
        vocab_size: int,
        pad_token_id: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        integer_fields = {
            "vocab_size": vocab_size,
            "pad_token_id": pad_token_id,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ModelConfigurationError(f"{name} must be an integer")

        if vocab_size < 2:
            raise ModelConfigurationError("vocab_size must be at least 2")
        if not 0 <= pad_token_id < vocab_size:
            raise ModelConfigurationError(
                "pad_token_id must be inside the model vocabulary"
            )
        if embedding_dim < 1:
            raise ModelConfigurationError("embedding_dim must be positive")
        if hidden_dim < 1:
            raise ModelConfigurationError("hidden_dim must be positive")
        if num_layers < 2:
            raise ModelConfigurationError("num_layers must be at least 2")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
            raise ModelConfigurationError("dropout must be a real number")
        if not 0.0 <= float(dropout) < 1.0:
            raise ModelConfigurationError("dropout must be in the range [0, 1)")

    @property
    def num_parameters(self) -> int:
        """Return the number of trainable scalar parameters."""

        return sum(parameter.numel() for parameter in self.parameters())

    def initial_hidden(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Create a zero recurrent state compatible with this model."""

        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ModelInputError("batch_size must be a positive integer")

        reference = self.embedding.weight
        resolved_device = reference.device if device is None else torch.device(device)
        resolved_dtype = reference.dtype if dtype is None else dtype
        if not resolved_dtype.is_floating_point:
            raise ModelInputError("hidden-state dtype must be floating point")
        return torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_dim,
            device=resolved_device,
            dtype=resolved_dtype,
        )

    def _validate_inputs(self, input_ids: Tensor, hidden: Tensor | None) -> None:
        if not isinstance(input_ids, Tensor):
            raise ModelInputError("input_ids must be a torch.Tensor")
        if input_ids.ndim != 2:
            raise ModelInputError("input_ids must have shape (batch, time)")
        if input_ids.dtype != torch.long:
            raise ModelInputError("input_ids must use torch.long token identifiers")
        if input_ids.shape[0] < 1 or input_ids.shape[1] < 1:
            raise ModelInputError("input_ids cannot have an empty batch or time axis")

        minimum_id, maximum_id = torch.aminmax(input_ids)
        if minimum_id.item() < 0 or maximum_id.item() >= self.vocab_size:
            raise ModelInputError(
                f"input_ids must be in the range [0, {self.vocab_size})"
            )

        parameter = self.embedding.weight
        if input_ids.device != parameter.device:
            raise ModelInputError("input_ids and model parameters must share a device")

        if hidden is None:
            return
        if not isinstance(hidden, Tensor):
            raise ModelInputError("hidden must be a torch.Tensor or None")
        expected_shape = (self.num_layers, input_ids.shape[0], self.hidden_dim)
        if tuple(hidden.shape) != expected_shape:
            raise ModelInputError(f"hidden must have shape {expected_shape}")
        if not hidden.dtype.is_floating_point:
            raise ModelInputError("hidden must use a floating-point dtype")
        if hidden.device != parameter.device:
            raise ModelInputError("hidden and model parameters must share a device")
        if hidden.dtype != parameter.dtype:
            raise ModelInputError("hidden and model parameters must share a dtype")

    @overload
    def forward(
        self,
        input_ids: Tensor,
        hidden: Tensor | None = None,
        *,
        return_hidden: bool = False,
    ) -> Tensor: ...

    @overload
    def forward(
        self,
        input_ids: Tensor,
        hidden: Tensor | None = None,
        *,
        return_hidden: bool,
    ) -> Tensor | tuple[Tensor, Tensor]: ...

    def forward(
        self,
        input_ids: Tensor,
        hidden: Tensor | None = None,
        *,
        return_hidden: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Compute next-token logits and, optionally, the final hidden state."""

        if not isinstance(return_hidden, bool):
            raise ModelInputError("return_hidden must be a boolean")
        self._validate_inputs(input_ids, hidden)

        embeddings = self.embedding(input_ids)
        recurrent_output, final_hidden = self.gru(embeddings, hidden)
        logits = self.output_projection(recurrent_output)
        if return_hidden:
            return logits, final_hidden
        return logits


__all__ = [
    "GRUModel",
    "ModelConfigurationError",
    "ModelInputError",
]
