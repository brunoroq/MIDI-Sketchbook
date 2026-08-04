"""Unit tests for the autoregressive GRU model."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from midi_idea_generator.model import (  # noqa: E402
    GRUModel,
    ModelConfigurationError,
    ModelInputError,
)


def _small_model(**overrides: object) -> GRUModel:
    arguments: dict[str, object] = {
        "vocab_size": 17,
        "pad_token_id": 0,
        "embedding_dim": 5,
        "hidden_dim": 7,
        "num_layers": 2,
        "dropout": 0.0,
    }
    arguments.update(overrides)
    return GRUModel(**arguments)  # type: ignore[arg-type]


def test_forward_returns_unnormalised_batch_first_logits() -> None:
    model = _small_model()
    input_ids = torch.tensor([[1, 4, 9, 2], [1, 8, 2, 0]], dtype=torch.long)

    logits = model(input_ids)

    assert logits.shape == (2, 4, 17)
    assert logits.dtype == model.embedding.weight.dtype
    assert model.gru.batch_first is True
    assert model.gru.bidirectional is False
    # An unconstrained projection should not accidentally behave like softmax.
    assert not torch.allclose(logits.sum(dim=-1), torch.ones(2, 4))


def test_forward_accepts_and_optionally_returns_hidden_state() -> None:
    model = _small_model(num_layers=3, hidden_dim=11)
    input_ids = torch.tensor([[1, 3, 2], [1, 6, 2]], dtype=torch.long)
    initial = model.initial_hidden(batch_size=2)

    logits, final = model(input_ids, initial, return_hidden=True)

    assert logits.shape == (2, 3, 17)
    assert initial.shape == (3, 2, 11)
    assert final.shape == (3, 2, 11)
    assert torch.count_nonzero(initial) == 0
    assert torch.count_nonzero(final) > 0


def test_padding_embedding_is_zero_and_receives_no_gradient() -> None:
    model = _small_model(pad_token_id=3)
    input_ids = torch.tensor([[1, 5, 3, 3], [1, 7, 2, 3]], dtype=torch.long)

    loss = model(input_ids).square().mean()
    loss.backward()

    assert torch.count_nonzero(model.embedding.weight[3]) == 0
    assert model.embedding.weight.grad is not None
    assert torch.count_nonzero(model.embedding.weight.grad[3]) == 0
    assert torch.count_nonzero(model.embedding.weight.grad[1]) > 0


def test_model_supports_full_next_token_backward_pass() -> None:
    model = _small_model()
    inputs = torch.tensor([[1, 4, 6], [1, 8, 5]], dtype=torch.long)
    targets = torch.tensor([[4, 6, 2], [8, 5, 2]], dtype=torch.long)

    logits = model(inputs)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, model.vocab_size), targets.reshape(-1)
    )
    loss.backward()

    assert torch.isfinite(loss)
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"vocab_size": 1}, "vocab_size"),
        ({"vocab_size": True}, "vocab_size"),
        ({"pad_token_id": -1}, "pad_token_id"),
        ({"pad_token_id": 17}, "pad_token_id"),
        ({"embedding_dim": 0}, "embedding_dim"),
        ({"hidden_dim": 0}, "hidden_dim"),
        ({"num_layers": 1}, "num_layers"),
        ({"dropout": -0.01}, "dropout"),
        ({"dropout": 1.0}, "dropout"),
        ({"dropout": True}, "dropout"),
    ],
)
def test_invalid_model_configuration_is_rejected(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ModelConfigurationError, match=message):
        _small_model(**overrides)


@pytest.mark.parametrize(
    ("input_ids", "message"),
    [
        (torch.tensor([1, 2], dtype=torch.long), "shape"),
        (torch.empty((0, 2), dtype=torch.long), "empty"),
        (torch.empty((2, 0), dtype=torch.long), "empty"),
        (torch.tensor([[1.0, 2.0]]), "torch.long"),
        (torch.tensor([[-1, 2]], dtype=torch.long), "range"),
        (torch.tensor([[1, 17]], dtype=torch.long), "range"),
    ],
)
def test_invalid_token_tensors_are_rejected(input_ids: object, message: str) -> None:
    with pytest.raises(ModelInputError, match=message):
        _small_model()(input_ids)  # type: ignore[arg-type]


def test_invalid_hidden_state_is_rejected() -> None:
    model = _small_model()
    input_ids = torch.tensor([[1, 2], [1, 2]], dtype=torch.long)

    with pytest.raises(ModelInputError, match="shape"):
        model(input_ids, torch.zeros(2, 1, 7))
    with pytest.raises(ModelInputError, match="floating-point"):
        model(input_ids, torch.zeros(2, 2, 7, dtype=torch.long))
    with pytest.raises(ModelInputError, match="return_hidden"):
        model(input_ids, return_hidden=1)  # type: ignore[arg-type]
    with pytest.raises(ModelInputError, match="batch_size"):
        model.initial_hidden(0)
    with pytest.raises(ModelInputError, match="floating point"):
        model.initial_hidden(2, dtype=torch.long)


def test_parameter_count_matches_gru_architecture() -> None:
    vocab_size = 17
    embedding_dim = 5
    hidden_dim = 7
    num_layers = 3
    model = _small_model(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.25,
    )

    embedding_parameters = vocab_size * embedding_dim
    first_gru_layer = 3 * hidden_dim * embedding_dim + 3 * hidden_dim**2 + 6 * hidden_dim
    later_gru_layers = (num_layers - 1) * (
        6 * hidden_dim**2 + 6 * hidden_dim
    )
    projection_parameters = hidden_dim * vocab_size + vocab_size
    expected = (
        embedding_parameters
        + first_gru_layer
        + later_gru_layers
        + projection_parameters
    )

    assert model.num_parameters == expected
    assert model.num_parameters == sum(p.numel() for p in model.parameters())
    assert model.gru.dropout == pytest.approx(0.25)

