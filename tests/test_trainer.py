"""Fast behavioural and checkpoint tests for the Stage 3 trainer."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from midi_idea_generator.model import GRUModel
from midi_idea_generator import trainer
from midi_idea_generator.trainer import (
    CHECKPOINT_SCHEMA_VERSION,
    TrainingError,
    resolve_device,
    run_training,
    seed_everything,
    token_cross_entropy,
)
from midi_idea_generator.training_config import (
    DataConfig,
    ModelConfig,
    OptimizationConfig,
    TrainingConfig,
    TrainingPathsConfig,
)


class _SyntheticDataset:
    """Tiny deterministic dataset implementing the trainer-facing contract."""

    vocabulary_size = 12
    pad_token_id = 0
    tokenization_run_id = "synthetic-tokenization-run"
    tokenizer_sha256 = "1" * 64
    configuration_sha256 = "2" * 64
    tokenization_manifest_sha256 = "3" * 64
    technique_token_ids = frozenset({9, 10})
    token_type_by_id = (
        "PAD",
        "BOS",
        "EOS",
        "Bar",
        "Position",
        "Pitch",
        "Duration",
        "Pitch",
        "PitchBend",
        "Technique",
        "Technique",
        "Rest",
    )

    _IDS = {
        "train": (
            (1, 3, 4, 5, 2),
            (1, 4, 6, 2),
            (1, 5, 7, 8, 2),
        ),
        "validation": (
            (1, 3, 6, 2),
            (1, 5, 8, 2),
        ),
    }

    def __init__(self, _manifest_path: Path, split: str, **_kwargs: object) -> None:
        self.split = split
        self._ids = self._IDS[split]

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, index: int) -> dict[str, object]:
        ids = self._ids[index]
        return {
            "input_ids": ids[:-1],
            "target_ids": ids[1:],
            "unknown_technique_decision_mask": (False,) * (len(ids) - 1),
            "length": len(ids) - 1,
            "sequence_id": f"{self.split}-{index}",
            "split": self.split,
        }


class _NullSummaryWriter:
    """Avoid filesystem-heavy TensorBoard event writing in integration tests."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def add_scalar(self, *_args: object, **_kwargs: object) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


def _training_config(root: Path, output_name: str) -> TrainingConfig:
    return TrainingConfig(
        seed=1729,
        device="cpu",
        project_root=root,
        paths=TrainingPathsConfig(
            tokenization_manifest_path=root / "manifest.json",
            checkpoints_dir=root / output_name / "checkpoints",
            tensorboard_log_dir=root / output_name / "tensorboard",
            resume_from=None,
        ),
        model=ModelConfig(
            architecture="gru",
            embedding_dim=4,
            hidden_dim=6,
            num_layers=2,
            dropout=0.15,
        ),
        data=DataConfig(
            max_sequence_length=16,
            batch_size=2,
            num_workers=0,
        ),
        training=OptimizationConfig(
            epochs=2,
            learning_rate=1e-2,
            weight_decay=0.0,
            gradient_clip=1.0,
            mixed_precision="off",
            checkpoint_every_epochs=1,
            early_stopping_patience=10,
            early_stopping_min_delta=0.0,
        ),
    )


def test_resolve_device_supports_cpu_and_auto_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device("auto") == torch.device("cpu")
    with pytest.raises(TrainingError, match="CUDA was requested"):
        resolve_device("cuda")


def test_resolve_device_selects_available_cuda_without_initialising_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert resolve_device("auto") == torch.device("cuda")
    assert resolve_device("cuda") == torch.device("cuda")
    with pytest.raises(TrainingError, match="device must be"):
        resolve_device("mps")


def test_seed_everything_reproduces_python_numpy_and_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    seed_everything(91)
    first = (
        random.random(),
        np.random.random(4),
        torch.rand(4),
    )
    seed_everything(91)
    second = (
        random.random(),
        np.random.random(4),
        torch.rand(4),
    )

    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)


@pytest.mark.parametrize("invalid_seed", [-1, True, 1.5])
def test_seed_everything_rejects_invalid_values(invalid_seed: object) -> None:
    with pytest.raises(TrainingError, match="non-negative integer"):
        seed_everything(invalid_seed)  # type: ignore[arg-type]


def test_token_cross_entropy_ignores_pad_in_loss_count_and_gradient() -> None:
    logits = torch.tensor(
        [
            [
                [0.2, 1.7, -0.1, 0.3],
                [5.0, -5.0, 2.0, 1.0],
                [-0.2, 0.1, 1.3, 0.0],
            ]
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    targets = torch.tensor([[1, 0, 2]], dtype=torch.long)

    loss_sum, num_tokens = token_cross_entropy(logits, targets, pad_token_id=0)
    expected = F.cross_entropy(
        torch.stack((logits[0, 0], logits[0, 2])),
        torch.tensor([1, 2]),
        reduction="sum",
    )
    loss_sum.backward()

    assert num_tokens.item() == 2
    torch.testing.assert_close(loss_sum.detach(), expected.detach())
    assert logits.grad is not None
    torch.testing.assert_close(logits.grad[0, 1], torch.zeros(4))
    assert torch.count_nonzero(logits.grad[0, 0]).item() > 0
    assert torch.count_nonzero(logits.grad[0, 2]).item() > 0


def test_partial_label_loss_trains_base_target_without_technique_gradients() -> None:
    logits = torch.tensor(
        [[[0.0, 0.5, 1.0, 4.0, 3.0, -0.5]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    targets = torch.tensor([[2]], dtype=torch.long)
    unknown = torch.tensor([[True]], dtype=torch.bool)

    loss_sum, num_tokens = token_cross_entropy(
        logits,
        targets,
        pad_token_id=0,
        unknown_technique_decision_mask=unknown,
        technique_token_ids=(3, 4),
    )
    expected = F.cross_entropy(
        logits[0, 0, [0, 1, 2, 5]].unsqueeze(0),
        torch.tensor([2]),
        reduction="sum",
    )
    loss_sum.backward()

    assert num_tokens.item() == 1
    torch.testing.assert_close(loss_sum.detach(), expected.detach())
    assert logits.grad is not None
    torch.testing.assert_close(logits.grad[0, 0, 3:5], torch.zeros(2))
    assert logits.grad[0, 0, 2].item() < 0
    assert torch.count_nonzero(logits.grad[0, 0, [0, 1, 5]]).item() == 3


def test_complete_post_duration_decision_uses_full_vocabulary_loss() -> None:
    logits = torch.tensor(
        [[[0.0, 0.5, 1.0, 4.0, 3.0, -0.5]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    targets = torch.tensor([[2]], dtype=torch.long)

    loss_sum, _ = token_cross_entropy(
        logits,
        targets,
        pad_token_id=0,
        unknown_technique_decision_mask=torch.tensor([[False]]),
        technique_token_ids=(3, 4),
    )
    expected = F.cross_entropy(logits[0], targets[0], reduction="sum")
    loss_sum.backward()

    torch.testing.assert_close(loss_sum.detach(), expected.detach())
    assert logits.grad is not None
    assert logits.grad[0, 0, 3].item() > 0
    assert logits.grad[0, 0, 4].item() > 0


def test_partial_label_loss_rejects_technique_target_as_unknown() -> None:
    logits = torch.zeros((1, 1, 6), dtype=torch.float32)

    with pytest.raises(TrainingError, match="cannot be a Technique"):
        token_cross_entropy(
            logits,
            torch.tensor([[3]], dtype=torch.long),
            pad_token_id=0,
            unknown_technique_decision_mask=torch.tensor([[True]]),
            technique_token_ids=(3, 4),
        )


def test_token_cross_entropy_rejects_all_padding() -> None:
    logits = torch.zeros((2, 3, 5), dtype=torch.float32)
    targets = torch.zeros((2, 3), dtype=torch.long)

    with pytest.raises(TrainingError, match="only padding"):
        token_cross_entropy(logits, targets, pad_token_id=0)


def test_train_epoch_backpropagates_and_counts_only_real_targets() -> None:
    seed_everything(7)
    model = GRUModel(
        9,
        0,
        embedding_dim=4,
        hidden_dim=6,
        num_layers=2,
        dropout=0.0,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    batch = {
        "input_ids": torch.tensor([[1, 3, 4], [1, 4, 0]], dtype=torch.long),
        "target_ids": torch.tensor([[3, 4, 2], [4, 2, 0]], dtype=torch.long),
        "unknown_technique_decision_mask": torch.zeros(
            (2, 3), dtype=torch.bool
        ),
    }
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    loss, token_count, gradient_norm, diagnostics = trainer._train_epoch(
        model,
        [batch],  # type: ignore[arg-type]
        optimizer,
        scaler,
        torch.device("cpu"),
        pad_token_id=0,
        gradient_clip=1.0,
        amp_enabled=False,
        technique_token_ids=(7, 8),
        token_type_by_id=(
            "PAD",
            "BOS",
            "EOS",
            "Bar",
            "Position",
            "Pitch",
            "Duration",
            "Technique",
            "Technique",
        ),
    )

    assert token_count == 5
    assert math.isfinite(loss) and loss > 0
    assert math.isfinite(gradient_norm) and gradient_norm > 0
    assert diagnostics["total"]["count"] == 5
    assert diagnostics["post_duration_unknown"]["count"] == 0
    assert diagnostics["total"]["objective_nll"] == pytest.approx(loss)
    assert any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in model.named_parameters()
    )
    torch.testing.assert_close(model.embedding.weight[0], torch.zeros(4))


def test_atomic_checkpoint_round_trip_uses_weights_only(tmp_path: Path) -> None:
    checkpoint = tmp_path / "nested" / "latest.pt"
    payload: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "weights": torch.arange(5),
        "metadata": {"epoch": 3},
    }

    trainer._atomic_torch_save(checkpoint, payload)
    restored = trainer._load_checkpoint(checkpoint, torch.device("cpu"))

    assert checkpoint.is_file()
    assert restored["metadata"] == {"epoch": 3}
    torch.testing.assert_close(restored["weights"], payload["weights"])
    assert not list(checkpoint.parent.glob(f".{checkpoint.name}-*.tmp"))


def test_load_checkpoint_explicitly_requests_safe_weights_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}

    def fake_load(path: Path, **kwargs: object) -> dict[str, int]:
        observed["path"] = path
        observed.update(kwargs)
        return {"schema_version": CHECKPOINT_SCHEMA_VERSION}

    checkpoint = tmp_path / "checkpoint.pt"
    monkeypatch.setattr(trainer.torch, "load", fake_load)

    trainer._load_checkpoint(checkpoint, torch.device("cpu"))

    assert observed == {
        "path": checkpoint,
        "map_location": torch.device("cpu"),
        "weights_only": True,
    }


def test_corrupt_checkpoint_is_wrapped_as_training_error(tmp_path: Path) -> None:
    checkpoint = tmp_path / "corrupt.pt"
    checkpoint.write_bytes(b"not a PyTorch checkpoint")

    with pytest.raises(TrainingError, match="Could not load checkpoint"):
        trainer._load_checkpoint(checkpoint, torch.device("cpu"))


def test_atomic_checkpoint_failure_preserves_previous_file_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "latest.pt"
    checkpoint.write_bytes(b"previous complete checkpoint")

    def interrupted_save(_payload: object, path: Path) -> None:
        Path(path).write_bytes(b"partial replacement")
        raise RuntimeError("simulated write interruption")

    monkeypatch.setattr(trainer.torch, "save", interrupted_save)

    with pytest.raises(TrainingError, match="Could not save checkpoint"):
        trainer._atomic_torch_save(
            checkpoint, {"schema_version": CHECKPOINT_SCHEMA_VERSION}
        )

    assert checkpoint.read_bytes() == b"previous complete checkpoint"
    assert not list(tmp_path.glob(f".{checkpoint.name}-*.tmp"))


@pytest.mark.parametrize(
    "unsafe_run_id",
    ["../escape", "run/subdirectory", "/tmp/absolute", ".", ".."],
)
def test_checkpoint_training_run_id_rejects_path_components(
    unsafe_run_id: str,
) -> None:
    with pytest.raises(TrainingError):
        trainer._checkpoint_run_id({"training_run_id": unsafe_run_id})


def test_resume_matches_uninterrupted_training_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(trainer, "TokenizedSequenceDataset", _SyntheticDataset)
    monkeypatch.setattr(trainer, "SummaryWriter", _NullSummaryWriter)

    uninterrupted = run_training(
        _training_config(tmp_path, "uninterrupted"), epochs_override=2
    )
    first_half = run_training(
        _training_config(tmp_path, "resumed"), epochs_override=1
    )
    resume_config = _training_config(tmp_path, "resumed")
    resume_config = replace(
        resume_config,
        paths=replace(resume_config.paths, resume_from=first_half.latest_checkpoint),
    )
    resumed = run_training(resume_config, epochs_override=2)

    uninterrupted_payload = trainer._load_checkpoint(
        uninterrupted.latest_checkpoint, torch.device("cpu")
    )
    resumed_payload = trainer._load_checkpoint(
        resumed.latest_checkpoint, torch.device("cpu")
    )
    uninterrupted_state = uninterrupted_payload["model_state_dict"]
    resumed_state = resumed_payload["model_state_dict"]

    assert resumed.start_epoch == 2
    assert resumed.completed_epochs == 2
    assert [metrics.epoch for metrics in resumed.history] == [1, 2]
    assert isinstance(uninterrupted_state, dict)
    assert isinstance(resumed_state, dict)
    assert uninterrupted_state.keys() == resumed_state.keys()
    for name in uninterrupted_state:
        torch.testing.assert_close(
            uninterrupted_state[name], resumed_state[name], rtol=0, atol=0
        )
    assert resumed.history[-1].train_loss == pytest.approx(
        uninterrupted.history[-1].train_loss, rel=0, abs=0
    )
    assert resumed.history[-1].validation_loss == pytest.approx(
        uninterrupted.history[-1].validation_loss, rel=0, abs=0
    )
    assert resumed.history[-1].train_metrics == (
        uninterrupted.history[-1].train_metrics
    )
    assert resumed.history[-1].validation_metrics == (
        uninterrupted.history[-1].validation_metrics
    )


def test_latest_checkpoint_is_published_after_epoch_and_best(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(trainer, "TokenizedSequenceDataset", _SyntheticDataset)
    monkeypatch.setattr(trainer, "SummaryWriter", _NullSummaryWriter)
    saved_names: list[str] = []
    original_save = trainer._atomic_torch_save

    def recording_save(path: Path, payload: dict[str, Any]) -> None:
        saved_names.append(path.name)
        original_save(path, payload)

    monkeypatch.setattr(trainer, "_atomic_torch_save", recording_save)

    run_training(_training_config(tmp_path, "ordered"), epochs_override=1)

    assert saved_names == ["epoch-0001.pt", "best.pt", "latest.pt"]


def test_failed_best_checkpoint_does_not_publish_latest_commit_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(trainer, "TokenizedSequenceDataset", _SyntheticDataset)
    monkeypatch.setattr(trainer, "SummaryWriter", _NullSummaryWriter)
    original_save = trainer._atomic_torch_save

    def fail_on_best(path: Path, payload: dict[str, Any]) -> None:
        if path.name == "best.pt":
            raise TrainingError("simulated best checkpoint failure")
        original_save(path, payload)

    monkeypatch.setattr(trainer, "_atomic_torch_save", fail_on_best)

    with pytest.raises(TrainingError, match="simulated best checkpoint failure"):
        run_training(_training_config(tmp_path, "failed-best"), epochs_override=1)

    run_dirs = list((tmp_path / "failed-best" / "checkpoints").iterdir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "epoch-0001.pt").is_file()
    assert not (run_dirs[0] / "best.pt").exists()
    assert not (run_dirs[0] / "latest.pt").exists()


def test_training_rejects_resume_from_best_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(trainer, "TokenizedSequenceDataset", _SyntheticDataset)
    monkeypatch.setattr(trainer, "SummaryWriter", _NullSummaryWriter)
    config = _training_config(tmp_path, "no-branch")
    first_epoch = run_training(config, epochs_override=1)
    resume_from_best = replace(
        config,
        paths=replace(config.paths, resume_from=first_epoch.best_checkpoint),
    )

    with pytest.raises(TrainingError, match=r"only resume.*latest\.pt"):
        run_training(resume_from_best, epochs_override=2)
