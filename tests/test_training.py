"""Tests for training infrastructure: config, data, checkpoint, trainer."""

import json
import os

import torch
import pytest

from atomiclm.model.decoder import Decoder
from atomiclm.tokenizer.tokenizer import BasicTokenizer
from atomiclm.training.config import TrainingConfig, ModelConfig, DataConfig, OptimConfig
from atomiclm.training.data import TextDataset, create_dataloaders
from atomiclm.training.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint
from atomiclm.training.train import (
    Trainer,
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)

SAMPLE_TEXT = "The quick brown fox jumps over the lazy dog. " * 200
VOCAB_SIZE = 280
D_MODEL = 32
NUM_LAYERS = 2
NUM_HEADS = 2
D_FF = 64
MAX_SEQ_LEN = 64
SEQ_LEN = 16
BATCH_SIZE = 4


@pytest.fixture
def tokenizer():
    tok = BasicTokenizer()
    tok.train(SAMPLE_TEXT, vocab_size=VOCAB_SIZE, verbose=False)
    return tok


@pytest.fixture
def small_model():
    torch.manual_seed(42)
    return Decoder(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS, d_ff=D_FF, max_seq_len=MAX_SEQ_LEN, dropout=0.0,
    )


# ── Config Tests (in-memory) ────────────────────────────────────────


def test_config_defaults():
    config = TrainingConfig()
    assert config.device == "auto"
    assert config.seed == 42
    assert config.optim.scheduler == "cosine"
    assert config.model.dropout == 0.0


def test_config_resolve_device():
    assert TrainingConfig(device="cpu").resolve_device() == "cpu"
    resolved = TrainingConfig(device="auto").resolve_device()
    assert resolved in ("cuda", "mps", "cpu")


def test_config_json_roundtrip(tmp_path):
    config = TrainingConfig(
        model=ModelConfig(d_model=256, num_layers=6, num_heads=8, d_ff=1024),
        optim=OptimConfig(lr=1e-4, warmup_steps=200),
        max_steps=50000,
        run_name="test_run",
    )
    path = str(tmp_path / "config.json")
    config.to_json(path)
    loaded = TrainingConfig.from_json(path)

    assert loaded.model.d_model == 256
    assert loaded.optim.lr == 1e-4
    assert loaded.max_steps == 50000
    assert loaded.run_name == "test_run"


def test_config_partial_json(tmp_path):
    path = str(tmp_path / "partial.json")
    with open(path, "w") as f:
        json.dump({"model": {"d_model": 512}, "max_steps": 1000}, f)

    config = TrainingConfig.from_json(path)
    assert config.model.d_model == 512
    assert config.max_steps == 1000
    assert config.optim.lr == 3e-4  # default preserved


# ── Data Tests ───────────────────────────────────────────────────────


def test_text_dataset_shapes(tmp_path, tokenizer):
    (tmp_path / "train.txt").write_text(SAMPLE_TEXT)
    ds = TextDataset(str(tmp_path / "train.txt"), tokenizer, seq_len=SEQ_LEN)

    item = ds[0]
    assert item["input_ids"].shape == (SEQ_LEN,)
    assert item["target_ids"].shape == (SEQ_LEN,)


def test_text_dataset_shift(tmp_path, tokenizer):
    (tmp_path / "train.txt").write_text(SAMPLE_TEXT)
    ds = TextDataset(str(tmp_path / "train.txt"), tokenizer, seq_len=SEQ_LEN)

    item = ds[0]
    chunk = ds.token_ids[: SEQ_LEN + 1]
    assert (item["input_ids"] == chunk[:-1]).all()
    assert (item["target_ids"] == chunk[1:]).all()


def test_text_dataset_length(tmp_path, tokenizer):
    (tmp_path / "train.txt").write_text(SAMPLE_TEXT)
    ds = TextDataset(str(tmp_path / "train.txt"), tokenizer, seq_len=SEQ_LEN)
    assert len(ds) == len(ds.token_ids) - SEQ_LEN


def test_text_dataset_too_short(tmp_path, tokenizer):
    (tmp_path / "short.txt").write_text("hi")
    with pytest.raises(ValueError, match="Text too short"):
        TextDataset(str(tmp_path / "short.txt"), tokenizer, seq_len=9999)


def test_create_dataloaders_with_val(tmp_path, tokenizer):
    (tmp_path / "train.txt").write_text(SAMPLE_TEXT)
    (tmp_path / "val.txt").write_text(SAMPLE_TEXT[:len(SAMPLE_TEXT) // 2])
    data_config = DataConfig(
        train_path=str(tmp_path / "train.txt"),
        val_path=str(tmp_path / "val.txt"),
        tokenizer_path="", batch_size=BATCH_SIZE, seq_len=SEQ_LEN,
    )
    train_loader, val_loader = create_dataloaders(data_config, tokenizer)

    assert val_loader is not None
    batch = next(iter(train_loader))
    assert batch["input_ids"].shape == (BATCH_SIZE, SEQ_LEN)


def test_create_dataloaders_no_val(tmp_path, tokenizer):
    (tmp_path / "train.txt").write_text(SAMPLE_TEXT)
    data_config = DataConfig(
        train_path=str(tmp_path / "train.txt"),
        tokenizer_path="", batch_size=BATCH_SIZE, seq_len=SEQ_LEN,
    )
    _, val_loader = create_dataloaders(data_config, tokenizer)
    assert val_loader is None


# ── Checkpoint Tests (needs disk) ────────────────────────────────────


def test_checkpoint_roundtrip(tmp_path, small_model):
    optimizer = torch.optim.AdamW(small_model.parameters(), lr=1e-3)
    scheduler = get_constant_schedule_with_warmup(optimizer, warmup_steps=5)

    # Create non-trivial optimizer state
    ids = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN))
    logits, _ = small_model(ids)
    logits.sum().backward()
    optimizer.step()
    scheduler.step()

    config = TrainingConfig(device="cpu")
    ckpt_dir = str(tmp_path / "ckpt")
    save_checkpoint(ckpt_dir, small_model, optimizer, scheduler, 42, config)

    # Load into fresh model
    fresh = Decoder(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS, d_ff=D_FF, max_seq_len=MAX_SEQ_LEN,
    )
    state = load_checkpoint(os.path.join(ckpt_dir, "checkpoint_latest.pt"), fresh)

    assert state["global_step"] == 42
    for p1, p2 in zip(small_model.parameters(), fresh.parameters()):
        torch.testing.assert_close(p1, p2)


def test_find_latest_checkpoint(tmp_path):
    assert find_latest_checkpoint(str(tmp_path)) is None

    latest = str(tmp_path / "checkpoint_latest.pt")
    torch.save({}, latest)
    assert find_latest_checkpoint(str(tmp_path)) == latest


def test_checkpoint_best(tmp_path, small_model):
    optimizer = torch.optim.AdamW(small_model.parameters(), lr=1e-3)
    scheduler = get_constant_schedule_with_warmup(optimizer, warmup_steps=5)
    config = TrainingConfig(device="cpu")

    ckpt_dir = str(tmp_path / "ckpt")
    save_checkpoint(ckpt_dir, small_model, optimizer, scheduler, 10, config, is_best=True)

    assert os.path.exists(os.path.join(ckpt_dir, "checkpoint_best.pt"))
    assert os.path.exists(os.path.join(ckpt_dir, "checkpoint_latest.pt"))
    assert os.path.exists(os.path.join(ckpt_dir, "config.json"))


# ── LR Schedule Tests (in-memory) ───────────────────────────────────


def test_cosine_schedule_warmup():
    optimizer = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps=10, max_steps=100, min_lr_ratio=0.1,
    )
    lrs = []
    for _ in range(100):
        lrs.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()

    assert lrs[9] > lrs[0]       # warmup increases
    assert lrs[50] < lrs[10]     # cosine decays
    assert lrs[-1] >= 0.1 - 1e-6  # floor


def test_constant_schedule_warmup():
    optimizer = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
    scheduler = get_constant_schedule_with_warmup(optimizer, warmup_steps=10)
    lrs = []
    for _ in range(50):
        lrs.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()

    assert abs(lrs[15] - 1.0) < 1e-6
    assert abs(lrs[40] - 1.0) < 1e-6


# ── Trainer Tests ────────────────────────────────────────────────────


def _make_trainer(tmp_path, tokenizer, model, val=False):
    """Helper: create a Trainer with minimal config."""
    (tmp_path / "train.txt").write_text(SAMPLE_TEXT)
    data_cfg = DataConfig(
        train_path=str(tmp_path / "train.txt"),
        tokenizer_path="", batch_size=BATCH_SIZE, seq_len=SEQ_LEN,
    )
    if val:
        (tmp_path / "val.txt").write_text(SAMPLE_TEXT[:len(SAMPLE_TEXT) // 2])
        data_cfg.val_path = str(tmp_path / "val.txt")

    config = TrainingConfig(
        model=ModelConfig(
            vocab_size=VOCAB_SIZE, d_model=D_MODEL, num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS, d_ff=D_FF, max_seq_len=MAX_SEQ_LEN,
        ),
        data=data_cfg,
        optim=OptimConfig(lr=1e-3, warmup_steps=5),
        max_steps=20, eval_interval=10, log_interval=5, save_interval=10,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        device="cpu", seed=42,
    )
    train_loader, val_loader = create_dataloaders(config.data, tokenizer)
    return Trainer(config, model, tokenizer, train_loader, val_loader)


def test_trainer_single_step(tmp_path, tokenizer, small_model):
    trainer = _make_trainer(tmp_path, tokenizer, small_model)
    batch = next(iter(trainer.train_loader))
    loss = trainer._micro_step(batch)

    assert isinstance(loss, float)
    assert loss > 0
    assert not torch.isnan(torch.tensor(loss))


def test_trainer_validation(tmp_path, tokenizer, small_model):
    trainer = _make_trainer(tmp_path, tokenizer, small_model, val=True)
    val_loss = trainer.validate()

    assert isinstance(val_loss, float)
    assert val_loss > 0


def test_trainer_param_grouping(tmp_path, tokenizer, small_model):
    trainer = _make_trainer(tmp_path, tokenizer, small_model)
    groups = trainer.optimizer.param_groups

    assert len(groups) == 2
    assert groups[0]["weight_decay"] > 0
    assert groups[1]["weight_decay"] == 0.0


def test_trainer_seq_len_exceeds_max_seq_len(tmp_path, tokenizer, small_model):
    (tmp_path / "train.txt").write_text(SAMPLE_TEXT)
    config = TrainingConfig(
        model=ModelConfig(
            vocab_size=VOCAB_SIZE, d_model=D_MODEL, num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS, d_ff=D_FF, max_seq_len=MAX_SEQ_LEN,
        ),
        data=DataConfig(
            train_path=str(tmp_path / "train.txt"),
            tokenizer_path="", batch_size=BATCH_SIZE, seq_len=MAX_SEQ_LEN + 1,
        ),
        device="cpu",
    )
    train_loader, val_loader = create_dataloaders(config.data, tokenizer)
    with pytest.raises(ValueError, match="exceeds"):
        Trainer(config, small_model, tokenizer, train_loader, val_loader)


# ── Gradient Accumulation Tests ──────────────────────────────────────


def test_grad_accum_steps_default():
    assert OptimConfig().grad_accum_steps == 1


def test_grad_accum_steps_json_roundtrip(tmp_path):
    config = TrainingConfig(optim=OptimConfig(grad_accum_steps=4))
    path = str(tmp_path / "config.json")
    config.to_json(path)
    loaded = TrainingConfig.from_json(path)
    assert loaded.optim.grad_accum_steps == 4


def test_micro_step_scales_gradients(tmp_path, tokenizer, small_model):
    """_micro_step divides loss by grad_accum_steps before backward."""
    trainer = _make_trainer(tmp_path, tokenizer, small_model)
    batch = next(iter(trainer.train_loader))

    # grad_accum_steps=1 (default): gradients are unscaled
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer._micro_step(batch)
    grads_base = {
        n: p.grad.clone()
        for n, p in trainer.model.named_parameters()
        if p.grad is not None
    }

    # grad_accum_steps=4: each micro-step contributes 1/4 of the gradient
    trainer.config.optim.grad_accum_steps = 4
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer._micro_step(batch)

    for name, p in trainer.model.named_parameters():
        if p.grad is not None:
            torch.testing.assert_close(p.grad, grads_base[name] / 4)


def test_micro_step_returns_unscaled_loss(tmp_path, tokenizer, small_model):
    """_micro_step returns the unscaled loss regardless of grad_accum_steps."""
    trainer = _make_trainer(tmp_path, tokenizer, small_model)
    batch = next(iter(trainer.train_loader))

    trainer.optimizer.zero_grad(set_to_none=True)
    loss_1 = trainer._micro_step(batch)

    trainer.config.optim.grad_accum_steps = 4
    trainer.optimizer.zero_grad(set_to_none=True)
    loss_4 = trainer._micro_step(batch)

    # Same batch, same model weights → same unscaled loss
    assert abs(loss_1 - loss_4) < 1e-6


def test_grad_accum_accumulates_over_micro_batches(tmp_path, tokenizer, small_model):
    """Calling _micro_step N times without zero_grad sums the scaled gradients."""
    trainer = _make_trainer(tmp_path, tokenizer, small_model)
    train_iter = iter(trainer.train_loader)
    batch_a = next(train_iter)
    batch_b = next(train_iter)

    # Compute unscaled gradients for each batch individually
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer._micro_step(batch_a)
    grads_a = {
        n: p.grad.clone()
        for n, p in trainer.model.named_parameters()
        if p.grad is not None
    }

    trainer.optimizer.zero_grad(set_to_none=True)
    trainer._micro_step(batch_b)
    grads_b = {
        n: p.grad.clone()
        for n, p in trainer.model.named_parameters()
        if p.grad is not None
    }

    # Now accumulate both with grad_accum_steps=2
    trainer.config.optim.grad_accum_steps = 2
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer._micro_step(batch_a)
    trainer._micro_step(batch_b)

    # Accumulated gradient should equal mean of individual gradients
    for name, p in trainer.model.named_parameters():
        if p.grad is not None:
            expected = (grads_a[name] + grads_b[name]) / 2
            torch.testing.assert_close(p.grad, expected)


def test_end_to_end_with_grad_accumulation(tmp_path, tokenizer):
    """Training loop completes correctly with grad_accum_steps > 1."""
    torch.manual_seed(42)
    model = Decoder(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS, d_ff=D_FF, max_seq_len=MAX_SEQ_LEN, dropout=0.0,
    )
    (tmp_path / "train.txt").write_text(SAMPLE_TEXT)
    config = TrainingConfig(
        model=ModelConfig(
            vocab_size=VOCAB_SIZE, d_model=D_MODEL, num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS, d_ff=D_FF, max_seq_len=MAX_SEQ_LEN,
        ),
        data=DataConfig(
            train_path=str(tmp_path / "train.txt"),
            tokenizer_path="", batch_size=BATCH_SIZE, seq_len=SEQ_LEN,
        ),
        optim=OptimConfig(lr=1e-3, warmup_steps=5, grad_accum_steps=2),
        max_steps=10, eval_interval=5, log_interval=5, save_interval=10,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        device="cpu", seed=42,
    )
    train_loader, val_loader = create_dataloaders(config.data, tokenizer)
    trainer = Trainer(config, model, tokenizer, train_loader, val_loader)
    trainer.train()

    # global_step counts optimizer steps, not micro-steps
    assert trainer.global_step == 10

    run_dir = os.path.join(str(tmp_path / "checkpoints"), "run")
    assert os.path.exists(os.path.join(run_dir, "checkpoint_latest.pt"))


# ── Integration Test ─────────────────────────────────────────────────


def test_end_to_end_training(tmp_path, tokenizer):
    torch.manual_seed(42)
    model = Decoder(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS, d_ff=D_FF, max_seq_len=MAX_SEQ_LEN, dropout=0.0,
    )
    trainer = _make_trainer(tmp_path, tokenizer, model)
    trainer.train()

    assert trainer.global_step == 20

    # Checkpoint saved
    run_dir = os.path.join(str(tmp_path / "checkpoints"), "run")
    assert os.path.exists(os.path.join(run_dir, "checkpoint_latest.pt"))
    assert os.path.exists(os.path.join(run_dir, "training.csv"))

    # Resume and train more
    trainer.config.max_steps = 30
    trainer.config.resume_from = os.path.join(run_dir, "checkpoint_latest.pt")

    fresh = Decoder(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS, d_ff=D_FF, max_seq_len=MAX_SEQ_LEN, dropout=0.0,
    )
    trainer2 = _make_trainer(tmp_path, tokenizer, fresh)
    trainer2.config.max_steps = 30
    trainer2.config.resume_from = os.path.join(run_dir, "checkpoint_latest.pt")
    trainer2.train()

    assert trainer2.global_step == 30
