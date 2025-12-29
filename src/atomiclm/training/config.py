"""
Training configuration system using dataclasses and JSON.

All hyperparameters are defined as nested dataclasses that can be loaded
from and saved to JSON files. No external dependencies beyond stdlib.
"""

import json
from dataclasses import dataclass, field, asdict, fields
from typing import Optional, Any

import torch


def _resolve_device(device_str: str) -> str:
    """Resolve 'auto' to the best available device."""
    if device_str != "auto":
        return device_str
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _from_dict(cls, data: dict):
    """Recursively construct a dataclass from a (possibly nested) dict."""
    init_kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        # If the field type is itself a dataclass, recurse
        if hasattr(f.type, "__dataclass_fields__"):
            value = _from_dict(f.type, value)
        # Handle tuple fields stored as lists in JSON
        elif f.type is tuple and isinstance(value, list):
            value = tuple(value)
        init_kwargs[f.name] = value
    return cls(**init_kwargs)


@dataclass
class ModelConfig:
    """Model architecture hyperparameters."""

    vocab_size: int = 256
    d_model: int = 128
    num_layers: int = 4
    num_heads: int = 4
    d_ff: int = 512
    max_seq_len: int = 1024
    num_kv_heads: Optional[int] = None
    dropout: float = 0.0
    qkv_bias: bool = False
    use_gated_ffn: bool = False
    rope_base: float = 10000.0


@dataclass
class DataConfig:
    """Data loading configuration."""

    train_path: str = ""
    tokenizer_path: str = ""
    val_path: Optional[str] = None
    batch_size: int = 8
    seq_len: int = 512
    num_workers: int = 0


@dataclass
class OptimConfig:
    """Optimizer and learning rate schedule."""

    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    scheduler: str = "cosine"
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1


@dataclass
class TrainingConfig:
    """Complete training configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    max_steps: int = 10000
    eval_interval: int = 500
    eval_steps: int = 20
    log_interval: int = 10
    save_interval: int = 1000
    checkpoint_dir: str = "checkpoints"
    run_name: str = "run"
    resume_from: Optional[str] = None
    device: str = "auto"
    seed: int = 42

    @classmethod
    def from_json(cls, path: str) -> "TrainingConfig":
        """Load configuration from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return _from_dict(cls, data)

    def to_json(self, path: str) -> None:
        """Save configuration to a JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    def resolve_device(self) -> str:
        """Return the resolved device string (auto -> cuda/mps/cpu)."""
        return _resolve_device(self.device)
