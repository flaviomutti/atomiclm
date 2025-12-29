"""
Training loop for decoder-only language models.

Provides the Trainer class that orchestrates forward/backward passes,
optimizer steps, learning rate scheduling, validation, logging, and
checkpointing. Also includes standalone LR schedule functions.
"""

import math
import os
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..model.decoder import Decoder
from ..tokenizer.tokenizer import BasicTokenizer
from .config import TrainingConfig
from .checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint


class Trainer:
    """
    Training orchestrator for a decoder-only language model.

    Handles:
    - Training loop (forward, loss, backward, step)
    - Validation loop
    - Checkpointing (save/resume)
    - Logging (console + CSV)
    - Learning rate scheduling
    """

    def __init__(
        self,
        config: TrainingConfig,
        model: Decoder,
        tokenizer: BasicTokenizer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Validate seq_len fits within model's max_seq_len
        if config.data.seq_len > config.model.max_seq_len:
            raise ValueError(
                f"data.seq_len ({config.data.seq_len}) exceeds "
                f"model.max_seq_len ({config.model.max_seq_len})"
            )

        # Device
        self.device = torch.device(config.resolve_device())
        self.model = model.to(self.device)

        # Optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Training state
        self.global_step = 0
        self.best_val_loss = float("inf")

        # Logging
        self.run_dir = os.path.join(config.checkpoint_dir, config.run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.log_path = os.path.join(self.run_dir, "training.csv")

    def _create_optimizer(self) -> torch.optim.AdamW:
        """Create AdamW with parameter grouping (no decay on biases/norms/embeddings)."""
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() < 2 or "norm" in name or "tok_emb" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        groups = [
            {"params": decay_params, "weight_decay": self.config.optim.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        return torch.optim.AdamW(
            groups,
            lr=self.config.optim.lr,
            betas=self.config.optim.betas,
            eps=1e-8,
        )

    def _create_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        """Create LR scheduler based on config."""
        if self.config.optim.scheduler == "cosine":
            return get_cosine_schedule_with_warmup(
                self.optimizer,
                warmup_steps=self.config.optim.warmup_steps,
                max_steps=self.config.max_steps,
                min_lr_ratio=self.config.optim.min_lr_ratio,
            )
        elif self.config.optim.scheduler == "constant":
            return get_constant_schedule_with_warmup(
                self.optimizer,
                warmup_steps=self.config.optim.warmup_steps,
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.config.optim.scheduler}")

    def train_step(self, batch: dict) -> float:
        """
        Single training step: forward, loss, backward, clip, step.

        Returns:
            Scalar loss value.
        """
        self.model.train()

        input_ids = batch["input_ids"].to(self.device)
        target_ids = batch["target_ids"].to(self.device)

        logits, _ = self.model(input_ids)

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if self.config.optim.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.optim.grad_clip,
            )

        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    @torch.no_grad()
    def validate(self) -> float:
        """
        Run validation loop.

        Returns:
            Average loss over eval_steps batches (or all batches if fewer).
        """
        if self.val_loader is None:
            return float("nan")

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for i, batch in enumerate(self.val_loader):
            if i >= self.config.eval_steps:
                break

            input_ids = batch["input_ids"].to(self.device)
            target_ids = batch["target_ids"].to(self.device)

            logits, _ = self.model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
            )

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else float("nan")

    def train(self) -> None:
        """Main training loop."""
        num_params = sum(p.numel() for p in self.model.parameters())
        total_tokens = self.config.max_steps * self.config.data.batch_size * self.config.data.seq_len
        print(f"Run: {self.config.run_name}")
        print(f"Device: {self.device}")
        print(f"Parameters: {num_params:,}")
        print(f"Max steps: {self.config.max_steps:,}  ({total_tokens:,} tokens)")
        print("-" * 60)

        # Resume if configured
        if self.config.resume_from is not None:
            state = load_checkpoint(
                self.config.resume_from,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                device=str(self.device),
            )
            self.global_step = state["global_step"]
            print(f"Resumed from step {self.global_step}")
        else:
            # Auto-resume from latest checkpoint in run dir
            latest = find_latest_checkpoint(self.run_dir)
            if latest is not None:
                state = load_checkpoint(
                    latest,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    device=str(self.device),
                )
                self.global_step = state["global_step"]
                print(f"Auto-resumed from step {self.global_step}")

        # Initialize CSV log
        self._init_log()

        # Infinite iterator over training data
        train_iter = iter(self.train_loader)
        step_start = time.time()
        loss_accum = 0.0
        loss_count = 0

        while self.global_step < self.config.max_steps:
            # Get next batch (wrap around if exhausted)
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            loss = self.train_step(batch)
            self.global_step += 1
            loss_accum += loss
            loss_count += 1

            # Logging
            if self.global_step % self.config.log_interval == 0:
                elapsed = time.time() - step_start
                steps_per_sec = self.config.log_interval / elapsed
                tokens_per_sec = steps_per_sec * self.config.data.batch_size * self.config.data.seq_len
                remaining_steps = self.config.max_steps - self.global_step
                eta_sec = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                lr = self.scheduler.get_last_lr()[0]
                avg_loss = loss_accum / loss_count
                self._log(
                    step=self.global_step,
                    train_loss=avg_loss,
                    lr=lr,
                    tokens_per_sec=tokens_per_sec,
                    eta_sec=eta_sec,
                )
                loss_accum = 0.0
                loss_count = 0
                step_start = time.time()

            # Validation
            if self.global_step % self.config.eval_interval == 0:
                val_loss = self.validate()
                self._log(step=self.global_step, val_loss=val_loss)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save(is_best=True)

            # Periodic checkpoint
            if self.global_step % self.config.save_interval == 0:
                self._save()

        # Final checkpoint
        self._save()
        print(f"\nTraining complete at step {self.global_step}.")
        if self.best_val_loss < float("inf"):
            print(f"Best validation loss: {self.best_val_loss:.4f}")

    def _save(self, is_best: bool = False) -> None:
        """Save checkpoint."""
        save_checkpoint(
            checkpoint_dir=self.run_dir,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            global_step=self.global_step,
            config=self.config,
            is_best=is_best,
        )

    def _init_log(self) -> None:
        """Initialize CSV log file if it doesn't exist."""
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                f.write("step,train_loss,val_loss,lr,steps_per_sec\n")

    def _log(
        self,
        step: int,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        lr: Optional[float] = None,
        tokens_per_sec: Optional[float] = None,
        eta_sec: Optional[float] = None,
    ) -> None:
        """Log metrics to console and CSV."""
        parts = [f"step {step:>6d}"]
        if train_loss is not None:
            parts.append(f"train_loss={train_loss:.4f}")
        if val_loss is not None:
            parts.append(f"val_loss={val_loss:.4f}")
        if lr is not None:
            parts.append(f"lr={lr:.2e}")
        if tokens_per_sec is not None:
            parts.append(f"{tokens_per_sec:,.0f} tok/s")
        if eta_sec is not None:
            parts.append(f"ETA {_format_eta(eta_sec)}")
        print(" | ".join(parts))

        # Append to CSV
        with open(self.log_path, "a") as f:
            f.write(
                f"{step},"
                f"{train_loss if train_loss is not None else ''},"
                f"{val_loss if val_loss is not None else ''},"
                f"{lr if lr is not None else ''},"
                f"{tokens_per_sec if tokens_per_sec is not None else ''}\n"
            )


# ── Helpers ──────────────────────────────────────────────────────────


def _format_eta(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


# ── Learning Rate Schedules ──────────────────────────────────────────


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Cosine learning rate schedule with linear warmup.

    LR increases linearly from 0 to max_lr during warmup,
    then decreases following a cosine curve to min_lr = max_lr * min_lr_ratio.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, max_steps - warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_constant_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Constant LR with linear warmup."""

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
