"""
Data loading for language model training.

Handles text tokenization and batching for next-token prediction.
Each sample is a chunk of consecutive tokens split into input and
target (shifted by one position).
"""

from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader

from ..tokenizer.tokenizer import BasicTokenizer
from .config import DataConfig


class TextDataset(Dataset):
    """
    Language modeling dataset.

    Loads a text file, tokenizes it once on init, and provides
    random chunks of seq_len+1 tokens. Each chunk is split into
    input (first seq_len tokens) and target (last seq_len tokens,
    shifted by 1).
    """

    def __init__(
        self,
        text_path: str,
        tokenizer: BasicTokenizer,
        seq_len: int,
    ):
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()

        self.token_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        self.seq_len = seq_len

        if len(self.token_ids) < seq_len + 1:
            raise ValueError(
                f"Text too short: {len(self.token_ids)} tokens, "
                f"need at least {seq_len + 1}"
            )

    def __len__(self) -> int:
        return len(self.token_ids) - self.seq_len

    def __getitem__(self, idx: int) -> dict:
        chunk = self.token_ids[idx : idx + self.seq_len + 1]
        return {
            "input_ids": chunk[:-1],
            "target_ids": chunk[1:],
        }


def create_dataloaders(
    config: DataConfig,
    tokenizer: BasicTokenizer,
) -> tuple[DataLoader, Optional[DataLoader]]:
    """
    Create training and optional validation DataLoaders.

    Returns:
        (train_loader, val_loader) — val_loader is None if config.val_path
        is not provided.
    """
    train_dataset = TextDataset(
        text_path=config.train_path,
        tokenizer=tokenizer,
        seq_len=config.seq_len,
    )

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=use_pin_memory,
        drop_last=True,
    )

    val_loader = None
    if config.val_path is not None:
        val_dataset = TextDataset(
            text_path=config.val_path,
            tokenizer=tokenizer,
            seq_len=config.seq_len,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=use_pin_memory,
        )

    return train_loader, val_loader
