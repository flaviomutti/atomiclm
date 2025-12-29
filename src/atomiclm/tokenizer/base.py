from abc import ABC, abstractmethod
from typing import Iterable, List


class Tokenizer(ABC):
    """
    Abstract tokenizer interface.

    Implementations must be deterministic and reversible
    (decode(encode(x)) ≈ x).
    """

    @abstractmethod
    def train(self, text: str, vocab_size: int, verbose: bool = False):
        pass

    @abstractmethod
    def encode(self, text: str, allowed_special="none") -> List[int]:
        pass

    @abstractmethod
    def decode(self, ids: Iterable[int]) -> str:
        pass

    @abstractmethod
    def save(self, filename: str):
        pass

    @abstractmethod
    def load(self, model_file: str):
        pass
