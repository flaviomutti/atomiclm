from collections import defaultdict
import heapq
import json
import time
import regex as re

from .base import Tokenizer
from .constants import TOKENIZER_BASE_LENGTH, TOKENIZER_VERSION
from .patterns import GPT4_SPLIT_PATTERN
from .bpe_ops import merge, get_stats
from ..utils import format_eta


class BasicTokenizer(Tokenizer):

    def __init__(self, pattern=None):

        self.special_tokens = {}
        self.inverse_special_tokens = {}

        self.merges = {}
        self.merge_ranks = {}
        self._bytes_to_id = {}

        self.vocab = self._build_vocab()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)

    def _build_vocab(self):

        vocab = {idx: bytes([idx]) for idx in range(TOKENIZER_BASE_LENGTH)}

        used_ids = set(vocab.keys())

        for pair, idx in self.merges.items():
            if idx in used_ids:
                raise ValueError(f"Token id collision: id {idx} already assigned")

            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            used_ids.add(idx)

        self._bytes_to_id = {v: k for k, v in vocab.items()}

        return vocab

    def train(self, text: str, vocab_size: int, verbose=False, method=None):

        sequences = list(self._iter_sequences_from_text(text))
        return self._train_from_sequences(sequences, vocab_size, verbose, method)

    def train_from_iterator(
        self, iterator, vocab_size: int, verbose=False, method=None
    ):

        sequences = list(self._iter_sequences_from_iterator(iterator))
        return self._train_from_sequences(sequences, vocab_size, verbose, method)

    def _train_from_sequences(self, sequences, vocab_size, verbose, method):

        train_fn = self._train_heap
        if method == "basic":
            train_fn = self._train_basic

        return train_fn(sequences, vocab_size, verbose)

    def _iter_sequences_from_text(self, text: str):

        for chunk in re.findall(self.compiled_pattern, text):
            yield list(chunk.encode("utf-8"))

    def _iter_sequences_from_iterator(self, iterator):

        carry = ""

        for text in iterator:
            text = carry + text
            chunks = re.findall(self.compiled_pattern, text)

            for chunk in chunks[:-1]:
                yield list(chunk.encode("utf-8"))

            carry = chunks[-1] if chunks else ""

        if carry:
            yield list(carry.encode("utf-8"))

    def _train_heap(self, sequences, vocab_size, verbose=False):
        """
        Train a BPE tokenizer using a heap for efficient merges.
        """
        assert vocab_size > TOKENIZER_BASE_LENGTH
        num_merges = vocab_size - TOKENIZER_BASE_LENGTH

        self.merges = {}
        self.merge_ranks = {}

        token_bytes = {i: bytes([i]) for i in range(TOKENIZER_BASE_LENGTH)}
        bytes_to_id = {b: i for i, b in token_bytes.items()}

        pair_freq = defaultdict(int)
        pair_positions = defaultdict(set)

        for seq_idx, seq in enumerate(sequences):
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                pair_freq[pair] += 1
                pair_positions[pair].add((seq_idx, i))

        heap = []
        tie = 0
        for pair, freq in pair_freq.items():
            heap.append((-freq, tie, pair))
            tie += 1
        heapq.heapify(heap)

        merges_done = 0
        t_start = time.time()
        print_every = max(1, num_merges // 20)
        w = len(str(num_merges))
        while merges_done < num_merges and heap:
            while heap:
                neg_freq, _, pair = heapq.heappop(heap)

                freq = pair_freq.get(pair, 0)
                if freq == -neg_freq and freq > 0 and pair not in self.merges:
                    break
            else:
                break

            left, right = pair
            new_bytes = token_bytes[left] + token_bytes[right]

            if new_bytes in bytes_to_id:
                pair_freq[pair] = 0
                continue

            new_id = TOKENIZER_BASE_LENGTH + merges_done
            self.merges[pair] = new_id
            self.merge_ranks[pair] = merges_done
            merges_done += 1

            token_bytes[new_id] = new_bytes
            bytes_to_id[new_bytes] = new_id

            if verbose and (merges_done % print_every == 0 or merges_done == num_merges):
                elapsed = time.time() - t_start
                rate = merges_done / elapsed if elapsed > 0 else 0
                remaining = (num_merges - merges_done) / rate if rate > 0 else 0
                pct = 100.0 * merges_done / num_merges
                end = "\n" if merges_done == num_merges else ""
                print(
                    f"\r  [{merges_done:{w}d}/{num_merges}] {pct:5.1f}%"
                    f" | {rate:6.0f} merges/s | ETA {format_eta(remaining):<8s}",
                    end=end, flush=True,
                )

            # Sort by position descending so merging at later positions
            # doesn't shift earlier ones within the same sequence.
            occurrences = sorted(pair_positions[pair], key=lambda x: -x[1])
            pair_positions[pair].clear()
            pair_freq[pair] = 0

            for seq_idx, pos in occurrences:
                seq = sequences[seq_idx]

                if pos >= len(seq) - 1:
                    continue
                if seq[pos] != pair[0] or seq[pos + 1] != pair[1]:
                    continue

                if pos > 0:
                    left = (seq[pos - 1], seq[pos])
                    pair_freq[left] -= 1
                    pair_positions[left].discard((seq_idx, pos - 1))
                    heapq.heappush(heap, (-pair_freq[left], tie, left))
                    tie += 1

                if pos + 2 < len(seq):
                    right = (seq[pos + 1], seq[pos + 2])
                    pair_freq[right] -= 1
                    pair_positions[right].discard((seq_idx, pos + 1))
                    heapq.heappush(heap, (-pair_freq[right], tie, right))
                    tie += 1

                seq[pos : pos + 2] = [new_id]

                if pos > 0:
                    new_left = (seq[pos - 1], new_id)
                    pair_freq[new_left] += 1
                    pair_positions[new_left].add((seq_idx, pos))
                    heapq.heappush(heap, (-pair_freq[new_left], tie, new_left))
                    tie += 1

                if pos + 1 < len(seq):
                    new_right = (new_id, seq[pos + 1])
                    pair_freq[new_right] += 1
                    pair_positions[new_right].add((seq_idx, pos))
                    heapq.heappush(heap, (-pair_freq[new_right], tie, new_right))
                    tie += 1

        self.vocab = self._build_vocab()

    def _train_basic(self, sequences, vocab_size, verbose=False):

        assert vocab_size > TOKENIZER_BASE_LENGTH

        ids = [seq[:] for seq in sequences]
        num_merges = vocab_size - TOKENIZER_BASE_LENGTH

        self.merges = {}
        self.merge_ranks = {}

        t_start = time.time()
        print_every = max(1, num_merges // 20)
        w = len(str(num_merges))
        for i in range(num_merges):
            stats = {}
            for seq in ids:
                stats = get_stats(seq, stats)

            if not stats:
                break

            pair = max(stats, key=stats.get)
            idx = TOKENIZER_BASE_LENGTH + i

            ids = [merge(seq, pair, idx) for seq in ids]
            self.merges[pair] = idx
            self.merge_ranks[pair] = i

            done = i + 1
            if verbose and (done % print_every == 0 or done == num_merges):
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                remaining = (num_merges - done) / rate if rate > 0 else 0
                pct = 100.0 * done / num_merges
                end = "\n" if done == num_merges else ""
                print(
                    f"\r  [{done:{w}d}/{num_merges}] {pct:5.1f}%"
                    f" | {rate:6.0f} merges/s | ETA {format_eta(remaining):<8s}",
                    end=end, flush=True,
                )

        self.vocab = self._build_vocab()

    def register_special_tokens(self, special_tokens):

        self.special_tokens = special_tokens
        self.inverse_special_tokens = {
            idx: tok for tok, idx in self.special_tokens.items()
        }

    def encode(self, text, allowed_special="none"):

        special_tokens_subset = self._get_allowed_special_subset(text, allowed_special)

        missing = set(self.merges) - set(self.merge_ranks)
        if missing:
            raise RuntimeError(f"Missing merge_ranks for pairs: {missing}")

        if not special_tokens_subset:
            return self._encode_basic(text)

        special_pattern = (
            "(" + "|".join(re.escape(k) for k in special_tokens_subset) + ")"
        )
        special_chunks = re.split(special_pattern, text)

        ids = []
        for chunk in special_chunks:
            if chunk in special_tokens_subset:
                ids.append(special_tokens_subset[chunk])
                continue

            ids.extend(self._encode_basic(chunk))

        return ids

    def decode(self, ids):

        byte_stream = []

        for idx in ids:
            if idx in self.vocab:
                byte_stream.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                byte_stream.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"Invalid token id: {idx}")

        tokens = b"".join(byte_stream)

        return tokens.decode("utf-8", errors="replace")

    def save(self, filename):

        data = {
            "version": TOKENIZER_VERSION,
            "type": "bpe",
            "base_vocab_size": TOKENIZER_BASE_LENGTH,
            "pattern": self.pattern,
            "merges": [[pair[0], pair[1], idx] for pair, idx in self.merges.items()],
            "merge_ranks": [
                [pair[0], pair[1], rank] for pair, rank in self.merge_ranks.items()
            ],
            "special_tokens": self.special_tokens,
        }
        with open(filename + ".json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, filename):

        assert filename.endswith(".json")
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["version"] == TOKENIZER_VERSION
        assert data["type"] == "bpe"

        self.pattern = data["pattern"]
        self.special_tokens = data.get("special_tokens", {})
        self.inverse_special_tokens = {
            idx: tok for tok, idx in self.special_tokens.items()
        }

        self.merges = {(pair[0], pair[1]): pair[2] for pair in data["merges"]}
        self.merge_ranks = {
            (pair[0], pair[1]): rank
            for pair, rank in [([p[0], p[1]], p[2]) for p in data["merge_ranks"]]
        }

        self.vocab = self._build_vocab()

    def _encode_basic(self, text):

        text_chunks = re.findall(self.compiled_pattern, text)

        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunks(chunk_bytes)
            ids.extend(chunk_ids)

        return ids

    def _encode_chunks(self, text_bytes):

        ids = list(text_bytes)

        while len(ids) >= 2:
            # Find the adjacent pair whose merged bytes have the lowest token ID.
            # Token IDs are assigned in merge order (256, 257, ...) so they
            # double as priority ranks — lower ID means higher priority.
            best_id = None
            for i in range(len(ids) - 1):
                merged_bytes = self.vocab[ids[i]] + self.vocab[ids[i + 1]]
                merged_id = self._bytes_to_id.get(merged_bytes)
                if merged_id is not None:
                    if best_id is None or merged_id < best_id:
                        best_id = merged_id

            if best_id is None:
                break

            # Merge all adjacent pairs that produce best_id (may come from
            # different token-ID pairs that share the same byte sequence).
            target_bytes = self.vocab[best_id]
            new_ids = []
            i = 0
            while i < len(ids):
                if (
                    i < len(ids) - 1
                    and self.vocab[ids[i]] + self.vocab[ids[i + 1]] == target_bytes
                ):
                    new_ids.append(best_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids

        return ids

    def _get_allowed_special_subset(self, text, allowed_special):

        if allowed_special == "all":
            return self.special_tokens
        if allowed_special == "none":
            return {}
        if allowed_special == "none_raise":
            # validate no special token is in the text
            if any(token in text for token in self.special_tokens):
                raise ValueError(
                    "Text contains special tokens, but allowed_special='none_raise'"
                )
            return {}
        if isinstance(allowed_special, set):
            return {
                tok: idx
                for tok, idx in self.special_tokens.items()
                if tok in allowed_special
            }

        raise ValueError(f"allowed_special={allowed_special} not understood")

    def export_mergeable_ranks(self):

        mergeable_ranks = {}

        # Build vocabulary incrementally from low to high token IDs
        token_bytes = [bytes([i]) for i in range(TOKENIZER_BASE_LENGTH)]

        # Base tokens
        for i, b in enumerate(token_bytes):
            mergeable_ranks[b] = i

        sorted_merges = sorted(self.merges.items(), key=lambda item: item[1])

        for (left, right), merged_id in sorted_merges:
            merged_bytes = token_bytes[left] + token_bytes[right]

            if len(token_bytes) <= merged_id:
                token_bytes.extend([b""] * (merged_id + 1 - len(token_bytes)))

            token_bytes[merged_id] = merged_bytes
            mergeable_ranks[merged_bytes] = merged_id

        return mergeable_ranks
