import pytest

from atomiclm.tokenizer import BasicTokenizer


def test_base_vocab_is_bytes_identity():

    tok = BasicTokenizer()

    vocab = tok.vocab

    assert len(vocab) >= 256

    for i in range(256):
        assert i in vocab
        assert vocab[i] == bytes([i])


def test_merged_tokens_are_concatenation_of_parents():

    tok = BasicTokenizer()

    # Fake a simple merge: 'a' + 'b' -> token 256
    tok.merges = {(ord("a"), ord("b")): 256}
    tok.vocab = tok._build_vocab()

    assert tok.vocab[256] == b"ab"


def test_vocab_reconstruction_is_deterministic():

    tok = BasicTokenizer()

    tok.merges = {
        (1, 2): 256,
        (256, 3): 257,
    }

    vocab1 = tok._build_vocab()
    vocab2 = tok._build_vocab()

    assert vocab1 == vocab2


def test_encode_decode_roundtrip():

    tok = BasicTokenizer()
    text = "hello world"

    ids = tok.encode(text)
    decoded = tok.decode(ids)

    assert decoded == text


def test_export_mergeable_ranks_has_unique_bytes():

    tok = BasicTokenizer()
    mergeable_ranks = tok.export_mergeable_ranks()

    seen = set()
    for token_bytes in mergeable_ranks.keys():
        assert token_bytes not in seen
        seen.add(token_bytes)


def test_encode_empty_string():

    tok = BasicTokenizer()

    ids = tok.encode("")
    assert ids == []


def test_decode_empty_ids():

    tok = BasicTokenizer()

    text = tok.decode([])
    assert text == ""


def test_single_character_roundtrip():

    tok = BasicTokenizer()

    for ch in ["a", " ", "\n", "\t"]:
        ids = tok.encode(ch)
        assert tok.decode(ids) == ch


def test_utf8_roundtrip():

    tok = BasicTokenizer()

    text = "caffè ☕ naïve 🚀"
    ids = tok.encode(text)
    decoded = tok.decode(ids)

    assert decoded == text


def test_overlapping_merges_reconstruct_correctly():

    tok = BasicTokenizer()

    # a b -> ab
    # ab c -> abc
    tok.merges = {
        (ord("a"), ord("b")): 256,
        (256, ord("c")): 257,
    }

    tok.vocab = tok._build_vocab()

    assert tok.vocab[256] == b"ab"
    assert tok.vocab[257] == b"abc"


def test_merge_id_collision_overwrites_nothing():

    tok = BasicTokenizer()

    tok.merges = {
        (1, 2): 256,
        (3, 4): 256,  # illegal: same token id
    }

    with pytest.raises(ValueError):
        tok._build_vocab()


def test_export_mergeable_ranks_is_deterministic():

    tok = BasicTokenizer()

    r1 = tok.export_mergeable_ranks()
    r2 = tok.export_mergeable_ranks()

    assert r1 == r2


def test_no_empty_tokens_in_vocab():

    tok = BasicTokenizer()

    for b in tok.vocab.values():
        assert len(b) > 0


def test_training_stops_with_small_corpus():

    tok = BasicTokenizer()

    tok.train("aaaa", vocab_size=10_000)

    # vocab size is bounded by entropy
    assert len(tok.vocab) < 10_000


def test_encode_applies_multi_level_hierarchy():

    tok = BasicTokenizer()

    tok.merges = {
        (ord("a"), ord("b")): 256,  # ab
        (256, ord("c")): 257,  # abc
        (257, ord("d")): 258,  # abcd
    }

    tok.merge_ranks = {
        (ord("a"), ord("b")): 0,  # ab
        (256, ord("c")): 1,  # abc
        (257, ord("d")): 2,  # abcd
    }

    tok.vocab = tok._build_vocab()

    ids = tok.encode("abcd")

    assert ids == [258]


def test_encode_hierarchical_overlapping_merges():

    tok = BasicTokenizer()

    tok.merges = {
        (ord("a"), ord("b")): 256,
        (ord("b"), ord("c")): 257,
        (256, ord("c")): 258,
    }

    tok.merge_ranks = {
        (ord("a"), ord("b")): 0,  # ab
        (ord("b"), ord("c")): 1,  # bc
        (256, ord("c")): 2,  # abc
    }

    tok.vocab = tok._build_vocab()

    ids = tok.encode("abc")

    assert ids == [258]


def test_iter_sequences_text_vs_iterator_equivalent():

    tok = BasicTokenizer()

    text = "hello world\nhello"
    iterator = ["hello world\n", "hello"]

    seq_from_text = list(tok._iter_sequences_from_text(text))
    seq_from_iter = list(tok._iter_sequences_from_iterator(iterator))

    assert seq_from_text == seq_from_iter


def test_iter_sequences_iterator_chunk_boundaries_do_not_matter():

    tok = BasicTokenizer()

    text = "abc def ghi"

    iterator = ["a", "bc d", "ef g", "hi"]

    seq_from_text = list(tok._iter_sequences_from_text(text))
    seq_from_iter = list(tok._iter_sequences_from_iterator(iterator))

    assert seq_from_text == seq_from_iter


def test_iter_sequences_iterator_ignores_empty_strings():

    tok = BasicTokenizer()

    iterator = ["", "   ", "\n", "hi"]

    seqs = list(tok._iter_sequences_from_iterator(iterator))

    # Should only produce sequences from actual text chunks
    assert all(len(seq) > 0 for seq in seqs)


def test_heap_merges_adjacent_repeated_pairs():
    """Heap trainer should merge all occurrences of a pair, even when adjacent."""
    tok_heap = BasicTokenizer()
    tok_basic = BasicTokenizer()

    # "ababab" has pair (a,b) at three adjacent positions
    text = "ababab " * 50
    tok_heap.train(text, vocab_size=270, method=None)
    tok_basic.train(text, vocab_size=270, method="basic")

    assert tok_heap.merges == tok_basic.merges


def test_train_text_vs_iterator_produce_same_merges():

    tok1 = BasicTokenizer()
    tok2 = BasicTokenizer()

    text = "banana bandana banana"
    iterator = ["banana ", "bandana ", "banana"]

    tok1.train(text, vocab_size=300)
    tok2.train_from_iterator(iterator, vocab_size=300)

    assert tok1.merges == tok2.merges


def test_encode_after_training_text_vs_iterator_equivalent():

    tok1 = BasicTokenizer()
    tok2 = BasicTokenizer()

    corpus = "mississippi river"
    iterator = ["missi", "ssippi ", "river"]

    tok1.train(corpus, vocab_size=300)
    tok2.train_from_iterator(iterator, vocab_size=300)

    test_text = "mississippi"
    assert tok1.encode(test_text) == tok2.encode(test_text)


def test_train_from_iterator_is_deterministic():

    tok1 = BasicTokenizer()
    tok2 = BasicTokenizer()

    iterator = ["aab", "aac", "aab"]

    tok1.train_from_iterator(iterator, vocab_size=300)
    tok2.train_from_iterator(iterator, vocab_size=300)

    assert tok1.merges == tok2.merges


def test_iterator_and_text_do_not_swap_merge_ids():

    tok1 = BasicTokenizer()
    tok2 = BasicTokenizer()

    text = " iou"
    iterator = [" ", "iou"]

    tok1.train(text, vocab_size=300)
    tok2.train_from_iterator(iterator, vocab_size=300)

    assert tok1.merges == tok2.merges
