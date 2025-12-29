import pytest
import torch

from atomiclm.model.block import TransformerBlock

D_MODEL = 32
NUM_HEADS = 4
D_FF = 128
BATCH = 2
SEQ_LEN = 8


@pytest.fixture
def block():
    torch.manual_seed(42)
    m = TransformerBlock(
        d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_FF, dropout=0.0,
    )
    m.eval()
    return m


def test_output_shape(block):
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    out, cache = block(x)
    assert out.shape == (BATCH, SEQ_LEN, D_MODEL)
    assert cache is None


def test_causal_masking_through_block(block):
    """Appending tokens must not change output at earlier positions."""
    x = torch.randn(1, SEQ_LEN, D_MODEL)

    t = SEQ_LEN // 2
    out_short, _ = block(x[:, :t, :])
    out_full, _ = block(x)

    torch.testing.assert_close(out_full[:, :t, :], out_short)


def test_full_output_matches_cached_output(block):
    """Full-sequence output must match token-by-token KV-cached decoding."""
    x = torch.randn(1, SEQ_LEN, D_MODEL)

    full_out, _ = block(x)

    cache = {
        "k": torch.zeros(1, NUM_HEADS, SEQ_LEN, D_MODEL // NUM_HEADS),
        "v": torch.zeros(1, NUM_HEADS, SEQ_LEN, D_MODEL // NUM_HEADS),
        "pos": 0,
    }
    cached_outputs = []
    for i in range(SEQ_LEN):
        token = x[:, i : i + 1, :]
        out, cache = block(token, kv_cache=cache)
        cached_outputs.append(out)

    cached_out = torch.cat(cached_outputs, dim=1)
    torch.testing.assert_close(full_out, cached_out)


def test_multi_token_cached_matches_full(block):
    """Chunked cached decoding must match full-sequence output."""
    x = torch.randn(1, SEQ_LEN, D_MODEL)

    full_out, _ = block(x)

    cache = {
        "k": torch.zeros(1, NUM_HEADS, SEQ_LEN, D_MODEL // NUM_HEADS),
        "v": torch.zeros(1, NUM_HEADS, SEQ_LEN, D_MODEL // NUM_HEADS),
        "pos": 0,
    }

    half = SEQ_LEN // 2
    out1, cache = block(x[:, :half, :], kv_cache=cache)
    out2, cache = block(x[:, half:, :], kv_cache=cache)

    cached_out = torch.cat([out1, out2], dim=1)
    torch.testing.assert_close(full_out, cached_out)


def test_kv_cache_pos_advances(block):
    cache = {
        "k": torch.zeros(1, NUM_HEADS, SEQ_LEN, D_MODEL // NUM_HEADS),
        "v": torch.zeros(1, NUM_HEADS, SEQ_LEN, D_MODEL // NUM_HEADS),
        "pos": 0,
    }

    x = torch.randn(1, 3, D_MODEL)
    _, cache = block(x, kv_cache=cache)
    assert cache["pos"] == 3

    for _ in range(2):
        token = torch.randn(1, 1, D_MODEL)
        _, cache = block(token, kv_cache=cache)

    assert cache["pos"] == 5


def test_cache_in_training_mode_raises(block):
    block.train()

    cache = {
        "k": torch.zeros(1, NUM_HEADS, SEQ_LEN, D_MODEL // NUM_HEADS),
        "v": torch.zeros(1, NUM_HEADS, SEQ_LEN, D_MODEL // NUM_HEADS),
        "pos": 0,
    }
    x = torch.randn(1, 1, D_MODEL)

    with pytest.raises(AssertionError):
        block(x, kv_cache=cache)


def test_batch_consistency(block):
    """Identical sequences in different batch slots produce identical output."""
    single = torch.randn(1, SEQ_LEN, D_MODEL)
    batched = single.expand(BATCH, -1, -1)

    out, _ = block(batched)

    for i in range(1, BATCH):
        torch.testing.assert_close(out[0], out[i])


def test_gated_ffn_flag():
    """Constructing with use_gated_ffn=True uses the gated variant."""
    block = TransformerBlock(
        d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_FF, use_gated_ffn=True,
    )
    assert block.ffn.use_gated is True


def test_single_token_input(block):
    x = torch.randn(1, 1, D_MODEL)
    out, cache = block(x)
    assert out.shape == (1, 1, D_MODEL)
    assert cache is None
