import pytest
import torch

from atomiclm.model.attention import MultiHeadAttention

# Shared dimensions for tests
D_IN = 32
D_OUT = 32
NUM_HEADS = 4
BATCH = 2
SEQ_LEN = 8


@pytest.fixture
def mha():
    torch.manual_seed(0)
    m = MultiHeadAttention(d_in=D_IN, d_out=D_OUT, num_heads=NUM_HEADS, dropout=0.0)
    m.eval()
    return m


def test_output_shape(mha):

    x = torch.randn(BATCH, SEQ_LEN, D_IN)

    out, cache = mha(x)

    assert out.shape == (BATCH, SEQ_LEN, D_OUT)
    assert cache is None


def test_causal_masking(mha):
    """Appending tokens must not change output at earlier positions."""

    x = torch.randn(1, SEQ_LEN, D_IN)

    t = SEQ_LEN // 2
    out_short, _ = mha(x[:, :t, :])
    out_full, _ = mha(x)

    torch.testing.assert_close(out_full[:, :t, :], out_short)


def test_full_output_matches_cached_output(mha):
    """Full-sequence output must match token-by-token KV-cached decoding."""

    x = torch.randn(1, SEQ_LEN, D_IN)

    # Full sequence in one shot
    full_out, _ = mha(x)

    # Token-by-token with KV cache
    cache = {
        "k": torch.zeros(1, mha.num_heads, SEQ_LEN, mha.head_dim),
        "v": torch.zeros(1, mha.num_heads, SEQ_LEN, mha.head_dim),
        "pos": 0,
    }
    cached_outputs = []
    for i in range(SEQ_LEN):
        token = x[:, i : i + 1, :]
        out, cache = mha(token, kv_cache=cache)
        cached_outputs.append(out)

    cached_out = torch.cat(cached_outputs, dim=1)

    torch.testing.assert_close(full_out, cached_out)


def test_batch_consistency(mha):
    """Identical sequences in different batch slots produce identical output."""

    single = torch.randn(1, SEQ_LEN, D_IN)
    batched = single.expand(BATCH, -1, -1)

    out, _ = mha(batched)

    for i in range(1, BATCH):
        torch.testing.assert_close(out[0], out[i])


def test_single_token_input(mha):

    x = torch.randn(1, 1, D_IN)

    out, cache = mha(x)

    assert out.shape == (1, 1, D_OUT)
    assert cache is None


def test_kv_cache_pos_advances(mha):

    cache = {
        "k": torch.zeros(1, mha.num_heads, SEQ_LEN, mha.head_dim),
        "v": torch.zeros(1, mha.num_heads, SEQ_LEN, mha.head_dim),
        "pos": 0,
    }

    # Feed 3 tokens at once
    x = torch.randn(1, 3, D_IN)
    _, cache = mha(x, kv_cache=cache)
    assert cache["pos"] == 3

    # Feed 2 more tokens one by one
    for _ in range(2):
        token = torch.randn(1, 1, D_IN)
        _, cache = mha(token, kv_cache=cache)

    assert cache["pos"] == 5


def test_cache_influences_output(mha):
    """Output for a token must differ depending on cached context."""

    x = torch.randn(1, 3, D_IN)

    # Last token without any prior context
    out_no_ctx, _ = mha(x[:, 2:3, :])

    # Same token but with two prior tokens in cache
    cache = {
        "k": torch.zeros(1, mha.num_heads, 3, mha.head_dim),
        "v": torch.zeros(1, mha.num_heads, 3, mha.head_dim),
        "pos": 0,
    }
    mha(x[:, :2, :], kv_cache=cache)
    out_with_ctx, _ = mha(x[:, 2:3, :], kv_cache=cache)

    assert not torch.allclose(out_no_ctx, out_with_ctx)


def test_cache_overflow_raises(mha):
    """Writing past cache capacity must raise."""

    cache = {
        "k": torch.zeros(1, mha.num_heads, 2, mha.head_dim),
        "v": torch.zeros(1, mha.num_heads, 2, mha.head_dim),
        "pos": 0,
    }

    x = torch.randn(1, 3, D_IN)

    with pytest.raises(AssertionError):
        mha(x, kv_cache=cache)


def test_cache_in_training_mode_raises(mha):
    """KV cache is inference-only; using it in training mode must raise."""

    mha.train()

    cache = {
        "k": torch.zeros(1, mha.num_heads, SEQ_LEN, mha.head_dim),
        "v": torch.zeros(1, mha.num_heads, SEQ_LEN, mha.head_dim),
        "pos": 0,
    }

    x = torch.randn(1, 1, D_IN)

    with pytest.raises(AssertionError):
        mha(x, kv_cache=cache)


def test_custom_mask_with_causal_raises():
    """Passing attn_mask when causal=True must raise."""

    torch.manual_seed(0)
    mha = MultiHeadAttention(
        d_in=D_IN, d_out=D_OUT, num_heads=NUM_HEADS, dropout=0.0, causal=True,
    )
    mha.eval()

    x = torch.randn(1, SEQ_LEN, D_IN)
    mask = torch.zeros(SEQ_LEN, SEQ_LEN)

    with pytest.raises(AssertionError):
        mha(x, attn_mask=mask)


def test_float16_no_nan():
    """Large-magnitude float16 inputs must not produce NaN or Inf."""

    torch.manual_seed(0)
    mha = MultiHeadAttention(
        d_in=D_IN, d_out=D_OUT, num_heads=NUM_HEADS, dropout=0.0,
    ).half().eval()

    # float16 max is ~65504; scale near the boundary
    x = torch.randn(1, SEQ_LEN, D_IN, dtype=torch.float16) * 100.0

    out, _ = mha(x)

    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()


def test_multi_token_cached_matches_full(mha):
    """Feeding chunks of >1 token through cache must match full-sequence output."""

    x = torch.randn(1, SEQ_LEN, D_IN)

    # Full sequence in one shot
    full_out, _ = mha(x)

    # Two chunks: first half, then second half via cache
    cache = {
        "k": torch.zeros(1, mha.num_heads, SEQ_LEN, mha.head_dim),
        "v": torch.zeros(1, mha.num_heads, SEQ_LEN, mha.head_dim),
        "pos": 0,
    }

    half = SEQ_LEN // 2
    out1, cache = mha(x[:, :half, :], kv_cache=cache)
    out2, cache = mha(x[:, half:, :], kv_cache=cache)

    cached_out = torch.cat([out1, out2], dim=1)

    torch.testing.assert_close(full_out, cached_out)


# ============================================================
# GQA-specific tests
# ============================================================


def test_gqa_output_shape():
    """GQA output shape must be (b, t, d_model) regardless of num_kv_heads."""
    torch.manual_seed(0)
    num_kv_heads = 2
    mha_gqa = MultiHeadAttention(
        d_in=D_IN, d_out=D_OUT, num_heads=NUM_HEADS, num_kv_heads=num_kv_heads, dropout=0.0,
    )
    mha_gqa.eval()

    x = torch.randn(BATCH, SEQ_LEN, D_IN)
    out, cache = mha_gqa(x)

    assert out.shape == (BATCH, SEQ_LEN, D_OUT)
    assert cache is None


def test_gqa_kv_head_count():
    """GQA module should correctly store num_kv_heads and compute num_groups."""
    num_kv_heads = 2
    mha_gqa = MultiHeadAttention(
        d_in=D_IN, d_out=D_OUT, num_heads=NUM_HEADS, num_kv_heads=num_kv_heads,
    )

    assert mha_gqa.num_kv_heads == num_kv_heads
    assert mha_gqa.num_groups == NUM_HEADS // num_kv_heads


def test_gqa_cache_uses_num_kv_heads():
    """KV-cache for GQA must use num_kv_heads, not num_heads."""
    torch.manual_seed(0)
    num_kv_heads = 2
    mha_gqa = MultiHeadAttention(
        d_in=D_IN, d_out=D_OUT, num_heads=NUM_HEADS, num_kv_heads=num_kv_heads, dropout=0.0,
    )
    mha_gqa.eval()

    # Cache should have num_kv_heads in dim 1
    cache = {
        "k": torch.zeros(1, num_kv_heads, SEQ_LEN, mha_gqa.head_dim),
        "v": torch.zeros(1, num_kv_heads, SEQ_LEN, mha_gqa.head_dim),
        "pos": 0,
    }

    x = torch.randn(1, 1, D_IN)
    out, updated_cache = mha_gqa(x, kv_cache=cache)

    assert out.shape == (1, 1, D_OUT)
    assert updated_cache["k"].shape[1] == num_kv_heads
    assert updated_cache["v"].shape[1] == num_kv_heads


def test_gqa_cache_memory_reduction():
    """GQA cache should use less memory than MHA cache."""
    num_kv_heads = 2

    mha_standard = MultiHeadAttention(
        d_in=D_IN, d_out=D_OUT, num_heads=NUM_HEADS,
    )

    mha_gqa = MultiHeadAttention(
        d_in=D_IN, d_out=D_OUT, num_heads=NUM_HEADS, num_kv_heads=num_kv_heads,
    )

    # Standard MHA cache
    mha_cache_size = BATCH * mha_standard.num_heads * SEQ_LEN * mha_standard.head_dim

    # GQA cache
    gqa_cache_size = BATCH * mha_gqa.num_kv_heads * SEQ_LEN * mha_gqa.head_dim

    expected_reduction = NUM_HEADS // num_kv_heads
    assert mha_cache_size == gqa_cache_size * expected_reduction


def test_gqa_cached_generation_correctness():
    """GQA with cache must produce correct outputs (matches full forward)."""
    torch.manual_seed(0)
    num_kv_heads = 2
    mha_gqa = MultiHeadAttention(
        d_in=D_IN, d_out=D_OUT, num_heads=NUM_HEADS, num_kv_heads=num_kv_heads, dropout=0.0,
    )
    mha_gqa.eval()

    x = torch.randn(1, SEQ_LEN, D_IN)

    # Full sequence in one shot
    full_out, _ = mha_gqa(x)

    # Token-by-token with KV cache
    cache = {
        "k": torch.zeros(1, num_kv_heads, SEQ_LEN, mha_gqa.head_dim),
        "v": torch.zeros(1, num_kv_heads, SEQ_LEN, mha_gqa.head_dim),
        "pos": 0,
    }
    cached_outputs = []
    for i in range(SEQ_LEN):
        token = x[:, i : i + 1, :]
        out, cache = mha_gqa(token, kv_cache=cache)
        cached_outputs.append(out)

    cached_out = torch.cat(cached_outputs, dim=1)

    torch.testing.assert_close(full_out, cached_out)


def test_gqa_vs_mha_equivalence_when_equal():
    """When num_kv_heads == num_heads, GQA should behave like MHA."""
    torch.manual_seed(42)

    # GQA with num_kv_heads == num_heads (equivalent to MHA)
    mha_gqa = MultiHeadAttention(
        d_in=D_IN, d_out=D_OUT, num_heads=NUM_HEADS, num_kv_heads=NUM_HEADS, dropout=0.0,
    )
    mha_gqa.eval()

    x = torch.randn(BATCH, SEQ_LEN, D_IN)
    out_gqa, _ = mha_gqa(x)

    # Output shape should be identical
    assert out_gqa.shape == (BATCH, SEQ_LEN, D_OUT)

    # num_groups should be 1 (no expansion needed)
    assert mha_gqa.num_groups == 1


def test_gqa_invalid_head_ratio():
    """num_heads must be divisible by num_kv_heads."""
    with pytest.raises(AssertionError, match="must be divisible"):
        MultiHeadAttention(
            d_in=D_IN, d_out=D_OUT, num_heads=NUM_HEADS, num_kv_heads=3,  # 4 % 3 != 0
        )


def test_mqa_single_kv_head():
    """MQA (num_kv_heads=1) should work correctly."""
    torch.manual_seed(0)
    mqa = MultiHeadAttention(
        d_in=D_IN, d_out=D_OUT, num_heads=NUM_HEADS, num_kv_heads=1, dropout=0.0,
    )
    mqa.eval()

    x = torch.randn(BATCH, SEQ_LEN, D_IN)
    out, cache = mqa(x)

    assert out.shape == (BATCH, SEQ_LEN, D_OUT)
    assert mqa.num_groups == NUM_HEADS  # All queries share 1 KV head
    assert cache is None


def test_gqa_repeat_kv_helper():
    """_repeat_kv should correctly expand KV heads."""
    torch.manual_seed(0)

    # Input: (b=2, num_kv_heads=2, t=4, d_h=8)
    kv = torch.randn(2, 2, 4, 8)

    # Expand by num_groups=2: should get (2, 4, 4, 8)
    expanded = MultiHeadAttention._repeat_kv(kv, num_groups=2)

    assert expanded.shape == (2, 4, 4, 8)

    # Each original KV head should appear num_groups times consecutively
    torch.testing.assert_close(expanded[:, 0, :, :], kv[:, 0, :, :])
    torch.testing.assert_close(expanded[:, 1, :, :], kv[:, 0, :, :])
    torch.testing.assert_close(expanded[:, 2, :, :], kv[:, 1, :, :])
    torch.testing.assert_close(expanded[:, 3, :, :], kv[:, 1, :, :])
