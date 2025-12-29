import torch
import pytest

from atomiclm.model.rope import RotaryEmbedding

HEAD_DIM = 16
MAX_SEQ_LEN = 64
BATCH = 2
NUM_HEADS = 4
SEQ_LEN = 8


@pytest.fixture
def rope():
    return RotaryEmbedding(head_dim=HEAD_DIM, max_seq_len=MAX_SEQ_LEN)


def test_output_shape(rope):
    """apply_rotary must preserve tensor shape."""
    x = torch.randn(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    freqs_cis = rope(SEQ_LEN)

    out = RotaryEmbedding.apply_rotary(x, freqs_cis)

    assert out.shape == x.shape


def test_rotation_equivariance(rope):
    """Rotating Q and K by the same angle preserves their dot product."""
    q = torch.randn(1, 1, 1, HEAD_DIM)
    k = torch.randn(1, 1, 1, HEAD_DIM)

    dot_before = (q * k).sum()

    freqs_cis = rope(1, offset=5)  # arbitrary position
    q_rot = RotaryEmbedding.apply_rotary(q, freqs_cis)
    k_rot = RotaryEmbedding.apply_rotary(k, freqs_cis)

    dot_after = (q_rot * k_rot).sum()

    torch.testing.assert_close(dot_before, dot_after, atol=1e-5, rtol=1e-5)


def test_position_sensitivity(rope):
    """Same token at different positions must get different representations."""
    x = torch.randn(1, 1, 1, HEAD_DIM)

    freqs_pos0 = rope(1, offset=0)
    freqs_pos5 = rope(1, offset=5)

    out_pos0 = RotaryEmbedding.apply_rotary(x, freqs_pos0)
    out_pos5 = RotaryEmbedding.apply_rotary(x, freqs_pos5)

    assert not torch.allclose(out_pos0, out_pos5)


def test_relative_distance(rope):
    """Q at pos i dot K at pos j depends only on i-j."""
    q = torch.randn(1, 1, 1, HEAD_DIM)
    k = torch.randn(1, 1, 1, HEAD_DIM)

    # Pair 1: Q@3, K@1 -> distance = 2
    q_rot_3 = RotaryEmbedding.apply_rotary(q, rope(1, offset=3))
    k_rot_1 = RotaryEmbedding.apply_rotary(k, rope(1, offset=1))
    dot_a = (q_rot_3 * k_rot_1).sum()

    # Pair 2: Q@10, K@8 -> distance = 2
    q_rot_10 = RotaryEmbedding.apply_rotary(q, rope(1, offset=10))
    k_rot_8 = RotaryEmbedding.apply_rotary(k, rope(1, offset=8))
    dot_b = (q_rot_10 * k_rot_8).sum()

    torch.testing.assert_close(dot_a, dot_b, atol=1e-5, rtol=1e-5)


def test_offset_matches_full(rope):
    """rope(t, offset=p) must equal rope(p+t)[p:]."""
    p, t = 5, 3

    from_offset = rope(t, offset=p)
    from_full = rope(p + t)[p:]

    torch.testing.assert_close(from_offset, from_full)


def test_dtype_preserved(rope):
    """Output dtype should match input dtype."""
    x = torch.randn(1, 1, SEQ_LEN, HEAD_DIM)
    freqs_cis = rope(SEQ_LEN)

    out = RotaryEmbedding.apply_rotary(x, freqs_cis)
    assert out.dtype == x.dtype

    x_half = x.half()
    out_half = RotaryEmbedding.apply_rotary(x_half, freqs_cis)
    assert out_half.dtype == torch.float16
