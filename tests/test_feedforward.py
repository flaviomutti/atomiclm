import pytest
import torch

from atomiclm.model.feedforward import FeedForward

D_MODEL = 32
D_FF = 128
BATCH = 2
SEQ_LEN = 8


@pytest.fixture
def ffn_standard():
    torch.manual_seed(42)
    m = FeedForward(d_model=D_MODEL, d_ff=D_FF, dropout=0.0, use_gated=False)
    m.eval()
    return m


@pytest.fixture
def ffn_gated():
    torch.manual_seed(42)
    m = FeedForward(d_model=D_MODEL, d_ff=D_FF, dropout=0.0, use_gated=True)
    m.eval()
    return m


def test_output_shape_standard(ffn_standard):
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    out = ffn_standard(x)
    assert out.shape == (BATCH, SEQ_LEN, D_MODEL)


def test_output_shape_gated(ffn_gated):
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    out = ffn_gated(x)
    assert out.shape == (BATCH, SEQ_LEN, D_MODEL)


def test_standard_vs_gated_differ():
    """Standard and gated variants should produce different outputs."""
    torch.manual_seed(0)
    x = torch.randn(1, SEQ_LEN, D_MODEL)

    torch.manual_seed(42)
    standard = FeedForward(D_MODEL, D_FF, use_gated=False).eval()
    torch.manual_seed(42)
    gated = FeedForward(D_MODEL, D_FF, use_gated=True).eval()

    out_std = standard(x)
    out_gated = gated(x)
    assert not torch.allclose(out_std, out_gated)


def test_batch_consistency(ffn_standard):
    """Identical sequences in different batch slots produce identical output."""
    single = torch.randn(1, SEQ_LEN, D_MODEL)
    batched = single.expand(BATCH, -1, -1)
    out = ffn_standard(batched)

    for i in range(1, BATCH):
        torch.testing.assert_close(out[0], out[i])


def test_gradient_flows_standard(ffn_standard):
    ffn_standard.train()
    x = torch.randn(1, SEQ_LEN, D_MODEL, requires_grad=True)
    out = ffn_standard(x)
    out.sum().backward()

    assert x.grad is not None
    for p in ffn_standard.parameters():
        assert p.grad is not None


def test_gradient_flows_gated(ffn_gated):
    ffn_gated.train()
    x = torch.randn(1, SEQ_LEN, D_MODEL, requires_grad=True)
    out = ffn_gated(x)
    out.sum().backward()

    assert x.grad is not None
    for p in ffn_gated.parameters():
        assert p.grad is not None


def test_parameter_count_gated():
    """Gated variant has 3 bias-free projections; standard has 2 with bias."""
    standard = FeedForward(D_MODEL, D_FF, use_gated=False)
    gated = FeedForward(D_MODEL, D_FF, use_gated=True)

    # Standard: fc1 weight + bias + fc2 weight + bias
    std_params = sum(p.numel() for p in standard.parameters())
    expected_std = D_MODEL * D_FF + D_FF + D_FF * D_MODEL + D_MODEL
    assert std_params == expected_std

    # Gated: 3 weight matrices, no bias
    gated_params = sum(p.numel() for p in gated.parameters())
    expected_gated = 3 * D_MODEL * D_FF
    assert gated_params == expected_gated
