import pytest
import torch

from atomiclm.model.norm import RMSNorm

D_MODEL = 32
BATCH = 2
SEQ_LEN = 8


@pytest.fixture
def norm():
    torch.manual_seed(42)
    return RMSNorm(d_model=D_MODEL)


def test_output_shape(norm):
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    out = norm(x)
    assert out.shape == (BATCH, SEQ_LEN, D_MODEL)


def test_unit_rms(norm):
    """After normalization with weight=ones, RMS along last dim should be ~1."""
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    out = norm(x)
    rms = out.pow(2).mean(-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-5, rtol=1e-5)


def test_weight_scaling(norm):
    """Setting weight to constant c should scale output by c."""
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)

    out_ones = norm(x)

    with torch.no_grad():
        norm.weight.fill_(3.0)
    out_scaled = norm(x)

    torch.testing.assert_close(out_scaled, out_ones * 3.0)


def test_zero_input(norm):
    """All-zero input must produce all-zero output (no NaN)."""
    x = torch.zeros(1, 1, D_MODEL)
    out = norm(x)
    assert torch.isfinite(out).all()
    assert (out == 0).all()


def test_gradient_flows(norm):
    x = torch.randn(1, SEQ_LEN, D_MODEL, requires_grad=True)
    out = norm(x)
    out.sum().backward()

    assert x.grad is not None
    assert norm.weight.grad is not None


def test_deterministic(norm):
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    out1 = norm(x)
    out2 = norm(x)
    torch.testing.assert_close(out1, out2)
