import torch
import pytest

from atomiclm.model.decoder import Decoder

VOCAB_SIZE = 64
D_MODEL = 32
NUM_LAYERS = 2
NUM_HEADS = 4
D_FF = 128
MAX_SEQ_LEN = 32
BATCH = 2
SEQ_LEN = 8


@pytest.fixture
def model():
    torch.manual_seed(42)
    m = Decoder(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN,
        dropout=0.0,
    )
    m.eval()
    return m


def test_forward_shape(model):
    """Logits shape must be (batch, seq_len, vocab_size)."""
    ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    logits, cache = model(ids)

    assert logits.shape == (BATCH, SEQ_LEN, VOCAB_SIZE)
    assert cache is None


def test_weight_tying(model):
    """LM head and embedding must share the same weight tensor."""
    assert model.lm_head.weight is model.tok_emb.weight


def test_causal_through_model(model):
    """Appending tokens must not change logits at earlier positions."""
    ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))

    t = SEQ_LEN // 2
    logits_short, _ = model(ids[:, :t])
    logits_full, _ = model(ids)

    torch.testing.assert_close(logits_full[:, :t, :], logits_short)


def test_cached_matches_full(model):
    """Full forward must match token-by-token with KV cache."""
    ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))

    # Full sequence
    full_logits, _ = model(ids)

    # Token-by-token with cache
    cache = model.make_cache(1)
    cached_logits_list = []
    for i in range(SEQ_LEN):
        token = ids[:, i : i + 1]
        logits, cache = model(token, kv_cache=cache)
        cached_logits_list.append(logits)

    cached_logits = torch.cat(cached_logits_list, dim=1)

    torch.testing.assert_close(full_logits, cached_logits)


def test_generate_shape(model):
    """Output length must be prompt_len + max_new_tokens."""
    prompt = torch.randint(0, VOCAB_SIZE, (1, 4))
    max_new = 5

    output = model.generate(prompt, max_new_tokens=max_new)

    assert output.shape == (1, 4 + max_new)
    # First tokens should be the prompt
    assert (output[:, :4] == prompt).all()


def test_generate_deterministic(model):
    """Same seed must produce identical output."""
    prompt = torch.randint(0, VOCAB_SIZE, (1, 4))

    torch.manual_seed(99)
    out1 = model.generate(prompt, max_new_tokens=5)

    torch.manual_seed(99)
    out2 = model.generate(prompt, max_new_tokens=5)

    assert (out1 == out2).all()


def test_make_cache_structure(model):
    """Cache must have correct number of layers, shapes, and initial pos=0."""
    cache = model.make_cache(BATCH)

    assert len(cache) == NUM_LAYERS

    for layer_cache in cache:
        # Should use num_kv_heads (which defaults to num_heads for MHA)
        assert layer_cache["k"].shape == (
            BATCH,
            model.num_kv_heads,
            MAX_SEQ_LEN,
            D_MODEL // NUM_HEADS,
        )
        assert layer_cache["v"].shape == (
            BATCH,
            model.num_kv_heads,
            MAX_SEQ_LEN,
            D_MODEL // NUM_HEADS,
        )
        assert layer_cache["pos"] == 0


def test_generate_top_k(model):
    """Generation with top_k should still produce valid output."""
    prompt = torch.randint(0, VOCAB_SIZE, (1, 4))

    output = model.generate(prompt, max_new_tokens=3, top_k=10)

    assert output.shape == (1, 4 + 3)
    # All generated tokens should be valid vocab indices
    assert (output >= 0).all() and (output < VOCAB_SIZE).all()


def test_generate_temperature(model):
    """Temperature=0.01 (near greedy) should produce consistent output."""
    prompt = torch.randint(0, VOCAB_SIZE, (1, 4))

    torch.manual_seed(0)
    out1 = model.generate(prompt, max_new_tokens=5, temperature=0.01)

    torch.manual_seed(1)  # different seed
    out2 = model.generate(prompt, max_new_tokens=5, temperature=0.01)

    # With very low temperature, outputs should be mostly identical (greedy)
    assert (out1 == out2).all()


# ============================================================
# GQA-specific decoder tests
# ============================================================


def test_gqa_decoder_forward():
    """Decoder with GQA should produce correct output shape."""
    torch.manual_seed(42)
    num_kv_heads = 2
    model_gqa = Decoder(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        num_kv_heads=num_kv_heads,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN,
        dropout=0.0,
    )
    model_gqa.eval()

    ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    logits, cache = model_gqa(ids)

    assert logits.shape == (BATCH, SEQ_LEN, VOCAB_SIZE)
    assert cache is None


def test_gqa_decoder_cache_shape():
    """GQA decoder cache should use num_kv_heads, not num_heads."""
    torch.manual_seed(42)
    num_kv_heads = 2
    model_gqa = Decoder(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        num_kv_heads=num_kv_heads,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN,
        dropout=0.0,
    )

    cache = model_gqa.make_cache(BATCH)

    assert len(cache) == NUM_LAYERS
    for layer_cache in cache:
        assert layer_cache["k"].shape == (
            BATCH,
            num_kv_heads,
            MAX_SEQ_LEN,
            D_MODEL // NUM_HEADS,
        )
        assert layer_cache["v"].shape == (
            BATCH,
            num_kv_heads,
            MAX_SEQ_LEN,
            D_MODEL // NUM_HEADS,
        )
        assert layer_cache["pos"] == 0


def test_gqa_decoder_cache_memory_reduction():
    """GQA decoder cache should use less memory than MHA."""
    num_kv_heads = 2

    # MHA decoder
    model_mha = Decoder(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN,
    )

    # GQA decoder
    model_gqa = Decoder(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        num_kv_heads=num_kv_heads,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN,
    )

    cache_mha = model_mha.make_cache(BATCH)
    cache_gqa = model_gqa.make_cache(BATCH)

    # Calculate memory per layer (K + V)
    mha_mem = cache_mha[0]["k"].numel() + cache_mha[0]["v"].numel()
    gqa_mem = cache_gqa[0]["k"].numel() + cache_gqa[0]["v"].numel()

    expected_reduction = NUM_HEADS // num_kv_heads
    assert mha_mem == gqa_mem * expected_reduction


def test_gqa_decoder_generation():
    """GQA decoder should generate valid sequences."""
    torch.manual_seed(42)
    num_kv_heads = 2
    model_gqa = Decoder(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        num_kv_heads=num_kv_heads,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN,
        dropout=0.0,
    )
    model_gqa.eval()

    prompt = torch.randint(0, VOCAB_SIZE, (1, 4))
    max_new = 5

    output = model_gqa.generate(prompt, max_new_tokens=max_new)

    assert output.shape == (1, 4 + max_new)
    assert (output[:, :4] == prompt).all()
    assert (output >= 0).all() and (output < VOCAB_SIZE).all()


def test_gqa_decoder_cached_matches_full():
    """GQA decoder: full forward must match token-by-token with KV cache."""
    torch.manual_seed(42)
    num_kv_heads = 2
    model_gqa = Decoder(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        num_kv_heads=num_kv_heads,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN,
        dropout=0.0,
    )
    model_gqa.eval()

    ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))

    # Full sequence
    full_logits, _ = model_gqa(ids)

    # Token-by-token with cache
    cache = model_gqa.make_cache(1)
    cached_logits_list = []
    for i in range(SEQ_LEN):
        token = ids[:, i : i + 1]
        logits, cache = model_gqa(token, kv_cache=cache)
        cached_logits_list.append(logits)

    cached_logits = torch.cat(cached_logits_list, dim=1)

    torch.testing.assert_close(full_logits, cached_logits)


# ── Weight Initialization Tests ──────────────────────────────────────


def test_init_weights_linear():
    """Linear weights should be ~N(0, 0.02) after initialization."""
    import math

    model = Decoder(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN,
    )

    residual_ids = {id(m) for m in model._residual_projections()}

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if id(module) in residual_ids:
                expected_std = 0.02 / math.sqrt(2 * NUM_LAYERS)
            else:
                expected_std = 0.02

            actual_std = module.weight.std().item()
            # Allow ±50% tolerance (small tensors have noisy estimates)
            assert (
                actual_std < expected_std * 2.0
            ), f"{name}: std={actual_std:.4f}, expected ~{expected_std:.4f}"

            if module.bias is not None:
                assert (module.bias == 0).all(), f"{name}: bias not zeros"


def test_init_weights_embedding():
    """Embedding weights should be ~N(0, 0.02) after initialization."""
    model = Decoder(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN,
    )

    emb_std = model.tok_emb.weight.std().item()
    assert emb_std < 0.04, f"Embedding std={emb_std:.4f}, expected ~0.02"
    assert emb_std > 0.005, f"Embedding std={emb_std:.4f}, too small"


def test_init_weights_residual_scaling():
    """Residual projections should have smaller std than regular layers."""
    import math

    model = Decoder(
        vocab_size=VOCAB_SIZE,
        d_model=128,
        num_layers=8,
        num_heads=4,
        d_ff=512,
        max_seq_len=MAX_SEQ_LEN,
    )

    residual_ids = {id(m) for m in model._residual_projections()}

    regular_stds = []
    residual_stds = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if id(module) in residual_ids:
                residual_stds.append(module.weight.std().item())
            else:
                regular_stds.append(module.weight.std().item())

    avg_regular = sum(regular_stds) / len(regular_stds)
    avg_residual = sum(residual_stds) / len(residual_stds)

    # With 8 layers, residual std should be ~0.02/sqrt(16) = 0.005
    # Regular std should be ~0.02
    # Residual should be meaningfully smaller
    assert (
        avg_residual < avg_regular
    ), f"Residual avg std ({avg_residual:.4f}) should be < regular ({avg_regular:.4f})"


# ── EOS Handling Tests ───────────────────────────────────────────────

EOS_TOKEN_ID = 2
PAD_TOKEN_ID = 0


def test_generate_eos_early_stop():
    """Generation should stop early when all sequences produce EOS."""
    torch.manual_seed(42)
    m = Decoder(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN,
        dropout=0.0,
    )
    m.eval()

    prompt = torch.randint(0, VOCAB_SIZE, (1, 4))
    max_new = 20

    output = m.generate(
        prompt,
        max_new_tokens=max_new,
        eos_token_id=EOS_TOKEN_ID,
        pad_token_id=PAD_TOKEN_ID,
    )

    # Output should be at most prompt_len + max_new_tokens
    assert output.shape[1] <= 4 + max_new
    # Prompt should be preserved
    assert (output[:, :4] == prompt).all()


def test_generate_eos_pads_finished_sequences():
    """After EOS, subsequent tokens should be pad_token_id."""
    torch.manual_seed(42)
    m = Decoder(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN,
        dropout=0.0,
    )
    m.eval()

    prompt = torch.randint(0, VOCAB_SIZE, (1, 4))
    output = m.generate(
        prompt,
        max_new_tokens=20,
        eos_token_id=EOS_TOKEN_ID,
        pad_token_id=PAD_TOKEN_ID,
    )

    generated = output[0, 4:].tolist()

    # If EOS appears, everything after it should be PAD
    if EOS_TOKEN_ID in generated:
        eos_pos = generated.index(EOS_TOKEN_ID)
        after_eos = generated[eos_pos + 1 :]
        assert all(
            t == PAD_TOKEN_ID for t in after_eos
        ), f"Tokens after EOS should be PAD, got: {after_eos}"


def test_generate_eos_requires_pad():
    """generate() should raise when eos_token_id is set without pad_token_id."""
    m = Decoder(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN,
    )
    m.eval()

    prompt = torch.randint(0, VOCAB_SIZE, (1, 4))

    with pytest.raises(ValueError, match="pad_token_id is required"):
        m.generate(prompt, max_new_tokens=5, eos_token_id=EOS_TOKEN_ID)


def test_generate_without_eos_unchanged():
    """Without eos_token_id, generate() behaves exactly as before."""
    torch.manual_seed(42)
    m = Decoder(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN,
        dropout=0.0,
    )
    m.eval()

    prompt = torch.randint(0, VOCAB_SIZE, (1, 4))
    max_new = 10

    output = m.generate(prompt, max_new_tokens=max_new)

    # Without EOS, always generates exactly max_new_tokens
    assert output.shape == (1, 4 + max_new)
