"""Wan model loading utilities."""

from contextlib import contextmanager
from pathlib import Path
import resource
import sys
import time

import mlx.core as mx
import mlx.nn as nn


@contextmanager
def _profile_stage(label, enabled=False):
    """Diagnostic boundaries synchronize MLX; peak values cover the process lifetime."""
    if not enabled:
        yield
        return

    def memory():
        gib = 1024**3
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_bytes = rss if sys.platform == "darwin" else rss * 1024
        return (
            f"MLX active={mx.get_active_memory() / gib:.2f} GiB, "
            f"cache={mx.get_cache_memory() / gib:.2f} GiB, "
            f"peak={mx.get_peak_memory() / gib:.2f} GiB, "
            f"process peak RSS={rss_bytes / gib:.2f} GiB"
        )

    print(f"[profile] {label}: start | {memory()}", flush=True)
    mx.synchronize()
    start = time.perf_counter()
    yield
    mx.synchronize()
    print(
        f"[profile] {label}: {time.perf_counter() - start:.3f}s | {memory()}",
        flush=True,
    )


def load_wan_model(
    model_path: Path,
    config,
    quantization: dict | None = None,
    loras: list | None = None,
):
    """Load and initialize WanModel, with optional quantization and LoRA support.

    Args:
        model_path: Path to model safetensors file
        config: WanModelConfig
        quantization: Optional dict with 'bits' and 'group_size' keys.
                      If provided, creates QuantizedLinear stubs before loading.
        loras: Optional list of (lora_path, strength) tuples to apply.
    """
    from mlx_video.models.wan_2.wan_2 import WanModel

    model = WanModel(config)

    if quantization:
        from mlx_video.models.wan_2.convert import _quantize_predicate

        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            class_predicate=lambda path, m: _quantize_predicate(path, m),
        )

    weights = mx.load(str(model_path))

    # Apply LoRAs: dequantize+merge for quantized models, weight merge for bf16
    if loras:
        if quantization:
            # Dequantize LoRA-targeted layers, merge delta, replace with bf16 Linear.
            # Non-LoRA layers stay 4-bit. Zero per-step overhead.
            from mlx_video.models.wan_2.convert import _load_lora_configs
            from mlx_video.lora import apply_loras_to_model

            model.load_weights(list(weights.items()), strict=False)
            mx.eval(model.parameters())
            module_to_loras = _load_lora_configs(loras)
            apply_loras_to_model(model, module_to_loras)
            mx.eval(model.parameters())
            return model
        else:
            # Weight merging: fold LoRA into bf16 weights before loading
            from mlx_video.models.wan_2.convert import load_and_apply_loras

            weights = load_and_apply_loras(dict(weights), loras)

    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
    return model


def load_t5_encoder(model_path: Path, config, *, profile=False):
    """Load T5 text encoder.

    Preserves checkpoint precision to avoid expanding BF16 weights to FP32.
    Attention scores and softmax are still computed in float32.
    """
    from mlx_video.models.wan_2.text_encoder import T5Encoder

    with _profile_stage("T5 construction", profile):
        encoder = T5Encoder(
            vocab_size=config.t5_vocab_size,
            dim=config.t5_dim,
            dim_attn=config.t5_dim_attn,
            dim_ffn=config.t5_dim_ffn,
            num_heads=config.t5_num_heads,
            num_layers=config.t5_num_layers,
            num_buckets=config.t5_num_buckets,
            shared_pos=False,
        )
    with _profile_stage("T5 checkpoint load call (may defer I/O)", profile):
        weights = mx.load(str(model_path))
    if profile:
        sizes = {}
        for weight in weights.values():
            dtype = str(weight.dtype)
            sizes[dtype] = sizes.get(dtype, 0) + weight.nbytes
        print(f"[profile] T5 checkpoint bytes by dtype: {sizes}", flush=True)
    with _profile_stage("T5 weight assignment", profile):
        # weights = {k: v.astype(mx.float32) for k, v in weights.items()}
        encoder.load_weights(list(weights.items()))
    with _profile_stage("T5 weight evaluation (includes deferred load)", profile):
        mx.eval(encoder.parameters())
    return encoder


def load_vae_decoder(model_path: Path, config=None):
    """Load VAE decoder (skips encoder weights with strict=False).

    For Wan2.2 (vae_z_dim=48), uses Wan22VAEDecoder.
    For Wan2.1 (vae_z_dim=16), uses WanVAE.
    """
    is_wan22 = config is not None and config.vae_z_dim == 48

    if is_wan22:
        from mlx_video.models.wan_2.vae22 import Wan22VAEDecoder

        vae = Wan22VAEDecoder(z_dim=48)
    else:
        from mlx_video.models.wan_2.vae import WanVAE

        vae = WanVAE(z_dim=16)

    weights = mx.load(str(model_path))
    # Upcast VAE weights to float32 for quality — official Wan2.2 runs VAE in float32
    weights = {k: v.astype(mx.float32) for k, v in weights.items()}
    vae.load_weights(list(weights.items()), strict=False)
    mx.eval(vae.parameters())
    return vae


def load_vae_encoder(model_path: Path, config=None):
    """Load VAE encoder for I2V image encoding.

    For Wan2.2 TI2V (vae_z_dim=48), uses Wan22VAEEncoder.
    For Wan2.1/I2V-14B (vae_z_dim=16), uses WanVAE with encoder=True.
    """
    if config is not None and config.vae_z_dim == 16:
        from mlx_video.models.wan_2.vae import WanVAE

        vae = WanVAE(z_dim=16, encoder=True)
    else:
        from mlx_video.models.wan_2.vae22 import Wan22VAEEncoder

        vae = Wan22VAEEncoder(z_dim=config.vae_z_dim if config else 48)

    weights = mx.load(str(model_path))
    weights = {k: v.astype(mx.float32) for k, v in weights.items()}
    vae.load_weights(list(weights.items()), strict=False)
    mx.eval(vae.parameters())
    return vae


def _clean_text(text: str) -> str:
    """Clean text matching official Wan2.2 tokenizer preprocessing.

    Applies ftfy.fix_text (fixes mojibake, normalizes fullwidth chars),
    double HTML unescape, and whitespace normalization. Critical for
    correct tokenization of the Chinese negative prompt.
    """
    import html
    import re

    try:
        import ftfy

        text = ftfy.fix_text(text)
    except ImportError:
        pass
    text = html.unescape(html.unescape(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def encode_text(
    encoder,
    tokenizer,
    prompt: str,
    text_len: int = 512,
    *,
    profile=False,
    profile_label="prompt",
) -> mx.array:
    """Encode text prompt using T5 encoder.

    Args:
        encoder: T5Encoder model
        tokenizer: HuggingFace tokenizer
        prompt: Text prompt
        text_len: Maximum text length

    Returns:
        Text embeddings [L, dim]
    """
    with _profile_stage(f"T5 {profile_label}: cleaning/tokenization", profile):
        prompt = _clean_text(prompt)
        tokens = tokenizer(
            prompt,
            max_length=text_len,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
    if profile:
        print(
            f"[profile] T5 {profile_label}: "
            f"tokens={int(tokens['attention_mask'].sum())}, "
            f"padded_length={tokens['input_ids'].shape[-1]}",
            flush=True,
        )
    with _profile_stage(f"T5 {profile_label}: input arrays", profile):
        ids = mx.array(tokens["input_ids"])
        mask = mx.array(tokens["attention_mask"])

    with _profile_stage(f"T5 {profile_label}: forward graph construction", profile):
        embeddings = encoder(ids, mask=mask)

    # Return only non-padding tokens; item() is also a synchronization boundary.
    with _profile_stage(f"T5 {profile_label}: length extraction and slicing", profile):
        seq_len = int(mask.sum().item())
        embeddings = embeddings[0, :seq_len]
    if profile:
        with _profile_stage(f"T5 {profile_label}: output evaluation", profile):
            mx.eval(embeddings)
    return embeddings
