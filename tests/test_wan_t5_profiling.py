"""Exercise profiling with tiny weights; never load the full T5 checkpoint."""

from contextlib import redirect_stdout
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from mlx_video.models.wan_2.text_encoder import T5Encoder
from mlx_video.models.wan_2.utils import _profile_stage, encode_text, load_t5_encoder


class TestT5Profiling(unittest.TestCase):
    def test_disabled_stage_does_not_synchronize_or_log(self):
        output = io.StringIO()
        with patch.object(mx, "synchronize") as synchronize, redirect_stdout(output):
            with _profile_stage("disabled"):
                pass
        synchronize.assert_not_called()
        self.assertEqual(output.getvalue(), "")

    def test_profiled_encoding_matches_baseline(self):
        encoder = T5Encoder(
            vocab_size=16, dim=8, dim_attn=8, dim_ffn=16,
            num_heads=2, num_layers=1,
        )

        def tokenizer(prompt, **kwargs):
            return {
                "input_ids": np.array([[2, 3, 1, 0]], dtype=np.int32),
                "attention_mask": np.array([[1, 1, 1, 0]], dtype=np.int32),
            }

        baseline = encode_text(encoder, tokenizer, "test", 4)
        mx.eval(baseline)
        output = io.StringIO()
        with redirect_stdout(output):
            profiled = encode_text(
                encoder, tokenizer, "test", 4,
                profile=True, profile_label="positive",
            )
        np.testing.assert_array_equal(np.array(baseline), np.array(profiled))
        self.assertIn("tokens=3, padded_length=4", output.getvalue())
        self.assertIn("positive: output evaluation", output.getvalue())
        self.assertIn("MLX active=", output.getvalue())
        self.assertIn("process peak RSS=", output.getvalue())

    def test_profiled_loading_preserves_checkpoint_precision(self):
        encoder = T5Encoder(
            vocab_size=16, dim=8, dim_attn=8, dim_ffn=16,
            num_heads=2, num_layers=1,
        )
        from mlx.utils import tree_flatten

        weights = {k: v.astype(mx.bfloat16) for k, v in tree_flatten(encoder.parameters())}
        config = SimpleNamespace(
            t5_vocab_size=16, t5_dim=8, t5_dim_attn=8, t5_dim_ffn=16,
            t5_num_heads=2, t5_num_layers=1, t5_num_buckets=32,
        )
        output = io.StringIO()
        with TemporaryDirectory() as directory, redirect_stdout(output):
            path = Path(directory) / "t5.safetensors"
            mx.save_safetensors(str(path), weights)
            loaded = load_t5_encoder(path, config, profile=True)
        for name, weight in tree_flatten(loaded.parameters()):
            self.assertEqual(weight.dtype, mx.bfloat16)
            np.testing.assert_array_equal(
                np.array(weight.astype(mx.float32)),
                np.array(weights[name].astype(mx.float32)),
            )
        self.assertIn("checkpoint bytes by dtype", output.getvalue())
        self.assertIn("T5 weight evaluation", output.getvalue())
