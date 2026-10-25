"""tools/test_bonsai_router.py - no network, no GPU."""
from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from bonsai.router import Backend, classify_model, refuse_reason


class RouterTest(unittest.TestCase):
    def test_strata_pack_dir(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "index.txt").write_text("# align 256 pool 1 tensors 0\n", encoding="utf-8")
            (p / "dense.bin").write_bytes(b"\0")
            c = classify_model(p)
            self.assertEqual(c.backend, Backend.STRATA)

    def test_pq2_filename(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "Ternary-Bonsai-2-27B-PQ2_0.gguf"
            p.write_bytes(b"GGUF" + b"\0" * 64)
            c = classify_model(p)
            self.assertEqual(c.backend, Backend.PRISM)
            self.assertEqual(c.packing, "PQ2_0")
            self.assertTrue(c.needs_prism_binary)
            self.assertIsNotNone(refuse_reason(c, have_prism_binary=False))
            self.assertIsNone(refuse_reason(c, have_prism_binary=True))

    def test_ptq1_type_id_in_blob(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "model.gguf"
            p.write_bytes(b"GGUF" + b"\0" * 32 + struct.pack("<I", 143) + b"\0" * 32)
            c = classify_model(p)
            self.assertEqual(c.backend, Backend.PRISM)
            self.assertTrue(c.needs_prism_binary)

    def test_q2_upstream(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "Ternary-Bonsai-2-27B-Q2_0-prism-fork-required.gguf"
            p.write_bytes(b"GGUF" + b"\0" * 64)
            c = classify_model(p)
            self.assertEqual(c.packing, "Q2_0")
            self.assertFalse(c.needs_prism_binary)


if __name__ == "__main__":
    unittest.main()
