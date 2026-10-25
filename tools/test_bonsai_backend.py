"""Unit tests for serve.bonsai_backend helpers (no GPU, no network)."""
from __future__ import annotations

import unittest

from serve.bonsai_backend import build_server_args


class BuildArgsTest(unittest.TestCase):
    def test_cpu_and_timeouts(self):
        args = build_server_args(
            exe="llama-server",
            model="m.gguf",
            host="127.0.0.1",
            port=8081,
            ctx=2048,
            ngl=0,
            fit=False,
            reasoning="off",
            reasoning_budget=0,
            server_timeout_s=3600,
        )
        self.assertIn("--timeout", args)
        self.assertEqual(args[args.index("--timeout") + 1], "3600")
        self.assertEqual(args[args.index("-ngl") + 1], "0")
        self.assertEqual(args[args.index("--reasoning") + 1], "off")
        self.assertNotIn("--fit", args)

    def test_gpu_fit_like_lm_studio(self):
        args = build_server_args(
            exe="llama-server",
            model="m.gguf",
            host="127.0.0.1",
            port=8081,
            ctx=8192,
            ngl=40,
            fit=True,
            reasoning="off",
            reasoning_budget=0,
            server_timeout_s=3600,
        )
        self.assertEqual(args[args.index("-ngl") + 1], "40")
        self.assertEqual(args[args.index("--fit") + 1], "on")

    def test_omit_ngl_when_none(self):
        args = build_server_args(
            exe="llama-server",
            model="m.gguf",
            host="127.0.0.1",
            port=8081,
            ctx=8192,
            ngl=None,
            fit=True,
            reasoning="on",
            reasoning_budget=-1,
            server_timeout_s=7200,
        )
        self.assertNotIn("-ngl", args)
        self.assertEqual(args[args.index("--fit") + 1], "on")


if __name__ == "__main__":
    unittest.main()
