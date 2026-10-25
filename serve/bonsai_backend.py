"""PrismML llama.cpp subprocess backend for Ternary-Bonsai GGUF.

Starts `llama-server` (Prism pin or STRATA_PRISM_LLAMA) and exposes a small OpenAI
HTTP client used by `--engine bonsai`. Does not reimplement PQ2_0/PTQ1_0 kernels.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]


def _looks_prism_install(path: str | Path) -> bool:
    """Stock winget/upstream llama-server is not Prism (PQ2_0/PTQ1_0 would be unknown types)."""
    p = Path(path)
    parts = " ".join(p.parts).lower()
    if "prism" in parts or "prism-llama" in parts:
        return True
    pin = ROOT / "third_party" / "prism-llama" / "PIN.json"
    if pin.is_file():
        try:
            # Any binary under the pinned tree counts once fetch_prism wrote PIN.json
            return str(p.resolve()).lower().startswith(str((ROOT / "third_party" / "prism-llama").resolve()).lower())
        except OSError:
            return False
    return False


def resolve_llama_server(*, require_prism: bool = True) -> str | None:
    """Locate llama-server. When require_prism is True, ignore stock PATH installs."""
    env = os.environ.get("STRATA_PRISM_LLAMA")
    if env and Path(env).exists():
        return env  # user override is trusted
    base = ROOT / "third_party" / "prism-llama"
    if base.is_dir():
        for name in ("llama-server.exe", "llama-server", "bin/llama-server.exe", "bin/llama-server"):
            p = base / name
            if p.exists():
                return str(p)
        for p in base.rglob("llama-server*"):
            if p.is_file() and "server" in p.name.lower() and not p.name.endswith(".dll"):
                return str(p)
    which = shutil.which("llama-server")
    if which and (not require_prism or _looks_prism_install(which)):
        return which
    # Upstream-only Q2_0 path may set LLAMA_SERVER explicitly
    env2 = os.environ.get("LLAMA_SERVER")
    if env2 and Path(env2).exists() and not require_prism:
        return env2
    return None


def have_prism_binary() -> bool:
    return resolve_llama_server(require_prism=True) is not None


class BonsaiServer:
    """Manage a llama-server process for one GGUF."""

    def __init__(self, model: str, host: str = "127.0.0.1", port: int = 8081, ngl: int = 99,
                 ctx: int = 8192, exe: str | None = None, extra_args: list[str] | None = None):
        self.model = model
        self.host = host
        self.port = port
        self.ngl = ngl
        self.ctx = ctx
        self.exe = exe or resolve_llama_server(require_prism=True) or resolve_llama_server(require_prism=False)
        self.extra_args = list(extra_args or [])
        self.proc: subprocess.Popen | None = None
        self.base = f"http://{host}:{port}"

    def start(self, timeout_s: float = 120.0) -> None:
        if not self.exe:
            raise FileNotFoundError(
                "llama-server not found. Run: python tools/bonsai/fetch_prism.py "
                "and extract it, or set STRATA_PRISM_LLAMA to the Prism binary."
            )
        args = [
            self.exe, "-m", self.model, "--host", self.host, "--port", str(self.port),
            "-c", str(self.ctx), "-ngl", str(self.ngl),
        ] + self.extra_args
        self.proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        deadline = time.time() + timeout_s
        last_err = ""
        while time.time() < deadline:
            if self.proc.poll() is not None:
                err = (self.proc.stderr.read() or b"").decode("utf-8", "replace") if self.proc.stderr else ""
                raise RuntimeError(f"llama-server exited early: {err[-2000:]}")
            try:
                with urllib.request.urlopen(self.base + "/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except Exception as e:
                last_err = str(e)
            time.sleep(0.5)
        raise TimeoutError(f"llama-server did not become healthy: {last_err}")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def chat(self, messages: list[dict], max_tokens: int = 64, temperature: float = 1.0,
             stream: bool = False) -> dict | Iterator[bytes]:
        body = json.dumps({
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.base + "/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            resp = urllib.request.urlopen(req, timeout=600)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"bonsai chat failed: {e.read().decode('utf-8', 'replace')}") from e
        if stream:
            return resp
        return json.loads(resp.read().decode("utf-8"))


class BonsaiEngine:
    """Minimal Engine-shaped adapter: generate() yields fake token ids from decoded text bytes.

    Used so serve.Service can drive a Bonsai model without Strata's pack/tokenizer contract.
    Detokenization of the yielded ids is identity Latin-1 of UTF-8 bytes (good enough for smoke).
    """

    def __init__(self, model: str, **server_kw):
        self.server = BonsaiServer(model, **server_kw)
        self.server.start()
        self.max_context = server_kw.get("ctx", 8192)
        self.info: dict = {"backend": "prism-llama", "model": model}

    def close(self) -> None:
        self.server.stop()

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        # ids are ignored for routing; Service still builds a prompt — we recover text poorly.
        # Prefer chat() via the bonsai HTTP path. This fallback decodes ids as bytes when < 256.
        prompt = bytes(i for i in ids if 0 <= i < 256).decode("utf-8", "replace")
        temp = 1.0
        if isinstance(sampling, dict) and "temperature" in sampling:
            temp = float(sampling["temperature"])
        out = self.server.chat([{"role": "user", "content": prompt or "Hi"}], max_tokens=max_new,
                               temperature=temp)
        text = out["choices"][0]["message"]["content"]
        for b in text.encode("utf-8"):
            if cancel is not None and getattr(cancel, "is_set", lambda: False)():
                break
            yield b
