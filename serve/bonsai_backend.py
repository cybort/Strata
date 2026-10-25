"""PrismML llama.cpp subprocess backend for Ternary-Bonsai GGUF.

Starts `llama-server` (Prism pin or STRATA_PRISM_LLAMA) and exposes an OpenAI HTTP
client used by `--engine bonsai`. Partial GPU offload matches LM Studio: `-ngl N`
(layers in VRAM) plus optional `--fit on` to auto-trim to free VRAM.
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
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]

# Client HTTP timeout for /v1/chat/completions (seconds). Thinking + CPU can exceed 10 minutes.
DEFAULT_CHAT_TIMEOUT_S = int(os.environ.get("STRATA_BONSAI_CHAT_TIMEOUT", "3600"))
DEFAULT_SERVER_TIMEOUT_S = int(os.environ.get("STRATA_BONSAI_SERVER_TIMEOUT", "3600"))
DEFAULT_HEALTH_TIMEOUT_S = float(os.environ.get("STRATA_BONSAI_HEALTH_TIMEOUT", "300"))


def _looks_prism_install(path: str | Path) -> bool:
    """Stock winget/upstream llama-server is not Prism (PQ2_0/PTQ1_0 would be unknown types)."""
    p = Path(path)
    parts = " ".join(p.parts).lower()
    if "prism" in parts or "prism-llama" in parts:
        return True
    pin = ROOT / "third_party" / "prism-llama" / "PIN.json"
    if pin.is_file():
        try:
            return str(p.resolve()).lower().startswith(
                str((ROOT / "third_party" / "prism-llama").resolve()).lower()
            )
        except OSError:
            return False
    return False


def resolve_llama_server(*, require_prism: bool = True) -> str | None:
    """Locate llama-server. When require_prism is True, ignore stock PATH installs."""
    env = os.environ.get("STRATA_PRISM_LLAMA")
    if env and Path(env).exists():
        return env
    base = ROOT / "third_party" / "prism-llama"
    if base.is_dir():
        # Prefer a CUDA build tree if both cpu/ and cuda/ exist
        for name in (
            "cuda/llama-server.exe",
            "cuda/bin/llama-server.exe",
            "bin/llama-server.exe",
            "llama-server.exe",
            "llama-server",
            "bin/llama-server",
        ):
            p = base / name
            if p.exists():
                return str(p)
        for p in base.rglob("llama-server*"):
            if p.is_file() and "server" in p.name.lower() and not p.name.endswith(".dll"):
                return str(p)
    which = shutil.which("llama-server")
    if which and (not require_prism or _looks_prism_install(which)):
        return which
    env2 = os.environ.get("LLAMA_SERVER")
    if env2 and Path(env2).exists() and not require_prism:
        return env2
    return None


def have_prism_binary() -> bool:
    return resolve_llama_server(require_prism=True) is not None


def binary_supports_cuda(exe: str | None) -> bool:
    """Heuristic: CUDA Prism zips ship ggml-cuda / cudart next to llama-server."""
    if not exe:
        return False
    p = Path(exe).resolve().parent
    markers = ("ggml-cuda.dll", "ggml-cuda.so", "cudart64_12.dll", "cudart64_13.dll", "libcudart.so")
    if any((p / m).exists() for m in markers):
        return True
    # Name / path hints
    blob = str(p).lower()
    return "cuda" in blob and "cpu" not in Path(exe).name.lower()


def build_server_args(
    *,
    exe: str,
    model: str,
    host: str,
    port: int,
    ctx: int,
    ngl: int | None,
    fit: bool,
    reasoning: str,
    reasoning_budget: int,
    server_timeout_s: int,
    extra_args: list[str] | None = None,
) -> list[str]:
    """CLI for llama-server. `ngl=None` means omit and let --fit choose; `ngl=0` is CPU-only."""
    args = [
        exe,
        "-m", model,
        "--host", host,
        "--port", str(port),
        "-c", str(ctx),
        "--timeout", str(server_timeout_s),
        "--reasoning", reasoning,
        "--reasoning-budget", str(reasoning_budget),
    ]
    if ngl is not None:
        args += ["-ngl", str(int(ngl))]
    if fit:
        args += ["--fit", "on"]
    if extra_args:
        args += list(extra_args)
    return args


class BonsaiServer:
    """Manage a llama-server process for one GGUF (optional partial GPU offload)."""

    def __init__(
        self,
        model: str,
        host: str = "127.0.0.1",
        port: int = 8081,
        ngl: int | None = 99,
        ctx: int = 8192,
        exe: str | None = None,
        extra_args: list[str] | None = None,
        fit: bool = True,
        reasoning: str = "off",
        reasoning_budget: int = 0,
        chat_timeout_s: int = DEFAULT_CHAT_TIMEOUT_S,
        server_timeout_s: int = DEFAULT_SERVER_TIMEOUT_S,
        health_timeout_s: float = DEFAULT_HEALTH_TIMEOUT_S,
    ):
        self.model = model
        self.host = host
        self.port = port
        self.ngl = ngl
        self.ctx = ctx
        self.fit = fit
        self.reasoning = reasoning
        self.reasoning_budget = reasoning_budget
        self.chat_timeout_s = chat_timeout_s
        self.server_timeout_s = server_timeout_s
        self.health_timeout_s = health_timeout_s
        self.exe = exe or resolve_llama_server(require_prism=True) or resolve_llama_server(require_prism=False)
        self.extra_args = list(extra_args or [])
        self.proc: subprocess.Popen | None = None
        self.base = f"http://{host}:{port}"
        self._stderr_tail = ""

    def start(self) -> None:
        if not self.exe:
            raise FileNotFoundError(
                "llama-server not found. Run: python tools/bonsai/fetch_prism.py --variant cuda "
                "and extract it, or set STRATA_PRISM_LLAMA to the Prism binary."
            )
        ngl = self.ngl
        if ngl is not None and ngl > 0 and not binary_supports_cuda(self.exe):
            # CPU-only Prism zip: -ngl > 0 is silently ignored / fails — force CPU and say so.
            print(
                f"[bonsai] {self.exe} looks CPU-only; using -ngl 0. "
                "For LM Studio-style GPU offload: python tools/bonsai/fetch_prism.py --variant cuda",
                flush=True,
            )
            ngl = 0
            # fit is useless without a GPU backend
            fit = False
        else:
            fit = self.fit

        args = build_server_args(
            exe=self.exe,
            model=self.model,
            host=self.host,
            port=self.port,
            ctx=self.ctx,
            ngl=ngl,
            fit=fit,
            reasoning=self.reasoning,
            reasoning_budget=self.reasoning_budget,
            server_timeout_s=self.server_timeout_s,
            extra_args=self.extra_args,
        )
        print(f"[bonsai] starting: {' '.join(args)}", flush=True)
        self.proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        deadline = time.time() + self.health_timeout_s
        last_err = ""
        while time.time() < deadline:
            if self.proc.poll() is not None:
                err = (self.proc.stderr.read() or b"").decode("utf-8", "replace") if self.proc.stderr else ""
                self._stderr_tail = err[-4000:]
                raise RuntimeError(f"llama-server exited early: {self._stderr_tail}")
            try:
                with urllib.request.urlopen(self.base + "/health", timeout=2) as r:
                    if r.status == 200:
                        print(
                            f"[bonsai] ready on {self.base} (ngl={ngl}, fit={fit}, "
                            f"reasoning={self.reasoning}, chat_timeout={self.chat_timeout_s}s)",
                            flush=True,
                        )
                        return
            except Exception as e:
                last_err = str(e)
            time.sleep(0.5)
        raise TimeoutError(f"llama-server did not become healthy in {self.health_timeout_s}s: {last_err}")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 64,
        temperature: float = 1.0,
        stream: bool = False,
        extra_body: dict[str, Any] | None = None,
    ) -> dict | Iterator[bytes]:
        payload: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        # Mirror server defaults on the request so a client that enables thinking cannot hang forever.
        if self.reasoning == "off":
            payload.setdefault("chat_template_kwargs", {})
            if isinstance(payload["chat_template_kwargs"], dict):
                payload["chat_template_kwargs"].setdefault("enable_thinking", False)
        if extra_body:
            payload.update(extra_body)
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base + "/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            resp = urllib.request.urlopen(req, timeout=self.chat_timeout_s)
        except TimeoutError as e:
            raise TimeoutError(
                f"bonsai chat timed out after {self.chat_timeout_s}s "
                f"(raise STRATA_BONSAI_CHAT_TIMEOUT, lower max_tokens, set reasoning=off, or use CUDA -ngl)"
            ) from e
        except urllib.error.URLError as e:
            if isinstance(e.reason, TimeoutError) or "timed out" in str(e).lower():
                raise TimeoutError(
                    f"bonsai chat timed out after {self.chat_timeout_s}s "
                    f"(raise STRATA_BONSAI_CHAT_TIMEOUT, or offload layers with ngl/fit on a CUDA Prism build)"
                ) from e
            raise
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"bonsai chat failed: {e.read().decode('utf-8', 'replace')}") from e
        if stream:
            return resp
        return json.loads(resp.read().decode("utf-8"))


class BonsaiEngine:
    """Minimal Engine-shaped adapter: generate() yields UTF-8 bytes as token-id stand-ins."""

    def __init__(self, model: str, **server_kw):
        self.server = BonsaiServer(model, **server_kw)
        self.server.start()
        self.max_context = server_kw.get("ctx", 8192)
        self.info: dict = {
            "backend": "prism-llama",
            "model": model,
            "ngl": server_kw.get("ngl", 99),
            "fit": server_kw.get("fit", True),
            "cuda": binary_supports_cuda(self.server.exe),
        }

    def close(self) -> None:
        self.server.stop()

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        prompt = bytes(i for i in ids if 0 <= i < 256).decode("utf-8", "replace")
        temp = 1.0
        if isinstance(sampling, dict) and "temperature" in sampling:
            temp = float(sampling["temperature"])
        out = self.server.chat(
            [{"role": "user", "content": prompt or "Hi"}],
            max_tokens=max(1, int(max_new)),
            temperature=temp,
        )
        text = (out.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        for b in text.encode("utf-8"):
            if cancel is not None and getattr(cancel, "is_set", lambda: False)():
                break
            yield b
