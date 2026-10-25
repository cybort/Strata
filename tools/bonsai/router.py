"""Classify a model path as Strata Flash-Next pack vs PrismML Ternary-Bonsai GGUF."""
from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Backend(str, Enum):
    STRATA = "strata"  # Qwen3.8-Flash-Next pack (index.txt + dense.bin)
    PRISM = "prism"  # Ternary-Bonsai GGUF (PQ2_0 / PTQ1_0 / Q2_0+Hadamard)
    UNKNOWN = "unknown"


# ggml type ids from PrismML-Eng/llama.cpp (prism branch)
GGML_TYPE_PQ2_0 = 142
GGML_TYPE_PTQ1_0 = 143


@dataclass(frozen=True)
class Classification:
    backend: Backend
    reason: str
    packing: str | None = None  # PQ2_0 | PTQ1_0 | Q2_0 | None
    needs_prism_binary: bool = False


def _looks_like_strata_pack(p: Path) -> bool:
    return (p / "index.txt").is_file() and (p / "dense.bin").is_file()


def _gguf_magic_ok(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(4) == b"GGUF"
    except OSError:
        return False


def _scan_gguf_hints(path: Path) -> tuple[str | None, bool, str]:
    """Best-effort: filename + first 4 MiB for type ids / hadamard key strings.

    Full GGUF parsing is left to llama.cpp; this only routes and warns.
    """
    name = path.name.upper()
    packing = None
    if "PQ2_0" in name or "PQ2" in name:
        packing = "PQ2_0"
    elif "PTQ1_0" in name or "PTQ1" in name:
        packing = "PTQ1_0"
    elif "Q2_0" in name:
        packing = "Q2_0"

    needs_prism = packing in ("PQ2_0", "PTQ1_0")
    hint = f"filename packing={packing}" if packing else "gguf file"

    try:
        with path.open("rb") as f:
            blob = f.read(4 * 1024 * 1024)
    except OSError as e:
        return packing, needs_prism, f"{hint}; read failed: {e}"

    if b"prism.hadamard" in blob or b"hadamard" in blob.lower():
        needs_prism = needs_prism or packing != "Q2_0"
        hint += "; prism.hadamard metadata present"
    # type id bytes little-endian (heuristic)
    if struct.pack("<I", GGML_TYPE_PQ2_0) in blob:
        packing = packing or "PQ2_0"
        needs_prism = True
        hint += "; type 142 (PQ2_0) seen"
    if struct.pack("<I", GGML_TYPE_PTQ1_0) in blob:
        packing = packing or "PTQ1_0"
        needs_prism = True
        hint += "; type 143 (PTQ1_0) seen"
    if b"Ternary-Bonsai" in blob or b"bonsai" in blob.lower() or "BONSAI" in name:
        hint += "; bonsai"
        if packing in ("PQ2_0", "PTQ1_0", None):
            needs_prism = True
    return packing, needs_prism, hint


def classify_model(path: str | os.PathLike[str]) -> Classification:
    p = Path(path)
    if p.is_dir():
        if _looks_like_strata_pack(p):
            return Classification(Backend.STRATA, "pack has index.txt + dense.bin")
        # directory of GGUFs: prefer language model file
        ggufs = sorted(p.glob("*.gguf"))
        if not ggufs:
            return Classification(Backend.UNKNOWN, f"directory has no index.txt/dense.bin or *.gguf: {p}")
        # Prefer non-mmproj
        cands = [g for g in ggufs if "mmproj" not in g.name.lower()] or ggufs
        return classify_model(cands[0])

    if not p.is_file():
        return Classification(Backend.UNKNOWN, f"path does not exist: {p}")

    if p.suffix.lower() == ".gguf" or _gguf_magic_ok(p):
        if not _gguf_magic_ok(p):
            return Classification(Backend.UNKNOWN, f"not a GGUF magic: {p}")
        packing, needs_prism, hint = _scan_gguf_hints(p)
        return Classification(
            Backend.PRISM,
            hint,
            packing=packing,
            needs_prism_binary=needs_prism if packing != "Q2_0" else False,
        )

    if p.name == "index.txt" or p.name == "dense.bin":
        return classify_model(p.parent)

    return Classification(Backend.UNKNOWN, f"unrecognized model path: {p}")


def refuse_reason(c: Classification, have_prism_binary: bool) -> str | None:
    """Return an error string if this classification must not be run, else None."""
    if c.backend == Backend.UNKNOWN:
        return c.reason
    if c.backend == Backend.PRISM and c.needs_prism_binary and not have_prism_binary:
        return (
            f"model needs PrismML llama.cpp (PQ2_0/PTQ1_0 or Hadamard): {c.reason}. "
            "Install via tools/bonsai/fetch_prism.py, or use the Q2_0 pack from "
            "prism-ml/Ternary-Bonsai-2-27B-gguf-dev with upstream llama.cpp b11555+."
        )
    return None
