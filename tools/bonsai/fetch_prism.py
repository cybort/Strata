#!/usr/bin/env python3
"""Pin and download a PrismML llama.cpp release binary for Bonsai GGUF.

Default pin: prism-b10777 (docs.prismml.com Formats page).
Prefer the CUDA zip for LM Studio-style `-ngl` / `--fit` GPU offload.

Usage:
  python tools/bonsai/fetch_prism.py --variant cuda
  python tools/bonsai/fetch_prism.py --variant cpu --dry-run
  python tools/bonsai/fetch_prism.py --tag prism-b10777-e3de715 --extract
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TAG = "prism-b10777-e3de715"
PIN_FILE = "PIN.json"
API = "https://api.github.com/repos/PrismML-Eng/llama.cpp/releases/tags/{tag}"


def pick_asset(assets: list[dict], system: str, machine: str, variant: str) -> dict | None:
    sys_l = system.lower()
    mac_l = machine.lower()
    variant = variant.lower()

    def scored() -> list[tuple[int, dict]]:
        out: list[tuple[int, dict]] = []
        for a in assets:
            n = a["name"].lower()
            score = 0
            if sys_l.startswith("win"):
                if "win" not in n and "windows" not in n:
                    continue
                if "arm64" in n and "arm" not in mac_l and "aarch" not in mac_l:
                    continue
                if variant == "cuda":
                    # Prefer llama-prism-*-cuda-* (not the huge cudart-* companion zip)
                    if "cudart" in n:
                        score = 5
                    elif "cuda-13.3" in n or "cuda-13" in n:
                        score = 100
                    elif "cuda-12.4" in n or "cuda-12" in n:
                        score = 90
                    elif "cuda" in n:
                        score = 80
                    else:
                        continue
                elif variant == "vulkan":
                    if "vulkan" not in n:
                        continue
                    score = 50
                elif variant == "cpu":
                    if "cpu" not in n or "cuda" in n or "hip" in n or "vulkan" in n:
                        continue
                    score = 50
                else:
                    continue
            elif sys_l == "darwin":
                if "macos" not in n:
                    continue
                score = 50
            else:
                if "linux" not in n:
                    continue
                if variant == "cuda" and "cuda" in n:
                    score = 80 if "cudart" not in n else 5
                elif variant == "cpu" and "cpu" in n:
                    score = 50
                else:
                    continue
            out.append((score, a))
        out.sort(key=lambda t: (-t[0], t[1].get("size") or 0))
        return out

    ranked = scored()
    return ranked[0][1] if ranked else None


def extract_zip(zip_path: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)
    # Find llama-server
    for p in dest.rglob("llama-server*"):
        if p.is_file() and p.suffix.lower() in (".exe", "") and "server" in p.name.lower():
            if not p.name.endswith(".dll"):
                return p
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default=os.environ.get("STRATA_PRISM_TAG", DEFAULT_TAG))
    ap.add_argument("--dir", type=Path, default=ROOT / "third_party" / "prism-llama")
    ap.add_argument(
        "--variant",
        choices=["cuda", "cpu", "vulkan"],
        default=os.environ.get("STRATA_PRISM_VARIANT", "cuda"),
        help="cuda (default): GPU -ngl/--fit like LM Studio; cpu: no VRAM offload",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--extract", action="store_true", help="unzip into --dir/<variant>/")
    args = ap.parse_args()

    url = API.format(tag=args.tag)
    print(f"[bonsai] fetching release metadata {url}", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            rel = json.load(r)
    except Exception as e:
        print(f"[bonsai] GitHub API failed ({e}).", file=sys.stderr)
        print(
            "Manual install:\n"
            f"  1. Open https://github.com/PrismML-Eng/llama.cpp/releases/tag/{args.tag}\n"
            f"  2. Download llama-prism-*-bin-win-cuda-*.zip (not cudart-*) into {args.dir}\n"
            "  3. Extract and set STRATA_PRISM_LLAMA to llama-server.exe",
            file=sys.stderr,
        )
        return 1

    assets = rel.get("assets") or []
    asset = pick_asset(assets, platform.system(), platform.machine(), args.variant)
    if not asset:
        print(f"[bonsai] no asset matched variant={args.variant}; available: {[a['name'] for a in assets]}",
              file=sys.stderr)
        return 1

    pin = {
        "tag": args.tag,
        "variant": args.variant,
        "asset": asset["name"],
        "url": asset["browser_download_url"],
        "size": asset.get("size"),
        "system": platform.system(),
        "machine": platform.machine(),
    }
    args.dir.mkdir(parents=True, exist_ok=True)
    pin_path = args.dir / PIN_FILE
    if args.dry_run:
        print(json.dumps(pin, indent=2))
        return 0

    dest = args.dir / asset["name"]
    if dest.exists() and dest.stat().st_size == asset.get("size"):
        print(f"[bonsai] already have {dest}", flush=True)
    else:
        print(f"[bonsai] downloading {asset['browser_download_url']} -> {dest}", flush=True)
        urllib.request.urlretrieve(asset["browser_download_url"], dest)
    pin_path.write_text(json.dumps(pin, indent=2) + "\n", encoding="utf-8")
    print(f"[bonsai] wrote {pin_path}", flush=True)

    if args.extract or args.variant:
        out = args.dir / args.variant
        server = extract_zip(dest, out)
        print(f"[bonsai] extracted to {out}", flush=True)
        print(f"[bonsai] set STRATA_PRISM_LLAMA={server}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
