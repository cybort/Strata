#!/usr/bin/env python3
"""Pin and download a PrismML llama.cpp release binary for Bonsai GGUF.

Default pin: prism-b10777 (docs.prismml.com Formats page, Oct 2026).
Does not push; writes under third_party/prism-llama/ by default.

Usage:
  python tools/bonsai/fetch_prism.py
  python tools/bonsai/fetch_prism.py --dry-run
  python tools/bonsai/fetch_prism.py --tag prism-b10777-e3de715
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TAG = "prism-b10777-e3de715"
PIN_FILE = "PIN.json"
API = "https://api.github.com/repos/PrismML-Eng/llama.cpp/releases/tags/{tag}"


def pick_asset(assets: list[dict], system: str, machine: str) -> dict | None:
    names = [a["name"] for a in assets]
    sys_l = system.lower()
    mac_l = machine.lower()

    def find(*needles: str) -> dict | None:
        for a in assets:
            n = a["name"].lower()
            if all(x.lower() in n for x in needles):
                return a
        return None

    if sys_l.startswith("win"):
        return find("win", "cuda") or find("windows", "cuda") or find("win", "cpu") or find("windows")
    if sys_l == "darwin":
        return find("macos", "arm") or find("macos", "apple") or find("macos")
    # linux
    if "aarch" in mac_l or "arm" in mac_l:
        return find("linux", "arm") or find("linux-arm")
    return find("linux", "cuda") or find("ubuntu", "cuda") or find("linux", "cpu") or find("linux")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default=os.environ.get("STRATA_PRISM_TAG", DEFAULT_TAG))
    ap.add_argument("--dir", type=Path, default=ROOT / "third_party" / "prism-llama")
    ap.add_argument("--dry-run", action="store_true")
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
            f"  2. Download the archive for your OS into {args.dir}\n"
            "  3. Extract so llama-server (or llama-cli) is on PATH or under that dir\n"
            "  4. Write PIN.json with {\"tag\": \"%s\", \"asset\": \"...\"}" % args.tag,
            file=sys.stderr,
        )
        return 1

    assets = rel.get("assets") or []
    asset = pick_asset(assets, platform.system(), platform.machine())
    if not asset:
        print(f"[bonsai] no asset matched; available: {[a['name'] for a in assets]}", file=sys.stderr)
        return 1

    pin = {
        "tag": args.tag,
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
    print(f"[bonsai] downloading {asset['browser_download_url']} -> {dest}", flush=True)
    urllib.request.urlretrieve(asset["browser_download_url"], dest)
    pin_path.write_text(json.dumps(pin, indent=2) + "\n", encoding="utf-8")
    print(f"[bonsai] wrote {pin_path}", flush=True)
    print(f"[bonsai] extract the archive under {args.dir} and set STRATA_PRISM_LLAMA to llama-server", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
