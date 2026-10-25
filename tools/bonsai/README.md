# Bonsai dual-runtime (PrismML llama.cpp)

Strata’s Flash-Next MoE engine does not implement Prism `PQ2_0` / `PTQ1_0` or the activation Hadamard.
Those formats run through **PrismML’s llama.cpp** (`--engine bonsai`).

## Install Prism (prefer CUDA for GPU offload)

```powershell
# CUDA build: enables -ngl / --fit like LM Studio (recommended on NVIDIA)
python tools/bonsai/fetch_prism.py --variant cuda --extract

# CPU-only (no VRAM offload; slow on 27B)
python tools/bonsai/fetch_prism.py --variant cpu --extract
```

Or set `STRATA_PRISM_LLAMA` to `llama-server.exe`.

Pin: `prism-b10777` — see https://docs.prismml.com/download/formats.

## Partial GPU offload (LM Studio style)

Same mechanism as LM Studio / llama.cpp:

| Config | Meaning |
| --- | --- |
| `"ngl": 99` | Put as many layers as possible on GPU (default) |
| `"ngl": 40` | Exactly 40 layers on GPU, rest on CPU/RAM |
| `"ngl": 0` | CPU only |
| `"fit": true` | Auto-shrink unset knobs to free VRAM (`--fit on`) |

Requires a **CUDA** Prism zip. A CPU zip with `ngl>0` is forced to `-ngl 0` with a warning.

## Timeouts / thinking (avoids 600s hangs)

Bonsai 2 defaults to long “thinking” traces. This backend defaults to:

- `"reasoning": "off"`, `"reasoning_budget": 0`
- `"chat_timeout_s": 3600`, `"server_timeout_s": 3600` (override with `STRATA_BONSAI_CHAT_TIMEOUT`)

For full thinking mode set `"reasoning": "on"` and raise timeouts / use CUDA `-ngl`.

## Serve

```powershell
python -m serve.server --engine bonsai --config tools/bonsai/example.bonsai.json --port 8080
```

## Routing

| Path | Backend |
| --- | --- |
| `index.txt` + `dense.bin` pack | Strata |
| Bonsai `PQ2_0` / `PTQ1_0` GGUF | Prism (needs Prism binary) |
| Bonsai `Q2_0` GGUF | Prism or upstream llama.cpp `b11555+` |
