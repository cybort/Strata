# Bonsai dual-runtime (PrismML llama.cpp)

Strata’s Flash-Next MoE engine does not implement Prism `PQ2_0` (ggml 142) / `PTQ1_0` (143) or the
activation Hadamard. Those formats run through **PrismML’s llama.cpp** as a second backend.

## Install the Prism binary

```powershell
python tools/bonsai/fetch_prism.py
# Extract the archive under third_party/prism-llama/
# Or: set STRATA_PRISM_LLAMA to the full path of llama-server
```

Pin: `prism-b10777` (see https://docs.prismml.com/download/formats).

## Download a model card / GGUF

```powershell
hf download prism-ml/Ternary-Bonsai-2-27B-gguf README.md --local-dir K:\models\Ternary-Bonsai-2-27B-gguf
hf download prism-ml/Ternary-Bonsai-2-27B-gguf Ternary-Bonsai-2-27B-PQ2_0.gguf --local-dir K:\models\Ternary-Bonsai-2-27B-gguf
```

## Serve

`bonsai.json` example:

```json
{
  "model": "K:/models/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf",
  "model_name": "ternary-bonsai-2-27b",
  "llama_port": 8081,
  "ngl": 99,
  "ctx": 8192
}
```

```powershell
python -m serve.server --engine bonsai --config bonsai.json --port 8080
```

## Routing

`tools/bonsai/router.py` chooses:

| Path | Backend |
| --- | --- |
| Directory with `index.txt` + `dense.bin` | Strata (`--engine strata`) |
| `*.gguf` Bonsai / PQ2 / PTQ1 | Prism (`--engine bonsai`, needs Prism binary) |
| `*Q2_0*.gguf` | Prism or upstream llama.cpp `b11555+` |
