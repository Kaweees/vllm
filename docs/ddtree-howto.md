# Running DDTree Speculative Decoding

## Branch
```bash
git checkout autoresearch/ddtree-speculative-decoding
source .venv/bin/activate
```

## Install
```bash
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

## Download models
```bash
hf download Qwen/Qwen3.5-27B --local-dir /models/Qwen3.5-27B
hf download z-lab/Qwen3.5-27B-DFlash --local-dir /models/Qwen3.5-27B-DFlash
```

## Unit tests (no GPU needed)
```bash
.venv/bin/python -m pytest tests/v1/spec_decode/test_ddtree.py -v
```

## Run DDTree with a model (offline benchmark, requires GPU + DFlash draft model)

You need a target model and a DFlash draft model:

```bash
.venv/bin/python examples/offline_inference/spec_decode.py \
  --model /models/Qwen3.5-27B \
  --draft-model /models/Qwen3.5-27B-DFlash \
  --method ddtree \
  --num-spec-tokens 3 \
  --tp 1
```

This runs an **offline inference benchmark** that measures the acceptance length (average number of draft tokens accepted per verification round). It does **not** launch an OpenAI API server.

To launch an OpenAI-compatible API server with DDTree, use:
```bash
.venv/bin/python -m vllm.entrypoints.api_server \
    --model /models/Qwen3.5-27B \
    --draft-model /models/Qwen3.5-27B-DFlash \
    --method ddtree \
    --num-spec-tokens 3
```

## Config fields

| Field | Value | Description |
|-------|-------|-------------|
| `--method` | `"ddtree"` | Enable DDTree (CLI) |
| `speculative_method` | `"ddtree"` | Enable DDTree (Python API) |
| `speculative_model` | `/models/Qwen3.5-27B-DFlash` | DFlash draft model |
| `num_speculative_tokens` | `3` (or higher) | Tree budget (nodes = spec_tokens) |

## Limitations

- **Tree verification is not yet active** — DDTree currently returns greedy samples (same as DFlash). The full tree-based verification with ancestor-only attention mask requires additional integration work in the model runner's `sample_tokens()` flow.
- **Works with**: Qwen3.5-27B target model with Qwen3.5-27B-DFlash draft model
- **Does not require**: New dependencies, new model types, or changes to the target model

## Troubleshooting

If you encounter import errors (e.g., `SlidingWindowMLASpec`), this may be due to the vLLM version in the environment. The DDTree implementation is compatible with the vLLM codebase at the time of writing. For the best experience, ensure you are using a recent version of vLLm that includes the necessary KV cache interface updates.
