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

## Run DDTree with a model (requires GPU + DFlash draft model)

You need a target model and a DFlash draft model:

```bash
.venv/bin/python examples/offline_inference/spec_decode.py \
  --model /models/Qwen3.5-27B \
  --draft-model /models/Qwen3.5-27B-DFlash \
  --speculative-method ddtree \
  --num-spec-tokens 3 \
  --tp 1
```

Or via the Python API:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="/models/Qwen3.5-27B",
    speculative_model="/models/Qwen3.5-27B-DFlash",
    speculative_method="ddtree",
    num_speculative_tokens=3,
    tensor_parallel_size=1,
)

prompts = ["Hello, my name is"]
sampling_params = SamplingParams(temperature=0.0, max_tokens=64)
outputs = llm.generate(prompts, sampling_params)
```

## Config fields

| Field | Value | Description |
|-------|-------|-------------|
| `speculative_method` | `"ddtree"` | Enable DDTree |
| `speculative_model` | `/models/Qwen3.5-27B-DFlash` | DFlash draft model |
| `num_speculative_tokens` | `3` (or higher) | Tree budget (nodes = spec_tokens) |

## Limitations

- **Tree verification is not yet active** — DDTree currently returns greedy samples (same as DFlash). The full tree-based verification with ancestor-only attention mask requires additional integration work in the model runner's `sample_tokens()` flow.
- **Works with**: Qwen3.5-27B target model with Qwen3.5-27B-DFlash draft model
- **Does not require**: New dependencies, new model types, or changes to the target model
