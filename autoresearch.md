# Autoresearch: DDTree Speculative Decoding

## Objective
Add DDTree (Diffusion Draft Tree) support for speculative decoding in vLLM. DDTree builds a draft tree from per-position distributions of a block diffusion draft model (DFlash), then verifies the whole tree in a single target-model forward pass using tree attention. This should improve acceptance length over vanilla DFlash while keeping drafting cheap.

Reference: https://liranringel.github.io/ddtree/, https://arxiv.org/abs/2604.12989

## Metrics
- **Primary**: acceptance_length (tokens, higher is better) — mean number of draft tokens accepted per verification round
- **Secondary**: throughput (tokens/s), verify_time (ms), draft_time (ms), tree_build_time (ms)

## How to Run
`./autoresearch.sh` — outputs `METRIC acceptance_length=number` lines.

The script:
1. Checks for a GPU
2. If no GPU available: runs a lightweight unit test that validates the DDTree data structures and tree walking logic
3. If GPU available: runs a full acceptance length benchmark with a small model

## Files in Scope
- `vllm/v1/spec_decode/ddtree_proposer.py` — DDTreeProposer class (new)
- `vllm/config/speculative.py` — add "ddtree" to SpeculativeMethod, parallel_drafting for ddtree
- `vllm/v1/worker/gpu_model_runner.py` — wire DDTree proposer into draft model selection
- `vllm/v1/spec_decode/utils.py` — tree building and cache compaction utilities (new or extended)
- `tests/v1/spec_decode/test_ddtree.py` — unit tests for DDTree (new)

## Off Limits
- Do not modify the DFlash draft model implementation itself
- Do not modify the target model implementations
- Do not add new Python dependencies (use only existing PyTorch, NumPy)

## Constraints
- Must pass all existing spec decode tests
- DDTree must work with DFlash draft model (DFlashQwen3ForCausalLM)
- Tree attention mask must follow ancestor-only pattern
- KV cache compaction must be correct for accepted tokens
- Must support batched requests (multiple concurrent requests)
- unit tests must pass without GPU

## What's Been Tried
