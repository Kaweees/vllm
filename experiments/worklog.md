# DDTree Speculative Decoding — Worklog

## Session Started
- **Date**: 2026-04-26
- **Goal**: Add DDTree support for speculative decoding in vLLM
- **Reference**: https://arxiv.org/abs/2604.12989, https://github.com/liranringel/ddtree
- **Baseline**: DFlashProposer (existing) — DDTree builds on top of DFlash

## Run 1: DDTreeProposer implementation — unit_test_passed=1 (KEEP)
- Timestamp: 2026-04-26 10:15
- What changed: Full DDTree implementation including tree building, attention mask compilation, tree walking, KV cache compaction, DDTreeProposer class, config wiring, GPUModelRunner integration
- Result: 16/16 unit tests passed
- Insight: DDTree architecture cleanly extends DFlashProposer. The tree building uses best-first heap search on log-probs, and the attention mask is built as an ancestor-only visibility matrix.
- Next: Add GPU acceptance length benchmark

## Key Architectural Insights
- DDTree = DFlash draft model + best-first tree construction + tree attention verification
- vLLM already has `TreeAttentionMetadata` support (used by MTP)
- DFlashProposer extends `SpecDecodeBaseProposer` with cross-attention and parallel drafting
- DDTreeProposer extends DFlashProposer to reuse draft model infrastructure
- Key differentiator: DFlash outputs one trajectory; DDTree outputs a tree of candidate trajectories
- Verification uses ancestor-only attention mask for tree structure
- Tree building: best-first heap on accumulated log-probabilities, limited by node budget
- KV cache compaction must preserve only accepted path tokens

## What's Been Tried

### Round 1: Initial Implementation
- **Approach**: DDTreeProposer extending DFlashProposer, reusing draft model loading and hidden state processing
- **Result**: KEEP — 16/16 unit tests pass
- **Files added**: ddtree_utils.py, ddtree_proposer.py, test_ddtree.py
- **Config changes**: Added "ddtree" to SpeculativeMethod, parallel_drafting, use_ddtree() helper
- **Runner changes**: Wired into GPUModelRunner draft model selection and propose_draft_token_ids

## Next Ideas
1. Implement GPU acceptance length benchmark (full integration test with DFlash draft model)
2. Add tree_budget tuning experiments
3. Implement CUDA graph support for DDTree verification
4. Add multi-request batching tests
5. Optimize tree building for speed (currently CPU-based)
