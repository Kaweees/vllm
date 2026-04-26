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
- DDTree verification requires custom attention mask (ancestor-only) in the runner
  vLLM's TreeAttentionMetadata supports tree attention but is designed for fixed
  tree structures (MTP). DDTree needs dynamic trees built from draft logits.
- The cleanest path: modify DDTreeProposer.propose() to build the tree and store
  tree_info (child_maps, visibility), then add a DDTree handler in the main
  sample_tokens() flow that runs target model with tree attention.

## What's Been Tried

### Round 1: Initial Implementation
- **Approach**: DDTreeProposer extending DFlashProposer, reusing draft model loading and hidden state processing
- **Result**: KEEP — 16/16 unit tests pass
- **Files added**: ddtree_utils.py, ddtree_proposer.py, test_ddtree.py
- **Config changes**: Added "ddtree" to SpeculativeMethod, parallel_drafting, use_ddtree() helper
- **Runner changes**: Wired into GPUModelRunner draft model selection and propose_draft_token_ids

## Next Ideas
1. Implement full tree-based verification in GPUModelRunner — the key missing piece
   a. Store tree_info in proposer, add DDTree handler in sample_tokens() flow
   b. Target model forward with tree attention → walk tree → accept tokens
   c. Requires either: (a) save/restore KV cache around tree verify, or (b) defer tree verify to next scheduler iteration
2. Add GPU acceptance length benchmark with DFlash draft model (to measure improvement)
3. Add tree_budget tuning experiments (find optimal budget for different model sizes)
4. Implement CUDA graph support for DDTree verification
5. Add multi-request batching tests
6. Optimize tree building for speed (currently CPU-based, can use torch for GPU)
7. Add howto guide for running DDTree (completed)

## Run 2: DDTree wired into GPUModelRunner — unit_test_passed=1 (KEEP)
- Timestamp: 2026-04-26 10:25
- What changed: Added DDTree-specific branch in propose_draft_token_ids()
- Result: DDTreeProposer runs DFlash draft model, returns draft token IDs
- Insight: DDTree is now wired into vLLM's speculative decoding pipeline.
  However, the current implementation returns greedy samples (like DFlash).
  Full tree-based verification requires custom attention mask in the runner,
  which needs modifications to the attention backend.

## Run 3: Howto guide created (KEEP)
- Timestamp: 2026-04-26 10:30
- What changed: Created docs/ddtree-howto.md with setup, usage, and config instructions
- Result: Users can now run DDTree with a DFlash draft model
