# DDTree Speculative Decoding — Worklog

## Session Started
- **Date**: 2026-04-26
- **Goal**: Add DDTree support for speculative decoding in vLLM
- **Reference**: https://arxiv.org/abs/2604.12989, https://github.com/liranringel/ddtree
- **Baseline**: DFlashProposer (existing) — DDTree builds on top of DFlash

## Key Architectural Insights
- DDTree = DFlash draft model + best-first tree construction + tree attention verification
- vLLM already has `TreeAttentionMetadata` support (used by MTP)
- DFlashProposer extends `SpecDecodeBaseProposer` with cross-attention and parallel drafting
- DDTreeProposer should extend DFlashProposer to reuse draft model infrastructure
- Key differentiator: DFlash outputs one trajectory; DDTree outputs a tree of candidate trajectories
- Verification uses ancestor-only attention mask for tree structure

## Next Ideas
1. Start with DDTreeProposer implementing just the tree building logic (no GPU benchmark yet)
2. Add "ddtree" to SpeculativeMethod enum
3. Wire into GPUModelRunner
4. Implement propose() with tree verification
5. Add unit tests for tree building, walking, and attention mask generation
6. Tune tree_budget parameter
