# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.spec_decode.ddtree_utils import (
    DDTreeInfo,
    build_ddtree_tree,
    compile_ddtree_tree,
    follow_verified_tree,
    compact_kv_cache,
    _compact_appended_window,
)


class TestBuildDDTreeTree:
    """Test DDTree building from draft logits."""

    def test_empty_budget(self):
        """Empty budget returns single root node."""
        draft_logits = torch.randn(3, 100)
        node_ids, depths, parents, child_maps, visibility = build_ddtree_tree(
            draft_logits, budget=0
        )
        assert node_ids.numel() == 0
        assert depths.numel() == 0
        assert parents == [-1]
        assert child_maps == [{}]
        assert visibility.shape == (1, 1)
        assert visibility[0, 0] == True

    def test_empty_logits(self):
        """Empty logits returns single root node."""
        draft_logits = torch.empty(0, 100)
        node_ids, depths, parents, child_maps, visibility = build_ddtree_tree(
            draft_logits, budget=5
        )
        assert node_ids.numel() == 0
        assert visibility.shape == (1, 1)
        assert visibility[0, 0] == True

    def test_single_node(self):
        """Budget=1 should create one child of root."""
        torch.manual_seed(42)
        draft_logits = torch.randn(3, 100)
        node_ids, depths, parents, child_maps, visibility = build_ddtree_tree(
            draft_logits, budget=1
        )
        assert node_ids.numel() == 1
        assert depths[0] == 1  # child of root is at depth 1
        assert parents[0] == -1  # root
        assert parents[1] == 0  # node 1's parent is root (0)
        assert len(child_maps[0]) == 1  # root has one child
        assert visibility.shape == (2, 2)
        # Root visible to itself
        assert visibility[0, 0] == True
        # Node 1 visible to root and itself
        assert visibility[1, 0] == True
        assert visibility[1, 1] == True

    def test_tree_structure(self):
        """Budget > 1 should create proper tree structure."""
        torch.manual_seed(42)
        draft_logits = torch.randn(3, 50)
        node_ids, depths, parents, child_maps, visibility = build_ddtree_tree(
            draft_logits, budget=10
        )
        assert node_ids.numel() <= 10
        assert node_ids.numel() == depths.numel()
        assert len(parents) == 1 + node_ids.numel()
        assert len(child_maps) == 1 + node_ids.numel()

        # Every node (except root) has a valid parent
        for i in range(1, len(parents)):
            assert 0 <= parents[i] < i, f"Node {i} has invalid parent {parents[i]}"

        # Visibility matrix should be upper triangular with diagonal
        assert visibility.shape == (len(parents), len(parents))
        for i in range(len(parents)):
            assert visibility[i, i] == True, f"Node {i} not visible to itself"
            for j in range(i + 1, len(parents)):
                assert visibility[i, j] == False, (
                    f"Node {i} should not see node {j} (lower triangular)"
                )

    def test_budget_limit(self):
        """Tree should never exceed budget nodes."""
        torch.manual_seed(42)
        draft_logits = torch.randn(3, 100)
        for budget in [1, 5, 10, 20, 50, 100]:
            node_ids, _, _, _, _ = build_ddtree_tree(draft_logits, budget=budget)
            assert node_ids.numel() <= budget, (
                f"Budget {budget}: got {node_ids.numel()} nodes"
            )

    def test_best_first_selection(self):
        """Higher log-prob paths should be selected first."""
        # Create logits where position 0 has very high prob for token 10,
        # position 1 has very high prob for token 20, etc.
        draft_logits = torch.zeros(3, 100)
        draft_logits[0, 10] = 10.0  # position 0, token 10
        draft_logits[1, 20] = 10.0  # position 1, token 20
        draft_logits[2, 30] = 10.0  # position 2, token 30

        node_ids, depths, parents, child_maps, _ = build_ddtree_tree(
            draft_logits, budget=3
        )
        assert node_ids.numel() == 3
        # The highest-prob path should be selected: 10 -> 20 -> 30
        # Node at depth 1 should be token 10
        depth_1_mask = depths == 1
        if depth_1_mask.any():
            assert node_ids[depth_1_mask].item() == 10
        # Node at depth 2 should be token 20
        depth_2_mask = depths == 2
        if depth_2_mask.any():
            assert node_ids[depth_2_mask].item() == 20


class TestCompileDDTreeTree:
    """Test DDTree tensor compilation for target model verification."""

    def test_compile_basic(self):
        """Basic compilation with small tree."""
        torch.manual_seed(42)
        node_ids = torch.tensor([5, 10, 15])
        depths = torch.tensor([1, 2, 3])
        parents = [-1, 0, 0, 1]
        child_maps = [{5: 1}, {10: 2}, {}, {}]

        # Create draft logits for visibility
        draft_logits = torch.randn(3, 50)
        _, _, _, _, visibility = build_ddtree_tree(draft_logits, budget=3)

        device = torch.device("cpu")
        dtype = torch.float16

        verify_input_ids, verify_position_ids, attention_mask, new_start, new_length = (
            compile_ddtree_tree(
                root_token_id=0,
                start=10,
                node_token_ids=node_ids,
                node_depths=depths,
                visibility=visibility,
                past_length=10,
                dtype=dtype,
                device=device,
                verify_input_ids_buffer=torch.empty((1, 10), dtype=torch.long, device=device),
                verify_position_ids_buffer=torch.empty((1, 10), dtype=torch.long, device=device),
                attention_mask_buffer=torch.zeros(
                    (1, 1, 10, 100), dtype=dtype, device=device
                ),
                tree_visibility_buffer=torch.empty((10, 10), dtype=torch.bool, device=device),
                previous_tree_start=0,
                previous_tree_length=0,
            )
        )

        assert verify_input_ids.shape == (1, 4)  # root + 3 nodes
        assert verify_position_ids.shape == (1, 4)
        assert attention_mask.shape[2] == 4  # query length = tree length
        assert new_length == 4
        assert new_start == 10

    def test_attention_mask_structure(self):
        """Attention mask should encode ancestor-only visibility."""
        torch.manual_seed(42)
        node_ids = torch.tensor([5, 10, 15])
        depths = torch.tensor([1, 2, 3])
        _, _, _, _, visibility = build_ddtree_tree(torch.randn(3, 50), budget=3)

        attention_mask_buffer = torch.zeros(
            (1, 1, 10, 100), dtype=torch.float16, device=torch.device("cpu")
        )

        verify_input_ids, verify_position_ids, attention_mask, _, _ = (
            compile_ddtree_tree(
                root_token_id=0,
                start=10,
                node_token_ids=node_ids,
                node_depths=depths,
                visibility=visibility,
                past_length=10,
                dtype=torch.float16,
                device=torch.device("cpu"),
                verify_input_ids_buffer=torch.empty((1, 10), dtype=torch.long, device="cpu"),
                verify_position_ids_buffer=torch.empty((1, 10), dtype=torch.long, device="cpu"),
                attention_mask_buffer=attention_mask_buffer,
                tree_visibility_buffer=torch.empty((10, 10), dtype=torch.bool, device="cpu"),
                previous_tree_start=0,
                previous_tree_length=0,
            )
        )

        # attention_mask[..., :4, 10:14] should match visibility
        tree_block = attention_mask[0, 0, :4, 10:14]
        log_min = torch.finfo(torch.float16).min
        for i in range(4):
            for j in range(4):
                if visibility[i, j]:
                    assert tree_block[i, j] == 0.0
                else:
                    assert tree_block[i, j] == log_min


class TestFollowVerifiedTree:
    """Test DDTree walking for acceptance determination."""

    def test_full_match(self):
        """All tokens match the tree."""
        child_maps = [
            {5: 1},       # root -> node 1 (token 5)
            {10: 2},      # node 1 -> node 2 (token 10)
            {15: 3},      # node 2 -> node 3 (token 15)
            {},           # node 3 has no children
        ]
        # Posterior tokens: [5, 10, 15, ...]
        posterior = torch.tensor([[5, 10, 15, 99]])

        accepted, next_token = follow_verified_tree(child_maps, posterior)
        assert accepted == [0, 1, 2, 3]
        assert next_token == 99

    def test_partial_match(self):
        """First token matches, second doesn't."""
        child_maps = [
            {5: 1},
            {10: 2},
            {},
        ]
        # Posterior: [5, 99, ...] - matches token 5 but not 10
        posterior = torch.tensor([[5, 99, 7, 3]])

        accepted, next_token = follow_verified_tree(child_maps, posterior)
        assert accepted == [0, 1]
        assert next_token == 99

    def test_no_match(self):
        """No tokens match the tree."""
        child_maps = [
            {5: 1},
            {},
        ]
        # Posterior: [99, ...] - no match
        posterior = torch.tensor([[99, 7, 3]])

        accepted, next_token = follow_verified_tree(child_maps, posterior)
        assert accepted == [0]
        assert next_token == 99

    def test_empty_tree(self):
        """Empty child maps (root only)."""
        child_maps = [{}]
        posterior = torch.tensor([[5, 7, 3]])

        accepted, next_token = follow_verified_tree(child_maps, posterior)
        assert accepted == [0]
        assert next_token == 5


class TestCompactKVCache:
    """Test KV cache compaction."""

    def test_empty_keep_indices(self):
        """Empty keep_indices should crop to past_length."""
        # Create a mock DynamicCache-like object
        class MockCache:
            def __init__(self):
                self.cropped = None

            def crop(self, length):
                self.cropped = length

        cache = MockCache()
        compact_kv_cache(cache, past_length=10, keep_current_indices=[])
        assert cache.cropped == 10

    def test_full_keep_indices(self):
        """All indices kept - should crop to past + count."""
        class MockCache:
            def __init__(self):
                self.cropped = None

            def crop(self, length):
                self.cropped = length

        cache = MockCache()
        compact_kv_cache(cache, past_length=10, keep_current_indices=[0, 1, 2])
        assert cache.cropped == 13


class TestDDTreeInfo:
    """Test DDTreeInfo helper properties."""

    def test_info_properties_empty(self):
        """Empty tree info."""
        info = DDTreeInfo(
            node_token_ids=torch.empty(0, dtype=torch.long),
            node_depths=torch.empty(0, dtype=torch.long),
            parents=[-1],
            child_maps=[{}],
            visibility=torch.tensor([[True]]),
        )
        assert info.num_nodes == 0
        assert info.tree_length == 1
        assert info.max_depth == 0

    def test_info_properties_nonempty(self):
        """Non-empty tree info."""
        info = DDTreeInfo(
            node_token_ids=torch.tensor([5, 10, 15, 20]),
            node_depths=torch.tensor([1, 2, 2, 3]),
            parents=[-1, 0, 0, 1],
            child_maps=[{5: 1, 10: 2}, {15: 3}, {}, {}, {}],
            visibility=torch.ones(5, 5, dtype=torch.bool),
        )
        assert info.num_nodes == 4
        assert info.tree_length == 5
        assert info.max_depth == 3
