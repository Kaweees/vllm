# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DDTree (Diffusion Draft Tree) tree building and attention mask utilities.

DDTree builds a draft tree from per-position distributions of a block diffusion
draft model (DFlash), then verifies the whole tree in a single target-model forward
pass using ancestor-only attention mask.

Reference: https://arxiv.org/abs/2604.12989
"""

import heapq
from typing import Any

import torch
from typing_extensions import override

from vllm.logger import init_logger

logger = init_logger(__name__)


class DDTreeInfo:
    """Holds the tree structure built from DFlash draft logits.

    This class encapsulates all the data needed to verify a DDTree draft:
    - node_token_ids: token IDs for each tree node (0-indexed, root at 0)
    - node_depths: depth of each node (root=0, children of root=1, etc.)
    - parents: parent index for each node (-1 for root)
    - child_maps: mapping from parent index -> {token_id -> child index}
    - visibility: ancestor-only attention mask (bool matrix)
    """

    def __init__(
        self,
        node_token_ids: torch.Tensor,
        node_depths: torch.Tensor,
        parents: list[int],
        child_maps: list[dict[int, int]],
        visibility: torch.Tensor,
    ):
        self.node_token_ids = node_token_ids
        self.node_depths = node_depths
        self.parents = parents
        self.child_maps = child_maps
        self.visibility = visibility

    @property
    def num_nodes(self) -> int:
        """Total number of nodes in the tree (excluding root)."""
        return self.node_token_ids.numel()

    @property
    def tree_length(self) -> int:
        """Total tree length including root."""
        return 1 + self.num_nodes

    @property
    def max_depth(self) -> int:
        """Maximum depth of the tree."""
        if self.node_depths.numel() == 0:
            return 0
        return int(self.node_depths.max().item())


def build_ddtree_tree(
    draft_logits: torch.Tensor,
    budget: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[dict[int, int]], torch.Tensor]:
    """Build a DDTree from draft model per-position distributions.

    Uses a best-first heap algorithm to select the most likely tree nodes
    under a fixed node budget.

    Args:
        draft_logits: [block_size, vocab_size] logits for each position in the block.
            Position 0 corresponds to depth 1 (child of root).
        budget: maximum number of nodes to include in the tree (excluding root).

    Returns:
        Tuple of (node_token_ids, node_depths, parents, child_maps, visibility)
        where:
        - node_token_ids: [num_nodes] token IDs for tree nodes (excluding root)
        - node_depths: [num_nodes] depths of each node
        - parents: [tree_length] parent index for each node (-1 for root)
        - child_maps: [tree_length] list of {token_id -> child_index} dicts
        - visibility: [tree_length, tree_length] ancestor-only attention mask
    """
    if budget <= 0 or draft_logits.shape[0] == 0:
        visibility = torch.zeros((1, 1), dtype=torch.bool)
        visibility[0, 0] = True
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            [-1],
            [dict()],
            visibility,
        )

    topk = min(budget, draft_logits.shape[-1])
    depth_limit = int(draft_logits.shape[0])

    # Compute log probabilities (for best-first search priority)
    logits = draft_logits.float()
    top_logits, top_token_ids = torch.topk(logits, k=topk, dim=-1)
    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
    top_log_probs = (top_logits - log_z)  # log probabilities

    # Convert to CPU for heap operations
    top_log_probs_cpu = top_log_probs.to(device="cpu", dtype=torch.float32)
    top_token_ids_cpu = top_token_ids.to(device="cpu", dtype=torch.long)

    top_log_probs_np = top_log_probs_cpu.numpy()
    top_token_ids_np = top_token_ids_cpu.numpy()

    # Best-first heap search
    # Heap entries: (-log_w, ranks_tuple, parent_index, depth, rank, log_w)
    # where log_w is the accumulated log probability of the path
    first_logw = float(top_log_probs_np[0, 0])
    heap: list[tuple[float, tuple[int, ...], int, int, int, float]] = [
        (-first_logw, (0,), 0, 1, 0, first_logw)
    ]

    node_token_ids_np = torch.empty(budget, dtype=torch.long, device="cpu")
    node_depths_np = torch.empty(budget, dtype=torch.long, device="cpu")
    parents_np = torch.empty(budget + 1, dtype=torch.int32, device="cpu")
    parents_np[0] = -1
    child_maps: list[dict[int, int]] = [dict()]
    node_count = 0

    while heap and node_count < budget:
        _, ranks, parent_index, depth, rank, logw = heapq.heappop(heap)

        token_id = int(top_token_ids_np[depth - 1, rank])
        current_index = node_count + 1  # +1 because root is index 0
        node_token_ids_np[node_count] = token_id
        node_depths_np[node_count] = depth
        parents_np[current_index] = parent_index
        child_maps.append(dict())
        child_maps[parent_index][token_id] = current_index
        node_count += 1

        # Push sibling (same depth, next rank)
        if rank + 1 < topk:
            sibling_ranks = ranks[:-1] + (rank + 1,)
            sibling_logw = (
                logw
                - float(top_log_probs_np[depth - 1, rank])
                + float(top_log_probs_np[depth - 1, rank + 1])
            )
            heapq.heappush(
                heap, (-sibling_logw, sibling_ranks, parent_index, depth, rank + 1, sibling_logw)
            )

        # Push child (deeper depth, first rank)
        if depth < depth_limit:
            child_ranks = ranks + (0,)
            child_logw = logw + float(top_log_probs_np[depth, 0])
            heapq.heappush(
                heap, (-child_logw, child_ranks, current_index, depth + 1, 0, child_logw)
            )

    # Build ancestor-only visibility matrix
    current_length = 1 + node_count
    visibility_np = torch.zeros(
        (current_length, current_length), dtype=torch.bool, device="cpu"
    )
    visibility_np[0, 0] = True
    for index in range(1, current_length):
        parent_index = int(parents_np[index].item())
        # Node is visible to all its ancestors (same rows as parent) + itself
        visibility_np[index, :index] = visibility_np[parent_index, :index]
        visibility_np[index, index] = True

    node_token_ids = node_token_ids_np[:node_count]
    node_depths = node_depths_np[:node_count]
    parents = parents_np[:current_length].tolist()

    return node_token_ids, node_depths, parents, child_maps, visibility_np


def compile_ddtree_tree(
    root_token_id: int,
    start: int,
    node_token_ids: torch.Tensor,
    node_depths: torch.Tensor,
    visibility: torch.Tensor,
    past_length: int,
    dtype: torch.dtype,
    device: torch.device,
    verify_input_ids_buffer: torch.Tensor,
    verify_position_ids_buffer: torch.Tensor,
    attention_mask_buffer: torch.Tensor,
    tree_visibility_buffer: torch.Tensor,
    previous_tree_start: int,
    previous_tree_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Prepare DDTree tensors for target model verification.

    Sets up the input IDs, positions, and attention mask for the target model
    to verify the entire DDTree in one forward pass.

    Args:
        root_token_id: the root token (previous bonus token)
        start: current generation position
        node_token_ids: tree node token IDs
        node_depths: tree node depths
        visibility: ancestor-only attention mask
        past_length: number of tokens already in the KV cache
        dtype: target model dtype
        device: target device
        verify_input_ids_buffer: pre-allocated buffer for input IDs
        verify_position_ids_buffer: pre-allocated buffer for position IDs
        attention_mask_buffer: pre-allocated attention mask buffer
        tree_visibility_buffer: pre-allocated visibility buffer
        previous_tree_start: start position of previous tree in attention mask
        previous_tree_length: length of previous tree (0 for first tree)

    Returns:
        (verify_input_ids, verify_position_ids, attention_mask, new_previous_tree_start, new_previous_tree_length)
    """
    current_length = 1 + int(node_token_ids.numel())

    # Clear attention mask for previous tree
    if previous_tree_length > 0:
        attention_mask_buffer[0, 0, :previous_tree_length, previous_tree_start : previous_tree_start + previous_tree_length] = 0

    # Fill input IDs
    verify_input_ids = verify_input_ids_buffer[:, :current_length]
    verify_input_ids[0, 0] = root_token_id
    if current_length > 1:
        verify_input_ids[0, 1:current_length].copy_(node_token_ids, non_blocking=False)

    # Fill position IDs
    verify_position_ids = verify_position_ids_buffer[:, :current_length]
    verify_position_ids[0, 0] = start
    if current_length > 1:
        verify_position_ids[0, 1:current_length].copy_(node_depths, non_blocking=False)
        verify_position_ids[0, 1:current_length].add_(start)

    # Fill visibility mask
    visibility_buf = tree_visibility_buffer[:current_length, :current_length]
    visibility_buf.copy_(visibility, non_blocking=False)

    # Build attention mask block: ancestor-only mask with log_min for non-ancestors
    tree_block = attention_mask_buffer[0, 0, :current_length, past_length : past_length + current_length]
    tree_block.fill_(torch.finfo(dtype).min)
    tree_block.masked_fill_(visibility_buf, 0)

    attention_mask = attention_mask_buffer[:, :, :current_length, : past_length + current_length]

    return (
        verify_input_ids,
        verify_position_ids,
        attention_mask,
        past_length,
        current_length,
    )


def follow_verified_tree(
    child_maps: list[dict[int, int]],
    posterior: torch.Tensor,
) -> tuple[list[int], int]:
    """Walk the DDTree following the target model's sampled tokens.

    Starting from the root, walks down the tree as long as the target model's
    sampled token matches a child in the tree. Returns the accepted indices
    and the first unmatched token.

    Args:
        child_maps: [tree_length] list of {token_id -> child_index} dicts
            built during tree construction
        posterior: [1, tree_length] sampled tokens from the target model

    Returns:
        (accepted_indices, next_token) where:
        - accepted_indices: list of tree node indices that were accepted
        - next_token: the first token that did not match (or the last matched token)
    """
    posterior_tokens = posterior[0].tolist()
    accepted_indices = [0]  # root is always accepted
    current_index = 0
    next_token = int(posterior_tokens[current_index])

    while next_token in child_maps[current_index]:
        current_index = child_maps[current_index][next_token]
        accepted_indices.append(current_index)
        next_token = int(posterior_tokens[current_index])

    return accepted_indices, next_token


def _compact_appended_window(
    cache_tensor: torch.Tensor, past_length: int, keep_current_indices: list[int]
) -> None:
    """Compact the appended window of a KV cache tensor.

    Moves the kept tokens from the appended window to the front,
    discarding unkept tokens. This is called after accepting a DDTree path
    to remove unaccepted tokens from the KV cache.

    Args:
        cache_tensor: KV cache tensor [num_layers, num_heads, seq_len, head_size]
            or similar layout where seq_len is second-to-last dimension.
        past_length: number of tokens before the current append window
        keep_current_indices: indices within the appended window to keep
    """
    seq_dim = cache_tensor.dim() - 2
    current_length = cache_tensor.size(seq_dim) - past_length
    if current_length <= 0:
        return

    keep_count = len(keep_current_indices)
    if keep_count == 0 or keep_count == current_length:
        return

    keep_tensor = torch.tensor(
        keep_current_indices, dtype=torch.long, device=cache_tensor.device
    )
    kept_tail = cache_tensor.narrow(seq_dim, past_length, current_length).index_select(
        seq_dim, keep_tensor
    )
    cache_tensor.narrow(seq_dim, past_length, keep_count).copy_(kept_tail)


def compact_kv_cache(
    past_key_values: Any,
    past_length: int,
    keep_current_indices: list[int],
) -> None:
    """Compact the KV cache after accepting a DDTree path.

    Removes unaccepted tokens from the KV cache tail, keeping only the
    accepted path tokens. This is essential for maintaining cache efficiency.

    Args:
        past_key_values: DynamicCache or similar KV cache object
        past_length: number of tokens before the current tree
        keep_current_indices: indices of tokens to keep from the current tree
    """
    if len(keep_current_indices) == 0:
        past_key_values.crop(past_length)
        return

    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        for layer_idx in range(len(past_key_values.key_cache)):
            key_cache = past_key_values.key_cache[layer_idx]
            value_cache = past_key_values.value_cache[layer_idx]
            _compact_appended_window(key_cache, past_length, keep_current_indices)
            _compact_appended_window(value_cache, past_length, keep_current_indices)
        past_key_values.crop(past_length + len(keep_current_indices))
        return

    if hasattr(past_key_values, "layers"):
        for layer in past_key_values.layers:
            if not hasattr(layer, "keys") or layer.keys is None or layer.keys.numel() == 0:
                continue
            _compact_appended_window(layer.keys, past_length, keep_current_indices)
            _compact_appended_window(layer.values, past_length, keep_current_indices)
        past_key_values.crop(past_length + len(keep_current_indices))
        return

    # Fallback: just crop (less efficient)
    logger.warning(
        "KV cache compaction: unsupported cache layout, falling back to crop only"
    )
    past_key_values.crop(past_length + len(keep_current_indices))
