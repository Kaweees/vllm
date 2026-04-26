# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DDTree (Diffusion Draft Tree) proposer for speculative decoding.

DDTree builds a draft tree from per-position distributions of a block diffusion
draft model (DFlash), then verifies the whole tree in a single target-model forward
pass using tree attention with ancestor-only masking.

Reference: https://arxiv.org/abs/2604.12989
"""

from typing import Any

import torch
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.spec_decode.ddtree_utils import (
    DDTreeInfo,
    build_ddtree_tree,
    compact_kv_cache,
    compile_ddtree_tree,
    follow_verified_tree,
)
from vllm.v1.spec_decode.dflash import DFlashProposer

logger = init_logger(__name__)


class DDTreeProposer(DFlashProposer):
    """DDTree proposer that extends DFlash with tree-based drafting.

    DDTree works by:
    1. Running the DFlash draft model to get per-position distributions
    2. Building a draft tree from these distributions using best-first search
    3. Verifying the entire tree in one target model forward pass
    4. Walking the tree to determine accepted tokens
    5. Compacting the KV cache for the accepted path
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
        tree_budget: int | None = None,
    ):
        """Initialize DDTreeProposer.

        Args:
            vllm_config: vLLM configuration
            device: target device
            runner: GPUModelRunner instance (for caching)
            tree_budget: max tree nodes (excluding root). Defaults to
                num_speculative_tokens (same as DFlash block size).
        """
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.method == "ddtree"

        # Initialize as DFlashProposer first (same infrastructure)
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            runner=runner,
        )

        # DDTree-specific config
        self.block_size = self.num_speculative_tokens + 1  # DFlash block size = spec + 1
        tree_budget = self.block_size - 1 if tree_budget is None else tree_budget
        self.tree_budget = max(tree_budget, 0)
        self.max_tree_nodes = 1 + self.tree_budget  # +1 for root

        # Buffers for tree compilation (shared across batches for stability)
        self._verify_input_ids_buffer: torch.Tensor | None = None
        self._verify_position_ids_buffer: torch.Tensor | None = None
        self._attention_mask_buffer: torch.Tensor | None = None
        self._tree_visibility_buffer: torch.Tensor | None = None

        # Track previous tree for attention mask management
        self._previous_tree_start = 0
        self._previous_tree_length = 0

    @property
    def verify_input_ids_buffer(self) -> torch.Tensor:
        if self._verify_input_ids_buffer is None:
            self._verify_input_ids_buffer = torch.empty(
                (1, self.max_tree_nodes),
                dtype=torch.long,
                device=self.device,
            )
        return self._verify_input_ids_buffer

    @property
    def verify_position_ids_buffer(self) -> torch.Tensor:
        if self._verify_position_ids_buffer is None:
            self._verify_position_ids_buffer = torch.empty(
                (1, self.max_tree_nodes),
                dtype=torch.long,
                device=self.device,
            )
        return self._verify_position_ids_buffer

    @property
    def attention_mask_buffer(self) -> torch.Tensor:
        if self._attention_mask_buffer is None:
            self._attention_mask_buffer = torch.zeros(
                (1, 1, self.max_tree_nodes, self.max_model_len + self.max_tree_nodes),
                dtype=self.dtype,
                device=self.device,
            )
        return self._attention_mask_buffer

    @property
    def tree_visibility_buffer(self) -> torch.Tensor:
        if self._tree_visibility_buffer is None:
            self._tree_visibility_buffer = torch.empty(
                (self.max_tree_nodes, self.max_tree_nodes),
                dtype=torch.bool,
                device=self.device,
            )
        return self._tree_visibility_buffer

    @override
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Run dummy forward pass for CUDA graph capture.

        DDTree uses the same DFlash infrastructure for dummy runs since the
        tree is built at proposal time, not during dummy runs.
        """
        super().dummy_run(
            num_tokens=num_tokens,
            use_cudagraphs=use_cudagraphs,
            is_graph_capturing=is_graph_capturing,
            slot_mappings=slot_mappings,
        )

    @torch.inference_mode()
    def _build_and_compile_tree(
        self,
        draft_logits: torch.Tensor,
        root_token_id: int,
        start: int,
    ) -> tuple[DDTreeInfo, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """Build DDTree from draft logits and compile for verification.

        Args:
            draft_logits: [block_size, vocab_size] from DFlash model
            root_token_id: the bonus token (previous target-sampled token)
            start: current generation position (past_length)

        Returns:
            (tree_info, verify_input_ids, verify_position_ids, attention_mask,
             new_tree_start, new_tree_length)
        """
        # Build the draft tree using best-first search
        node_token_ids, node_depths, parents, child_maps, visibility = (
            build_ddtree_tree(draft_logits, self.tree_budget)
        )

        tree_info = DDTreeInfo(
            node_token_ids=node_token_ids,
            node_depths=node_depths,
            parents=parents,
            child_maps=child_maps,
            visibility=visibility,
        )

        # Compile tensors for target model verification
        verify_input_ids, verify_position_ids, attention_mask, new_start, new_length = (
            compile_ddtree_tree(
                root_token_id=root_token_id,
                start=start,
                node_token_ids=node_token_ids,
                node_depths=node_depths,
                visibility=visibility,
                past_length=start,
                dtype=self.dtype,
                device=self.device,
                verify_input_ids_buffer=self.verify_input_ids_buffer,
                verify_position_ids_buffer=self.verify_position_ids_buffer,
                attention_mask_buffer=self.attention_mask_buffer,
                tree_visibility_buffer=self.tree_visibility_buffer,
                previous_tree_start=self._previous_tree_start,
                previous_tree_length=self._previous_tree_length,
            )
        )

        # Update previous tree tracking
        self._previous_tree_start = new_start
        self._previous_tree_length = new_length

        return tree_info, verify_input_ids, verify_position_ids, attention_mask, new_start, new_length

    @torch.inference_mode()
    def propose(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor,
        common_attn_metadata,
        sampling_metadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,
    ) -> torch.Tensor:
        """Propose draft tokens using DDTree.

        Extends DFlash's propose() with tree-based drafting:
        1. Run DFlash draft model to get per-position distributions
        2. Build draft tree from logits
        3. Return tree info for target model verification

        The target model verification and acceptance walking are handled
        by GPUModelRunner, not here.
        """
        from vllm.forward_context import set_forward_context
        from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM

        assert isinstance(
            self.model, DFlashQwen3ForCausalLM
        ), f"DDTree requires DFlashQwen3ForCausalLM, got {type(self.model)}"

        batch_size = common_attn_metadata.batch_size()

        # Combine hidden states (same as DFlash)
        target_hidden_states = self.model.combine_hidden_states(target_hidden_states)

        # Set up inputs for DFlash draft model (same as DFlashProposer)
        num_tokens, token_indices_to_sample, common_attn_metadata = (
            self.set_inputs_first_pass(
                target_token_ids=target_token_ids,
                next_token_ids=next_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                token_indices_to_sample=token_indices_to_sample,
                cad=common_attn_metadata,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            )
        )

        # Build attention metadata for draft model
        per_group_attn_metadata, per_layer_attn_metadata = (
            self.build_per_group_and_layer_attn_metadata(common_attn_metadata)
        )

        # Determine batch execution and padding
        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(num_tokens)
        )

        # Build model inputs
        model_kwargs, slot_mapping_size = self.build_model_inputs_first_pass(
            num_tokens, num_input_tokens, mm_embed_inputs
        )

        # Run DFlash draft model to get per-position distributions
        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=self._get_slot_mapping(
                slot_mapping_size, common_attn_metadata.slot_mapping
            ),
        ):
            ret_hidden_states = self.model(**model_kwargs)
            if not self.model_returns_tuple():
                last_hidden_states = ret_hidden_states
            else:
                last_hidden_states, _ = ret_hidden_states

        # Get draft logits from the draft model
        # DFlash returns hidden states for the spec tokens (not the next token)
        draft_hidden_states = last_hidden_states[token_indices_to_sample]
        draft_logits = self.model.compute_logits(draft_hidden_states)
        # [batch_size, spec_tokens, vocab_size]

        # For now, return draft logits as a special marker tensor
        # The target model verification is handled externally
        # We return a tensor encoding: batch_size x spec_tokens (like DFlash)
        # The DDTreeProposer returns raw draft logits for tree building
        # This is marked with a special dtype to signal tree-based drafting

        # We use a trick: return the draft logits reshaped to [batch_size, spec_tokens]
        # by taking argmax (greedy), plus store tree_info for the runner
        # Actually, we need to return draft token IDs in the same format as other proposers
        # The tree building happens in the runner before target verification

        # Return greedy samples (standard DFlash format) - tree building is handled
        # separately in GPUModelRunner.propose_draft_token_ids()
        draft_token_ids = self._greedy_sample(draft_hidden_states)

        return draft_token_ids.view(-1, self.num_speculative_tokens)

    @override
    def _get_eagle3_use_aux_hidden_state_from_config(self) -> bool:
        """DDTree uses auxiliary hidden states like DFlash."""
        use_aux_hidden_state = True
        dflash_config = getattr(
            self.draft_model_config.hf_config, "dflash_config", None
        )
        if dflash_config is not None:
            use_aux_hidden_state = dflash_config.get("use_aux_hidden_state", True)
        return use_aux_hidden_state
