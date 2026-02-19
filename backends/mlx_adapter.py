# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright 2026 Halcyon AI Research (jeff@halcyon.ie)
"""
MLX MoE adapter: BackendAdapter for Qwen1.5-MoE on Apple Silicon via MLX.

Captures routing telemetry via class-level __call__ monkeypatch (MLX has
no forward hooks). Converts mx.array -> numpy -> torch for LayerSnapshot
compatibility with the existing Bob substrate.

Supports forward() and forward_with_motif() (expert bias via logit offset).

Usage::

    from mlx_lm import load
    model, tokenizer = load("mlx-community/Qwen1.5-MoE-A2.7B-4bit")
    adapter = MLXMoEAdapter(model, tokenizer)

    result = adapter.forward(input_ids_list, target_ids_list)
    # result.snapshots[0].selected_experts -> torch.Tensor [T, top_k]
"""

import math
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from backends.adapter import (
    ForwardResult,
    LayerMotif,
    LayerSnapshot,
    MotifSpec,
    OverlapKind,
)


class _RoutingStore:
    """Thread-local-ish store for captured routing data per forward pass."""

    def __init__(self):
        self.data: Dict[int, dict] = {}
        self.enabled: bool = True
        self.motif: Optional[MotifSpec] = None
        self.memory_bias: Optional[Dict[int, list]] = None
        self.topk_override: Optional[int] = None  # Set to e.g. 1 to collapse routing

    def clear(self):
        self.data.clear()

    def store(self, layer_id: int, gates: mx.array, inds: mx.array, scores: mx.array):
        if not self.enabled:
            return
        self.data[layer_id] = {
            "gates": np.array(gates),
            "inds": np.array(inds),
            "scores": np.array(scores),
        }


def _make_patched_call(store: _RoutingStore):
    """Create a patched __call__ for the MoE block class."""

    def patched_call(self, x: mx.array):
        lid = getattr(self, "_capture_layer_id", -1)

        # Compute gate logits
        gates = self.gate(x)

        # Memory bias (always, before motif)
        if store.memory_bias is not None and lid in store.memory_bias:
            gates = gates + mx.array(store.memory_bias[lid])

        # Apply motif bias if active for this layer
        motif = store.motif
        if motif is not None and lid in motif.layers:
            lm = motif.layers[lid]
            num_experts = gates.shape[-1]
            # Build bias: positive for preferred experts, zero for others
            bias = mx.zeros(num_experts)
            for eid, weight in zip(lm.expert_ids, lm.weights):
                # Scale bias by weight and bias_strength
                bias_val = lm.bias_strength * weight
                bias = bias.at[eid].add(bias_val)
            gates = gates + bias

        gates = mx.softmax(gates, axis=-1, precise=True)

        k = self.top_k
        # Routing perturbation: collapse top-k to override value
        if store.topk_override is not None:
            k = store.topk_override
        inds = mx.stop_gradient(
            mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k]
        )
        scores = mx.take_along_axis(gates, inds, axis=-1)

        # Capture routing
        if store.enabled:
            mx.eval(gates, inds, scores)
            store.store(lid, gates, inds, scores)

        # Standard expert computation
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        shared_expert_output = self.shared_expert(x)
        shared_expert_output = (
            mx.sigmoid(self.shared_expert_gate(x)) * shared_expert_output
        )

        return y + shared_expert_output

    return patched_call


class MLXMoEAdapter:
    """
    BackendAdapter for MLX-based Qwen MoE models.

    Monkeypatches Qwen2MoeSparseMoeBlock.__call__ at the class level
    to intercept routing decisions. Produces LayerSnapshot with torch
    tensors for compatibility with Bob substrate.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self._store = _RoutingStore()
        self._patched = False
        self._moe_class = None

        # Discover MoE blocks and patch
        self._moe_layers: Dict[int, object] = {}
        self._discover_and_patch()

        # Extract config from first MoE block
        first_moe = next(iter(self._moe_layers.values()))
        self.num_experts = first_moe.num_experts
        self.top_k = first_moe.top_k
        self.num_layers = len(self._moe_layers)

    def _discover_and_patch(self):
        """Find MoE blocks and install class-level monkeypatch."""
        layers = self.model.model.layers

        for i, layer in enumerate(layers):
            moe = layer.mlp
            if not hasattr(moe, 'gate') or not hasattr(moe, 'switch_mlp'):
                continue
            moe._capture_layer_id = i
            self._moe_layers[i] = moe
            if self._moe_class is None:
                self._moe_class = type(moe)

        if self._moe_class is None:
            raise ValueError("No MoE blocks found in model")

        # Save original and patch class
        self._moe_class._original_call = self._moe_class.__call__
        self._moe_class.__call__ = _make_patched_call(self._store)
        self._patched = True

    def set_topk_override(self, k: Optional[int]):
        """Override routing top-k for perturbation experiments.

        Set to 1 to collapse routing to top-1 (optionality collapse).
        Set to None to restore normal top-k routing.
        """
        self._store.topk_override = k

    # --- BackendAdapter protocol ---

    @property
    def adapter_version(self) -> str:
        return "mlx-v1"

    @property
    def supports_overlap(self) -> bool:
        return False

    @property
    def overlap_kind(self) -> OverlapKind:
        return OverlapKind.NONE

    def forward(
        self,
        inputs: List[int],
        targets: Optional[List[int]] = None,
        memory_bias: Optional[Dict[int, list]] = None,
    ) -> ForwardResult:
        """Forward pass with routing capture. inputs/targets are token id lists."""
        return self._run_forward(inputs, targets, motif=None, memory_bias=memory_bias)

    def forward_with_motif(
        self,
        inputs: List[int],
        motif: MotifSpec,
        targets: Optional[List[int]] = None,
        memory_bias: Optional[Dict[int, list]] = None,
    ) -> ForwardResult:
        """Forward pass with expert bias applied via motif."""
        return self._run_forward(inputs, targets, motif=motif, memory_bias=memory_bias)

    def forward_counterfactual(
        self,
        inputs: List[int],
        targets: Optional[List[int]] = None,
    ) -> ForwardResult:
        raise NotImplementedError("MLX adapter does not support counterfactual")

    def get_context_embedding(self) -> torch.Tensor:
        raise NotImplementedError("MLX adapter does not support context embedding")

    def _run_forward(
        self,
        inputs: List[int],
        targets: Optional[List[int]],
        motif: Optional[MotifSpec],
        memory_bias: Optional[Dict[int, list]] = None,
    ) -> ForwardResult:
        """Core forward pass with optional motif and memory bias."""
        self._store.clear()
        self._store.enabled = True
        self._store.motif = motif
        self._store.memory_bias = memory_bias

        # Tokenize if needed
        input_ids = mx.array([inputs])
        num_tokens = len(inputs)

        # Forward pass
        logits = self.model(input_ids)
        mx.eval(logits)

        # Compute loss if targets provided
        loss_val = None
        if targets is not None:
            target_ids = mx.array([targets])
            # Cross-entropy loss
            logits_flat = logits[0, :-1, :]  # [T-1, vocab]
            targets_flat = target_ids[0, 1:]  # [T-1]
            # Manual cross-entropy: -log(softmax(logits)[target])
            log_probs = mx.log(mx.softmax(logits_flat, axis=-1) + 1e-8)
            # Gather target log probs
            target_log_probs = mx.take_along_axis(
                log_probs, targets_flat[:, None], axis=-1
            ).squeeze(-1)
            loss_mx = -target_log_probs.mean()
            mx.eval(loss_mx)
            loss_val = torch.tensor(float(loss_mx.item()))

        # Build snapshots from captured routing
        snapshots = self._build_snapshots()

        # Count expert invocations
        expert_invocations = 0
        for snap in snapshots:
            expert_invocations += int(snap.selected_experts.shape[0]) * int(snap.selected_experts.shape[1])

        self._store.motif = None
        self._store.memory_bias = None

        return ForwardResult(
            loss=loss_val,
            logits=None,  # Don't convert full logits to torch (too expensive)
            snapshots=snapshots,
            expert_invocations=expert_invocations,
            tokens_processed=num_tokens,
        )

    def _build_snapshots(self) -> List[LayerSnapshot]:
        """Convert captured routing data to LayerSnapshot objects."""
        snapshots = []

        for layer_id in sorted(self._store.data.keys()):
            d = self._store.data[layer_id]
            gates_np = d["gates"]     # [1, T, num_experts] or [T, num_experts]
            inds_np = d["inds"]       # [1, T, top_k] or [T, top_k]
            scores_np = d["scores"]   # [1, T, top_k] or [T, top_k]

            # Flatten batch dimension if present
            if gates_np.ndim == 3:
                gates_np = gates_np.reshape(-1, gates_np.shape[-1])
                inds_np = inds_np.reshape(-1, inds_np.shape[-1])
                scores_np = scores_np.reshape(-1, scores_np.shape[-1])

            num_tokens = inds_np.shape[0]
            num_experts = gates_np.shape[-1]

            # Router scores (post-softmax gates)
            router_scores = torch.from_numpy(gates_np.astype(np.float32))

            # Selected experts
            selected_experts = torch.from_numpy(inds_np.astype(np.int64))

            # Routing weights
            routing_weights = torch.from_numpy(scores_np.astype(np.float32))

            # Expert usage: selection frequency per expert (sums to ~top_k)
            flat_inds = inds_np.reshape(-1)
            usage_counts = np.bincount(flat_inds, minlength=num_experts).astype(np.float32)
            # Normalize: fraction of tokens that selected each expert
            total_selections = max(num_tokens * inds_np.shape[-1], 1)
            usage = usage_counts / total_selections
            expert_usage = torch.from_numpy(usage)

            # Mean entropy
            log_gates = np.log(gates_np + 1e-8)
            entropy = -(gates_np * log_gates).sum(axis=-1)
            mean_entropy = float(entropy.mean())

            snapshots.append(LayerSnapshot(
                layer_id=layer_id,
                router_scores=router_scores,
                selected_experts=selected_experts,
                routing_weights=routing_weights,
                expert_usage=expert_usage,
                mean_entropy=mean_entropy,
            ))

        return snapshots

    def unpatch(self):
        """Restore original __call__ if needed."""
        if self._patched and self._moe_class is not None:
            self._moe_class.__call__ = self._moe_class._original_call
            self._patched = False
