# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright 2026 Halcyon AI Research (jeff@halcyon.ie)
"""
Association basins: memory-to-routing bias interface.

Each graph node can accumulate a routing signature — a record of which
experts handled that entity's context well in past sessions. When the
entity is activated (cue-triggered by alias match), the signature
produces a pre-softmax logit adjustment. Same mechanism as scars.

bias_scale HARD CEILING: 0.2. Memory whispers. Consequence shouts.

Spec reference: docs/chronomoe_unified_memory_v1.md, Sections 5-6
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from bob_core.graph import RelationalGraph


# --- Data structures ---

@dataclass
class AssociationBasin:
    """Routing signature for an entity context.

    Records which experts were active when this entity was relevant
    and outcomes were positive.
    """
    node_id: str
    bias_vector: List[List[float]]   # [num_layers][num_experts]
    strength: float = 0.1            # overall confidence, 0.0-1.0
    update_count: int = 0
    last_updated: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "bias_vector": self.bias_vector,
            "strength": round(self.strength, 4),
            "update_count": self.update_count,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "AssociationBasin":
        return cls(
            node_id=d["node_id"],
            bias_vector=d["bias_vector"],
            strength=d.get("strength", 0.1),
            update_count=d.get("update_count", 0),
            last_updated=d.get("last_updated"),
        )


# --- Basin Store ---

class BasinStore:
    """Manages association basins and computes aggregate memory bias.

    Args:
        bias_scale: Global scaling factor for memory influence.
            Start conservative (0.1). HARD CEILING 0.2.
        hard_ceiling: Maximum absolute bias value per expert.
            Asserted, not convention.
    """

    HARD_CEILING = 0.2

    def __init__(self, bias_scale: float = 0.1, hard_ceiling: float = 0.2):
        assert bias_scale <= self.HARD_CEILING, (
            f"bias_scale must be <= {self.HARD_CEILING}, got {bias_scale}"
        )
        assert hard_ceiling <= self.HARD_CEILING, (
            f"hard_ceiling must be <= {self.HARD_CEILING}, got {hard_ceiling}"
        )
        self.bias_scale = bias_scale
        self.hard_ceiling = hard_ceiling
        self._basins: Dict[str, AssociationBasin] = {}

    def get_or_create_basin(
        self,
        node_id: str,
        num_layers: int,
        num_experts: int,
    ) -> AssociationBasin:
        """Get an existing basin or create a new one with zero bias."""
        if node_id not in self._basins:
            self._basins[node_id] = AssociationBasin(
                node_id=node_id,
                bias_vector=[[0.0] * num_experts for _ in range(num_layers)],
            )
        return self._basins[node_id]

    def get_basin(self, node_id: str) -> Optional[AssociationBasin]:
        return self._basins.get(node_id)

    @property
    def basin_count(self) -> int:
        return len(self._basins)

    def to_dict(self) -> Dict:
        return {
            "bias_scale": self.bias_scale,
            "hard_ceiling": self.hard_ceiling,
            "basins": [b.to_dict() for b in self._basins.values()],
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "BasinStore":
        store = cls(
            bias_scale=d.get("bias_scale", 0.1),
            hard_ceiling=d.get("hard_ceiling", 0.2),
        )
        for bd in d.get("basins", []):
            basin = AssociationBasin.from_dict(bd)
            store._basins[basin.node_id] = basin
        return store


# --- Entity Linking ---

def link_entities(
    token_spans: List[str],
    graph: RelationalGraph,
) -> Dict[str, float]:
    """Scan input tokens for entity mentions via exact alias lookup.

    Returns activation map: {node_id: activation_strength}.
    Unambiguous match: 1.0. Ambiguous (multiple candidates): 0.8. No match: skip.

    No embedding search. No fuzzy matching.
    """
    activations: Dict[str, float] = {}

    for span in token_spans:
        candidates = graph.alias_table.lookup(span)

        if len(candidates) == 1:
            activations[candidates[0]] = 1.0
        elif len(candidates) > 1:
            # Ambiguous: activate all candidates at reduced strength
            # (simple approach for Phase 1; spec has graph-proximity disambiguation)
            for cid in candidates:
                if cid not in activations or activations[cid] < 0.8:
                    activations[cid] = 0.8

    return activations


# --- Activation Diffusion ---

def diffuse_activation(
    activations: Dict[str, float],
    graph: RelationalGraph,
    depth: int = 1,
    decay: float = 0.3,
) -> Dict[str, float]:
    """Spread activation from directly-mentioned entities to neighbours.

    depth=1: only immediate neighbours (sufficient for Phase 1).
    decay=0.3: neighbours get 30% of source activation.
    Uses MAX not SUM to prevent runaway when multiple paths converge.

    Returns updated activation map including diffused neighbours.
    """
    diffused = dict(activations)

    for hop in range(depth):
        hop_decay = decay ** (hop + 1)
        new_activations: Dict[str, float] = {}

        for node_id, strength in list(diffused.items()):
            neighbours = graph.get_neighbours(node_id)
            for neighbour_id, _relation in neighbours:
                neighbour_strength = strength * hop_decay
                # MAX not SUM to prevent runaway
                if neighbour_id in new_activations:
                    new_activations[neighbour_id] = max(
                        new_activations[neighbour_id],
                        neighbour_strength,
                    )
                else:
                    new_activations[neighbour_id] = neighbour_strength

        # Merge: max of existing and new (don't override direct mentions)
        for node_id, strength in new_activations.items():
            if node_id not in diffused:
                diffused[node_id] = strength
            else:
                diffused[node_id] = max(diffused[node_id], strength)

    return diffused


# --- Memory Bias Computation ---

@dataclass
class MemoryBiasDiagnostics:
    """Magnitude diagnostics for memory bias. Always logged."""
    mean_abs_bias: float = 0.0
    max_abs_bias: float = 0.0
    bias_to_logit_ratio: float = 0.0  # Requires logit_std from caller
    n_active_nodes: int = 0
    n_active_basins: int = 0

    def to_dict(self) -> Dict:
        return {
            "mean_abs_bias": round(self.mean_abs_bias, 6),
            "max_abs_bias": round(self.max_abs_bias, 6),
            "bias_to_logit_ratio": round(self.bias_to_logit_ratio, 6),
            "n_active_nodes": self.n_active_nodes,
            "n_active_basins": self.n_active_basins,
        }


def compute_memory_bias(
    activations: Dict[str, float],
    basin_store: BasinStore,
    num_layers: int,
    num_experts: int,
) -> Tuple[Optional[Dict[int, List[float]]], MemoryBiasDiagnostics]:
    """Compute aggregate memory bias across all active basins.

    Returns (bias_field, diagnostics).
    bias_field: {layer_id: [num_experts]} or None if nothing to apply.
    diagnostics: magnitude stats for telemetry.

    HARD CEILING enforced: abs(result[layer]).max() <= basin_store.hard_ceiling.

    When bias_scale=0.0, returns (None, diagnostics) to guarantee
    B1 = Condition 0 (no floating-point epsilon drift).
    """
    diag = MemoryBiasDiagnostics(
        n_active_nodes=sum(1 for s in activations.values() if s > 0),
    )

    # Short-circuit: zero scale means no bias (B1 guarantee)
    if basin_store.bias_scale == 0.0:
        return None, diag

    # Short-circuit: no activations
    if not activations:
        return None, diag

    # Accumulate weighted bias across active basins
    bias_field: Dict[int, List[float]] = {}
    active_basins = 0

    for node_id, activation_strength in activations.items():
        basin = basin_store.get_basin(node_id)
        if basin is None or basin.strength <= 0:
            continue
        active_basins += 1

        for layer_idx in range(min(num_layers, len(basin.bias_vector))):
            if layer_idx not in bias_field:
                bias_field[layer_idx] = [0.0] * num_experts

            layer_bias = basin.bias_vector[layer_idx]
            for expert_idx in range(min(num_experts, len(layer_bias))):
                bias_field[layer_idx][expert_idx] += (
                    activation_strength
                    * basin.strength
                    * layer_bias[expert_idx]
                )

    diag.n_active_basins = active_basins

    if not bias_field:
        return None, diag

    # Apply scale and enforce hard ceiling
    scale = basin_store.bias_scale
    ceiling = basin_store.hard_ceiling
    total_abs = 0.0
    max_abs = 0.0
    count = 0

    for layer_idx in bias_field:
        for expert_idx in range(len(bias_field[layer_idx])):
            val = bias_field[layer_idx][expert_idx] * scale
            # Clamp to hard ceiling
            val = max(-ceiling, min(ceiling, val))
            bias_field[layer_idx][expert_idx] = val
            abs_val = abs(val)
            total_abs += abs_val
            if abs_val > max_abs:
                max_abs = abs_val
            count += 1

    diag.mean_abs_bias = total_abs / max(count, 1)
    diag.max_abs_bias = max_abs

    # If all zeros after scaling, return None
    if max_abs < 1e-10:
        return None, diag

    return bias_field, diag
