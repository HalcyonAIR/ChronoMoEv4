# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright 2026 Halcyon AI Research (jeff@halcyon.ie)
"""
DecisionTrace: the atomic artifact. Every decision produces one trace.

The trace is constructed in bob_core from adapter raw ingredients.
The adapter provides snapshots; Bob constructs meaning.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class DecisionTrace:
    """
    Full trace of one routing decision. THE unit of truth.

    Every field logged for falsifiability. If the system gets cheaper
    but quality drops, traces will show it. If it gets cheaper and
    quality holds, we have the artifact.
    """
    step: int
    context_class: int
    governance_state: str
    path: str                          # "cheap" or "full"
    expert_ids: Tuple[int, ...]        # Which experts actually fired
    expert_invocations: int            # THE cost metric
    tokens_processed: int
    loss: float                        # THE quality metric

    # Gate signals (for falsifiability)
    routing_stability: float
    debt_level: float
    motif_survival: float
    gate_passed: bool
    stability_passed: bool
    debt_passed: bool
    survival_passed: bool

    # Motif info
    motif_id: Optional[int] = None

    # Governor fields (all Optional, backward compatible)
    governor_decision: Optional[str] = None
    governor_reasons: Optional[List[str]] = None
    forced_exploration: bool = False
    medium_activation: Optional[float] = None
    scar_debt: Optional[float] = None
    cost_cheap_fraction: Optional[float] = None
    commitment_id: Optional[int] = None
    identity_weight: Optional[float] = None

    # Geometry fields (backward compatible)
    router_entropy: Optional[float] = None       # Mean per-token entropy across MoE layers
    churn: Optional[float] = None                # Raw Jaccard distance from previous step
    scar_hit: Optional[bool] = None              # Was this region in a scar neighborhood?
    baseline_loss: Optional[float] = None        # Expensive-path running avg for this class
    neff: Optional[float] = None                 # Min Neff across layers (worst-case, 1/Herfindahl)
    flipflop_ema: Optional[float] = None         # Medium clock flipflop EMA
    routing_weights_top: Optional[List[float]] = None  # Mean routing weights for top-k slots

    # Governor tightening fields (backward compatible)
    scar_overlap: Optional[float] = None         # Actual scar severity score (not just bool)
    escalation_count: Optional[int] = None       # Cumulative ESCALATE decisions so far
    gate_pass_rate: Optional[float] = None       # Gate-pass rate (independent of governor)
    governor_allow_rate: Optional[float] = None  # Governor allow rate (independent of gate)

    # Three-clock activations (backward compatible)
    fast_activation: Optional[float] = None      # Fast clock activation (0-1, reflex)
    slow_activation: Optional[float] = None      # Slow clock activation (0-1, constitution)

    # Low-Neff funnel detector (backward compatible)
    neff_collapse: Optional[bool] = None         # True when Neff below floor for K steps
    neff_floor: Optional[float] = None           # Calibrated Neff floor (p10 of warmup)
    neff_per_layer: Optional[List[float]] = None  # Per-layer Neff (1/Herfindahl per MoE layer)
    neff_collapse_layers: Optional[List[int]] = None  # Layer indices currently in collapse

    # Triad monitors (backward compatible)
    angel_score: Optional[float] = None          # Peak angel score across layers
    devil_score: Optional[float] = None          # Peak devil score across layers
    maniac_score: Optional[float] = None         # Peak maniac score across layers
    angel_flag: Optional[bool] = None            # Any layer angel-flagged
    devil_flag: Optional[bool] = None            # Any layer devil-flagged
    maniac_flag: Optional[bool] = None           # Any layer maniac-flagged
    triad_intervention: Optional[str] = None     # "devil"/"angel"/"maniac"/None
    triad_intervention_layer: Optional[int] = None  # Layer index for intervention

    # Conflict register (backward compatible)
    conflict_index: Optional[float] = None       # angel_peak * devil_peak
    conflict_mean: Optional[float] = None        # Rolling mean over 50 steps
    conflict_mode: Optional[str] = None          # "A" or "B"
    conflict_trending: Optional[bool] = None     # Rising?

    # Memory system (backward compatible)
    memory_nodes_active: Optional[int] = None    # Entities activated this step
    memory_query_rendered: Optional[bool] = None # Did a fact packet render?
    memory_bias_applied: Optional[bool] = None   # Was memory bias active?
    memory_bias_max: Optional[float] = None      # Max |bias| across layers
    memory_bias_to_logit_ratio: Optional[float] = None  # Bias audibility
    memory_active_basins: Optional[int] = None   # Basins contributing bias
    memory_active_entities: Optional[int] = None # Entities contributing bias

    def to_dict(self) -> Dict:
        d = {
            "step": self.step,
            "context_class": self.context_class,
            "governance_state": self.governance_state,
            "path": self.path,
            "expert_ids": list(self.expert_ids),
            "expert_invocations": self.expert_invocations,
            "tokens_processed": self.tokens_processed,
            "loss": round(self.loss, 6),
            "routing_stability": round(self.routing_stability, 4),
            "debt_level": round(self.debt_level, 4),
            "motif_survival": round(self.motif_survival, 4),
            "gate_passed": self.gate_passed,
            "stability_passed": self.stability_passed,
            "debt_passed": self.debt_passed,
            "survival_passed": self.survival_passed,
            "motif_id": self.motif_id,
        }
        # Governor fields: include when set
        if self.governor_decision is not None:
            d["governor_decision"] = self.governor_decision
        if self.governor_reasons is not None:
            d["governor_reasons"] = self.governor_reasons
        if self.forced_exploration:
            d["forced_exploration"] = True
        if self.medium_activation is not None:
            d["medium_activation"] = round(self.medium_activation, 4)
        if self.scar_debt is not None:
            d["scar_debt"] = round(self.scar_debt, 4)
        if self.cost_cheap_fraction is not None:
            d["cost_cheap_fraction"] = round(self.cost_cheap_fraction, 4)
        if self.commitment_id is not None:
            d["commitment_id"] = self.commitment_id
        if self.identity_weight is not None:
            d["identity_weight"] = round(self.identity_weight, 4)
        # Geometry fields
        if self.router_entropy is not None:
            d["router_entropy"] = round(self.router_entropy, 4)
        if self.churn is not None:
            d["churn"] = round(self.churn, 4)
        if self.scar_hit is not None:
            d["scar_hit"] = self.scar_hit
        if self.baseline_loss is not None:
            d["baseline_loss"] = round(self.baseline_loss, 4)
        if self.neff is not None:
            d["neff"] = round(self.neff, 6)
        if self.flipflop_ema is not None:
            d["flipflop_ema"] = round(self.flipflop_ema, 4)
        if self.routing_weights_top is not None:
            d["routing_wts"] = [round(w, 4) for w in self.routing_weights_top]
        # Governor tightening fields
        if self.scar_overlap is not None:
            d["scar_overlap"] = round(self.scar_overlap, 4)
        if self.escalation_count is not None:
            d["escalation_count"] = self.escalation_count
        if self.gate_pass_rate is not None:
            d["gate_pass_rate"] = round(self.gate_pass_rate, 4)
        if self.governor_allow_rate is not None:
            d["governor_allow_rate"] = round(self.governor_allow_rate, 4)
        # Three-clock activations
        if self.fast_activation is not None:
            d["fast_activation"] = round(self.fast_activation, 4)
        if self.slow_activation is not None:
            d["slow_activation"] = round(self.slow_activation, 4)
        # Low-Neff funnel detector
        if self.neff_collapse is not None:
            d["neff_collapse"] = self.neff_collapse
        if self.neff_floor is not None:
            d["neff_floor"] = round(self.neff_floor, 4)
        if self.neff_per_layer is not None:
            d["neff_per_layer"] = [round(n, 6) for n in self.neff_per_layer]
        if self.neff_collapse_layers is not None and self.neff_collapse_layers:
            d["neff_collapse_layers"] = self.neff_collapse_layers
        # Triad monitors
        if self.angel_score is not None:
            d["angel_score"] = round(self.angel_score, 6)
        if self.devil_score is not None:
            d["devil_score"] = round(self.devil_score, 6)
        if self.maniac_score is not None:
            d["maniac_score"] = round(self.maniac_score, 6)
        if self.angel_flag is not None:
            d["angel_flag"] = self.angel_flag
        if self.devil_flag is not None:
            d["devil_flag"] = self.devil_flag
        if self.maniac_flag is not None:
            d["maniac_flag"] = self.maniac_flag
        if self.triad_intervention is not None:
            d["triad_intervention"] = self.triad_intervention
        if self.triad_intervention_layer is not None:
            d["triad_intervention_layer"] = self.triad_intervention_layer
        # Conflict register
        if self.conflict_index is not None:
            d["conflict_index"] = round(self.conflict_index, 6)
        if self.conflict_mean is not None:
            d["conflict_mean"] = round(self.conflict_mean, 6)
        if self.conflict_mode is not None:
            d["conflict_mode"] = self.conflict_mode
        if self.conflict_trending is not None:
            d["conflict_trending"] = self.conflict_trending
        # Memory system
        if self.memory_nodes_active is not None:
            d["memory_nodes_active"] = self.memory_nodes_active
        if self.memory_query_rendered is not None:
            d["memory_query_rendered"] = self.memory_query_rendered
        if self.memory_bias_applied is not None:
            d["memory_bias_applied"] = self.memory_bias_applied
        if self.memory_bias_max is not None:
            d["memory_bias_max"] = round(self.memory_bias_max, 6)
        if self.memory_bias_to_logit_ratio is not None:
            d["memory_bias_to_logit_ratio"] = round(self.memory_bias_to_logit_ratio, 6)
        if self.memory_active_basins is not None:
            d["memory_active_basins"] = self.memory_active_basins
        if self.memory_active_entities is not None:
            d["memory_active_entities"] = self.memory_active_entities
        return d
