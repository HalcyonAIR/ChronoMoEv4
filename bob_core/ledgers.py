# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright 2026 Halcyon AI Research (jeff@halcyon.ie)
"""
Ledgers: structured records for commitments, scars, and costs.

Three independent ledgers, unified by BobCore:
- CommitmentLedger: locked routing decisions with full lifecycle
- ScarLedger: per-routing-region harm records with decay
- CostLedger: cost as governance signal (felt, not just logged)

Governance state is stored as numeric coordinates, not strings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math

from backends.adapter import LayerSnapshot


@dataclass
class GovernanceCoords:
    """Numeric governance state. Coarse now, refinable later.

    Stored with each commitment so we know what regime produced it.
    """
    fast: float = 0.0             # Fast clock activation (single-step)
    medium: float = 0.0           # Medium clock activation (run-level stability)
    slow: float = 0.0             # Slow clock / identity pressure
    debt: float = 0.0             # Current scar debt level
    time_since_commit: int = 0    # Steps since last commit in this context
    commit_count: int = 0         # Total commits in this context


@dataclass
class RoutingVector:
    """Structured routing observation. Raw material for future geometric promotion."""
    layer_id: int
    probabilities: Tuple[float, ...]       # Full distribution, sums to 1.0
    delta_from_uniform: Tuple[float, ...]  # probabilities - 1/num_experts
    selected_experts: Tuple[int, ...]      # Top-k expert IDs
    routing_weights: Tuple[float, ...]     # Weights for selected experts
    num_experts: int

    @classmethod
    def from_snapshot(cls, snapshot: LayerSnapshot) -> 'RoutingVector':
        """Build from LayerSnapshot. Computes delta from uniform."""
        scores = snapshot.router_scores
        # Softmax to get probabilities (scores may be pre-softmax logits)
        if scores.dim() == 2:
            # Average across tokens: [B*T, num_experts] -> [num_experts]
            avg_scores = scores.mean(dim=0)
        else:
            avg_scores = scores

        # Normalize to probabilities
        probs = avg_scores.softmax(dim=-1)
        probs_tuple = tuple(float(p) for p in probs)
        num_experts = len(probs_tuple)
        uniform = 1.0 / num_experts
        delta = tuple(p - uniform for p in probs_tuple)

        # Selected experts and weights from first token (representative)
        sel = snapshot.selected_experts[0]  # [top_k]
        wts = snapshot.routing_weights[0]   # [top_k]
        selected = tuple(int(e) for e in sel)
        weights = tuple(float(w) for w in wts)

        return cls(
            layer_id=snapshot.layer_id,
            probabilities=probs_tuple,
            delta_from_uniform=delta,
            selected_experts=selected,
            routing_weights=weights,
            num_experts=num_experts,
        )


@dataclass
class Commitment:
    """A locked routing decision with full lifecycle and provenance."""
    commitment_id: int
    context_class: int
    routing_signature: Tuple[int, ...]  # Sorted expert IDs (the "region")
    governance_coords: GovernanceCoords
    loss_at_proposal: float
    baseline_loss: float
    status: str  # "proposed" | "active" | "fulfilled" | "broken" | "expired"
    created_step: int
    activated_step: Optional[int] = None
    resolved_step: Optional[int] = None
    motif_id: Optional[int] = None
    routing_vector: Optional[RoutingVector] = None


@dataclass
class Scar:
    """Per-routing-region harm record. Not a float — an object with provenance."""
    scar_id: int
    routing_region: Tuple[int, ...]  # Sorted expert IDs
    severity: float                   # 0.0 to 1.0
    created_step: int
    last_triggered_step: int
    trigger_count: int = 1

    def decayed_severity(self, current_step: int, half_life: int = 500) -> float:
        age = max(0, current_step - self.last_triggered_step)
        return self.severity * math.pow(0.5, age / half_life)


@dataclass
class CostSignal:
    """Cost as governance input — felt, not just logged."""
    cheap_count: int
    expensive_count: int
    cheap_fraction: float
    avg_invocations: float
    escalation_rate: float
    exploration_frequency: float


class CommitmentLedger:
    """Full commitment lifecycle: propose → activate → resolve.

    Bounded storage per context class. Evicts oldest resolved.
    Tracks expensive-path running average for baseline comparison.
    """

    def __init__(self, max_per_class: int = 10, compound_factor: float = 0.5):
        self.max_per_class = max_per_class
        self._commitments: Dict[int, List[Commitment]] = {}  # context_class -> list
        self._next_id: int = 0
        self._expensive_loss_sums: Dict[int, float] = {}
        self._expensive_loss_counts: Dict[int, int] = {}
        self._last_commit_step: Dict[int, int] = {}  # context_class -> step
        self._commit_counts: Dict[int, int] = {}      # context_class -> count

    def _ensure_class(self, context_class: int) -> None:
        if context_class not in self._commitments:
            self._commitments[context_class] = []
            self._expensive_loss_sums[context_class] = 0.0
            self._expensive_loss_counts[context_class] = 0
            self._last_commit_step[context_class] = 0
            self._commit_counts[context_class] = 0

    def propose(
        self,
        context_class: int,
        routing_sig: Tuple[int, ...],
        governance_coords: GovernanceCoords,
        loss: float,
        baseline: float,
        step: int,
        motif_id: Optional[int] = None,
        routing_vector: Optional[RoutingVector] = None,
    ) -> Commitment:
        self._ensure_class(context_class)
        c = Commitment(
            commitment_id=self._next_id,
            context_class=context_class,
            routing_signature=routing_sig,
            governance_coords=governance_coords,
            loss_at_proposal=loss,
            baseline_loss=baseline,
            status="proposed",
            created_step=step,
            motif_id=motif_id,
            routing_vector=routing_vector,
        )
        self._next_id += 1
        self._commitments[context_class].append(c)
        self._evict_if_needed(context_class)
        return c

    def activate(self, commitment_id: int, step: int) -> None:
        c = self._find(commitment_id)
        if c and c.status == "proposed":
            c.status = "active"
            c.activated_step = step
            self._last_commit_step[c.context_class] = step
            self._commit_counts[c.context_class] = (
                self._commit_counts.get(c.context_class, 0) + 1
            )

    def resolve(self, commitment_id: int, loss: float, step: int) -> str:
        c = self._find(commitment_id)
        if not c or c.status != "active":
            return "not_found"
        success = loss < c.baseline_loss * 1.2  # same multiplier semantics
        c.status = "fulfilled" if success else "broken"
        c.resolved_step = step
        return c.status

    def get_active(self, context_class: int) -> List[Commitment]:
        self._ensure_class(context_class)
        return [c for c in self._commitments[context_class] if c.status == "active"]

    def baseline_loss(self, context_class: int) -> float:
        self._ensure_class(context_class)
        if self._expensive_loss_counts[context_class] > 0:
            return (
                self._expensive_loss_sums[context_class]
                / self._expensive_loss_counts[context_class]
            )
        return float("inf")

    def record_expensive_loss(self, context_class: int, loss: float) -> None:
        self._ensure_class(context_class)
        self._expensive_loss_sums[context_class] += loss
        self._expensive_loss_counts[context_class] += 1

    def time_since_commit(self, context_class: int, current_step: int) -> int:
        self._ensure_class(context_class)
        last = self._last_commit_step.get(context_class, 0)
        return current_step - last

    def commit_count(self, context_class: int) -> int:
        return self._commit_counts.get(context_class, 0)

    def _find(self, commitment_id: int) -> Optional[Commitment]:
        for commitments in self._commitments.values():
            for c in commitments:
                if c.commitment_id == commitment_id:
                    return c
        return None

    def _evict_if_needed(self, context_class: int) -> None:
        commitments = self._commitments[context_class]
        if len(commitments) <= self.max_per_class:
            return
        # Evict oldest resolved
        resolved = [
            (i, c) for i, c in enumerate(commitments)
            if c.status in ("fulfilled", "broken", "expired")
        ]
        if resolved:
            commitments.pop(resolved[0][0])


class ScarLedger:
    """Per-routing-region harm records with decay and compounding.

    Key invariant: failure on experts (1,3) does NOT increase debt for experts (0,2).

    Scars decay with half_life and have a cooldown window. A scar only blocks
    commits if it was triggered within the cooldown period AND its decayed
    severity exceeds the neighborhood threshold. This prevents ancient scars
    from permanently blocking regions.
    """

    def __init__(self, compound_factor: float = 0.3, half_life: int = 200,
                 cooldown_steps: int = 100, enabled: bool = True,
                 debt_cap: float = 1.0):
        self._scars: Dict[Tuple[int, ...], Scar] = {}  # routing_region -> Scar
        self._next_id: int = 0
        self.compound_factor = compound_factor
        self.half_life = half_life
        self.cooldown_steps = cooldown_steps
        self.enabled = enabled
        self.debt_cap = debt_cap

    def record_harm(
        self, routing_region: Tuple[int, ...], severity: float, step: int
    ) -> Scar:
        if not self.enabled:
            return Scar(scar_id=-1, routing_region=tuple(sorted(routing_region)),
                        severity=0.0, created_step=step, last_triggered_step=step)
        region = tuple(sorted(routing_region))
        if region in self._scars:
            scar = self._scars[region]
            scar.severity = min(1.0, scar.severity + severity * self.compound_factor)
            scar.last_triggered_step = step
            scar.trigger_count += 1
            return scar
        scar = Scar(
            scar_id=self._next_id,
            routing_region=region,
            severity=min(1.0, severity),
            created_step=step,
            last_triggered_step=step,
        )
        self._next_id += 1
        self._scars[region] = scar
        return scar

    def debt_level(self, routing_region: Tuple[int, ...], current_step: int) -> float:
        region = tuple(sorted(routing_region))
        scar = self._scars.get(region)
        if not scar:
            return 0.0
        return min(1.0, scar.decayed_severity(current_step, self.half_life))

    def total_debt(self, current_step: int) -> float:
        total = sum(
            s.decayed_severity(current_step, self.half_life)
            for s in self._scars.values()
        )
        return min(self.debt_cap, total)

    def scar_severity_score(
        self, routing_region: Tuple[int, ...], current_step: int,
    ) -> float:
        """Return the actual decayed severity for this region. 0.0 if no scar.

        Use this to log overlap scores per verdict for cross-model debugging.
        """
        if not self.enabled:
            return 0.0
        region = tuple(sorted(routing_region))
        scar = self._scars.get(region)
        if not scar:
            return 0.0
        age = current_step - scar.last_triggered_step
        if age > self.cooldown_steps:
            return 0.0
        return scar.decayed_severity(current_step, self.half_life)

    def is_in_scar_neighborhood(
        self, routing_region: Tuple[int, ...], current_step: int,
        threshold: float = 0.3,
    ) -> bool:
        """Check if a region is actively scarred.

        A scar blocks only if:
        1. It was triggered within the cooldown window, AND
        2. Its decayed severity exceeds the threshold

        This prevents ancient scars from permanently blocking regions.
        """
        if not self.enabled:
            return False
        region = tuple(sorted(routing_region))
        scar = self._scars.get(region)
        if not scar:
            return False
        # Cooldown check: scar must be recently active to block
        age = current_step - scar.last_triggered_step
        if age > self.cooldown_steps:
            return False
        return scar.decayed_severity(current_step, self.half_life) > threshold


class CostLedger:
    """Cost as governance signal. Bounded sliding window."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._was_cheap: List[bool] = []
        self._invocations: List[int] = []
        self._was_exploration: List[bool] = []
        self._was_blocked: List[bool] = []

    def record(
        self, was_cheap: bool, invocations: int,
        was_exploration: bool = False, was_blocked: bool = False,
    ) -> None:
        self._was_cheap.append(was_cheap)
        self._invocations.append(invocations)
        self._was_exploration.append(was_exploration)
        self._was_blocked.append(was_blocked)
        # Trim to window
        if len(self._was_cheap) > self.window_size:
            self._was_cheap = self._was_cheap[-self.window_size:]
            self._invocations = self._invocations[-self.window_size:]
            self._was_exploration = self._was_exploration[-self.window_size:]
            self._was_blocked = self._was_blocked[-self.window_size:]

    def get_signal(self) -> CostSignal:
        n = len(self._was_cheap)
        if n == 0:
            return CostSignal(0, 0, 0.0, 0.0, 0.0, 0.0)
        cheap = sum(self._was_cheap)
        expensive = n - cheap
        return CostSignal(
            cheap_count=cheap,
            expensive_count=expensive,
            cheap_fraction=cheap / n,
            avg_invocations=sum(self._invocations) / n,
            escalation_rate=sum(self._was_blocked) / n,
            exploration_frequency=sum(self._was_exploration) / n,
        )

    def cost_pressure(self) -> float:
        """0.0 = all cheap, 1.0 = all expensive."""
        n = len(self._was_cheap)
        if n == 0:
            return 0.5
        return 1.0 - sum(self._was_cheap) / n

    def cheap_eligibility(self) -> float:
        """How eligible we are for cheap path. Feeds governor."""
        signal = self.get_signal()
        # If we're mostly cheap and escalation is low, eligibility is high
        return signal.cheap_fraction * (1.0 - signal.escalation_rate)


class BobCore:
    """Unified wrapper for all three ledgers.

    Single entry point: process_outcome handles commitment lifecycle + scar + cost.
    """

    def __init__(self, success_multiplier: float = 1.2, debt_cap: float = 1.0):
        self.commitments = CommitmentLedger()
        self.scars = ScarLedger(debt_cap=debt_cap)
        self.costs = CostLedger()
        self.success_multiplier = success_multiplier

    def process_outcome(
        self,
        context_class: int,
        step: int,
        loss: float,
        was_cheap: bool,
        expert_ids: Tuple[int, ...],
        routing_vector: Optional[RoutingVector],
        governance_coords: GovernanceCoords,
        expert_invocations: int,
        was_exploration: bool = False,
        was_blocked: bool = False,
        motif_id: Optional[int] = None,
    ) -> Optional[Commitment]:
        """Single entry point: commitment lifecycle + scar (if failure) + cost."""
        routing_sig = tuple(sorted(expert_ids))
        baseline = self.commitments.baseline_loss(context_class)
        success = self.determine_success(loss, baseline)

        # Track expensive-path baseline
        if not was_cheap:
            self.commitments.record_expensive_loss(context_class, loss)

        # Cost ledger
        self.costs.record(was_cheap, expert_invocations, was_exploration, was_blocked)

        # Commitment lifecycle
        commitment = None
        if was_cheap and not was_blocked:
            # This was a commit — propose and activate immediately
            commitment = self.commitments.propose(
                context_class=context_class,
                routing_sig=routing_sig,
                governance_coords=governance_coords,
                loss=loss,
                baseline=baseline if baseline != float("inf") else loss,
                step=step,
                motif_id=motif_id,
                routing_vector=routing_vector,
            )
            self.commitments.activate(commitment.commitment_id, step)

            # Resolve immediately based on outcome
            status = self.commitments.resolve(commitment.commitment_id, loss, step)
            if status == "broken":
                # Record harm at this routing region
                severity = min(1.0, (loss - baseline) / max(baseline, 1e-6))
                severity = max(0.0, severity)
                self.scars.record_harm(routing_sig, severity, step)

        elif not success and not was_cheap:
            # Expensive path failure — scar the region only on meaningful failures.
            # Threshold 0.3 = loss must exceed baseline by 30% to create a scar.
            # This prevents normal variance from scarring every region.
            if baseline != float("inf"):
                severity = min(1.0, max(0.0, (loss - baseline) / max(baseline, 1e-6)))
                if severity > 0.3:
                    self.scars.record_harm(routing_sig, severity, step)

        return commitment

    def determine_success(self, loss: float, baseline: float) -> bool:
        if baseline == float("inf"):
            return True  # No baseline yet, assume success
        return loss < baseline * self.success_multiplier

    def get_debt_for_region(self, routing_region: Tuple[int, ...], step: int) -> float:
        return self.scars.debt_level(routing_region, step)

    def get_governance_coords(
        self, context_class: int, step: int,
        fast: float = 0.0, medium: float = 0.0, slow: float = 0.0,
    ) -> GovernanceCoords:
        """Build governance coordinates from current state."""
        return GovernanceCoords(
            fast=fast,
            medium=medium,
            slow=slow,
            debt=self.scars.total_debt(step),
            time_since_commit=self.commitments.time_since_commit(context_class, step),
            commit_count=self.commitments.commit_count(context_class),
        )
