# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright 2026 Halcyon AI Research (jeff@halcyon.ie)
"""
Motif store with recency-weighted survival and compound structural gate.

A motif is a validated routing pattern (which experts fired at which layers)
for a context class. The store tracks survival evidence with recency weighting
to prevent survival inflation.

The compound gate requires ALL THREE signals to independently pass:
  1. Routing stability >= threshold
  2. Debt level <= threshold
  3. Motif survival >= threshold

No weighted combination. No confidence. Pass all three or pay full cost.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
import math

from backends.adapter import MotifSpec, LayerMotif


@dataclass
class SurvivalEvent:
    """One success/failure observation for a motif."""
    step: int
    success: bool


@dataclass
class MotifRecord:
    """
    A stored routing motif with recency-weighted survival tracking.

    Unlike the v1 CommitmentCache which used raw counts, survival here
    is computed with exponential recency weighting. Old successes decay.
    A motif must CONTINUE succeeding to maintain gate access.
    """
    motif_id: int
    context_class: int
    motif_spec: MotifSpec
    created_step: int
    routing_key: Tuple[int, ...] = ()  # Coarse key for stability/matching
    events: List[SurvivalEvent] = field(default_factory=list)

    def record_event(self, step: int, success: bool) -> None:
        """Record a success or failure at this step."""
        self.events.append(SurvivalEvent(step=step, success=success))

    def survival_rate(self, current_step: int, half_life: int = 200) -> float:
        """
        Recency-weighted survival rate.

        Recent events count more than old ones. A motif that succeeded
        100 times last epoch but failed 5 times today has lower survival
        than one that succeeded 20 times today.

        Returns 0.0 if no events recorded.
        """
        if not self.events:
            return 0.0

        weighted_success = 0.0
        weighted_total = 0.0

        for event in self.events:
            age = max(0, current_step - event.step)
            weight = math.pow(0.5, age / half_life)
            weighted_total += weight
            if event.success:
                weighted_success += weight

        return weighted_success / weighted_total if weighted_total > 0 else 0.0

    @property
    def total_events(self) -> int:
        return len(self.events)

    @property
    def last_used_step(self) -> int:
        return self.events[-1].step if self.events else self.created_step


@dataclass
class GateSignals:
    """The three independent signals for the compound gate."""
    routing_stability: float   # 0-1: how consistently the router picks the same motif
    debt_level: float          # 0-1: accumulated failure pressure
    motif_survival: float      # 0-1: recency-weighted success rate


@dataclass
class GateThresholds:
    """Per-signal thresholds. All must pass independently."""
    stability_min: float = 0.6    # Router must be this stable for this class
    debt_max: float = 0.5         # Debt must be at most this high
    survival_min: float = 0.7     # Motif must have survived this well recently


@dataclass
class GateResult:
    """Full result of a gate evaluation for falsifiability."""
    passed: bool
    signals: GateSignals
    thresholds_used: GateThresholds
    governance_state: str
    # Which signal(s) failed, if any
    stability_passed: bool = True
    debt_passed: bool = True
    survival_passed: bool = True


class CompoundGate:
    """
    Structural gate: all three signals must independently pass.

    Not confidence-based. Not a weighted combination.
    Model confidence (softmax entropy) is explicitly excluded.

    Governance modulation:
      DRIFT -> always blocked
      EQUILIBRIUM -> normal thresholds
      SETTLEMENT -> thresholds * 0.7 (more aggressive)
    """

    def __init__(self, thresholds: Optional[GateThresholds] = None):
        self.thresholds = thresholds or GateThresholds()

    def evaluate(self, signals: GateSignals, governance_state: str) -> GateResult:
        """Evaluate the compound gate. Returns full result for telemetry."""
        if governance_state == "DRIFT":
            return GateResult(
                passed=False,
                signals=signals,
                thresholds_used=self.thresholds,
                governance_state=governance_state,
                stability_passed=False,
                debt_passed=False,
                survival_passed=False,
            )

        # Governance-modulated thresholds
        stability_thresh = self.thresholds.stability_min
        debt_thresh = self.thresholds.debt_max
        survival_thresh = self.thresholds.survival_min

        if governance_state == "SETTLEMENT":
            stability_thresh *= 0.7
            survival_thresh *= 0.7
            debt_thresh = min(1.0, debt_thresh * 1.3)  # More permissive on debt too

        effective_thresholds = GateThresholds(
            stability_min=stability_thresh,
            debt_max=debt_thresh,
            survival_min=survival_thresh,
        )

        # All three must independently pass
        s_pass = signals.routing_stability >= stability_thresh
        d_pass = signals.debt_level <= debt_thresh
        v_pass = signals.motif_survival >= survival_thresh

        return GateResult(
            passed=s_pass and d_pass and v_pass,
            signals=signals,
            thresholds_used=effective_thresholds,
            governance_state=governance_state,
            stability_passed=s_pass,
            debt_passed=d_pass,
            survival_passed=v_pass,
        )


class MotifStore:
    """
    Per-context-class motif library with routing stability tracking.

    Stores validated routing patterns. Tracks which experts the router
    picks for each context class to measure routing stability.
    Manages debt (failure accumulation) per class.
    """

    def __init__(
        self,
        max_motifs_per_class: int = 5,
        stability_window: int = 50,
        survival_half_life: int = 200,
        success_multiplier: float = 1.2,
        debt_decay: float = 0.95,
    ):
        self.max_motifs_per_class = max_motifs_per_class
        self.stability_window = stability_window
        self.survival_half_life = survival_half_life
        self.success_multiplier = success_multiplier
        self.debt_decay = debt_decay

        # Per-class state
        self._motifs: Dict[int, List[MotifRecord]] = {}
        self._routing_history: Dict[int, deque] = {}
        self._loss_sums: Dict[int, float] = {}
        self._loss_counts: Dict[int, int] = {}
        self._expensive_loss_sums: Dict[int, float] = {}
        self._expensive_loss_counts: Dict[int, int] = {}
        self._debt: Dict[int, float] = {}
        self._cheap_counts: Dict[int, int] = {}
        self._expensive_counts: Dict[int, int] = {}
        self._next_motif_id: int = 0

    def _ensure_class(self, context_class: int) -> None:
        """Lazily initialize state for a context class."""
        if context_class not in self._motifs:
            self._motifs[context_class] = []
            self._routing_history[context_class] = deque(maxlen=self.stability_window)
            self._loss_sums[context_class] = 0.0
            self._loss_counts[context_class] = 0
            self._expensive_loss_sums[context_class] = 0.0
            self._expensive_loss_counts[context_class] = 0
            self._debt[context_class] = 0.0
            self._cheap_counts[context_class] = 0
            self._expensive_counts[context_class] = 0

    def get_top_motif(self, context_class: int, current_step: int) -> Optional[MotifRecord]:
        """Get the best motif for a context class, ranked by recency-weighted survival."""
        self._ensure_class(context_class)
        motifs = self._motifs[context_class]
        if not motifs:
            return None

        # Rank by recency-weighted survival
        ranked = sorted(
            motifs,
            key=lambda m: m.survival_rate(current_step, self.survival_half_life),
            reverse=True,
        )
        return ranked[0]

    def routing_stability(self, context_class: int) -> float:
        """
        Fraction of recent routing decisions that match the most common pattern.

        Measured as the mode frequency in the routing history window.
        """
        self._ensure_class(context_class)
        history = self._routing_history[context_class]
        if len(history) < 5:
            return 0.0

        # Count how often each pattern appears
        pattern_counts: Dict[Tuple, int] = {}
        for pattern in history:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        if not pattern_counts:
            return 0.0

        max_count = max(pattern_counts.values())
        return max_count / len(history)

    def debt_level(self, context_class: int) -> float:
        """Current debt level for a context class. Range [0, 1]."""
        self._ensure_class(context_class)
        return min(1.0, self._debt[context_class])

    def get_gate_signals(
        self,
        context_class: int,
        current_step: int,
    ) -> GateSignals:
        """Compute the three gate signals for a context class."""
        self._ensure_class(context_class)

        stability = self.routing_stability(context_class)
        debt = self.debt_level(context_class)

        top_motif = self.get_top_motif(context_class, current_step)
        survival = (
            top_motif.survival_rate(current_step, self.survival_half_life)
            if top_motif
            else 0.0
        )

        return GateSignals(
            routing_stability=stability,
            debt_level=debt,
            motif_survival=survival,
        )

    def record_routing(
        self,
        context_class: int,
        expert_ids: Tuple[int, ...],
    ) -> None:
        """Record a routing decision for stability tracking."""
        self._ensure_class(context_class)
        self._routing_history[context_class].append(expert_ids)

    def update(
        self,
        context_class: int,
        expert_ids: Tuple[int, ...],
        motif_spec: MotifSpec,
        loss: float,
        step: int,
        was_cheap: bool,
        routing_key: Optional[Tuple[int, ...]] = None,
    ) -> None:
        """Update motif library after a decision."""
        self._ensure_class(context_class)

        # Use routing_key for stability tracking (coarser than full expert_ids)
        if routing_key is None:
            routing_key = expert_ids
        self.record_routing(context_class, routing_key)

        # Update running average loss
        self._loss_sums[context_class] += loss
        self._loss_counts[context_class] += 1

        # Track cheap/expensive counts and expensive-path baseline
        if was_cheap:
            self._cheap_counts[context_class] += 1
        else:
            self._expensive_counts[context_class] += 1
            self._expensive_loss_sums[context_class] += loss
            self._expensive_loss_counts[context_class] += 1

        # Determine success: compare against expensive-path baseline
        # This prevents cheap-path losses from inflating the threshold
        if self._expensive_loss_counts[context_class] > 0:
            baseline_loss = (
                self._expensive_loss_sums[context_class]
                / self._expensive_loss_counts[context_class]
            )
        else:
            baseline_loss = (
                self._loss_sums[context_class] / self._loss_counts[context_class]
            )
        success = loss < baseline_loss * self.success_multiplier

        # Update debt: failures increase, successes decay
        if not success:
            self._debt[context_class] = min(
                1.0, self._debt[context_class] + 0.1
            )
        else:
            self._debt[context_class] *= self.debt_decay

        # Find existing motif or create new one (match on routing_key)
        existing_idx = self._find_motif(context_class, routing_key)

        if existing_idx is not None:
            # Update existing motif (also update motif_spec to latest)
            existing = self._motifs[context_class][existing_idx]
            existing.record_event(step, success)
            existing.motif_spec = motif_spec  # keep motif fresh
        elif not was_cheap:
            # Only create new motifs from expensive path decisions
            motifs = self._motifs[context_class]
            if len(motifs) < self.max_motifs_per_class:
                new_motif = MotifRecord(
                    motif_id=self._next_motif_id,
                    context_class=context_class,
                    motif_spec=motif_spec,
                    created_step=step,
                    routing_key=routing_key,
                )
                new_motif.record_event(step, success)
                motifs.append(new_motif)
                self._next_motif_id += 1
            else:
                # Evict worst motif if new pattern is emerging
                worst_idx = min(
                    range(len(motifs)),
                    key=lambda i: motifs[i].survival_rate(step, self.survival_half_life),
                )
                worst = motifs[worst_idx]
                if (
                    worst.total_events >= 5
                    and worst.survival_rate(step, self.survival_half_life) < 0.4
                ):
                    new_motif = MotifRecord(
                        motif_id=self._next_motif_id,
                        context_class=context_class,
                        motif_spec=motif_spec,
                        created_step=step,
                        routing_key=routing_key,
                    )
                    new_motif.record_event(step, success)
                    motifs[worst_idx] = new_motif
                    self._next_motif_id += 1

    def _find_motif(
        self, context_class: int, routing_key: Tuple[int, ...]
    ) -> Optional[int]:
        """Find motif index by routing_key, or None."""
        for i, m in enumerate(self._motifs[context_class]):
            if m.routing_key == routing_key:
                return i
        return None

    def get_stats(self) -> Dict:
        """Summary statistics for telemetry."""
        stats = {"per_class": {}}
        total_cheap = 0
        total_expensive = 0

        for cc in sorted(self._motifs.keys()):
            cheap = self._cheap_counts.get(cc, 0)
            expensive = self._expensive_counts.get(cc, 0)
            total = cheap + expensive
            total_cheap += cheap
            total_expensive += expensive

            class_stats = {
                "motif_count": len(self._motifs[cc]),
                "routing_stability": round(self.routing_stability(cc), 4),
                "debt_level": round(self.debt_level(cc), 4),
                "cheap_count": cheap,
                "expensive_count": expensive,
                "hit_rate": round(cheap / total, 4) if total > 0 else 0.0,
            }
            stats["per_class"][cc] = class_stats

        grand_total = total_cheap + total_expensive
        stats["overall_hit_rate"] = (
            round(total_cheap / grand_total, 4) if grand_total > 0 else 0.0
        )
        stats["total_cheap"] = total_cheap
        stats["total_expensive"] = total_expensive
        return stats
