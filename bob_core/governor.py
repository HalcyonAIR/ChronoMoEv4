# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright 2026 Halcyon AI Research (jeff@halcyon.ie)
"""
Governor: sits ABOVE compound gate. Governs commit authorization.

Deliberately boring. The contribution is not cleverness. It's that
the rule exists, is measurable, and sits above routing.

Pure reader: reads fast, medium, and slow clock activations.
Never writes to any clock's state.

Minimal deterministic rule set:
- COMMIT requires: fast < Tf AND medium < Tm AND debt < Td AND not in scar
- If any fail: BLOCK
- If hard violation active: ESCALATE

Guardrail: minimum commit rate prevents governor from "winning" by never
committing. If authorized commits in last W steps < min_commits, thresholds
are temporarily relaxed (with decay, not snap-back).
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from bob_core.ledgers import BobCore, CostSignal
from bob_core.medium_clock import MediumClock
from bob_core.motifs import GateResult


class GovernorDecision(Enum):
    ALLOW = "allow"
    BLOCK = "block"
    ESCALATE = "escalate"


@dataclass
class GovernorVerdict:
    """Full decision with reasons — every decision logged."""
    decision: GovernorDecision
    reasons: List[str]
    cost_signal: CostSignal
    debt_level: float
    medium_activation: float
    fast_activation: float = 0.0
    slow_activation: float = 0.0
    thresholds_relaxed: bool = False
    scar_overlap: Optional[float] = None       # Actual scar severity score (for debugging)
    escalation_count: int = 0                   # Cumulative escalations so far


class BobGovernor:
    """Sits ABOVE compound gate. Governs commit authorization.

    Pure reader of three clocks:
    - Fast clock: immediate instability (exploration pressure)
    - Medium clock: regime confidence (sustained instability)
    - Slow clock: constitutional stress (threshold envelope)

    ALLOW: gate passed AND all conditions met
    BLOCK: gate passed BUT governor says no
    ESCALATE: hard violation (3+ conditions fail OR debt > 0.9)

    Governor never writes to any clock's state.
    """

    def __init__(
        self,
        bob_core: BobCore,
        medium_clock: MediumClock,
        fast_clock=None,
        slow_clock=None,
        fast_threshold: float = 0.5,
        medium_threshold: float = 0.5,
        debt_threshold: float = 0.7,
        commit_rate_window: int = 200,
        min_commits_per_window: int = 10,
        relaxation_amount: float = 0.1,
        relaxation_decay_steps: int = 50,
        conflict_register=None,
    ):
        self.bob_core = bob_core
        self.medium_clock = medium_clock
        self.fast_clock = fast_clock
        self.slow_clock = slow_clock
        self.fast_threshold = fast_threshold
        self.medium_threshold = medium_threshold
        self.debt_threshold = debt_threshold
        self._decisions: List[GovernorVerdict] = []

        # Conflict register (optional, for Mode A/B)
        self.conflict_register = conflict_register

        # Minimum commit rate guardrail (governor self-correction)
        self.commit_rate_window = commit_rate_window
        self.min_commits_per_window = min_commits_per_window
        self.relaxation_amount = relaxation_amount
        self._recent_allows: deque = deque(maxlen=commit_rate_window)

        # Threshold relaxation with decay (not snap-back)
        self._relaxation_level: float = 0.0
        self._relaxation_decay_rate: float = 1.0 / relaxation_decay_steps

        # ESCALATE counter (distinct from BLOCK in telemetry)
        self._escalation_count: int = 0

    def _effective_thresholds(self) -> Tuple[float, float, float, bool]:
        """Return (medium_thresh, debt_thresh, scar_thresh, relaxed).

        Composes two sources:
        1. Slow clock threshold envelope (constitutional stress)
        2. Commit rate guardrail (governor self-correction with decay)

        These are independent and additive.
        """
        # Start with base thresholds (prefer calibrated if available)
        if hasattr(self.medium_clock, 'calibrated') and self.medium_clock.calibrated:
            base_medium = self.medium_clock.governor_threshold
        else:
            base_medium = self.medium_threshold
        base_debt = self.debt_threshold
        base_scar = 0.3

        # Slow clock envelope: reshapes background field
        if self.slow_clock is not None:
            base_medium, base_debt, base_scar = self.slow_clock.threshold_envelope(
                base_medium, base_debt, base_scar
            )

        # Commit rate guardrail: governor self-correction
        recent_allows = sum(self._recent_allows)
        starving = (len(self._recent_allows) >= self.commit_rate_window
                    and recent_allows < self.min_commits_per_window)

        if starving:
            self._relaxation_level = min(1.0, self._relaxation_level + 0.2)
        else:
            self._relaxation_level = max(0.0,
                                         self._relaxation_level - self._relaxation_decay_rate)

        if self._relaxation_level > 0.0:
            r = self._relaxation_level * self.relaxation_amount
            return (base_medium + r, base_debt + r, base_scar + r, True)

        # Conflict register Mode B: favour stable commit under pressure
        # If both fast and medium are calm, permit slightly elevated risk
        if (self.conflict_register is not None
                and self.conflict_register.mode == "B"):
            fast_calm = True
            if self.fast_clock is not None:
                fast_thresh = self.fast_threshold
                if hasattr(self.fast_clock, 'calibrated') and self.fast_clock.calibrated:
                    fast_thresh = self.fast_clock.governor_threshold
                fast_calm = self.fast_clock.activation < fast_thresh
            medium_calm = self.medium_clock.activation < base_medium
            if fast_calm and medium_calm:
                # 15% more permissive for stable trajectories under conflict
                return (base_medium * 1.15, base_debt, base_scar, False)

        return (base_medium, base_debt, base_scar, False)

    def evaluate_commit(
        self,
        context_class: int,
        expert_ids: Tuple[int, ...],
        gate_result: GateResult,
        step: int,
    ) -> GovernorVerdict:
        """Should Bob authorize this commit (cheap path)?

        Called only when gate_result.passed is True.
        Pure reader: never writes to any clock state.
        """
        routing_region = tuple(sorted(expert_ids))
        reasons: List[str] = []

        cost_signal = self.bob_core.costs.get_signal()
        debt = self.bob_core.get_debt_for_region(routing_region, step)
        medium_act = self.medium_clock.activation
        fast_act = self.fast_clock.activation if self.fast_clock is not None else 0.0
        slow_act = self.slow_clock.activation if self.slow_clock is not None else 0.0

        medium_thresh, debt_thresh, scar_thresh, relaxed = self._effective_thresholds()

        # Check 1: Scar neighborhood — log actual severity score
        scar_overlap = self.bob_core.scars.scar_severity_score(routing_region, step)
        in_scar = self.bob_core.scars.is_in_scar_neighborhood(
            routing_region, step, threshold=scar_thresh
        )
        if in_scar:
            reasons.append(f"scar_neighborhood: overlap={scar_overlap:.3f} debt={debt:.3f}")

        # Check 2: Medium clock (regime instability)
        if medium_act > medium_thresh:
            reasons.append(f"medium_unstable: {medium_act:.3f} > {medium_thresh:.2f}")

        # Check 3: Debt level
        if debt > debt_thresh:
            reasons.append(f"debt_high: {debt:.3f} > {debt_thresh:.2f}")

        # Check 4: Escalation rate
        if cost_signal.escalation_rate > 0.5:
            reasons.append(f"escalation_rate_high: {cost_signal.escalation_rate:.3f}")

        # Check 5: Fast clock (immediate instability)
        fast_thresh = self.fast_threshold
        if self.fast_clock is not None and hasattr(self.fast_clock, 'calibrated') and self.fast_clock.calibrated:
            fast_thresh = self.fast_clock.governor_threshold
        if fast_act > fast_thresh:
            reasons.append(f"fast_unstable: {fast_act:.3f} > {fast_thresh:.2f}")

        if reasons:
            if len(reasons) >= 3 or debt > 0.9:
                decision = GovernorDecision.ESCALATE
                self._escalation_count += 1
            else:
                decision = GovernorDecision.BLOCK
        else:
            decision = GovernorDecision.ALLOW

        # Track for commit rate guardrail
        self._recent_allows.append(1 if decision == GovernorDecision.ALLOW else 0)

        verdict = GovernorVerdict(
            decision=decision,
            reasons=reasons,
            cost_signal=cost_signal,
            debt_level=debt,
            medium_activation=medium_act,
            fast_activation=fast_act,
            slow_activation=slow_act,
            thresholds_relaxed=relaxed,
            scar_overlap=scar_overlap if scar_overlap > 0.0 else None,
            escalation_count=self._escalation_count,
        )
        self._decisions.append(verdict)
        return verdict

    @property
    def decisions(self) -> List[GovernorVerdict]:
        return self._decisions

    @property
    def blocks_count(self) -> int:
        return sum(
            1 for d in self._decisions
            if d.decision in (GovernorDecision.BLOCK, GovernorDecision.ESCALATE)
        )

    @property
    def allows_count(self) -> int:
        return sum(
            1 for d in self._decisions
            if d.decision == GovernorDecision.ALLOW
        )

    @property
    def escalation_count(self) -> int:
        return self._escalation_count
