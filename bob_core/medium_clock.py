# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright 2026 Halcyon AI Research (jeff@halcyon.ie)
"""
Medium clock: stateful filter that outputs instability signals.

Medium clock is the stabilizer across a run — not across a single step
and not across the whole lifetime. It detects instability (oscillation,
dithering, repeated undoing), not just failure.

Medium clock is dynamics, not distance. Distance is slow clock (identity pressure).
Dynamics is stability pressure.

Explicitly separate file — not buried in governor.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class MediumClockState:
    """Small vector updated each step."""
    churn_ema: float = 0.0         # EMA of routing-change magnitude (Jaccard distance)
    flipflop_ema: float = 0.0      # EMA of cheap/full alternation
    outcome_var_ema: float = 0.0   # EMA of |loss_change| between same-class steps
    provisional_active: bool = False
    provisional_ttl: int = 0
    last_churn: float = 0.0        # Raw Jaccard distance from most recent tick


class MediumClock:
    """Stateful filter that outputs instability signals.

    Detects oscillation, dithering, repeated undoing across a run.
    """

    def __init__(
        self,
        ema_alpha: float = 0.1,
        instability_threshold: float = 0.5,
        governor_threshold: float = 0.5,
        churn_weight: float = 0.4,
        flipflop_weight: float = 0.3,
        outcome_weight: float = 0.3,
        calibration_steps: int = 0,
        calibration_percentile: float = 90.0,
    ):
        self.ema_alpha = ema_alpha
        self.instability_threshold = instability_threshold
        self.governor_threshold = governor_threshold
        self._weights = (churn_weight, flipflop_weight, outcome_weight)
        self._state = MediumClockState()
        self._last_path: Optional[str] = None

        # Calibration state
        self._calibration_steps = calibration_steps
        self._calibration_percentile = calibration_percentile
        self._calibration_samples: List[float] = []
        self._calibrated = calibration_steps == 0
        self._tick_count = 0

    def tick(
        self,
        prev_expert_ids: Optional[Tuple[int, ...]],
        curr_expert_ids: Tuple[int, ...],
        prev_loss: Optional[float],
        curr_loss: float,
        path: str,
    ) -> MediumClockState:
        """Update all EMAs from this step's observations.

        Reads routing signals only. No governor decisions in state.

        churn_ema: Jaccard distance between prev/curr expert sets
        flipflop_ema: 1.0 if path alternated from last step, else 0.0
        outcome_var_ema: |curr_loss - prev_loss|
        """
        alpha = self.ema_alpha

        # Churn: Jaccard distance between expert sets
        if prev_expert_ids is not None and curr_expert_ids:
            prev_set = set(prev_expert_ids)
            curr_set = set(curr_expert_ids)
            union = prev_set | curr_set
            intersection = prev_set & curr_set
            churn = 1.0 - (len(intersection) / len(union)) if union else 0.0
        else:
            churn = 0.0

        self._state.last_churn = churn

        # Flipflop: did the path alternate?
        if self._last_path is not None and path != self._last_path:
            flipflop = 1.0
        else:
            flipflop = 0.0
        self._last_path = path

        # Outcome variance: |loss change|
        if prev_loss is not None:
            outcome_var = abs(curr_loss - prev_loss)
        else:
            outcome_var = 0.0

        # Update EMAs (3 signals, no escalation)
        self._state.churn_ema = (1 - alpha) * self._state.churn_ema + alpha * churn
        self._state.flipflop_ema = (1 - alpha) * self._state.flipflop_ema + alpha * flipflop
        self._state.outcome_var_ema = (1 - alpha) * self._state.outcome_var_ema + alpha * outcome_var

        # Tick provisional
        if self._state.provisional_active:
            self._state.provisional_ttl -= 1
            if self._state.provisional_ttl <= 0:
                self._state.provisional_active = False

        self._tick_count += 1

        # Calibration: collect samples, then freeze threshold
        if not self._calibrated:
            self._calibration_samples.append(self.activation)
            if self._tick_count >= self._calibration_steps:
                self._finalize_calibration()

        return self._state

    def _finalize_calibration(self):
        """Set governor_threshold from observed activation distribution."""
        if self._calibration_samples:
            samples = sorted(self._calibration_samples)
            n = len(samples)
            idx = (self._calibration_percentile / 100.0) * (n - 1)
            lo = int(idx)
            hi = min(lo + 1, n - 1)
            frac = idx - lo
            threshold = samples[lo] * (1 - frac) + samples[hi] * frac
            self.governor_threshold = max(0.1, threshold)

        self._calibrated = True
        self._calibration_samples = []

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    @property
    def activation(self) -> float:
        """0-1 instability signal. Weighted sum of 3 EMAs, clipped.

        Reads routing signals only. No governor decisions in this computation.
        """
        w = self._weights
        raw = (
            w[0] * self._state.churn_ema
            + w[1] * self._state.flipflop_ema
            + w[2] * self._state.outcome_var_ema
        )
        return max(0.0, min(1.0, raw))

    @property
    def state(self) -> MediumClockState:
        return self._state

    def propose_provisional(self, ttl: int = 20) -> None:
        """Start provisional commitment monitoring."""
        self._state.provisional_active = True
        self._state.provisional_ttl = ttl

    def check_provisional(self, loss_trend: float, debt_slope: float) -> str:
        """Returns 'promote', 'kill', or 'continue'.

        loss_trend: negative = improving, positive = degrading
        debt_slope: positive = debt growing, negative = healing
        """
        if not self._state.provisional_active:
            return "continue"

        if self._state.provisional_ttl <= 0:
            # TTL expired — promote if stable, kill if not
            if loss_trend <= 0 and debt_slope <= 0:
                return "promote"
            return "kill"

        # Still active — check for early kill
        if self.activation > self.instability_threshold * 1.5:
            return "kill"

        return "continue"
