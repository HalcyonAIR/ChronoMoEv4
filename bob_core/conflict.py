# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright 2026 Halcyon AI Research (jeff@halcyon.ie)
"""
Conflict register: tracks angel-devil tension as a control signal.

The conflict index measures how much internal disagreement exists at
each step (angel_peak * devil_peak). A rolling window produces stats
that select between Mode A (normal) and Mode B (damp + bold commit).

Mode B is the correct response to internal conflict: damping, not
exploration. Bold but calm under pressure.

Implementation: circular buffer of length 50. Fixed memory. No
growing state. Updated every step from TriadSummary.

Spec reference: docs/chronomoe_spec_v03_triad.md, Section 6.7
"""

from collections import deque
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ConflictState:
    """Snapshot of the conflict register at one step."""
    index: float          # angel_peak * devil_peak (this step)
    mean_50: float        # Rolling mean over last 50 steps
    max_spike: float      # Rolling max over last 50 steps
    trending: bool        # Current > mean (rising?)
    mode: str             # "A" or "B"


class ConflictRegister:
    """Circular buffer tracking angel-devil conflict over time.

    Args:
        buffer_size: Rolling window length. Default 50 per spec.
        calibration_steps: Steps to collect samples before setting
            the Mode B threshold (mu). 0 = use default threshold.
        calibration_percentile: Percentile for mu threshold.
            Default 90.0 = Mode B triggers when conflict is in the
            top 10% of warmup distribution.
    """

    def __init__(
        self,
        buffer_size: int = 50,
        calibration_steps: int = 0,
        calibration_percentile: float = 90.0,
    ):
        self._buffer: deque = deque(maxlen=buffer_size)
        self._mu: float = 0.01  # Default Mode B threshold (before calibration)
        self._calibration_steps = calibration_steps
        self._calibration_percentile = calibration_percentile
        self._cal_samples: List[float] = []
        self._calibrated = calibration_steps == 0
        self._tick_count = 0

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    @property
    def mode(self) -> str:
        """Current behavioral mode: 'A' (normal) or 'B' (damp + bold)."""
        if len(self._buffer) == 0:
            return "A"
        current = self._buffer[-1]
        mean = sum(self._buffer) / len(self._buffer)
        trending = current > mean
        high_conflict = trending and mean > self._mu
        return "B" if high_conflict else "A"

    def update(self, angel_peak: float, devil_peak: float) -> ConflictState:
        """Record one step's conflict and return current state.

        Args:
            angel_peak: Max angel score across layers this step.
            devil_peak: Max devil score across layers this step.

        Returns:
            ConflictState with current index, rolling stats, and mode.
        """
        self._tick_count += 1

        # Option A from spec: product (high only when BOTH monitors active)
        conflict = angel_peak * devil_peak
        self._buffer.append(conflict)

        # Calibration
        if not self._calibrated:
            self._cal_samples.append(conflict)
            if self._tick_count >= self._calibration_steps:
                self._finalize_calibration()

        # Compute rolling stats
        buf = list(self._buffer)
        mean_50 = sum(buf) / len(buf)
        max_spike = max(buf)
        trending = conflict > mean_50

        # Mode selection
        high_conflict = trending and mean_50 > self._mu
        mode = "B" if high_conflict else "A"

        return ConflictState(
            index=conflict,
            mean_50=mean_50,
            max_spike=max_spike,
            trending=trending,
            mode=mode,
        )

    def _finalize_calibration(self):
        """Set mu threshold from warmup samples."""
        if self._cal_samples:
            s = sorted(self._cal_samples)
            n = len(s)
            idx = (self._calibration_percentile / 100.0) * (n - 1)
            lo = int(idx)
            hi = min(lo + 1, n - 1)
            frac = idx - lo
            self._mu = s[lo] * (1 - frac) + s[hi] * frac
        self._calibrated = True
        self._cal_samples.clear()
