# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright 2026 Halcyon AI Research (jeff@halcyon.ie)
"""
Fast clock: the reflex.

Two parallel detectors, same clock:

1. Volatility detector (original): watches rate-of-change signals
   (churn, entropy delta, loss delta, expert flips). Fires on instability.

2. Low-Neff detector (funnel alarm): watches the absolute effective
   expert count. Fires when Neff drops below the calibrated floor
   for K consecutive steps. Detects optionality collapse — the
   anti-maniac signal.

Both detectors feed into the same soft-override path: relax gate
thresholds, don't bypass governance entirely.

Calibration: during warmup, collects activation AND Neff samples.
Sets explore_threshold to p90 of activation, neff_floor to p10 of Neff.
Both are model-relative, not absolute.

Alpha 0.3: responds in ~3 steps, forgets in ~8.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class FastClockState:
    """Fast-timescale EMAs. Updated every step."""
    churn_ema: float = 0.0          # EMA of Jaccard distance between expert sets
    entropy_delta_ema: float = 0.0  # EMA of |entropy change| between steps
    loss_delta_ema: float = 0.0     # EMA of |loss change| between steps
    expert_flip_ema: float = 0.0    # EMA of dominant-expert changes


class FastClock:
    """Continuous high-frequency oscillator with funnel detection.

    Reads raw per-step routing signals. Produces two independent alarms:
    - exploration_pressure: volatility above calibrated threshold
    - neff_collapse: effective expert count below calibrated floor
      for K consecutive steps

    Never writes to medium or slow clock state.
    """

    def __init__(
        self,
        ema_alpha: float = 0.3,
        explore_threshold: float = 0.4,
        governor_threshold: float = 0.5,
        churn_weight: float = 0.30,
        entropy_weight: float = 0.25,
        loss_weight: float = 0.25,
        flip_weight: float = 0.20,
        calibration_steps: int = 0,
        calibration_percentile: float = 90.0,
        neff_floor_percentile: float = 10.0,
        neff_collapse_window: int = 3,
    ):
        self.ema_alpha = ema_alpha
        self.explore_threshold = explore_threshold
        self.governor_threshold = governor_threshold
        self._weights = (churn_weight, entropy_weight, loss_weight, flip_weight)
        self._state = FastClockState()

        # Calibration state (volatility)
        self._calibration_steps = calibration_steps
        self._calibration_percentile = calibration_percentile
        self._calibration_samples: List[float] = []
        self._calibrated = calibration_steps == 0

        # Calibration state (Neff floor — per-layer)
        self._neff_floor_percentile = neff_floor_percentile
        self._neff_calibration_samples: List[List[float]] = []  # list of per-layer vectors
        self.neff_floor: Optional[float] = None  # Scalar floor (min of per-layer floors)
        self._neff_floors_per_layer: Optional[List[float]] = None  # Per-layer p10 floors

        # Low-Neff detector state (per-layer consecutive counters, max trigger)
        self._neff_collapse_window = neff_collapse_window
        self._consecutive_low_neff: Optional[List[int]] = None  # Per-layer counters
        self._num_layers: Optional[int] = None

        self._tick_count = 0

    def tick(
        self,
        prev_expert_ids: Optional[Tuple[int, ...]],
        curr_expert_ids: Tuple[int, ...],
        prev_loss: Optional[float],
        curr_loss: float,
        prev_entropy: Optional[float],
        curr_entropy: Optional[float],
        neff: Optional[float] = None,
        neff_per_layer: Optional[List[float]] = None,
    ) -> FastClockState:
        """Update all EMAs and Neff detector from this step's raw signals.

        Called once per step, after execution, with actual data.

        neff_per_layer: per-layer Neff values. Each layer gets its own
        consecutive counter; collapse fires when ANY layer has been below
        its calibrated floor for K consecutive steps (max trigger).
        Falls back to scalar neff if neff_per_layer not provided.
        """
        alpha = self.ema_alpha

        # Churn: Jaccard distance between consecutive expert sets
        if prev_expert_ids is not None and curr_expert_ids:
            prev_set = set(prev_expert_ids)
            curr_set = set(curr_expert_ids)
            union = prev_set | curr_set
            intersection = prev_set & curr_set
            churn = 1.0 - (len(intersection) / len(union)) if union else 0.0
        else:
            churn = 0.0

        # Entropy delta: how much did router entropy shift?
        if prev_entropy is not None and curr_entropy is not None:
            entropy_delta = abs(curr_entropy - prev_entropy)
        else:
            entropy_delta = 0.0

        # Loss delta: how much did loss jump?
        if prev_loss is not None:
            loss_delta = abs(curr_loss - prev_loss)
        else:
            loss_delta = 0.0

        # Expert flip: did the dominant expert change?
        if prev_expert_ids and curr_expert_ids:
            expert_flip = 1.0 if prev_expert_ids[0] != curr_expert_ids[0] else 0.0
        else:
            expert_flip = 0.0

        # Update EMAs
        self._state.churn_ema = (1 - alpha) * self._state.churn_ema + alpha * churn
        self._state.entropy_delta_ema = (1 - alpha) * self._state.entropy_delta_ema + alpha * entropy_delta
        self._state.loss_delta_ema = (1 - alpha) * self._state.loss_delta_ema + alpha * loss_delta
        self._state.expert_flip_ema = (1 - alpha) * self._state.expert_flip_ema + alpha * expert_flip

        # Update per-layer Neff detector (max trigger)
        if neff_per_layer is not None:
            n_layers = len(neff_per_layer)
            # Initialize per-layer counters on first observation
            if self._consecutive_low_neff is None or self._num_layers != n_layers:
                self._consecutive_low_neff = [0] * n_layers
                self._num_layers = n_layers
            # Update each layer's consecutive counter
            if self._neff_floors_per_layer is not None:
                for i, (layer_neff, floor) in enumerate(
                    zip(neff_per_layer, self._neff_floors_per_layer)
                ):
                    if layer_neff < floor:
                        self._consecutive_low_neff[i] += 1
                    else:
                        self._consecutive_low_neff[i] = 0
        elif neff is not None:
            # Scalar fallback: single-layer behavior
            if self._consecutive_low_neff is None:
                self._consecutive_low_neff = [0]
                self._num_layers = 1
            if self.neff_floor is not None:
                if neff < self.neff_floor:
                    self._consecutive_low_neff[0] += 1
                else:
                    self._consecutive_low_neff[0] = 0

        self._tick_count += 1

        # Calibration: collect samples, then freeze thresholds
        if not self._calibrated:
            self._calibration_samples.append(self.activation)
            if neff_per_layer is not None:
                self._neff_calibration_samples.append(neff_per_layer)
            elif neff is not None:
                self._neff_calibration_samples.append([neff])
            if self._tick_count >= self._calibration_steps:
                self._finalize_calibration()

        return self._state

    def _finalize_calibration(self):
        """Set explore_threshold and per-layer neff_floors from observed distributions."""
        # Volatility threshold (p90)
        if self._calibration_samples:
            samples = sorted(self._calibration_samples)
            n = len(samples)
            idx = (self._calibration_percentile / 100.0) * (n - 1)
            lo = int(idx)
            hi = min(lo + 1, n - 1)
            frac = idx - lo
            threshold = samples[lo] * (1 - frac) + samples[hi] * frac
            self.explore_threshold = max(0.1, threshold)
            # Governor threshold at same percentile (model-relative)
            self.governor_threshold = max(0.1, threshold)

        # Neff floor (p10) — per-layer calibration
        if self._neff_calibration_samples:
            n_layers = len(self._neff_calibration_samples[0])
            per_layer_floors = []
            for layer_idx in range(n_layers):
                layer_samples = sorted(
                    s[layer_idx] for s in self._neff_calibration_samples
                )
                n = len(layer_samples)
                idx = (self._neff_floor_percentile / 100.0) * (n - 1)
                lo_i = int(idx)
                hi_i = min(lo_i + 1, n - 1)
                frac = idx - lo_i
                floor = layer_samples[lo_i] * (1 - frac) + layer_samples[hi_i] * frac
                per_layer_floors.append(floor)

            self._neff_floors_per_layer = per_layer_floors
            self.neff_floor = min(per_layer_floors)  # Scalar summary (telemetry)
            self._num_layers = n_layers
            self._consecutive_low_neff = [0] * n_layers

        self._calibrated = True
        self._calibration_samples = []
        self._neff_calibration_samples = []

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    @property
    def activation(self) -> float:
        """0-1 immediate instability signal. Weighted sum of EMAs, clipped."""
        w = self._weights
        raw = (
            w[0] * self._state.churn_ema
            + w[1] * self._state.entropy_delta_ema
            + w[2] * self._state.loss_delta_ema
            + w[3] * self._state.expert_flip_ema
        )
        return max(0.0, min(1.0, raw))

    @property
    def exploration_pressure(self) -> bool:
        """Volatility alarm: should the system take an honest look right now?

        Returns False during calibration. After calibration, returns True
        when activation exceeds the calibrated threshold.
        """
        if not self._calibrated:
            return False
        return self.activation > self.explore_threshold

    @property
    def neff_collapse(self) -> bool:
        """Funnel alarm: has ANY layer's Neff collapsed?

        Per-layer with max trigger: each layer tracks its own consecutive
        below-floor count. Fires when ANY single layer has been below its
        calibrated floor for neff_collapse_window consecutive steps.

        This prevents one layer from silently rigidifying while the
        average hides it.

        Returns False if not calibrated or no Neff floors established.
        """
        if not self._calibrated or self._consecutive_low_neff is None:
            return False
        return max(self._consecutive_low_neff) >= self._neff_collapse_window

    @property
    def neff_collapse_layers(self) -> Optional[List[int]]:
        """Which layers are currently in collapse (for telemetry)."""
        if not self._calibrated or self._consecutive_low_neff is None:
            return None
        return [
            i for i, c in enumerate(self._consecutive_low_neff)
            if c >= self._neff_collapse_window
        ]

    @property
    def state(self) -> FastClockState:
        return self._state
