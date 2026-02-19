# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright 2026 Halcyon AI Research (jeff@halcyon.ie)
"""
Slow clock: the constitution.

Holds what the system has decided matters long-term. Anything that survives
medium curation gets promoted: stable scars, stable constraints. Slow isn't
reactive. It reshapes the background field that the other clocks operate within.

Decays slowly, updates slowly, and mostly changes how future situations
are interpreted — by adjusting the threshold envelope that the governor
operates within.

Alpha 0.02: ~50 steps to respond, ~150 steps to forget.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class SlowClockState:
    """Constitutional-timescale EMAs. Barely moves step-to-step."""
    scar_pressure_ema: float = 0.0   # Slow EMA of aggregate scar debt
    stability_ema: float = 0.5       # Slow EMA of routing stability (starts neutral)


class SlowClock:
    """Very slow integrator that reshapes the threshold envelope.

    Reads aggregate scar debt and routing stability from the environment.
    Produces threshold adjustments that tighten or loosen governor behavior.
    Never writes to fast or medium clock state.
    """

    def __init__(
        self,
        ema_alpha: float = 0.02,
        tighten_amount: float = 0.1,
        loosen_amount: float = 0.05,
        stress_scaling: float = 1.0,
        max_tighten_fraction: float = 0.15,
    ):
        self.ema_alpha = ema_alpha
        self.tighten_amount = tighten_amount
        self.loosen_amount = loosen_amount
        self.stress_scaling = stress_scaling
        self.max_tighten_fraction = max_tighten_fraction
        self._state = SlowClockState()

    def tick(
        self,
        scar_debt: float,
        routing_stability: float,
    ) -> SlowClockState:
        """Update constitutional EMAs from environmental signals.

        Called once per step, after execution.

        scar_debt: aggregate scar debt from ScarLedger.total_debt()
        routing_stability: current routing stability from MotifStore
        """
        alpha = self.ema_alpha

        self._state.scar_pressure_ema = (
            (1 - alpha) * self._state.scar_pressure_ema + alpha * scar_debt
        )
        self._state.stability_ema = (
            (1 - alpha) * self._state.stability_ema + alpha * routing_stability
        )

        return self._state

    @property
    def activation(self) -> float:
        """0-1 constitutional stress signal.

        High scar pressure + low stability = high activation.
        System under sustained constitutional stress.
        """
        stress = self._state.scar_pressure_ema * (1.0 - self._state.stability_ema)
        return max(0.0, min(1.0, stress * self.stress_scaling))

    def threshold_envelope(
        self,
        base_medium_thresh: float,
        base_debt_thresh: float,
        base_scar_thresh: float,
    ) -> Tuple[float, float, float]:
        """Adjust governor thresholds based on constitutional state.

        High activation → tighter thresholds (lower values, harder to pass)
        Low activation → looser thresholds (higher values, easier to pass)

        Returns (medium_thresh, debt_thresh, scar_thresh).
        """
        act = self.activation

        if act > 0.5:
            # Constitutional stress: tighten (clamped by envelope floor)
            tighten = (act - 0.5) * 2.0 * self.tighten_amount
            max_med = base_medium_thresh * self.max_tighten_fraction
            max_debt = base_debt_thresh * self.max_tighten_fraction
            max_scar = base_scar_thresh * self.max_tighten_fraction
            return (
                base_medium_thresh - min(tighten, max_med),
                base_debt_thresh - min(tighten, max_debt),
                base_scar_thresh - min(tighten, max_scar),
            )
        elif act < 0.2:
            # Constitutional rest: loosen
            loosen = (0.2 - act) * 5.0 * self.loosen_amount
            return (
                base_medium_thresh + loosen,
                base_debt_thresh + loosen,
                base_scar_thresh + loosen,
            )
        else:
            # Neutral zone: no adjustment
            return (base_medium_thresh, base_debt_thresh, base_scar_thresh)

    @property
    def state(self) -> SlowClockState:
        return self._state
