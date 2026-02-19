# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright 2026 Halcyon AI Research (jeff@halcyon.ie)
"""
Triad monitors: Angel, Devil, Maniac.

Three routing-level observers that read the softmax distribution pi
from each MoE layer and compute per-layer scores. They do NOT route
tokens. They feed signals into the existing governor/clock/scar system.

Angel: "Am I losing options?" — optionality collapsing into a dominant expert.
Devil: "Am I chasing power at the expense of flexibility?" — Venus flytrap.
Maniac: "Have I stopped exploring?" — routing frozen into habit.

All signals derive from LayerSnapshot.router_scores [T, num_experts].
Calibration follows the same collect-then-freeze pattern as FastClock.

Spec reference: docs/chronomoe_spec_v03_triad.md
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from backends.adapter import LayerSnapshot


@dataclass
class LayerMonitorState:
    """Per-layer state tracked across steps."""
    pi_bar: Optional[List[float]] = None   # Habit EMA [E], eta=0.01
    ema_pi1: float = 0.0                   # EMA of top-1 weight, alpha=0.3
    prev_neff: Optional[float] = None      # Previous N_eff for Q_A velocity
    devil_consecutive: int = 0             # Consecutive devil trigger steps
    maniac_consecutive: int = 0            # Consecutive stagnation steps


@dataclass
class TriadScores:
    """Per-layer monitor scores and flags for one step."""
    angel_score: float
    devil_score: float
    maniac_score: float
    angel_flag: bool
    devil_flag: bool
    maniac_flag: bool
    # Raw signals for telemetry
    q_a: float       # N_eff velocity
    g: float         # Gap ratio
    c: float         # Commitment acceleration
    d_kl: float      # Divergence from habit
    neff: float      # Effective expert count


@dataclass
class TriadSummary:
    """Aggregated monitor output for one step (across all layers)."""
    angel_peak: float                          # Max angel score across layers
    devil_peak: float                          # Max devil score across layers
    maniac_peak: float                         # Max maniac score across layers
    angel_flag: bool                           # Any layer angel-flagged
    devil_flag: bool                           # Any layer devil-flagged
    maniac_flag: bool                          # Any layer maniac-flagged
    intervention: Optional[str] = None         # "devil"/"angel"/"maniac"/None
    intervention_layer: Optional[int] = None   # Layer index for intervention
    per_layer: Optional[List[TriadScores]] = None  # Full per-layer detail


class TriadMonitor:
    """Observer for Angel, Devil, Maniac routing monitors.

    Reads LayerSnapshot.router_scores, computes derived signals per layer,
    calibrates thresholds from warmup, and produces TriadSummary each step.

    Does not modify routing. Does not replace clocks or governor.

    Args:
        calibration_steps: Steps to collect samples before freezing thresholds.
            0 means use default thresholds (no calibration).
        dkl_settle_steps: Steps to skip before collecting D_KL samples.
            Pi_bar starts equal to pi, so early D_KL is artificially zero.
            Skipping the first N steps lets the habit diverge from current
            routing before we start measuring stagnation thresholds.
            Default 5. Set to 0 to collect from the start.
        habit_eta: EMA rate for pi_bar (routing habit). Slow = genuine habit.
        pi1_alpha: EMA rate for top-1 weight smoothing (C signal).
        devil_consecutive_threshold: Consecutive steps of devil condition
            before flag fires. Default 2 per spec.
        maniac_consecutive_threshold: Consecutive steps of low D_KL before
            maniac flag fires. Default 3.
        interventions_enabled: If True, select_intervention returns actions.
            Phase 1-3: False (observe only). Phase 4: True.
        store_per_layer: If True, include per-layer scores in TriadSummary.
    """

    def __init__(
        self,
        calibration_steps: int = 50,
        dkl_settle_steps: int = 5,
        habit_eta: float = 0.01,
        pi1_alpha: float = 0.3,
        devil_consecutive_threshold: int = 2,
        maniac_consecutive_threshold: int = 3,
        interventions_enabled: bool = False,
        store_per_layer: bool = False,
    ):
        self._calibration_steps = calibration_steps
        self._dkl_settle_steps = dkl_settle_steps
        self._habit_eta = habit_eta
        self._pi1_alpha = pi1_alpha
        self._devil_consecutive_threshold = devil_consecutive_threshold
        self._maniac_consecutive_threshold = maniac_consecutive_threshold
        self._interventions_enabled = interventions_enabled
        self._store_per_layer = store_per_layer

        # Per-layer monitor state (keyed by layer_id)
        self._layer_states: Dict[int, LayerMonitorState] = {}

        # Calibration: per-layer sample buffers
        self._calibrated = calibration_steps == 0
        self._tick_count = 0
        self._cal_q_a: Dict[int, List[float]] = {}     # Q_A samples per layer
        self._cal_g: Dict[int, List[float]] = {}        # G samples per layer
        self._cal_c: Dict[int, List[float]] = {}        # C samples per layer
        self._cal_d_kl: Dict[int, List[float]] = {}     # D_KL samples per layer

        # Calibrated thresholds per layer (set after calibration)
        # alpha[l] = p10(Q_A)  — rare negative Q_A = rare collapse
        # beta[l]  = p90(G)    — high gap = high dominance
        # gamma[l] = p90(C)    — high commitment acceleration
        # delta[l] = p10(D_KL) — low divergence = rare stagnation
        self._alpha: Dict[int, float] = {}
        self._beta: Dict[int, float] = {}
        self._gamma: Dict[int, float] = {}
        self._delta: Dict[int, float] = {}

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    def tick(self, snapshots: List[LayerSnapshot]) -> TriadSummary:
        """Process one step's routing snapshots through all three monitors.

        Called once per step, after forward execution, with the full set
        of layer snapshots. Same timing as clock ticks.

        Returns TriadSummary with peak scores, flags, and optional intervention.
        """
        self._tick_count += 1

        per_layer_scores: List[TriadScores] = []
        layer_ids: List[int] = []

        for snap in snapshots:
            lid = snap.layer_id
            layer_ids.append(lid)

            # Ensure per-layer state exists
            if lid not in self._layer_states:
                self._layer_states[lid] = LayerMonitorState()

            state = self._layer_states[lid]

            # Compute signals from router_scores [T, E]
            signals = self._compute_signals(snap, state, lid)
            q_a, g, c, d_kl, neff = signals

            # Collect calibration samples
            if not self._calibrated:
                self._cal_q_a.setdefault(lid, []).append(q_a)
                self._cal_g.setdefault(lid, []).append(g)
                self._cal_c.setdefault(lid, []).append(c)
                # D_KL settle: skip early samples where pi_bar ≈ pi
                # (D_KL starts at 0 because habit is initialized from first pi)
                if self._tick_count > self._dkl_settle_steps:
                    self._cal_d_kl.setdefault(lid, []).append(d_kl)

            # Compute scores and flags
            scores = self._score_layer(lid, state, q_a, g, c, d_kl, neff)
            per_layer_scores.append(scores)

        # Finalize calibration if threshold reached
        if not self._calibrated and self._tick_count >= self._calibration_steps:
            self._finalize_calibration()

        # Aggregate across layers
        angel_peak = max((s.angel_score for s in per_layer_scores), default=0.0)
        devil_peak = max((s.devil_score for s in per_layer_scores), default=0.0)
        maniac_peak = max((s.maniac_score for s in per_layer_scores), default=0.0)

        angel_flag = any(s.angel_flag for s in per_layer_scores)
        devil_flag = any(s.devil_flag for s in per_layer_scores)
        maniac_flag = any(s.maniac_flag for s in per_layer_scores)

        # Intervention selection: max ONE, priority Devil > Angel > Maniac
        intervention = None
        intervention_layer = None
        if self._interventions_enabled and self._calibrated:
            intervention, intervention_layer = self._select_intervention(
                per_layer_scores, layer_ids
            )

        return TriadSummary(
            angel_peak=angel_peak,
            devil_peak=devil_peak,
            maniac_peak=maniac_peak,
            angel_flag=angel_flag,
            devil_flag=devil_flag,
            maniac_flag=maniac_flag,
            intervention=intervention,
            intervention_layer=intervention_layer,
            per_layer=per_layer_scores if self._store_per_layer else None,
        )

    def _compute_signals(
        self,
        snap: LayerSnapshot,
        state: LayerMonitorState,
        layer_id: int,
    ) -> Tuple[float, float, float, float, float]:
        """Compute Q_A, G, C, D_KL, N_eff from one layer's routing.

        All signals derive from router_scores [T, E] (post-softmax).

        Returns (q_a, g, c, d_kl, neff).
        """
        # router_scores: [T, E] post-softmax distribution
        pi_tensor = snap.router_scores  # torch.Tensor [T, E]

        # Average routing across tokens -> pi [E]
        pi = pi_tensor.float().mean(dim=0)  # [E]
        pi_list = pi.tolist()
        num_experts = len(pi_list)

        # Ensure non-negative and normalize (softmax should already be, but safety)
        pi_sum = sum(pi_list)
        if pi_sum > 0:
            pi_list = [p / pi_sum for p in pi_list]

        # Sort for pi1, pi2
        sorted_pi = sorted(pi_list, reverse=True)
        pi1 = sorted_pi[0] if sorted_pi else 0.0
        pi2 = sorted_pi[1] if len(sorted_pi) > 1 else 0.0

        # N_eff = exp(H(pi)) where H = -sum(p * log(p))
        entropy = 0.0
        for p in pi_list:
            if p > 1e-10:
                entropy -= p * math.log(p)
        neff = math.exp(entropy)

        # Q_A: N_eff velocity
        q_a = 0.0
        if state.prev_neff is not None:
            q_a = neff - state.prev_neff
        state.prev_neff = neff

        # G: gap ratio = (pi1 - pi2) / pi1
        g = 0.0
        if pi1 > 1e-10:
            g = (pi1 - pi2) / pi1

        # C: commitment acceleration (EMA-smoothed pi1 velocity)
        # Use EMA of pi1, then velocity is delta of EMA
        prev_ema_pi1 = state.ema_pi1
        alpha = self._pi1_alpha
        state.ema_pi1 = (1 - alpha) * state.ema_pi1 + alpha * pi1
        c = state.ema_pi1 - prev_ema_pi1

        # D_KL: KL divergence from habit (pi || pi_bar)
        d_kl = 0.0
        if state.pi_bar is not None:
            for p, q in zip(pi_list, state.pi_bar):
                if p > 1e-10 and q > 1e-10:
                    d_kl += p * math.log(p / q)

        # Update habit EMA: pi_bar <- (1-eta)*pi_bar + eta*pi
        eta = self._habit_eta
        if state.pi_bar is None:
            state.pi_bar = list(pi_list)
        else:
            state.pi_bar = [
                (1 - eta) * q + eta * p
                for p, q in zip(pi_list, state.pi_bar)
            ]

        return (q_a, g, c, d_kl, neff)

    def _score_layer(
        self,
        layer_id: int,
        state: LayerMonitorState,
        q_a: float,
        g: float,
        c: float,
        d_kl: float,
        neff: float,
    ) -> TriadScores:
        """Compute Angel/Devil/Maniac scores and flags for one layer."""

        # Get calibrated thresholds (or defaults before calibration)
        alpha = self._alpha.get(layer_id, -0.5)   # Q_A threshold (p10, negative)
        beta = self._beta.get(layer_id, 0.8)      # G threshold (p90, high gap)
        gamma = self._gamma.get(layer_id, 0.01)    # C threshold (p90, high accel)
        delta = self._delta.get(layer_id, 0.01)    # D_KL threshold (p10, low div)

        # --- Angel: optionality collapsing into dominant expert ---
        # Score: max(0, -Q_A * G)
        angel_score = max(0.0, -q_a * g)
        # Flag: Q_A < alpha AND G > beta (instant, no consecutive requirement)
        angel_flag = (q_a < alpha) and (g > beta)

        # --- Devil: confidence rising while options fall ---
        # Score: max(0, C * (-Q_A))
        devil_score = max(0.0, c * (-q_a))
        # Flag: C > gamma AND Q_A < 0 for consecutive threshold steps
        devil_condition = (c > gamma) and (q_a < 0)
        if devil_condition:
            state.devil_consecutive += 1
        else:
            state.devil_consecutive = 0
        devil_flag = state.devil_consecutive >= self._devil_consecutive_threshold

        # --- Maniac: sustained stagnation ---
        # Score: max(0, (delta - D_KL)) * n_consecutive
        maniac_condition = d_kl < delta
        if maniac_condition:
            state.maniac_consecutive += 1
        else:
            state.maniac_consecutive = 0
        maniac_score = max(0.0, delta - d_kl) * state.maniac_consecutive
        maniac_flag = state.maniac_consecutive >= self._maniac_consecutive_threshold

        return TriadScores(
            angel_score=angel_score,
            devil_score=devil_score,
            maniac_score=maniac_score,
            angel_flag=angel_flag,
            devil_flag=devil_flag,
            maniac_flag=maniac_flag,
            q_a=q_a,
            g=g,
            c=c,
            d_kl=d_kl,
            neff=neff,
        )

    def _select_intervention(
        self,
        per_layer_scores: List[TriadScores],
        layer_ids: List[int],
    ) -> Tuple[Optional[str], Optional[int]]:
        """Select at most ONE intervention. Priority: Devil > Angel > Maniac.

        Returns (intervention_type, layer_id) or (None, None).
        """
        # Collect triggered layers with scores
        devil_triggers = [
            (lid, s.devil_score)
            for lid, s in zip(layer_ids, per_layer_scores)
            if s.devil_flag
        ]
        angel_triggers = [
            (lid, s.angel_score)
            for lid, s in zip(layer_ids, per_layer_scores)
            if s.angel_flag
        ]
        maniac_triggers = [
            (lid, s.maniac_score)
            for lid, s in zip(layer_ids, per_layer_scores)
            if s.maniac_flag
        ]

        if devil_triggers:
            target = max(devil_triggers, key=lambda x: x[1])
            return ("devil", target[0])
        elif angel_triggers:
            target = max(angel_triggers, key=lambda x: x[1])
            return ("angel", target[0])
        elif maniac_triggers:
            target = max(maniac_triggers, key=lambda x: x[1])
            return ("maniac", target[0])

        return (None, None)

    def _finalize_calibration(self):
        """Compute per-layer percentile thresholds from warmup samples."""
        for lid in self._cal_q_a:
            self._alpha[lid] = _percentile(self._cal_q_a[lid], 10.0)
            self._beta[lid] = _percentile(self._cal_g[lid], 90.0)
            self._gamma[lid] = _percentile(self._cal_c[lid], 90.0)
            self._delta[lid] = _percentile(self._cal_d_kl[lid], 10.0)

        self._calibrated = True

        # Free calibration buffers
        self._cal_q_a.clear()
        self._cal_g.clear()
        self._cal_c.clear()
        self._cal_d_kl.clear()

    def get_calibration_summary(self) -> Optional[Dict]:
        """Return calibrated thresholds for logging. None if not calibrated."""
        if not self._calibrated or not self._alpha:
            return None
        return {
            "alpha": dict(self._alpha),
            "beta": dict(self._beta),
            "gamma": dict(self._gamma),
            "delta": dict(self._delta),
        }

    def get_calibration_diagnostics(self) -> Optional[Dict]:
        """Return full calibration distribution stats. Only available during calibration.

        Returns per-layer {signal: {min, p10, median, p90, max, n_samples}}
        for Q_A, G, C, D_KL. Call before calibration finalizes to inspect.
        """
        if self._calibrated:
            return None  # Buffers already freed
        if not self._cal_q_a:
            return None

        result = {}
        for lid in sorted(self._cal_q_a.keys()):
            layer_stats = {}
            for name, samples in [
                ("Q_A", self._cal_q_a.get(lid, [])),
                ("G", self._cal_g.get(lid, [])),
                ("C", self._cal_c.get(lid, [])),
                ("D_KL", self._cal_d_kl.get(lid, [])),
            ]:
                if not samples:
                    continue
                layer_stats[name] = {
                    "min": min(samples),
                    "p10": _percentile(samples, 10.0),
                    "median": _percentile(samples, 50.0),
                    "p90": _percentile(samples, 90.0),
                    "max": max(samples),
                    "n": len(samples),
                }
            result[lid] = layer_stats
        return result


def _percentile(samples: List[float], pct: float) -> float:
    """Compute percentile using linear interpolation (same as FastClock)."""
    if not samples:
        return 0.0
    s = sorted(samples)
    n = len(s)
    idx = (pct / 100.0) * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac
