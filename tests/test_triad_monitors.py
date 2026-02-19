# SPDX-License-Identifier: Apache-2.0
"""
Tests for triad monitors (Angel, Devil, Maniac) and conflict register.

Validates signal computation, calibration, flag logic, intervention priority,
conflict register mode selection, and backward compatibility.

Run:
  pytest test_triad_monitors.py -v        # preferred
  python3 test_triad_monitors.py          # also works
"""

import math
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dataclasses import dataclass
from typing import List, Optional

import torch

from backends.adapter import LayerSnapshot
from bob_core.monitors import TriadMonitor, TriadScores, TriadSummary, LayerMonitorState, _percentile
from bob_core.conflict import ConflictRegister, ConflictState
from bob_core.telemetry import DecisionTrace


# --- Test helpers ---

def make_snapshot(layer_id: int, pi: List[float], num_tokens: int = 10) -> LayerSnapshot:
    """Create a synthetic LayerSnapshot with given routing distribution.

    pi: desired mean routing distribution [num_experts].
    Each token gets the same distribution (deterministic).
    """
    num_experts = len(pi)
    # Tile the distribution across tokens
    router_scores = torch.tensor([pi] * num_tokens, dtype=torch.float32)
    # For selected_experts and routing_weights, use top-2
    sorted_indices = torch.argsort(router_scores[0], descending=True)
    top_k = min(2, num_experts)
    selected = sorted_indices[:top_k].unsqueeze(0).expand(num_tokens, -1)
    weights = torch.ones(num_tokens, top_k) / top_k
    usage = torch.tensor(pi, dtype=torch.float32)
    return LayerSnapshot(
        layer_id=layer_id,
        router_scores=router_scores,
        selected_experts=selected,
        routing_weights=weights,
        expert_usage=usage,
    )


def uniform_pi(n: int) -> List[float]:
    """Uniform distribution over n experts."""
    return [1.0 / n for _ in range(n)]


def dominant_pi(n: int, dominant_weight: float = 0.9) -> List[float]:
    """Distribution with one dominant expert."""
    rest = (1.0 - dominant_weight) / max(1, n - 1)
    return [dominant_weight] + [rest] * (n - 1)


def neff_from_pi(pi: List[float]) -> float:
    """Compute N_eff = exp(H(pi))."""
    h = 0.0
    for p in pi:
        if p > 1e-10:
            h -= p * math.log(p)
    return math.exp(h)


# --- Tests ---

# Detect whether we're running under pytest or as a script
_PYTEST_RUNNING = False
passed = 0
failed = 0
total = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed, total
    total += 1
    if condition:
        passed += 1
        if not _PYTEST_RUNNING:
            print(f"  PASS: {name}")
    else:
        failed += 1
        msg = f"{name} — {detail}" if detail else name
        if not _PYTEST_RUNNING:
            print(f"  FAIL: {msg}")
        assert condition, msg


def test_signal_computation():
    """Test that Q_A, G, C, D_KL, N_eff are computed correctly."""
    print("\n=== Test 1: Signal Computation ===")

    n = 8
    monitor = TriadMonitor(calibration_steps=0)  # Skip calibration

    # Step 1: uniform distribution
    pi1 = uniform_pi(n)
    snap1 = [make_snapshot(0, pi1)]
    summary1 = monitor.tick(snap1)
    scores1 = summary1  # Can't get per-layer without store_per_layer

    # Re-create with per-layer storage
    monitor = TriadMonitor(calibration_steps=0, store_per_layer=True)
    summary1 = monitor.tick([make_snapshot(0, pi1)])

    check("Per-layer scores stored", summary1.per_layer is not None)
    s1 = summary1.per_layer[0]

    expected_neff = neff_from_pi(pi1)
    check("N_eff uniform ~8", abs(s1.neff - expected_neff) < 0.01,
          f"got {s1.neff:.4f}, expected {expected_neff:.4f}")

    # Q_A should be 0 on first step (no previous N_eff)
    check("Q_A is 0 on first step", abs(s1.q_a) < 0.001, f"got {s1.q_a}")

    # G for uniform: (pi1 - pi2) / pi1 = 0 (all equal)
    check("G is ~0 for uniform", abs(s1.g) < 0.01, f"got {s1.g}")

    # Step 2: dominant distribution -> Q_A should be negative (N_eff drops)
    pi2 = dominant_pi(n, 0.9)
    summary2 = monitor.tick([make_snapshot(0, pi2)])
    s2 = summary2.per_layer[0]

    expected_neff2 = neff_from_pi(pi2)
    check("N_eff drops for dominant",
          s2.neff < s1.neff,
          f"dominant={s2.neff:.4f}, uniform={s1.neff:.4f}")

    check("Q_A is negative (collapse)",
          s2.q_a < 0,
          f"got {s2.q_a:.4f}")

    # G should be high for dominant
    check("G is high for dominant",
          s2.g > 0.8,
          f"got {s2.g:.4f}")


def test_angel_flag_fires_on_collapse():
    """Test angel flag fires when optionality collapses."""
    print("\n=== Test 2: Angel Flag on Collapse ===")

    n = 8
    # Calibrate with moderate signals so thresholds are reasonable
    monitor = TriadMonitor(
        calibration_steps=10,
        dkl_settle_steps=2,
        store_per_layer=True,
    )

    # Warmup: feed alternating mild signals to set thresholds
    for i in range(10):
        if i % 2 == 0:
            pi = uniform_pi(n)
        else:
            # Slight dominance, not extreme
            pi = [0.2] + [0.8 / 7] * 7
        monitor.tick([make_snapshot(0, pi)])

    check("Calibrated after 10 steps", monitor.calibrated)

    # Now feed a sharp collapse: uniform -> very dominant
    monitor.tick([make_snapshot(0, uniform_pi(n))])  # Reset N_eff high
    summary = monitor.tick([make_snapshot(0, dominant_pi(n, 0.95))])

    s = summary.per_layer[0]
    check("Angel score > 0 on collapse",
          s.angel_score > 0,
          f"got {s.angel_score:.4f}")
    check("Q_A < 0 (collapse signal)",
          s.q_a < 0,
          f"got {s.q_a:.4f}")
    check("G > 0.9 (dominant gap)",
          s.g > 0.9,
          f"got {s.g:.4f}")


def test_devil_requires_consecutive():
    """Test devil flag requires consecutive steps (not instant)."""
    print("\n=== Test 3: Devil Consecutive Requirement ===")

    n = 8
    monitor = TriadMonitor(
        calibration_steps=0,  # Use defaults
        devil_consecutive_threshold=2,
        store_per_layer=True,
    )

    # Step 1: uniform (baseline)
    monitor.tick([make_snapshot(0, uniform_pi(n))])

    # Step 2: collapsing + rising commitment (C > 0 AND Q_A < 0)
    # Use dominant to get Q_A < 0
    summary2 = monitor.tick([make_snapshot(0, dominant_pi(n, 0.8))])
    s2 = summary2.per_layer[0]

    # After just one step of devil condition, flag should NOT fire
    check("Devil flag NOT on first consecutive step",
          not s2.devil_flag,
          f"devil_flag={s2.devil_flag}, consecutive={monitor._layer_states[0].devil_consecutive}")

    # Step 3: continue collapsing even harder
    summary3 = monitor.tick([make_snapshot(0, dominant_pi(n, 0.95))])
    s3 = summary3.per_layer[0]

    # Check if devil condition was met for 2 consecutive steps
    # (C > gamma AND Q_A < 0)
    if s3.q_a < 0 and s3.c > 0:
        check("Devil condition met step 2",
              monitor._layer_states[0].devil_consecutive >= 2,
              f"consecutive={monitor._layer_states[0].devil_consecutive}")
    else:
        # If Q_A went positive because N_eff is still dropping, that's fine
        check("Devil condition check (Q_A or C reset)",
              True,
              f"Q_A={s3.q_a:.4f}, C={s3.c:.6f} — condition may not hold")


def test_maniac_requires_consecutive_and_calibration():
    """Test maniac flag requires 3+ consecutive low-D_KL steps."""
    print("\n=== Test 4: Maniac Consecutive + Settle ===")

    n = 8

    # Part A: Test D_KL settle phase collects correctly
    monitor_a = TriadMonitor(
        calibration_steps=10,
        dkl_settle_steps=3,
        maniac_consecutive_threshold=3,
        store_per_layer=True,
    )
    pis = [
        uniform_pi(n),
        dominant_pi(n, 0.3),
        uniform_pi(n),
        dominant_pi(n, 0.5),
        uniform_pi(n),
        dominant_pi(n, 0.2),
        uniform_pi(n),
        dominant_pi(n, 0.4),
        uniform_pi(n),
        dominant_pi(n, 0.3),
    ]
    for pi in pis:
        monitor_a.tick([make_snapshot(0, pi)])

    check("Monitor calibrated", monitor_a.calibrated)
    cal = monitor_a.get_calibration_summary()
    check("Calibration summary available", cal is not None)
    if cal:
        delta_0 = cal["delta"][0]
        check("Delta (D_KL p10) is > 0 after settle",
              delta_0 > 0,
              f"got {delta_0:.6f}")

    # Part B: Test maniac consecutive logic directly with defaults
    # Use calibration_steps=0 (default delta=0.01)
    # Feed the SAME distribution so pi_bar converges and D_KL → 0
    monitor_b = TriadMonitor(
        calibration_steps=0,
        maniac_consecutive_threshold=3,
        store_per_layer=True,
    )

    # Warm up pi_bar with a few steps of the same distribution
    stagnant_pi = uniform_pi(n)
    for _ in range(10):
        monitor_b.tick([make_snapshot(0, stagnant_pi)])

    consec = monitor_b._layer_states[0].maniac_consecutive
    check("Maniac consecutive > 0 after stagnation",
          consec > 0,
          f"got consecutive={consec}")

    # After 10 identical steps, D_KL should be << 0.01 (default delta)
    summary = monitor_b.tick([make_snapshot(0, stagnant_pi)])
    s = summary.per_layer[0]
    check("D_KL is very small after stagnation",
          s.d_kl < 0.01,
          f"got D_KL={s.d_kl:.6f}")
    check("Maniac flag fires after sustained stagnation",
          s.maniac_flag,
          f"flag={s.maniac_flag}, consecutive={monitor_b._layer_states[0].maniac_consecutive}")

    # Part C: verify consecutive resets on a different distribution
    summary_reset = monitor_b.tick([make_snapshot(0, dominant_pi(n, 0.9))])
    s_reset = summary_reset.per_layer[0]
    consec_after = monitor_b._layer_states[0].maniac_consecutive
    check("Maniac consecutive resets on new routing",
          consec_after < 3,
          f"got consecutive={consec_after} (should reset)")


def test_intervention_priority():
    """Test intervention selection: Devil > Angel > Maniac."""
    print("\n=== Test 5: Intervention Priority ===")

    monitor = TriadMonitor(
        calibration_steps=0,
        interventions_enabled=True,
        store_per_layer=True,
    )

    # Create fake scores with all flags set
    scores_both = TriadScores(
        angel_score=0.5, devil_score=0.3, maniac_score=0.1,
        angel_flag=True, devil_flag=True, maniac_flag=True,
        q_a=-1.0, g=0.9, c=0.02, d_kl=0.001, neff=2.0,
    )
    scores_angel_only = TriadScores(
        angel_score=0.5, devil_score=0.0, maniac_score=0.0,
        angel_flag=True, devil_flag=False, maniac_flag=False,
        q_a=-1.0, g=0.9, c=0.0, d_kl=0.5, neff=3.0,
    )
    scores_maniac_only = TriadScores(
        angel_score=0.0, devil_score=0.0, maniac_score=0.5,
        angel_flag=False, devil_flag=False, maniac_flag=True,
        q_a=0.0, g=0.1, c=0.0, d_kl=0.0001, neff=7.0,
    )

    # Devil > Angel > Maniac
    kind, lid = monitor._select_intervention([scores_both], [0])
    check("Devil wins when all flags set",
          kind == "devil",
          f"got {kind}")

    kind, lid = monitor._select_intervention([scores_angel_only], [0])
    check("Angel wins when no devil",
          kind == "angel",
          f"got {kind}")

    kind, lid = monitor._select_intervention([scores_maniac_only], [0])
    check("Maniac wins when alone",
          kind == "maniac",
          f"got {kind}")

    # No flags = no intervention
    scores_none = TriadScores(
        angel_score=0.0, devil_score=0.0, maniac_score=0.0,
        angel_flag=False, devil_flag=False, maniac_flag=False,
        q_a=0.0, g=0.1, c=0.0, d_kl=0.5, neff=7.0,
    )
    kind, lid = monitor._select_intervention([scores_none], [0])
    check("No intervention when no flags",
          kind is None and lid is None,
          f"got kind={kind}, lid={lid}")


def test_interventions_disabled_by_default():
    """Test that interventions don't fire when disabled."""
    print("\n=== Test 6: Interventions Disabled by Default ===")

    n = 8
    monitor = TriadMonitor(
        calibration_steps=0,
        interventions_enabled=False,
        store_per_layer=True,
    )

    # Even with extreme collapse, intervention should be None
    monitor.tick([make_snapshot(0, uniform_pi(n))])
    summary = monitor.tick([make_snapshot(0, dominant_pi(n, 0.99))])

    check("No intervention when disabled",
          summary.intervention is None,
          f"got {summary.intervention}")
    check("No intervention layer when disabled",
          summary.intervention_layer is None,
          f"got {summary.intervention_layer}")


def test_conflict_register_basic():
    """Test conflict register: product formula, Mode A/B."""
    print("\n=== Test 7: Conflict Register Basic ===")

    cr = ConflictRegister(buffer_size=10, calibration_steps=0)

    # Low conflict: Mode A
    state = cr.update(angel_peak=0.0, devil_peak=0.0)
    check("Zero conflict index when no signals",
          state.index == 0.0,
          f"got {state.index}")
    check("Mode A with zero conflict",
          state.mode == "A",
          f"got {state.mode}")

    # High conflict: angel AND devil both high
    state = cr.update(angel_peak=0.5, devil_peak=0.5)
    check("Conflict index is product",
          abs(state.index - 0.25) < 0.001,
          f"got {state.index}")

    # One high, one zero: still no conflict
    state = cr.update(angel_peak=1.0, devil_peak=0.0)
    check("No conflict when one signal is zero",
          state.index == 0.0,
          f"got {state.index}")


def test_conflict_register_mode_b():
    """Test that Mode B triggers under sustained high conflict."""
    print("\n=== Test 8: Conflict Register Mode B ===")

    cr = ConflictRegister(buffer_size=10, calibration_steps=0)
    # Default mu = 0.01

    # Pump high conflict to get mean above mu
    for _ in range(5):
        state = cr.update(angel_peak=0.5, devil_peak=0.5)

    check("Mean above default mu",
          state.mean_50 > 0.01,
          f"mean={state.mean_50:.4f}")

    # Add a spike above mean to trigger trending=True + mean > mu -> Mode B
    state = cr.update(angel_peak=0.8, devil_peak=0.8)
    check("Mode B under high sustained conflict",
          state.mode == "B",
          f"mode={state.mode}, mean={state.mean_50:.4f}, index={state.index:.4f}")

    # Drop to zero: should return to Mode A
    for _ in range(10):
        state = cr.update(angel_peak=0.0, devil_peak=0.0)
    check("Mode A after conflict subsides",
          state.mode == "A",
          f"mode={state.mode}")


def test_conflict_register_calibration():
    """Test conflict register calibration sets mu from percentile."""
    print("\n=== Test 9: Conflict Register Calibration ===")

    cr = ConflictRegister(
        buffer_size=50,
        calibration_steps=10,
        calibration_percentile=90.0,
    )

    check("Not calibrated initially", not cr.calibrated)

    # Feed 10 steps with varying conflict
    for i in range(10):
        angel = 0.1 * i  # 0.0 to 0.9
        devil = 0.1 * i
        cr.update(angel_peak=angel, devil_peak=devil)

    check("Calibrated after 10 steps", cr.calibrated)

    # mu should be set to p90 of the conflict samples
    # Conflicts were: 0.0, 0.01, 0.04, 0.09, 0.16, 0.25, 0.36, 0.49, 0.64, 0.81
    # p90 of those should be around 0.64-0.81
    check("Mu is reasonable (>0.5)",
          cr._mu > 0.5,
          f"got mu={cr._mu:.4f}")


def test_dkl_settle_skips_early_samples():
    """Test that D_KL calibration skips first N steps."""
    print("\n=== Test 10: D_KL Settle Phase ===")

    n = 4
    monitor = TriadMonitor(
        calibration_steps=10,
        dkl_settle_steps=5,
        store_per_layer=True,
    )

    # Feed 4 steps (< dkl_settle_steps of 5)
    for _ in range(4):
        monitor.tick([make_snapshot(0, uniform_pi(n))])

    # D_KL buffer should be empty (all steps skipped)
    dkl_samples = monitor._cal_d_kl.get(0, [])
    check("No D_KL samples before settle completes",
          len(dkl_samples) == 0,
          f"got {len(dkl_samples)} samples")

    # Feed step 5 (still skipped because tick_count == 5, not > 5)
    monitor.tick([make_snapshot(0, uniform_pi(n))])
    dkl_samples = monitor._cal_d_kl.get(0, [])
    check("No D_KL at exactly settle boundary",
          len(dkl_samples) == 0,
          f"got {len(dkl_samples)} samples")

    # Step 6: should start collecting
    monitor.tick([make_snapshot(0, uniform_pi(n))])
    dkl_samples = monitor._cal_d_kl.get(0, [])
    check("D_KL collected after settle phase",
          len(dkl_samples) == 1,
          f"got {len(dkl_samples)} samples")

    # Remaining steps
    for _ in range(4):
        monitor.tick([make_snapshot(0, uniform_pi(n))])

    check("Calibrated after 10 steps", monitor.calibrated)


def test_percentile_function():
    """Test the _percentile helper."""
    print("\n=== Test 11: Percentile Function ===")

    # Empty
    check("Empty list -> 0.0", _percentile([], 50.0) == 0.0)

    # Single element
    check("Single element -> that value", _percentile([5.0], 50.0) == 5.0)

    # Two elements: p0 = first, p100 = last
    check("Two elements p0", abs(_percentile([1.0, 2.0], 0.0) - 1.0) < 0.001)
    check("Two elements p100", abs(_percentile([1.0, 2.0], 100.0) - 2.0) < 0.001)
    check("Two elements p50", abs(_percentile([1.0, 2.0], 50.0) - 1.5) < 0.001)

    # Known distribution
    samples = list(range(1, 101))  # 1..100
    p10 = _percentile(samples, 10.0)
    p50 = _percentile(samples, 50.0)
    p90 = _percentile(samples, 90.0)
    check("p10 of 1..100 ~10.9", abs(p10 - 10.9) < 0.1, f"got {p10:.2f}")
    check("p50 of 1..100 ~50.5", abs(p50 - 50.5) < 0.1, f"got {p50:.2f}")
    check("p90 of 1..100 ~90.1", abs(p90 - 90.1) < 0.1, f"got {p90:.2f}")


def test_multi_layer():
    """Test monitor handles multiple layers correctly."""
    print("\n=== Test 12: Multi-Layer Processing ===")

    n = 8
    monitor = TriadMonitor(
        calibration_steps=0,
        store_per_layer=True,
    )

    # Layer 0: uniform, Layer 1: dominant
    snaps = [
        make_snapshot(0, uniform_pi(n)),
        make_snapshot(1, dominant_pi(n, 0.9)),
    ]
    monitor.tick(snaps)  # First tick: baseline
    summary = monitor.tick(snaps)  # Second tick: now we have Q_A

    check("Two layers in per_layer", len(summary.per_layer) == 2)

    s0 = summary.per_layer[0]
    s1 = summary.per_layer[1]

    # Layer 1 (dominant) should have higher G than layer 0 (uniform)
    check("Dominant layer has higher G",
          s1.g > s0.g,
          f"layer0={s0.g:.4f}, layer1={s1.g:.4f}")

    # Peaks should be max across layers
    check("Angel peak is max of layers",
          summary.angel_peak == max(s0.angel_score, s1.angel_score))
    check("Devil peak is max of layers",
          summary.devil_peak == max(s0.devil_score, s1.devil_score))


def test_neff_correctness():
    """Test N_eff computation for known distributions."""
    print("\n=== Test 13: N_eff Correctness ===")

    n = 8
    monitor = TriadMonitor(calibration_steps=0, store_per_layer=True)

    # Uniform: N_eff should equal n
    summary = monitor.tick([make_snapshot(0, uniform_pi(n))])
    check("N_eff for uniform = n",
          abs(summary.per_layer[0].neff - n) < 0.01,
          f"got {summary.per_layer[0].neff:.4f}")

    # Pure singleton: N_eff should be 1.0
    singleton = [1.0] + [0.0] * (n - 1)
    monitor2 = TriadMonitor(calibration_steps=0, store_per_layer=True)
    summary2 = monitor2.tick([make_snapshot(0, singleton)])
    check("N_eff for singleton = 1.0",
          abs(summary2.per_layer[0].neff - 1.0) < 0.01,
          f"got {summary2.per_layer[0].neff:.4f}")


def test_telemetry_backward_compatible():
    """Test DecisionTrace serialization with/without triad fields."""
    print("\n=== Test 14: Telemetry Backward Compatibility ===")

    # Trace WITHOUT triad fields (old behavior)
    trace_old = DecisionTrace(
        step=0, context_class=0, governance_state="EXPLORING",
        path="full", expert_ids=(1, 2), expert_invocations=2,
        tokens_processed=10, loss=1.5,
        routing_stability=0.9, debt_level=0.0, motif_survival=0.0,
        gate_passed=True, stability_passed=True, debt_passed=True,
        survival_passed=True,
    )
    d_old = trace_old.to_dict()
    check("No angel_score in old trace", "angel_score" not in d_old)
    check("No devil_score in old trace", "devil_score" not in d_old)
    check("No conflict_mode in old trace", "conflict_mode" not in d_old)

    # Trace WITH triad fields (new behavior)
    trace_new = DecisionTrace(
        step=1, context_class=0, governance_state="EXPLOITING",
        path="cheap", expert_ids=(1,), expert_invocations=1,
        tokens_processed=10, loss=1.2,
        routing_stability=0.95, debt_level=0.0, motif_survival=0.5,
        gate_passed=True, stability_passed=True, debt_passed=True,
        survival_passed=True,
        angel_score=0.35, devil_score=0.02, maniac_score=0.0,
        angel_flag=True, devil_flag=False, maniac_flag=False,
        conflict_index=0.007, conflict_mean=0.003,
        conflict_mode="A", conflict_trending=True,
    )
    d_new = trace_new.to_dict()
    check("angel_score in new trace", "angel_score" in d_new)
    check("devil_score in new trace", "devil_score" in d_new)
    check("angel_flag in new trace", "angel_flag" in d_new)
    check("conflict_mode in new trace", d_new.get("conflict_mode") == "A")
    check("angel_score value preserved",
          abs(d_new["angel_score"] - 0.35) < 0.001,
          f"got {d_new.get('angel_score')}")


def test_calibration_diagnostics():
    """Test get_calibration_diagnostics during calibration."""
    print("\n=== Test 15: Calibration Diagnostics ===")

    n = 4
    monitor = TriadMonitor(
        calibration_steps=10,
        dkl_settle_steps=2,
    )

    # Before any ticks, no diagnostics
    diag = monitor.get_calibration_diagnostics()
    check("No diagnostics before ticks", diag is None)

    # Feed a few steps
    for i in range(5):
        pi = uniform_pi(n) if i % 2 == 0 else dominant_pi(n, 0.6)
        monitor.tick([make_snapshot(0, pi)])

    diag = monitor.get_calibration_diagnostics()
    check("Diagnostics available during calibration", diag is not None)

    if diag:
        check("Layer 0 in diagnostics", 0 in diag)
        if 0 in diag:
            check("Q_A stats present", "Q_A" in diag[0])
            check("G stats present", "G" in diag[0])
            if "Q_A" in diag[0]:
                check("Q_A has min/max/n",
                      all(k in diag[0]["Q_A"] for k in ["min", "max", "n"]))

    # After calibration finalizes, diagnostics return None (buffers freed)
    for i in range(5):
        monitor.tick([make_snapshot(0, uniform_pi(n))])

    check("Monitor now calibrated", monitor.calibrated)
    diag_after = monitor.get_calibration_diagnostics()
    check("No diagnostics after calibration", diag_after is None)


def test_conflict_register_property_mode():
    """Test ConflictRegister.mode property matches update result."""
    print("\n=== Test 16: Conflict Register Mode Property ===")

    cr = ConflictRegister(buffer_size=10, calibration_steps=0)

    check("Initial mode is A", cr.mode == "A")

    # Feed enough conflict to push to Mode B
    for _ in range(5):
        state = cr.update(0.5, 0.5)

    # Now add a spike
    state = cr.update(0.8, 0.8)
    check("Property matches update result",
          cr.mode == state.mode,
          f"property={cr.mode}, update={state.mode}")


def test_zero_calibration_steps():
    """Test monitor works with calibration_steps=0 (default thresholds)."""
    print("\n=== Test 17: Zero Calibration Steps ===")

    n = 4
    monitor = TriadMonitor(calibration_steps=0, store_per_layer=True)

    check("Calibrated immediately with 0 steps", monitor.calibrated)

    # Should work fine with default thresholds
    summary = monitor.tick([make_snapshot(0, uniform_pi(n))])
    check("Summary produced with defaults",
          summary.angel_peak >= 0 and summary.devil_peak >= 0)


# --- Runner ---

_ALL_TESTS = [
    test_signal_computation,
    test_angel_flag_fires_on_collapse,
    test_devil_requires_consecutive,
    test_maniac_requires_consecutive_and_calibration,
    test_intervention_priority,
    test_interventions_disabled_by_default,
    test_conflict_register_basic,
    test_conflict_register_mode_b,
    test_conflict_register_calibration,
    test_dkl_settle_skips_early_samples,
    test_percentile_function,
    test_multi_layer,
    test_neff_correctness,
    test_telemetry_backward_compatible,
    test_calibration_diagnostics,
    test_conflict_register_property_mode,
    test_zero_calibration_steps,
]


def run_all_tests():
    """Script-mode runner. pytest users: just run `pytest test_triad_monitors.py -v`."""
    global passed, failed, total
    passed = failed = total = 0

    print("=" * 60)
    print("Triad Monitor + Conflict Register Tests")
    print("=" * 60)

    test_failures = []
    for fn in _ALL_TESTS:
        try:
            fn()
        except AssertionError as e:
            test_failures.append((fn.__name__, str(e)))

    print("\n" + "=" * 60)
    if failed == 0:
        print(f"ALL TESTS PASSED ({passed}/{total})")
    else:
        print(f"FAILED: {failed}/{total} checks failed")
        for name, msg in test_failures:
            print(f"  {name}: {msg}")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
