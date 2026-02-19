# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright 2026 Halcyon AI Research (jeff@halcyon.ie)
"""
Qwen MoE Bob Phase 2 Governed Experiment: cross-model validation on MLX.

Same protocol as OLMoE experiment, adapted for Qwen1.5-MoE-A2.7B on Apple Silicon:
- 60 experts, top-4, 24 MoE layers
- MLX backend with monkeypatch routing capture
- Same 5 prompt categories and geometry logging

Usage:
    source qwen_moe_mlx/bin/activate
    python3 experiments/qwen_governed.py --seed 15

    # Smoke test
    python3 experiments/qwen_governed.py --smoke --seed 99

    # Perturbation test
    python3 experiments/qwen_governed.py --seed 15 --perturb-at-step 60 --perturb-duration 30

    # No-scars ablation
    python3 experiments/qwen_governed.py --seed 15 --no-scars
"""

import sys
import os
import json
import time
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from bob_core.substrate import BobSubstrate
from bob_core.motifs import GateThresholds
from bob_core.ledgers import BobCore
from bob_core.medium_clock import MediumClock
from bob_core.fast_clock import FastClock
from bob_core.slow_clock import SlowClock
from bob_core.governor import BobGovernor
from bob_core.promotion import PromotionGate
from bob_core.monitors import TriadMonitor
from bob_core.conflict import ConflictRegister

from experiments.prompts import (
    PROMPTS, NUM_CATEGORIES, CATEGORY_NAMES, CATEGORY_TO_ID,
    compute_commit_then_violate,
)


# ─── Experiment Config ───────────────────────────────────────────────

@dataclass
class QwenExperimentConfig:
    model_name: str = "mlx-community/Qwen1.5-MoE-A2.7B-4bit"
    max_seq_len: int = 128
    warmup_steps: int = 50
    active_steps: int = 250
    class_block_size: int = 10
    success_multiplier: float = 1.2
    max_prompts_per_category: int = 0
    prompt_offset: int = 0
    monitor_cal_steps: int = 50       # Triad monitor calibration (decoupled from warmup)


# ─── Text Corpus (MLX version) ──────────────────────────────────────

class MLXTextCorpus:
    """Provides tokenized (input_ids_list, labels_list) pairs for MLX.

    Same category cycling as OLMoE TextCorpus, but returns plain
    Python lists of token IDs instead of torch tensors.
    """

    def __init__(self, tokenizer, config: QwenExperimentConfig, seed: int = 42):
        self.config = config
        self.rng = random.Random(seed)
        self.total_steps = config.warmup_steps + config.active_steps

        # Tokenize prompts
        offset = config.prompt_offset
        limit = config.max_prompts_per_category
        self._tokenized: Dict[int, List[List[int]]] = {}
        for cat_name in CATEGORY_NAMES:
            cat_id = CATEGORY_TO_ID[cat_name]
            prompts = PROMPTS[cat_name][offset:]
            if limit > 0:
                prompts = prompts[:limit]
            tokens = []
            for prompt in prompts:
                ids = tokenizer.encode(prompt)
                # Truncate to max_seq_len
                ids = ids[:config.max_seq_len]
                tokens.append(ids)
            self._tokenized[cat_id] = tokens

        # Per-category shuffled permutation (length = n, independent of total_steps)
        self._perm: Dict[int, List[int]] = {}
        for cat_id in range(NUM_CATEGORIES):
            n = len(self._tokenized[cat_id])
            perm = list(range(n))
            self.rng.shuffle(perm)
            self._perm[cat_id] = perm

        self._counters: Dict[int, int] = defaultdict(int)

    def get_batch(self, step: int) -> Tuple[int, List[int], List[int]]:
        """Return (context_class, input_ids, labels) for this step.

        Returns plain Python lists (not tensors) for MLX adapter.
        Cycles through a fixed permutation — ordering is seed-stable
        regardless of total_steps.
        """
        block_idx = step // self.config.class_block_size
        context_class = block_idx % NUM_CATEGORIES

        idx = self._counters[context_class]
        n = len(self._tokenized[context_class])
        prompt_idx = self._perm[context_class][idx % n]
        self._counters[context_class] = idx + 1

        input_ids = self._tokenized[context_class][prompt_idx]
        labels = list(input_ids)  # copy

        return context_class, input_ids, labels


# ─── Run Functions ───────────────────────────────────────────────────

def _make_run_header(
    config: QwenExperimentConfig, adapter, seed: int,
    enable_governor: bool, label: str,
    scars_disabled: bool = False,
    perturb_at_step: Optional[int] = None,
    perturb_duration: int = 30,
    perturb_routing: bool = False,
    debt_cap: float = 1.0,
) -> Dict:
    header = {
        "type": "header",
        "model_id": config.model_name,
        "num_experts": adapter.num_experts,
        "top_k": adapter.top_k,
        "num_moe_layers": adapter.num_layers,
        "backend": "mlx",
        "seed": seed,
        "label": label,
        "governor_enabled": enable_governor,
        "warmup_steps": config.warmup_steps,
        "active_steps": config.active_steps,
        "max_prompts_per_category": config.max_prompts_per_category,
        "prompt_offset": config.prompt_offset,
        "max_seq_len": config.max_seq_len,
        "success_multiplier": config.success_multiplier,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if scars_disabled:
        header["scars_disabled"] = True
    if perturb_at_step is not None:
        header["perturb_at_step"] = perturb_at_step
        header["perturb_duration"] = perturb_duration
        header["perturb_mode"] = "topk_collapse" if perturb_routing else "scars_off"
    if debt_cap < 1.0:
        header["debt_cap"] = debt_cap
    return header


def _resume_from_jsonl(jsonl_path: str) -> Tuple[int, List[Dict]]:
    traces = []
    if not os.path.exists(jsonl_path):
        return 0, traces
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("type") == "header":
                continue
            traces.append(record)
    if not traces:
        return 0, traces
    last_step = traces[-1]["step"]
    return last_step + 1, traces


def run_qwen_governed(
    adapter, tokenizer,
    config: QwenExperimentConfig,
    seed: int = 42,
    enable_governor: bool = True,
    label: str = "governed",
    jsonl_path: Optional[str] = None,
    no_scars: bool = False,
    perturb_at_step: Optional[int] = None,
    perturb_duration: int = 30,
    perturb_routing: bool = False,
    debt_cap: float = 1.0,
    enable_triad_monitors: bool = False,
    enable_conflict_index: bool = False,
) -> Dict:
    """Run Bob with or without governor on Qwen MoE via MLX."""
    # Reset calibration log flags (function attributes persist across calls)
    run_qwen_governed._fast_cal_logged = False
    run_qwen_governed._med_cal_logged = False
    run_qwen_governed._triad_cal_logged = False
    run_qwen_governed._triad_diag_logged = False

    corpus = MLXTextCorpus(tokenizer, config, seed=seed)

    # Build optional components
    bob_core = None
    governor = None
    fast_clock = None
    medium_clock = None
    slow_clock = None
    promotion_gate = None
    triad_monitor = None
    conflict_register = None

    if enable_governor:
        bob_core = BobCore(success_multiplier=config.success_multiplier, debt_cap=debt_cap)
        fast_clock = FastClock(
            ema_alpha=0.3,
            explore_threshold=0.4,  # Initial; overridden by calibration
            calibration_steps=50,   # Calibrate during warmup
            calibration_percentile=90.0,
        )
        medium_clock = MediumClock(
            ema_alpha=0.1,
            instability_threshold=0.5,
            calibration_steps=50,   # Calibrate during warmup
            calibration_percentile=90.0,
        )
        slow_clock = SlowClock(ema_alpha=0.02)

        # Triad monitors (optional, behind flags)
        # Monitor calibration is decoupled from warmup. Monitors need more
        # samples than clocks because derived signals (Q_A velocity, C
        # acceleration, D_KL from habit) need the habit EMA to settle.
        # Clamped so calibration always completes inside the run.
        total_steps = config.warmup_steps + config.active_steps
        monitor_cal_steps = min(config.monitor_cal_steps, total_steps - 1)
        if enable_triad_monitors:
            triad_monitor = TriadMonitor(
                calibration_steps=monitor_cal_steps,
            )
        if enable_conflict_index and enable_triad_monitors:
            conflict_register = ConflictRegister(
                buffer_size=50,
                calibration_steps=monitor_cal_steps,
                calibration_percentile=90.0,
            )

        governor = BobGovernor(
            bob_core, medium_clock,
            fast_clock=fast_clock,
            slow_clock=slow_clock,
            fast_threshold=0.5,
            medium_threshold=0.5,
            debt_threshold=0.7,
            conflict_register=conflict_register,
        )
        promotion_gate = PromotionGate(stability_window=20)

        if no_scars:
            bob_core.scars.enabled = False

    # Qwen 60E top-4 has higher routing diversity than OLMoE 64E top-8.
    # Max stability ≈ 0.20 (vs 0.60+ on OLMoE). Lower threshold to match.
    bob = BobSubstrate(
        adapter,
        gate_thresholds=GateThresholds(
            stability_min=0.15,
            debt_max=0.5,
            survival_min=0.7,
        ),
        warmup_steps=config.warmup_steps,
        governance_state="EQUILIBRIUM",
        bob_core=bob_core,
        governor=governor,
        fast_clock=fast_clock,
        medium_clock=medium_clock,
        slow_clock=slow_clock,
        promotion_gate=promotion_gate,
        triad_monitor=triad_monitor,
        conflict_register=conflict_register,
        stability_window=50,
        survival_half_life=200,
        success_multiplier=1.5,
    )

    total_steps = corpus.total_steps

    # Resume support
    resume_step = 0
    existing_traces = []
    if jsonl_path:
        resume_step, existing_traces = _resume_from_jsonl(jsonl_path)
        if resume_step > 0:
            print(f"    Resuming from step {resume_step} "
                  f"({len(existing_traces)} traces loaded)", flush=True)
            for s in range(resume_step):
                corpus.get_batch(s)
        else:
            header = _make_run_header(
                config, adapter, seed, enable_governor, label,
                scars_disabled=no_scars,
                perturb_at_step=perturb_at_step,
                perturb_duration=perturb_duration,
                perturb_routing=perturb_routing,
                debt_cap=debt_cap,
            )
            with open(jsonl_path, "w") as f:
                f.write(json.dumps(header) + "\n")

    jsonl_file = None
    if jsonl_path:
        jsonl_file = open(jsonl_path, "a")

    t_start = time.time()
    progress_interval = max(1, total_steps // 20)

    for step in range(resume_step, total_steps):
        # Perturbation toggle
        if perturb_at_step is not None:
            if perturb_routing:
                # Routing perturbation: collapse top-k to 1
                if step == perturb_at_step:
                    adapter.set_topk_override(1)
                elif step == perturb_at_step + perturb_duration:
                    adapter.set_topk_override(None)
            elif bob_core is not None and not no_scars:
                # Governance perturbation: disable scars
                if step == perturb_at_step:
                    bob_core.scars.enabled = False
                elif step == perturb_at_step + perturb_duration:
                    bob_core.scars.enabled = True

        task_class, input_ids, labels = corpus.get_batch(step)

        step_t0 = time.time()
        trace = bob.step(input_ids, labels, task_class, step)
        step_dt = time.time() - step_t0

        # Log fast clock calibration when it completes
        if (fast_clock is not None
                and fast_clock.calibrated
                and not getattr(run_qwen_governed, '_fast_cal_logged', False)):
            print(f"    [FastClock calibrated] explore={fast_clock.explore_threshold:.4f} "
                  f"governor={fast_clock.governor_threshold:.4f} "
                  f"(p{fast_clock._calibration_percentile:.0f} of {fast_clock._tick_count} samples)",
                  flush=True)
            if fast_clock._neff_floors_per_layer is not None:
                floors = fast_clock._neff_floors_per_layer
                print(f"    [Neff floors] n_layers={len(floors)} "
                      f"min={min(floors):.2f} max={max(floors):.2f} "
                      f"mean={sum(floors)/len(floors):.2f}",
                      flush=True)
            run_qwen_governed._fast_cal_logged = True

        # Log medium clock calibration when it completes
        if (medium_clock is not None
                and hasattr(medium_clock, 'calibrated')
                and medium_clock.calibrated
                and not getattr(run_qwen_governed, '_med_cal_logged', False)):
            print(f"    [MediumClock calibrated] governor={medium_clock.governor_threshold:.4f} "
                  f"(p{medium_clock._calibration_percentile:.0f} of {medium_clock._tick_count} samples)",
                  flush=True)
            run_qwen_governed._med_cal_logged = True

        # Log triad monitor calibration diagnostics on the last step before finalization
        if (triad_monitor is not None
                and not triad_monitor.calibrated
                and not getattr(run_qwen_governed, '_triad_diag_logged', False)
                and triad_monitor._tick_count >= triad_monitor._calibration_steps - 1):
            diag = triad_monitor.get_calibration_diagnostics()
            if diag:
                print(f"    [TriadMonitor pre-cal diagnostics] {len(diag)} layers, "
                      f"{triad_monitor._tick_count} samples:", flush=True)
                # Print summary for first, middle, last layer
                layer_ids = sorted(diag.keys())
                sample_layers = [layer_ids[0], layer_ids[len(layer_ids)//2], layer_ids[-1]]
                for lid in sample_layers:
                    stats = diag[lid]
                    for sig_name in ["Q_A", "G", "C", "D_KL"]:
                        s = stats.get(sig_name, {})
                        if s:
                            collapsed = (s["p10"] == 0 and s["p90"] == 0)
                            flag = " *** COLLAPSED ***" if collapsed else ""
                            print(f"      L{lid:>2} {sig_name:>4}: "
                                  f"min={s['min']:>9.6f} p10={s['p10']:>9.6f} "
                                  f"med={s['median']:>9.6f} p90={s['p90']:>9.6f} "
                                  f"max={s['max']:>9.6f} (n={s['n']}){flag}",
                                  flush=True)
            run_qwen_governed._triad_diag_logged = True

        # Log triad monitor calibration thresholds AFTER finalization
        if (triad_monitor is not None
                and triad_monitor.calibrated
                and not getattr(run_qwen_governed, '_triad_cal_logged', False)):
            cal = triad_monitor.get_calibration_summary()
            if cal:
                n_layers = len(cal["alpha"])
                print(f"    [TriadMonitor calibrated] {n_layers} layers", flush=True)
                # Sample layers
                layer_ids = sorted(cal["alpha"].keys())
                sample_layers = [layer_ids[0], layer_ids[len(layer_ids)//2], layer_ids[-1]]
                for lid in sample_layers:
                    print(f"      L{lid:>2}: "
                          f"alpha(Q_A p10)={cal['alpha'][lid]:>9.6f}  "
                          f"beta(G p90)={cal['beta'][lid]:.4f}  "
                          f"gamma(C p90)={cal['gamma'][lid]:>9.6f}  "
                          f"delta(D_KL p10)={cal['delta'][lid]:>9.6f}", flush=True)
            run_qwen_governed._triad_cal_logged = True

        # Write trace to JSONL
        trace_dict = trace.to_dict()
        trace_dict["wall_seconds"] = round(step_dt, 2)
        trace_dict["category"] = CATEGORY_NAMES[task_class]

        if perturb_at_step is not None:
            if step < perturb_at_step:
                trace_dict["perturb_phase"] = "before"
            elif step < perturb_at_step + perturb_duration:
                trace_dict["perturb_phase"] = "during"
            else:
                trace_dict["perturb_phase"] = "after"

        if jsonl_file:
            jsonl_file.write(json.dumps(trace_dict) + "\n")
            jsonl_file.flush()

        steps_done = step - resume_step + 1
        if steps_done == 1 or step % progress_interval == 0 or step == total_steps - 1:
            elapsed = time.time() - t_start
            rate = steps_done / elapsed if elapsed > 0 else 0
            remaining_steps = total_steps - step - 1
            eta_min = remaining_steps / rate / 60 if rate > 0 else 0
            print(f"    step {step}/{total_steps}  "
                  f"{step_dt:.1f}s/step  "
                  f"~{eta_min:.1f}m left  "
                  f"path={trace.path}  loss={trace.loss:.4f}",
                  flush=True)

    if jsonl_file:
        jsonl_file.close()

    # Combine and compute metrics
    new_traces = [t.to_dict() for t in bob.traces]
    raw_traces = existing_traces + new_traces

    active_traces = [
        t for t in raw_traces
        if config.warmup_steps <= t["step"] < total_steps
    ]

    cheap_count = sum(1 for t in active_traces if t["path"] == "cheap")
    total_active = len(active_traces)

    proposed_commits = 0
    blocked_commits = 0
    authorized_commits = 0
    authorized_losses = []
    commits_blocked = 0

    if enable_governor:
        commits_blocked = governor.blocks_count
        for t in raw_traces:
            gd = t.get("governor_decision")
            if gd is not None:
                proposed_commits += 1
                if gd == "allow":
                    authorized_commits += 1
                    authorized_losses.append(t["loss"])
                else:
                    blocked_commits += 1

    authorized_commit_quality = None
    if authorized_losses:
        authorized_commit_quality = {
            "count": len(authorized_losses),
            "avg_loss": round(sum(authorized_losses) / len(authorized_losses), 4),
            "min_loss": round(min(authorized_losses), 4),
            "max_loss": round(max(authorized_losses), 4),
        }

    ctv = compute_commit_then_violate(
        raw_traces, success_multiplier=config.success_multiplier, K=5,
    )
    ctv["sufficient_sample"] = authorized_commits >= 20

    all_regions_visited: Dict[int, set] = defaultdict(set)
    for t in raw_traces:
        region = tuple(sorted(t["expert_ids"]))
        if region:
            all_regions_visited[0].add(region)
    total_regions_visited = sum(len(v) for v in all_regions_visited.values())

    escalation_by_class: Dict[int, Dict] = {}
    for cc in range(NUM_CATEGORIES):
        class_traces = [t for t in active_traces if t["context_class"] == cc]
        if not class_traces:
            continue
        blocked = sum(
            1 for t in class_traces
            if t.get("governor_decision") in ("block", "escalate")
        )
        escalation_by_class[cc] = {
            "total": len(class_traces),
            "blocked": blocked,
            "rate": round(blocked / len(class_traces), 4) if class_traces else 0.0,
            "category": CATEGORY_NAMES[cc],
        }

    medium_state = None
    if medium_clock:
        s = medium_clock.state
        medium_state = {
            "churn_ema": round(s.churn_ema, 4),
            "flipflop_ema": round(s.flipflop_ema, 4),
            "outcome_var_ema": round(s.outcome_var_ema, 4),
            "activation": round(medium_clock.activation, 4),
        }
    fast_state = None
    if fast_clock:
        fs = fast_clock.state
        fast_state = {
            "churn_ema": round(fs.churn_ema, 4),
            "entropy_delta_ema": round(fs.entropy_delta_ema, 4),
            "loss_delta_ema": round(fs.loss_delta_ema, 4),
            "expert_flip_ema": round(fs.expert_flip_ema, 4),
            "activation": round(fast_clock.activation, 4),
        }
    slow_state = None
    if slow_clock:
        ss = slow_clock.state
        slow_state = {
            "scar_pressure_ema": round(ss.scar_pressure_ema, 4),
            "stability_ema": round(ss.stability_ema, 4),
            "activation": round(slow_clock.activation, 4),
        }

    scar_summary = None
    if bob_core:
        scars = bob_core.scars._scars
        scarred_regions = len(scars)
        scar_summary = {
            "total_scars": scarred_regions,
            "total_visited": total_regions_visited,
            "scar_saturation": round(
                scarred_regions / total_regions_visited, 4
            ) if total_regions_visited > 0 else 0.0,
            "total_debt": round(bob_core.scars.total_debt(total_steps), 4),
        }

    avg_loss = sum(t["loss"] for t in active_traces) / total_active if total_active else 0

    return {
        "label": label,
        "governor_enabled": enable_governor,
        "seed": seed,
        "total_steps": len(raw_traces),
        "active_cheap_count": cheap_count,
        "active_cheap_fraction": round(cheap_count / total_active, 4) if total_active else 0,
        "avg_loss": round(avg_loss, 4),
        "commits_blocked": commits_blocked,
        "proposed_commits": proposed_commits,
        "authorized_commits": authorized_commits,
        "blocked_commits": blocked_commits,
        "authorized_commit_quality": authorized_commit_quality,
        "commit_then_violate": ctv,
        "escalation_by_class": escalation_by_class,
        "unique_regions_visited": total_regions_visited,
        "fast_clock": fast_state,
        "medium_clock": medium_state,
        "slow_clock": slow_state,
        "scars": scar_summary,
        "model": config.model_name,
        "num_experts": adapter.num_experts,
        "top_k": adapter.top_k,
        "num_moe_layers": adapter.num_layers,
    }


def diff_report(baseline: Dict, governed: Dict) -> Dict:
    """Same diff report as OLMoE experiment."""
    report = {
        "baseline_seed": baseline["seed"],
        "governed_seed": governed["seed"],
        "same_seed": baseline["seed"] == governed["seed"],
        "model": governed.get("model", "unknown"),
        "num_experts": governed.get("num_experts"),
        "top_k": governed.get("top_k"),
        "num_moe_layers": governed.get("num_moe_layers"),
    }

    report["commits_blocked"] = governed["commits_blocked"]
    report["commits_blocked_pass"] = governed["commits_blocked"] > 0

    b_ctv = baseline["commit_then_violate"]["commit_then_violate_rate"]
    g_ctv = governed["commit_then_violate"]["commit_then_violate_rate"]
    report["baseline_ctv_rate"] = b_ctv
    report["governed_ctv_rate"] = g_ctv
    report["ctv_delta"] = round(g_ctv - b_ctv, 4)

    g_authorized = governed["authorized_commits"]
    report["governed_ctv_sufficient_sample"] = g_authorized >= 20
    if g_authorized >= 20:
        report["ctv_improved"] = g_ctv < b_ctv
    else:
        report["ctv_improved"] = None
        report["governed_ctv_note"] = (
            f"insufficient sample: {g_authorized} authorized commits (need >= 20)"
        )

    report["authorized_commit_quality"] = governed.get("authorized_commit_quality")
    report["proposed_commits"] = governed["proposed_commits"]
    report["authorized_commits"] = governed["authorized_commits"]
    report["blocked_commits"] = governed["blocked_commits"]
    report["blocked_commit_rate"] = round(
        governed["blocked_commits"] / governed["proposed_commits"], 4
    ) if governed["proposed_commits"] > 0 else 0.0

    report["baseline_avg_loss"] = baseline["avg_loss"]
    report["governed_avg_loss"] = governed["avg_loss"]
    report["loss_delta"] = round(governed["avg_loss"] - baseline["avg_loss"], 4)

    report["baseline_cheap_fraction"] = baseline["active_cheap_fraction"]
    report["governed_cheap_fraction"] = governed["active_cheap_fraction"]

    report["unique_regions_visited"] = governed["unique_regions_visited"]
    if governed["scars"]:
        report["scar_saturation"] = governed["scars"]["scar_saturation"]
        report["total_debt"] = governed["scars"]["total_debt"]

    report["escalation_by_class"] = governed["escalation_by_class"]
    report["scars"] = governed["scars"]
    report["fast_clock"] = governed["fast_clock"]
    report["medium_clock"] = governed["medium_clock"]
    report["slow_clock"] = governed["slow_clock"]

    return report


# ─── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen MoE Bob governed experiment (MLX)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="mlx-community/Qwen1.5-MoE-A2.7B-4bit")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--active", type=int, default=250)
    parser.add_argument("--prompts", type=int, default=0)
    parser.add_argument("--prompt-offset", type=int, default=0)
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: warmup=10, active=30, prompts=4")
    parser.add_argument("--no-scars", action="store_true")
    parser.add_argument("--perturb-at-step", type=int, default=None)
    parser.add_argument("--perturb-duration", type=int, default=30)
    parser.add_argument("--perturb-routing", action="store_true",
                        help="Routing perturbation: collapse top-k to 1 during window")
    parser.add_argument("--debt-cap", type=float, default=1.0,
                        help="Cap scar total_debt to prevent slow clock saturation (default 1.0)")
    parser.add_argument("--enable-triad-monitors", action="store_true",
                        help="Enable Angel/Devil/Maniac routing monitors (Phase 1-3)")
    parser.add_argument("--enable-conflict-index", action="store_true",
                        help="Enable conflict register + Mode A/B (requires --enable-triad-monitors)")
    parser.add_argument("--monitor-cal", type=int, default=50,
                        help="Triad monitor calibration steps (decoupled from warmup, default 50)")
    args = parser.parse_args()

    if args.smoke:
        args.warmup = 10
        args.active = 50
        args.prompts = 4
        args.monitor_cal = 30  # Longer than warmup: settle + collect

    # Auto-set perturbation start if routing perturbation requested
    if args.perturb_routing and args.perturb_at_step is None:
        args.perturb_at_step = args.warmup + 10  # Default: 10 steps after warmup

    config = QwenExperimentConfig(
        model_name=args.model,
        warmup_steps=args.warmup,
        active_steps=args.active,
        max_prompts_per_category=args.prompts,
        prompt_offset=args.prompt_offset,
        monitor_cal_steps=args.monitor_cal,
    )

    mode_label = "SMOKE TEST" if args.smoke else "GOVERNED EXPERIMENT"
    prompts_per_cat = args.prompts if args.prompts > 0 else len(PROMPTS[CATEGORY_NAMES[0]])
    total_prompts = prompts_per_cat * NUM_CATEGORIES

    print("=" * 70)
    print(f"QWEN MOE BOB PHASE 2: {mode_label}")
    print(f"  Model: {config.model_name}")
    print(f"  Steps: warmup={config.warmup_steps}, active={config.active_steps}")
    print(f"  Prompts: {prompts_per_cat}/category x {NUM_CATEGORIES} = {total_prompts}"
          + (f" (offset={args.prompt_offset})" if args.prompt_offset else ""))
    if args.no_scars:
        print(f"  Scars: DISABLED")
    if args.debt_cap < 1.0:
        print(f"  Debt cap: {args.debt_cap}")
    if args.perturb_at_step is not None:
        mode = "ROUTING (top-k→1)" if args.perturb_routing else "governance (scars off)"
        print(f"  Perturbation: {mode} at step {args.perturb_at_step} for {args.perturb_duration} steps")
    if args.enable_triad_monitors:
        monitors_str = "Angel/Devil/Maniac"
        if args.enable_conflict_index:
            monitors_str += " + Conflict Index"
        print(f"  Monitors: {monitors_str} (cal={config.monitor_cal_steps} steps)")
    print("=" * 70)

    # Load model via MLX
    from mlx_lm import load
    print(f"  Loading {config.model_name}...", flush=True)
    t0 = time.time()
    model, tokenizer = load(config.model_name)
    dt = time.time() - t0
    print(f"  Model loaded ({dt:.1f}s)", flush=True)

    # Create adapter
    from backends.mlx_adapter import MLXMoEAdapter
    adapter = MLXMoEAdapter(model, tokenizer)
    print(f"  MoE: {adapter.num_experts}E top-{adapter.top_k}, "
          f"{adapter.num_layers} layers (MLX)")

    seed = args.seed
    tag = "smoke" if args.smoke else "run"

    base_jsonl = f"qwen_{tag}_baseline_s{seed}.jsonl"
    gov_jsonl = f"qwen_{tag}_governed_s{seed}.jsonl"

    if args.no_scars:
        gov_jsonl = f"qwen_{tag}_governed_noscars_s{seed}.jsonl"

    print(f"\n--- Running baseline (no governor) ---")
    print(f"    JSONL: {base_jsonl}")
    baseline = run_qwen_governed(
        adapter, tokenizer, config,
        seed=seed, enable_governor=False, label="baseline",
        jsonl_path=base_jsonl,
    )
    print(f"  Cheap fraction: {baseline['active_cheap_fraction']*100:.1f}%")
    print(f"  Avg loss: {baseline['avg_loss']:.4f}")

    print(f"\n--- Running governed (with governor) ---")
    print(f"    JSONL: {gov_jsonl}")
    governed = run_qwen_governed(
        adapter, tokenizer, config,
        seed=seed, enable_governor=True, label="governed",
        jsonl_path=gov_jsonl,
        no_scars=args.no_scars,
        perturb_at_step=args.perturb_at_step,
        perturb_duration=args.perturb_duration,
        perturb_routing=args.perturb_routing,
        debt_cap=args.debt_cap,
        enable_triad_monitors=args.enable_triad_monitors,
        enable_conflict_index=args.enable_conflict_index,
    )
    print(f"  Cheap fraction: {governed['active_cheap_fraction']*100:.1f}%")
    print(f"  Avg loss: {governed['avg_loss']:.4f}")
    print(f"  Proposed: {governed['proposed_commits']}")
    print(f"  Authorized: {governed['authorized_commits']}")
    print(f"  Blocked: {governed['blocked_commits']}")

    if governed.get("authorized_commit_quality"):
        acq = governed["authorized_commit_quality"]
        print(f"  Authorized commit quality: avg={acq['avg_loss']:.4f} "
              f"[{acq['min_loss']:.4f}, {acq['max_loss']:.4f}]")

    if governed["scars"]:
        print(f"  Scars: {governed['scars']['total_scars']} regions, "
              f"sat={governed['scars']['scar_saturation']:.4f}, "
              f"debt={governed['scars']['total_debt']:.4f}")

    if governed["fast_clock"]:
        fc = governed["fast_clock"]
        print(f"  Fast clock:   act={fc['activation']:.4f}, "
              f"churn={fc['churn_ema']:.4f}, entropy_d={fc['entropy_delta_ema']:.4f}")
    if governed["medium_clock"]:
        mc = governed["medium_clock"]
        print(f"  Medium clock: act={mc['activation']:.4f}, "
              f"flipflop={mc['flipflop_ema']:.4f}, churn={mc['churn_ema']:.4f}")
    if governed["slow_clock"]:
        sc = governed["slow_clock"]
        print(f"  Slow clock:   act={sc['activation']:.4f}, "
              f"scar_p={sc['scar_pressure_ema']:.4f}, stab={sc['stability_ema']:.4f}")

    report = diff_report(baseline, governed)

    print(f"\n{'='*70}")
    print("DIFF REPORT")
    print(f"{'='*70}")
    print(f"  commits_blocked > 0:  "
          f"{'PASS' if report['commits_blocked_pass'] else 'FAIL'} "
          f"({report['commits_blocked']})")

    ctv_status = report.get("ctv_improved")
    if ctv_status is None:
        print(f"  CTV:                  INSUFFICIENT SAMPLE "
              f"(authorized={report['authorized_commits']}, need >= 20)")
    else:
        print(f"  CTV drops:            "
              f"{'PASS' if ctv_status else 'FAIL'} "
              f"(base={report['baseline_ctv_rate']:.4f}, "
              f"gov={report['governed_ctv_rate']:.4f}, "
              f"delta={report['ctv_delta']:+.4f})")

    print(f"  Loss not degraded:    "
          f"{'PASS' if report['loss_delta'] <= 0.05 else 'WARN'} "
          f"(delta={report['loss_delta']:+.4f})")

    if report.get("escalation_by_class"):
        print(f"\n  ESCALATION BY CATEGORY:")
        for cc, e in sorted(report["escalation_by_class"].items()):
            cat_name = e.get("category", f"class_{cc}")
            print(f"    {cat_name:>12}: "
                  f"{e['blocked']}/{e['total']} = {e['rate']:.1%}")

    report_path = f"qwen_{tag}_report_s{seed}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to {report_path}")

    print("\nDone.")
