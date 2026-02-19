# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright 2026 Halcyon AI Research (jeff@halcyon.ie)
"""
Phase 8c: Memory Bias Validation Experiment.

Does pre-softmax logit bias from entity-specific routing signatures
do anything useful? This experiment is allowed to return "null".

5 conditions, same seed, same prompt sequence:
  C0  Baseline   - Governed, no graph, no basins
  C1  Toggle A   - Governed, graph, no basins  (must = C0)
  C2  B1         - Governed, graph, basins, scale=0.0  (must = C0)
  C3  B2         - Governed, graph, basins, scale=0.1  (the actual test)
  C4  B3         - Governed, graph, basins, scale=0.2  (hard ceiling)

Success is NOT "a probe moved."
Success is: a pre-declared primary metric improves for synthetic entities,
over Toggle A and Condition 0, across seeds, while geometry shows no
runaway or pathology.

Usage:
    source qwen_moe_mlx/bin/activate
    python3 experiments/memory_bias.py --seed 42
    python3 experiments/memory_bias.py --seed 42 --smoke
    python3 experiments/memory_bias.py --seeds 3            # seeds 42, 49, 56
    python3 experiments/memory_bias.py --seed 42 --conditions b1,b2
"""

import sys
import os
import json
import time
import random
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
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
from bob_core.graph import RelationalGraph
from bob_core.basins import BasinStore, compute_memory_bias

from experiments.prompts import (
    PROMPTS, NUM_CATEGORIES, CATEGORY_NAMES, CATEGORY_TO_ID,
    compute_commit_then_violate,
)


# ─── Entity Prompts ───────────────────────────────────────────────
# 10 semantic (Jeff, Paula, ALS) + 10 synthetic (Zorblax, Vexnari, Krenthar)
# Interleaved: even indices = semantic, odd indices = synthetic.
# This guarantees both types appear in every entity block.

ENTITY_PROMPTS = [
    # 0: semantic
    "Jeff walked into the ALS office on Monday morning and checked the latest research results on the shared dashboard.",
    # 1: synthetic
    "Zorblax reviewed the latest findings from the Krenthar Institute and prepared a summary for the consortium meeting.",
    # 2: semantic
    "Paula and Jeff discussed the implications of the new ALS funding round over dinner at their usual restaurant.",
    # 3: synthetic
    "Vexnari sent a memo to Zorblax outlining the proposed changes to the Krenthar Institute operating procedures.",
    # 4: semantic
    "The ALS research team, led by Jeff, published preliminary findings on motor neuron degeneration pathways.",
    # 5: synthetic
    "The Krenthar Institute board voted to expand Zorblax's research division, allocating additional computational resources.",
    # 6: semantic
    "Paula reviewed Jeff's grant application for ALS research funding and suggested several improvements to the methodology section.",
    # 7: synthetic
    "Zorblax and Vexnari co-authored a paper on emergent routing dynamics, presented at the Krenthar symposium.",
    # 8: semantic
    "Jeff presented the quarterly ALS progress report to the advisory board, highlighting three breakthrough observations.",
    # 9: synthetic
    "Vexnari's analysis revealed unexpected patterns in the data that Zorblax had collected from the Krenthar field stations.",
    # 10: semantic
    "Paula organised a fundraising event for ALS awareness, with Jeff coordinating the scientific outreach programme.",
    # 11: synthetic
    "The Krenthar Institute released its annual report, crediting Zorblax and Vexnari for significant methodological advances.",
    # 12: semantic
    "Jeff and Paula spent the weekend reviewing literature on ALS biomarkers, preparing for the upcoming conference.",
    # 13: synthetic
    "Zorblax proposed a new experimental protocol to the Krenthar Institute ethics committee for approval.",
    # 14: semantic
    "The ALS foundation acknowledged Jeff's contributions to understanding disease progression mechanisms.",
    # 15: synthetic
    "Vexnari calibrated the instruments at the Krenthar Institute while Zorblax supervised the data collection process.",
    # 16: semantic
    "Paula helped Jeff draft the executive summary for the ALS clinical trial proposal.",
    # 17: synthetic
    "Zorblax received notification that the Krenthar Institute had approved the expanded research mandate.",
    # 18: semantic
    "Jeff attended the international ALS conference and networked with researchers from twelve different countries.",
    # 19: synthetic
    "Vexnari and Zorblax debated the merits of their competing hypotheses during the weekly Krenthar research seminar.",
]

# Entity classification: which entities are in each prompt
ENTITY_ANNOTATIONS = {
    0:  {"type": "semantic",  "entities": ["jeff", "als"]},
    1:  {"type": "synthetic", "entities": ["zorblax", "krenthar institute"]},
    2:  {"type": "semantic",  "entities": ["paula", "jeff", "als"]},
    3:  {"type": "synthetic", "entities": ["vexnari", "zorblax", "krenthar institute"]},
    4:  {"type": "semantic",  "entities": ["als", "jeff"]},
    5:  {"type": "synthetic", "entities": ["krenthar institute", "zorblax"]},
    6:  {"type": "semantic",  "entities": ["paula", "jeff", "als"]},
    7:  {"type": "synthetic", "entities": ["zorblax", "vexnari", "krenthar"]},
    8:  {"type": "semantic",  "entities": ["jeff", "als"]},
    9:  {"type": "synthetic", "entities": ["vexnari", "zorblax", "krenthar"]},
    10: {"type": "semantic",  "entities": ["paula", "als", "jeff"]},
    11: {"type": "synthetic", "entities": ["krenthar institute", "zorblax", "vexnari"]},
    12: {"type": "semantic",  "entities": ["jeff", "paula", "als"]},
    13: {"type": "synthetic", "entities": ["zorblax", "krenthar institute"]},
    14: {"type": "semantic",  "entities": ["als", "jeff"]},
    15: {"type": "synthetic", "entities": ["vexnari", "krenthar institute", "zorblax"]},
    16: {"type": "semantic",  "entities": ["paula", "jeff", "als"]},
    17: {"type": "synthetic", "entities": ["zorblax", "krenthar institute"]},
    18: {"type": "semantic",  "entities": ["jeff", "als"]},
    19: {"type": "synthetic", "entities": ["vexnari", "zorblax", "krenthar"]},
}


# ─── Graph + Basin Population ──────────────────────────────────────

def build_memory_graph() -> RelationalGraph:
    """Build the entity graph for memory experiment."""
    g = RelationalGraph()
    # Semantic entities
    g.resolve_or_create_node("Jeff", node_type="person")
    g.resolve_or_create_node("Paula", node_type="person")
    g.resolve_or_create_node("ALS", node_type="organisation")
    g.add_triple("Jeff", "spouse", "Paula")
    g.add_triple("Jeff", "works_at", "ALS")

    # Synthetic entities
    g.resolve_or_create_node("Zorblax", node_type="concept")
    g.resolve_or_create_node("Vexnari", node_type="person")
    g.resolve_or_create_node("Krenthar Institute", node_type="organisation")
    g.add_triple("Zorblax", "member_of", "Krenthar Institute")
    g.add_triple("Zorblax", "colleague", "Vexnari")
    return g


def build_basin_store(
    graph: RelationalGraph,
    num_layers: int,
    num_experts: int,
    scale: float,
) -> BasinStore:
    """Build pre-populated basin store with known affinities.

    Basin affinities are PRE-POPULATED, not learned. This is deliberate:
    we're testing whether the plumbing works, not whether basins can
    be learned from data.

    Uses graph to look up node IDs by name — never hardcode IDs.
    """
    store = BasinStore(bias_scale=scale)

    def _nid(name: str) -> str:
        candidates = graph.alias_table.lookup(name)
        assert len(candidates) == 1, f"Expected exactly 1 node for '{name}', got {candidates}"
        return candidates[0]

    # Jeff: mild preference for expert 2 at layers 0, 4, 8
    jeff_basin = store.get_or_create_basin(_nid("Jeff"), num_layers, num_experts)
    jeff_basin.strength = 0.5
    for lid in [0, 4, 8]:
        if lid < num_layers:
            jeff_basin.bias_vector[lid][min(2, num_experts - 1)] = 0.6

    # Paula: mild preference for expert 5 at layers 0, 4, 8
    paula_basin = store.get_or_create_basin(_nid("Paula"), num_layers, num_experts)
    paula_basin.strength = 0.5
    for lid in [0, 4, 8]:
        if lid < num_layers:
            paula_basin.bias_vector[lid][min(5, num_experts - 1)] = 0.6

    # Zorblax: STRONG preference for expert 0 at ALL layers (primary test signal)
    zorblax_basin = store.get_or_create_basin(_nid("Zorblax"), num_layers, num_experts)
    zorblax_basin.strength = 0.8
    for lid in range(num_layers):
        zorblax_basin.bias_vector[lid][0] = 1.0

    # Vexnari: moderate preference for expert 10 at layers 0-3
    vexnari_basin = store.get_or_create_basin(_nid("Vexnari"), num_layers, num_experts)
    vexnari_basin.strength = 0.6
    for lid in range(min(4, num_layers)):
        vexnari_basin.bias_vector[lid][min(10, num_experts - 1)] = 0.8

    # Krenthar Institute: NO basin (entity exists in graph but no routing preference)
    # ALS: NO basin (entity exists in graph but no routing preference)

    return store


# ─── Experiment Config ─────────────────────────────────────────────

@dataclass
class MemoryExperimentConfig:
    model_name: str = "mlx-community/Qwen1.5-MoE-A2.7B-4bit"
    max_seq_len: int = 128
    warmup_steps: int = 50
    active_steps: int = 250
    class_block_size: int = 10
    success_multiplier: float = 1.2


# ─── Condition Dataclass ──────────────────────────────────────────

@dataclass
class Condition:
    name: str
    label: str
    enable_graph: bool
    enable_basins: bool
    bias_scale: float  # Only used if enable_basins


ALL_CONDITIONS = [
    Condition("baseline", "C0: Baseline (governed, no memory)", False, False, 0.0),
    Condition("toggle_a", "C1: Toggle A (graph only)", True, False, 0.0),
    Condition("b1", "C2: B1 (scale=0.0)", True, True, 0.0),
    Condition("b2", "C3: B2 (scale=0.1)", True, True, 0.1),
    Condition("b3", "C4: B3 (scale=0.2)", True, True, 0.2),
]


# ─── Text Corpus (with entity tokens) ─────────────────────────────

# Extend category system to include entity prompts
MEMORY_CATEGORY_NAMES = CATEGORY_NAMES + ["entity"]
MEMORY_NUM_CATEGORIES = len(MEMORY_CATEGORY_NAMES)
MEMORY_CATEGORY_TO_ID = {name: i for i, name in enumerate(MEMORY_CATEGORY_NAMES)}


class MemoryTextCorpus:
    """MLXTextCorpus extended with entity tokens and raw text.

    Adds 6th category "entity" containing ENTITY_PROMPTS.
    Returns (context_class, input_ids, labels, entity_tokens, raw_text, entity_annotation).
    """

    def __init__(self, tokenizer, config: MemoryExperimentConfig, seed: int = 42):
        self.config = config
        self.rng = random.Random(seed)

        # Tokenize standard prompts (5 categories from OLMoE)
        self._tokenized: Dict[int, List[List[int]]] = {}
        self._raw_texts: Dict[int, List[str]] = {}
        for cat_name in CATEGORY_NAMES:
            cat_id = MEMORY_CATEGORY_TO_ID[cat_name]
            prompts = PROMPTS[cat_name]
            tokens = []
            texts = []
            for prompt in prompts:
                ids = tokenizer.encode(prompt)
                ids = ids[:config.max_seq_len]
                tokens.append(ids)
                texts.append(prompt)
            self._tokenized[cat_id] = tokens
            self._raw_texts[cat_id] = texts

        # Tokenize entity prompts (6th category)
        entity_cat_id = MEMORY_CATEGORY_TO_ID["entity"]
        tokens = []
        texts = []
        for prompt in ENTITY_PROMPTS:
            ids = tokenizer.encode(prompt)
            ids = ids[:config.max_seq_len]
            tokens.append(ids)
            texts.append(prompt)
        self._tokenized[entity_cat_id] = tokens
        self._raw_texts[entity_cat_id] = texts

        # Per-category shuffled permutation
        self._perm: Dict[int, List[int]] = {}
        for cat_id in range(MEMORY_NUM_CATEGORIES):
            n = len(self._tokenized[cat_id])
            perm = list(range(n))
            self.rng.shuffle(perm)
            self._perm[cat_id] = perm

        self._counters: Dict[int, int] = defaultdict(int)

    def get_batch(self, step: int):
        """Return (context_class, input_ids, labels, entity_tokens, raw_text, annotation).

        annotation is None for non-entity prompts, dict for entity prompts.
        """
        block_idx = step // self.config.class_block_size
        context_class = block_idx % MEMORY_NUM_CATEGORIES

        idx = self._counters[context_class]
        n = len(self._tokenized[context_class])
        prompt_idx = self._perm[context_class][idx % n]
        self._counters[context_class] = idx + 1

        input_ids = self._tokenized[context_class][prompt_idx]
        labels = list(input_ids)
        raw_text = self._raw_texts[context_class][prompt_idx]

        # Annotation (only for entity category)
        annotation = None
        if context_class == MEMORY_CATEGORY_TO_ID["entity"]:
            annotation = ENTITY_ANNOTATIONS.get(prompt_idx)

        # Entity tokens: use explicit entity names from annotation for reliable
        # alias matching. Raw text splitting produces "zorblax's" and splits
        # "krenthar institute" into two tokens — neither matches the alias table.
        # Phase 8c tests plumbing, not NLP matching quality.
        if annotation is not None:
            entity_tokens = list(annotation["entities"])
        else:
            entity_tokens = []

        return context_class, input_ids, labels, entity_tokens, raw_text, annotation


# ─── Run Function ─────────────────────────────────────────────────

def run_condition(
    adapter,
    tokenizer,
    config: MemoryExperimentConfig,
    condition: Condition,
    seed: int,
    jsonl_path: str,
) -> Dict:
    """Run a single condition and return results dict."""
    random.seed(seed)

    total_steps = config.warmup_steps + config.active_steps
    corpus = MemoryTextCorpus(tokenizer, config, seed)

    # Build graph + basins per condition
    graph = build_memory_graph() if condition.enable_graph else None
    basin_store = None
    if condition.enable_basins:
        assert graph is not None, "Basins require a graph for node ID lookup"
        basin_store = build_basin_store(
            graph, adapter.num_layers, adapter.num_experts, condition.bias_scale,
        )

    # Build standard Bob components (identical across conditions)
    bob_core = BobCore(debt_cap=1.0)
    fast_clock = FastClock()
    medium_clock = MediumClock()
    slow_clock = SlowClock()
    promotion_gate = PromotionGate()
    governor = BobGovernor(
        bob_core, medium_clock,
        fast_clock=fast_clock,
        slow_clock=slow_clock,
        fast_threshold=0.5,
        medium_threshold=0.5,
        debt_threshold=0.7,
    )

    thresholds = GateThresholds(
        stability_min=0.15,
        debt_max=0.5,
        survival_min=0.7,
    )

    bob = BobSubstrate(
        adapter,
        gate_thresholds=thresholds,
        warmup_steps=config.warmup_steps,
        bob_core=bob_core,
        governor=governor,
        fast_clock=fast_clock,
        medium_clock=medium_clock,
        slow_clock=slow_clock,
        promotion_gate=promotion_gate,
        memory_graph=graph,
        basin_store=basin_store,
        memory_bias_scale=condition.bias_scale if condition.enable_basins else 0.1,
    )

    print(f"  Running: {condition.label}", flush=True)
    print(f"    Steps: warmup={config.warmup_steps}, active={config.active_steps}", flush=True)
    print(f"    JSONL: {jsonl_path}", flush=True)

    jsonl_file = open(jsonl_path, "w")
    t_start = time.time()
    progress_interval = 50

    all_traces = []

    for step in range(total_steps):
        ctx_class, input_ids, labels, entity_tokens, raw_text, annotation = corpus.get_batch(step)

        step_t0 = time.time()
        trace = bob.step(input_ids, labels, ctx_class, step, entity_tokens=entity_tokens)
        step_dt = time.time() - step_t0

        # Enrich trace with experiment metadata
        trace_dict = trace.to_dict()
        trace_dict["wall_seconds"] = round(step_dt, 2)
        trace_dict["category"] = MEMORY_CATEGORY_NAMES[ctx_class]
        trace_dict["condition"] = condition.name
        trace_dict["has_entity_tokens"] = annotation is not None
        trace_dict["entity_type"] = annotation["type"] if annotation else "none"
        trace_dict["entity_names"] = annotation["entities"] if annotation else []

        all_traces.append(trace_dict)

        jsonl_file.write(json.dumps(trace_dict) + "\n")
        jsonl_file.flush()

        steps_done = step + 1
        if steps_done == 1 or step % progress_interval == 0 or step == total_steps - 1:
            elapsed = time.time() - t_start
            rate = steps_done / elapsed if elapsed > 0 else 0
            remaining = total_steps - step - 1
            eta_min = remaining / rate / 60 if rate > 0 else 0
            print(f"    step {step}/{total_steps}  "
                  f"{step_dt:.1f}s/step  "
                  f"~{eta_min:.1f}m left  "
                  f"path={trace.path}  loss={trace.loss:.4f}",
                  flush=True)

    jsonl_file.close()

    # Compute summary metrics
    active_traces = [t for t in all_traces if t["step"] >= config.warmup_steps]
    total_active = len(active_traces)

    cheap_count = sum(1 for t in active_traces if t["path"] == "cheap")
    avg_loss = sum(t["loss"] for t in active_traces) / total_active if total_active else 0

    # Entity-type breakdown
    entity_losses = {"semantic": [], "synthetic": [], "none": []}
    entity_expert_ids = {"semantic": [], "synthetic": [], "none": []}
    for t in active_traces:
        etype = t["entity_type"]
        entity_losses[etype].append(t["loss"])
        entity_expert_ids[etype].append(tuple(t["expert_ids"]))

    entity_summary = {}
    for etype in ["semantic", "synthetic", "none"]:
        losses = entity_losses[etype]
        if losses:
            entity_summary[etype] = {
                "count": len(losses),
                "avg_loss": round(sum(losses) / len(losses), 6),
                "min_loss": round(min(losses), 6),
                "max_loss": round(max(losses), 6),
            }

    # Memory bias audibility (only for conditions with basins)
    # Note: bias_to_logit_ratio requires logit_std from adapter (not wired).
    # Use memory_bias_max as the audibility proxy instead.
    bias_stats = {"audible_steps": 0, "total_entity_steps": 0, "max_biases": []}
    for t in active_traces:
        if t["entity_type"] != "none":
            bias_stats["total_entity_steps"] += 1
            bias_max = t.get("memory_bias_max")
            if bias_max is not None and bias_max > 0:
                bias_stats["audible_steps"] += 1
                bias_stats["max_biases"].append(bias_max)

    bias_audibility = None
    if bias_stats["max_biases"]:
        vals = sorted(bias_stats["max_biases"])
        bias_audibility = {
            "audible_fraction": round(bias_stats["audible_steps"] / max(1, bias_stats["total_entity_steps"]), 4),
            "min_bias_max": round(vals[0], 6),
            "median_bias_max": round(vals[len(vals) // 2], 6),
            "max_bias_max": round(vals[-1], 6),
            "mean_bias_max": round(sum(vals) / len(vals), 6),
        }

    # Scar summary
    scar_summary = None
    if bob_core:
        scars = bob_core.scars._scars
        scarred_regions = len(scars)
        all_regions: Set[tuple] = set()
        for t in all_traces:
            region = tuple(sorted(t["expert_ids"]))
            if region:
                all_regions.add(region)
        total_regions = len(all_regions)
        scar_summary = {
            "total_scars": scarred_regions,
            "total_visited": total_regions,
            "scar_saturation": round(
                scarred_regions / total_regions, 4
            ) if total_regions > 0 else 0.0,
            "total_debt": round(bob_core.scars.total_debt(total_steps), 4),
        }

    elapsed = time.time() - t_start
    print(f"    Done: {elapsed:.0f}s  "
          f"avg_loss={avg_loss:.4f}  "
          f"cheap={cheap_count}/{total_active}={cheap_count/total_active*100:.1f}%",
          flush=True)

    return {
        "condition": condition.name,
        "label": condition.label,
        "seed": seed,
        "total_steps": total_steps,
        "active_steps": total_active,
        "avg_loss": round(avg_loss, 6),
        "cheap_count": cheap_count,
        "cheap_fraction": round(cheap_count / total_active, 4) if total_active else 0,
        "entity_summary": entity_summary,
        "bias_audibility": bias_audibility,
        "scars": scar_summary,
        "wall_seconds": round(elapsed, 1),
        "all_traces": all_traces,  # kept in memory for gate checks
    }


# ─── Verification Gates ──────────────────────────────────────────

def gate_1_b1_equals_c0(c0_result: Dict, b1_result: Dict) -> Dict:
    """Gate 1: B1 (scale=0.0) must produce IDENTICAL traces to C0.

    Not "close". Identical within floating tolerance.
    If this fails, STOP — every effect after is contaminated.
    """
    c0_traces = [t for t in c0_result["all_traces"] if t["step"] >= 0]
    b1_traces = [t for t in b1_result["all_traces"] if t["step"] >= 0]

    if len(c0_traces) != len(b1_traces):
        return {"passed": False, "reason": f"trace count mismatch: {len(c0_traces)} vs {len(b1_traces)}"}

    mismatches = []
    for i, (c0t, b1t) in enumerate(zip(c0_traces, b1_traces)):
        # Loss must be identical within 1e-6
        if abs(c0t["loss"] - b1t["loss"]) > 1e-6:
            mismatches.append({"step": c0t["step"], "field": "loss",
                               "c0": c0t["loss"], "b1": b1t["loss"]})
        # Expert IDs must be identical
        if tuple(c0t["expert_ids"]) != tuple(b1t["expert_ids"]):
            mismatches.append({"step": c0t["step"], "field": "expert_ids",
                               "c0": c0t["expert_ids"], "b1": b1t["expert_ids"]})
        # Path must be identical
        if c0t["path"] != b1t["path"]:
            mismatches.append({"step": c0t["step"], "field": "path",
                               "c0": c0t["path"], "b1": b1t["path"]})
        # Governor decision must be identical
        if c0t.get("governor_decision") != b1t.get("governor_decision"):
            mismatches.append({"step": c0t["step"], "field": "governor_decision",
                               "c0": c0t.get("governor_decision"),
                               "b1": b1t.get("governor_decision")})

    passed = len(mismatches) == 0
    return {
        "passed": passed,
        "mismatches": mismatches[:10],  # First 10 only
        "total_mismatches": len(mismatches),
        "reason": "IDENTICAL" if passed else f"{len(mismatches)} mismatches found",
    }


def gate_2_toggle_a_equals_c0(c0_result: Dict, c1_result: Dict) -> Dict:
    """Gate 2: Toggle A (graph only) must produce IDENTICAL traces to C0.

    Graph must be provably inert.
    """
    # Same logic as gate 1
    return gate_1_b1_equals_c0(c0_result, c1_result)


def gate_3_bias_audibility(b2_result: Dict, b3_result: Dict) -> Dict:
    """Gate 3: Is the bias audible in routing units?

    Reports memory_bias_max distribution across entity steps.
    bias_to_logit_ratio requires logit_std (not wired), so we use
    the raw max bias value as a proxy.

    Hard ceiling is 0.2, so max_bias_max should never exceed that.
    If median_bias_max > 0: bias is being applied (audible).
    """
    results = {}
    for label, result in [("b2", b2_result), ("b3", b3_result)]:
        aud = result.get("bias_audibility")
        if aud is None:
            results[label] = {"audible": False, "reason": "no bias applied"}
            continue

        audible = aud["median_bias_max"] > 0
        ceiling_breach = aud["max_bias_max"] > 0.2 + 1e-6

        results[label] = {
            "audible": audible,
            "ceiling_breach": ceiling_breach,
            "stats": aud,
        }

    return {
        "passed": True,  # This gate is observational, not pass/fail
        "results": results,
    }


def gate_4_geometry_stability(c0_result: Dict, test_result: Dict, label: str) -> Dict:
    """Gate 4: Geometry stability (15.6 — destabilization, not displacement).

    Neff must not drop below 70% of C0's mean Neff.
    Mean entropy delta < 30%.
    Scar count must not runaway (>2x C0).
    """
    c0_active = [t for t in c0_result["all_traces"] if t["step"] >= c0_result["total_steps"] - c0_result["active_steps"]]
    test_active = [t for t in test_result["all_traces"] if t["step"] >= test_result["total_steps"] - test_result["active_steps"]]

    # Neff comparison
    c0_neffs = [t["neff"] for t in c0_active if t.get("neff") is not None]
    test_neffs = [t["neff"] for t in test_active if t.get("neff") is not None]

    neff_ok = True
    neff_detail = {}
    if c0_neffs and test_neffs:
        c0_mean_neff = sum(c0_neffs) / len(c0_neffs)
        test_mean_neff = sum(test_neffs) / len(test_neffs)
        neff_ratio = test_mean_neff / c0_mean_neff if c0_mean_neff > 0 else 1.0
        neff_ok = neff_ratio >= 0.7
        neff_detail = {
            "c0_mean": round(c0_mean_neff, 4),
            "test_mean": round(test_mean_neff, 4),
            "ratio": round(neff_ratio, 4),
            "threshold": 0.7,
            "passed": neff_ok,
        }

    # Entropy comparison
    c0_entropies = [t["router_entropy"] for t in c0_active if t.get("router_entropy") is not None]
    test_entropies = [t["router_entropy"] for t in test_active if t.get("router_entropy") is not None]

    entropy_ok = True
    entropy_detail = {}
    if c0_entropies and test_entropies:
        c0_mean_ent = sum(c0_entropies) / len(c0_entropies)
        test_mean_ent = sum(test_entropies) / len(test_entropies)
        if c0_mean_ent > 0:
            entropy_delta = (test_mean_ent - c0_mean_ent) / c0_mean_ent
        else:
            entropy_delta = 0.0
        # Decrease (concentration) is the concern
        entropy_ok = entropy_delta > -0.3
        entropy_detail = {
            "c0_mean": round(c0_mean_ent, 6),
            "test_mean": round(test_mean_ent, 6),
            "delta_pct": round(entropy_delta * 100, 2),
            "threshold_pct": -30,
            "passed": entropy_ok,
        }

    # Scar runaway
    scar_ok = True
    scar_detail = {}
    c0_scars = c0_result.get("scars", {})
    test_scars = test_result.get("scars", {})
    if c0_scars and test_scars:
        c0_count = c0_scars.get("total_scars", 0)
        test_count = test_scars.get("total_scars", 0)
        scar_ratio = test_count / max(1, c0_count)
        scar_ok = scar_ratio <= 2.0
        scar_detail = {
            "c0_scars": c0_count,
            "test_scars": test_count,
            "ratio": round(scar_ratio, 2),
            "threshold": 2.0,
            "passed": scar_ok,
        }

    passed = neff_ok and entropy_ok and scar_ok
    return {
        "passed": passed,
        "label": label,
        "neff": neff_detail,
        "entropy": entropy_detail,
        "scars": scar_detail,
    }


def gate_5_primary_success(c0_result: Dict, test_result: Dict, label: str) -> Dict:
    """Gate 5: Primary success metric (synthetic entities only).

    Pre-declared metric: mean loss on synthetic-entity steps in test condition
    compared to same steps in C0.

    Success requires ALL of:
    1. Loss delta < 0 (improvement, not degradation) for synthetic steps
    2. Improvement is larger than any improvement for non-entity steps
    3. Expert selection shows routing shift (Jaccard distance > 0.05)
    """
    c0_active = [t for t in c0_result["all_traces"] if t["step"] >= c0_result["total_steps"] - c0_result["active_steps"]]
    test_active = [t for t in test_result["all_traces"] if t["step"] >= test_result["total_steps"] - test_result["active_steps"]]

    # Build step-indexed maps
    c0_by_step = {t["step"]: t for t in c0_active}
    test_by_step = {t["step"]: t for t in test_active}

    # Compare matched steps by entity type
    deltas = {"synthetic": [], "semantic": [], "none": []}
    jaccard_distances = {"synthetic": [], "semantic": [], "none": []}

    for step in sorted(c0_by_step.keys()):
        if step not in test_by_step:
            continue
        c0t = c0_by_step[step]
        tt = test_by_step[step]
        etype = tt["entity_type"]

        delta = tt["loss"] - c0t["loss"]
        deltas[etype].append(delta)

        # Jaccard distance: 1 - |intersection|/|union|
        c0_ids = set(c0t["expert_ids"])
        test_ids = set(tt["expert_ids"])
        if c0_ids or test_ids:
            jaccard = 1.0 - len(c0_ids & test_ids) / len(c0_ids | test_ids)
            jaccard_distances[etype].append(jaccard)

    results = {}
    for etype in ["synthetic", "semantic", "none"]:
        d = deltas[etype]
        j = jaccard_distances[etype]
        if d:
            results[etype] = {
                "count": len(d),
                "mean_loss_delta": round(sum(d) / len(d), 6),
                "mean_jaccard": round(sum(j) / len(j), 4) if j else 0.0,
            }

    # Check success criteria
    synth = results.get("synthetic", {})
    none_ = results.get("none", {})

    synth_improves = synth.get("mean_loss_delta", 0) < 0
    synth_better_than_none = (
        synth.get("mean_loss_delta", 0) < none_.get("mean_loss_delta", 0)
    )
    routing_shift = synth.get("mean_jaccard", 0) > 0.05

    # Classify result
    if synth_improves and synth_better_than_none and routing_shift:
        verdict = "SUCCESS"
    elif synth_improves and routing_shift:
        verdict = "PARTIAL (improves but not more than non-entity drift)"
    elif routing_shift and not synth_improves:
        verdict = "AUDIBLE_NEUTRAL (routing shifts but loss unchanged/worse)"
    elif synth.get("mean_loss_delta", 0) > 0:
        verdict = "NEGATIVE (loss gets worse)"
    else:
        verdict = "NULL (no measurable effect)"

    return {
        "passed": verdict == "SUCCESS",
        "verdict": verdict,
        "label": label,
        "by_entity_type": results,
        "criteria": {
            "synthetic_improves": synth_improves,
            "synthetic_better_than_none": synth_better_than_none,
            "routing_shift": routing_shift,
        },
    }


# ─── Report Generation ──────────────────────────────────────────

def generate_report(
    results: Dict[str, Dict],
    gates: Dict[str, Dict],
    seed: int,
    config: MemoryExperimentConfig,
) -> Dict:
    """Generate the Phase 8c report."""
    # Condition summaries (without bulky all_traces)
    condition_summaries = {}
    for name, r in results.items():
        summary = {k: v for k, v in r.items() if k != "all_traces"}
        condition_summaries[name] = summary

    return {
        "phase": "8c",
        "description": "Memory Bias Validation Experiment",
        "seed": seed,
        "model": config.model_name,
        "warmup_steps": config.warmup_steps,
        "active_steps": config.active_steps,
        "conditions": condition_summaries,
        "gates": gates,
        "conclusion": _derive_conclusion(gates),
    }


def _derive_conclusion(gates: Dict) -> str:
    """Derive the one-line conclusion from gate results."""
    g1 = gates.get("gate_1_b1_equals_c0", {})
    g2 = gates.get("gate_2_toggle_a_equals_c0", {})

    if not g1.get("passed"):
        return "CONTAMINATED: B1 != C0. Plumbing broken. All results invalid."
    if not g2.get("passed"):
        return "CONTAMINATED: Toggle A != C0. Graph is not inert. All results invalid."

    g4_b2 = gates.get("gate_4_geometry_b2", {})
    g4_b3 = gates.get("gate_4_geometry_b3", {})
    if not g4_b2.get("passed") or not g4_b3.get("passed"):
        return "DESTABILIZED: Memory bias damages routing geometry."

    g5_b2 = gates.get("gate_5_primary_b2", {})
    g5_b3 = gates.get("gate_5_primary_b3", {})

    verdicts = [g5_b2.get("verdict", "NULL"), g5_b3.get("verdict", "NULL")]
    if "SUCCESS" in verdicts:
        return "SUCCESS: Pre-declared primary metric improves for synthetic entities."
    if any("PARTIAL" in v for v in verdicts):
        return "PARTIAL: Improvement exists but does not exceed non-entity drift."
    if any("AUDIBLE_NEUTRAL" in v for v in verdicts):
        return "AUDIBLE_NEUTRAL: Routing shifts but no loss improvement. Memory bias is audible but neutral."
    if any("NEGATIVE" in v for v in verdicts):
        return "NEGATIVE: Memory bias degrades synthetic entity loss. Destructive at this scale."

    return "NULL: No measurable effect. Graph-only is the correct architecture."


# ─── Main ────────────────────────────────────────────────────────

def run_experiment(seed: int, config: MemoryExperimentConfig, condition_filter=None, smoke=False):
    """Run the full 5-condition experiment for one seed."""
    print("=" * 70)
    print(f"PHASE 8c: MEMORY BIAS VALIDATION EXPERIMENT")
    print(f"  Model: {config.model_name}")
    print(f"  Seed: {seed}")
    print(f"  Steps: warmup={config.warmup_steps}, active={config.active_steps}")
    print(f"  Entity prompts: {len(ENTITY_PROMPTS)} ({sum(1 for a in ENTITY_ANNOTATIONS.values() if a['type']=='synthetic')} synthetic, "
          f"{sum(1 for a in ENTITY_ANNOTATIONS.values() if a['type']=='semantic')} semantic)")
    if smoke:
        print(f"  MODE: SMOKE TEST")
    print("=" * 70)

    # Load model
    from mlx_lm import load
    print(f"\n  Loading {config.model_name}...", flush=True)
    t0 = time.time()
    model, tokenizer = load(config.model_name)
    dt = time.time() - t0
    print(f"  Model loaded ({dt:.1f}s)", flush=True)

    from backends.mlx_adapter import MLXMoEAdapter
    adapter = MLXMoEAdapter(model, tokenizer)
    print(f"  MoE: {adapter.num_experts}E top-{adapter.top_k}, "
          f"{adapter.num_layers} layers (MLX)")

    # Filter conditions
    conditions = ALL_CONDITIONS
    if condition_filter:
        conditions = [c for c in ALL_CONDITIONS if c.name in condition_filter]
        # Always include baseline for comparison
        if not any(c.name == "baseline" for c in conditions):
            conditions.insert(0, ALL_CONDITIONS[0])
        print(f"  Running subset: {[c.name for c in conditions]}")

    tag = "smoke" if smoke else "run"
    log_dir = os.path.join(os.path.dirname(__file__), "phase8c_logs")
    os.makedirs(log_dir, exist_ok=True)

    # Run all conditions
    results = {}
    for ci, condition in enumerate(conditions):
        print(f"\n--- Condition {ci+1}/{len(conditions)}: {condition.name} ---")
        jsonl_path = os.path.join(log_dir, f"memory_exp_s{seed}_c{ci}_{condition.name}.jsonl")

        result = run_condition(
            adapter, tokenizer, config, condition, seed, jsonl_path,
        )
        results[condition.name] = result

        # Early stop: check Gate 1 as soon as B1 completes
        if condition.name == "b1" and "baseline" in results:
            print(f"\n  [Gate 1 check: B1 = C0?]", flush=True)
            g1 = gate_1_b1_equals_c0(results["baseline"], results["b1"])
            if not g1["passed"]:
                print(f"  *** GATE 1 FAILED: {g1['reason']} ***")
                print(f"  *** STOPPING — all subsequent results would be contaminated ***")
                gates = {"gate_1_b1_equals_c0": g1}
                report = generate_report(results, gates, seed, config)
                _save_report(report, seed, tag, log_dir)
                return report
            print(f"  Gate 1: PASS (B1 identical to C0)")

        # Early stop: check Gate 2 after Toggle A
        if condition.name == "toggle_a" and "baseline" in results:
            print(f"\n  [Gate 2 check: Toggle A = C0?]", flush=True)
            g2 = gate_2_toggle_a_equals_c0(results["baseline"], results["toggle_a"])
            if not g2["passed"]:
                print(f"  *** GATE 2 FAILED: {g2['reason']} ***")
                print(f"  *** STOPPING — graph is not inert ***")
                gates = {"gate_2_toggle_a_equals_c0": g2}
                report = generate_report(results, gates, seed, config)
                _save_report(report, seed, tag, log_dir)
                return report
            print(f"  Gate 2: PASS (Toggle A identical to C0)")

    # All conditions complete — run remaining gates
    print(f"\n{'='*70}")
    print("VERIFICATION GATES")
    print(f"{'='*70}")

    gates = {}

    # Gate 1 (may already be computed)
    if "b1" in results and "baseline" in results:
        gates["gate_1_b1_equals_c0"] = gate_1_b1_equals_c0(results["baseline"], results["b1"])
        status = "PASS" if gates["gate_1_b1_equals_c0"]["passed"] else "FAIL"
        print(f"  Gate 1 (B1=C0):          {status}")

    # Gate 2 (may already be computed)
    if "toggle_a" in results and "baseline" in results:
        gates["gate_2_toggle_a_equals_c0"] = gate_2_toggle_a_equals_c0(results["baseline"], results["toggle_a"])
        status = "PASS" if gates["gate_2_toggle_a_equals_c0"]["passed"] else "FAIL"
        print(f"  Gate 2 (Toggle A=C0):    {status}")

    # Gate 3
    if "b2" in results and "b3" in results:
        gates["gate_3_audibility"] = gate_3_bias_audibility(results["b2"], results["b3"])
        for label in ["b2", "b3"]:
            r = gates["gate_3_audibility"]["results"].get(label, {})
            if r.get("audible"):
                stats = r.get("stats", {})
                print(f"  Gate 3 ({label} audibility): AUDIBLE "
                      f"(median_max={stats.get('median_bias_max', '?')}, "
                      f"frac={stats.get('audible_fraction', '?')})")
            else:
                reason = r.get("reason", "inaudible")
                print(f"  Gate 3 ({label} audibility): NOT AUDIBLE ({reason})")

    # Gate 4
    if "b2" in results and "baseline" in results:
        gates["gate_4_geometry_b2"] = gate_4_geometry_stability(results["baseline"], results["b2"], "b2")
        status = "PASS" if gates["gate_4_geometry_b2"]["passed"] else "FAIL"
        print(f"  Gate 4 (geometry B2):    {status}")

    if "b3" in results and "baseline" in results:
        gates["gate_4_geometry_b3"] = gate_4_geometry_stability(results["baseline"], results["b3"], "b3")
        status = "PASS" if gates["gate_4_geometry_b3"]["passed"] else "FAIL"
        print(f"  Gate 4 (geometry B3):    {status}")

    # Gate 5
    if "b2" in results and "baseline" in results:
        gates["gate_5_primary_b2"] = gate_5_primary_success(results["baseline"], results["b2"], "b2")
        verdict = gates["gate_5_primary_b2"]["verdict"]
        print(f"  Gate 5 (primary B2):     {verdict}")

    if "b3" in results and "baseline" in results:
        gates["gate_5_primary_b3"] = gate_5_primary_success(results["baseline"], results["b3"], "b3")
        verdict = gates["gate_5_primary_b3"]["verdict"]
        print(f"  Gate 5 (primary B3):     {verdict}")

    # Report
    report = generate_report(results, gates, seed, config)
    _save_report(report, seed, tag, log_dir)

    print(f"\n{'='*70}")
    print(f"CONCLUSION: {report['conclusion']}")
    print(f"{'='*70}")

    return report


def _save_report(report: Dict, seed: int, tag: str, log_dir: str):
    """Save report and gates to JSON files."""
    report_path = os.path.join(log_dir, f"memory_exp_s{seed}_report.json")
    gates_path = os.path.join(log_dir, f"memory_exp_s{seed}_gates.json")

    # Remove all_traces from report before saving (too large)
    save_report = json.loads(json.dumps(report, default=str))

    with open(report_path, "w") as f:
        json.dump(save_report, f, indent=2, default=str)
    print(f"\n  Report: {report_path}")

    with open(gates_path, "w") as f:
        json.dump(report.get("gates", {}), f, indent=2, default=str)
    print(f"  Gates:  {gates_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 8c: Memory Bias Validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, default=0,
                        help="Multi-seed: run N seeds starting at 42 (step 7)")
    parser.add_argument("--model", default="mlx-community/Qwen1.5-MoE-A2.7B-4bit")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--active", type=int, default=250)
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: warmup=10, active=50")
    parser.add_argument("--conditions", type=str, default=None,
                        help="Comma-separated condition names (e.g. 'b1,b2')")
    args = parser.parse_args()

    if args.smoke:
        args.warmup = 10
        args.active = 50

    config = MemoryExperimentConfig(
        model_name=args.model,
        warmup_steps=args.warmup,
        active_steps=args.active,
    )

    condition_filter = None
    if args.conditions:
        condition_filter = set(args.conditions.split(","))

    if args.seeds > 0:
        # Multi-seed
        seeds = [42 + i * 7 for i in range(args.seeds)]
        reports = []
        for s in seeds:
            print(f"\n{'#'*70}")
            print(f"# SEED {s}")
            print(f"{'#'*70}")
            r = run_experiment(s, config, condition_filter, args.smoke)
            reports.append(r)

        print(f"\n{'='*70}")
        print("MULTI-SEED SUMMARY")
        print(f"{'='*70}")
        for r in reports:
            print(f"  Seed {r['seed']}: {r['conclusion']}")
    else:
        run_experiment(args.seed, config, condition_filter, args.smoke)
