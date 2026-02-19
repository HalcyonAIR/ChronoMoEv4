# SPDX-License-Identifier: Apache-2.0
"""
Deterministic B1 equivalence test.

Asserts that BobSubstrate with graph + basins at scale=0.0 produces
IDENTICAL traces to BobSubstrate without graph/basins.

This is the unit-test version of Gate 1 from memory_experiment.py.
If this fails, the wiring is not clean.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from backends.adapter import ForwardResult, LayerSnapshot
from bob_core.substrate import BobSubstrate
from bob_core.motifs import GateThresholds
from bob_core.ledgers import BobCore
from bob_core.medium_clock import MediumClock
from bob_core.fast_clock import FastClock
from bob_core.slow_clock import SlowClock
from bob_core.governor import BobGovernor
from bob_core.promotion import PromotionGate
from bob_core.graph import RelationalGraph
from bob_core.basins import BasinStore


# --- Mock Adapter ---

class _MockAdapter:
    """Deterministic adapter that returns constant results.

    Returns the same loss, expert_ids, and routing data regardless of
    memory_bias. This isolates the test to the substrate's decision
    logic: does the code path diverge when graph/basins are present
    at scale=0.0?
    """
    num_experts = 8
    num_layers = 4
    top_k = 2
    adapter_version = "mock-b1-test"
    supports_overlap = False
    overlap_kind = "none"

    def forward(self, inputs, targets=None, memory_bias=None):
        # Deterministic snapshot per layer
        snapshots = []
        for lid in range(self.num_layers):
            selected = torch.tensor([[lid % self.num_experts, (lid + 1) % self.num_experts]])
            weights = torch.tensor([[0.6, 0.4]])
            usage = torch.zeros(self.num_experts)
            usage[lid % self.num_experts] = 0.6
            usage[(lid + 1) % self.num_experts] = 0.4
            snapshots.append(LayerSnapshot(
                layer_id=lid,
                router_scores=torch.randn(1, self.num_experts),
                selected_experts=selected,
                routing_weights=weights,
                expert_usage=usage,
                mean_entropy=2.5,
            ))
        return ForwardResult(
            loss=torch.tensor(1.5),
            logits=None,
            snapshots=snapshots,
            expert_invocations=self.num_layers * self.top_k,
            tokens_processed=1,
        )

    def forward_with_motif(self, inputs, motif, targets=None, memory_bias=None):
        return self.forward(inputs, targets, memory_bias=memory_bias)


# --- Graph + Basin builders ---

def _build_graph():
    g = RelationalGraph()
    g.resolve_or_create_node("Jeff", node_type="person")
    g.resolve_or_create_node("Paula", node_type="person")
    g.resolve_or_create_node("Zorblax", node_type="concept")
    g.add_triple("Jeff", "spouse", "Paula")
    return g


def _build_basin_store(graph, num_layers, num_experts, scale):
    store = BasinStore(bias_scale=scale)
    # Zorblax: strong affinity for expert 0 at all layers
    candidates = graph.alias_table.lookup("Zorblax")
    assert len(candidates) == 1
    zorblax_id = candidates[0]
    basin = store.get_or_create_basin(zorblax_id, num_layers, num_experts)
    basin.strength = 0.8
    for lid in range(num_layers):
        basin.bias_vector[lid][0] = 1.0
    return store


def _build_substrate(adapter, graph=None, basin_store=None, scale=0.1):
    """Build a fully-wired BobSubstrate."""
    bob_core = BobCore(debt_cap=1.0)
    fast_clock = FastClock()
    medium_clock = MediumClock()
    slow_clock = SlowClock()
    promotion_gate = PromotionGate()
    governor = BobGovernor(
        bob_core, medium_clock,
        fast_clock=fast_clock,
        slow_clock=slow_clock,
    )
    return BobSubstrate(
        adapter,
        gate_thresholds=GateThresholds(stability_min=0.15, debt_max=0.5, survival_min=0.7),
        warmup_steps=5,
        bob_core=bob_core,
        governor=governor,
        fast_clock=fast_clock,
        medium_clock=medium_clock,
        slow_clock=slow_clock,
        promotion_gate=promotion_gate,
        memory_graph=graph,
        basin_store=basin_store,
        memory_bias_scale=scale,
    )


# --- Tests ---

def test_b1_equals_c0():
    """B1 (graph + basins, scale=0.0) must produce identical traces to C0 (no graph, no basins).

    "Identical" means: loss, expert_ids, path, governor_decision match
    at every step. Not "close". Identical within floating tolerance.
    """
    adapter = _MockAdapter()

    # C0: no graph, no basins
    bob_c0 = _build_substrate(adapter, graph=None, basin_store=None)

    # B1: graph + basins, scale=0.0
    graph = _build_graph()
    basin_store = _build_basin_store(graph, adapter.num_layers, adapter.num_experts, scale=0.0)
    bob_b1 = _build_substrate(adapter, graph=graph, basin_store=basin_store, scale=0.0)

    total_steps = 20
    entity_tokens_with_match = ["zorblax", "jeff"]
    entity_tokens_without = ["hello", "world"]

    for step in range(total_steps):
        ctx_class = (step // 5) % 3
        inputs = [1, 2, 3, 4, 5]
        targets = [1, 2, 3, 4, 5]

        # Alternate: some steps have entity tokens that would match, some don't
        entity_tokens = entity_tokens_with_match if step % 2 == 0 else entity_tokens_without

        trace_c0 = bob_c0.step(inputs, targets, ctx_class, step, entity_tokens=None)
        trace_b1 = bob_b1.step(inputs, targets, ctx_class, step, entity_tokens=entity_tokens)

        # Assert identity
        assert abs(trace_c0.loss - trace_b1.loss) < 1e-6, (
            f"Step {step}: loss mismatch c0={trace_c0.loss} b1={trace_b1.loss}"
        )
        assert trace_c0.expert_ids == trace_b1.expert_ids, (
            f"Step {step}: expert_ids mismatch c0={trace_c0.expert_ids} b1={trace_b1.expert_ids}"
        )
        assert trace_c0.path == trace_b1.path, (
            f"Step {step}: path mismatch c0={trace_c0.path} b1={trace_b1.path}"
        )
        assert trace_c0.governor_decision == trace_b1.governor_decision, (
            f"Step {step}: governor mismatch c0={trace_c0.governor_decision} b1={trace_b1.governor_decision}"
        )
        assert trace_c0.gate_passed == trace_b1.gate_passed, (
            f"Step {step}: gate_passed mismatch c0={trace_c0.gate_passed} b1={trace_b1.gate_passed}"
        )


def test_b1_memory_bias_is_none():
    """With scale=0.0, compute_memory_bias must return None (B1 guarantee).

    This verifies the short-circuit: scale=0.0 → bias_field=None → no effect.
    """
    from bob_core.basins import compute_memory_bias, link_entities, diffuse_activation

    graph = _build_graph()
    store = _build_basin_store(graph, num_layers=4, num_experts=8, scale=0.0)

    # Entity tokens that definitely match
    activations = link_entities(["zorblax"], graph)
    assert len(activations) > 0, "Zorblax should match"

    diffused = diffuse_activation(activations, graph)
    bias_field, diag = compute_memory_bias(diffused, store, 4, 8)

    assert bias_field is None, "B1 guarantee: scale=0.0 must return None"


def test_b2_memory_bias_is_not_none():
    """With scale=0.1, compute_memory_bias must return a non-None field.

    Control: verify that at scale > 0, bias IS produced.
    """
    from bob_core.basins import compute_memory_bias, link_entities, diffuse_activation

    graph = _build_graph()
    store = _build_basin_store(graph, num_layers=4, num_experts=8, scale=0.1)

    activations = link_entities(["zorblax"], graph)
    diffused = diffuse_activation(activations, graph)
    bias_field, diag = compute_memory_bias(diffused, store, 4, 8)

    assert bias_field is not None, "B2: scale=0.1 must produce bias"
    assert diag.max_abs_bias > 0, "B2: bias should have nonzero values"
    assert diag.max_abs_bias <= 0.2 + 1e-10, "Hard ceiling must hold"


def test_determinism_across_runs():
    """Same inputs must produce same traces across 3 independent runs.

    Tests that there are no stochastic components in the decision path.
    """
    results = []
    for _ in range(3):
        adapter = _MockAdapter()
        bob = _build_substrate(adapter)
        traces = []
        for step in range(10):
            trace = bob.step([1, 2, 3], [1, 2, 3], step % 3, step)
            traces.append((trace.loss, trace.path, trace.expert_ids, trace.gate_passed))
        results.append(traces)

    for i in range(1, 3):
        for step in range(10):
            assert results[0][step] == results[i][step], (
                f"Run {i} step {step}: traces diverged. "
                f"Run 0={results[0][step]} Run {i}={results[i][step]}"
            )


# --- Script runner ---

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed + failed} tests passed")
    if failed:
        sys.exit(1)
