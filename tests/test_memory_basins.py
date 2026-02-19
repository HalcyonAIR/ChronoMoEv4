# SPDX-License-Identifier: Apache-2.0
"""
Tests for association basins + memory bias (Phase 8b).

Covers: BasinStore, entity linking, activation diffusion,
memory bias computation, hard ceiling, B1 guarantee,
serialization, and substrate backward compatibility.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from bob_core.basins import (
    AssociationBasin,
    BasinStore,
    MemoryBiasDiagnostics,
    link_entities,
    diffuse_activation,
    compute_memory_bias,
)
from bob_core.graph import RelationalGraph


# --- Helpers ---

def _make_graph():
    """Build a small test graph: Jeff --spouse--> Paula, Jeff --works_at--> ALS."""
    g = RelationalGraph()
    jeff = g.resolve_or_create_node("Jeff", node_type="person")
    paula = g.resolve_or_create_node("Paula", node_type="person")
    als = g.resolve_or_create_node("ALS", node_type="organisation")
    # add_triple takes names, not IDs (it resolves internally)
    g.add_triple("Jeff", "spouse", "Paula")
    g.add_triple("Jeff", "works_at", "ALS")
    return g, jeff, paula, als


def _make_basin_store(bias_scale=0.1, num_layers=2, num_experts=4):
    """Create a BasinStore with one basin that has known affinities."""
    store = BasinStore(bias_scale=bias_scale)
    basin = store.get_or_create_basin("node_0001", num_layers, num_experts)
    # Set known affinities: layer 0 prefers expert 1, layer 1 prefers expert 3
    basin.bias_vector[0][1] = 1.0
    basin.bias_vector[1][3] = 0.5
    basin.strength = 0.5
    return store


# --- Tests ---

def test_hard_ceiling_rejects_high_scale():
    """BasinStore rejects bias_scale > 0.2."""
    try:
        BasinStore(bias_scale=0.3)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass

    # At ceiling is fine
    store = BasinStore(bias_scale=0.2)
    assert store.bias_scale == 0.2


def test_hard_ceiling_rejects_high_ceiling():
    """BasinStore rejects hard_ceiling > 0.2."""
    try:
        BasinStore(bias_scale=0.1, hard_ceiling=0.3)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass


def test_basin_creation():
    """get_or_create_basin returns correct shape."""
    store = BasinStore()
    basin = store.get_or_create_basin("node_0001", num_layers=4, num_experts=8)
    assert basin.node_id == "node_0001"
    assert len(basin.bias_vector) == 4
    assert len(basin.bias_vector[0]) == 8
    assert all(v == 0.0 for row in basin.bias_vector for v in row)
    assert basin.strength == 0.1
    assert store.basin_count == 1


def test_basin_get_existing():
    """get_or_create_basin returns same basin on second call."""
    store = BasinStore()
    b1 = store.get_or_create_basin("node_0001", 2, 4)
    b1.strength = 0.9
    b2 = store.get_or_create_basin("node_0001", 2, 4)
    assert b2.strength == 0.9
    assert b1 is b2
    assert store.basin_count == 1


def test_single_basin_bias():
    """Known affinities produce expected output values."""
    store = _make_basin_store(bias_scale=0.1, num_layers=2, num_experts=4)
    # Activate node_0001 at strength 1.0
    activations = {"node_0001": 1.0}
    bias_field, diag = compute_memory_bias(activations, store, 2, 4)

    assert bias_field is not None
    # layer 0, expert 1: activation(1.0) * basin_strength(0.5) * affinity(1.0) * scale(0.1) = 0.05
    assert abs(bias_field[0][1] - 0.05) < 1e-6
    # layer 1, expert 3: 1.0 * 0.5 * 0.5 * 0.1 = 0.025
    assert abs(bias_field[1][3] - 0.025) < 1e-6
    # Other experts should be 0
    assert abs(bias_field[0][0]) < 1e-10
    assert abs(bias_field[0][2]) < 1e-10


def test_multi_basin_aggregation():
    """Two basins active produce additive blend."""
    store = BasinStore(bias_scale=0.1)
    b1 = store.get_or_create_basin("node_A", 2, 4)
    b1.bias_vector[0][0] = 1.0
    b1.strength = 0.5

    b2 = store.get_or_create_basin("node_B", 2, 4)
    b2.bias_vector[0][0] = 0.8
    b2.strength = 0.4

    activations = {"node_A": 1.0, "node_B": 1.0}
    bias_field, diag = compute_memory_bias(activations, store, 2, 4)

    assert bias_field is not None
    # layer 0, expert 0: (1.0 * 0.5 * 1.0 + 1.0 * 0.4 * 0.8) * 0.1 = (0.5 + 0.32) * 0.1 = 0.082
    assert abs(bias_field[0][0] - 0.082) < 1e-6
    assert diag.n_active_basins == 2


def test_no_active_basins():
    """No matching basins returns None, not zeros."""
    store = BasinStore()
    activations = {"unknown_node": 1.0}
    bias_field, diag = compute_memory_bias(activations, store, 2, 4)
    assert bias_field is None
    assert diag.n_active_basins == 0


def test_empty_activations():
    """Empty activations returns None."""
    store = _make_basin_store()
    bias_field, diag = compute_memory_bias({}, store, 2, 4)
    assert bias_field is None


def test_bias_scale_zero_b1_guarantee():
    """bias_scale=0.0 returns None (B1 = Condition 0 guarantee)."""
    store = BasinStore(bias_scale=0.0)
    basin = store.get_or_create_basin("node_0001", 2, 4)
    basin.bias_vector[0][1] = 1.0
    basin.strength = 1.0

    activations = {"node_0001": 1.0}
    bias_field, diag = compute_memory_bias(activations, store, 2, 4)
    assert bias_field is None, "B1 guarantee: scale=0.0 must return None"


def test_hard_ceiling_clamps_output():
    """Output values clamped to hard_ceiling even with large affinities."""
    store = BasinStore(bias_scale=0.2, hard_ceiling=0.2)
    basin = store.get_or_create_basin("node_0001", 2, 4)
    basin.bias_vector[0][0] = 10.0  # Huge affinity
    basin.strength = 1.0

    activations = {"node_0001": 1.0}
    bias_field, diag = compute_memory_bias(activations, store, 2, 4)

    assert bias_field is not None
    # 1.0 * 1.0 * 10.0 * 0.2 = 2.0, but clamped to 0.2
    assert abs(bias_field[0][0] - 0.2) < 1e-6
    assert diag.max_abs_bias <= 0.2 + 1e-10


def test_hard_ceiling_clamps_negative():
    """Negative affinities also clamped to -hard_ceiling."""
    store = BasinStore(bias_scale=0.2, hard_ceiling=0.2)
    basin = store.get_or_create_basin("node_0001", 1, 4)
    basin.bias_vector[0][2] = -10.0
    basin.strength = 1.0

    activations = {"node_0001": 1.0}
    bias_field, diag = compute_memory_bias(activations, store, 1, 4)

    assert bias_field is not None
    assert abs(bias_field[0][2] - (-0.2)) < 1e-6


def test_diagnostics():
    """Diagnostics compute mean_abs, max_abs correctly."""
    store = _make_basin_store(bias_scale=0.1, num_layers=2, num_experts=4)
    activations = {"node_0001": 1.0}
    bias_field, diag = compute_memory_bias(activations, store, 2, 4)

    assert diag.n_active_nodes == 1
    assert diag.n_active_basins == 1
    assert diag.max_abs_bias > 0
    assert diag.mean_abs_bias > 0
    assert diag.max_abs_bias >= diag.mean_abs_bias

    # Verify to_dict round-trip
    d = diag.to_dict()
    assert "mean_abs_bias" in d
    assert "max_abs_bias" in d


def test_entity_linking_exact():
    """Exact alias match gives activation 1.0."""
    g, jeff_id, paula_id, als_id = _make_graph()
    activations = link_entities(["Jeff"], g)
    assert jeff_id in activations
    assert activations[jeff_id] == 1.0


def test_entity_linking_case_insensitive():
    """Alias matching is case-insensitive (via alias table normalisation)."""
    g, jeff_id, _, _ = _make_graph()
    activations = link_entities(["jeff"], g)
    assert jeff_id in activations
    assert activations[jeff_id] == 1.0


def test_entity_linking_unknown():
    """Unknown tokens produce no activations."""
    g, _, _, _ = _make_graph()
    activations = link_entities(["Zorblax", "Krenthar"], g)
    assert len(activations) == 0


def test_entity_linking_ambiguous():
    """Ambiguous alias (multiple nodes) gives activation 0.8."""
    g = RelationalGraph()
    n1 = g.resolve_or_create_node("Sam", node_type="person")
    n2 = g.resolve_or_create_node("Sam Smith", node_type="person")
    # Both have "sam" as an alias (Sam Smith has "sam smith" and Sam has "sam")
    # For ambiguity, we need to manually add "sam" as alias to n2
    g.nodes[n2].aliases.append("Sam")
    g.rebuild_alias_table()

    activations = link_entities(["Sam"], g)
    # Both nodes activated at 0.8
    assert n1 in activations
    assert n2 in activations
    assert activations[n1] == 0.8
    assert activations[n2] == 0.8


def test_diffusion_1hop():
    """1-hop diffusion spreads activation to neighbours."""
    g, jeff_id, paula_id, als_id = _make_graph()
    activations = {jeff_id: 1.0}
    diffused = diffuse_activation(activations, g, depth=1, decay=0.3)

    # Jeff stays at 1.0
    assert diffused[jeff_id] == 1.0
    # Paula gets 1.0 * 0.3 = 0.3
    assert abs(diffused[paula_id] - 0.3) < 1e-6
    # ALS gets 1.0 * 0.3 = 0.3
    assert abs(diffused[als_id] - 0.3) < 1e-6


def test_diffusion_max_not_sum():
    """Converging paths use MAX not SUM."""
    g = RelationalGraph()
    # A --spouse--> B, A --friend--> C, B --friend--> C
    a = g.resolve_or_create_node("Alice", node_type="person")
    b = g.resolve_or_create_node("Bob", node_type="person")
    c = g.resolve_or_create_node("Charlie", node_type="person")
    g.add_triple("Alice", "spouse", "Bob")
    g.add_triple("Alice", "friend", "Charlie")
    g.add_triple("Bob", "friend", "Charlie")

    # Activate both A and B
    activations = {a: 1.0, b: 0.8}
    diffused = diffuse_activation(activations, g, depth=1, decay=0.3)

    # C reached from A (1.0*0.3=0.3) and B (0.8*0.3=0.24): MAX = 0.3
    assert abs(diffused[c] - 0.3) < 1e-6


def test_diffusion_no_override_direct():
    """Diffusion doesn't reduce directly-mentioned entity activation."""
    g, jeff_id, paula_id, _ = _make_graph()
    # Both directly mentioned
    activations = {jeff_id: 1.0, paula_id: 1.0}
    diffused = diffuse_activation(activations, g, depth=1, decay=0.3)

    # Paula stays at 1.0 (direct), not reduced to 0.3 (from Jeff)
    assert diffused[paula_id] == 1.0
    assert diffused[jeff_id] == 1.0


def test_basin_serialization_roundtrip():
    """BasinStore to_dict → from_dict preserves state."""
    store = _make_basin_store(bias_scale=0.15, num_layers=2, num_experts=4)
    d = store.to_dict()
    restored = BasinStore.from_dict(d)

    assert restored.bias_scale == 0.15
    assert restored.hard_ceiling == 0.2
    assert restored.basin_count == 1

    basin = restored.get_basin("node_0001")
    assert basin is not None
    assert basin.strength == 0.5
    assert abs(basin.bias_vector[0][1] - 1.0) < 1e-6
    assert abs(basin.bias_vector[1][3] - 0.5) < 1e-6


def test_association_basin_serialization():
    """AssociationBasin to_dict → from_dict preserves all fields."""
    basin = AssociationBasin(
        node_id="test_node",
        bias_vector=[[0.1, 0.2], [0.3, 0.4]],
        strength=0.75,
        update_count=5,
        last_updated="2026-02-18T12:00:00",
    )
    d = basin.to_dict()
    restored = AssociationBasin.from_dict(d)

    assert restored.node_id == "test_node"
    assert restored.strength == 0.75
    assert restored.update_count == 5
    assert restored.last_updated == "2026-02-18T12:00:00"
    assert restored.bias_vector == [[0.1, 0.2], [0.3, 0.4]]


def test_zero_strength_basin_ignored():
    """Basin with strength=0 is not included in bias computation."""
    store = BasinStore(bias_scale=0.1)
    basin = store.get_or_create_basin("node_0001", 2, 4)
    basin.bias_vector[0][0] = 1.0
    basin.strength = 0.0  # Zero strength

    activations = {"node_0001": 1.0}
    bias_field, diag = compute_memory_bias(activations, store, 2, 4)
    assert bias_field is None
    assert diag.n_active_basins == 0


def test_partial_activation_scales_bias():
    """Activation strength < 1.0 proportionally reduces bias."""
    store = _make_basin_store(bias_scale=0.1, num_layers=2, num_experts=4)
    # Full activation
    bias_full, _ = compute_memory_bias({"node_0001": 1.0}, store, 2, 4)
    # Half activation
    bias_half, _ = compute_memory_bias({"node_0001": 0.5}, store, 2, 4)

    assert bias_full is not None and bias_half is not None
    # Half activation should give half the bias
    ratio = bias_half[0][1] / bias_full[0][1]
    assert abs(ratio - 0.5) < 1e-6


def test_substrate_backward_compat():
    """BobSubstrate without memory params preserves all existing behavior."""
    from bob_core.substrate import BobSubstrate

    # Create a minimal mock adapter
    class MockAdapter:
        num_experts = 4
        num_layers = 2
        top_k = 2
        adapter_version = "mock-v1"
        supports_overlap = False
        overlap_kind = "none"

        def forward(self, inputs, targets=None, memory_bias=None):
            from backends.adapter import ForwardResult
            import torch
            return ForwardResult(
                loss=torch.tensor(1.0),
                logits=None,
                snapshots=[],
                expert_invocations=8,
                tokens_processed=4,
            )

        def forward_with_motif(self, inputs, motif, targets=None, memory_bias=None):
            return self.forward(inputs, targets, memory_bias=memory_bias)

    # Should construct without errors, no memory params
    bob = BobSubstrate(MockAdapter())
    assert bob.memory_graph is None
    assert bob.basin_store is None
    assert bob.memory_bias_scale == 0.1


# --- Script runner (backward compat) ---

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
