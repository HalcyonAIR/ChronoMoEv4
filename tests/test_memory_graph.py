# SPDX-License-Identifier: Apache-2.0
"""
Tests for the relational graph (Phase 8a, Toggle A).

Validates node/edge CRUD, alias resolution, contradiction detection,
assertion extraction, rendering, bounds enforcement, and serialization.

Run:
  pytest test_memory_graph.py -v        # preferred
  python3 test_memory_graph.py          # also works
"""

import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from typing import List

from bob_core.graph import (
    RelationalGraph,
    GraphNode,
    GraphEdge,
    AliasTable,
    NodeMetadata,
    DetectedTriple,
    TripleResult,
    TripleOutcome,
    ALLOWED_RELATIONS,
    RELATION_TEMPLATES,
)

# --- Test helpers ---

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


def fresh_graph() -> RelationalGraph:
    """Create a fresh graph with some test data."""
    g = RelationalGraph()
    g.add_triple("Jeff", "spouse", "Paula", session="s1", turn=1)
    g.add_triple("Jeff", "works_at", "ALS Minerals", session="s1", turn=2)
    g.add_triple("Jeff", "grandchild", "Brogan", session="s1", turn=3)
    return g


# --- Tests ---

def test_node_crud():
    """Test basic node creation and retrieval."""
    print("\n=== Test 1: Node CRUD ===")

    g = RelationalGraph()
    node = g.add_node(["Paula", "my wife"], node_type="person")

    check("Node created", node is not None)
    check("Node has ID", node.node_id.startswith("node_"))
    check("Node has aliases", node.aliases == ["Paula", "my wife"])
    check("Node type set", node.node_type == "person")

    # Retrieve by ID
    retrieved = g.get_node(node.node_id)
    check("Retrieve by ID", retrieved is not None and retrieved.node_id == node.node_id)

    # Retrieve by alias
    by_alias = g.get_node_by_alias("Paula")
    check("Retrieve by alias", by_alias is not None and by_alias.node_id == node.node_id)

    by_alias2 = g.get_node_by_alias("my wife")
    check("Retrieve by second alias", by_alias2 is not None and by_alias2.node_id == node.node_id)


def test_alias_normalisation():
    """Test that alias lookup is case-insensitive and whitespace-tolerant."""
    print("\n=== Test 2: Alias Normalisation ===")

    g = RelationalGraph()
    g.add_node(["Paula"], node_type="person")

    check("Lowercase match", g.get_node_by_alias("paula") is not None)
    check("Uppercase match", g.get_node_by_alias("PAULA") is not None)
    check("Mixed case match", g.get_node_by_alias("pAuLa") is not None)
    check("Whitespace stripped", g.get_node_by_alias("  Paula  ") is not None)
    check("No match for wrong name", g.get_node_by_alias("Sarah") is None)


def test_edge_crud():
    """Test edge creation and retrieval."""
    print("\n=== Test 3: Edge CRUD ===")

    g = fresh_graph()

    check("Three edges created", g.edge_count == 3)

    # Get Jeff's node
    jeff = g.get_node_by_alias("Jeff")
    check("Jeff exists", jeff is not None)

    edges = g.get_edges_for_node(jeff.node_id)
    check("Jeff has 3 edges", len(edges) == 3)

    # Check specific edge exists
    paula = g.get_node_by_alias("Paula")
    check("has_edge works",
          g.has_edge(jeff.node_id, "spouse", paula.node_id))


def test_allowed_relations():
    """Test that invalid relations are rejected."""
    print("\n=== Test 4: Allowed Relations ===")

    g = RelationalGraph()
    result = g.add_triple("Jeff", "likes", "pizza")

    check("Invalid relation rejected",
          result.result == TripleResult.INVALID_RELATION,
          f"got {result.result}")

    check("No edges created for invalid relation",
          g.edge_count == 0)

    # Valid relation works
    result2 = g.add_triple("Jeff", "spouse", "Paula")
    check("Valid relation accepted",
          result2.result == TripleResult.CREATED)


def test_contradiction_detection():
    """Test contradiction detection for same subject+relation, different object."""
    print("\n=== Test 5: Contradiction Detection ===")

    g = RelationalGraph()
    r1 = g.add_triple("Jeff", "spouse", "Paula")
    check("First triple created", r1.result == TripleResult.CREATED)

    # Try to add conflicting spouse
    r2 = g.add_triple("Jeff", "spouse", "Sarah")
    check("Contradiction detected",
          r2.result == TripleResult.CONTRADICTION,
          f"got {r2.result}")

    check("Conflicting edge returned",
          r2.conflicting_edge is not None)

    check("Message mentions both names",
          "Paula" in r2.message and "Sarah" in r2.message,
          f"message: {r2.message}")

    # Original triple unchanged
    check("Original edge still exists", g.edge_count == 1)


def test_duplicate_reinforcement():
    """Test that adding the same triple reinforces it."""
    print("\n=== Test 6: Duplicate Reinforcement ===")

    g = RelationalGraph()
    r1 = g.add_triple("Jeff", "spouse", "Paula")
    check("First: created", r1.result == TripleResult.CREATED)

    r2 = g.add_triple("Jeff", "spouse", "Paula")
    check("Second: reinforced", r2.result == TripleResult.REINFORCED)

    check("Still one edge", g.edge_count == 1)
    check("Usage count incremented",
          r2.edge.usage_count == 1,
          f"got {r2.edge.usage_count}")
    check("Confidence increased",
          r2.edge.confidence > 0.8,
          f"got {r2.edge.confidence}")


def test_resolve_or_create():
    """Test resolve_or_create_node returns existing or creates new."""
    print("\n=== Test 7: Resolve or Create ===")

    g = RelationalGraph()
    g.add_node(["Paula"], node_type="person")

    # Existing alias → return existing ID
    id1 = g.resolve_or_create_node("Paula")
    check("Existing resolved", id1 == g.get_node_by_alias("Paula").node_id)

    # New name → create new node
    initial_count = g.node_count
    id2 = g.resolve_or_create_node("Brogan")
    check("New node created", g.node_count == initial_count + 1)
    check("New node has ID", id2.startswith("node_"))


def test_neighbour_traversal():
    """Test 1-hop neighbour retrieval."""
    print("\n=== Test 8: Neighbour Traversal ===")

    g = fresh_graph()
    jeff = g.get_node_by_alias("Jeff")

    neighbours = g.get_neighbours(jeff.node_id)
    check("Jeff has 3 neighbours", len(neighbours) == 3)

    neighbour_ids = [n[0] for n in neighbours]
    paula = g.get_node_by_alias("Paula")
    check("Paula is a neighbour", paula.node_id in neighbour_ids)


def test_render_fact_packet():
    """Test templated rendering of facts."""
    print("\n=== Test 9: Render Fact Packet ===")

    g = fresh_graph()
    jeff = g.get_node_by_alias("Jeff")

    packet = g.render_fact_packet(jeff.node_id)
    check("Packet is not None", packet is not None)
    check("Contains 'Fact:'", "Fact:" in packet)
    check("Contains 'Paula'", "Paula" in packet)
    check("Contains relation text",
          "is married to" in packet or "works at" in packet,
          f"packet: {packet}")

    # Unknown node returns None
    check("Unknown node returns None",
          g.render_fact_packet("nonexistent") is None)

    # Max facts limit
    packet_1 = g.render_fact_packet(jeff.node_id, max_facts=1)
    check("Max facts limits output",
          packet_1.count("Fact:") == 1,
          f"got {packet_1.count('Fact:')} facts")


def test_detect_assertions():
    """Test conservative pattern matching for relational statements."""
    print("\n=== Test 10: Detect Assertions ===")

    # Spouse
    triples = RelationalGraph.detect_assertions("Paula is my wife")
    check("Detects spouse",
          len(triples) >= 1 and any(t.relation == "spouse" for t in triples),
          f"got {[(t.relation, t.object_) for t in triples]}")

    # Workplace
    triples = RelationalGraph.detect_assertions("I work at ALS Minerals")
    check("Detects workplace",
          len(triples) >= 1 and any(t.relation == "works_at" for t in triples))

    # Grandchild
    triples = RelationalGraph.detect_assertions("Brogan is my grandson")
    check("Detects grandchild",
          len(triples) >= 1 and any(t.relation == "grandchild" for t in triples))

    # Lives in
    triples = RelationalGraph.detect_assertions("I live in Ireland")
    check("Detects location",
          len(triples) >= 1 and any(t.relation == "lives_in" for t in triples))

    # No assertion in unrelated text
    triples = RelationalGraph.detect_assertions("explain quicksort")
    check("No assertion in unrelated text", len(triples) == 0)


def test_enforce_bounds():
    """Test bounds enforcement evicts lowest-confidence entries."""
    print("\n=== Test 11: Enforce Bounds ===")

    g = RelationalGraph(max_nodes=200, max_edges=500)

    # Add some triples
    g.add_triple("A", "colleague", "B", session="s1", turn=1)
    g.add_triple("C", "colleague", "D", session="s1", turn=2)

    # Lower one edge's confidence below threshold
    g.edges[0].confidence = 0.1

    g.enforce_bounds(min_confidence=0.3)
    check("Low-confidence edge removed",
          g.edge_count == 1,
          f"got {g.edge_count}")

    # Orphan nodes should be removed too
    check("Orphan nodes removed",
          g.node_count == 2,
          f"got {g.node_count}")


def test_enforce_bounds_edge_cap():
    """Test that enforce_bounds respects max_edges."""
    print("\n=== Test 12: Enforce Bounds Edge Cap ===")

    g = RelationalGraph(max_nodes=200, max_edges=3)

    # Add 5 edges (over the cap)
    g.add_triple("A", "colleague", "B")
    g.add_triple("C", "colleague", "D")
    g.add_triple("E", "colleague", "F")
    g.add_triple("G", "colleague", "H")
    g.add_triple("I", "colleague", "J")
    check("5 edges before bounds", g.edge_count == 5)

    g.enforce_bounds()
    check("Edge count capped",
          g.edge_count <= 3,
          f"got {g.edge_count}")


def test_serialization_roundtrip():
    """Test to_dict() → from_dict() produces identical graph."""
    print("\n=== Test 13: Serialization Round-Trip ===")

    g = fresh_graph()

    # Serialize
    d = g.to_dict()
    check("Dict has version", d["version"] == "0.4")
    check("Dict has nodes", len(d["nodes"]) == g.node_count)
    check("Dict has edges", len(d["edges"]) == g.edge_count)

    # JSON round-trip (proves it's JSON-serializable)
    json_str = json.dumps(d)
    d2 = json.loads(json_str)

    # Deserialize
    g2 = RelationalGraph.from_dict(d2)

    check("Same node count", g2.node_count == g.node_count)
    check("Same edge count", g2.edge_count == g.edge_count)

    # Alias table rebuilt
    paula = g2.get_node_by_alias("Paula")
    check("Alias table rebuilt (Paula found)", paula is not None)

    jeff = g2.get_node_by_alias("Jeff")
    check("Alias table rebuilt (Jeff found)", jeff is not None)

    # Edge data preserved
    jeff_edges = g2.get_edges_for_node(jeff.node_id)
    check("Edges preserved",
          len(jeff_edges) == 3,
          f"got {len(jeff_edges)}")


def test_rebuild_alias_table():
    """Test alias table reconstruction from nodes."""
    print("\n=== Test 14: Rebuild Alias Table ===")

    g = fresh_graph()

    # Clear and rebuild
    g.alias_table.clear()
    check("Alias table cleared",
          g.get_node_by_alias("Paula") is None)

    g.rebuild_alias_table()
    check("Alias table rebuilt",
          g.get_node_by_alias("Paula") is not None)


def test_empty_graph():
    """Test that empty graph operations don't crash."""
    print("\n=== Test 15: Empty Graph ===")

    g = RelationalGraph()

    check("Empty node count", g.node_count == 0)
    check("Empty edge count", g.edge_count == 0)
    check("Get nonexistent node", g.get_node("x") is None)
    check("Get nonexistent alias", g.get_node_by_alias("x") is None)
    check("Empty neighbours", g.get_neighbours("x") == [])
    check("Empty render", g.render_fact_packet("x") is None)
    check("Empty edges for node", g.get_edges_for_node("x") == [])

    # Bounds enforcement on empty graph
    g.enforce_bounds()
    check("Bounds on empty graph OK", g.node_count == 0)


def test_remove_node():
    """Test node removal cascades to edges and aliases."""
    print("\n=== Test 16: Remove Node ===")

    g = fresh_graph()
    paula = g.get_node_by_alias("Paula")

    initial_edges = g.edge_count
    g.remove_node(paula.node_id)

    check("Node removed", g.get_node(paula.node_id) is None)
    check("Alias removed", g.get_node_by_alias("Paula") is None)
    check("Edges reduced", g.edge_count < initial_edges)


def test_relation_templates_complete():
    """Test that every allowed relation has a template."""
    print("\n=== Test 17: Relation Templates ===")

    for rel in ALLOWED_RELATIONS:
        check(f"Template for '{rel}'",
              rel in RELATION_TEMPLATES,
              f"missing template")


def test_metadata_defaults():
    """Test NodeMetadata defaults and serialization."""
    print("\n=== Test 18: Metadata Defaults ===")

    m = NodeMetadata()
    check("Default confidence 0.8", m.confidence == 0.8)
    check("Default usage 0", m.usage_count == 0)
    check("Default source user_assertion", m.source_type == "user_assertion")

    d = m.to_dict()
    m2 = NodeMetadata.from_dict(d)
    check("Metadata round-trip confidence", m2.confidence == m.confidence)
    check("Metadata round-trip source", m2.source_type == m.source_type)


def test_multiple_nodes_same_alias():
    """Test disambiguation when alias maps to multiple nodes."""
    print("\n=== Test 19: Ambiguous Alias ===")

    g = RelationalGraph()
    n1 = g.add_node(["Paula"], node_type="person")
    n2 = g.add_node(["Paula"], node_type="person")

    # Both should be in alias table
    candidates = g.alias_table.lookup("Paula")
    check("Two candidates for same alias",
          len(candidates) == 2,
          f"got {len(candidates)}")

    # get_node_by_alias returns one (most recent or first)
    result = g.get_node_by_alias("Paula")
    check("Disambiguation returns a node", result is not None)


def test_primary_alias():
    """Test primary_alias returns first alias."""
    print("\n=== Test 20: Primary Alias ===")

    g = RelationalGraph()
    node = g.add_node(["Dr. Paula Smith", "Paula", "my wife"])

    check("Primary alias is first",
          node.primary_alias() == "Dr. Paula Smith")


def test_graph_counts():
    """Test node_count and edge_count properties."""
    print("\n=== Test 21: Graph Counts ===")

    g = RelationalGraph()
    check("Initial nodes 0", g.node_count == 0)
    check("Initial edges 0", g.edge_count == 0)

    g.add_triple("Jeff", "spouse", "Paula")
    check("After triple: nodes", g.node_count == 2)
    check("After triple: edges", g.edge_count == 1)


def test_detect_assertions_husband():
    """Test assertion detection for 'husband' variant."""
    print("\n=== Test 22: Detect Husband Assertion ===")

    triples = RelationalGraph.detect_assertions("Jeff is my husband")
    check("Detects husband as spouse",
          len(triples) >= 1 and any(t.relation == "spouse" and t.object_ == "Jeff" for t in triples),
          f"got {[(t.relation, t.object_) for t in triples]}")


def test_detect_assertions_founded():
    """Test assertion detection for 'I founded X'."""
    print("\n=== Test 23: Detect Founded ===")

    triples = RelationalGraph.detect_assertions("I founded Halcyon AI Research")
    check("Detects founded",
          len(triples) >= 1 and any(t.relation == "founded" for t in triples),
          f"got {[(t.relation, t.object_) for t in triples]}")


def test_add_triple_rejects_node_ids():
    """add_triple() raises ValueError if passed node_ids instead of names."""
    print("\n=== Test 25: add_triple rejects node_ids ===")
    g = RelationalGraph()
    jeff = g.resolve_or_create_node("Jeff", node_type="person")
    paula = g.resolve_or_create_node("Paula", node_type="person")

    # Passing node_ids should raise
    try:
        g.add_triple(jeff, "spouse", paula)
        check("Rejects node_id args", False, "Should have raised ValueError")
    except ValueError as e:
        check("Rejects node_id args", "entity names" in str(e), str(e))

    # Passing names should work fine
    result = g.add_triple("Jeff", "spouse", "Paula")
    check("Accepts name args", result.result == TripleResult.CREATED)


def test_telemetry_backward_compat():
    """Test that new memory fields on DecisionTrace are backward compatible."""
    print("\n=== Test 25: Telemetry Backward Compat ===")

    from bob_core.telemetry import DecisionTrace

    # Old-style trace without memory fields
    trace = DecisionTrace(
        step=0, context_class=0, governance_state="EXPLORING",
        path="full", expert_ids=(1, 2), expert_invocations=2,
        tokens_processed=10, loss=1.5,
        routing_stability=0.9, debt_level=0.0, motif_survival=0.0,
        gate_passed=True, stability_passed=True, debt_passed=True,
        survival_passed=True,
    )
    d = trace.to_dict()
    check("No memory_nodes_active in old trace",
          "memory_nodes_active" not in d)
    check("No memory_bias_applied in old trace",
          "memory_bias_applied" not in d)

    # New-style trace with memory fields
    trace2 = DecisionTrace(
        step=1, context_class=0, governance_state="EXPLOITING",
        path="cheap", expert_ids=(1,), expert_invocations=1,
        tokens_processed=10, loss=1.2,
        routing_stability=0.95, debt_level=0.0, motif_survival=0.5,
        gate_passed=True, stability_passed=True, debt_passed=True,
        survival_passed=True,
        memory_nodes_active=3,
        memory_bias_applied=True,
        memory_bias_max=0.12,
    )
    d2 = trace2.to_dict()
    check("memory_nodes_active in new trace",
          d2.get("memory_nodes_active") == 3)
    check("memory_bias_applied in new trace",
          d2.get("memory_bias_applied") is True)
    check("memory_bias_max in new trace",
          abs(d2.get("memory_bias_max", 0) - 0.12) < 0.001)


# --- Runner ---

_ALL_TESTS = [
    test_node_crud,
    test_alias_normalisation,
    test_edge_crud,
    test_allowed_relations,
    test_contradiction_detection,
    test_duplicate_reinforcement,
    test_resolve_or_create,
    test_neighbour_traversal,
    test_render_fact_packet,
    test_detect_assertions,
    test_enforce_bounds,
    test_enforce_bounds_edge_cap,
    test_serialization_roundtrip,
    test_rebuild_alias_table,
    test_empty_graph,
    test_remove_node,
    test_relation_templates_complete,
    test_metadata_defaults,
    test_multiple_nodes_same_alias,
    test_primary_alias,
    test_graph_counts,
    test_detect_assertions_husband,
    test_detect_assertions_founded,
    test_telemetry_backward_compat,
]


def run_all_tests():
    """Script-mode runner. pytest users: just run `pytest test_memory_graph.py -v`."""
    global passed, failed, total
    passed = failed = total = 0

    print("=" * 60)
    print("Relational Graph Tests (Phase 8a)")
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
