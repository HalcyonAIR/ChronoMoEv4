# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright 2026 Halcyon AI Research (jeff@halcyon.ie)
"""
Relational graph: typed triples for stable world-facts.

The graph stores entities (people, places, organisations) and their
relationships as typed triples. It does NOT store preferences, episodes,
opinions, or transient state. A rich personal graph is 50-100 triples.

Alias table maps surface forms to node IDs via exact match (normalised).
No embedding search. No fuzzy matching.

Spec reference: docs/chronomoe_unified_memory_v1.md, Sections 3-7
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


# --- Allowed relations (stable world-facts only) ---

ALLOWED_RELATIONS = frozenset({
    # People
    "spouse", "child", "grandchild", "stepchild", "parent",
    "sibling", "colleague", "friend",
    # Roles
    "role_at", "reports_to", "manages",
    # Places
    "lives_in", "works_at", "studies_at", "located_in",
    # Organisations
    "runs", "member_of", "employed_by", "founded",
})

RELATION_TEMPLATES = {
    "spouse": "is married to",
    "child": "has child",
    "grandchild": "has grandchild",
    "stepchild": "has stepchild",
    "parent": "has parent",
    "sibling": "is a sibling of",
    "colleague": "is a colleague of",
    "friend": "is a friend of",
    "role_at": "has role at",
    "reports_to": "reports to",
    "manages": "manages",
    "lives_in": "lives in",
    "works_at": "works at",
    "studies_at": "studies at",
    "located_in": "is located in",
    "runs": "runs",
    "member_of": "is a member of",
    "employed_by": "is employed by",
    "founded": "founded",
}

# Family relation aliases for assertion detection
_FAMILY_MAP = {
    "wife": "spouse", "husband": "spouse", "spouse": "spouse", "partner": "spouse",
    "son": "child", "daughter": "child", "child": "child",
    "grandson": "grandchild", "granddaughter": "grandchild", "grandchild": "grandchild",
    "stepson": "stepchild", "stepdaughter": "stepchild", "stepchild": "stepchild",
    "father": "parent", "mother": "parent", "parent": "parent", "dad": "parent", "mom": "parent",
    "brother": "sibling", "sister": "sibling", "sibling": "sibling",
}


# --- Result type for triple operations ---

class TripleResult(Enum):
    CREATED = "created"
    REINFORCED = "reinforced"
    CONTRADICTION = "contradiction"
    INVALID_RELATION = "invalid_relation"


@dataclass
class TripleOutcome:
    """Result of an add_triple operation."""
    result: TripleResult
    edge: Optional["GraphEdge"] = None
    conflicting_edge: Optional["GraphEdge"] = None
    message: str = ""


# --- Data structures ---

@dataclass
class NodeMetadata:
    """Provenance and usage tracking for a graph node."""
    provenance_session: str = ""
    provenance_turn: int = 0
    source_type: str = "user_assertion"  # "user_assertion" | "trusted_channel"
    confidence: float = 0.8
    usage_count: int = 0
    last_confirmed: Optional[str] = None   # ISO datetime string
    last_activated: Optional[str] = None

    def to_dict(self) -> Dict:
        d = {
            "provenance_session": self.provenance_session,
            "provenance_turn": self.provenance_turn,
            "source_type": self.source_type,
            "confidence": round(self.confidence, 4),
            "usage_count": self.usage_count,
        }
        if self.last_confirmed is not None:
            d["last_confirmed"] = self.last_confirmed
        if self.last_activated is not None:
            d["last_activated"] = self.last_activated
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "NodeMetadata":
        return cls(
            provenance_session=d.get("provenance_session", ""),
            provenance_turn=d.get("provenance_turn", 0),
            source_type=d.get("source_type", "user_assertion"),
            confidence=d.get("confidence", 0.8),
            usage_count=d.get("usage_count", 0),
            last_confirmed=d.get("last_confirmed"),
            last_activated=d.get("last_activated"),
        )


@dataclass
class GraphNode:
    """An entity in the relational graph."""
    node_id: str                    # stable internal ID ("node_0001")
    aliases: List[str]              # surface forms ("Paula", "my wife")
    node_type: str                  # "person", "place", "organisation", "role", "unknown"
    metadata: NodeMetadata = field(default_factory=NodeMetadata)

    def primary_alias(self) -> str:
        """Return the first (canonical) alias."""
        return self.aliases[0] if self.aliases else self.node_id

    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "aliases": list(self.aliases),
            "node_type": self.node_type,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "GraphNode":
        return cls(
            node_id=d["node_id"],
            aliases=d["aliases"],
            node_type=d["node_type"],
            metadata=NodeMetadata.from_dict(d.get("metadata", {})),
        )


@dataclass
class GraphEdge:
    """A typed triple: subject -[relation]-> object."""
    subject_id: str
    relation: str
    object_id: str
    provenance_session: str = ""
    provenance_turn: int = 0
    confidence: float = 0.8
    usage_count: int = 0

    def to_dict(self) -> Dict:
        return {
            "subject_id": self.subject_id,
            "relation": self.relation,
            "object_id": self.object_id,
            "provenance_session": self.provenance_session,
            "provenance_turn": self.provenance_turn,
            "confidence": round(self.confidence, 4),
            "usage_count": self.usage_count,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "GraphEdge":
        return cls(
            subject_id=d["subject_id"],
            relation=d["relation"],
            object_id=d["object_id"],
            provenance_session=d.get("provenance_session", ""),
            provenance_turn=d.get("provenance_turn", 0),
            confidence=d.get("confidence", 0.8),
            usage_count=d.get("usage_count", 0),
        )


# --- Assertion detection ---

@dataclass
class DetectedTriple:
    """A candidate triple extracted from user text."""
    subject: str
    relation: str
    object_: str


# --- Alias Table ---

class AliasTable:
    """Maps surface strings to graph node IDs. Exact match with normalisation."""

    def __init__(self):
        self._table: Dict[str, List[str]] = {}

    def add(self, alias: str, node_id: str):
        key = self._normalise(alias)
        if key not in self._table:
            self._table[key] = []
        if node_id not in self._table[key]:
            self._table[key].append(node_id)

    def remove(self, alias: str, node_id: str):
        key = self._normalise(alias)
        if key in self._table:
            self._table[key] = [nid for nid in self._table[key] if nid != node_id]
            if not self._table[key]:
                del self._table[key]

    def lookup(self, token_span: str) -> List[str]:
        key = self._normalise(token_span)
        return list(self._table.get(key, []))

    def clear(self):
        self._table.clear()

    @staticmethod
    def _normalise(s: str) -> str:
        return s.strip().lower()


# --- Relational Graph ---

class RelationalGraph:
    """In-memory relational graph of entities and typed triples.

    Small by design: 50-200 nodes, 50-500 edges. Bounded.
    Only stores stable world-facts, not preferences or episodes.

    Args:
        max_nodes: Hard cap on node count. Default 200.
        max_edges: Hard cap on edge count. Default 500.
    """

    def __init__(self, max_nodes: int = 200, max_edges: int = 500):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.alias_table: AliasTable = AliasTable()
        self._max_nodes = max_nodes
        self._max_edges = max_edges
        self._next_id = 1

    def _generate_id(self) -> str:
        nid = f"node_{self._next_id:04d}"
        self._next_id += 1
        return nid

    # --- Node operations ---

    def add_node(
        self,
        aliases: List[str],
        node_type: str = "unknown",
        session: str = "",
        turn: int = 0,
    ) -> GraphNode:
        """Create a new node with given aliases. Returns the node."""
        node_id = self._generate_id()
        metadata = NodeMetadata(
            provenance_session=session,
            provenance_turn=turn,
        )
        node = GraphNode(
            node_id=node_id,
            aliases=list(aliases),
            node_type=node_type,
            metadata=metadata,
        )
        self.nodes[node_id] = node
        for alias in aliases:
            self.alias_table.add(alias, node_id)
        return node

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by its ID."""
        return self.nodes.get(node_id)

    def get_node_by_alias(self, alias: str) -> Optional[GraphNode]:
        """Get a node by any of its aliases. Returns first match."""
        candidates = self.alias_table.lookup(alias)
        if len(candidates) == 1:
            return self.nodes.get(candidates[0])
        elif len(candidates) > 1:
            # Return most recently confirmed
            best = None
            for nid in candidates:
                node = self.nodes.get(nid)
                if node and (best is None or
                        (node.metadata.last_confirmed or "") >
                        (best.metadata.last_confirmed or "")):
                    best = node
            return best
        return None

    def resolve_or_create_node(
        self,
        name: str,
        node_type: str = "unknown",
        session: str = "",
        turn: int = 0,
    ) -> str:
        """If an entity with this alias exists, return its ID. Otherwise create."""
        candidates = self.alias_table.lookup(name)
        if len(candidates) == 1:
            return candidates[0]
        elif len(candidates) > 1:
            # Return most recently confirmed
            best_id = candidates[0]
            best_time = ""
            for nid in candidates:
                node = self.nodes.get(nid)
                if node and (node.metadata.last_confirmed or "") > best_time:
                    best_time = node.metadata.last_confirmed or ""
                    best_id = nid
            return best_id
        else:
            node = self.add_node([name], node_type=node_type, session=session, turn=turn)
            return node.node_id

    def remove_node(self, node_id: str):
        """Remove a node and all its edges."""
        node = self.nodes.pop(node_id, None)
        if node:
            for alias in node.aliases:
                self.alias_table.remove(alias, node_id)
            self.edges = [
                e for e in self.edges
                if e.subject_id != node_id and e.object_id != node_id
            ]

    # --- Edge operations ---

    def add_triple(
        self,
        subject_name: str,
        relation: str,
        object_name: str,
        session: str = "",
        turn: int = 0,
    ) -> TripleOutcome:
        """Add a typed triple. Only user can create facts.

        Takes entity NAMES (e.g. "Jeff", "Paula"), not node IDs.
        Use resolve_or_create_node() first if you need IDs for other purposes.

        Returns TripleOutcome with result type and details.
        Contradiction: same subject+relation, different object → flagged.
        Duplicate: same triple → reinforced (usage_count++, confidence nudge).
        """
        # Guard: catch accidental node_id usage (silent duplicate creator)
        import re as _re
        _NODE_ID_PAT = _re.compile(r"^node_\d{4,}$")
        for name, role in [(subject_name, "subject"), (object_name, "object")]:
            if _NODE_ID_PAT.match(name):
                raise ValueError(
                    f"add_triple() takes entity names, not node IDs. "
                    f"Got {role}='{name}'. "
                    f"Use entity names like 'Jeff' or 'Paula' instead."
                )

        # Validate relation
        if relation not in ALLOWED_RELATIONS:
            return TripleOutcome(
                result=TripleResult.INVALID_RELATION,
                message=f"Relation '{relation}' not in allowed set",
            )

        # Resolve or create nodes
        subject_id = self.resolve_or_create_node(subject_name, session=session, turn=turn)
        object_id = self.resolve_or_create_node(object_name, session=session, turn=turn)

        # Check for contradiction: same subject+relation, different object
        for edge in self.edges:
            if edge.subject_id == subject_id and edge.relation == relation:
                if edge.object_id != object_id:
                    obj_node = self.nodes.get(edge.object_id)
                    obj_name = obj_node.primary_alias() if obj_node else edge.object_id
                    return TripleOutcome(
                        result=TripleResult.CONTRADICTION,
                        conflicting_edge=edge,
                        message=(
                            f"Existing: {subject_name} {relation} {obj_name}. "
                            f"Proposed: {subject_name} {relation} {object_name}. Confirm?"
                        ),
                    )

        # Check for duplicate: reinforce
        for edge in self.edges:
            if (edge.subject_id == subject_id and
                    edge.relation == relation and
                    edge.object_id == object_id):
                edge.usage_count += 1
                edge.confidence = min(1.0, edge.confidence + 0.05)
                return TripleOutcome(
                    result=TripleResult.REINFORCED,
                    edge=edge,
                    message="Triple reinforced",
                )

        # Create new edge
        edge = GraphEdge(
            subject_id=subject_id,
            relation=relation,
            object_id=object_id,
            provenance_session=session,
            provenance_turn=turn,
            confidence=0.8,
            usage_count=0,
        )
        self.edges.append(edge)
        return TripleOutcome(
            result=TripleResult.CREATED,
            edge=edge,
            message="Triple created",
        )

    def has_edge(self, subject_id: str, relation: str, object_id: str) -> bool:
        """Check if a specific triple exists."""
        return any(
            e.subject_id == subject_id and e.relation == relation and e.object_id == object_id
            for e in self.edges
        )

    def get_edges_for_node(self, node_id: str) -> List[GraphEdge]:
        """Get all edges where node is subject or object."""
        return [
            e for e in self.edges
            if e.subject_id == node_id or e.object_id == node_id
        ]

    def get_neighbours(self, node_id: str) -> List[Tuple[str, str]]:
        """Get 1-hop neighbours as (neighbour_id, relation) pairs."""
        neighbours = []
        for edge in self.edges:
            if edge.subject_id == node_id:
                neighbours.append((edge.object_id, edge.relation))
            elif edge.object_id == node_id:
                neighbours.append((edge.subject_id, edge.relation))
        return neighbours

    def get_primary_alias(self, node_id: str) -> str:
        """Get the canonical name for a node."""
        node = self.nodes.get(node_id)
        return node.primary_alias() if node else node_id

    # --- Rendering ---

    def render_fact_packet(self, node_id: str, max_facts: int = 5) -> Optional[str]:
        """Render known facts about a node as templated text.

        Only called on explicit factual queries. NOT during normal routing.
        Returns None if no facts exist.
        """
        edges = self.get_edges_for_node(node_id)
        if not edges:
            return None

        # Sort by confidence * (usage_count + 1) — most established first
        edges.sort(key=lambda e: e.confidence * (e.usage_count + 1), reverse=True)
        edges = edges[:max_facts]

        lines = []
        for edge in edges:
            subject_name = self.get_primary_alias(edge.subject_id)
            object_name = self.get_primary_alias(edge.object_id)
            relation_text = RELATION_TEMPLATES.get(edge.relation, edge.relation)
            lines.append(f"Fact: {subject_name} {relation_text} {object_name}.")

        return "\n".join(lines)

    # --- Assertion detection ---

    @staticmethod
    def detect_assertions(text: str) -> List[DetectedTriple]:
        """Extract candidate triples from user text via conservative pattern matching.

        Returns candidates for validation. Ambiguous cases should be confirmed.
        """
        candidates = []

        # "X is my {relation}"
        m = re.search(r"(\w+(?:\s+\w+)?)\s+is\s+my\s+(\w+)", text, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            role = m.group(2).strip().lower()
            if role in _FAMILY_MAP:
                candidates.append(DetectedTriple(
                    subject="user",
                    relation=_FAMILY_MAP[role],
                    object_=name,
                ))

        # "I work at X"
        m = re.search(r"I\s+work\s+at\s+(.+?)(?:\.|,|$)", text, re.IGNORECASE)
        if m:
            candidates.append(DetectedTriple(
                subject="user",
                relation="works_at",
                object_=m.group(1).strip(),
            ))

        # "I run X"
        m = re.search(r"I\s+run\s+(.+?)(?:\.|,|$)", text, re.IGNORECASE)
        if m:
            candidates.append(DetectedTriple(
                subject="user",
                relation="runs",
                object_=m.group(1).strip(),
            ))

        # "I study at X" / "I am studying at X"
        m = re.search(r"I\s+(?:study|am\s+studying)\s+at\s+(.+?)(?:\.|,|$)", text, re.IGNORECASE)
        if m:
            candidates.append(DetectedTriple(
                subject="user",
                relation="studies_at",
                object_=m.group(1).strip(),
            ))

        # "I live in X"
        m = re.search(r"I\s+live\s+in\s+(.+?)(?:\.|,|$)", text, re.IGNORECASE)
        if m:
            candidates.append(DetectedTriple(
                subject="user",
                relation="lives_in",
                object_=m.group(1).strip(),
            ))

        # "I founded X"
        m = re.search(r"I\s+founded\s+(.+?)(?:\.|,|$)", text, re.IGNORECASE)
        if m:
            candidates.append(DetectedTriple(
                subject="user",
                relation="founded",
                object_=m.group(1).strip(),
            ))

        return candidates

    # --- Bounds enforcement ---

    def enforce_bounds(
        self,
        max_nodes: Optional[int] = None,
        max_edges: Optional[int] = None,
        min_confidence: float = 0.3,
    ):
        """Prevent unbounded growth. Remove lowest-confidence entries."""
        max_n = max_nodes or self._max_nodes
        max_e = max_edges or self._max_edges

        # Remove edges below confidence threshold
        self.edges = [e for e in self.edges if e.confidence >= min_confidence]

        # If still over edge limit, remove least-used
        if len(self.edges) > max_e:
            self.edges.sort(key=lambda e: e.usage_count)
            self.edges = self.edges[-max_e:]

        # Remove orphan nodes (no edges)
        connected: Set[str] = set()
        for e in self.edges:
            connected.add(e.subject_id)
            connected.add(e.object_id)
        orphans = [nid for nid in self.nodes if nid not in connected]
        for nid in orphans:
            self.remove_node(nid)

        # Hard node cap
        if len(self.nodes) > max_n:
            nodes_by_recency = sorted(
                self.nodes.values(),
                key=lambda n: n.metadata.last_activated or "",
            )
            to_remove = [n.node_id for n in nodes_by_recency[:len(self.nodes) - max_n]]
            for nid in to_remove:
                self.remove_node(nid)

    # --- Serialization ---

    def to_dict(self) -> Dict:
        """Serialize graph to a JSON-compatible dict."""
        return {
            "version": "0.4",
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
            "next_id": self._next_id,
            "max_nodes": self._max_nodes,
            "max_edges": self._max_edges,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "RelationalGraph":
        """Deserialize graph from a dict."""
        graph = cls(
            max_nodes=d.get("max_nodes", 200),
            max_edges=d.get("max_edges", 500),
        )
        graph._next_id = d.get("next_id", 1)

        for nd in d.get("nodes", []):
            node = GraphNode.from_dict(nd)
            graph.nodes[node.node_id] = node

        for ed in d.get("edges", []):
            graph.edges.append(GraphEdge.from_dict(ed))

        graph.rebuild_alias_table()
        return graph

    def rebuild_alias_table(self):
        """Reconstruct alias table from current nodes."""
        self.alias_table.clear()
        for node in self.nodes.values():
            for alias in node.aliases:
                self.alias_table.add(alias, node.node_id)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)
