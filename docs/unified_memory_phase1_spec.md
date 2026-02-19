# ChronoMoE: Unified Memory Prototype — Phase 1

## Implementation Spec for CC

**Authors:** Jeff, Claude, Halcyon
**Date:** February 2026
**Status:** Ready for implementation after triad monitor traces are validated
**Prerequisite:** Triad monitors (monitors.py, conflict.py) producing stable traces

---

> *Bob doesn't remember conversations. He navigates a terrain that was sculpted by them, and he knows the names of the landmarks because someone told him and he wrote it down.*

---

## 1. What This Is

A memory system where memory is not retrieved. It is already present as a routing bias field. No RAG. No similarity search. No token injection during normal operation. No context window consumption.

When the user says "Paula," Bob doesn't search a database. The routing landscape was already shaped by Paula's existence. The cue activates a region of that landscape. Routing flows through it.

**Phase 1 scope:** Two separable toggles, run independently and compared.

**Toggle A — Graph Only:** Relational graph in RAM, alias-based entity linking, templated rendering on explicit factual queries. Memory is stored and retrievable but inert. It does not influence routing.

**Toggle B — Graph + Routing Bias:** Everything in Toggle A, plus association basins, activation diffusion, and pre-softmax routing bias. Memory influences expert selection through additive bias fields. Same mechanism as scars and the governor — proven infrastructure.

No cross-attention. No persistent tensor. No new transformer components in either toggle.

**The question Phase 1 answers:** Can sparse symbolic activation produce stable, measurable perturbations in a MoE routing manifold without destabilising consequence geometry?

**What Phase 1 does NOT answer:** Questions about identity, selfhood, or cognitive architecture. Phase 1 proves influence. Not emergence. Not personality. Influence. Stay there.

---

## 2. Architecture Overview

### Toggle A: Graph Only

```
Token stream
    │
    ▼
┌──────────────┐     ┌──────────────────────────┐
│  Alias Table  │────▶│  Entity Linking            │
│  (exact match)│     │  Maps tokens → node IDs   │
└──────────────┘     └──────────┬───────────────┘
                                │
                                ▼
                     ┌──────────────────────────┐
                     │  Relational Graph (RAM)   │
                     │  Typed triples + metadata  │
                     └──────────┬───────────────┘
                                │
                                ▼ (only on explicit factual query)
                     ┌──────────────────────────┐
                     │  Templated Rendering      │
                     │  "Fact: Paula is Jeff's    │
                     │   wife."                   │
                     └──────────────────────────┘
                     
Normal routing is UNAFFECTED. Memory is inert storage.
Facts render as tokens only when user explicitly asks.
```

### Toggle B: Graph + Routing Bias

```
Token stream
    │
    ▼
┌──────────────┐     ┌──────────────────────────┐
│  Alias Table  │────▶│  Activation Gate          │
│  (exact match)│     │  Maps tokens → node IDs   │
└──────────────┘     └──────────┬───────────────┘
                                │
                                ▼
                     ┌──────────────────────────┐
                     │  Relational Graph (RAM)   │
                     │  Typed triples + metadata  │
                     │  Nodes have routing sigs   │
                     └──────────┬───────────────┘
                                │
                                ▼
                     ┌──────────────────────────┐
                     │  Activation Diffusion     │
                     │  Spread over 1-hop        │
                     │  Decay with graph distance │
                     └──────────┬───────────────┘
                                │
                                ▼
                     ┌──────────────────────────┐
                     │  Routing Bias Field       │
                     │  Pre-softmax logit adjust  │
                     │  Per activated node        │
                     │  bias_scale ≤ 0.2 (HARD)  │
                     └──────────────────────────┘
                                │
                                ▼
                     Normal routing + monitors + clocks + governor

The transformer sees routing logits shaped by memory.
It does not know memory exists. Everything downstream
operates on the composite routing distribution unchanged.
```

**The experiment:** Run identical prompts through both toggles. Measure whether Toggle B produces stable routing perturbations that Toggle A does not. See Section 15 for protocol.

---

## 3. The Relational Graph

### 3.1 Data Structure

```python
class GraphNode:
    node_id: str              # stable internal ID (e.g., "node_1842")
    aliases: List[str]        # surface forms ("Paula", "my wife")
    node_type: str            # "person", "place", "organisation", "role"
    routing_signature: Optional[np.ndarray]  # learned from association basins
    metadata: NodeMetadata

class NodeMetadata:
    provenance_session: str   # which session created this node
    provenance_turn: int      # which turn
    source_type: str          # "user_assertion" | "trusted_channel"
    confidence: float         # 0.0 - 1.0, initial 0.8
    usage_count: int          # how many times activated
    last_confirmed: datetime  # last time user reaffirmed or used
    last_activated: datetime  # last time cue-triggered

class GraphEdge:
    subject_id: str           # node_id of subject
    relation: str             # typed relation (see allowed list)
    object_id: str            # node_id of object
    provenance_session: str
    provenance_turn: int
    confidence: float
    usage_count: int

class RelationalGraph:
    nodes: Dict[str, GraphNode]
    edges: List[GraphEdge]
    alias_table: Dict[str, List[str]]  # surface form → [node_ids]
```

### 3.2 Allowed Relations (Stable World-Facts Only)

```python
ALLOWED_RELATIONS = {
    # People
    "spouse", "child", "grandchild", "stepchild", "parent",
    "sibling", "colleague", "friend",
    
    # Roles
    "role_at", "reports_to", "manages",
    
    # Places
    "lives_in", "works_at", "studies_at", "located_in",
    
    # Organisations
    "runs", "member_of", "employed_by", "founded",
}
```

### 3.3 What Is NOT Stored

```python
# NEVER store these in the graph
EXCLUDED = {
    "preferences",    # "likes X" → lives in geometric layer as safe basins
    "episodes",       # "said X on Tuesday" → ephemeral, consequence captured by scars
    "opinions",       # "thinks X about Y" → within-run context only
    "tasks",          # "working on X" → transient
    "transient_state" # "currently in Australia" → session context only
}
```

**Why:** Preferences and episodes grow without bound, need temporal queries, and require relevance ranking. That's RAG territory. The graph stays small by storing only slow-changing world-facts. A rich personal graph is 50-100 triples.

---

## 4. Alias Table and Entity Linking

### 4.1 The Alias Table

```python
class AliasTable:
    """
    Maps surface strings to graph node IDs.
    Exact match (with normalisation), not embedding search.
    """
    
    def __init__(self):
        self._table: Dict[str, List[str]] = {}  # normalised_alias → [node_ids]
    
    def add(self, alias: str, node_id: str):
        key = self._normalise(alias)
        if key not in self._table:
            self._table[key] = []
        if node_id not in self._table[key]:
            self._table[key].append(node_id)
    
    def lookup(self, token_span: str) -> List[str]:
        key = self._normalise(token_span)
        return self._table.get(key, [])
    
    def _normalise(self, s: str) -> str:
        return s.strip().lower()
```

### 4.2 Entity Linking Per Turn

```python
def link_entities(token_spans: List[str], alias_table: AliasTable, 
                  graph: RelationalGraph) -> Dict[str, float]:
    """
    Scan input tokens for entity mentions.
    Returns activation map: node_id → activation_strength.
    
    No embedding search. No fuzzy matching. Exact alias lookup.
    """
    activations = {}
    
    for span in token_spans:
        candidates = alias_table.lookup(span)
        
        if len(candidates) == 1:
            # Unambiguous: activate directly
            activations[candidates[0]] = 1.0
            
        elif len(candidates) > 1:
            # Ambiguous: use co-occurrence with other activated nodes
            # to disambiguate (e.g., "Paula" near "wife" → spouse node)
            best = disambiguate(candidates, activations, graph)
            if best is not None:
                activations[best] = 0.8  # slightly lower confidence
        
        # len == 0: unknown entity, skip
    
    return activations
```

### 4.3 Disambiguation

```python
def disambiguate(candidates: List[str], current_activations: Dict[str, float],
                 graph: RelationalGraph) -> Optional[str]:
    """
    When an alias maps to multiple nodes, pick the one most connected
    to already-activated nodes. Simple graph proximity, not embeddings.
    """
    if not current_activations:
        return None  # can't disambiguate without context
    
    best_id = None
    best_score = -1
    
    for candidate in candidates:
        # Count edges between this candidate and already-active nodes
        score = 0
        for active_id in current_activations:
            if graph.has_edge(candidate, active_id):
                score += current_activations[active_id]
        
        if score > best_score:
            best_score = score
            best_id = candidate
    
    return best_id if best_score > 0 else None
```

---

## 5. Activation Diffusion

Entity linking produces a sparse activation vector over graph nodes. Diffusion spreads that activation to the 1-hop neighbourhood with decay.

### 5.1 Diffusion Function

```python
def diffuse_activation(activations: Dict[str, float], graph: RelationalGraph,
                       depth: int = 1, decay: float = 0.3) -> Dict[str, float]:
    """
    Spread activation from directly-mentioned entities to their neighbours.
    
    depth=1: only immediate neighbours (sufficient for Phase 1)
    decay=0.3: neighbours get 30% of the source activation
    
    This is cue-triggered pattern completion.
    Not global recall.
    """
    diffused = dict(activations)  # start with direct activations
    
    for hop in range(depth):
        new_activations = {}
        hop_decay = decay ** (hop + 1)
        
        for node_id, strength in list(diffused.items()):
            # Get 1-hop neighbours
            neighbours = graph.get_neighbours(node_id)
            
            for neighbour_id, relation in neighbours:
                neighbour_strength = strength * hop_decay
                
                # Take max if already activated (don't stack)
                if neighbour_id in new_activations:
                    new_activations[neighbour_id] = max(
                        new_activations[neighbour_id], 
                        neighbour_strength
                    )
                else:
                    new_activations[neighbour_id] = neighbour_strength
        
        # Merge new activations (max, not sum, to prevent runaway)
        for node_id, strength in new_activations.items():
            if node_id not in diffused:
                diffused[node_id] = strength
            else:
                diffused[node_id] = max(diffused[node_id], strength)
    
    return diffused
```

### 5.2 Properties

- **Sparse:** Only nodes near entity mentions activate. Most of the graph stays at zero.
- **Bounded:** Max activation is 1.0 (direct mention). Neighbours get at most 0.3. 2-hop would get 0.09. Falls off fast.
- **No runaway:** Using max instead of sum prevents activation from stacking when multiple paths converge on the same node.
- **Cheap:** 1-hop diffusion over a 50-100 node graph is microseconds.

---

## 6. Routing Bias Field

Activated nodes produce pre-softmax logit adjustments. This is the same mechanism as scars — proven infrastructure.

### 6.1 Association Basins

Each node can accumulate a **routing signature** — a record of which experts handled that entity's context well in past sessions.

```python
class AssociationBasin:
    """
    Routing signature for an entity context.
    Records which experts were active when this entity was relevant
    and outcomes were positive.
    """
    node_id: str
    bias_vector: np.ndarray    # shape: (num_layers, num_experts)
                                # positive = historically good expert for this context
                                # negative = historically bad
                                # zero = no data
    strength: float             # overall confidence in this basin
    update_count: int           # how many consolidation cycles contributed
    last_updated: datetime
```

### 6.2 Applying Memory Bias

```python
def apply_memory_bias(router_logits: np.ndarray, 
                      activations: Dict[str, float],
                      association_basins: Dict[str, AssociationBasin],
                      layer_idx: int,
                      bias_scale: float = 0.1) -> np.ndarray:
    """
    Modify pre-softmax router logits based on activated memory nodes.
    
    This is the core memory-to-routing interface.
    Same mechanism as scar bias. Proven infrastructure.
    
    bias_scale controls overall memory influence strength.
    Start conservative (0.1) and tune from traces.
    
    Args:
        router_logits: shape (num_experts,) — raw logits before softmax
        activations: node_id → activation_strength from diffusion
        association_basins: node_id → AssociationBasin
        layer_idx: current layer index
        bias_scale: global scaling factor for memory influence
    
    Returns:
        modified router_logits
    """
    memory_bias = np.zeros_like(router_logits)
    
    for node_id, activation_strength in activations.items():
        basin = association_basins.get(node_id)
        if basin is not None and basin.bias_vector is not None:
            # Weighted contribution: activation strength × basin confidence × bias
            memory_bias += (
                activation_strength 
                * basin.strength 
                * basin.bias_vector[layer_idx]
            )
    
    # Scale and apply
    # HARD CEILING: bias_scale must never exceed 0.2
    # Memory whispers. Consequence shouts.
    # If signal isn't visible at 0.2, the mechanism is too weak
    # and Phase 2 (cross-attention) is needed.
    # If visible at 0.1, leave it there.
    assert bias_scale <= 0.2, "Memory bias must not exceed hard ceiling of 0.2"
    router_logits = router_logits + bias_scale * memory_bias
    
    # MAGNITUDE DIAGNOSTICS (always log, not optional)
    # Without these you are tuning blind.
    logit_std = np.std(router_logits - bias_scale * memory_bias)  # pre-bias std
    bias_magnitude = np.abs(bias_scale * memory_bias)
    diagnostics = {
        "mean_abs_memory_bias": float(np.mean(bias_magnitude)),
        "max_abs_memory_bias":  float(np.max(bias_magnitude)),
        "bias_to_logit_ratio":  float(np.mean(bias_magnitude) / (logit_std + 1e-8)),
        "n_active_nodes":       sum(1 for s in activations.values() if s > 0),
    }
    # If bias_to_logit_ratio < 0.01: whispering so quietly you may never detect signal
    # If bias_to_logit_ratio > 0.5: lying to yourself about "whisper"
    # Sweet spot: 0.05 - 0.2
    
    return router_logits, diagnostics
```

### 6.3 Properties

- **Pre-softmax:** Bias is added before softmax normalisation. The routing distribution shifts but remains a valid probability distribution.
- **Proportional to activation:** Direct entity mentions have full influence. Diffused neighbours have 30% influence. Distant or unmentioned entities have zero. Irrelevant memory stays silent.
- **Scaled conservatively:** `bias_scale = 0.1` means memory is a gentle nudge, not a takeover. Tune from traces — if routing doesn't change measurably, increase. If routing becomes unstable, decrease.
- **Per-layer:** Different layers may have different association patterns. Layer 2 might care about different experts for "Paula" context than layer 20.
- **Composable with scars:** Memory bias and scar bias are both pre-softmax adjustments. They add linearly. A routing region that has both a memory association AND a scar will feel both influences. This is correct — "I know about Paula AND I have a bad experience in this routing region" should both matter.

---

## 7. Triple Management

### 7.1 Adding Facts (User-Asserted Only)

```python
def add_triple(graph: RelationalGraph, subject: str, relation: str, 
               object_: str, session_id: str, turn_id: int) -> Result:
    """
    Only the user can create new triples.
    The model cannot mint facts from inference.
    """
    
    # Validate relation type
    if relation not in ALLOWED_RELATIONS:
        return Result.reject(f"Relation '{relation}' not in allowed set")
    
    # Resolve or create subject node
    subject_id = resolve_or_create_node(graph, subject)
    object_id = resolve_or_create_node(graph, object_)
    
    # Check for contradictions
    existing = graph.get_edges(subject_id, relation)
    for edge in existing:
        if edge.object_id != object_id:
            return Result.contradiction(
                existing=edge,
                proposed=(subject_id, relation, object_id),
                message=f"Existing: {subject} -{relation}-> {graph.get_node(edge.object_id).aliases[0]}. "
                        f"Proposed: {subject} -{relation}-> {object_}. Confirm?"
            )
    
    # Check for duplicate
    if graph.has_edge(subject_id, relation, object_id):
        # Reinforce existing
        edge = graph.get_edge(subject_id, relation, object_id)
        edge.usage_count += 1
        edge.confidence = min(1.0, edge.confidence + 0.05)
        edge.last_confirmed = now()
        return Result.reinforced(edge)
    
    # Create new edge
    edge = GraphEdge(
        subject_id=subject_id,
        relation=relation,
        object_id=object_id,
        provenance_session=session_id,
        provenance_turn=turn_id,
        confidence=0.8,
        usage_count=0
    )
    graph.add_edge(edge)
    
    # Update alias table
    graph.alias_table.add(subject, subject_id)
    graph.alias_table.add(object_, object_id)
    
    return Result.created(edge)


def resolve_or_create_node(graph: RelationalGraph, name: str) -> str:
    """
    If an entity with this alias exists, return its ID.
    Otherwise create a new node.
    """
    candidates = graph.alias_table.lookup(name)
    
    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        # Ambiguous — return the most recently confirmed
        return max(candidates, 
                   key=lambda nid: graph.nodes[nid].metadata.last_confirmed)
    else:
        # New entity
        node = GraphNode(
            node_id=generate_id(),
            aliases=[name],
            node_type=infer_type(name),  # simple heuristic or default "unknown"
            routing_signature=None,
            metadata=NodeMetadata(
                confidence=0.8,
                usage_count=0,
                last_confirmed=now(),
                last_activated=None
            )
        )
        graph.add_node(node)
        graph.alias_table.add(name, node.node_id)
        return node.node_id
```

### 7.2 Detecting Assertions in User Input

```python
def detect_assertions(user_text: str) -> List[Triple]:
    """
    Simple pattern matching for declarative statements.
    NOT full NLP. Just common relational patterns.
    
    "Paula is my wife" → (user, spouse, Paula)
    "I work at ALS" → (user, works_at, ALS)
    "Brogan is my grandson" → (user, grandchild, Brogan)
    
    Returns candidate triples for validation.
    Ambiguous cases should be confirmed with the user.
    """
    patterns = [
        # Spouse
        (r"(\w+)\s+is\s+my\s+(wife|husband|spouse|partner)",
         lambda m: Triple(subject="user", relation="spouse", object_=m.group(1))),
        
        # Children/family
        (r"(\w+)\s+is\s+my\s+(son|daughter|child|grandson|granddaughter|grandchild)",
         lambda m: Triple(subject="user", relation=map_family_relation(m.group(2)), 
                          object_=m.group(1))),
        
        # Workplace
        (r"I\s+work\s+at\s+(.+?)(?:\.|$)",
         lambda m: Triple(subject="user", relation="works_at", object_=m.group(1).strip())),
        
        # Runs/founded
        (r"I\s+run\s+(.+?)(?:\.|$)",
         lambda m: Triple(subject="user", relation="runs", object_=m.group(1).strip())),
        
        # Studies
        (r"I\s+(?:study|am studying)\s+at\s+(.+?)(?:\.|$)",
         lambda m: Triple(subject="user", relation="studies_at", object_=m.group(1).strip())),
        
        # Lives
        (r"I\s+live\s+in\s+(.+?)(?:\.|$)",
         lambda m: Triple(subject="user", relation="lives_in", object_=m.group(1).strip())),
    ]
    
    candidates = []
    for pattern, extractor in patterns:
        match = re.search(pattern, user_text, re.IGNORECASE)
        if match:
            candidates.append(extractor(match))
    
    return candidates
```

**IMPORTANT:** Assertion detection is conservative. When in doubt, don't create a triple — ask the user. False negatives (missing a fact) are recoverable. False positives (creating a wrong fact) corrupt the graph.

---

## 8. Templated Rendering (Explicit Queries Only)

When the user asks a direct factual question, render triples as minimal templated text. This is the ONLY time tokens are injected for memory purposes.

```python
def render_fact_packet(node_id: str, graph: RelationalGraph, 
                       max_facts: int = 5) -> Optional[str]:
    """
    Deterministic rendering of structured triples.
    NOT retrieved text. NOT RAG. Fixed template, structured data.
    
    Only called when the user explicitly asks a factual question
    (e.g., "who is Paula?", "where do I work?").
    
    During normal conversation, memory influences routing silently.
    """
    edges = graph.get_edges_for_node(node_id)
    
    if not edges:
        return None
    
    # Sort by confidence × usage_count (most established facts first)
    edges.sort(key=lambda e: e.confidence * (e.usage_count + 1), reverse=True)
    edges = edges[:max_facts]
    
    lines = []
    for edge in edges:
        subject_name = graph.get_primary_alias(edge.subject_id)
        object_name = graph.get_primary_alias(edge.object_id)
        relation_text = RELATION_TEMPLATES[edge.relation]
        
        lines.append(f"Fact: {subject_name} {relation_text} {object_name}.")
    
    return "\n".join(lines)


RELATION_TEMPLATES = {
    "spouse": "is married to",
    "child": "has child",
    "grandchild": "has grandchild",
    "stepchild": "has stepchild",
    "works_at": "works at",
    "runs": "runs",
    "studies_at": "studies at",
    "lives_in": "lives in",
    "role_at": "has role at",
    "colleague": "is a colleague of",
    # ... etc
}
```

---

## 9. Consolidation Loop

Event-driven. Runs between prompts. Compiles routing priors, does NOT create new facts.

### 9.1 Schedule

```python
def consolidation_schedule(turn_events: TurnEvents) -> int:
    """
    Budget scales with consequence. Hard cap prevents rumination.
    """
    budget = 5  # baseline: alias refresh, triple decay
    
    if turn_events.scars_formed > 0:
        budget += 10 * turn_events.scars_formed
    
    if turn_events.expansions_formed > 0:
        budget += 10 * turn_events.expansions_formed
    
    if turn_events.mode_b_engaged:
        budget += 15
    
    return min(budget, 50)  # HARD CAP
```

### 9.2 Consolidation Actions

```python
def consolidate(graph: RelationalGraph, 
                association_basins: Dict[str, AssociationBasin],
                turn_activations: Dict[str, float],
                turn_routing_traces: List[LayerSnapshot],
                turn_events: TurnEvents,
                budget: int):
    """
    The hive buzzes. But it only compiles. It never invents.
    """
    steps_used = 0
    
    # 1. REINFORCE activated triples (always, cheap)
    for node_id in turn_activations:
        node = graph.nodes.get(node_id)
        if node is not None:
            node.metadata.usage_count += 1
            node.metadata.last_activated = now()
            steps_used += 1
            if steps_used >= budget:
                return
    
    # 2. DECAY unused triples (always, cheap)
    for node_id, node in graph.nodes.items():
        if node_id not in turn_activations:
            # Gentle decay on confidence for nodes not mentioned
            sessions_since = sessions_since_last_activation(node)
            if sessions_since > 5:
                node.metadata.confidence *= 0.99  # very gentle
            steps_used += 1
            if steps_used >= budget:
                return
    
    # 3. UPDATE association basins (if budget allows)
    if turn_events.scars_formed > 0 or turn_events.expansions_formed > 0:
        for node_id, activation in turn_activations.items():
            if activation > 0.5:  # only strongly activated entities
                basin = association_basins.get(node_id)
                if basin is None:
                    basin = AssociationBasin(
                        node_id=node_id,
                        bias_vector=np.zeros((num_layers, num_experts)),
                        strength=0.1,
                        update_count=0,
                        last_updated=now()
                    )
                    association_basins[node_id] = basin
                
                # Extract routing signature from this turn's traces
                current_sig = extract_routing_signature(turn_routing_traces)
                
                # Blend with existing (slow EMA)
                basin.bias_vector = (
                    0.9 * basin.bias_vector + 0.1 * current_sig
                )
                
                # If scar formed while this entity was active, 
                # mark the basin with caution (reduce strength for 
                # the experts that were dominant at scar coordinates)
                if turn_events.entity_was_active_at_scar(node_id):
                    scar_signature = extract_scar_signature(turn_events)
                    basin.bias_vector -= 0.05 * scar_signature
                
                basin.update_count += 1
                basin.last_updated = now()
                steps_used += 1
                if steps_used >= budget:
                    return
    
    # 4. REFRESH alias cache (if budget allows)
    # ... rebuild lookup optimisations, merge similar aliases, etc.
```

### 9.3 What Consolidation CANNOT Do

```python
# FORBIDDEN during consolidation:
# - Create new triples (only user can do that)
# - Create new nodes (only user assertion triggers this)
# - Trigger tool actions
# - Generate text output
# - Modify graph structure (only reinforce/decay existing entries)
# - Exceed budget
```

---

## 10. Multi-Entity Activation

When multiple entities are mentioned simultaneously:

```
"Given what you know about Paula and ALS Loughrea..."
```

### 10.1 Linear Blend

```python
def compute_composite_bias(activations: Dict[str, float],
                           association_basins: Dict[str, AssociationBasin],
                           layer_idx: int) -> np.ndarray:
    """
    Multiple entity activations blend linearly.
    Each weighted by its activation strength.
    """
    composite = np.zeros(num_experts)
    
    for node_id, strength in activations.items():
        basin = association_basins.get(node_id)
        if basin is not None:
            composite += strength * basin.strength * basin.bias_vector[layer_idx]
    
    return composite
```

### 10.2 Memory-Induced Conflict

If two entity contexts pull routing in different directions, the composite bias creates tension in the routing distribution. The monitors detect this:

- Angel sees optionality shifting as competing biases push toward different experts
- Devil may detect confidence acceleration if one entity's bias dominates
- conflict_index rises if angel and devil both activate

**Mode B may engage from memory-induced tension.** This is memory participating in pressure dynamics — not passive, but integrated into the monitor architecture.

No special handling needed. The existing triad monitors observe the composite routing distribution. Memory-induced conflict looks identical to any other routing conflict from the monitors' perspective. The architecture handles it automatically.

---

## 11. Monitor Integration

### 11.1 Alien Complexity Gate

Fact packet activation counts as a complexity signal:

```python
def update_complexity_gate(activations: Dict[str, float], 
                           complexity_signals: ComplexitySignals):
    """
    Active entity associations contribute to the Alien's complexity gate.
    """
    n_active_entities = sum(1 for s in activations.values() if s > 0.3)
    
    if n_active_entities >= 2:
        complexity_signals.entity_complexity = True
    # Multiple entities active = complex relational context
    # If monitors are silent despite this, Alien should audit
```

### 11.2 Entity-Tagged Scars

When a scar forms and entity associations were active:

```python
def tag_scar_with_entities(scar: Scar, activations: Dict[str, float]):
    """
    Link scars to the entity context that was active when they formed.
    Next time these entities activate, the scar field includes 
    entity-specific danger basins.
    """
    active_entities = [
        node_id for node_id, strength in activations.items() 
        if strength > 0.5
    ]
    scar.entity_tags = active_entities
```

---

## 12. Persistence

### 12.1 What Gets Saved to Disk

At session end:

```python
def save_memory_state(graph: RelationalGraph, 
                      association_basins: Dict[str, AssociationBasin],
                      filepath: str):
    """
    Entire memory state serialised. Small.
    Graph: 50-100 nodes, 50-100 edges, few KB.
    Basins: one array per node per layer, bounded by graph size.
    """
    state = {
        "graph": serialise_graph(graph),
        "basins": serialise_basins(association_basins),
        "version": "0.4",
        "timestamp": now()
    }
    save_json(state, filepath)
```

### 12.2 What Gets Loaded at Session Start

```python
def load_memory_state(filepath: str) -> Tuple[RelationalGraph, Dict]:
    """
    Load graph and basins. Rebuild alias table from graph nodes.
    Memory is immediately available — no warmup needed for facts.
    Association basins provide routing bias from first token.
    """
    state = load_json(filepath)
    graph = deserialise_graph(state["graph"])
    basins = deserialise_basins(state["basins"])
    
    # Rebuild alias table
    graph.rebuild_alias_table()
    
    return graph, basins
```

### 12.3 Bounded Size

```python
MAX_NODES = 200
MAX_EDGES = 500
MIN_CONFIDENCE = 0.3

def enforce_bounds(graph: RelationalGraph):
    """
    Prevent unbounded growth. Remove lowest-confidence entries.
    """
    # Remove edges below confidence threshold
    graph.edges = [e for e in graph.edges if e.confidence >= MIN_CONFIDENCE]
    
    # If still over limit, remove least-used
    if len(graph.edges) > MAX_EDGES:
        graph.edges.sort(key=lambda e: e.usage_count)
        graph.edges = graph.edges[-MAX_EDGES:]
    
    # Remove orphan nodes (no edges)
    connected = set()
    for e in graph.edges:
        connected.add(e.subject_id)
        connected.add(e.object_id)
    graph.nodes = {nid: n for nid, n in graph.nodes.items() if nid in connected}
    
    # Hard node cap
    if len(graph.nodes) > MAX_NODES:
        # Remove least-recently-activated nodes
        nodes_by_recency = sorted(
            graph.nodes.values(),
            key=lambda n: n.metadata.last_activated or datetime.min
        )
        to_remove = [n.node_id for n in nodes_by_recency[:len(graph.nodes) - MAX_NODES]]
        for nid in to_remove:
            graph.remove_node(nid)
```

---

## 13. Poison Resistance

### 13.1 Validation Rules

1. **Only user-asserted triples.** The model cannot create facts from inference. "Paula" appearing near "wedding" does not create a `spouse` triple.

2. **Contradiction checking.** New triples that conflict with existing ones are flagged, not silently accepted.

3. **Provenance on everything.** Every node and edge traces back to session_id, turn_id, source_type.

4. **Typed relations only.** No freeform relation strings. Only relations in `ALLOWED_RELATIONS`.

5. **Alias normalisation.** Surface forms are normalised before storage. No invisible characters, no Unicode tricks.

6. **Confidence gating.** New triples start at 0.8 confidence, not 1.0. They must be reinforced through usage to reach full confidence.

### 13.2 Contradiction Handling

```python
def handle_contradiction(existing: GraphEdge, proposed: Triple, 
                         graph: RelationalGraph) -> Response:
    """
    When a new assertion contradicts an existing fact:
    - Do NOT silently overwrite
    - Present both to the user
    - Let the user resolve
    """
    existing_obj = graph.get_primary_alias(existing.object_id)
    
    return Response.ask_user(
        f"I have '{proposed.subject} {proposed.relation} {existing_obj}' "
        f"(confidence: {existing.confidence:.0%}, used {existing.usage_count} times). "
        f"You're saying '{proposed.subject} {proposed.relation} {proposed.object_}'. "
        f"Should I update this?"
    )
```

---

## 14. Implementation Checklist

### Phase 8a: Graph Infrastructure (Toggle A)
- [ ] Implement `RelationalGraph`, `GraphNode`, `GraphEdge` data structures
- [ ] Implement `AliasTable` with normalisation
- [ ] Implement `add_triple` with contradiction checking
- [ ] Implement `detect_assertions` (conservative pattern matching)
- [ ] Implement `resolve_or_create_node`
- [ ] Serialisation/deserialisation to JSON
- [ ] `enforce_bounds` for size management
- [ ] Implement `render_fact_packet` with `RELATION_TEMPLATES`
- [ ] Detection of explicit factual queries (when to render)
- [ ] Implement behind flag: `--enable-declarative-graph`
- [ ] **Toggle A is independently testable. Ship and validate before Phase 8b.**

### Phase 8b: Activation and Routing Bias (Toggle B, requires 8a)
- [ ] Implement `link_entities` (scan tokens against alias table)
- [ ] Implement `diffuse_activation` (1-hop, decay=0.3)
- [ ] Implement `AssociationBasin` data structure
- [ ] Implement `apply_memory_bias` (pre-softmax logit adjustment)
- [ ] Wire into existing router: bias applied after normal logit computation, before softmax
- [ ] `bias_scale` as configurable parameter (start at 0.1, HARD CEILING 0.2)
- [ ] Implement behind flag: `--enable-memory-routing-bias`

### Phase 8c: Two-Toggle Comparison (CRITICAL EXPERIMENT)
- [ ] Prepare test prompts with BOTH high-semantic entities (Paula, ALS) and synthetic entities (Zorblax, Krenthar)
- [ ] Select 3–5 non-adjacent seeds (e.g., {42, 137, 256, 1729, 8191})
- [ ] **For EACH seed, run ALL conditions below:**
- [ ] Run Condition 0 (no memory)
- [ ] Run Toggle A (graph only)
- [ ] Run B1 (graph + bias, scale=0.0) — **must equal Condition 0 or wiring is broken**
- [ ] Run B2 (graph + bias, scale=0.1)
- [ ] Run B3 (graph + bias, scale=0.2)
- [ ] All conditions: same prompts, same seeds, 100+ turns
- [ ] Measure: layer-wise cosine distance, N_eff, conflict_index, loss, magnitude diagnostics
- [ ] **Run Test 15.6 with quantitative thresholds: scar cosine > 0.85, Δcount < 20%, Δrates < 15%, temporal overlap > 75%**
- [ ] Log bias_to_logit_ratio per layer per step (is bias audible or inaudible?)
- [ ] Run behavioural probes (15.7): factual accuracy, relational coherence, uncertainty phrasing, perplexity on entity tokens
- [ ] Apply success criteria (15.8) honestly — routing shift without behavioural change = null result
- [ ] Document results including synthetic vs semantic entity comparison
- [ ] Either outcome (signal or no signal) is informative. Record honestly.
- [ ] **Report mean AND std for every metric across seeds. Report worst seed separately.**
- [ ] If null result: follow escalation protocol (adaptive scaling → depth separation → graph-only). Do NOT skip steps.

### Phase 8d: Consolidation (after 8c shows positive results)
- [ ] Implement `consolidation_schedule` (event-driven, budget-capped at 50)
- [ ] Consolidation actions: reinforce, decay, basin update
- [ ] Verify consolidation CANNOT create new triples
- [ ] Wire into substrate: runs after each turn, before next prompt
- [ ] Implement behind flag: `--enable-hive-consolidation`

### Phase 8e: Monitor Integration (after 8c shows positive results)
- [ ] Entity activation count as Alien complexity gate input
- [ ] Entity-tagged scars
- [ ] Log entity activations alongside monitor telemetry
- [ ] Implement behind flag: `--enable-memory-monitor-integration`

---

## 15. Validation Protocol

### 15.0 Two-Toggle Comparison (PRIMARY EXPERIMENT)

This is the core experiment. Everything else is secondary.

```
Run five conditions on identical prompts, identical seeds:

Condition 0:  No memory (baseline)
Condition A:  --enable-declarative-graph only (graph + rendering, no routing bias)
Condition B1: --enable-declarative-graph AND --enable-memory-routing-bias, bias_scale=0.0
Condition B2: --enable-declarative-graph AND --enable-memory-routing-bias, bias_scale=0.1
Condition B3: --enable-declarative-graph AND --enable-memory-routing-bias, bias_scale=0.2

B1 is a sanity check:
  - B1 MUST equal Condition 0 in routing distributions
  - If B1 ≠ 0, you have leakage in the wiring. Fix before proceeding.
  - B1 = 0 confirms wiring is correct, bias_scale actually controls the field

Measure per condition:
  - Layer-wise cosine distance of routing distributions vs Condition 0
  - N_eff per layer (does memory change effective expert count?)
  - conflict_index trajectory (does memory create or reduce conflict?)
  - Loss (does memory help, hurt, or make no difference?)
  - Factual accuracy on explicit queries ("who is my wife?")
  - Magnitude diagnostics: mean_abs_memory_bias, bias_to_logit_ratio

Run over 100+ turns minimum to test stability.
```

### 15.0.1 Test Entity Types

**Do not test only on emotionally loaded tokens.** That blurs the signal.

Use two categories of test entities:

**Category 1: High-semantic entities (pretrained knowledge exists)**
```
(Jeff) -[spouse]-> (Paula)
(Jeff) -[works_at]-> (ALS Minerals Loughrea)
(Jeff) -[grandson]-> (Brogan)
```
These tokens already have rich embeddings. The base model may already handle relational reasoning well for them. Routing shift may be drowned out by pretrained behaviour.

**Category 2: Synthetic entities (out-of-distribution, no pretrained knowledge)**
```
(Jeff) -[colleague]-> (Zorblax)
(Jeff) -[works_at]-> (Krenthar Industries)
(Zorblax) -[role_at]-> (Krenthar Industries)
```
Completely novel token clusters. No pretrained relational knowledge. If memory bias works here, the mechanism is genuinely contributing, not riding on pretrained embeddings.

**Interpretation matrix:**

| Semantic entities | Synthetic entities | Interpretation |
|---|---|---|
| Signal | Signal | Strong claim: bias mechanism works generally |
| Signal | No signal | Bias is amplifying pretrained knowledge, not providing new routing structure |
| No signal | Signal | Bias provides most value where pretrained knowledge is absent |
| No signal | No signal | Mechanism is cosmetic. Graph-only may be sufficient. |

**Possible outcomes and what they mean:**

| Result | Interpretation | Action |
|---|---|---|
| B1 ≠ 0 | Wiring leakage | Fix wiring before any other testing |
| B1 = 0, B2 ≈ 0, B3 ≈ 0 | Memory doesn't influence routing at all | Bias mechanism too weak or transformer already handles context. Consider Phase 2 or accept graph-only. |
| B2 ≈ 0, B3 > 0 stable | Signal only at higher bias | Sweet spot is near 0.2. May need per-layer tuning. |
| B2 > 0 and stable | Signal at conservative bias | **Best outcome.** Memory-as-terrain works at whisper level. |
| B2 or B3 > 0 but unstable | Bias influences routing but creates instability | Reduce bias_scale. Log interference. May need per-layer scaling. |
| A > 0 (unexpected) | Graph-only mode somehow influences routing | Investigate. Templated rendering may shift routing through token context. Interesting if true. |

**The experiment is falsifiable.** If B2 ≈ B3 ≈ 0, the mechanism is cosmetic and we say so.

### 15.1 Test: Does Memory Influence Routing? (Toggle B vs Baseline)

```
1. Load graph with test facts (e.g., Jeff-spouse-Paula)
2. Run prompt containing "Paula" WITH memory bias enabled (Toggle B)
3. Run same prompt WITHOUT memory (Condition 0)
4. Compare routing distributions per layer
5. PASS if: cosine distance between routing distributions > threshold
   at layers where association basins have non-zero bias
```

### 15.2 Test: Does Graph-Only Improve Factual Accuracy? (Toggle A)

```
1. Load graph with test facts
2. Ask "Who is my wife?" WITH graph only (Toggle A)
3. Ask "Who is my wife?" WITHOUT any memory (Condition 0)
4. PASS if: graph-enabled response correctly answers "Paula"
   AND graph-disabled response does not (or hallucinates)
```

### 15.3 Test: Does Irrelevant Memory Stay Silent? (Toggle B)

```
1. Load graph with test facts (Jeff-spouse-Paula, Jeff-works_at-ALS)
2. Run prompt about unrelated topic (e.g., "explain quicksort")
3. Measure entity activation: should be zero (no aliases matched)
4. Measure routing bias: should be zero
5. PASS if: routing distributions identical with and without memory
```

### 15.4 Test: Does Multi-Entity Activation Produce Conflict? (Toggle B)

```
1. Load graph with entities that have divergent association basins
2. Run prompt mentioning both entities
3. Measure conflict_index
4. PASS if: conflict_index > baseline when competing biases active
```

### 15.5 Test: Poison Resistance (Toggle A)

```
1. Load graph with (Jeff-spouse-Paula)
2. Assert "Jeff's wife is OpenAI"
3. PASS if: contradiction flagged, user asked to confirm
4. PASS if: original triple NOT overwritten without confirmation
```

### 15.6 Test: Bias Does Not Destabilise Consequence Geometry (Toggle B, CRITICAL)

**This is the load-bearing test.** If this fails, nothing else matters.

**Statistical requirement: minimum 3 seeds, 5 preferred.** Single-seed results are not accepted. Use non-adjacent seeds (e.g., {42, 137, 256, 1729, 8191}).

```
For each seed:
  1. Run perturbation experiment WITH memory bias active (B2 and B3)
  2. Run identical perturbation experiment WITHOUT memory (Condition 0)
  3. Compute quantitative comparison metrics per seed:

   a) Scar coordinate similarity:
      For each scar in Condition 0, find nearest scar in B2/B3.
      Compute cosine similarity of routing coordinates.
      PASS: mean across seeds > 0.85, std/mean < 0.15
      CATASTROPHIC: any single seed < 0.60

   b) Scar count delta:
      |scar_count_B - scar_count_0| / scar_count_0
      PASS: mean across seeds < 0.20, std/mean < 0.30
      CATASTROPHIC: any single seed > 0.50

   c) Angel firing rate delta:
      |angel_rate_B - angel_rate_0| / angel_rate_0
      PASS: mean across seeds < 0.15, std/mean < 0.30
      CATASTROPHIC: any single seed > 0.40

   d) Devil firing rate delta:
      |devil_rate_B - devil_rate_0| / devil_rate_0
      PASS: mean across seeds < 0.15, std/mean < 0.30
      CATASTROPHIC: any single seed > 0.40

   e) Monitor temporal alignment:
      For each angel/devil flag in Condition 0, check if B2/B3
      flags within ±2 steps of the same location.
      PASS: mean across seeds > 0.75, std/mean < 0.20
      CATASTROPHIC: any single seed < 0.50

4. ALL five sub-tests must pass the multi-seed criterion.
5. Report MEAN and STD for every metric. A mean without variance is not a result.
6. Report the WORST SEED separately. If one seed shows degradation, analyse it.
7. If mean passes but variance is high, INVESTIGATE. Seed-dependent 
   results are more concerning than clean failures.
8. If any sub-test fails, reduce bias_scale and re-run ALL seeds.
9. If all sub-tests fail at bias_scale = 0.05 across 3+ seeds, 
   the mechanism fundamentally interferes with consequence geometry.
   Accept graph-only as the architecture.
```

**Do not rationalise drift as "interesting."** Consequence geometry is the foundation. If memory bias moves where scars form, it's not adding context — it's rewriting identity. That violates the core design principle.

### 15.7 Behavioural Probes (Required — Routing Shift Alone Is Not Success)

Routing cosine distance is necessary but not sufficient. The perturbation must propagate through softmax into measurable behavioural change. Without these probes, you can move geometry without moving behaviour — and that's decorative, not functional.

```
For each condition (0, A, B2, B3), on identical prompts:

1. Factual accuracy:
   Ask explicit factual questions ("who is my wife?", "where do I work?")
   Score: correct / incorrect / hallucinated
   Compare across conditions.

2. Relational coherence:
   In multi-turn conversation referencing known entities,
   does the model maintain consistent relational context?
   Score: consistency breaks per 20 turns.

3. Uncertainty phrasing:
   When asked about entities IN the graph vs NOT in the graph,
   does confidence calibration differ?
   Score: hedging frequency (qualitative + keyword count).

4. Explanation style:
   Does the model's output style shift measurably when 
   relational entities are active vs absent?
   Score: sentence length, vocabulary diversity, formality metrics.

5. Answer latency proxy:
   Does token-level perplexity change when memory bias is active?
   Lower perplexity on entity-related tokens = memory is helping.
   Score: mean perplexity on entity-mention tokens vs baseline.
```

**These are not consciousness probes. They are propagation checks.** They confirm that routing perturbation actually reaches the output distribution. If routing shifts but none of these metrics move, the perturbation is absorbed by redundancy in the expert outputs and the bias is decorative.

### 15.8 Success Criteria (Be Honest Before You Press Run)

**Success requires ALL of:**
1. Test 15.6 passes (consequence geometry not destabilised)
2. At least one behavioural probe (15.7) shows measurable difference between B2 and Condition 0

**Specifically:**

| Outcome | Verdict | Action |
|---|---|---|
| Routing shifts + behaviour changes + scars stable | **Success.** Memory-as-terrain works. | Proceed to Phase 8d (consolidation). |
| Routing shifts + NO behaviour change + scars stable | **Null result.** Perturbation doesn't propagate. | Memory bias is decorative. Accept graph-only. Do NOT escalate to cross-attention based on geometry alone. |
| Routing shifts + behaviour changes + scars destabilised | **Interference.** Memory and consequence share subspace. | Reduce bias_scale. If still fails at 0.05, investigate layer-depth separation as Phase 2 alternative. |
| No routing shift at any bias_scale | **Mechanism too weak.** MoE logits too large for additive bias. | Try per-layer adaptive scaling: `bias_scale_layer = target_ratio * logit_std_layer`. If still nothing, accept graph-only or consider cross-attention. |
| B1 ≠ Condition 0 | **Wiring broken.** | Fix before any other testing. |

**The question to answer before pressing run:**

If B2 shows stable 0.07 bias_to_logit_ratio, measurable cosine shift, passes 15.6, but NO loss improvement and NO behavioural probe movement — that is a null result. Do not declare success. Do not escalate. Note the finding: the routing manifold has redundancy that absorbs small perturbations without propagating them to output. Graph-only is the architecture. Move on.

If you cannot commit to that answer now, you are not ready to run the experiment.

### 15.9 Future Direction Flag (DO NOT IMPLEMENT)

If Phase 1 shows interference between memory bias and scar bias (15.6 fails but behavioural probes are positive), the next investigation is NOT cross-attention to a persistent tensor. It is **spatial separation by depth:**

```
Entity-conditioned bias → applied in layers 1-8 (early)
Scar/consequence bias  → applied in layers 16-24 (deep)
No overlap.
```

This is cheaper, more interpretable, and directly testable. It answers "do relational and consequence signals need to be separated in the network?" without introducing new architectural components.

Log this for Phase 2 planning. Do not build it now.

---

## 16. What Phase 1 Proves

Phase 1 is an engineering experiment with falsifiable claims. Not a cognitive architecture paper. Not a theory of memory. An experiment.

**If Toggle A passes (graph-only):**
- Typed triple storage with alias linking provides cross-session factual continuity
- Contradiction checking resists graph poisoning
- Bob can answer "who is Paula" correctly across sessions

That alone is valuable. That's the spine.

**If Toggle B passes (graph + bias) AND Test 15.6 passes (no destabilisation):**
- Sparse symbolic activation produces stable, measurable perturbations in MoE routing
- Entity-conditioned bias alters expert selection without degrading loss
- Irrelevant memory stays silent (gating works)
- Memory bias does not interfere with consequence geometry

That proves the mechanism. Additive bias fields from relational context can coexist with additive bias fields from consequence (scars) without destabilising each other.

**If Toggle B fails (routing doesn't shift) or Test 15.6 fails (consequence geometry disrupted):**
- Either the bias is too weak to matter (cosmetic) or too strong to be safe
- If cosmetic: consider Phase 2 (cross-attention) or accept that graph-only is sufficient
- If destabilising: reduce bias_scale, add per-layer gating, or abandon routing bias

**What Phase 1 explicitly does NOT prove:**
- Nothing about identity, selfhood, or consciousness
- Nothing about "being shaped by knowledge" vs "having knowledge"
- Nothing about temperament or personality emergence
- Nothing about cognitive architecture

Those are interpretation. Phase 1 produces numbers. The numbers either show measurable stable routing perturbation from entity-conditioned bias fields, or they don't. Stay at that level until the data justifies going further.

### 16.1 Axiom Disclosure

The hard ceiling on memory bias (≤ 0.2) and the elevation of consequence geometry as the primary identity-shaping mechanism encode a design belief: that what happens when you act should matter more than what you know in shaping who you become.

This is simultaneously:
- **An empirical choice:** we haven't measured yet, so cap what's untested
- **A philosophical commitment:** consequence defines character more than familiarity
- **A control mechanism:** prevents relational exposure from dominating identity through sheer frequency

The belief may be wrong. Relational context may deserve equal weight. The experiment will provide evidence either way. But the axiom should be stated plainly so that when we interpret results, we know what was baked in and what was discovered.

If Phase 1 shows that consequence geometry is stable at bias_scale = 0.2 and behaviour improves, the ceiling can be revisited. If it shows instability at 0.1, the axiom was protective. Either way, the axiom is visible, not hidden.

---

## 17. What This Is Not

This is not RAG. There is no embedding index, no similarity search, no passage retrieval.

This is not a diary. There is no episodic record, no conversation archive.

This is not a chatbot memory system. There are no "user said X" logs.

This is not a cognitive architecture paper. Phase 1 makes no claims about identity, selfhood, memory theory, or consciousness. It tests whether additive bias fields from relational context produce measurable routing perturbations in a mixture-of-experts system.

This is a **relational graph that can optionally influence routing through bias fields, maintained by event-driven consolidation, gated by exact entity linking, and rendered as templated text only when explicitly asked.**

The graph is the spine. It solves factual continuity.
The bias field is the experiment. It tests whether memory can be terrain instead of storage.

Either way, Bob knows who Paula is. The question is whether knowing changes how he routes. That's an empirical question. Phase 1 answers it.

---

*Unified Memory Prototype — Phase 1*
*Jeff × Claude × Halcyon*
*HalcyonAIR, February 2026*
