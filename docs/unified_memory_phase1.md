# ChronoMoE: Unified Memory Prototype — Phase 1

**Implementation Spec for ChronoMoEv4**

Authors: Jeff, Claude, Halcyon
Date: February 2026
Status: Ready for implementation after triad monitor traces are validated
Prerequisite: Triad monitors (monitors.py, conflict.py) producing stable traces

---

> Bob doesn't remember conversations. He navigates a terrain that was sculpted by them, and he knows the names of the landmarks because someone told him and he wrote it down.

---

## 1. What This Is

A memory system where memory is not retrieved. It is already present as a routing bias field. No RAG. No similarity search. No token injection during normal operation. No context window consumption.

When the user says "Paula," Bob doesn't search a database. The routing landscape was already shaped by Paula's existence. The cue activates a region of that landscape. Routing flows through it.

**Phase 1 scope:** Two separable toggles, run independently and compared.

**Toggle A — Graph Only:** Relational graph in RAM, alias-based entity linking, templated rendering on explicit factual queries. Memory is stored and retrievable but inert. It does not influence routing.

**Toggle B — Graph + Routing Bias:** Everything in Toggle A, plus association basins, activation diffusion, and pre-softmax routing bias. Memory influences expert selection through additive bias fields. Same mechanism as scars and the governor — proven infrastructure.

No cross-attention. No persistent tensor. No new transformer components in either toggle.

The question Phase 1 answers: Can sparse symbolic activation produce stable, measurable perturbations in a MoE routing manifold without destabilising consequence geometry?

What Phase 1 does NOT answer: Questions about identity, selfhood, or cognitive architecture. Phase 1 proves influence. Not emergence. Not personality. Influence. Stay there.

---

## 2. Architecture Overview

### Toggle A: Graph Only

    Token stream
         |
         v
    Alias Table -----> Entity Linking (Maps tokens to node IDs)
                            |
                            v
                    Relational Graph (RAM)
                    Typed triples + metadata
                            |
                            v  (only on explicit factual query)
                    Templated Rendering
                    "Fact: Paula is Jeff's wife."

Normal routing is UNAFFECTED. Memory is inert storage. Facts render as tokens only when user explicitly asks.

### Toggle B: Graph + Routing Bias

    Token stream
         |
         v
    Alias Table -----> Activation Gate (Maps tokens to node IDs)
                            |
                            v
                    Relational Graph (RAM)
                    Typed triples + metadata
                    Nodes have routing sigs
                            |
                            v
                    Activation Diffusion
                    Spread over 1-hop
                    Decay with graph distance
                            |
                            v
                    Routing Bias Field
                    Pre-softmax logit adjust
                    Per activated node
                    bias_scale <= 0.2 (HARD)
                            |
                            v
                    Normal routing + monitors + clocks + governor

The transformer sees routing logits shaped by memory. It does not know memory exists. Everything downstream operates on the composite routing distribution unchanged.

The experiment: Run identical prompts through both toggles. Measure whether Toggle B produces stable routing perturbations that Toggle A does not. See Section 15 for protocol.

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
    alias_table: Dict[str, List[str]]  # surface form -> [node_ids]
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
    "preferences",    # "likes X" -> lives in geometric layer as safe basins
    "episodes",       # "said X on Tuesday" -> ephemeral, consequence captured by scars
    "opinions",       # "thinks X about Y" -> within-run context only
    "tasks",          # "working on X" -> transient
    "transient_state" # "currently in Australia" -> session context only
}
```

Why: Preferences and episodes grow without bound, need temporal queries, and require relevance ranking. That's RAG territory. The graph stays small by storing only slow-changing world-facts. A rich personal graph is 50-100 triples.

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
        self._table: Dict[str, List[str]] = {}  # normalised_alias -> [node_ids]

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
    Returns activation map: node_id -> activation_strength.

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

```python
def diffuse_activation(activations: Dict[str, float], graph: RelationalGraph,
                       depth: int = 1, decay: float = 0.3) -> Dict[str, float]:
    """
    Spread activation from directly-mentioned entities to their neighbours.

    depth=1: only immediate neighbours (sufficient for Phase 1)
    decay=0.3: neighbours get 30% of the source activation

    This is cue-triggered pattern completion. Not global recall.
    """
    diffused = dict(activations)

    for hop in range(depth):
        new_activations = {}
        hop_decay = decay ** (hop + 1)

        for node_id, strength in list(diffused.items()):
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

**Properties:**
- Sparse: Only nodes near entity mentions activate.
- Bounded: Max activation is 1.0 (direct mention). Neighbours get at most 0.3.
- No runaway: Using max instead of sum prevents stacking.
- Cheap: 1-hop over 50-100 nodes is microseconds.

---

## 6. Routing Bias Field

Activated nodes produce pre-softmax logit adjustments. Same mechanism as scars — proven infrastructure.

### 6.1 Association Basins

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
    strength: float            # overall confidence in this basin
    update_count: int          # how many consolidation cycles contributed
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

    HARD CEILING: bias_scale must never exceed 0.2
    Memory whispers. Consequence shouts.
    If signal isn't visible at 0.2, the mechanism is too weak
    and Phase 2 (cross-attention) is needed.
    """
    memory_bias = np.zeros_like(router_logits)

    for node_id, activation_strength in activations.items():
        basin = association_basins.get(node_id)
        if basin is not None and basin.bias_vector is not None:
            memory_bias += (
                activation_strength
                * basin.strength
                * basin.bias_vector[layer_idx]
            )

    assert bias_scale <= 0.2, "Memory bias must not exceed hard ceiling of 0.2"
    router_logits = router_logits + bias_scale * memory_bias

    # MAGNITUDE DIAGNOSTICS (always log)
    logit_std = np.std(router_logits - bias_scale * memory_bias)
    bias_magnitude = np.abs(bias_scale * memory_bias)
    diagnostics = {
        "mean_abs_memory_bias": float(np.mean(bias_magnitude)),
        "max_abs_memory_bias":  float(np.max(bias_magnitude)),
        "bias_to_logit_ratio":  float(np.mean(bias_magnitude) / (logit_std + 1e-8)),
        "n_active_nodes":       sum(1 for s in activations.values() if s > 0),
    }
    # Sweet spot: bias_to_logit_ratio 0.05-0.2

    return router_logits, diagnostics
```

**Properties:**
- Pre-softmax: Bias added before softmax. Routing remains valid probability distribution.
- Proportional to activation: Diffused neighbours have 30% influence. Unmentioned entities have zero.
- Scaled conservatively: bias_scale=0.1 is a gentle nudge.
- Composable with scars: Memory bias and scar bias are both pre-softmax. They add linearly.

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

    if relation not in ALLOWED_RELATIONS:
        return Result.reject(f"Relation '{relation}' not in allowed set")

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
        edge = graph.get_edge(subject_id, relation, object_id)
        edge.usage_count += 1
        edge.confidence = min(1.0, edge.confidence + 0.05)
        return Result.reinforced(edge)

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
    graph.alias_table.add(subject, subject_id)
    graph.alias_table.add(object_, object_id)

    return Result.created(edge)
```

### 7.2 Detecting Assertions in User Input

```python
def detect_assertions(user_text: str) -> List[Triple]:
    """
    Simple pattern matching for declarative statements.
    NOT full NLP. Just common relational patterns.

    "Paula is my wife" -> (user, spouse, Paula)
    "I work at ALS" -> (user, works_at, ALS)
    "Brogan is my grandson" -> (user, grandchild, Brogan)

    When in doubt, don't create a triple — ask the user.
    False negatives are recoverable. False positives corrupt the graph.
    """
    patterns = [
        (r"(\w+)\s+is\s+my\s+(wife|husband|spouse|partner)",
         lambda m: Triple(subject="user", relation="spouse", object_=m.group(1))),

        (r"(\w+)\s+is\s+my\s+(son|daughter|child|grandson|granddaughter|grandchild)",
         lambda m: Triple(subject="user", relation=map_family_relation(m.group(2)),
                           object_=m.group(1))),

        (r"I\s+work\s+at\s+(.+?)(?:\.|$)",
         lambda m: Triple(subject="user", relation="works_at", object_=m.group(1).strip())),

        (r"I\s+run\s+(.+?)(?:\.|$)",
         lambda m: Triple(subject="user", relation="runs", object_=m.group(1).strip())),

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

---

## 8. Templated Rendering (Explicit Queries Only)

```python
def render_fact_packet(node_id: str, graph: RelationalGraph,
                        max_facts: int = 5) -> Optional[str]:
    """
    Deterministic rendering of structured triples.
    NOT retrieved text. NOT RAG. Fixed template, structured data.

    Only called when the user explicitly asks a factual question.
    During normal conversation, memory influences routing silently.
    """
    edges = graph.get_edges_for_node(node_id)
    if not edges:
        return None

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
    "works_at": "works at",
    "runs": "runs",
    "studies_at": "studies at",
    "lives_in": "lives in",
    "colleague": "is a colleague of",
    # ... etc
}
```

---

## 9. Consolidation Loop

Event-driven. Runs between prompts. Compiles routing priors. Does NOT create new facts.

### 9.1 Schedule

```python
def consolidation_schedule(turn_events: TurnEvents) -> int:
    """Budget scales with consequence. Hard cap prevents rumination."""
    budget = 5  # baseline: alias refresh, triple decay

    if turn_events.scars_formed > 0:
        budget += 10 * turn_events.scars_formed

    if turn_events.expansions_formed > 0:
        budget += 10 * turn_events.expansions_formed

    if turn_events.mode_b_engaged:
        budget += 15

    return min(budget, 50)  # HARD CAP
```

### 9.2 What Consolidation CANNOT Do

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

    "Given what you know about Paula and ALS Loughrea..."

Multiple entity activations blend linearly. Each weighted by its activation strength. If two entity contexts pull routing in different directions, the composite bias creates tension. The monitors detect this. Angel sees optionality shifting. Devil may detect confidence acceleration. conflict_index rises.

No special handling needed. The existing triad monitors observe the composite routing distribution. Memory-induced conflict looks identical to any other routing conflict from the monitors' perspective.

---

## 11. Monitor Integration

### 11.1 Alien Complexity Gate

Active entity associations contribute to the Alien's complexity gate. Two or more entities active simultaneously = complex relational context flag.

### 11.2 Entity-Tagged Scars

When a scar forms and entity associations were active, the scar is tagged with those entity IDs. Next time these entities activate, the scar field includes entity-specific danger basins.

---

## 12. Persistence

### 12.1 Bounded Size

```python
MAX_NODES = 200
MAX_EDGES = 500
MIN_CONFIDENCE = 0.3
```

Graph: 50-100 nodes, 50-100 edges, few KB. Basins: one array per node per layer. Entire memory state fits in a small JSON file. Memory is immediately available at session start — no warmup needed.

---

## 13. Poison Resistance

1. **Only user-asserted triples.** The model cannot create facts from inference.
2. **Contradiction checking.** New triples conflicting with existing ones are flagged, not silently accepted.
3. **Provenance on everything.** Every node and edge traces back to session_id, turn_id, source_type.
4. **Typed relations only.** No freeform relation strings.
5. **Confidence gating.** New triples start at 0.8, not 1.0. Must be reinforced through usage.

When a new assertion contradicts an existing fact: do NOT silently overwrite. Present both to the user. Let the user resolve.

---

## 14. Implementation Checklist

### Phase 8a: Graph Infrastructure (Toggle A)
- [ ] RelationalGraph, GraphNode, GraphEdge data structures
- [ ] AliasTable with normalisation
- [ ] add_triple with contradiction checking
- [ ] detect_assertions (conservative pattern matching)
- [ ] Serialisation/deserialisation to JSON
- [ ] enforce_bounds for size management
- [ ] render_fact_packet with RELATION_TEMPLATES
- [ ] Behind flag: --enable-declarative-graph
- [ ] Toggle A independently testable. Ship and validate before Phase 8b.

### Phase 8b: Activation and Routing Bias (Toggle B, requires 8a)
- [ ] link_entities (scan tokens against alias table)
- [ ] diffuse_activation (1-hop, decay=0.3)
- [ ] AssociationBasin data structure
- [ ] apply_memory_bias (pre-softmax logit adjustment)
- [ ] Wire into existing router: bias applied after normal logit computation, before softmax
- [ ] bias_scale configurable (start 0.1, HARD CEILING 0.2)
- [ ] Behind flag: --enable-memory-routing-bias

### Phase 8c: Two-Toggle Comparison (CRITICAL EXPERIMENT)
- [ ] Prepare test prompts with high-semantic entities (Paula, ALS) AND synthetic entities (Zorblax, Krenthar)
- [ ] Run Condition 0 (no memory)
- [ ] Run Toggle A (graph only)
- [ ] Run B1 (bias_scale=0.0) — must equal Condition 0 or wiring is broken
- [ ] Run B2 (bias_scale=0.1)
- [ ] Run B3 (bias_scale=0.2)
- [ ] All conditions: same prompts, same seeds, 100+ turns
- [ ] Measure: layer-wise cosine distance, N_eff, conflict_index, loss, magnitude diagnostics
- [ ] Run Test 15.6 with quantitative thresholds
- [ ] Run behavioural probes (15.7)
- [ ] Apply success criteria (15.8) honestly
- [ ] Either outcome informative. Record honestly.

### Phase 8d: Consolidation (after 8c positive)
- [ ] consolidation_schedule (event-driven, budget-capped at 50)
- [ ] Consolidation actions: reinforce, decay, basin update
- [ ] Verify consolidation CANNOT create new triples
- [ ] Behind flag: --enable-hive-consolidation

### Phase 8e: Monitor Integration (after 8c positive)
- [ ] Entity activation count as Alien complexity gate input
- [ ] Entity-tagged scars
- [ ] Behind flag: --enable-memory-monitor-integration

---

## 15. Validation Protocol

### 15.0 Two-Toggle Comparison (PRIMARY EXPERIMENT)

Run five conditions on identical prompts, identical seeds:

| Condition | Config |
|-----------|--------|
| Condition 0 | No memory (baseline) |
| Condition A | --enable-declarative-graph only |
| Condition B1 | Graph + bias, bias_scale=0.0 (sanity check: must equal C0) |
| Condition B2 | Graph + bias, bias_scale=0.1 |
| Condition B3 | Graph + bias, bias_scale=0.2 |

**B1 is a sanity check.** B1 MUST equal Condition 0. If B1 != 0, you have wiring leakage. Fix before proceeding.

Measure per condition: layer-wise cosine distance vs C0, N_eff per layer, conflict_index trajectory, loss, factual accuracy, magnitude diagnostics.

Run over 100+ turns minimum.

### 15.0.1 Test Entity Types

**Category 1: High-semantic entities** (pretrained knowledge exists)

    (Jeff) -[spouse]-> (Paula)
    (Jeff) -[works_at]-> (ALS Minerals Loughrea)
    (Jeff) -[grandson]-> (Brogan)

**Category 2: Synthetic entities** (out-of-distribution, no pretrained knowledge)

    (Jeff) -[colleague]-> (Zorblax)
    (Jeff) -[works_at]-> (Krenthar Industries)
    (Zorblax) -[role_at]-> (Krenthar Industries)

If bias mechanism works on synthetic entities, the mechanism is genuinely contributing — not riding on pretrained embeddings.

**Interpretation matrix:**

| Semantic | Synthetic | Interpretation |
|----------|-----------|----------------|
| Signal | Signal | Strong claim: bias mechanism works generally |
| Signal | No signal | Bias amplifying pretrained knowledge only |
| No signal | Signal | Bias most valuable where pretrained knowledge is absent |
| No signal | No signal | Mechanism is cosmetic. Graph-only may be sufficient. |

### 15.6 Test: Bias Does Not Destabilise Consequence Geometry (CRITICAL)

This is the load-bearing test. If this fails, nothing else matters.

Run perturbation experiment WITH memory bias (B2 and B3) vs WITHOUT (Condition 0).

Quantitative thresholds:

| Sub-test | Metric | Pass Threshold |
|----------|--------|----------------|
| Scar coordinate similarity | Cosine similarity of routing coordinates | > 0.85 |
| Scar count delta | Absolute change / baseline count | < 20% |
| Angel firing rate delta | Absolute change / baseline rate | < 15% |
| Devil firing rate delta | Absolute change / baseline rate | < 15% |
| Monitor temporal alignment | Flags within ±2 steps of same location | > 75% overlap |

ALL five sub-tests must pass. If any fail, reduce bias_scale and re-run.

Do not rationalise drift as "interesting." Consequence geometry is the foundation. If memory bias moves where scars form, it's not adding context — it's rewriting identity.

### 15.7 Behavioural Probes (Required)

Routing cosine distance is necessary but not sufficient. For each condition, on identical prompts:

1. **Factual accuracy:** Ask explicit factual questions. Score: correct / incorrect / hallucinated.
2. **Relational coherence:** Consistency breaks per 20 turns in multi-entity conversation.
3. **Uncertainty phrasing:** Hedging frequency for entities IN vs NOT IN graph.
4. **Explanation style:** Sentence length, vocabulary diversity, formality metrics.
5. **Perplexity on entity tokens:** Lower perplexity = memory is helping.

These confirm routing perturbation actually reaches the output distribution. If routing shifts but none of these metrics move, the perturbation is absorbed by expert redundancy and the bias is decorative.

### 15.8 Success Criteria

Success requires ALL of:
- Test 15.6 passes (consequence geometry not destabilised)
- At least one behavioural probe (15.7) shows measurable difference between B2 and Condition 0

| Outcome | Verdict | Action |
|---------|---------|--------|
| Routing shifts + behaviour changes + scars stable | SUCCESS | Proceed to Phase 8d |
| Routing shifts + NO behaviour change + scars stable | Null result | Accept graph-only. Do NOT escalate. |
| Routing shifts + behaviour changes + scars destabilised | Interference | Reduce bias_scale. If still fails at 0.05, investigate layer-depth separation. |
| No routing shift at any bias_scale | Mechanism too weak | Try adaptive per-layer scaling. If still nothing, accept graph-only. |
| B1 != Condition 0 | Wiring broken | Fix before any other testing. |

---

## 16. What Phase 1 Proves

Phase 1 is an engineering experiment with falsifiable claims. Not a cognitive architecture paper. Not a theory of memory. An experiment.

**If Toggle A passes:** Typed triple storage with alias linking provides cross-session factual continuity. Contradiction checking resists graph poisoning. Bob can answer "who is Paula" correctly across sessions. That alone is valuable. That's the spine.

**If Toggle B passes AND Test 15.6 passes:** Sparse symbolic activation produces stable, measurable perturbations in MoE routing. Entity-conditioned bias alters expert selection without degrading loss. Irrelevant memory stays silent. Memory bias does not interfere with consequence geometry.

**What Phase 1 does NOT prove:** Nothing about identity, selfhood, or consciousness. Phase 1 produces numbers. Stay at that level until data justifies going further.

### 16.1 Axiom Disclosure

The hard ceiling on memory bias (<=0.2) and the elevation of consequence geometry as the primary identity-shaping mechanism encode a design belief: **that what happens when you act should matter more than what you know in shaping who you become.**

This is simultaneously:
- An empirical choice: we haven't measured yet, so cap what's untested
- A philosophical commitment: consequence defines character more than familiarity
- A control mechanism: prevents relational exposure from dominating identity through sheer frequency

The belief may be wrong. The axiom should be stated plainly so when we interpret results, we know what was baked in and what was discovered.

---

## 17. What This Is Not

- Not RAG. No embedding index, no similarity search, no passage retrieval.
- Not a diary. No episodic record, no conversation archive.
- Not a chatbot memory system. No "user said X" logs.
- Not a cognitive architecture paper.

This is a relational graph that can optionally influence routing through bias fields, maintained by event-driven consolidation, gated by exact entity linking, and rendered as templated text only when explicitly asked.

**The graph is the spine.** It solves factual continuity. **The bias field is the experiment.** It tests whether memory can be terrain instead of storage.

Either way, Bob knows who Paula is. The question is whether knowing changes how he routes. That's an empirical question. Phase 1 answers it.

---

*Unified Memory Prototype — Phase 1*
*Jeff x Claude x Halcyon*
*HalcyonAIR, February 2026*
