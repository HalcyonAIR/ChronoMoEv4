# ChronoMoEv4 Bible

Technical wiring document. How Bob works on Qwen1.5-MoE-A2.7B via MLX.

**Version:** v4.0.0-alpha1
**Model:** `mlx-community/Qwen1.5-MoE-A2.7B-4bit` (60 experts, top-4, 24 MoE layers)
**Backend:** MLX on Apple Silicon
**License:** PolyForm Noncommercial 1.0.0

---

## 1. What Bob Is

Bob is a control plane that sits between an MoE model's router and its expert dispatch. It observes routing decisions (which experts were selected, with what weights), accumulates consequences (scars from bad outcomes, motifs from good ones), and learns when to take the cheap path (reuse a known-good routing pattern instead of paying full expert cost). Bob does not process tokens. Bob does not participate in routing. Bob decides how much thinking is necessary.

---

## 2. Architecture

```
mlx_lm.load("mlx-community/Qwen1.5-MoE-A2.7B-4bit")
    |
    v
MLXMoEAdapter (backends/mlx_adapter.py)
    |  - monkeypatches Qwen2MoeSparseMoeBlock.__call__
    |  - captures: gates [T, 60], inds [T, 4], scores [T, 4]
    |  - injects: memory_bias (pre-softmax), motif_bias (pre-softmax)
    |
    v
BobSubstrate (bob_core/substrate.py)
    |  - orchestrates: observe -> decide -> execute -> update -> log
    |  - reads: FastClock, MediumClock, SlowClock activations
    |  - evaluates: CompoundGate (3 sub-gates)
    |  - consults: BobGovernor (ALLOW / BLOCK / ESCALATE)
    |  - outputs: DecisionTrace per step
    |
    +--- MotifStore: routing pattern library per context class
    +--- BobCore: CommitmentLedger + ScarLedger + CostLedger
    +--- FastClock: immediate volatility (alpha=0.3)
    +--- MediumClock: regime stability (alpha=0.1)
    +--- SlowClock: constitutional envelope (alpha=0.02)
    +--- BobGovernor: commit authorization (pure reader)
    +--- PromotionGate: per-class stability tracking
    +--- TriadMonitor: Angel/Devil/Maniac per layer (optional)
    +--- ConflictRegister: angel-devil co-occurrence (optional)
    +--- RelationalGraph: typed entity triples (optional)
    +--- BasinStore: entity routing affinities (optional)
```

---

## 3. MLX Hook Point

**Location:** `backends/mlx_adapter.py`, function `_make_patched_call()` (line 63).

**Mechanism:** Class-level monkeypatch. MLX has no forward hooks. The adapter replaces `Qwen2MoeSparseMoeBlock.__call__` at the class level (not per-instance) so all 24 MoE layers share the patched method. The original is saved as `_original_call` for restoration via `adapter.unpatch()`.

**Layer identification:** Each MoE block gets `_capture_layer_id = i` set during discovery (line 153). The patched call reads this attribute to know which layer it is.

**Discovery path:** `model.model.layers[i].mlp` — iterates all transformer layers, checks for `gate` and `switch_mlp` attributes. Only MoE layers are captured.

**Captured tensors per layer per forward pass:**

| Tensor | Shape | Type | What it is |
|--------|-------|------|-----------|
| `gates` | `[T, 60]` | `mx.array` → `np.float32` | Post-softmax routing distribution over 60 experts |
| `inds` | `[T, 4]` | `mx.array` → `np.int64` | Selected expert indices (top-4) |
| `scores` | `[T, 4]` | `mx.array` → `np.float32` | Post-softmax weights for selected experts |

Where T = number of tokens in the input sequence.

**Gradient isolation:** Inference only. No training loop. `mx.stop_gradient()` wraps `mx.argpartition` (line 95). No gradients flow through the routing capture. Bob cannot backpropagate into the model.

**Conversion chain:** `mx.array` → `mx.eval()` (force compute) → `np.array()` → `torch.from_numpy()` → `LayerSnapshot`. The torch tensors in `LayerSnapshot` are for compatibility with Bob's substrate, which uses torch for Jaccard distance, entropy, and Neff calculations.

---

## 4. Pre-Softmax Bias Injection

**Location:** `backends/mlx_adapter.py`, inside `patched_call()` (lines 72-87).

**Order of operations:**

```
1. gates = self.gate(x)              # raw logits [T, 60] from nn.Linear
2. gates += memory_bias[lid]         # additive memory bias (if present)
3. gates += motif_bias               # additive motif bias (if present)
4. gates = softmax(gates)            # routing distribution
5. inds = argpartition(top-4)        # expert selection
6. scores = gather(gates, inds)      # routing weights
```

Memory bias is applied BEFORE motif bias. Both are applied BEFORE softmax. Both are additive offsets to raw gate logits.

**Memory bias format:** `Dict[int, list]` mapping `layer_id → [60 floats]`. Set on `_RoutingStore.memory_bias` before forward pass. The substrate computes this from `link_entities → diffuse_activation → compute_memory_bias` (see Section 9).

**Motif bias format:** `MotifSpec` containing per-layer `LayerMotif` with `expert_ids`, `weights`, and `bias_strength`. Constructs a bias vector: `bias[expert_id] = bias_strength * weight` for each preferred expert.

**Hard ceiling:** Memory bias values are clamped to `[-0.2, 0.2]` in `compute_memory_bias()` (bob_core/basins.py). The `BasinStore` constructor asserts `bias_scale <= 0.2` and `hard_ceiling <= 0.2`. Motif bias has no hard ceiling (it is consequence-derived).

---

## 5. Three Clocks

All clocks tick once per step, after execution. They read routing signals. They never write to each other's state.

### 5.1 Fast Clock (bob_core/fast_clock.py)

| Parameter | Default | What it means |
|-----------|---------|--------------|
| `ema_alpha` | 0.3 | Responds in ~3 steps, forgets in ~8 |
| `explore_threshold` | 0.4 | Activation above this triggers exploration pressure |
| `governor_threshold` | 0.5 | Activation above this triggers governor caution |
| `calibration_steps` | 0 (or 50 in experiments) | Steps to collect samples before freezing thresholds |
| `calibration_percentile` | 90.0 | Sets thresholds to p90 of warmup activations |

**Signals (4 EMAs, weighted):**

| Signal | Weight | Formula |
|--------|--------|---------|
| `churn_ema` | 0.30 | Jaccard distance between prev/curr expert sets |
| `entropy_delta_ema` | 0.25 | `|entropy_curr - entropy_prev|` |
| `loss_delta_ema` | 0.25 | `|loss_curr - loss_prev|` |
| `expert_flip_ema` | 0.20 | 1.0 if dominant expert changed, else 0.0 |

**Activation:** `sum(weight_i * signal_i)` — single scalar 0.0-1.0.

**Two alarms:**
- `exploration_pressure`: activation > explore_threshold
- `neff_collapse`: effective expert count below calibrated floor for K consecutive steps (K=3)

**Neff floor calibration:** During warmup, collects per-layer Neff values. Sets floor to p10 of each layer's distribution, then takes the minimum across layers.

### 5.2 Medium Clock (bob_core/medium_clock.py)

| Parameter | Default | What it means |
|-----------|---------|--------------|
| `ema_alpha` | 0.1 | Responds in ~10 steps, forgets in ~25 |
| `instability_threshold` | 0.5 | Above this, regime is unstable |
| `governor_threshold` | 0.5 | Used by governor for commit decisions |
| `calibration_steps` | 0 (or 50 in experiments) | Steps to collect before freezing |
| `calibration_percentile` | 90.0 | p90 of warmup activations |

**Signals (3 EMAs, weighted):**

| Signal | Weight | Formula |
|--------|--------|---------|
| `churn_ema` | 0.4 | Jaccard distance (same as fast clock but slower EMA) |
| `flipflop_ema` | 0.3 | 1.0 if path alternated (cheap→full or full→cheap), else 0.0 |
| `outcome_var_ema` | 0.3 | `|loss_curr - loss_prev|` for same-class steps |

**Activation:** `sum(weight_i * signal_i)` — single scalar 0.0-1.0.

**Role in bias modulation:** Motif bias strength = `base_bias * (1 - medium_activation) * promotion_score`. When medium activation is high (unstable), bias weakens. When stable, bias is strong.

### 5.3 Slow Clock (bob_core/slow_clock.py)

| Parameter | Default | What it means |
|-----------|---------|--------------|
| `ema_alpha` | 0.02 | Responds in ~50 steps, forgets in ~150 |
| `tighten_amount` | 0.1 | How much to tighten governor thresholds under stress |
| `loosen_amount` | 0.05 | How much to loosen when stable |
| `max_tighten_fraction` | 0.15 | Maximum threshold tightening (15%) |

**Signals (2 EMAs):**

| Signal | Formula |
|--------|---------|
| `scar_pressure_ema` | Slow EMA of aggregate scar debt |
| `stability_ema` | Slow EMA of routing stability (starts at 0.5) |

**Activation:** `max(0, scar_pressure - stability)`. High scar debt with low stability = constitutional stress.

**Effect:** Adjusts the governor's threshold envelope. High slow activation tightens thresholds (harder to commit). Low activation loosens them.

---

## 6. Governor

**Location:** `bob_core/governor.py`, class `BobGovernor`.

**Constructor:** `BobGovernor(bob_core, medium_clock, fast_clock=None, slow_clock=None, fast_threshold=0.5, medium_threshold=0.5, debt_threshold=0.7, ...)`

**Decision rule:**

```python
ALLOW iff:
    fast_activation  < fast_threshold   AND
    medium_activation < medium_threshold AND
    debt_level       < debt_threshold   AND
    not in_scar_neighborhood

BLOCK if any condition fails.

ESCALATE if:
    3+ conditions fail   OR
    debt_level > 0.9
```

**ESCALATE response (substrate, not governor):** Temporarily lowers fast clock's `explore_threshold` by 0.15 (minimum 0.15). Recovers over 10 steps. This makes exploration more likely for a few steps after a hard violation.

**Minimum commit rate guardrail:** If authorized commits in last 200 steps < 10, thresholds are temporarily relaxed (by `relaxation_amount=0.1`). Decays back over 50 steps. Prevents the governor from "winning" by never committing.

**Governor is a pure reader.** It reads clock activations and ledger state. It never writes to any clock or ledger. The substrate acts on the verdict.

---

## 7. Compound Gate

**Location:** `bob_core/motifs.py`, class `CompoundGate`.

Three sub-gates, ALL must pass independently:

| Sub-gate | Signal | Default threshold | Passes when |
|----------|--------|-------------------|-------------|
| Stability | `routing_stability` | 0.6 (Qwen: 0.15) | `stability >= stability_min` |
| Debt | `debt_level` | 0.5 | `debt <= debt_max` |
| Survival | `motif_survival` | 0.7 | `survival >= survival_min` |

**Qwen override:** Qwen 60E top-4 has lower routing stability than OLMoE 64E top-8 (max ~0.20 vs ~0.60+). Experiments use `stability_min=0.15`.

**Fast exploration relaxation:** When `fast_clock.exploration_pressure` is true, the substrate creates a relaxed gate: `stability_min * 0.5`, `debt_max * 2.0`, `survival_min * 0.5`. The gate is still checked. The governor is still consulted. Exploration relaxes thresholds, it does not bypass governance.

---

## 8. Scars

**Location:** `bob_core/ledgers.py`, class `Scar` and `ScarLedger`.

**Data structure:**

```python
@dataclass
class Scar:
    scar_id: int
    routing_region: Tuple[int, ...]  # sorted expert IDs
    severity: float                   # 0.0 to 1.0
    created_step: int
    last_triggered_step: int
    trigger_count: int = 1
```

**Decay:** `decayed_severity = severity * 0.5^(age / half_life)`, where `age = current_step - last_triggered_step`, `half_life = 500`.

**Total debt:** Sum of all scars' decayed severities. Used as the debt signal for the compound gate and governor.

**Scar neighborhood check:** The governor checks whether the candidate routing region overlaps with any scarred region. Overlap is computed as Jaccard similarity of expert ID sets.

**Governance coordinates:** Each commitment records the clock activations at creation time (`GovernanceCoords`: fast, medium, slow, debt, time_since_commit, commit_count). This is provenance for debugging.

---

## 9. Memory System

### 9.1 Relational Graph (bob_core/graph.py)

Typed triple store. Entities (people, places, organisations) connected by 34 allowed relations. Each node has a stable internal ID, alias list, node type, and metadata (confidence, usage count, provenance).

**34 allowed relations:** spouse, child, grandchild, stepchild, parent, sibling, colleague, friend, role_at, reports_to, manages, lives_in, works_at, studies_at, located_in, runs, member_of, employed_by, founded, pet, hobby, religion, nationality, language, qualification, alma_mater, born_in, born_on, died_on, event_attended, organisation_member, board_member, advisor, mentors.

**Alias table:** Maps normalised surface strings to node IDs. Exact match with case-insensitive normalisation. No embedding search. No fuzzy matching.

**Bounds:** MAX_NODES=200, MAX_EDGES=500, MIN_CONFIDENCE=0.3. `enforce_bounds()` removes lowest-confidence entries.

**The graph is inert.** It does not touch routing. Phase 8c confirmed: Toggle A (graph only) produces traces identical to Condition 0 (no graph).

### 9.2 Association Basins (bob_core/basins.py)

Per-entity routing affinity vectors. Each basin is `[num_layers][num_experts]` floats recording which experts historically handled this entity's context.

**Data structure:**

```python
@dataclass
class AssociationBasin:
    node_id: str
    bias_vector: List[List[float]]   # [num_layers][num_experts]
    strength: float                   # 0.0-1.0, overall confidence
    update_count: int
    last_updated: str
```

**BasinStore constraints:**
- `bias_scale <= 0.2` (asserted in constructor)
- `hard_ceiling <= 0.2` (asserted in constructor)
- Output clamped to `[-hard_ceiling, hard_ceiling]`

### 9.3 Memory Bias Pipeline

```
entity_tokens (from input)
    |
    v
link_entities(tokens, graph)
    |  - alias table lookup per token
    |  - unambiguous match: activation = 1.0
    |  - ambiguous match (multiple nodes): activation = 0.8
    |  - no match: skip
    |
    v
activations: Dict[node_id, float]
    |
    v
diffuse_activation(activations, graph, depth=1, decay=0.3)
    |  - 1-hop neighbours get source_activation * 0.3
    |  - MAX not SUM (no runaway from converging paths)
    |  - direct mentions are never reduced
    |
    v
diffused: Dict[node_id, float]
    |
    v
compute_memory_bias(diffused, basin_store, num_layers, num_experts)
    |  - for each active node with a basin:
    |      bias += activation * basin.strength * basin.bias_vector
    |  - scale by bias_scale
    |  - clamp to [-hard_ceiling, hard_ceiling]
    |  - if scale=0.0: return None (short circuit)
    |  - if no active basins: return None
    |
    v
bias_field: Dict[layer_id, List[float]] or None
    |
    v
adapter._store.memory_bias = bias_field  (injected pre-softmax)
```

### 9.4 B1 Guarantee

`bias_scale=0.0` → `compute_memory_bias()` returns `None` → `memory_bias=None` → no injection → traces identical to no-memory baseline.

Verified by:
- `tests/test_b1_equivalence.py`: 20-step deterministic assertion of loss, expert_ids, path, governor_decision identity
- `experiments/phase8c_logs/memory_exp_s42_report.json`: Gate 1 PASS (0 mismatches across 60 steps)

---

## 10. Triad Monitors

**Location:** `bob_core/monitors.py` (TriadMonitor), `bob_core/conflict.py` (ConflictRegister).

**Status:** Implemented and validated. Interventions gated behind `interventions_enabled=False` (monitoring only).

### 10.1 Observable Signals (per layer, per step)

| Signal | Formula | What it measures |
|--------|---------|-----------------|
| Q_A | `N_eff[t] - N_eff[t-1]` | Optionality velocity |
| G | `(pi1 - pi2) / pi1` | Winner dominance (gap ratio) |
| C | `EMA(pi1)[t] - EMA(pi1)[t-1]` | Commitment acceleration (EMA-smoothed) |
| D_KL | `KL(pi[t] \|\| pi_bar)` | Divergence from routing habit |

Where `N_eff = exp(H(pi))`, `H(pi) = -sum(pi_k * log(pi_k))`, `pi_bar` is a slow EMA of the routing distribution.

### 10.2 Calibration

During warmup (configurable, default 50-100 steps), collect per-layer empirical distributions. Freeze percentile thresholds:

| Threshold | Percentile | Signal | Used by |
|-----------|-----------|--------|---------|
| alpha | p10 of Q_A | Rare collapse | Angel |
| beta | p90 of G | High dominance | Angel |
| gamma | p90 of C | High acceleration | Devil |
| delta | p10 of D_KL | Low divergence | Maniac |

### 10.3 Monitor Triggers

**Angel** (optionality collapse):
```
angel_flag = (Q_A < alpha) AND (G > beta)
score = max(0, -Q_A * G)
```

**Devil** (Venus flytrap — confidence + collapse):
```
devil_flag = (C > gamma) for 2+ consecutive AND (Q_A < 0)
score = max(0, C * (-Q_A))
```

**Maniac** (stagnation):
```
maniac_flag = (D_KL < delta) for n consecutive
score = (delta - D_KL) * n_consecutive
```

**Intervention priority:** Devil > Angel > Maniac. Maximum one intervention per step.

### 10.4 Conflict Register

```python
conflict_index[t] = angel_peak[t] * devil_peak[t]
```

Product formula: high only when BOTH angel and devil are active simultaneously. Stored in a 50-step circular buffer.

**Mode A (normal):** conflict low/stable. All monitors at baseline.
**Mode B (calm-bold):** conflict high and trending up. Devil requires 3 consecutive steps (was 2). Damping increases. Fewer interventions, but better-supported commits.

---

## 11. Running It

### 11.1 Install

```bash
pip install -r requirements.txt
# requirements.txt: mlx>=0.21.0, mlx-lm>=0.21.0, torch>=2.0, numpy>=1.24, transformers>=4.37
```

### 11.2 Test (no model needed)

```bash
python3 -m pytest tests/ -v
# 70 tests, ~0.5s
```

### 11.3 Governed Experiment

```python
from mlx_lm import load
from backends.mlx_adapter import MLXMoEAdapter
from bob_core.substrate import BobSubstrate
from bob_core.motifs import GateThresholds
from bob_core.ledgers import BobCore
from bob_core.medium_clock import MediumClock
from bob_core.fast_clock import FastClock
from bob_core.slow_clock import SlowClock
from bob_core.governor import BobGovernor
from bob_core.promotion import PromotionGate

# Load model
model, tokenizer = load("mlx-community/Qwen1.5-MoE-A2.7B-4bit")
adapter = MLXMoEAdapter(model, tokenizer)
# adapter.num_experts=60, adapter.top_k=4, adapter.num_layers=24

# Build components
bob_core = BobCore(debt_cap=1.0)
fast_clock = FastClock(ema_alpha=0.3, calibration_steps=50, calibration_percentile=90.0)
medium_clock = MediumClock(ema_alpha=0.1, calibration_steps=50, calibration_percentile=90.0)
slow_clock = SlowClock(ema_alpha=0.02)
promotion_gate = PromotionGate(stability_window=20)
governor = BobGovernor(
    bob_core, medium_clock,
    fast_clock=fast_clock,
    slow_clock=slow_clock,
    fast_threshold=0.5,
    medium_threshold=0.5,
    debt_threshold=0.7,
)

# Build substrate
bob = BobSubstrate(
    adapter,
    gate_thresholds=GateThresholds(stability_min=0.15, debt_max=0.5, survival_min=0.7),
    warmup_steps=50,
    bob_core=bob_core,
    governor=governor,
    fast_clock=fast_clock,
    medium_clock=medium_clock,
    slow_clock=slow_clock,
    promotion_gate=promotion_gate,
)

# Run
input_ids = tokenizer.encode("Explain quicksort")
labels = list(input_ids)
trace = bob.step(input_ids, labels, context_class=0, step=0)
# trace.path: "full" or "cheap"
# trace.loss: float
# trace.expert_ids: tuple of selected expert IDs
# trace.gate_passed: bool
# trace.governor_decision: "allow" | "block" | "escalate" | None
```

### 11.4 Memory Experiment

```python
from bob_core.graph import RelationalGraph
from bob_core.basins import BasinStore

# Build graph
graph = RelationalGraph()
graph.resolve_or_create_node("Jeff", node_type="person")
graph.resolve_or_create_node("Paula", node_type="person")
graph.add_triple("Jeff", "spouse", "Paula")

# Build basins
store = BasinStore(bias_scale=0.1)  # hard ceiling 0.2
node_id = graph.alias_table.lookup("Jeff")[0]
basin = store.get_or_create_basin(node_id, num_layers=24, num_experts=60)
basin.strength = 0.5
basin.bias_vector[0][2] = 0.6  # layer 0, expert 2 affinity

# Wire into substrate
bob = BobSubstrate(
    adapter,
    # ... same as above ...
    memory_graph=graph,
    basin_store=store,
    memory_bias_scale=0.1,
)

# Step with entity tokens
trace = bob.step(input_ids, labels, context_class=0, step=50,
                 entity_tokens=["jeff", "paula"])
# trace.memory_bias_applied: bool
# trace.memory_bias_max: float (max absolute bias value)
# trace.memory_nodes_active: int
```

---

## 12. Determinism Invariants

1. **Fixed seed produces identical traces.** Three independent runs with the same inputs produce byte-identical loss, path, expert_ids, and governor_decision. Verified by `test_determinism_across_runs()`.

2. **B1 = Condition 0.** Scale=0.0 with graph+basins present produces traces identical (within 1e-6) to no graph, no basins. Not close. Identical. Verified by `test_b1_equals_c0()` and Gate 1 of Phase 8c.

3. **Toggle A = Condition 0.** Graph present without basins produces traces identical to no graph. Graph is provably inert. Verified by Gate 2 of Phase 8c.

4. **Hard ceiling never exceeded.** `memory_bias_max <= 0.2` for all steps, all conditions. Asserted in `BasinStore` constructor and clamped in `compute_memory_bias` output. Verified by Gate 3 of Phase 8c.

5. **Governor is deterministic given same inputs.** No stochastic components in the decision path. The governor reads three scalar activations, one debt level, and one scar overlap score. Same inputs → same verdict.

6. **No clock writes to another clock.** Fast, Medium, and Slow clocks read routing signals. They never write to each other's state. The governor reads all three. The substrate acts on the governor's verdict. One-directional data flow: `signals → clocks → governor → substrate → execution`.

---

## 13. Phase 8c Results

Five conditions, seed 42, Qwen1.5-MoE-A2.7B-4bit, warmup=10, active=50.

| Condition | Avg Loss | Cheap Fraction |
|-----------|----------|---------------|
| C0 Baseline | 2.4716 | 4% |
| C1 Toggle A (graph only) | 2.4716 | 4% |
| C2 B1 (scale=0.0) | 2.4716 | 4% |
| C3 B2 (scale=0.1) | 2.4711 | 4% |
| C4 B3 (scale=0.2) | 2.4719 | 4% |

| Gate | Result |
|------|--------|
| Gate 1 (B1=C0) | PASS: 0 mismatches |
| Gate 2 (Toggle A=C0) | PASS: 0 mismatches |
| Gate 3 (audibility) | B2: median_max=0.08, B3: median_max=0.16, 100% coverage |
| Gate 4 (geometry B2) | PASS: Neff ratio=1.0001, entropy delta=-0.0% |
| Gate 4 (geometry B3) | PASS: Neff ratio=0.9998, entropy delta=-0.0% |
| Gate 5 (primary B2) | NULL: no routing shift (Jaccard=0.0) |
| Gate 5 (primary B3) | NEGATIVE: synthetic loss +0.002 |

**Conclusion:** Memory bias is audible (it reaches the router) but does not improve synthetic entity loss at tested scales. Routing geometry is preserved (no destabilisation). Graph-only is the correct architecture at this point. This is a valid and publishable null result.

---

## 14. File Inventory

| File | Lines | What it does |
|------|-------|-------------|
| `bob_core/substrate.py` | ~710 | Main orchestrator: step(), gate, governor, memory bias |
| `bob_core/motifs.py` | ~280 | MotifStore, CompoundGate, GateThresholds, survival tracking |
| `bob_core/ledgers.py` | ~400 | Commitments, Scars, Costs, GovernanceCoords |
| `bob_core/governor.py` | ~250 | ALLOW/BLOCK/ESCALATE, minimum commit rate guardrail |
| `bob_core/fast_clock.py` | ~220 | Volatility + Neff collapse detection, calibration |
| `bob_core/medium_clock.py` | ~150 | Regime stability, churn/flipflop/outcome variance |
| `bob_core/slow_clock.py` | ~100 | Constitutional envelope, scar pressure + stability |
| `bob_core/promotion.py` | ~80 | Per-class stability tracking |
| `bob_core/monitors.py` | ~290 | Angel/Devil/Maniac signals, calibration, flag logic |
| `bob_core/conflict.py` | ~80 | Conflict register, Mode A/B |
| `bob_core/graph.py` | ~500 | Relational graph, alias table, triple management |
| `bob_core/basins.py` | ~300 | Association basins, memory bias computation |
| `bob_core/telemetry.py` | ~150 | DecisionTrace dataclass |
| `bob_core/identity.py` | ~30 | Identity boundary detection |
| `backends/adapter.py` | ~120 | Protocol definition (ForwardResult, LayerSnapshot, etc.) |
| `backends/mlx_adapter.py` | ~330 | Qwen MLX monkeypatch adapter |
| `experiments/prompts.py` | ~1400 | 199 prompts across 5 categories (zero non-standard imports) |
| `experiments/qwen_governed.py` | ~830 | Phase 2 governed experiment |
| `experiments/memory_bias.py` | ~1040 | Phase 8c memory bias validation |
