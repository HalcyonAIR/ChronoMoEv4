# ChronoMoEv4

**Bob**: consequence-accumulating control plane for Mixture-of-Experts models.

Bob observes MoE routing decisions, accumulates consequences, and learns when
to take the cheap path. The model does the thinking. Bob decides how much
thinking is necessary.

**Version**: v4.0.0-alpha1
**License**: PolyForm Noncommercial 1.0.0 (see `bob_core/LICENSE`)
**Backend**: Qwen1.5-MoE-A2.7B on Apple Silicon via MLX

---

## Structure

```
bob_core/          Control plane (15 modules)
backends/          Adapter layer (MLX only)
experiments/       Experiment scripts + shared prompt corpus
tests/             70 tests (triad monitors, relational graph, association basins, B1 equivalence)
designs/           Specifications (triad monitor spec)
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests (no model needed)
python3 -m pytest tests/ -v

# Verify imports
python3 -c "from bob_core import BobSubstrate, BOB_CORE_VERSION; print(BOB_CORE_VERSION)"

# Run governed experiment (requires Qwen model + MLX)
source qwen_moe_mlx/bin/activate
python3 experiments/qwen_governed.py --smoke --seed 42

# Run memory bias validation (Phase 8c)
python3 experiments/memory_bias.py --smoke --seed 42
```

## What Bob Does

| Component | What it does |
|-----------|-------------|
| Motif store + compound gate | Recognises repeated routing patterns, offers cheap-path shortcut |
| Three clocks (fast/medium/slow) | Detect instability at different timescales, modulate intervention |
| Governor + ledgers | Authorise or block cheap-path commits based on scar/cost/commitment history |
| Triad monitors | Detect angel/devil/maniac expert pathologies per layer |
| Conflict register | Track angel-devil co-occurrence, mode switching |
| Relational graph | Store typed entity triples, alias lookup. Inert -- does NOT touch routing |
| Association basins | Pre-softmax logit bias injection wiring. Plumbing only -- see Phase 8c |

## Phase 8c: Memory Bias Validation

5-condition A/B experiment testing whether pre-softmax logit bias from
entity-specific routing signatures does anything useful.

| Gate | What it checks |
|------|---------------|
| Gate 1 | B1 (scale=0.0) identical to baseline -- plumbing clean |
| Gate 2 | Toggle A (graph only) identical to baseline -- graph inert |
| Gate 3 | Bias audibility in routing units |
| Gate 4 | Geometry stability (Neff, entropy, scars) |
| Gate 5 | Primary success metric (synthetic entity loss) |

Results: `experiments/phase8c_logs/`

**Conclusion**: Memory bias is audible but does not improve synthetic entity
loss at tested scales. Graph-only is the correct architecture at this point.

## Tests

```
70 passed in 0.5s

tests/test_triad_monitors.py      17 tests  Angel/Devil/Maniac monitors + conflict register
tests/test_memory_graph.py         25 tests  Relational graph CRUD, serialization, assertion detection
tests/test_memory_basins.py        24 tests  Association basins, entity linking, bias computation
tests/test_b1_equivalence.py        4 tests  Deterministic B1=C0 guarantee, cross-run determinism
```
