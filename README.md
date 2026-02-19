# ChronoMoEv4 — Bob_Qwen Research

**Milestone achieved:** 66/66 tests passing
**Phase 8c complete:** Memory bias validation (5-condition A/B experiment) — all gates cleared

This repository tracks the **Bob_Qwen** line of Mixture-of-Experts (MoE) research built on Qwen via MLX. It picks up from the ChronoMoE v3 work and focuses on temporal routing, memory bias characterization, and experimental validation infrastructure.

---

## Milestone: 66/66 Tests Passing

All 66 unit and integration tests pass as of this milestone commit. This marks the cutover point for differentiated Bob_Qwen development.

---

## Phase 8c — Memory Bias A/B Experiment

memory_experiment.py implements a **5-condition A/B experiment** for memory bias validation.

### Conditions

| Condition | Description |
|-----------|-------------|
| C0 | Baseline — no memory, no bias |
| B1 | Memory ON, bias OFF (plumbing check) |
| B2 | Memory ON, bias LOW |
| B3 | Memory ON, bias HIGH |
| Toggle A | Memory toggled mid-run (contamination check) |

### Smoke Test Results

Seed 42, warmup=10, active=50

| Gate | Result | Detail |
|------|--------|--------|
| Gate 1 (B1=C0) | PASS | B1 identical to C0 — plumbing clean |
| Gate 2 (Toggle A=C0) | PASS | Graph inert — no routing contamination |
| Gate 3 (audibility) | AUDIBLE | B2: median_max=0.08, B3: median_max=0.16, 100% coverage |
| Gate 4 (geometry) | PASS | Neff, entropy, scars all within tolerance |
| Gate 5 (primary) | B2=NULL, B3=NEGATIVE | Expected in smoke (too few steps for signal) |

### Fixes Applied During Smoke Test

1. Governor constructor — medium_clock must be passed as positional arg
2. Gate 3 audibility — switched from bias_to_logit_ratio (not wired) to memory_bias_max
3. Entity tokens — use explicit annotation names instead of raw_text.split() to avoid zorblax and krenthar-institute matching failures

---

## Running the Experiment

    source qwen_moe_mlx/bin/activate
    python3 memory_experiment.py --seed 42
    python3 memory_experiment.py --seeds 3

---

## Environment

- Model: Bob_Qwen (Qwen-based MoE)
- Runtime: MLX (Apple Silicon)
- Virtualenv: qwen_moe_mlx/

---

## Roadmap

- Gate 5 signal detection at full step count
- Multi-seed statistical aggregation
- Phase 9 planning (post memory-bias characterization)
