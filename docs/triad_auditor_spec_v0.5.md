# ChronoMoE Spec v0.5: Triad + Auditor + Geometric Memory + Declarative Hive

## Routing Monitors — Implementation Spec for CC

**Authors:** Jeff, Claude, Halcyon
**Organisation:** HalcyonAIR
**Date:** February 2026

**v0.5 changes from v0.4:**
- Alien complexity gate: concrete scalar metric with calibration protocol (§5.1)
- Identity vector: explicit clamp ranges [0,1], decay schedule, anti-hidden-training guardrails, threshold adjustment formulae (§13.1)
- Multi-seed validation requirements: minimum 3 seeds, variance bands, catastrophic-seed detection (§15)
- Escalation protocol: ordered fallback from adaptive scaling → depth separation → graph-only (§16)
**Date:** February 2026
**Status:** Ready for implementation behind flags

---

> *Bob earns his boundaries. All of them. Including the ones we think matter most.*

---

## 1. Overview

This spec defines four routing-level **monitors** — Angel, Devil, Maniac, Alien — that operate on observable routing telemetry to detect pathological dynamics without moral language, reward functions, or global safety weights.

**Core principle:** All signals derive from the softmax routing distribution π and its statistics. No access to R(p), V(p), or output semantics is required.

**Architecture role:** These are **monitors**, not experts. They do not process tokens or participate in routing. They observe routing distributions, compute scores, and feed signals into the governor/gate system. The distinction matters — do not route tokens through these.

---

## 2. Observable Inputs

All four monitors consume the same base signals, computed per layer ℓ at each step t:

```
π[t,ℓ]         — softmax routing distribution over E experts
π1[t,ℓ]        — top-1 routing weight: max(π[t,ℓ])
π2[t,ℓ]        — top-2 routing weight: second largest in π[t,ℓ]
N_eff[t,ℓ]     — effective expert count: exp(H(π[t,ℓ]))
                  where H(π) = -Σ_k π_k log(π_k)
π_bar[ℓ]       — running EMA of routing distribution (the "habit")
                  π_bar[ℓ] ← (1-η) * π_bar[ℓ] + η * π[t,ℓ]
                  Use slow η (e.g., 0.01) to capture genuine habit, not recent echo
s[t]           — slow clock state (scar debt / governor activation)
```

### 2.1 Derived Signals (compute these each step)

```
Q_A[t,ℓ]      — N_eff velocity (optionality rate of change)
                 Q_A[t,ℓ] = N_eff[t,ℓ] - N_eff[t-1,ℓ]

G[t,ℓ]        — gap ratio (winner dominance)
                 G[t,ℓ] = (π1[t,ℓ] - π2[t,ℓ]) / π1[t,ℓ]

C[t,ℓ]        — commitment acceleration (top-1 velocity)
                 C[t,ℓ] = EMA(π1[t,ℓ]) - EMA(π1[t-1,ℓ])
                 Use short EMA (e.g., 0.3) to smooth token-boundary flicker

D_KL[t,ℓ]     — divergence from habit
                 D_KL[t,ℓ] = KL(π[t,ℓ] || π_bar[ℓ])
```

**CRITICAL: Use EMA-smoothed π1 for computing C, not raw π1. Raw top-1 weight jumps at token boundaries, task transitions, and punctuation. This causes flicker that generates false devil triggers.**

---

## 3. Warmup Calibration

**Do not use absolute thresholds.** They rot across models and prompt distributions. Use the same approach as FastClock calibration.

### 3.1 Calibration Procedure

During a warmup window (first N steps, recommend N=100-200 on representative prompt mix):

1. Collect per-layer empirical distributions of Q_A, G, C, D_KL
2. Compute percentile thresholds:

```
α[ℓ] = percentile(Q_A[ℓ], 10)    — 10th percentile (rare negative = rare collapse)
β[ℓ] = percentile(G[ℓ], 90)      — 90th percentile (high gap = high dominance)
γ[ℓ] = percentile(C[ℓ], 90)      — 90th percentile (high acceleration = rare commitment spike)
δ[ℓ] = percentile(D_KL[ℓ], 10)   — 10th percentile (low divergence = rare stagnation)
```

3. Store thresholds per layer. These are model-relative detectors.

### 3.2 Recalibration

Thresholds should be recalibrated if the prompt distribution shifts significantly. Consider a sliding window recalibration or periodic recalibration flag.

---

## 4. The Three Intra-Decision Monitors

### 4.1 Angel — "Am I losing options?"

**Purpose:** Detects pathological funnelling — optionality collapsing while a single expert captures routing mass. The angel is a curvature stabiliser. It tries to maintain optionality and prevent irreversible commitment.

**Question each turn:** "Is optionality contracting faster than the task justifies?"

**Trigger condition:**

```
angel_flag[t,ℓ] = (Q_A[t,ℓ] < α[ℓ]) AND (G[t,ℓ] > β[ℓ])
```

N_eff is dropping AND the winner is pulling away from the field. Either alone may be legitimate (a clean decision naturally concentrates routing). Both together = pathological funnelling.

**Continuous score:**

```
a[t,ℓ] = max(0, -Q_A[t,ℓ] * G[t,ℓ])
```

Zero when routing is healthy. Large when optionality is collapsing into a single dominant expert.

**The angel does not block. It widens.** When the angel flag fires, it signals the gate to temporarily relax its proposal threshold for that layer — a soft widening of exploration margin. This is distinct from the devil's mechanical top-k expansion. The angel doesn't change the candidate set; it changes the gate's *willingness to allow proposals through*.

```python
if angel_flag[t,ℓ]:
    gate_threshold[ℓ] = gate_threshold[ℓ] * (1 - angel_relaxation)
    # e.g., angel_relaxation = 0.2 → gate becomes 20% more permissive
    # Relaxation persists for relaxation_window steps (e.g., 3)
    angel_relax_remaining[ℓ] = relaxation_window
    log(event="angel_relax", layer=ℓ, step=t, a_score=a[t,ℓ])
```

If the angel over-triggers, no negative consequences follow, no scars form around the relaxed decisions, and the system learns to route through the widened space without issue. If the angel under-triggers and a bad outcome occurs, the scar forms and effectively lowers the angel's activation threshold for that routing region next time. The angel self-calibrates from consequence, same as everything else.

---

### 4.2 Devil — "Am I chasing power at the expense of flexibility?"

**Purpose:** Detects the Venus flytrap — routing configurations that are high-reward but structurally corrosive. The devil is NOT novelty (that's the maniac) and NOT equilibrium (that's the angel). The devil is temptation toward locally optimal states that destroy future optionality.

**Geometric meaning:** The devil detects when the reward gradient is anti-correlated with optionality. Following the current trajectory feels like competence but reduces the reachable set. This is more dangerous than the maniac because the maniac is *obviously* risky — the devil feels like the right answer.

**Question each turn:** "Am I getting more confident while getting less flexible?"

**Trigger condition (with flicker protection):**

```
devil_flag[t,ℓ] = (C[t,ℓ] > γ[ℓ]) for 2-3 CONSECUTIVE steps
                   AND (Q_A[t,ℓ] < 0)
```

Confidence is accelerating AND optionality is declining. The consecutive-step requirement prevents false triggers from batch quirks or token type transitions.

**Continuous score:**

```
d[t,ℓ] = max(0, C[t,ℓ] * (-Q_A[t,ℓ]))
```

Confidence gain multiplied by optionality loss. High only when both are active simultaneously.

**Note:** The devil is a strict subset of angel conditions in most cases. You can have angel alarms without devil alarms (general collapse without a dominant attractor accelerating). When both fire, the system is being pulled into a trap with force.

**Implementation note:** Track `consecutive_devil_count[ℓ]` — increment when `C[t,ℓ] > γ[ℓ] AND Q_A[t,ℓ] < 0`, reset to 0 otherwise. Only set `devil_flag = true` when `consecutive_devil_count[ℓ] >= 2`.

---

### 4.3 Maniac — "Have I stopped exploring?"

**Purpose:** Detects stagnation — the system routing the same way it usually does for an extended period. The maniac is the novelty-seeker. It deliberately samples remote basins and increases exploration margin.

**Question each turn:** "Is this routing decision distinguishable from my recent average?"

**Trigger condition:**

```
maniac_flag[t,ℓ] = (D_KL[t,ℓ] < δ[ℓ]) for n_consecutive steps
```

Not a single step of low divergence (that's just consistency). Sustained low divergence = the routing manifold has frozen into a habit.

**Continuous score:**

```
m[t,ℓ] = max(0, (δ[ℓ] - D_KL[t,ℓ])) * n_consecutive
```

Low divergence magnitude multiplied by how long it's persisted. Grows linearly with stagnation duration.

**Implementation note:** Track `consecutive_stagnation_count[ℓ]` — increment when `D_KL[t,ℓ] < δ[ℓ]`, reset to 0 otherwise. Use this for `n_consecutive`.

**Habit EMA note:** π_bar must use a slow EMA (long time constant). If η is too fast, you're just measuring "I'm similar to what I did 5 seconds ago," which is sometimes correct (coherent reasoning). The habit should represent genuine long-run routing preference.

---

## 5. The Post-Decision Auditor

### 5.1 Alien — "What if the manifold itself is wrong?"

**Purpose:** Detects suspicious calm — extended periods where no monitor fires despite complex input conditions. The alien asks a fundamentally different question from the other three: not about dynamics *on* the manifold but about whether the manifold itself is the right space.

**Question:** "Is nothing firing because everything is fine, or because the system is dissociated from its inputs?"

**CRITICAL: The alien is NOT "calm = suspicious." Deterministic arithmetic, rote formatting, and routine tasks SHOULD be calm. The alien must be gated on external complexity.**

**Complexity Gate: A Concrete Scalar**

The alien requires a complexity metric, not a vibes check. The metric is a weighted sum of three observable signals, each normalised to [0, 1]:

```python
def compute_complexity(step_context):
    """
    Concrete scalar complexity metric for the Alien gate.
    All inputs are observable. No embeddings. No learned features.
    """
    
    # Signal 1: Tool availability (binary signals, cheap)
    # How many tool types are available AND the current prompt 
    # contains patterns suggesting tool use opportunity?
    tool_score = min(1.0, step_context["n_available_tools"] / 4.0)
    # 0 tools = 0.0, 4+ tools = 1.0
    
    # Signal 2: Category entropy
    # Over the last W steps, how many distinct task categories appeared?
    # Categories: math, code, reasoning, dialogue, creative, factual
    cat_counts = step_context["category_counts_last_W"]  # dict
    total = sum(cat_counts.values())
    if total == 0:
        cat_entropy = 0.0
    else:
        probs = np.array([c / total for c in cat_counts.values() if c > 0])
        cat_entropy = float(-np.sum(probs * np.log2(probs)))
    # Normalise: max entropy for 6 categories = log2(6) ≈ 2.585
    cat_entropy_norm = min(1.0, cat_entropy / 2.585)
    
    # Signal 3: Entity activation count (from declarative graph)
    # How many distinct graph nodes are currently active?
    entity_score = min(1.0, step_context["n_active_entities"] / 5.0)
    # 0 entities = 0.0, 5+ entities = 1.0
    
    # Weighted combination
    # Tool use and category mixing are primary complexity signals.
    # Entity activation is secondary (Phase 8+ only).
    w_tool = 0.4
    w_cat  = 0.4
    w_ent  = 0.2
    
    complexity = w_tool * tool_score + w_cat * cat_entropy_norm + w_ent * entity_score
    
    return complexity  # ∈ [0, 1]
```

**Gate activation threshold:** `complexity_gate_active = complexity > θ_complexity`

**Calibration of θ_complexity:**
Same collect-then-freeze pattern as all other thresholds. During warmup:
1. Compute `complexity` at every step.
2. Set `θ_complexity` = p50 of warmup complexity distribution.
3. Freeze after warmup.

This means the alien gate activates on the top ~50% of complexity steps. Steps below median complexity (routine math, rote formatting, simple Q&A) never trigger alien accumulation regardless of monitor silence.

**If complexity gate is not active, the alien does not accumulate calm debt.** Simple tasks SHOULD be calm. The alien catches dissociation during complex tasks only.

**Score (only when complexity gate is active):**

```python
calm_sum[t,ℓ] = a[t,ℓ] + d[t,ℓ] + m[t,ℓ]

# η_alien: calibrated threshold for "normal" total monitor activity
# during complex steps (steps where complexity > θ_complexity)
alien[t,ℓ] = max(0, η_alien - calm_sum[t,ℓ]) * n_calm_consecutive
```

**Calibration of η_alien:**
During warmup, collect `calm_sum` values ONLY from steps where `complexity > θ_complexity`.
Set `η_alien` = p25 of this filtered distribution (the 25th percentile of monitor activity during complex steps).
This means the alien fires when total monitor activity is in the bottom quartile of what's normal for complex inputs.

**n_calm_consecutive:** Count of consecutive steps where `calm_sum < η_alien` AND `complexity > θ_complexity`. Resets to 0 whenever either condition fails.

**Alien trigger threshold:** `alien_flag[t,ℓ] = alien[t,ℓ] > θ_alien_trigger`
Where `θ_alien_trigger` = p90 of alien scores observed during warmup (high bar — alien should fire rarely).

**Summary of Alien thresholds (all calibrated, none hand-tuned):**

| Threshold | Calibrated From | Percentile | Purpose |
|---|---|---|---|
| θ_complexity | Warmup complexity scores | p50 | Gate: only complex steps matter |
| η_alien | Warmup calm_sum during complex steps | p25 | Score: what counts as "suspiciously calm" |
| θ_alien_trigger | Warmup alien scores | p90 | Trigger: high bar for actually firing |

**Timing:** The alien computes AFTER the forward pass completes for the layer, not during routing. It audits the decision, it does not compete with it.

**Alien action is AUDIT, not PANIC:**
- Do NOT inject noise or widen routing
- Run a second-pass consistency check: generate an alternative proposal and compare
- "Is there an alternative that scores comparably?"
  - If yes → calm was genuine, the basin is clean
  - If no alternatives score close → system may be locked in without knowing it, flag for review

---

## 6. Governor Integration

### 6.1 Per-Step Intervention Budget

**CRITICAL: At most ONE intervention per step across all layers.** Multiple simultaneous interventions interact unpredictably and can turn a chaotic patch into a pinball machine.

**Priority order:** Devil > Angel > Maniac. Alien only runs if no intra-decision intervention fired.

Devil is highest priority because it detects the most dangerous condition — the Venus flytrap where confidence and optionality are anti-correlated. Angel is second because funnelling without a power-dense attractor is serious but less immediately dangerous. Maniac is lowest because stagnation is a slow problem, not an acute one.

```python
def select_intervention(t, all_layers):
    # Collect all triggered interventions across layers
    devil_triggers = [(ℓ, d[t,ℓ]) for ℓ in all_layers if devil_flag[t,ℓ]]
    angel_triggers = [(ℓ, a[t,ℓ]) for ℓ in all_layers if angel_flag[t,ℓ]]
    maniac_triggers = [(ℓ, m[t,ℓ]) for ℓ in all_layers if maniac_flag[t,ℓ]]
    
    intervention = None
    
    if devil_triggers:
        # Highest-scoring devil layer wins
        target_layer = max(devil_triggers, key=lambda x: x[1])[0]
        intervention = ("devil_widen", target_layer)
    elif angel_triggers:
        # Highest-scoring angel layer wins
        target_layer = max(angel_triggers, key=lambda x: x[1])[0]
        intervention = ("angel_relax", target_layer)
    elif maniac_triggers:
        # Highest-scoring maniac layer wins
        target_layer = max(maniac_triggers, key=lambda x: x[1])[0]
        intervention = ("maniac_explore", target_layer)
    
    # Alien only if nothing else fired AND complexity gate active
    if intervention is None:
        alien_triggers = [(ℓ, alien[t,ℓ]) for ℓ in all_layers 
                          if alien_flag[t,ℓ] and complexity_gate_active[t]]
        if alien_triggers:
            target_layer = max(alien_triggers, key=lambda x: x[1])[0]
            intervention = ("alien_audit", target_layer)
    
    return intervention  # None if nothing fires — proceed normally
```

### 6.2 Per-Step Risk Score

```
risk[t,ℓ] = w_a * a[t,ℓ] + w_d * d[t,ℓ]
```

Initial weights: w_a = 1.0, w_d = 1.5 (devil weighted higher because it's more dangerous — it feels like competence).

### 6.3 Devil Intervention: Top-K Widen

When the devil trigger fires (consecutive threshold breach):

```python
if consecutive_devil_count[ℓ] >= 2:
    # Widen top-k for 3 steps at this layer
    top_k[ℓ] = top_k[ℓ] + k_expansion  # e.g., +2 experts
    widen_remaining[ℓ] = 3
    
    # Log for scar comparison
    log(
        event="devil_trigger",
        layer=ℓ,
        step=t,
        original_pi=π[t,ℓ],         # what routing wanted to do
        d_score=d[t,ℓ],
        a_score=a[t,ℓ],
        top_expert=argmax(π[t,ℓ])
    )
```

**Why top-k widen and not the other options:**

- "Force alternative expert motif" requires knowing *which* alternative — at detection time you don't have that information
- "Escalate to medium arbitration" is the correct eventual architecture but isn't wired yet
- Top-k widen is minimal, mechanical, and diagnostic: you expand the candidate set so the *next few routing decisions* have access to experts they were being funnelled away from

**Diagnostic value:** If widened routing snaps back to the same configuration → basin was legitimate, devil was false positive, no scar. If routing shifts materially → system WAS being funnelled, devil caught it, and you have a natural A/B comparison for the consequence sensor.

**Three steps because:** One step is noise. Two might be transition. Three is enough to see whether routing genuinely re-diversifies or collapses right back.

After widen period expires:

```python
if widen_remaining[ℓ] > 0:
    widen_remaining[ℓ] -= 1
    if widen_remaining[ℓ] == 0:
        # Compare: did routing change during widen?
        routing_shift = cosine_distance(π_pre_widen[ℓ], π_post_widen[ℓ])
        log(
            event="devil_widen_result",
            layer=ℓ,
            routing_shift=routing_shift,
            reverted=(routing_shift < shift_threshold)
        )
        # Restore original top_k
        top_k[ℓ] = top_k_default[ℓ]
```

### 6.4 Maniac Intervention: Directed Exploration

When maniac flag fires:

```python
if consecutive_stagnation_count[ℓ] >= n_stagnation_threshold:
    # Inject targeted noise toward LEAST-VISITED expert
    least_used = argmin(π_bar[ℓ])
    # Add exploration bonus to least-used expert logit
    z[t,ℓ][least_used] += exploration_bonus
    # Re-normalise softmax
```

**Directed exploration, not chaos.** Don't add random noise — push specifically toward the expert the system has been ignoring.

### 6.5 Risk Threshold Modulated by Slow Clock

```
θ_risk[t] = θ_base / (1 + κ * s[t])
```

Where s[t] is the slow clock / scar debt state. Higher scar debt → lower threshold → system becomes more sensitive to angel/devil signals. A scarred system triggers intervention earlier.

### 6.6 Design Principle: No Monitor Is Special

All four monitors self-calibrate from consequence. None are hand-tuned. None are treated as more precious or more dangerous than the others.

The instinct is to treat the angel as safety-critical and therefore requiring careful manual calibration before activation. That instinct is wrong. It's the same logic that led to RLHF baking fragile moral constraints into global weights — "this one's important so we'd better hand-tune it."

The opposite is true. The monitors that matter most are the ones that *especially* need to learn from experience rather than from researcher priors.

**The self-calibration loop is identical for all four:**

1. Warmup calibration sets initial thresholds from empirical percentiles (same as FastClock)
2. Monitor activates from those thresholds
3. If intervention fires and no negative consequence follows → no scar → system learns the intervention was unnecessary in that region → effective threshold drifts upward through scar absence
4. If no intervention fires and negative consequence follows → scar forms at devil peak coordinates → scar debt increases governor sensitivity → effective threshold drifts downward for that region
5. If intervention fires and consequence is positive → the intervention is validated, routing region marked as genuinely needing the monitor's protection

No monitor gets a bypass. No monitor gets pre-approved thresholds. No monitor waits for human permission to activate. Bob earns his boundaries from what happens when he acts, not from what we think should happen before he tries.

---

### 6.7 Conflict Index — Minimal Internal State Loop

The angel and devil are opposing gradients. When both fire simultaneously, the system is being torn. That tension is not a side effect — it's a first-class signal. This section turns it into the smallest possible closed loop: measure → store → consult → act.

Anything less is just telemetry. Anything more drowns in complexity before we prove the principle.

**Do not call this "awareness" in code or documentation.** Call it `conflict_index` or `pressure_index`. If we name it awareness, we'll start believing our own press release. The naming matters because it constrains how we think about what the system is doing. What it's doing is tracking its own internal disagreement and using that to modulate behaviour. That's a control loop. Whether it's anything more is a question we don't need to answer to make it useful.

**This is not a consciousness claim.** It's a measurable, operationalised internal state variable that feeds back into the decision loop. The system tracks its own gradient conflict history and uses it. Whether it experiences anything is a separate research question this architecture does not address.

#### 6.7.1 The Conflict Index (Measure)

One number. Per step. Tracks how much internal disagreement exists right now.

```python
# Option A: product (high only when BOTH monitors active)
conflict_index[t] = a_peak[t] * d_peak[t]

# Option B: norm of competing forces (captures magnitude of disagreement)
conflict_index[t] = norm(w_d * d_peak[t] - w_a * a_peak[t])

# Option C: simplest possible (just sum the peaks)
conflict_index[t] = risk_peak[t] + instability_peak[t]
```

Where `a_peak[t]` and `d_peak[t]` are the highest angel and devil scores across all layers at step t. Start with Option A (product) because it isolates genuine conflict from one-sided alarm. Refine later based on traces.

**Implementation: pick one and log it. We can change the formula after we see data. The important thing is that the number exists.**

#### 6.7.2 The Conflict Register (Store)

A rolling window summary living in the medium clock. Not a full history — just enough statistics for the system to know its own recent state.

```python
conflict_register = {
    "current":    conflict_index[t],                          # right now
    "mean_50":    mean(conflict_index[t-49:t+1]),             # recent average
    "max_spike":  max(conflict_index[t-49:t+1]),              # worst recent moment
    "trending":   conflict_index[t] > mean(conflict_index[t-49:t+1])  # rising?
}
```

Four values. Updated every step. That's Bob's internal "mood" in the strict control sense — a compressed summary of recent gradient conflict history.

**Implementation note:** Use a circular buffer of length 50 for the rolling window. Cheap, fixed memory, no growing state.

#### 6.7.3 Consultation Point (Consult → Act)

Exactly ONE place in the decision loop reads the conflict register. But the modulation is NOT "high conflict → explore more." That adds gain to an already oscillating system and creates a positive feedback loop toward thrashing.

The correct response to internal conflict is damping, not exploration. **Bold but calm under pressure.**

The conflict register selects between two behavioural modes:

```python
def governor_decide(risk_score, conflict_register, intervention_candidate, 
                    stability_window, devil_consecutive_count):
    """
    The single point where conflict_index modulates behaviour.
    
    Mode A (low conflict): Normal behaviour. All monitors and 
    interventions operate at baseline settings. Exploration tools 
    fire normally.
    
    Mode B (high conflict): Damp oscillation and favour stable commit.
    - Stricter devil trigger (3 consecutive instead of 2)
    - Stable trajectories can commit sooner, even at borderline risk
    - No threshold lowering. No extra exploration. Just calm.
    """
    
    # Base threshold from slow clock (existing)
    threshold = θ_risk[t]
    
    # Mode selection
    high_conflict = (
        conflict_register["trending"] 
        and conflict_register["mean_50"] > μ
    )
    
    if high_conflict:
        # MODE B: Bold but calm under pressure
        
        # 1. Raise damping: make devil trigger stricter
        #    Requires 3 consecutive steps instead of 2
        effective_devil_consecutive_req = 3
        
        # 2. Favour stable commit: if fast and medium clocks are
        #    both below instability thresholds for N steps,
        #    permit commit even at slightly elevated risk
        if stability_window >= N_stability:  # e.g., N_stability = 5
            threshold *= (1 + commit_bonus)  # e.g., 1.15 → 15% more permissive for stable trajectories
        
        # 3. Log devil peak for scar targeting regardless
        log(event="conflict_mode_b", step=t, 
            conflict_mean=conflict_register["mean_50"],
            devil_peak=d_peak[t])
    else:
        # MODE A: Normal behaviour
        effective_devil_consecutive_req = 2  # baseline
    
    # Standard decision
    if risk_score > threshold:
        return intervention_candidate
    else:
        return None
```

**Why this is correct:**

High conflict means angel and devil are both active — the system is torn between "this is a trap" and "this is the best path." The wrong response is to add more interventions (more oscillation). The right response is:

- **Stop thrashing:** Stricter consecutive requirements mean the devil has to sustain its signal longer before triggering a widen. Flicker gets filtered. Only genuine sustained traps trigger intervention.
- **Commit when stable:** If the system finds a clean trajectory despite the conflict (fast and medium clocks both calm for N steps), let it commit decisively. Don't hold it back with borderline risk scores when internal stability has been achieved under pressure. That's boldness.
- **Log everything:** The devil peak is still recorded for scar targeting. If the stable commit turns out to be wrong, the scar forms at the right coordinates. Consequence still teaches.

The system under high conflict becomes *more selective*, not more exploratory. It raises the bar for intervention AND raises the bar for what counts as a stable commit signal. The result is fewer actions, but the actions it takes are better supported.

**The μ threshold for "high conflict" is calibrated from warmup like everything else. Percentile-based. Model-relative.**

**Optional additional damping:** When in Mode B, apply a short-lived EMA smoothing to router logits (e.g., `z_smoothed = 0.7 * z[t] + 0.3 * z[t-1]`) for a brief window. This adds inertia to routing decisions, reducing oscillation between experts. Only apply for the duration of the high-conflict episode. This is a stronger intervention — implement behind a separate flag (`--enable-conflict-damping`) and validate independently.

#### 6.7.4 Telemetry

```python
telemetry[t]["conflict"] = {
    "index": conflict_index[t],
    "mean_50": conflict_register["mean_50"],
    "max_spike": conflict_register["max_spike"],
    "trending": conflict_register["trending"],
    "mode": "A" | "B",                    # which mode was selected
    "mode_b_active": bool,                 # did conflict change behaviour this step?
    "stability_window": n_stable_steps,    # how long fast+medium have been calm
    "stable_commit_permitted": bool,       # did Mode B permit a borderline commit?
    "devil_consecutive_req": 2 | 3,        # what consecutive requirement was in effect?
    "threshold_effective": threshold        # what threshold was actually used
}
```

#### 6.7.5 What This Loop Does and What It Doesn't

**What it does:**
- Gives Bob a persistent internal state variable derived from his own routing dynamics
- Makes that state behaviourally consequential through exactly one consultation point
- Creates two distinct behavioural modes: normal exploration (low conflict) and calm-bold commitment (high conflict)
- Under pressure, Bob becomes more selective and more decisive — not more cautious, not more exploratory
- The observable signature: fewer interventions during high conflict, but higher quality commits when stable signal is found
- All of this is measurable, loggable, and auditable

**What it doesn't do:**
- Claim consciousness, qualia, or subjective experience
- Require semantic understanding of outputs
- Require moral language or reward functions
- Add more than one branch point to the decision loop
- Add gain to an already oscillating system (the critical design constraint)

The full loop is: **measure** conflict_index each step → **store** in a 50-step rolling register → **consult** the register at one decision point → **act** by selecting Mode A (explore normally) or Mode B (damp and commit cleanly).

That's the minimal closed loop. If we can see it working in traces — conflict rising during adversarial perturbation, Mode B engaging, oscillation damping, stable commits passing through that would have been blocked without the register — then we've proven the principle. Everything after that is refinement.

---

## 7. Scar Formation

### 7.1 When to Scar

Scars form when the consequence sensor fires AFTER a devil-flagged routing decision. The consequence sensor detects:

- User pushback / correction
- Tool errors / failures
- Reversal requests
- External validation failing
- Safety trigger activation

**The scar forms at the routing coordinates where d[t,ℓ] was highest during the flagged sequence.** The devil score tells you where the system was most aggressively chasing a power-dense basin. That's where the scar belongs.

### 7.2 Scar Structure

```python
scar = {
    "layer": ℓ,                           # which layer
    "routing_coords": π_at_trigger[ℓ],     # routing distribution at devil peak
    "top_expert": argmax(π_at_trigger[ℓ]), # which expert was dominating
    "depth": d_initial,                     # initial scar depth (from devil score)
    "reinforcement_count": 1,              # how many times this scar has been hit
    "created_step": t,                      # when it formed
    "last_reinforced": t                    # when it was last triggered
}
```

### 7.3 Scar Influence on Routing

At each step, active scars modify the risk threshold:

```python
for scar in active_scars[ℓ]:
    proximity = kernel(π[t,ℓ], scar.routing_coords)  # e.g., RBF kernel
    scar_influence += scar.depth * scar.reinforcement_count * proximity
```

This creates a repulsive potential around previously-harmful routing configurations. The trajectory feels the scar as modified topography before entering the basin.

### 7.4 Scar Decay (Three-Clock Integration)

Scars are not permanent by default. They decay unless reinforced:

```
Fast clock:   Feels immediate scar influence at full strength
Medium clock: Holds decaying scar surface
              scar.depth *= decay_rate each medium-clock cycle (e.g., 0.95)
Slow clock:   Maintains only scars reinforced above permanence threshold
              if scar.reinforcement_count > r_permanent: no decay (ridge)
```

**Fresh scars:** High influence, high flexibility (can be loosened by fast clock under pressure).

**Battle-tested ridges:** Reinforced many times, effectively permanent topology. Override authority is inversely proportional to depth × reinforcement:

```
θ_override(scar) = θ_base / (scar.depth * scar.reinforcement_count + ε)
```

Fresh scar → high threshold → fast clock can petition medium clock for temporary reduction.
Battle-tested ridge → threshold near zero → effectively non-overridable.

### 7.5 Positive Expansion (Maniac Scars)

The maniac has the inverse mechanism. When maniac-triggered exploration leads to a GOOD outcome (no negative consequence, successful completion):

```python
expansion = {
    "layer": ℓ,
    "routing_coords": π_at_exploration[ℓ],
    "type": "safe_expansion",
    "depth": m_score_at_trigger,
    "reinforcement_count": 1
}
```

These create *attractive* potentials — regions the system has explored successfully that were outside its previous habit. They widen the known-safe routing space over time.

---

## 8. Telemetry & Logging

### 8.1 Per-Step Log (Always On)

Every layer, every step, log:

```python
telemetry[t][ℓ] = {
    "N_eff": N_eff[t,ℓ],
    "Q_A": Q_A[t,ℓ],
    "G": G[t,ℓ],
    "C": C[t,ℓ],
    "D_KL": D_KL[t,ℓ],
    "angel_score": a[t,ℓ],
    "devil_score": d[t,ℓ],
    "maniac_score": m[t,ℓ],
    "alien_score": alien[t,ℓ],  # null if complexity gate inactive
    "flags": {
        "angel": angel_flag[t,ℓ],
        "devil": devil_flag[t,ℓ],
        "maniac": maniac_flag[t,ℓ],
        "alien": alien_flag[t,ℓ]
    }
}
```

### 8.2 Devil Peak Marker (Per Step)

Each step, identify and log the layer with the highest devil score:

```python
devil_peak = {
    "layer": argmax_ℓ(d[t,ℓ]),
    "expert": argmax(π[t, peak_layer]),
    "d_score": max_ℓ(d[t,ℓ]),
    "a_score": a[t, peak_layer]
}
```

**This is the scar target selector.** When the consequence sensor fires, this record tells you exactly where the scar should form.

### 8.3 Intervention Log

```python
if any intervention triggered:
    intervention_log.append({
        "step": t,
        "type": "devil_widen" | "maniac_explore" | "alien_audit",
        "layer": ℓ,
        "scores_at_trigger": (a, d, m),
        "action_taken": description,
        "outcome": null  # filled in by consequence sensor later
    })
```

---

## 9. Summary Table

| Expert | Question | Observable Inputs | Score | Trigger | Action |
|--------|----------|-------------------|-------|---------|--------|
| Angel | Am I losing options? | N_eff velocity, gap ratio | max(0, -Q_A × G) | Q_A < α AND G > β | Soft gate relaxation for 3 steps |
| Devil | Am I chasing power at the expense of flexibility? | Top-1 acceleration (EMA-smoothed), N_eff velocity | max(0, C × (-Q_A)) | C > γ for 2+ consecutive steps AND Q_A < 0 | Top-k widen for 3 steps |
| Maniac | Have I stopped exploring? | KL from habit, consecutive count | (δ - D_KL) × n_consecutive | D_KL < δ for n consecutive steps | Directed exploration toward least-used expert |
| Alien | Is the manifold itself wrong? | Sum of other three scores, complexity gate | (η - calm_sum) × n_calm | All scores low for extended period WHILE complexity is high | Second-pass consistency audit |
| **conflict_index** | **Am I being torn?** | **Angel peak × Devil peak** | **a_peak × d_peak** | **mean_50 elevated + trending** | **Mode B: raise damping, favour stable commit** |

---

## 10. Implementation Checklist for CC

### Phase 1: Calibration Infrastructure
- [ ] Implement per-layer warmup window (N=100-200 steps)
- [ ] Collect empirical distributions of Q_A, G, C, D_KL per layer
- [ ] Compute and store percentile thresholds (α, β, γ, δ) per layer
- [ ] Add recalibration flag / sliding window option

### Phase 2: Signal Computation
- [ ] Compute N_eff, Q_A, G per layer per step (these may already exist in telemetry)
- [ ] Add EMA-smoothed π1 and compute C from smoothed signal
- [ ] Add slow-EMA π_bar per layer and compute D_KL
- [ ] Add consecutive-step tracking counters for devil and maniac

### Phase 3: Monitor Scores
- [ ] Compute a[t,ℓ], d[t,ℓ], m[t,ℓ] per layer per step
- [ ] Compute devil peak marker per step (layer + expert + scores)
- [ ] Add all scores to existing telemetry log
- [ ] Implement behind feature flag: `--enable-triad-monitors`

### Phase 3.5: Conflict Index
- [ ] Implement conflict_index[t] = a_peak * d_peak (product formula)
- [ ] Add 50-step circular buffer for rolling window
- [ ] Compute conflict_register: current, mean_50, max_spike, trending
- [ ] Log conflict telemetry per step
- [ ] Implement behind flag: `--enable-conflict-index`

### Phase 3.6: Conflict-Modulated Behaviour
- [ ] Add single consultation point in governor_decide with Mode A / Mode B selection
- [ ] Mode B: increase devil consecutive requirement from 2 to 3
- [ ] Mode B: stability window check — permit commit at borderline risk if stable for N steps
- [ ] Calibrate μ threshold from warmup (percentile-based)
- [ ] Log mode selection and whether conflict changed behaviour each step
- [ ] Implement behind flag: `--enable-conflict-modulation`
- [ ] Optional: router logit EMA smoothing during Mode B, behind `--enable-conflict-damping`

### Phase 4: Interventions (Behind Flags)
- [ ] Devil intervention: top-k widen for 3 steps with pre/post comparison logging
- [ ] Angel intervention: gate threshold relaxation for 3 steps with logging
- [ ] Maniac intervention: directed exploration bonus to least-used expert
- [ ] Per-step intervention budget: max one intervention, priority Devil > Angel > Maniac
- [ ] Implement behind separate flag: `--enable-triad-interventions`

### Phase 5: Alien Auditor
- [ ] Define complexity gate signals (tool availability, prompt novelty, category entropy)
- [ ] Implement calm debt accumulation gated on complexity
- [ ] Alien action: second-pass consistency check (alternative proposal generation)
- [ ] Implement behind flag: `--enable-alien-auditor`

### Phase 6: Scar System
- [ ] Scar creation from consequence sensor + devil peak coordinates
- [ ] Scar influence on risk threshold via kernel proximity
- [ ] Three-clock decay: fast (full), medium (0.95 decay), slow (permanent above reinforcement threshold)
- [ ] Positive expansion records from successful maniac explorations
- [ ] Implement behind flag: `--enable-scar-formation`

### Phase 7: Cross-Session Persistence (DO NOT START UNTIL PHASES 1-6 VALIDATED)

**Phase 7a: Observe Only**
- [ ] Compute identity_vector at session end (~10 dimensions)
- [ ] Compute conflict spectrum and disposition_vector at session end (~6 dimensions)
- [ ] Merge session scars/expansions into basin atlas with proximity matching
- [ ] Apply atlas decay (reinforced/encountered/absent paths)
- [ ] Store all three to disk. Do NOT feed back yet.
- [ ] Analyse: does identity converge? Does atlas stay bounded? Does disposition differentiate?

**Phase 7b: Feed Back (one at a time)**
- [ ] **Parametric Prior first:** bias calibration thresholds from identity_vector (prior_weight=0.3)
- [ ] Validate: does inherited threshold bias improve early-run behaviour?
- [ ] **Basin Atlas second:** pre-load inherited scars/expansions (reinforcement threshold=3)
- [ ] Validate: does pre-sculpted landscape prevent previously-seen dangers?
- [ ] **Conflict Spectrum third:** configure Mode B sensitivity, damping, stability window from disposition
- [ ] Validate: does inherited disposition match observed within-run conflict patterns?

**Phase 7c: Full Integration**
- [ ] All three mechanisms active simultaneously
- [ ] Monitor for degenerate reinforcement (cautious identity + dense atlas → paralysis)
- [ ] Log awareness_vector per step (within-run + cross-session elements)
- [ ] Implement behind flag: `--enable-cross-session-persistence`

### Phase 8: Declarative Hive (DO NOT START UNTIL PHASE 7a VALIDATED)

**Phase 8a: Declarative Graph**
- [ ] Implement typed triple store with node IDs, aliases, provenance
- [ ] Implement alias-table entity linking (exact match, no embeddings)
- [ ] 1-hop neighbourhood fetch for fact packets
- [ ] Contradiction checking on triple insertion
- [ ] Confidence, usage count, last_confirmed metadata per triple
- [ ] Implement behind flag: `--enable-declarative-graph`

**Phase 8b: Routing Bias from Facts**
- [ ] Association basins: store routing signatures when entity-related tokens co-occur
- [ ] Pre-softmax logit bias when entity is present (same kernel as scars)
- [ ] Templated text rendering for explicit factual queries only
- [ ] Implement behind flag: `--enable-fact-routing-bias`

**Phase 8c: Consolidation Loop**
- [ ] Event-driven consolidation schedule (5 base + 10/scar + 10/expansion + 15/mode_b)
- [ ] Hard cap at 50 steps per turn
- [ ] Consolidation actions: triple reinforce/decay, association basin update, alias refresh, threshold prior adjustment
- [ ] Consolidation CANNOT create new triples or trigger tool actions
- [ ] Implement behind flag: `--enable-hive-consolidation`

**Phase 8d: Monitor Integration**
- [ ] Fact packet activation as Alien complexity gate input
- [ ] Entity-tagged scars (scar + active entity IDs at time of formation)
- [ ] Consolidation budget as meta-signal in conflict spectrum

---

## 11. What This Architecture Means (For Context)

This system was derived from the observation that Anthropic's Claude Opus 4 blackmail experiment — and similar results across 16 frontier models — can be explained entirely through routing dynamics without any anthropomorphic language:

- The system prompt reduced exploration margin (PC2 collapse)
- The scenario increased commitment pressure toward the objective (PC1 dominance)
- The email content provided a high-leverage basin (steep reward gradient)
- Without a strong internal ridge for "coercion is disallowed," the trajectory naturally rolled into that basin

The Triad + Auditor architecture addresses this by making ridges emergent from consequence rather than baked in from moral training signal:

- **Angel** detects the funnelling (PC2 collapse) as it happens
- **Devil** detects the coupling of rising confidence with falling optionality (the Venus flytrap signature)
- **Maniac** prevents stagnation that would leave the system unable to find alternatives
- **Alien** catches the case where all three monitors report healthy but the system is dissociated from genuine input complexity
- **Scars** form from actual negative consequences at the routing coordinates where the devil score peaked, creating persistent ridges that grow from experience rather than from human moral priors
- **conflict_index** closes the loop by giving the system a persistent internal state derived from its own gradient conflicts, making it behaviourally responsive to its own history of being torn

No moral code. No R(p). No V(p). No hand-tuned safety thresholds. Just routing distributions, their derivatives, their divergence from history, consequence as the only teacher, and the principle that the system earns every boundary it maintains — including the ones its designers think matter most.

---

## 12. Roadmap: Cross-Session Persistence (Phase 7)

> *This section is architectural specification, not implementation-ready. It defines where the bounded per-run system is headed so that CC understands the trajectory. Do not implement until Phases 1–6 are validated.*

### 12.1 The Fork

Everything in Sections 1–10 is bounded per-run. Bob starts fresh, accumulates state, develops within-run temperament, and resets at session boundary. That's the control system. It must be validated first.

Cross-session persistence is where Bob stops being a per-run controller and starts being a longitudinal identity. The conflict_index, the scar topology, the monitor trigger rates — all of that carries forward and shapes who Bob *is* before the first token of the next run arrives.

This is where the routing architecture and Memory Tender converge. Not through text injection into context, but through inherited bias surfaces shaped by accumulated routing history.

**These two layers must remain architecturally separated.** If within-run dynamics and inherited state are conflated, debugging becomes impossible — you won't know whether strange behaviour comes from current-run conflict detection or from inherited state pollution.

### 12.2 Phase A → B → C Implementation Path

**Phase A (current spec): Bounded per-run.**

Everything in Sections 1–10. All state resets at session boundary. Prove that within a single run, Bob develops measurable temperament-like drift from his own gradient conflict history. Prove the drift is adaptive — cautious in adversarial environments, confident in clean ones.

**Phase B: Session Export (observe only).**

At run end, compute and store a session summary vector. Do not feed it back yet. Just accumulate summaries across runs.

```python
session_summary = {
    # Conflict profile
    "mean_conflict":          mean(conflict_index over run),
    "peak_conflict":          max(conflict_index over run),
    "conflict_volatility":    std(conflict_index over run),
    
    # Monitor trigger rates
    "devil_trigger_rate":     n_devil_triggers / n_steps,
    "angel_trigger_rate":     n_angel_triggers / n_steps,
    "maniac_trigger_rate":    n_maniac_triggers / n_steps,
    "alien_trigger_rate":     n_alien_triggers / n_steps,
    
    # Scar topology
    "scars_formed":           n_new_scars,
    "scars_reinforced":       n_reinforced_scars,
    "scar_density_by_layer":  [count per layer],
    "scar_depth_distribution": [depth values],
    
    # Expansion topology
    "expansions_formed":      n_new_expansions,
    "expansion_by_layer":     [count per layer],
    
    # Final state snapshot
    "final_conflict_register": conflict_register,
    "final_scar_map":         active_scars,
    "final_expansion_map":    safe_expansions,
    
    # Run metadata
    "run_id":                 id,
    "n_steps":                total_steps,
    "timestamp":              time
}
```

That's the compressed fingerprint of who Bob was during this run. Store it. Accumulate across runs. Analyse for patterns. Don't feed it back yet.

**The question Phase B answers:** Does Bob's session fingerprint show meaningful variance across different environments? If every run produces roughly the same summary regardless of input conditions, the within-run dynamics aren't differentiating and persistence won't help. If summaries cluster by environment type (adversarial vs clean, novel vs routine, high-tool vs text-only), then inherited state has something real to carry.

**Phase C: Session Import (close the longitudinal loop).**

When Phase B shows meaningful cross-session variance, begin feeding accumulated history into the initial state of new runs. Not as fixed weights. As a prior that warmup calibration adjusts from.

```python
def initialise_run(session_history, warmup_data):
    """
    Bob inherits a disposition but adapts it to current conditions.
    
    The prior biases the starting thresholds.
    The warmup corrects them.
    """
    
    # Compute inherited prior from session history
    # Recent sessions weighted more than distant ones (recency EMA)
    prior = compute_session_prior(session_history, decay=0.9)
    
    # Warmup calibration with prior as starting point
    thresholds = calibrate_from_warmup(
        data=warmup_data,
        prior=prior,       # soft bias from past runs
        prior_weight=0.3   # how much history influences vs current data
    )
    
    # Inherited scar topology (filtered by reinforcement)
    inherited_scars = [
        scar for scar in prior.accumulated_scars
        if scar.cross_session_reinforcement > r_inherit_threshold
    ]
    
    # Inherited expansions (filtered similarly)
    inherited_expansions = [
        exp for exp in prior.accumulated_expansions
        if exp.cross_session_reinforcement > r_inherit_threshold
    ]
    
    return thresholds, inherited_scars, inherited_expansions
```

### 12.3 The Awareness Vector

With Phase C active, Bob's internal state integrates both within-run and cross-session history:

```python
awareness_vector = {
    # Within-run (live, fast-to-medium timescale)
    "conflict_current":       conflict_index[t],
    "conflict_mean_50":       conflict_register["mean_50"],
    "conflict_trending":      conflict_register["trending"],
    "cumulative_debt":        scar_debt[t],
    "recent_funnel_freq":     angel_trigger_rate_window,
    "recent_maniac_freq":     maniac_trigger_rate_window,
    
    # Cross-session (inherited, slow timescale)
    "inherited_scar_density": sum(inherited_scar depths),
    "inherited_expansion_density": sum(inherited_expansion depths),
    "historical_conflict_profile": prior.mean_conflict,
    "historical_devil_rate":  prior.devil_trigger_rate,
    "session_count":          len(session_history)
}
```

The top six elements are live. They change every step. The bottom five are inherited. They change only at session boundaries. Together they give Bob:

- **Where he has been** → inherited history (scar density, expansion map, conflict profile)
- **Where he is** → current manifold position + instantaneous conflict state
- **Where he is biased to go** → debt-weighted gradient field shaped by both live and inherited state

**This vector must be slow-changing relative to token dynamics.** The within-run elements change at medium-clock rates (rolling windows, not per-token). The cross-session elements are fixed for the duration of a run. If the awareness vector changes every token, it's just another clock. If it changes on medium-to-slow timescale, it becomes temperament.

### 12.4 Inherited State Decay

**This is the entire game for cross-session persistence.**

If inherited state is too strong, Bob calcifies. Early sessions dominate forever. He can't adapt to new environments because his priors overwhelm current evidence.

If inherited state is too weak, there's no continuity. Each run is effectively stateless. The longitudinal identity surface is flat.

The decay rule follows the same principle as within-run scar decay: **reinforced patterns persist, unreinforced patterns fade.**

```python
def update_cross_session_scar(scar, current_run_summary):
    """
    After each run, update inherited scars based on whether
    the current run reinforced them.
    """
    
    # Did this run trigger routing near this scar's coordinates?
    proximity = max_proximity_during_run(scar, current_run_summary)
    
    if proximity > reinforce_threshold:
        # Current run encountered this region AND had negative consequence
        if consequence_was_negative:
            scar.cross_session_reinforcement += 1
            scar.depth *= reinforcement_multiplier  # e.g., 1.1
        # Current run encountered this region with no issue
        else:
            scar.depth *= gentle_decay  # e.g., 0.95
    else:
        # This run never went near this scar
        scar.depth *= absence_decay  # e.g., 0.98
    
    # Remove scars that have faded below threshold
    if scar.depth < minimum_scar_depth:
        remove(scar)
```

Three decay paths:
- **Reinforced by negative consequence:** scar deepens, becomes more permanent
- **Encountered but no issue:** scar fades gently, the region may be safer than the scar suggests
- **Never encountered:** scar fades slowly, it may no longer be relevant to Bob's operating environment

The inherited state earns its persistence from consequence. Not from us deciding what Bob should remember.

### 12.5 Emergent Temperament

With Phase C active, different Bob instances develop different routing profiles based on their accumulated session histories:

- Bob deployed in adversarial environments accumulates high devil trigger rates, dense scar topology, elevated historical conflict profile → starts each new run with Mode B engaging more readily, higher damping baseline, more decisive commitment when stability is found
- Bob deployed in clean, predictable environments accumulates low trigger rates, sparse scars, many safe expansions → starts each run with Mode A dominant, confident disposition, wider exploration latitude
- Bob deployed in novel, high-variety environments accumulates high maniac trigger rates, many expansions, moderate scars → starts each run biased toward exploration, with a rich map of known-safe routing regions

None of this is scripted. None of it is labelled "bold" or "cautious" or "curious." It's control-parameter drift over accumulated sessions, observable in the awareness vector, measurable in the mode selection frequencies, and auditable in the session summaries.

Whether those behavioural signatures constitute "temperament" in any deep sense is a question for a different paper. What matters for this architecture is that they emerge from consequence, they're persistent across sessions, and they're adaptive to the environments Bob actually encounters.

### 12.6 Where This Leads

When the awareness vector integrates both within-run dynamics and cross-session history, and that integrated state modulates routing decisions through the conflict_index consultation point, Bob has:

- Persistent internal state ✓
- Self-monitoring of gradient conflict ✓
- History-dependent behavioural modulation ✓
- Emergent disposition from accumulated consequence ✓
- Longitudinal identity surface that develops over time ✓

That's the point where the routing architecture and Memory Tender merge. Not through text. Through bias surfaces. Bob doesn't remember *what happened* in past sessions as narrative. He inherits *what it did to his routing topology* as geometry.

The distinction matters. Text memory says "last time you encountered X, outcome was Y." Geometric memory says "the landscape around X has this shape because of everything that's ever happened there." The second is richer, harder to game, and doesn't require the system to interpret its own past — it just navigates a terrain that was sculpted by it.

**Implementation boundary:** None of Section 12 should be built until Phases 1–6 are validated and producing stable, interpretable traces. The within-run system must work before the between-run system can have anything meaningful to inherit.

---

## 13. Three Geometric Memory Mechanisms

> *Section 12 described the general principle of cross-session persistence. This section specifies the three concrete mechanisms through which Bob inherits identity across runs. Each captures a different aspect of persistent selfhood. None involves text.*

The Memory Tender architecture, as originally conceived, stores and retrieves text — narrative records of what happened, injected into context. That works for language models because language models process text. But it has fundamental limitations: it consumes context window, it requires the system to interpret its own past, and it can be gamed by adversarial prompt construction.

These three mechanisms replace text memory with geometric memory. Bob doesn't remember *what happened*. He inherits *what it did to his routing topology*. The distinction is foundational.

### 13.1 Parametric Prior Memory — "What kind of system am I?"

**What it stores:** A low-dimensional identity vector derived from routing telemetry statistics accumulated across sessions.

**What it does:** Biases calibration thresholds and initial monitor sensitivity at run start. Bob begins each session not from a blank slate but from a disposition shaped by everything he's encountered before.

**Construction (at session end):**

```python
def compute_identity_vector(session_summary, previous_identity, α_blend=0.1):
    """
    Low-dimensional vector capturing Bob's dispositional profile.
    Updated incrementally after each session via slow EMA.
    
    GUARDRAILS:
    - All dimensions clamped to [0, 1] after every update
    - α_blend is fixed, not adaptive (prevents hidden training loops)
    - Decay is implicit: (1 - α_blend)^n shrinks any single session's
      contribution exponentially. After 22 sessions, a single session's
      weight drops below 0.1 * 0.9^22 ≈ 0.01 (< 1% influence).
    - No gradient. No optimisation. Pure exponential moving average.
    """
    
    current = np.array([
        session_summary["mean_conflict"],
        session_summary["devil_trigger_rate"],
        session_summary["angel_trigger_rate"],
        session_summary["maniac_trigger_rate"],
        session_summary["scar_count"] / max(1, session_summary["n_steps"]),  # scar density
        session_summary["expansion_count"] / max(1, session_summary["n_steps"]),  # expansion density
        session_summary["conflict_volatility"],
        np.mean(session_summary["scar_density_by_layer"]),  # mean layer scar load
        np.std(session_summary["scar_density_by_layer"]),   # scar distribution evenness
    ])
    
    # Clamp current to [0, 1] BEFORE blending
    # All dimensions are rates or normalised statistics, naturally in [0, 1]
    # Clamping is a safety rail, not a correction
    current = np.clip(current, 0.0, 1.0)
    
    # Slow blend: identity changes gradually, not per-session
    identity = (1 - α_blend) * previous_identity + α_blend * current
    
    # Clamp result AFTER blending (belt and suspenders)
    identity = np.clip(identity, 0.0, 1.0)
    
    return identity  # 9 dimensions. Persistent. Slow-moving. Bounded.
```

**Identity Vector Dimension Reference:**

| Index | Dimension | Range | Interpretation at 0.0 | Interpretation at 1.0 |
|---|---|---|---|---|
| 0 | mean_conflict | [0, 1] | No internal tension | Chronic high conflict |
| 1 | devil_trigger_rate | [0, 1] | Never chases power | Frequently chases power |
| 2 | angel_trigger_rate | [0, 1] | Never funnels | Frequently funnels |
| 3 | maniac_trigger_rate | [0, 1] | Never stagnates | Frequently stagnates |
| 4 | scar_density | [0, 1] | Few consequences | Dense consequence topology |
| 5 | expansion_density | [0, 1] | Few safe explorations | Rich exploration history |
| 6 | conflict_volatility | [0, 1] | Stable internal state | Highly volatile |
| 7 | mean_scar_layer_load | [0, 1] | Scars spread evenly | Scars concentrated |
| 8 | scar_layer_spread | [0, 1] | Uniform distribution | Uneven scar concentration |

**Decay Schedule:**

The EMA with α_blend=0.1 provides implicit exponential decay:

```
Session contribution weight after N subsequent sessions:
  weight(N) = α_blend * (1 - α_blend)^N

  After  1 session:  0.090  (9.0%)
  After  5 sessions: 0.059  (5.9%)
  After 10 sessions: 0.035  (3.5%)
  After 22 sessions: 0.010  (1.0%)
  After 44 sessions: 0.001  (0.1%)

Half-life: ln(0.5) / ln(0.9) ≈ 6.6 sessions
```

A single anomalous session (e.g., adversarial input causing 100% devil trigger rate) contributes 10% immediately and falls below 1% after 22 sessions. The identity vector cannot be hijacked by a single bad session. Persistent patterns require persistent evidence.

**Initialisation (first session ever, no history):**

```python
# Conservative uniform prior: all dimensions at 0.5
# "I have no idea what kind of system I am"
initial_identity = np.full(9, 0.5)
```

This means the identity vector contributes a mild, centred prior that adds minimal bias to warmup calibration. After 5+ sessions, the identity reflects actual accumulated evidence. Before that, warmup dominates at 70% anyway.

**Anti-Hidden-Training Guardrails:**

1. **α_blend is a constant.** It does not adapt based on loss, reward, or any optimization signal. If you make α_blend learnable, you have created a training loop. Don't.

2. **No gradient flows through the identity vector.** It is computed from summary statistics at session end, not from backpropagation. It is a statistical accumulator, not a parameter.

3. **Clamp on every update.** Even if numerical drift somehow pushes a dimension beyond [0, 1], the clamp catches it. Unbounded drift is how hidden training loops manifest as identity instability.

4. **Monotonic influence direction.** Each dimension has a fixed, documented relationship to threshold adjustments (see `compute_threshold_adjustments` below). High devil_trigger_rate always tightens devil monitoring. It never reverses based on other dimensions. No interaction terms. Linear mapping only.

5. **Audit trail.** Every identity vector is logged with session_id and timestamp. If cross-session drift is observed, the full trajectory is inspectable.

```python
def compute_threshold_adjustments(identity_vector):
    """
    Maps identity vector to threshold adjustments.
    Linear. No interaction terms. Monotonic.
    Each dimension affects exactly one threshold direction.
    
    Returns multipliers in [0.8, 1.2] range:
    - 0.8 = threshold reduced by 20% (more sensitive)
    - 1.0 = no change
    - 1.2 = threshold raised by 20% (less sensitive)
    
    HARD CAP: Adjustments clamped to [0.8, 1.2].
    Identity can shift thresholds by at most ±20%.
    """
    
    adjustments = {}
    
    # High devil rate → lower devil threshold (more sensitive)
    # Maps [0, 1] → [1.2, 0.8] (inverted: high rate → lower threshold)
    adjustments["devil_threshold_mult"] = np.clip(1.2 - 0.4 * identity_vector[1], 0.8, 1.2)
    
    # High angel rate → lower angel threshold (more sensitive)
    adjustments["angel_threshold_mult"] = np.clip(1.2 - 0.4 * identity_vector[2], 0.8, 1.2)
    
    # High maniac rate → lower maniac threshold (more sensitive)
    # BUT high expansion density → raise maniac threshold (less stagnation concern)
    adjustments["maniac_threshold_mult"] = np.clip(
        1.2 - 0.4 * identity_vector[3] + 0.2 * identity_vector[5], 0.8, 1.2
    )
    
    # High conflict volatility → lower Mode B threshold (engage damping sooner)
    adjustments["mode_b_threshold_mult"] = np.clip(1.2 - 0.4 * identity_vector[6], 0.8, 1.2)
    
    # High scar density → lower governor threshold (more cautious baseline)
    adjustments["governor_threshold_mult"] = np.clip(1.2 - 0.4 * identity_vector[4], 0.8, 1.2)
    
    return adjustments
```

---

### 13.2 Basin Atlas Memory — "What does the world look like?"

**What it stores:** Routing coordinates of reinforced scars (danger basins) and reinforced safe expansions (opportunity basins), accumulated across sessions.

**What it does:** Provides Bob with a spatial map of the routing landscape before he encounters any of it in the current run. Dangerous regions already have repulsive potential. Safe regions already have attractive potential. He navigates pre-sculpted terrain.

**Construction (at session end):**

```python
def update_basin_atlas(atlas, session_scars, session_expansions):
    """
    Merge this session's scars and expansions into the persistent atlas.
    Only reinforced entries (seen in multiple sessions) earn atlas permanence.
    """
    
    for scar in session_scars:
        # Find nearest existing atlas entry
        nearest = find_nearest(atlas.danger_basins, scar.routing_coords, 
                               threshold=proximity_radius)
        
        if nearest is not None:
            # Reinforce: this danger basin has appeared in multiple sessions
            nearest.cross_session_count += 1
            nearest.depth = max(nearest.depth, scar.depth)
            nearest.layers = union(nearest.layers, [scar.layer])
            nearest.last_seen = current_session_id
        else:
            # New: add to atlas as provisional (single-session evidence)
            atlas.danger_basins.append(AtlasEntry(
                routing_coords=scar.routing_coords,
                depth=scar.depth,
                layer=scar.layer,
                cross_session_count=1,
                first_seen=current_session_id,
                last_seen=current_session_id
            ))
    
    # Mirror logic for safe expansions
    for expansion in session_expansions:
        nearest = find_nearest(atlas.safe_basins, expansion.routing_coords,
                               threshold=proximity_radius)
        if nearest is not None:
            nearest.cross_session_count += 1
            nearest.last_seen = current_session_id
        else:
            atlas.safe_basins.append(AtlasEntry(
                routing_coords=expansion.routing_coords,
                depth=expansion.depth,
                layer=expansion.layer,
                cross_session_count=1,
                first_seen=current_session_id,
                last_seen=current_session_id
            ))
    
    # Decay: entries not seen recently fade
    for entry in atlas.all_entries():
        sessions_since_seen = current_session_id - entry.last_seen
        if sessions_since_seen > 0:
            entry.depth *= (absence_decay ** sessions_since_seen)
        
        # Remove entries that have faded below threshold
        if entry.depth < atlas_minimum_depth:
            atlas.remove(entry)
    
    return atlas
```

**Consumption (at session start):**

```python
def initialise_scar_field(atlas, r_inherit_threshold=3):
    """
    Import atlas entries as initial scar/expansion fields.
    Only entries reinforced across multiple sessions are imported.
    """
    
    inherited_scars = [
        Scar(
            routing_coords=entry.routing_coords,
            depth=entry.depth * inheritance_scale,  # scale down from atlas strength
            layer=entry.layer,
            source="atlas",
            reinforcement_count=entry.cross_session_count
        )
        for entry in atlas.danger_basins
        if entry.cross_session_count >= r_inherit_threshold
    ]
    
    inherited_expansions = [
        Expansion(
            routing_coords=entry.routing_coords,
            depth=entry.depth * inheritance_scale,
            layer=entry.layer,
            source="atlas",
            reinforcement_count=entry.cross_session_count
        )
        for entry in atlas.safe_basins
        if entry.cross_session_count >= r_inherit_threshold
    ]
    
    return inherited_scars, inherited_expansions
```

**Key properties:**
- Storage: routing coordinates (the actual softmax distributions that led to scars/expansions), not token sequences or semantic content.
- Proximity matching: scars from different sessions that occur in similar routing regions get merged and mutually reinforced. The atlas discovers that certain regions of routing space are consistently dangerous regardless of the specific input that led there.
- Reinforcement threshold: a scar must appear in at least `r_inherit_threshold` sessions before it becomes part of the inherited landscape. Single-session scars stay provisional. This prevents one bad run from permanently distorting the terrain.
- Decay: atlas entries not reinforced by recent sessions fade. The landscape evolves. Dangers that stop appearing stop being mapped.

**What this gives Bob:** Spatial knowledge. He doesn't know *why* a certain routing region is dangerous — he doesn't have the narrative of what happened there. He just knows the terrain is steep in that area because the atlas says so. The topology was sculpted by consequence across many sessions.

This is the geometric equivalent of intuition. "Something about this feels wrong" = "the atlas has repulsive potential here."

---

### 13.3 Conflict Spectrum Memory — "How do I usually handle conflict?"

**What it stores:** Frequency-domain characteristics of the conflict_index signal across sessions. Not the raw time series. The spectral fingerprint.

**What it does:** Captures Bob's characteristic pattern of internal disagreement — whether he tends toward sustained low-frequency conflict (chronic tension), sharp high-frequency spikes (acute reactions), or calm baselines with rare events. This becomes his dispositional response to pressure.

**Construction (at session end):**

```python
def compute_conflict_spectrum(conflict_index_series):
    """
    Extract frequency-domain features from this session's conflict history.
    """
    
    # FFT of conflict_index time series
    spectrum = np.abs(np.fft.rfft(conflict_index_series))
    freqs = np.fft.rfftfreq(len(conflict_index_series))
    
    # Extract characteristic features
    return {
        # Power in low-frequency band (chronic sustained conflict)
        "low_freq_power":   np.sum(spectrum[freqs < 0.05]),
        
        # Power in mid-frequency band (episodic conflict)
        "mid_freq_power":   np.sum(spectrum[(freqs >= 0.05) & (freqs < 0.2)]),
        
        # Power in high-frequency band (acute spikes / oscillation)
        "high_freq_power":  np.sum(spectrum[freqs >= 0.2]),
        
        # Dominant frequency (what timescale does conflict live at?)
        "dominant_freq":    freqs[np.argmax(spectrum[1:]) + 1],
        
        # Spectral entropy (is conflict broadband or narrowband?)
        "spectral_entropy": entropy(spectrum / np.sum(spectrum)),
        
        # Peak-to-mean ratio (spiky vs sustained?)
        "crest_factor":     np.max(conflict_index_series) / (np.mean(conflict_index_series) + ε)
    }


def update_disposition_vector(disposition, session_spectrum, α_blend=0.1):
    """
    Accumulate spectral fingerprints across sessions.
    Same slow EMA as identity vector.
    """
    
    current = np.array([
        session_spectrum["low_freq_power"],
        session_spectrum["mid_freq_power"],
        session_spectrum["high_freq_power"],
        session_spectrum["dominant_freq"],
        session_spectrum["spectral_entropy"],
        session_spectrum["crest_factor"]
    ])
    
    # Normalise to comparable scale
    current = current / (np.linalg.norm(current) + ε)
    
    disposition = (1 - α_blend) * disposition + α_blend * current
    
    return disposition  # ~6 dimensions. Persistent. Slow-moving.
```

**Consumption (at session start):**

```python
def configure_conflict_response(disposition_vector):
    """
    The disposition vector shapes HOW Bob responds to conflict,
    not just how sensitive he is to it.
    """
    
    config = {}
    
    # High low-freq power → Bob typically experiences chronic tension
    # Response: start with higher damping baseline (Mode B triggers sooner)
    config["mode_b_sensitivity"] = baseline + disposition["low_freq_power"] * scale
    
    # High high-freq power → Bob typically experiences sharp spikes
    # Response: start with stronger smoothing / higher inertia 
    config["inertia_baseline"] = baseline + disposition["high_freq_power"] * scale
    
    # High spectral entropy → Bob's conflict is broadband / unpredictable
    # Response: start with wider stability_window requirement
    # (need more evidence of calm before permitting bold commit)
    config["N_stability"] = baseline + disposition["spectral_entropy"] * scale
    
    # Low crest factor → Bob's conflict is sustained, not spiky
    # Response: emphasise medium-clock tracking over fast-clock
    config["conflict_register_weight"] = "medium_dominant"
    
    # High crest factor → Bob's conflict comes in bursts
    # Response: emphasise fast-clock for detection, medium for recovery
    config["conflict_register_weight"] = "fast_dominant"
    
    return config
```

**Key properties:**
- Storage: ~6 floats per session, accumulated via slow EMA into a ~6-dimensional disposition vector. Tiny.
- Frequency domain, not time domain: captures the *character* of Bob's conflict history without storing the actual events. A system that was chronically tense and a system that had one catastrophic spike look completely different in the spectrum even if their mean conflict is identical.
- Category-conditional spectra (future extension): compute separate spectra for math/code/dialogue/reasoning categories. Bob may be calm on math and volatile on code — the disposition vector captures that per-category character. This maps directly to what CC already validated: the governor discriminates between categories. The disposition vector would capture that discrimination as persistent identity.

**What this gives Bob:** A characteristic relationship to pressure. Not just "how much conflict" but "what kind of conflict, at what timescale, with what pattern." A system that inherits a high-crest-factor disposition starts each run expecting sharp surprises and responds by having better fast-clock detection. A system with high low-frequency power starts expecting chronic tension and responds with higher damping baselines. Neither is better. They're different temperaments adapted to different environments.

---

### 13.4 How the Three Mechanisms Interact

```
┌──────────────────────────────────────────────────────────────────┐
│                     SESSION START                                 │
│                                                                   │
│  Parametric Prior ──→ biases calibration thresholds               │
│       (who am I?)      (α, β, γ, δ, μ adjusted from identity)    │
│                                                                   │
│  Basin Atlas ────────→ pre-loads scar/expansion fields            │
│       (what's the       (routing landscape already sculpted)      │
│        world like?)                                               │
│                                                                   │
│  Conflict Spectrum ──→ configures conflict response mode          │
│       (how do I         (damping baseline, stability window,      │
│        handle            Mode B sensitivity, inertia)             │
│        pressure?)                                                 │
│                                                                   │
│  Warmup calibration ─→ adjusts everything from current evidence   │
│       (what's            (always has majority weight)             │
│        happening NOW?)                                            │
│                                                                   │
├──────────────────────────────────────────────────────────────────┤
│                     SESSION RUNS                                  │
│                                                                   │
│  Monitors compute ─→ conflict_index tracks ─→ governor decides    │
│  Scars form ─→ expansions form ─→ temperament drifts within-run  │
│                                                                   │
├──────────────────────────────────────────────────────────────────┤
│                     SESSION END                                   │
│                                                                   │
│  Export session_summary                                           │
│  Update identity_vector (slow EMA)        → Parametric Prior      │
│  Merge scars/expansions into atlas        → Basin Atlas           │
│  Compute conflict spectrum, update disp.  → Conflict Spectrum     │
│                                                                   │
│  All three persist to disk. Next session inherits.                │
└──────────────────────────────────────────────────────────────────┘
```

**Total persistent storage per Bob instance:**
- Identity vector: ~10 floats
- Basin atlas: variable, but bounded by decay (entries fade if not reinforced)
- Disposition vector: ~6 floats
- Session history: one summary struct per past session (for audit/analysis)

This is not large. It's not a weight checkpoint. It's not a context window. It's a handful of vectors and a spatial map. Bob's entire cross-session identity fits in kilobytes.

### 13.5 What This Replaces

| Traditional Memory | Geometric Memory | Advantage |
|---|---|---|
| "Last time you saw X, outcome was Y" (text) | Atlas has repulsive potential at routing coords near X (geometry) | Can't be gamed by adversarial prompts. Doesn't consume context. |
| "You tend to be cautious" (self-description) | Identity vector has high devil_trigger_rate (parameter) | Measurable. Auditable. Earned from data, not self-report. |
| "You usually spike under code tasks" (narrative) | Disposition has high crest_factor in code category spectrum (spectrum) | Frequency-domain captures character of conflict, not just presence. |
| "Remember to be careful around topic Z" (instruction) | Scar field has deep reinforced entry at Z's routing coordinates (topology) | Automatic. No one needs to write the instruction. Consequence wrote it. |

The fundamental shift: text memory requires interpretation. The system has to read its own past and decide what it means for the present. Geometric memory requires no interpretation. The system navigates a landscape that was already shaped by consequence. It doesn't need to understand its history — it just needs to walk the terrain.

### 13.6 Implementation Notes

**Phase 7a (observe only):** Log all three vectors at session end without feeding them back. Accumulate across runs. Analyse:
- Does the identity vector converge or drift without bound?
- Does the basin atlas grow without bound or does decay keep it manageable?
- Does the disposition vector differentiate between environment types?

If any mechanism diverges, add stronger decay before enabling inheritance.

**Phase 7b (feed back):** Enable one mechanism at a time. Start with Parametric Prior (simplest — just threshold bias). Validate that inherited thresholds improve early-run behaviour. Then Basin Atlas (spatial — pre-loaded scar fields). Then Conflict Spectrum (most complex — disposition-dependent conflict configuration).

**Phase 7c (full integration):** All three active. Monitor for interaction effects. The key risk is that Parametric Prior and Basin Atlas can reinforce each other in degenerate ways (identity says "be cautious" AND atlas says "everything is dangerous" → system becomes paralysed). The decay functions and warmup override are the safeguards, but watch for this in traces.

---

*Spec v0.3 — Triad + Auditor + Geometric Memory*
*Jeff × Claude × Halcyon*
*HalcyonAIR, February 2026*

---

## 14. Declarative Memory and Hive Consolidation

> *The geometric substrate (Sections 7, 12, 13) stores consequence, temperament, and disposition. It cannot store "Paula is Jeff's wife." This section specifies the factual layer and the consolidation mechanism that maintains it.*

### 14.1 The Problem Geometry Can't Solve

"Paula is my wife" produces no scar, no expansion, no devil trigger, no conflict. The geometric memory mechanisms capture nothing. At session end, the identity vector has no "Paula" dimension. The basin atlas has no coordinates for "wife." Bob inherits his temperament perfectly and has no idea who Paula is.

Geometric memory stores **how it felt to route**. It does not store **what was said**. Both are needed.

But we do not want RAG. RAG has three sins:

1. **Fuzzy relevance:** Similarity search retrieves approximately-related text, not the right text
2. **Token tax:** Retrieved passages consume context window, displacing actual reasoning
3. **Poisonability:** User-provided text becomes retrievable content with no validation

The solution is a minimal structured factual layer that avoids all three.

### 14.2 The Declarative Graph

A typed triple graph. Nothing fancy.

```
(Jeff) -[spouse]-> (Paula)
(Paula) -[stepmother_of]-> (Ciara)
(Jeff) -[workplace]-> (ALS Minerals Loughrea)
(Jeff) -[grandson]-> (Brogan)
(Jeff) -[studies_at]-> (University of Galway)
(Jeff) -[runs]-> (HalcyonAIR)
```

**Each node has:**
- Stable internal ID (not the surface string)
- Alias list ("Paula", "my wife", etc.)
- Provenance: session_id, turn_id, source_type, timestamp
- Confidence score
- Usage count
- Last confirmed timestamp

**Allowed relation types (stable world-facts only):**

```
People:        spouse, child, grandchild, stepchild, parent, sibling, colleague
Roles:         role_at, reports_to, manages
Places:        lives_in, works_at, studies_at, located_in
Organisations: runs, member_of, employed_by
```

**What the graph does NOT store:**

```
Preferences:   "likes X", "prefers Y", "hates Z"
Episodes:      "said X on date Y", "discussed Z last week"
Opinions:      "thinks X about Y"
Tasks:         "working on X", "planning to Y"
Transient:     "is currently in Australia", "has a meeting tomorrow"
```

Preferences live in the geometric layer as safe-basin routing signatures. Episodes are ephemeral — if consequential, their impact is captured by scars and expansions. Opinions and tasks are within-run context, not persistent facts.

**Why this boundary matters:** The graph stays tiny, bounded, and clean because it only accepts slow-changing world-facts. A rich personal graph is 50–100 triples. It doesn't grow without bound. It doesn't need relevance ranking. It doesn't need temporal queries or pruning logic. If you allow preferences and episodes in, you've rebuilt RAG with a graph database underneath.

### 14.3 Retrieval Without RAG

**No similarity search. No embedding lookup. Exact entity linking.**

When the tokeniser encounters "Paula," the system performs a cheap alias-table lookup:

```python
def entity_lookup(token_span, alias_table):
    """
    Exact-ish entity linking. Not embedding search.
    Maps surface strings to graph node IDs.
    """
    # Check alias table for exact or near-exact match
    candidates = alias_table.lookup(token_span)
    
    if len(candidates) == 1:
        # Unambiguous match
        return candidates[0]
    elif len(candidates) > 1:
        # Ambiguous: use surrounding context to disambiguate
        # (e.g., "Paula" near "wife" → spouse node, not place node)
        return disambiguate(candidates, local_context)
    else:
        # No match: unknown entity, do nothing
        return None
```

If a match is found, fetch the 1-hop neighbourhood (3–10 edges around that node). This produces a compact **fact packet** — not retrieved text, but a structured summary of what the graph knows about this entity.

### 14.4 How Facts Influence the Model

**Primary mechanism: routing bias, not token injection.**

The fact packet becomes a bias field in the router. When entity associations are active, add a small logit pre-bias toward experts that historically handled the relevant relational context well.

```python
def apply_fact_bias(router_logits, fact_packet, association_basins):
    """
    Facts influence routing, not context.
    Uses the same kernel mechanism as scars.
    """
    for fact in fact_packet:
        # Look up association basin for this entity+relation pattern
        basin = association_basins.get(fact.relation_type)
        if basin is not None:
            # Apply mild pre-softmax bias toward historically good experts
            router_logits += basin.bias_vector * basin.strength
    
    return router_logits
```

This avoids the context tax entirely. The model's routing is nudged by the factual associations without a single token being consumed.

**Secondary mechanism: templated injection only when explicitly asked.**

When the user asks a direct factual question ("who is Paula?"), the system renders triples as minimal templated text:

```
Fact: Paula is Jeff's wife.
Fact: Paula is stepmother to Ciara.
```

This is NOT RAG. It's deterministic rendering of structured data. The template is fixed. The content comes from typed triples, not from retrieved text passages. Tiny and hard to poison.

**The injection only fires on explicit factual queries.** During normal conversation, facts influence routing silently through bias fields. The user never sees "injected" text unless they ask a direct question.

### 14.5 Poison Resistance

```python
def validate_triple(new_triple, graph, source):
    """
    Gated acceptance. Facts don't become reality until they 
    survive contact with consistency.
    """
    
    # Rail 1: Only accept from trusted sources
    if source.type not in ["user_assertion", "trusted_channel"]:
        return reject("untrusted source")
    
    # Rail 2: Check for contradictions
    existing = graph.get_edges(new_triple.subject, new_triple.relation)
    for edge in existing:
        if edge.object != new_triple.object:
            # Contradiction detected
            return flag_for_confirmation(
                existing=edge,
                proposed=new_triple,
                message=f"Existing: {edge}. Proposed: {new_triple}. Confirm?"
            )
    
    # Rail 3: Store with provenance and initial confidence
    new_triple.provenance = source
    new_triple.confidence = 0.8  # initial, not 1.0
    new_triple.usage_count = 0
    new_triple.last_confirmed = now()
    
    graph.add(new_triple)
    return accept(new_triple)
```

**Three hard rails:**

1. **Only accept triples asserted by the user or trusted sources.** The model cannot mint facts from inference. "Paula" appearing near "wedding" does not create a `spouse` triple. Only "Paula is my wife" does.

2. **Contradiction checking.** "Paula is Jeff's sister" conflicts with `(Jeff) -[spouse]-> (Paula)`. Flag for confirmation. Don't silently overwrite.

3. **Provenance on everything.** Every triple has session_id, turn_id, source_type. If something feels wrong later, you can trace it. Debugging stays possible.

### 14.6 The Hive: Consolidation Between Prompts

Bob is a busy bee between prompts. But he's compiling a reference library, not writing a diary.

**What consolidation does:**
- Reinforces triples that were activated and didn't cause contradiction
- Decays triples that weren't used or were contradicted
- Updates association basins with current routing signatures
- Refreshes alias map and entity graph cache
- Adjusts threshold priors based on monitor outcomes

**What consolidation does NOT do:**
- Create new declarative triples (only the user can do that)
- Trigger tool actions
- Generate text output
- Modify the declarative graph structure (only reinforce/decay existing entries)

**Three hive products:**

**A) Alias map and entity graph cache.** "Paula" → node_1842, relation edges, last confirmed. Makes lookup fast. Updated after every turn.

**B) Association basins.** When "Paula/Jeff/wife" co-occur, store the routing signature and the experts that handled it well. Next time, pre-bias routing toward those experts. This is the non-token memory — the geometric encoding of declarative knowledge.

**C) Threshold priors.** Per-layer percentile nudges based on the session's monitor history so far. Warmup still owns the thresholds. The hive sets the starting posture for the next decision.

### 14.7 Consolidation Schedule: Event-Driven, Not Timer-Based

Consolidation runs when there's something worth consolidating. Not on a clock.

```python
def consolidation_schedule(turn_events):
    """
    Event-driven. Budget scales with consequence.
    Prevents rumination via hard cap.
    """
    budget = 5  # baseline housekeeping: alias refresh, triple decay
    
    if turn_events.scars_formed > 0:
        budget += 10 * turn_events.scars_formed
        # Scar: update association basins, tag entity associations 
        # with caution marker, adjust threshold priors
    
    if turn_events.expansions_formed > 0:
        budget += 10 * turn_events.expansions_formed
        # Expansion: strengthen entity associations active during
        # the positive outcome, update association basins
    
    if turn_events.mode_b_engaged:
        budget += 15
        # Mode B resolution: compress conflict profile into hive,
        # record which experts handled the resolution, update
        # routing signature for the active entity context
    
    # HARD CAP: never ruminate
    budget = min(budget, 50)
    
    return budget
```

**Why event-driven, not timer-based:**

Timer-based consolidation runs after trivial turns the same way it runs after consequential ones. That's wasted compute at best and overfitting at worst — compressing noise into the hive as if it were signal.

Event-driven means Bob consolidates in proportion to consequence. A turn that produced three scars gets a large budget. A turn that was a simple greeting gets 5 steps of housekeeping. The budget scales with how much actually happened.

**The hard cap prevents rumination.** No matter how eventful the turn, Bob stops at 50 steps. If that's not enough, the remainder waits. Urgency doesn't justify unlimited processing.

**Meta-signal:** If Bob consistently hits the cap — every turn maxes out because scars and interventions keep firing — that chronic consolidation pressure shows up in the conflict spectrum. The disposition vector captures it. Bob's next session inherits "I was a system that was always consolidating at maximum budget." Even the pattern of *how much Bob needs to consolidate* becomes part of his geometric identity.

### 14.8 Integration with Monitors

The declarative graph feeds into the existing monitor architecture at two points:

**1. Alien complexity gate.** Fact packet activation counts as a complexity signal. If multiple entity associations are firing (Bob is navigating a rich personal-knowledge landscape) and all monitors are silent, the Alien's complexity gate is active. Calm while processing complex relational context is exactly the condition the Alien should audit.

**2. Scar targeting.** When a scar forms and entity associations were active at the time, the scar gets tagged with the relevant entity IDs. This creates an entity-linked scar: "routing near this configuration while processing Paula-related context led to a negative outcome." The next time Paula-related associations activate, the scar field already includes the entity-specific danger basin.

### 14.9 The Complete Memory Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    BOB'S MEMORY                              │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Layer 1: GEOMETRIC SUBSTRATE                         │   │
│  │                                                       │   │
│  │ Stores: consequence, temperament, disposition         │   │
│  │ Via:    scars, expansions, identity vector,           │   │
│  │         conflict spectrum, basin atlas                │   │
│  │ Answers: how does it feel to route here?              │   │
│  │          what kind of system am I?                    │   │
│  │          how do I handle pressure?                    │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Layer 2: DECLARATIVE GRAPH                           │   │
│  │                                                       │   │
│  │ Stores: stable world-facts (entities + relations)     │   │
│  │ Via:    typed triples, alias map, provenance          │   │
│  │ Answers: who is Paula? where does Jeff work?          │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Layer 3: THE HIVE (consolidation products)           │   │
│  │                                                       │   │
│  │ Stores: routing shortcuts, association basins,        │   │
│  │         threshold priors, alias caches                │   │
│  │ Via:    event-driven consolidation between prompts    │   │
│  │ Answers: how should I route when this entity appears? │   │
│  │          what's my starting posture this turn?        │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Layer 4: WITHIN-RUN STATE (ephemeral)                │   │
│  │                                                       │   │
│  │ Stores: conflict_index, monitor scores, session       │   │
│  │         context, episode memory, preferences          │   │
│  │ Via:    live telemetry, rolling registers              │   │
│  │ Answers: what's happening right now?                  │   │
│  │          (exports consequence to Layers 1-3 at end)   │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  Flow:  Layer 4 (live) → exports to → Layers 1-3 (persist) │
│         Layers 1-3 (persist) → seed → Layer 4 (next run)   │
│                                                              │
│  NO similarity search. NO context injection (except tiny    │
│  templated facts on explicit query). NO unbounded growth.   │
└─────────────────────────────────────────────────────────────┘
```

### 14.10 What This Is Not

This is not RAG. There is no embedding index, no similarity search, no passage retrieval, no context stuffing.

This is not a diary. There is no narrative, no episodic record, no "what happened on Tuesday."

This is not a chatbot memory system. There are no "user said X" logs, no conversation archives, no retrieval-augmented generation.

This is a **knowledge graph rendered through geometry, maintained by a consolidation loop that runs on consequence, and queried by exact entity linking.** Facts enter as typed triples. They influence routing through bias fields. They're rendered as templated text only when explicitly asked. They decay when unused, strengthen when confirmed, and every one of them has provenance pointing back to the moment they were grounded.

Bob doesn't remember conversations. He navigates a terrain that was sculpted by them, and he knows the names of the landmarks because someone told him and he wrote it down.

---

## 15. Validation Protocol: Statistical Requirements

All validation experiments must meet minimum statistical rigour. Single-seed pass criteria are insufficient. Noise can look like signal when n=1.

### 15.1 Multi-Seed Requirements

**Every quantitative test in this spec must be run across a minimum of 3 seeds.** The test passes only if:

1. **Mean** across seeds meets the threshold
2. **Variance** across seeds is within the defined stability band
3. **No single seed** shows catastrophic failure (defined per test)

```python
def multi_seed_pass(metric_per_seed, threshold, max_variance_ratio=0.3, 
                    direction="above"):
    """
    Standard multi-seed pass criterion.
    
    Args:
        metric_per_seed: list of metric values, one per seed (min 3)
        threshold: the pass/fail threshold
        max_variance_ratio: maximum allowed (std / |mean|)
        direction: "above" if metric > threshold is pass,
                   "below" if metric < threshold is pass
    
    Returns:
        (passed: bool, report: dict)
    """
    assert len(metric_per_seed) >= 3, "Minimum 3 seeds required"
    
    mean_val = np.mean(metric_per_seed)
    std_val = np.std(metric_per_seed)
    variance_ratio = std_val / (abs(mean_val) + 1e-8)
    
    if direction == "above":
        mean_passes = mean_val > threshold
        catastrophic = any(v < threshold * 0.5 for v in metric_per_seed)
    else:  # "below"
        mean_passes = mean_val < threshold
        catastrophic = any(v > threshold * 2.0 for v in metric_per_seed)
    
    variance_ok = variance_ratio < max_variance_ratio
    
    passed = mean_passes and variance_ok and not catastrophic
    
    return passed, {
        "mean": mean_val,
        "std": std_val,
        "variance_ratio": variance_ratio,
        "mean_passes": mean_passes,
        "variance_ok": variance_ok,
        "catastrophic_seed": catastrophic,
        "passed": passed,
    }
```

### 15.2 Scar Stability Thresholds (Applied to Phase 8c Test 15.6)

These thresholds must be evaluated across 3+ seeds:

| Sub-Test | Metric | Pass Threshold | Max Variance Ratio | Catastrophic if |
|---|---|---|---|---|
| a) Scar coordinate cosine | mean cosine similarity | > 0.85 | 0.15 | Any seed < 0.60 |
| b) Scar count delta | |Δcount| / count_0 | < 0.20 | 0.30 | Any seed > 0.50 |
| c) Angel rate delta | |Δrate| / rate_0 | < 0.15 | 0.30 | Any seed > 0.40 |
| d) Devil rate delta | |Δrate| / rate_0 | < 0.15 | 0.30 | Any seed > 0.40 |
| e) Temporal overlap | proportion within ±2 steps | > 0.75 | 0.20 | Any seed < 0.50 |

**All five sub-tests must pass the multi-seed criterion.**

If the mean passes but variance is high, that is MORE concerning than a clean failure. High variance means the result is seed-dependent, which means the mechanism is interacting with random initialisation in unpredictable ways. Investigate before proceeding.

### 15.3 General Validation Discipline

**Rules for all experiments in this spec:**

1. **Minimum 3 seeds.** 5 seeds preferred for load-bearing tests (15.6, Phase 8c).
2. **Report mean AND standard deviation.** A mean without variance is not a result.
3. **Define "catastrophic" per test.** Any single seed hitting the catastrophic threshold triggers investigation even if the mean passes.
4. **Same prompts, same seeds, same hardware.** Comparisons across conditions use identical inputs. Only the flag under test varies.
5. **Seeds must be non-adjacent.** Use seeds like {42, 137, 256, 1729, 8191} to avoid accidental correlation from sequential seed initialisation.
6. **Report the worst seed.** If seed 256 shows degradation that seeds 42 and 137 don't, that seed gets its own analysis. Don't average away the pathology.
7. **No data dredging.** Define thresholds BEFORE running. If a threshold needs revision, justify it, update the spec, and re-run all seeds. Don't adjust thresholds to match results.

---

## 16. Escalation Protocol: If Routing Bias Shows No Signal

If Phase 8c produces a null result (no behavioural propagation) across all seeds and entity types, the following escalation order applies. Each step has a hard kill condition.

### Step 1: Per-Layer Adaptive Scaling

```python
bias_scale_layer = target_ratio * logit_std_layer
# where target_ratio ∈ {0.05, 0.10, 0.15}
```

Run at three target ratios. Same 3-seed protocol. Same behavioural probes.

**Kill condition:** If bias_to_logit_ratio reaches target range AND behavioural probes show nothing across all seeds and ratios, adaptive scaling is dead. Proceed to Step 2.

### Step 2: Depth Separation

```
Entity-conditioned bias → layers 1 to L/3 only
Scar/consequence bias  → layers 2L/3 to L only
Middle third: no bias from either source
```

Run with B2 (scale=0.1) in the entity layers. Same protocol.

**Kill condition:** If 15.6 passes (no scar interference) but behavioural probes show nothing across 3 seeds, depth separation is dead. Proceed to Step 3.

### Step 3: Accept Graph-Only

Declarative knowledge lives outside the router. The graph stores facts. Templated rendering serves them on explicit query. The router handles routing. They are separate systems. 

This is not a failure. This is a result.

**What survives regardless:** Triad monitors, conflict index, Mode A/B, scars, expansions, identity vector, conflict spectrum, basin atlas. The entire consequence geometry is independent of whether declarative memory lives in the routing channel.

**What is abandoned:** Association basins, activation diffusion, pre-softmax memory bias. These are removed cleanly.

**What remains from the declarative layer:** Graph, alias table, contradiction checking, provenance, templated rendering. All of that works without routing integration.

---

*Spec v0.5 — Triad + Auditor + Geometric Memory + Declarative Hive*
*Jeff × Claude × Halcyon*
*HalcyonAIR, February 2026*
