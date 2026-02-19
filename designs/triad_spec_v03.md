# ChronoMoE Spec v0.3: Triad + Auditor

## Routing Monitors — Implementation Spec for CC

**Authors:** Jeff, Claude, Halcyon
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

**Complexity gate (require at least one):**
- High tool availability or tool-use opportunity in current context
- High category entropy across the prompt mix (mixing dialogue/math/code/reasoning)
- *Optional:* High novelty in prompt embedding space (input unlike recent prompts)

**Note on the embedding option:** Using a lightweight input embedding for the novelty check is acceptable as a gating signal, but be explicit that it is purely for gating and logging — NOT for memory injection or semantic routing. If you want to avoid embeddings entirely, use internal signals: per-layer activation variance or attention entropy dispersion as a complexity proxy. The embedding approach is cheaper and better calibrated; the internal-signal approach is purer but requires more tuning. Recommend starting with tool availability + category entropy (no embeddings needed), and adding the embedding gate later if those prove insufficient.

If the complexity gate is not active, the alien does not accumulate calm debt regardless of how quiet the other monitors are.

**Score (only when complexity gate is active):**

```
calm_sum[t,ℓ] = a[t,ℓ] + d[t,ℓ] + m[t,ℓ]

alien[t,ℓ] = max(0, η_alien - calm_sum[t,ℓ]) * n_calm_consecutive
```

Where η_alien is calibrated similarly to the other thresholds (e.g., percentile-based baseline for "normal" total monitor activity during complex tasks), and n_calm_consecutive tracks how many consecutive steps have had low total monitor activity while complexity is high.

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
- [ ] **7a: Session Export** — compute session_summary vector at run end, store to disk
- [ ] Accumulate session summaries across runs without feeding back
- [ ] Analyse cross-session variance (do summaries cluster by environment type?)
- [ ] **7b: Session Import** — compute prior from accumulated session history
- [ ] Implement initialise_run() with prior-weighted warmup calibration
- [ ] Inherited scar/expansion topology filtered by cross-session reinforcement count
- [ ] **7c: Awareness Vector** — integrate within-run + cross-session state
- [ ] Cross-session scar decay (three paths: reinforced, encountered, absent)
- [ ] Log awareness vector per step alongside existing telemetry
- [ ] Implement behind flag: `--enable-cross-session-persistence`

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

> *This section is architectural specification, not implementation-ready. It defines where the bounded per-run system is headed so that CC understands the trajectory. Do not implement until Phases 1-6 are validated.*

### 12.1 The Fork

Everything in Sections 1-10 is bounded per-run. Bob starts fresh, accumulates state, develops within-run temperament, and resets at session boundary. That's the control system. It must be validated first.

Cross-session persistence is where Bob stops being a per-run controller and starts being a longitudinal identity. The conflict_index, the scar topology, the monitor trigger rates — all of that carries forward and shapes who Bob *is* before the first token of the next run arrives.

This is where the routing architecture and Memory Tender converge. Not through text injection into context, but through inherited bias surfaces shaped by accumulated routing history.

**These two layers must remain architecturally separated.** If within-run dynamics and inherited state are conflated, debugging becomes impossible — you won't know whether strange behaviour comes from current-run conflict detection or from inherited state pollution.

### 12.2 Phase A → B → C Implementation Path

**Phase A (current spec): Bounded per-run.**

Everything in Sections 1-10. All state resets at session boundary. Prove that within a single run, Bob develops measurable temperament-like drift from his own gradient conflict history. Prove the drift is adaptive — cautious in adversarial environments, confident in clean ones.

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

### 12.3 The Three Memory Modalities

#### Parametric Prior Memory
At session end, export a low-dimensional "identity vector" derived from routing telemetry only. Feed that vector into calibration at the start of the next run. Not as tokens. As threshold bias and initial scar field. That's persistent temperament without text recall.

#### Basin Atlas Memory
Instead of storing text, store routing coordinates that led to reinforced scars or safe expansions. Over time you build a map of "danger basins" and "safe basins." The next run starts with those already embedded. That's spatial memory, not narrative memory.

#### Conflict Spectrum Memory
Store frequency spectra of conflict_index over sessions. Is Bob usually calm? Usually torn? Does he spike under certain categories? That becomes his disposition vector. The next run inherits that statistical fingerprint. Memory without tokens.

### 12.4 The Awareness Vector

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

### 12.5 Inherited State Decay

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

### 12.6 Emergent Temperament

With Phase C active, different Bob instances develop different routing profiles based on their accumulated session histories:

- Bob deployed in adversarial environments accumulates high devil trigger rates, dense scar topology, elevated historical conflict profile → starts each new run with Mode B engaging more readily, higher damping baseline, more decisive commitment when stability is found
- Bob deployed in clean, predictable environments accumulates low trigger rates, sparse scars, many safe expansions → starts each run with Mode A dominant, confident disposition, wider exploration latitude
- Bob deployed in novel, high-variety environments accumulates high maniac trigger rates, many expansions, moderate scars → starts each run biased toward exploration, with a rich map of known-safe routing regions

None of this is scripted. None of it is labelled "bold" or "cautious" or "curious." It's control-parameter drift over accumulated sessions, observable in the awareness vector, measurable in the mode selection frequencies, and auditable in the session summaries.

Whether those behavioural signatures constitute "temperament" in any deep sense is a question for a different paper. What matters for this architecture is that they emerge from consequence, they're persistent across sessions, and they're adaptive to the environments Bob actually encounters.

### 12.7 Where This Leads

When the awareness vector integrates both within-run dynamics and cross-session history, and that integrated state modulates routing decisions through the conflict_index consultation point, Bob has:

- Persistent internal state
- Self-monitoring of gradient conflict
- History-dependent behavioural modulation
- Emergent disposition from accumulated consequence
- Longitudinal identity surface that develops over time

That's the point where the routing architecture and Memory Tender merge. Not through text. Through bias surfaces. Bob doesn't remember *what happened* in past sessions as narrative. He inherits *what it did to his routing topology* as geometry.

The distinction matters. Text memory says "last time you encountered X, outcome was Y." Geometric memory says "the landscape around X has this shape because of everything that's ever happened there." The second is richer, harder to game, and doesn't require the system to interpret its own past — it just navigates a terrain that was sculpted by it.

**Implementation boundary:** None of Section 12 should be built until Phases 1-6 are validated and producing stable, interpretable traces. The within-run system must work before the between-run system can have anything meaningful to inherit.

---

*Spec v0.3 — Triad + Auditor*
*Jeff x Claude x Halcyon*
*HalcyonAIR, February 2026*
