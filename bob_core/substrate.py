# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright 2026 Halcyon AI Research (jeff@halcyon.ie)
"""
BobSubstrate: the main orchestrator.

observe() -> decide() -> update() -> log()

Bob reads adapter snapshots, evaluates the compound gate,
decides cheap or expensive path, and updates the motif store.

Phase 1: optional BobCore + Governor + MediumClock wiring.
If bob_core/governor are None -> existing behavior preserved.
"""

import math
from typing import Dict, List, Optional, Tuple

from backends.adapter import (
    BackendAdapter,
    ForwardResult,
    LayerMotif,
    LayerSnapshot,
    MotifSpec,
)
from bob_core.motifs import (
    CompoundGate,
    GateResult,
    GateSignals,
    GateThresholds,
    MotifRecord,
    MotifStore,
)
from bob_core.telemetry import DecisionTrace


class BobSubstrate:
    """
    Consequence-accumulating control plane for MoE models.

    Usage::

        adapter = SwissAdapter(model, moe_layers={0: model.moe})
        bob = BobSubstrate(adapter)

        for step in range(total_steps):
            task_class, inputs, targets = ladder.get_batch(step)
            trace = bob.step(inputs, targets, task_class, step)
            # trace.path is "cheap" or "full"
            # trace.expert_invocations is the cost
            # trace.loss is the quality

    Args:
        adapter: BackendAdapter implementation
        gate_thresholds: Thresholds for the compound gate
        warmup_steps: Steps before cheap path is offered
        governance_state: Fixed governance state (simplified for v1 experiment)
        bob_core: Optional BobCore for ledger-based lifecycle
        governor: Optional BobGovernor for commit authorization
        medium_clock: Optional MediumClock for instability detection
        fast_clock: Optional FastClock for immediate reflex
        slow_clock: Optional SlowClock for constitutional envelope
        promotion_gate: Optional PromotionGate for stability tracking
        motif_store_kwargs: Additional kwargs for MotifStore
    """

    def __init__(
        self,
        adapter: BackendAdapter,
        gate_thresholds: Optional[GateThresholds] = None,
        warmup_steps: int = 500,
        governance_state: str = "EQUILIBRIUM",
        bob_core=None,
        governor=None,
        medium_clock=None,
        fast_clock=None,
        slow_clock=None,
        promotion_gate=None,
        triad_monitor=None,
        conflict_register=None,
        memory_graph=None,
        basin_store=None,
        memory_bias_scale=0.1,
        **motif_store_kwargs,
    ):
        self.adapter = adapter
        self.gate = CompoundGate(gate_thresholds)
        self.store = MotifStore(**motif_store_kwargs)
        self.warmup_steps = warmup_steps
        self.governance_state = governance_state

        # Three overlapping clocks (all optional, backward compatible)
        self.fast_clock = fast_clock
        self.medium_clock = medium_clock
        self.slow_clock = slow_clock

        # Other components
        self.bob_core = bob_core
        self.governor = governor
        self.promotion_gate = promotion_gate

        # Triad monitors (all optional, backward compatible)
        self.triad_monitor = triad_monitor
        self.conflict_register = conflict_register

        # Memory system (all optional, backward compatible)
        self.memory_graph = memory_graph        # Optional[RelationalGraph]
        self.basin_store = basin_store          # Optional[BasinStore]
        self.memory_bias_scale = memory_bias_scale

        # Track previous step state for clock ticks
        self._prev_expert_ids: Optional[Tuple[int, ...]] = None
        self._prev_loss: Optional[float] = None
        self._prev_entropy: Optional[float] = None
        self._first_commit_step: Optional[int] = None

        # ESCALATE recovery: substrate temporarily lowers fast explore threshold
        # _pre_escalate_threshold is set at ESCALATE time (captures calibrated value)
        self._pre_escalate_threshold: Optional[float] = None
        self._escalation_recovery_steps: int = 0

        # Independent counters: gate vs governor attribution
        self._gate_eval_count: int = 0
        self._gate_pass_count: int = 0
        self._governor_eval_count: int = 0
        self._governor_allow_count: int = 0

        self.traces: List[DecisionTrace] = []

    def step(
        self,
        inputs,
        targets,
        context_class: int,
        step: int,
        entity_tokens: Optional[List[str]] = None,
    ) -> DecisionTrace:
        """
        Run one decision through Bob.

        Three overlapping clocks tick once per step, after execution:
        - Fast clock: immediate reflex (drives exploration pressure)
        - Medium clock: regime confidence (modulates bias strength)
        - Slow clock: constitutional envelope (reshapes governor thresholds)

        Step flow:
        0. Read previous-tick activations from all clocks
        1. Check fast_clock.exploration_pressure → bypass gate if true
        2. Gate evaluation
        3. If gate passed: governor.evaluate_commit() (pure reader)
        4. Execute (cheap or expensive)
        5. Post-execution: ledgers, motif store, promotion gate, identity
        6. Tick all clocks (single tick, actual data)
        7. Log DecisionTrace with all three activations

        Without governor/bob_core: existing behavior preserved.
        """
        was_blocked = False
        fast_exploration = False
        governor_decision = None
        governor_reasons = None
        scar_debt = None
        cost_cheap_fraction = None
        commitment_id = None
        identity_weight = None
        scar_overlap = None
        escalation_count = None

        # --- 0. Read previous-tick activations ---
        medium_activation = self.medium_clock.activation if self.medium_clock else None
        fast_activation = self.fast_clock.activation if self.fast_clock else None
        slow_activation = self.slow_clock.activation if self.slow_clock else None

        # Decay escalation-lowered fast threshold back to pre-escalate value
        if (self.fast_clock is not None
                and self._escalation_recovery_steps > 0):
            self._escalation_recovery_steps -= 1
            if self._escalation_recovery_steps <= 0 and self._pre_escalate_threshold is not None:
                self.fast_clock.explore_threshold = self._pre_escalate_threshold
                self._pre_escalate_threshold = None

        # --- 1. Fast clock alarms (volatility + funnel) ---
        if self.fast_clock is not None:
            if self.fast_clock.exploration_pressure or self.fast_clock.neff_collapse:
                fast_exploration = True

        # --- 2-3. Gate signals ---
        signals = self.store.get_gate_signals(context_class, step)

        # If bob_core exists, use scar-based debt instead of motif store debt
        if self.bob_core is not None:
            scar_debt = self.bob_core.scars.total_debt(step)

        # --- 4. Gate evaluation ---
        # Fast exploration is a soft override: it relaxes gate thresholds
        # (halves the stability/survival requirements, doubles debt allowance)
        # but does NOT bypass the gate or the governor. The gate and governor
        # always have a say.
        if step < self.warmup_steps:
            gate_result = GateResult(
                passed=False,
                signals=signals,
                thresholds_used=self.gate.thresholds,
                governance_state=self.governance_state,
                stability_passed=False,
                debt_passed=False,
                survival_passed=False,
            )
        elif fast_exploration:
            # Relaxed gate: lower bar but still check
            relaxed_thresholds = GateThresholds(
                stability_min=self.gate.thresholds.stability_min * 0.5,
                debt_max=min(1.0, self.gate.thresholds.debt_max * 2.0),
                survival_min=self.gate.thresholds.survival_min * 0.5,
            )
            relaxed_gate = CompoundGate(relaxed_thresholds)
            gate_result = relaxed_gate.evaluate(signals, self.governance_state)
        else:
            gate_result = self.gate.evaluate(signals, self.governance_state)

        # --- Track gate pass/fail independently ---
        if step >= self.warmup_steps:
            self._gate_eval_count += 1
            if gate_result.passed:
                self._gate_pass_count += 1

        # --- 5. Governor evaluation (if gate passed) ---
        top_motif = self.store.get_top_motif(context_class, step)

        if gate_result.passed and self.governor is not None and top_motif is not None:
            first_layer = next(iter(top_motif.motif_spec.layers.values()))
            candidate_ids = first_layer.expert_ids

            from bob_core.governor import GovernorDecision
            verdict = self.governor.evaluate_commit(
                context_class=context_class,
                expert_ids=candidate_ids,
                gate_result=gate_result,
                step=step,
            )
            governor_decision = verdict.decision.value
            governor_reasons = verdict.reasons
            cost_cheap_fraction = verdict.cost_signal.cheap_fraction
            scar_overlap = verdict.scar_overlap
            escalation_count = verdict.escalation_count

            # Track governor outcomes independently from gate
            self._governor_eval_count += 1
            if verdict.decision == GovernorDecision.ALLOW:
                self._governor_allow_count += 1

            if verdict.decision != GovernorDecision.ALLOW:
                # Governor blocked: force expensive path
                gate_result = GateResult(
                    passed=False,
                    signals=signals,
                    thresholds_used=self.gate.thresholds,
                    governance_state=self.governance_state,
                    stability_passed=gate_result.stability_passed,
                    debt_passed=False,
                    survival_passed=gate_result.survival_passed,
                )
                was_blocked = True

                # ESCALATE distinct substrate response: temporarily lower
                # fast clock's explore threshold so exploration is more likely
                # for the next few steps. This is a substrate action, not a
                # governor write — governor just said "this is serious."
                if verdict.decision == GovernorDecision.ESCALATE:
                    if self.fast_clock is not None:
                        # Save current (possibly calibrated) threshold before lowering
                        if self._pre_escalate_threshold is None:
                            self._pre_escalate_threshold = self.fast_clock.explore_threshold
                        self.fast_clock.explore_threshold = max(
                            0.15, self.fast_clock.explore_threshold - 0.15
                        )
                        self._escalation_recovery_steps = 10

        # --- Memory bias computation (before execution) ---
        memory_bias = None
        memory_diag = None
        if (self.basin_store is not None
                and self.memory_graph is not None
                and entity_tokens):
            from bob_core.basins import (
                link_entities, diffuse_activation, compute_memory_bias,
            )
            activations = link_entities(entity_tokens, self.memory_graph)
            diffused = diffuse_activation(activations, self.memory_graph)
            memory_bias, memory_diag = compute_memory_bias(
                diffused, self.basin_store,
                self.adapter.num_layers, self.adapter.num_experts,
            )

        # --- 6. Execute ---
        if gate_result.passed and top_motif is not None:
            # Compute bias strength: strong when stable, weak when unstable
            base_bias = 5.0
            instability_scale = 1.0 - (medium_activation or 0.0)  # 1.0=stable, 0.0=chaotic
            promotion_scale = 1.0
            if self.promotion_gate is not None:
                promotion_scale = self.promotion_gate.eligibility_score(context_class)
            bias = base_bias * instability_scale * max(0.3, promotion_scale)

            # Apply bias to all layers in the motif
            for lm in top_motif.motif_spec.layers.values():
                lm.bias_strength = bias

            result = self.adapter.forward_with_motif(
                inputs, top_motif.motif_spec, targets,
                memory_bias=memory_bias,
            )
            path = "cheap"
            motif_id = top_motif.motif_id
            first_layer = next(iter(top_motif.motif_spec.layers.values()))
            expert_ids = first_layer.expert_ids
        else:
            result = self.adapter.forward(inputs, targets, memory_bias=memory_bias)
            path = "full"
            motif_id = None
            expert_ids = ()

        loss_val = result.loss.item() if result.loss is not None else float("inf")

        # --- 7. Extract motif + routing vector ---
        actual_motif_spec, extracted_ids, routing_key = self._extract_motif(
            result, self.adapter.num_experts
        )
        if not expert_ids:
            expert_ids = extracted_ids

        # Build routing vectors from snapshots (if bob_core exists)
        routing_vector = None
        if self.bob_core is not None and result.snapshots:
            from bob_core.ledgers import RoutingVector
            routing_vector = RoutingVector.from_snapshot(result.snapshots[0])

        # --- Geometry: compute fields for trajectory logging ---
        router_entropy = None
        if result.snapshots:
            ents = [s.mean_entropy for s in result.snapshots if s.mean_entropy is not None]
            if ents:
                router_entropy = sum(ents) / len(ents)

        neff = None
        neff_per_layer = None
        if result.snapshots:
            layer_neffs = []
            for snap in result.snapshots:
                if snap.expert_usage is not None:
                    usage_sq = (snap.expert_usage * snap.expert_usage).sum().item()
                    if usage_sq > 0:
                        layer_neffs.append(1.0 / usage_sq)
            if layer_neffs:
                neff_per_layer = layer_neffs
                neff = min(layer_neffs)  # Worst layer drives collapse detection

        routing_weights_top = None
        if result.snapshots and result.snapshots[0].routing_weights is not None:
            mean_wts = result.snapshots[0].routing_weights.float().mean(dim=0)
            routing_weights_top = [round(float(w), 4) for w in mean_wts]

        # --- 8-9. BobCore outcome processing ---
        commitment = None
        if self.bob_core is not None:
            governance_coords = self.bob_core.get_governance_coords(
                context_class=context_class,
                step=step,
                medium=medium_activation or 0.0,
            )
            commitment = self.bob_core.process_outcome(
                context_class=context_class,
                step=step,
                loss=loss_val,
                was_cheap=(path == "cheap"),
                expert_ids=expert_ids,
                routing_vector=routing_vector,
                governance_coords=governance_coords,
                expert_invocations=result.expert_invocations,
                was_exploration=fast_exploration,
                was_blocked=was_blocked,
                motif_id=motif_id,
            )
            if commitment is not None:
                commitment_id = commitment.commitment_id
                if self._first_commit_step is None:
                    self._first_commit_step = step

        # --- 10. Update motif store ---
        self.store.update(
            context_class=context_class,
            expert_ids=expert_ids,
            motif_spec=actual_motif_spec,
            loss=loss_val,
            step=step,
            was_cheap=(path == "cheap"),
            routing_key=routing_key,
        )

        # --- 11. Promotion gate ---
        if self.promotion_gate is not None:
            self.promotion_gate.record(context_class, expert_ids, loss_val, step)

        # --- 12. Identity weight ---
        if self.bob_core is not None:
            from bob_core.identity import is_identity_event
            baseline = self.bob_core.commitments.baseline_loss(context_class)
            identity_weight = is_identity_event(
                path=path,
                step=step,
                first_commit_step=self._first_commit_step,
                loss=loss_val,
                baseline=baseline if baseline != float("inf") else loss_val,
            )

        # --- Tick all three clocks (single tick, after execution, actual data) ---
        if self.fast_clock is not None:
            self.fast_clock.tick(
                prev_expert_ids=self._prev_expert_ids,
                curr_expert_ids=expert_ids,
                prev_loss=self._prev_loss,
                curr_loss=loss_val,
                prev_entropy=self._prev_entropy,
                curr_entropy=router_entropy,
                neff=neff,
                neff_per_layer=neff_per_layer,
            )
            fast_activation = self.fast_clock.activation

        if self.medium_clock is not None:
            self.medium_clock.tick(
                prev_expert_ids=self._prev_expert_ids,
                curr_expert_ids=expert_ids,
                prev_loss=self._prev_loss,
                curr_loss=loss_val,
                path=path,
            )
            medium_activation = self.medium_clock.activation

        if self.slow_clock is not None:
            slow_scar_debt = self.bob_core.scars.total_debt(step) if self.bob_core else 0.0
            slow_stability = signals.routing_stability
            self.slow_clock.tick(
                scar_debt=slow_scar_debt,
                routing_stability=slow_stability,
            )
            slow_activation = self.slow_clock.activation

        # --- Tick triad monitors (after clocks, before trace) ---
        triad_summary = None
        conflict_state = None
        if self.triad_monitor is not None and result.snapshots:
            triad_summary = self.triad_monitor.tick(result.snapshots)
            if self.conflict_register is not None:
                conflict_state = self.conflict_register.update(
                    triad_summary.angel_peak, triad_summary.devil_peak
                )

        # --- Geometry: extract remaining fields ---
        churn_val = self.medium_clock.state.last_churn if self.medium_clock else None
        flipflop_ema_val = self.medium_clock.state.flipflop_ema if self.medium_clock else None

        scar_hit = None
        if self.bob_core is not None and expert_ids:
            region = tuple(sorted(expert_ids))
            scar_hit = self.bob_core.scars.is_in_scar_neighborhood(region, step)

        baseline_loss_val = None
        if self.bob_core is not None:
            bl = self.bob_core.commitments.baseline_loss(context_class)
            if bl != float("inf"):
                baseline_loss_val = bl

        # Update prev state for next step's clock ticks
        self._prev_expert_ids = expert_ids
        self._prev_loss = loss_val
        self._prev_entropy = router_entropy

        # --- 13. Log ---
        trace = DecisionTrace(
            step=step,
            context_class=context_class,
            governance_state=self.governance_state,
            path=path,
            expert_ids=expert_ids,
            expert_invocations=result.expert_invocations,
            tokens_processed=result.tokens_processed,
            loss=loss_val,
            routing_stability=signals.routing_stability,
            debt_level=signals.debt_level,
            motif_survival=signals.motif_survival,
            gate_passed=gate_result.passed,
            stability_passed=gate_result.stability_passed,
            debt_passed=gate_result.debt_passed,
            survival_passed=gate_result.survival_passed,
            motif_id=motif_id,
            governor_decision=governor_decision,
            governor_reasons=governor_reasons,
            forced_exploration=fast_exploration,
            medium_activation=medium_activation,
            scar_debt=scar_debt,
            cost_cheap_fraction=cost_cheap_fraction,
            commitment_id=commitment_id,
            identity_weight=identity_weight,
            router_entropy=router_entropy,
            churn=churn_val,
            scar_hit=scar_hit,
            baseline_loss=baseline_loss_val,
            neff=neff,
            flipflop_ema=flipflop_ema_val,
            routing_weights_top=routing_weights_top,
            scar_overlap=scar_overlap,
            escalation_count=escalation_count,
            gate_pass_rate=(
                self._gate_pass_count / self._gate_eval_count
                if self._gate_eval_count > 0 else None
            ),
            governor_allow_rate=(
                self._governor_allow_count / self._governor_eval_count
                if self._governor_eval_count > 0 else None
            ),
            fast_activation=fast_activation,
            slow_activation=slow_activation,
            neff_collapse=(
                self.fast_clock.neff_collapse if self.fast_clock is not None else None
            ),
            neff_floor=(
                self.fast_clock.neff_floor if self.fast_clock is not None else None
            ),
            neff_per_layer=neff_per_layer,
            neff_collapse_layers=(
                self.fast_clock.neff_collapse_layers if self.fast_clock is not None else None
            ),
            # Triad monitors
            angel_score=triad_summary.angel_peak if triad_summary else None,
            devil_score=triad_summary.devil_peak if triad_summary else None,
            maniac_score=triad_summary.maniac_peak if triad_summary else None,
            angel_flag=triad_summary.angel_flag if triad_summary else None,
            devil_flag=triad_summary.devil_flag if triad_summary else None,
            maniac_flag=triad_summary.maniac_flag if triad_summary else None,
            triad_intervention=triad_summary.intervention if triad_summary else None,
            triad_intervention_layer=triad_summary.intervention_layer if triad_summary else None,
            # Conflict register
            conflict_index=conflict_state.index if conflict_state else None,
            conflict_mean=conflict_state.mean_50 if conflict_state else None,
            conflict_mode=conflict_state.mode if conflict_state else None,
            conflict_trending=conflict_state.trending if conflict_state else None,
            # Memory system
            memory_nodes_active=(
                memory_diag.n_active_nodes if memory_diag else None
            ),
            memory_bias_applied=(
                memory_bias is not None if memory_diag is not None else None
            ),
            memory_bias_max=(
                memory_diag.max_abs_bias if memory_diag else None
            ),
            memory_bias_to_logit_ratio=(
                memory_diag.bias_to_logit_ratio if memory_diag else None
            ),
            memory_active_basins=(
                memory_diag.n_active_basins if memory_diag else None
            ),
            memory_active_entities=(
                memory_diag.n_active_nodes if memory_diag else None
            ),
        )
        self.traces.append(trace)
        return trace

    def _extract_motif(
        self,
        result: ForwardResult,
        num_experts: int,
    ) -> Tuple[MotifSpec, Tuple[int, ...], Tuple[int, ...]]:
        """
        Build a MotifSpec from actual token-level routing.

        Aggregates per-token expert selections across the batch to find
        experts that handle a disproportionate share of tokens. The motif
        includes only experts whose token-level selection frequency exceeds
        the uniform baseline by a meaningful margin.

        Returns (motif_spec, full_representative_ids, routing_key).
        - full_representative_ids: all dominant experts from first layer (for scars)
        - routing_key: top-2 dominant experts from first layer (for stability)
        """
        layers = {}
        representative_ids = ()
        routing_key = ()
        top_k = getattr(self.adapter, 'top_k', 2)

        # Dominance threshold: token frequency (fraction of tokens where expert
        # appears in top-k). Must be 2x the uniform expectation.
        # For 8E top-2: uniform = 0.25, threshold = 0.50
        # For 64E top-8: uniform = 0.125, threshold = 0.25
        uniform_token_freq = min(1.0, top_k / num_experts)
        dominance_multiplier = 2.0
        dominance_threshold = uniform_token_freq * dominance_multiplier

        for snap in result.snapshots:
            selected = snap.selected_experts  # [B*T, top_k]
            routing_w = snap.routing_weights  # [B*T, top_k]
            num_tokens = selected.shape[0]

            # Count per-expert: how many tokens selected this expert
            expert_counts: Dict[int, int] = {}
            expert_weight_sums: Dict[int, float] = {}
            for i in range(num_tokens):
                for j in range(selected.shape[1]):
                    eid = int(selected[i, j].item())
                    expert_counts[eid] = expert_counts.get(eid, 0) + 1
                    expert_weight_sums[eid] = (
                        expert_weight_sums.get(eid, 0.0)
                        + routing_w[i, j].item()
                    )

            if not expert_counts:
                continue

            # Find experts above dominance threshold (token frequency)
            dominant = [
                eid for eid, cnt in expert_counts.items()
                if cnt / num_tokens > dominance_threshold
            ]

            if dominant:
                # Specialization exists: use dominant experts
                dominant.sort(key=lambda e: expert_counts[e], reverse=True)
                # Ensure minimum experts for quality (at least top_k - 2)
                min_experts = max(2, top_k - 2)
                if len(dominant) < min_experts:
                    all_sorted = sorted(
                        expert_counts.items(), key=lambda x: x[1], reverse=True
                    )
                    for eid, cnt in all_sorted:
                        if eid not in dominant:
                            dominant.append(eid)
                        if len(dominant) >= min_experts:
                            break
                expert_ids = tuple(dominant)
            else:
                # No specialization: use the most common top-k set
                pair_counts: Dict[Tuple[int, ...], int] = {}
                for i in range(num_tokens):
                    pair = tuple(sorted(
                        int(selected[i, j].item())
                        for j in range(selected.shape[1])
                    ))
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1
                best_pair = max(pair_counts, key=pair_counts.get)
                expert_ids = best_pair

            # Compute weights from aggregate routing weights
            total_weight = sum(expert_weight_sums.get(e, 0) for e in expert_ids)
            if total_weight > 0:
                weights = tuple(
                    expert_weight_sums.get(e, 0) / total_weight
                    for e in expert_ids
                )
            else:
                weights = tuple(1.0 / len(expert_ids) for _ in expert_ids)

            layers[snap.layer_id] = LayerMotif(
                expert_ids=expert_ids,
                weights=weights,
            )

            if not representative_ids:
                representative_ids = expert_ids
                # Routing key: top-2 dominant, sorted for stable comparison
                key_experts = dominant[:2] if dominant else list(expert_ids[:2])
                routing_key = tuple(sorted(key_experts))

        return MotifSpec(layers=layers), representative_ids, routing_key

    def get_traces_window(self, last_n: int) -> List[DecisionTrace]:
        """Get the last N traces."""
        return self.traces[-last_n:]

    def get_pareto_data(self, window: int = 50) -> List[Dict]:
        """
        Get (cost, quality) pairs per window for Pareto analysis.

        Returns list of dicts with avg_cost, avg_loss, cheap_fraction per window.
        """
        data = []
        for i in range(0, len(self.traces), window):
            window_traces = self.traces[i : i + window]
            if not window_traces:
                continue

            avg_cost = sum(t.expert_invocations for t in window_traces) / len(
                window_traces
            )
            avg_loss = sum(t.loss for t in window_traces) / len(window_traces)
            cheap_count = sum(1 for t in window_traces if t.path == "cheap")
            cheap_frac = cheap_count / len(window_traces)

            data.append(
                {
                    "window_start": window_traces[0].step,
                    "window_end": window_traces[-1].step,
                    "avg_cost": avg_cost,
                    "avg_loss": avg_loss,
                    "cheap_fraction": cheap_frac,
                    "n_traces": len(window_traces),
                }
            )
        return data
