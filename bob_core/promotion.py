# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright 2026 Halcyon AI Research (jeff@halcyon.ie)
"""
Promotion gate: per-context-class stability tracker.

Sets cheap-path eligibility based on action distribution entropy
and outcome variance. Phase 1: statistics only.
Later (with routing vectors): will look at delta residuals.
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class _ClassStats:
    """Per-context-class tracking window."""
    expert_history: deque   # deque of Tuple[int, ...] (expert IDs)
    loss_history: deque     # deque of float
    steps: deque            # deque of int


class PromotionGate:
    """Per-context-class statistics tracker. Sets cheap-path eligibility.

    Phase 1: action distribution entropy + outcome variance.
    """

    def __init__(
        self,
        stability_window: int = 20,
        stability_threshold: float = 0.7,
    ):
        self.stability_window = stability_window
        self.stability_threshold = stability_threshold
        self._stats: Dict[int, _ClassStats] = {}

    def _ensure_class(self, context_class: int) -> None:
        if context_class not in self._stats:
            self._stats[context_class] = _ClassStats(
                expert_history=deque(maxlen=self.stability_window),
                loss_history=deque(maxlen=self.stability_window),
                steps=deque(maxlen=self.stability_window),
            )

    def record(
        self, context_class: int, expert_ids: Tuple[int, ...],
        loss: float, step: int,
    ) -> None:
        self._ensure_class(context_class)
        s = self._stats[context_class]
        s.expert_history.append(expert_ids)
        s.loss_history.append(loss)
        s.steps.append(step)

    def is_eligible(self, context_class: int) -> bool:
        """Low outcome variance AND low expert-selection entropy for M steps."""
        return self.eligibility_score(context_class) >= self.stability_threshold

    def eligibility_score(self, context_class: int) -> float:
        """0-1 score. Feeds into governor threshold adjustment."""
        self._ensure_class(context_class)
        s = self._stats[context_class]

        if len(s.expert_history) < self.stability_window:
            return 0.0

        # Expert-selection entropy (normalized)
        pattern_counts: Dict[Tuple[int, ...], int] = {}
        for ids in s.expert_history:
            pattern_counts[ids] = pattern_counts.get(ids, 0) + 1

        n = len(s.expert_history)
        entropy = 0.0
        for count in pattern_counts.values():
            p = count / n
            if p > 0:
                entropy -= p * math.log2(p)

        # Normalize entropy: max entropy = log2(n) when all unique
        max_entropy = math.log2(n) if n > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        # Outcome variance (coefficient of variation)
        losses = list(s.loss_history)
        mean_loss = sum(losses) / len(losses)
        if mean_loss > 0:
            variance = sum((l - mean_loss) ** 2 for l in losses) / len(losses)
            cv = math.sqrt(variance) / mean_loss
        else:
            cv = 0.0

        # Score: high when entropy is low AND variance is low
        # Both in [0,1], inverted so low = good
        entropy_score = max(0.0, 1.0 - normalized_entropy)
        variance_score = max(0.0, 1.0 - min(1.0, cv))

        return entropy_score * 0.6 + variance_score * 0.4
