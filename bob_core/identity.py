# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright 2026 Halcyon AI Research (jeff@halcyon.ie)
"""
Identity boundary: weights events by identity relevance.

Commit events are identity events. Pre-commit exploration is not.
Broken commitments are identity events (scars matter).
"""

from typing import Optional


def is_identity_event(
    path: str,
    step: int,
    first_commit_step: Optional[int],
    loss: float,
    baseline: float,
) -> float:
    """Returns weight 0.0-1.0 for how identity-relevant this event is.

    - Commit events (cheap path): weight 1.0
    - Broken commitments (loss >> baseline): weight 1.0
    - Full routing before first commit: weight 0.0
    - Full routing after first commit: weight 0.3
    """
    if path == "cheap":
        return 1.0

    # Full routing path
    if first_commit_step is None:
        # No commits yet — pre-identity exploration
        return 0.0

    if step < first_commit_step:
        return 0.0

    # After first commit, full routing has some identity weight
    # Broken commitments (loss significantly exceeds baseline) get full weight
    if baseline > 0 and loss > baseline * 1.5:
        return 1.0

    return 0.3
