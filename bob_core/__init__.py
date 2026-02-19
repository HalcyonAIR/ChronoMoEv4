# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright 2026 Halcyon AI Research (jeff@halcyon.ie)
"""
Bob: consequence-accumulating control plane for MoE models.

Bob observes routing decisions, accumulates motifs, and learns when
to take the cheap path. The model does the thinking. Bob decides
how much thinking is necessary.
"""

BOB_CORE_VERSION = "v4.0.0-alpha1"

from bob_core.motifs import MotifStore, MotifRecord, GateSignals, CompoundGate, GateThresholds, GateResult
from bob_core.substrate import BobSubstrate
from bob_core.telemetry import DecisionTrace
from bob_core.ledgers import (
    BobCore,
    CommitmentLedger,
    ScarLedger,
    CostLedger,
    Commitment,
    Scar,
    RoutingVector,
    CostSignal,
    GovernanceCoords,
)
from bob_core.medium_clock import MediumClock, MediumClockState
from bob_core.governor import BobGovernor, GovernorDecision, GovernorVerdict
from bob_core.identity import is_identity_event
from bob_core.promotion import PromotionGate
from bob_core.monitors import TriadMonitor, TriadScores, TriadSummary, LayerMonitorState
from bob_core.conflict import ConflictRegister, ConflictState
from bob_core.graph import (
    RelationalGraph,
    GraphNode,
    GraphEdge,
    AliasTable,
    NodeMetadata,
    DetectedTriple,
    TripleResult,
    TripleOutcome,
    ALLOWED_RELATIONS,
)
from bob_core.basins import (
    AssociationBasin,
    BasinStore,
    MemoryBiasDiagnostics,
)

__all__ = [
    "BOB_CORE_VERSION",
    # Original
    "MotifStore",
    "MotifRecord",
    "GateSignals",
    "GateResult",
    "CompoundGate",
    "GateThresholds",
    "BobSubstrate",
    "DecisionTrace",
    # Phase 1: Ledgers
    "BobCore",
    "CommitmentLedger",
    "ScarLedger",
    "CostLedger",
    "Commitment",
    "Scar",
    "RoutingVector",
    "CostSignal",
    "GovernanceCoords",
    # Phase 1: Medium Clock
    "MediumClock",
    "MediumClockState",
    # Phase 1: Governor
    "BobGovernor",
    "GovernorDecision",
    "GovernorVerdict",
    # Phase 1: Identity + Promotion
    "is_identity_event",
    "PromotionGate",
    # Phase 2: Triad Monitors
    "TriadMonitor",
    "TriadScores",
    "TriadSummary",
    "LayerMonitorState",
    # Phase 2: Conflict Register
    "ConflictRegister",
    "ConflictState",
    # Memory System: Relational Graph
    "RelationalGraph",
    "GraphNode",
    "GraphEdge",
    "AliasTable",
    "NodeMetadata",
    "DetectedTriple",
    "TripleResult",
    "TripleOutcome",
    "ALLOWED_RELATIONS",
    # Memory System: Association Basins
    "AssociationBasin",
    "BasinStore",
    "MemoryBiasDiagnostics",
]
