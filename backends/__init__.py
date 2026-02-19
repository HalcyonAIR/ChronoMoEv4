"""Backend adapters for Bob. Each adapter implements BackendAdapter protocol."""

from backends.adapter import (
    BackendAdapter,
    LayerSnapshot,
    LayerMotif,
    MotifSpec,
    ForwardResult,
    OverlapKind,
)

__all__ = [
    "BackendAdapter",
    "LayerSnapshot",
    "LayerMotif",
    "MotifSpec",
    "ForwardResult",
    "OverlapKind",
]
