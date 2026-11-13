"""Módulo de arquiteturas de modelos (Fases 2 e 3)."""

from physdual.models.graphdta import (
    DeepDTAStyleCNN3TargetEncoder,
    GINEncoder,
    GraphDTACNN3,
    GraphDTAGIN,
    OfficialGraphDTATargetEncoder,
    build_baseline_model,
)
from physdual.models.physdual import (
    ESMTargetEncoder,
    PhysDualPhase3,
    build_physdual_phase3_model,
)

__all__ = [
    "GINEncoder",
    "OfficialGraphDTATargetEncoder",
    "DeepDTAStyleCNN3TargetEncoder",
    "GraphDTAGIN",
    "GraphDTACNN3",
    "build_baseline_model",
    "ESMTargetEncoder",
    "PhysDualPhase3",
    "build_physdual_phase3_model",
]
