"""Módulo de arquiteturas de modelos (Fase 2)."""

from physdual.models.graphdta import (
    DeepDTAStyleCNN3TargetEncoder,
    GINEncoder,
    GraphDTACNN3,
    GraphDTAGIN,
    OfficialGraphDTATargetEncoder,
    build_baseline_model,
)

__all__ = [
    "GINEncoder",
    "OfficialGraphDTATargetEncoder",
    "DeepDTAStyleCNN3TargetEncoder",
    "GraphDTAGIN",
    "GraphDTACNN3",
    "build_baseline_model",
]
