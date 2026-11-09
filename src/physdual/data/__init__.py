"""Módulos de dados, splits, featurização molecular e cache ESM-2 do PhysDual-GNN (Fase 1)."""

from __future__ import annotations

from physdual.data.datasets import (
    EXPECTED_COUNTS,
    canonicalize_largest_fragment,
    load_raw_dataset,
    prepare_dataset,
    standardize_dataset,
    validate_protein_sequence,
)
from physdual.data.esm_cache import (
    compute_seq_hash,
    create_cache_tarball,
    extract_esm_embeddings_batch,
    generate_sliding_windows,
    get_embedding_cache_path,
    load_cached_protein_embedding,
    save_protein_embedding,
)
from physdual.data.featurizer_drug import (
    build_drug_graph_cache,
    graph_dict_to_pyg_data,
    load_drug_graph,
    load_drug_graph_cache,
    smiles_to_graph_dict,
    smiles_to_pyg_graph,
)
from physdual.data.splits import (
    create_splits,
    load_splits,
    save_splits,
    verify_split_leakage,
)

__all__ = [
    "EXPECTED_COUNTS",
    "build_drug_graph_cache",
    "canonicalize_largest_fragment",
    "compute_seq_hash",
    "create_cache_tarball",
    "create_splits",
    "extract_esm_embeddings_batch",
    "generate_sliding_windows",
    "get_embedding_cache_path",
    "graph_dict_to_pyg_data",
    "load_cached_protein_embedding",
    "load_drug_graph",
    "load_drug_graph_cache",
    "load_raw_dataset",
    "load_splits",
    "prepare_dataset",
    "save_protein_embedding",
    "save_splits",
    "smiles_to_graph_dict",
    "smiles_to_pyg_graph",
    "standardize_dataset",
    "validate_protein_sequence",
    "verify_split_leakage",
]
