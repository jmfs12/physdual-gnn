"""Módulos de dados, splits, featurização molecular, datamodule e cache ESM-2 do PhysDual-GNN."""

from __future__ import annotations

try:
    from physdual.data.datamodule import (
        PROTEIN_ALPHABET,
        PROTEIN_CHAR_TO_IDX,
        DTADataset,
        compute_aa_composition,
        dta_collate_fn,
        get_dta_dataloaders,
        prepare_local_cache,
        tokenize_protein_sequence,
    )
except ImportError:
    pass
from physdual.data.datasets import (
    EXPECTED_COUNTS,
    canonicalize_largest_fragment,
    download_canonical_dataset,
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
    create_group_kfold_splits,
    create_sequence_identity_kfold_splits,
    create_splits,
    load_splits,
    save_splits,
    verify_group_kfold_coverage,
    verify_sequence_identity_splits,
    verify_split_leakage,
)

__all__ = [
    "DTADataset",
    "EXPECTED_COUNTS",
    "PROTEIN_ALPHABET",
    "PROTEIN_CHAR_TO_IDX",
    "build_drug_graph_cache",
    "canonicalize_largest_fragment",
    "compute_aa_composition",
    "compute_seq_hash",
    "create_cache_tarball",
    "create_splits",
    "dta_collate_fn",
    "download_canonical_dataset",
    "extract_esm_embeddings_batch",
    "generate_sliding_windows",
    "get_dta_dataloaders",
    "get_embedding_cache_path",
    "graph_dict_to_pyg_data",
    "load_cached_protein_embedding",
    "load_drug_graph",
    "load_drug_graph_cache",
    "load_raw_dataset",
    "load_splits",
    "prepare_dataset",
    "prepare_local_cache",
    "save_protein_embedding",
    "save_splits",
    "smiles_to_graph_dict",
    "smiles_to_pyg_graph",
    "standardize_dataset",
    "tokenize_protein_sequence",
    "validate_protein_sequence",
    "verify_split_leakage",
]
