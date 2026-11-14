"""Módulos de dados, splits, featurização molecular, datamodule e cache ESM-2 do PhysDual-GNN."""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, str] = {
    # datamodule
    "PROTEIN_ALPHABET": "physdual.data.datamodule",
    "PROTEIN_CHAR_TO_IDX": "physdual.data.datamodule",
    "DTADataset": "physdual.data.datamodule",
    "compute_aa_composition": "physdual.data.datamodule",
    "dta_collate_fn": "physdual.data.datamodule",
    "get_dta_dataloaders": "physdual.data.datamodule",
    "prepare_local_cache": "physdual.data.datamodule",
    "tokenize_protein_sequence": "physdual.data.datamodule",
    # datasets
    "EXPECTED_COUNTS": "physdual.data.datasets",
    "canonicalize_largest_fragment": "physdual.data.datasets",
    "download_canonical_dataset": "physdual.data.datasets",
    "load_raw_dataset": "physdual.data.datasets",
    "prepare_dataset": "physdual.data.datasets",
    "standardize_dataset": "physdual.data.datasets",
    "validate_protein_sequence": "physdual.data.datasets",
    # esm_cache
    "compute_seq_hash": "physdual.data.esm_cache",
    "create_cache_tarball": "physdual.data.esm_cache",
    "extract_esm_embeddings_batch": "physdual.data.esm_cache",
    "generate_sliding_windows": "physdual.data.esm_cache",
    "get_embedding_cache_path": "physdual.data.esm_cache",
    "load_cached_protein_embedding": "physdual.data.esm_cache",
    "save_protein_embedding": "physdual.data.esm_cache",
    # featurizer_drug
    "build_drug_graph_cache": "physdual.data.featurizer_drug",
    "graph_dict_to_pyg_data": "physdual.data.featurizer_drug",
    "load_drug_graph": "physdual.data.featurizer_drug",
    "load_drug_graph_cache": "physdual.data.featurizer_drug",
    "smiles_to_graph_dict": "physdual.data.featurizer_drug",
    "smiles_to_pyg_graph": "physdual.data.featurizer_drug",
    # splits
    "create_group_kfold_splits": "physdual.data.splits",
    "create_sequence_identity_kfold_splits": "physdual.data.splits",
    "create_splits": "physdual.data.splits",
    "load_splits": "physdual.data.splits",
    "save_splits": "physdual.data.splits",
    "verify_group_kfold_coverage": "physdual.data.splits",
    "verify_sequence_identity_splits": "physdual.data.splits",
    "verify_split_leakage": "physdual.data.splits",
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    mod_name = _EXPORTS.get(name)
    if mod_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = importlib.import_module(mod_name)
    return getattr(mod, name)
