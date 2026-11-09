"""Testes unitários de ingestão, validação de procedência e padronização (Davis e KIBA)."""

from __future__ import annotations

import pandas as pd
import pytest

from physdual.data.datasets import (
    canonicalize_largest_fragment,
    standardize_dataset,
    validate_protein_sequence,
)


def test_standardize_davis_nm_dataset_and_colapse_reporting():
    """Valida auto-detecção em nível de dataset para Davis em nM, remoção de sais e relato by_id vs by_content."""
    raw_df = pd.DataFrame({
        "drug_id": ["D1", "D2", "D3"],
        "compound_iso_smiles": [
            "CC(=O)OC1=CC=CC=C1C(=O)O",
            "CC(=O)NC1=CC=C(O)C=C1",
            "CC(=O)OC1=CC=CC=C1C(=O)O.[Na+]",  # colapsa no mesmo SMILES de D1 após remoção do sal
        ],
        "target_id": ["T1", "T1", "T2"],
        "target_sequence": [
            "MKTVRQERLKSIVRILERSKEPVS",
            "MKTVRQERLKSIVRILERSKEPVS",
            "MKTVRQERLKSIVRILERSKEPVSXBUZO",  # inclui aminoácidos estendidos X,B,U,Z,O
        ],
        "affinity": [10.0, 1000.0, 10000.0],  # nM -> pKd: 8.0, 6.0, 5.0 (censurado)
    })
    std, report = standardize_dataset(raw_df, dataset_name="davis", affinity_unit="auto")
    assert list(std.columns) == [
        "drug_id", "smiles", "smiles_raw", "target_id", "sequence", "affinity"
    ]
    assert std.loc[0, "affinity"] == pytest.approx(8.0, abs=1e-2)
    assert std.loc[1, "affinity"] == pytest.approx(6.0, abs=1e-2)
    assert std.loc[2, "affinity"] == pytest.approx(5.0, abs=1e-2)
    assert report["affinity_unit_detected"] == "nM"
    assert report["davis_censored_count"] == 1
    assert report["smiles_modified_count"] >= 1
    assert report["num_drugs_by_id"] == 3
    assert report["num_drugs_by_canonical_smiles"] == 2
    assert report["num_targets_by_id"] == 2
    assert report["num_targets_by_sequence"] == 2
    assert report["ambiguous_aa_seq_count"] == 1
    assert report["ambiguous_aa_char_counts"]["X"] == 1
    assert report["ambiguous_aa_char_counts"]["U"] == 1
    assert report["ambiguous_aa_char_counts"]["O"] == 1
    assert report["raw_num_pairs"] == 3
    assert report["clean_num_pairs"] == 3
    assert report["num_pairs_dropped_recorded"] == 0
    assert std.loc[2, "smiles_raw"] == "CC(=O)OC1=CC=CC=C1C(=O)O.[Na+]"


def test_duplicate_content_pairs_not_silently_dropped():
    """Dois IDs distintos colapsando no mesmo (smiles, sequence) não devem ser descartados silenciosamente."""
    raw_df = pd.DataFrame({
        "drug_id": ["D1", "D2"],
        "smiles": ["CCO", "CCO.[Na+]"],
        "target_id": ["T1", "T1"],
        "sequence": ["MKTVRQ", "MKTVRQ"],
        "affinity": [7.0, 8.0],
    })
    std_keep, rep_keep = standardize_dataset(
        raw_df, dataset_name="davis", affinity_unit="pKd", duplicate_policy="keep_both"
    )
    assert len(std_keep) == 2
    assert rep_keep["duplicate_content_pairs_count"] == 1
    assert rep_keep["num_pairs_dropped_recorded"] == 0

    std_mean, rep_mean = standardize_dataset(
        raw_df, dataset_name="davis", affinity_unit="pKd", duplicate_policy="mean"
    )
    assert len(std_mean) == 1
    assert std_mean.loc[0, "affinity"] == pytest.approx(7.5, abs=1e-4)
    assert rep_mean["num_pairs_dropped_recorded"] == 1


def test_standardize_davis_pkd_dataset():
    """Dataset explicitamente em pKd não sofre nova transformação logarítmica."""
    raw_df = pd.DataFrame({
        "smiles": ["CC(=O)OC1=CC=CC=C1C(=O)O", "CC(=O)NC1=CC=C(O)C=C1"],
        "sequence": ["MKTVRQERLKSIVRILERSKEPVS", "ACDEFGHIKLMNPQRSTVWY"],
        "affinity": [7.5, 5.0],
    })
    std, report = standardize_dataset(raw_df, dataset_name="davis", affinity_unit="pKd")
    assert std.loc[0, "affinity"] == pytest.approx(7.5, abs=1e-4)
    assert std.loc[1, "affinity"] == pytest.approx(5.0, abs=1e-4)
    assert report["davis_censored_count"] == 1


def test_standardize_davis_auto_ambiguous_raises():
    """Amostra única ou dataset sem valores > 11 e sem distribuição completa levanta ValueError em modo auto."""
    raw_df = pd.DataFrame({
        "smiles": ["CC(=O)OC1=CC=CC=C1C(=O)O"],
        "sequence": ["MKTVRQERLKSIVRILERSKEPVS"],
        "affinity": [7.5],
    })
    with pytest.raises(ValueError, match="ambígua|affinity_unit"):
        standardize_dataset(raw_df, dataset_name="davis", affinity_unit="auto")


def test_kiba_never_converted_to_pkd():
    """KIBA usa kiba_score; valores > 11 jamais sofrem transformação logarítmica de pKd."""
    raw_df = pd.DataFrame({
        "smiles": ["CC(=O)OC1=CC=CC=C1C(=O)O", "CC(=O)NC1=CC=C(O)C=C1"],
        "sequence": ["MKTVRQERLKSIVRILERSKEPVS", "ACDEFGHIKLMNPQRSTVWY"],
        "affinity": [11.4, 14.2],
    })
    std, report = standardize_dataset(raw_df, dataset_name="kiba", affinity_unit="auto")
    assert std.loc[0, "affinity"] == pytest.approx(11.4, abs=1e-4)
    assert std.loc[1, "affinity"] == pytest.approx(14.2, abs=1e-4)
    assert report["affinity_unit_detected"] == "kiba_score"

    with pytest.raises(ValueError, match="KIBA|kiba_score"):
        standardize_dataset(raw_df, dataset_name="kiba", affinity_unit="nM")


def test_enforce_expected_counts_raises_on_mismatch():
    """Validação de procedência falha explicitamente na tabela bruta se contagens divergirem de 30.056 / 118.254."""
    raw_df = pd.DataFrame({
        "smiles": ["CC(=O)OC1=CC=CC=C1C(=O)O"],
        "sequence": ["MKTVRQERLKSIVRILERSKEPVS"],
        "affinity": [7.5],
    })
    with pytest.raises(AssertionError, match="30056|contagem"):
        standardize_dataset(
            raw_df, dataset_name="davis", affinity_unit="pKd", enforce_expected_counts=True
        )


def test_validate_protein_sequence_policy():
    """Mantém X, B, U, Z, O e remove quebras de linha, rejeitando caracteres inválidos."""
    clean = validate_protein_sequence("mktvrq\nXBUZO ")
    assert clean == "MKTVRQXBUZO"
    with pytest.raises(ValueError, match="caractere inválido"):
        validate_protein_sequence("MKTVRQ*J123")


def test_invalid_smiles_recorded_without_crashing_batch():
    """SMILES inválidos são registrados em smiles_failures sem abortar o lote."""
    smiles_canon, modified, err = canonicalize_largest_fragment("INVALID_SMILES_XYZ")
    assert smiles_canon is None
    assert err is not None

    raw_df = pd.DataFrame({
        "drug_id": ["D_OK", "D_BAD"],
        "smiles": ["CC(=O)OC1=CC=CC=C1C(=O)O", "INVALID_SMILES_XYZ"],
        "target_id": ["T1", "T1"],
        "sequence": ["MKTVRQERLKSIVRILERSKEPVS", "MKTVRQERLKSIVRILERSKEPVS"],
        "affinity": [7.5, 6.0],
    })
    std, report = standardize_dataset(raw_df, dataset_name="davis", affinity_unit="pKd")
    assert len(std) == 1
    assert report["smiles_failed_count"] == 1
    assert report["num_pairs_dropped_recorded"] == 1
    assert report["raw_num_pairs"] - report["clean_num_pairs"] == report["num_pairs_dropped_recorded"]
