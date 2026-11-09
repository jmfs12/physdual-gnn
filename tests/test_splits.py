"""Testes unitários de divisão de splits determinísticos sem vazamento por conteúdo."""

from __future__ import annotations

import pandas as pd
import pytest

from physdual.data.splits import (
    create_splits,
    load_splits,
    save_splits,
    verify_split_leakage,
)


@pytest.fixture
def synthetic_dta_df() -> pd.DataFrame:
    """Gera DataFrame sintético com 20 fármacos únicos e 10 alvos únicos (200 pares)."""
    records = []
    for d in range(20):
        for t in range(10):
            records.append({
                "drug_id": f"D{d}",
                "smiles": f"C{'C' * d}O",
                "target_id": f"T{t}",
                "sequence": f"MKTVRQ{'A' * (t + 1)}",
                "affinity": float(d + t),
            })
    return pd.DataFrame(records)


def test_cold_drug_split_has_zero_content_leakage(synthetic_dta_df: pd.DataFrame):
    splits = create_splits(synthetic_dta_df, split_type="cold_drug", seed=42)
    assert verify_split_leakage(synthetic_dta_df, splits, split_type="cold_drug")
    train_smiles = set(synthetic_dta_df.iloc[splits["train"]]["smiles"])
    val_smiles = set(synthetic_dta_df.iloc[splits["val"]]["smiles"])
    test_smiles = set(synthetic_dta_df.iloc[splits["test"]]["smiles"])
    assert train_smiles.isdisjoint(val_smiles)
    assert train_smiles.isdisjoint(test_smiles)
    assert val_smiles.isdisjoint(test_smiles)


def test_cold_target_split_has_zero_content_leakage(synthetic_dta_df: pd.DataFrame):
    splits = create_splits(synthetic_dta_df, split_type="cold_target", seed=42)
    assert verify_split_leakage(synthetic_dta_df, splits, split_type="cold_target")
    train_seqs = set(synthetic_dta_df.iloc[splits["train"]]["sequence"])
    val_seqs = set(synthetic_dta_df.iloc[splits["val"]]["sequence"])
    test_seqs = set(synthetic_dta_df.iloc[splits["test"]]["sequence"])
    assert train_seqs.isdisjoint(val_seqs)
    assert train_seqs.isdisjoint(test_seqs)
    assert val_seqs.isdisjoint(test_seqs)


def test_duplicate_content_with_different_ids_detected_and_grouped():
    """D0 e D1 têm IDs diferentes, mas o mesmo SMILES canônico ('CCO'); T0 e T1 têm a mesma sequência."""
    records = []
    for d in range(10):
        smiles = "CCO" if d in (0, 1) else f"C{'C' * d}N"
        for t in range(5):
            records.append({
                "drug_id": f"D{d}",
                "smiles": smiles,
                "target_id": f"T{t}",
                "sequence": "MKTVRQ" if t in (0, 1) else f"ACDEFG{'H' * t}",
                "affinity": 6.0,
            })
    df = pd.DataFrame(records)

    drug_splits = create_splits(df, split_type="cold_drug", seed=0)
    assert verify_split_leakage(df, drug_splits, split_type="cold_drug")

    target_splits = create_splits(df, split_type="cold_target", seed=0)
    assert verify_split_leakage(df, target_splits, split_type="cold_target")

    # Se forçarmos manualmente D0 no treino e D1 (mesmo SMILES) no teste, verify_split_leakage deve retornar False
    sub_df = df.iloc[[0, 5, 10]].reset_index(drop=True)
    corrupted_drug_splits = {"train": [0], "val": [2], "test": [1]}
    assert not verify_split_leakage(sub_df, corrupted_drug_splits, split_type="cold_drug")

    # Idem para target com mesma sequence em train e test
    sub_tgt_df = df.iloc[[0, 1, 2]].reset_index(drop=True)
    corrupted_target_splits = {"train": [0], "val": [2], "test": [1]}
    assert not verify_split_leakage(sub_tgt_df, corrupted_target_splits, split_type="cold_target")


def test_splits_determinism_proportions_and_non_empty(synthetic_dta_df: pd.DataFrame, tmp_path):
    s1 = create_splits(synthetic_dta_df, split_type="random", seed=1)
    s2 = create_splits(synthetic_dta_df, split_type="random", seed=1)
    s3 = create_splits(synthetic_dta_df, split_type="random", seed=2)

    assert s1 == s2
    assert s1 != s3

    for part in ("train", "val", "test"):
        assert len(s1[part]) > 0

    n = len(synthetic_dta_df)
    assert len(s1["train"]) == pytest.approx(0.8 * n, abs=2)
    assert len(s1["val"]) == pytest.approx(0.1 * n, abs=2)
    assert len(s1["test"]) == pytest.approx(0.1 * n, abs=2)

    out_file = save_splits(s1, tmp_path / "split_seed1.json")
    loaded = load_splits(out_file)
    assert loaded == s1
