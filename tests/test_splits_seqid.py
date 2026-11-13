"""Testes para partições por identidade de sequência (`seqid30` e `seqid40`)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from physdual.data.splits import (
    compute_global_sequence_identity,
    create_sequence_identity_kfold_splits,
    load_splits,
    verify_group_kfold_coverage,
    verify_sequence_identity_splits,
    verify_split_leakage,
)


def _make_family_df() -> pd.DataFrame:
    """Cria 10 proteínas sintéticas: 2 pares com alta identidade (~80%) e demais distintas, em 50 pares."""
    fam_a1 = "ACDEFGHIKLMNPQRSTVWY" * 3  # 60 aa
    fam_a2 = ("ACDEFGHIKLMNPQRSTVWY" * 2) + "ACDEFGHIKLMNPQRAAAAA"  # ~91% id com a1
    fam_b1 = "WYWVTSRQPNMLKIHGFEDC" * 3
    fam_b2 = ("WYWVTSRQPNMLKIHGFEDC" * 2) + "WYWVTSRQPNMLKIHMMMMM"
    short_p = "ACDEFGHIKLMNPQRSTVWY"  # 20 aa (< 0.34 de 60 aa -> ident <= 0.333 mesmo se prefixo)
    others = [
        "MMMMMMMMMMMMMMMMMMMMYYYYYYYYYYYYYYYYYYYYKKKKKKKKKKKKKKKKKKKK",
        "NNNNNNNNNNNNNNNNNNNNRRRRRRRRRRRRRRRRRRRRTTTTTTTTTTTTTTTTTTTT",
        "PPPPPPPPPPPPPPPPPPPPSSSSSSSSSSSSSSSSSSSSVVVVVVVVVVVVVVVVVVVV",
        "DDDDDDDDDDDDDDDDDDDDEEEEEEEEEEEEEEEEEEEEGGGGGGGGGGGGGGGGGGGG",
        "HHHHHHHHHHHHHHHHHHHHIIIIIIIIIIIIIIIIIIIILLLLLLLLLLLLLLLLLLLL",
    ]
    seqs = [fam_a1, fam_a2, fam_b1, fam_b2, short_p, *others]
    drugs = ["CCO", "CCN"]
    rows = []
    for d_i, smi in enumerate(drugs):
        for t_i, seq in enumerate(seqs):
            rows.append({
                "drug_id": f"D{d_i}",
                "smiles": smi,
                "target_id": f"T{t_i}",
                "sequence": seq,
                "affinity": 5.0 + 0.1 * t_i + 0.5 * d_i,
            })
    return pd.DataFrame(rows)


def test_identity_definition_uses_max_length_denominator() -> None:
    s60 = "ACDEFGHIKLMNPQRSTVWY" * 3
    s20 = "ACDEFGHIKLMNPQRSTVWY"
    ident = compute_global_sequence_identity(s60, s20)
    assert ident == pytest.approx(20.0 / 60.0, abs=1e-6)


def test_sequence_identity_kfold_keeps_homologs_together_and_zero_leakage() -> None:
    df = _make_family_df()
    by_tau = create_sequence_identity_kfold_splits(
        df,
        identity_thresholds=(0.40, 0.70),
        n_splits=5,
        group_col="sequence",
        n_workers=1,
    )
    assert set(by_tau.keys()) == {0.40, 0.70}

    for tau, folds in by_tau.items():
        assert len(folds) == 5
        assert verify_group_kfold_coverage(df, folds, split_type=f"cold_target_seqid{int(tau * 100)}")
        assert verify_sequence_identity_splits(df, folds, identity_threshold=tau)
        for f in folds:
            assert verify_split_leakage(df, f, split_type=f"cold_target_seqid{int(tau * 100)}")


def test_generated_davis_seqid_split_files_if_present() -> None:
    root = Path(__file__).resolve().parents[1]
    csv_path = root / "data" / "processed" / "davis_clean.csv"
    splits_dir = root / "data" / "processed" / "splits"
    if not csv_path.exists():
        pytest.skip("davis_clean.csv ausente no ambiente.")

    df = pd.read_csv(csv_path)
    for pct, exp_clusters in ((30, 120), (40, 199)):
        files = [splits_dir / f"davis_cold_target_seqid{pct}_fold{k}.json" for k in range(5)]
        if not all(p.exists() for p in files):
            pytest.skip(f"Splits seqid{pct} ainda não gerados em {splits_dir}.")
        folds = [load_splits(p) for p in files]
        assert verify_group_kfold_coverage(df, folds, split_type=f"cold_target_seqid{pct}", group_col="sequence")
        assert int(folds[0]["num_clusters_total"]) == exp_clusters
        # Conferir que metadados documentam definição e algoritmo
        raw = json.loads(files[0].read_text(encoding="utf-8"))
        assert raw["identity_definition"] == "exact_matches / max(len_a, len_b)"
        assert "single_linkage" in raw["clustering_algorithm"]
