"""Testes unitários do verificador do Gate de Pronto da Fase 1 (`scripts/verify_phase1.py`)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from physdual.data.splits import create_splits, save_splits  # noqa: E402
from verify_phase1 import run_phase1_verification  # noqa: E402


def _setup_mock_phase1_workspace(tmp_path: Path) -> tuple[Path, dict[str, dict[str, int]]]:
    """Cria uma árvore de artefatos da Fase 1 em `tmp_path` com mini-datasets sintéticos."""
    processed_dir = tmp_path / "data" / "processed"
    splits_dir = processed_dir / "splits"
    logs_dir = tmp_path / "outputs" / "logs"
    esm_dir = tmp_path / "cache" / "esm_embeddings"
    splits_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    esm_dir.mkdir(parents=True, exist_ok=True)

    expected_counts = {
        "davis": {"num_drugs": 6, "num_targets": 5, "num_pairs": 30},
        "kiba": {"num_drugs": 6, "num_targets": 5, "num_pairs": 30},
    }

    for ds, counts in expected_counts.items():
        rows = []
        for d in range(counts["num_drugs"]):
            for t in range(counts["num_targets"]):
                rows.append({
                    "drug_id": f"{ds}_D{d}",
                    "smiles": f"C{'C' * d}O",
                    "smiles_raw": f"C{'C' * d}O",
                    "target_id": f"{ds}_T{t}",
                    "sequence": f"MKTVRQ{'A' * (t + 1)}",
                    "affinity": 5.0 if (ds == "davis" and d == 0) else 7.5,
                })
        df = pd.DataFrame(rows)
        df.to_csv(processed_dir / f"{ds}_clean.csv", index=False)

        ing_report = {
            "dataset_name": ds,
            "raw_num_pairs": counts["num_pairs"],
            "clean_num_pairs": counts["num_pairs"],
            "num_pairs_dropped_recorded": 0,
            "num_drugs_by_id": counts["num_drugs"],
            "num_drugs_by_canonical_smiles": counts["num_drugs"],
            "num_targets_by_id": counts["num_targets"],
            "num_targets_by_sequence": counts["num_targets"],
            "duplicate_content_pairs_count": 0,
            "duplicate_policy": "keep_both",
            "ambiguous_aa_seq_count": 0,
            "ambiguous_aa_char_counts": {"X": 0, "B": 0, "U": 0, "Z": 0, "O": 0},
            "affinity_unit_detected": "pKd" if ds == "davis" else "kiba_score",
            "davis_censored_count": 5 if ds == "davis" else 0,
            "davis_censored_pct": 16.67 if ds == "davis" else 0.0,
            "smiles_modified_count": 0,
            "smiles_failed_count": 0,
            "smiles_failures": [],
        }
        (processed_dir / f"{ds}_ingestion_report.json").write_text(
            json.dumps(ing_report, indent=2), encoding="utf-8"
        )

        for st in ("random", "cold_drug", "cold_target"):
            for seed in (0, 1, 2, 3, 4):
                sp = create_splits(df, split_type=st, seed=seed)
                save_splits(sp, splits_dir / f"{ds}_{st}_seed{seed}.json")

    summary_report = {
        "model_name": "esm2_t33_650M_UR50D",
        "esm_dim": 1280,
        "windowed_sequences": {"count": 0, "target_ids": []},
        "davis_censoring": {"censored_count": 5, "censored_pct": 16.67},
        "smiles_audit": {"modified_count": 0, "failed_count": 0, "failures": []},
        "datasets": {},
    }
    (logs_dir / "phase1_summary_report.json").write_text(
        json.dumps(summary_report, indent=2), encoding="utf-8"
    )
    return tmp_path, expected_counts


def test_verify_phase1_detects_provenance_and_splits(tmp_path: Path):
    workspace, expected_counts = _setup_mock_phase1_workspace(tmp_path)

    # Sem torch local (ou skip_pt_checks=True), valida os gates de procedência, splits e relatório final
    res = run_phase1_verification(
        root=workspace,
        expected_counts_override=expected_counts,
        skip_pt_checks=True,
    )
    assert res["passed"] is True
    assert res["checks"]["4_splits_leakage_and_coverage"]["passed"] is True
    assert res["checks"]["5_provenance_and_zero_unrecorded_loss"]["passed"] is True
    assert res["checks"]["7_summary_report_completeness"]["passed"] is True


def test_verify_phase1_fails_when_split_has_content_leakage(tmp_path: Path):
    workspace, expected_counts = _setup_mock_phase1_workspace(tmp_path)
    bad_split_path = workspace / "data" / "processed" / "splits" / "davis_cold_drug_seed0.json"
    # Corrompe o split colocando o índice 0 (mesmo fármaco D0 do índice 1) em test enquanto 1 fica em train
    bad_splits = {
        "train": list(range(1, 25)),
        "val": list(range(25, 30)),
        "test": [0],
    }
    bad_split_path.write_text(json.dumps(bad_splits), encoding="utf-8")

    res = run_phase1_verification(
        root=workspace,
        expected_counts_override=expected_counts,
        skip_pt_checks=True,
    )
    assert res["passed"] is False
    assert res["checks"]["4_splits_leakage_and_coverage"]["passed"] is False
