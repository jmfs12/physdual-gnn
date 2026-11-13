#!/usr/bin/env python3
"""Gera e audita os splits 5-fold por identidade de sequência para o Davis (`seqid30` e `seqid40`).

Saídas geradas em `PHYSDUAL_ROOT`:
- `data/processed/splits/davis_cold_target_seqid30_fold{0..4}.json`
- `data/processed/splits/davis_cold_target_seqid40_fold{0..4}.json`
- `outputs/logs/davis_seqid_splits_summary.json`
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from physdual.data.splits import (  # noqa: E402
    create_sequence_identity_kfold_splits,
    save_splits,
    verify_group_kfold_coverage,
)
from physdual.utils.paths import get_project_root  # noqa: E402


def main() -> int:
    root = get_project_root()
    csv_path = root / "data" / "processed" / "davis_clean.csv"
    if not csv_path.exists():
        print(f"ERRO: arquivo não encontrado: {csv_path}", file=sys.stderr)
        return 2

    df = pd.read_csv(csv_path)
    splits_dir = root / "data" / "processed" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    thresholds = (0.30, 0.40)
    by_tau = create_sequence_identity_kfold_splits(
        df=df,
        identity_thresholds=thresholds,
        n_splits=5,
        group_col="sequence",
    )

    summary: dict[str, Any] = {
        "dataset": "davis",
        "num_rows": int(len(df)),
        "num_unique_sequences": int(df["sequence"].nunique()),
        "thresholds": {},
    }

    for tau, folds in by_tau.items():
        pct = int(round(tau * 100))
        split_name = f"cold_target_seqid{pct}"
        if not verify_group_kfold_coverage(df, folds, split_type=split_name, group_col="sequence"):
            raise RuntimeError(f"Falha de verificação GroupKFold em {split_name}!")

        fold_files: list[str] = []
        fold_stats: list[dict[str, Any]] = []
        for fold_dict in folds:
            fname = f"davis_{split_name}_fold{fold_dict['fold_id']}.json"
            out_path = splits_dir / fname
            save_splits(fold_dict, out_path)
            fold_files.append(fname)
            fold_stats.append({
                "fold_id": int(fold_dict["fold_id"]),
                "train_pairs": len(fold_dict["train"]),
                "val_pairs": len(fold_dict["val"]),
                "test_pairs": len(fold_dict["test"]),
                "train_unique_seqs": int(fold_dict["num_train_groups"]),
                "val_unique_seqs": int(fold_dict["num_val_groups"]),
                "test_unique_seqs": int(fold_dict["num_test_groups"]),
                "train_target_ids": int(fold_dict["num_train_target_ids"]),
                "val_target_ids": int(fold_dict["num_val_target_ids"]),
                "test_target_ids": int(fold_dict["num_test_target_ids"]),
                "test_clusters": int(fold_dict["fold_test_clusters"]),
            })

        summary["thresholds"][split_name] = {
            "identity_threshold": tau,
            "identity_definition": folds[0]["identity_definition"],
            "alignment_tool": folds[0]["alignment_tool"],
            "clustering_algorithm": folds[0]["clustering_algorithm"],
            "num_clusters_total": int(folds[0]["num_clusters_total"]),
            "singleton_clusters": int(folds[0]["singleton_clusters"]),
            "largest_cluster_unique_seqs": int(folds[0]["largest_cluster_unique_seqs"]),
            "largest_cluster_pairs": int(folds[0]["largest_cluster_pairs"]),
            "files": fold_files,
            "folds": fold_stats,
        }

    log_path = root / "outputs" / "logs" / "davis_seqid_splits_summary.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
