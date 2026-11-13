"""Testes de integração para `scripts/run_phase3_physdual.py` e `scripts/verify_phase3.py`."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from physdual.data.esm_cache import save_protein_embedding  # noqa: E402
from physdual.data.featurizer_drug import smiles_to_graph_dict  # noqa: E402
from physdual.utils.config import load_config  # noqa: E402


def _create_synthetic_phase3_workspace(tmp_path: Path) -> tuple[dict, Path]:
    proc_dir = tmp_path / "data" / "processed"
    splits_dir = proc_dir / "splits"
    esm_dir = tmp_path / "cache" / "esm_embeddings"
    res_dir = tmp_path / "outputs" / "results"
    splits_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    smiles_list = ["CCO", "CCN", "CCC", "CCCC"]
    seqs_list = ["MKTVRQ", "ACDEFG", "GHIKLM", "NPQRST"]
    rows = []
    for di, smi in enumerate(smiles_list):
        for ti, seq in enumerate(seqs_list):
            rows.append({
                "drug_id": f"D{di}",
                "smiles": smi,
                "target_id": f"T{ti}",
                "sequence": seq,
                "affinity": float(5.0 + 0.8 * di + 0.4 * ti),
            })
    df = pd.DataFrame(rows)  # 16 linhas
    df.to_csv(proc_dir / "davis_clean.csv", index=False)

    drug_cache = {
        smi: smiles_to_graph_dict(smi, drug_id=f"D{i}", featurizer_version="v2_onehot78_l1norm")
        for i, smi in enumerate(smiles_list)
    }
    torch.save(drug_cache, proc_dir / "davis_drug_graphs_v2_onehot78_l1norm.pt")

    for idx, seq in enumerate(seqs_list):
        emb = torch.randn(len(seq), 16, dtype=torch.float32) + idx * 0.5
        save_protein_embedding(seq, emb, model_name="esm2_test", cache_dir=esm_dir, windowed=False)

    # 2 dobras disjuntas em seqs_list
    split0 = {"train": [0, 1, 4, 5, 8, 9, 12, 13], "val": [2, 6, 10, 14], "test": [3, 7, 11, 15]}
    split1 = {"train": [2, 3, 6, 7, 10, 11, 14, 15], "val": [0, 4, 8, 12], "test": [1, 5, 9, 13]}
    (splits_dir / "davis_cold_target_seqid30_fold0.json").write_text(json.dumps(split0), encoding="utf-8")
    (splits_dir / "davis_cold_target_seqid30_fold1.json").write_text(json.dumps(split1), encoding="utf-8")

    # Mock de phase3_esm_cpu_baselines.json e baseline Phase 2 para validar comparação pareada
    esm_cpu = {
        "regimes": {
            "davis_seqid30_kfold": {
                "per_unit": [
                    {
                        "fold": 0,
                        "knn_esm_target": {"metrics": {"ci": 0.70, "mse": 0.90}},
                        "ridge_esm_target_only": {"metrics": {"ci": 0.55, "mse": 1.20}},
                        "ridge_esm": {"metrics": {"ci": 0.72, "mse": 0.85}},
                    },
                    {
                        "fold": 1,
                        "knn_esm_target": {"metrics": {"ci": 0.68, "mse": 0.95}},
                        "ridge_esm_target_only": {"metrics": {"ci": 0.54, "mse": 1.25}},
                        "ridge_esm": {"metrics": {"ci": 0.71, "mse": 0.88}},
                    },
                ],
                "mean_std": {
                    "knn_esm_target": {"ci": {"mean": 0.69}, "mse": {"mean": 0.925}},
                    "ridge_esm_target_only": {"ci": {"mean": 0.545}, "mse": {"mean": 1.225}},
                    "ridge_esm": {"ci": {"mean": 0.715}, "mse": {"mean": 0.865}},
                },
            }
        }
    }
    (res_dir / "phase3_esm_cpu_baselines.json").write_text(json.dumps(esm_cpu), encoding="utf-8")

    cnn3_mock = {
        "per_fold_results": {
            "0": {"test_metrics_full": {"ci": 0.71, "mse": 0.89}},
            "1": {"test_metrics_full": {"ci": 0.70, "mse": 0.91}},
        },
        "per_fold_mean_std": {"ci": {"mean": 0.705}, "mse": {"mean": 0.90}},
    }
    (res_dir / "baseline_graphdta_cnn3_davis_cold_target_seqid30_kfold.json").write_text(
        json.dumps(cnn3_mock), encoding="utf-8"
    )

    cfg = load_config(root=tmp_path)
    cfg["project_root"] = str(tmp_path)
    cfg["esm"]["model_name"] = "esm2_test"
    cfg["esm"]["esm_dim"] = 16
    cfg["physdual_phase3"]["esm_dim"] = 16
    cfg["physdual_phase3"]["prot_output_dim"] = 8
    cfg["physdual_phase3"]["mlp_hidden_dims"] = [32, 16]
    cfg["physdual_phase3"]["training"]["batch_size"] = 4
    cfg["physdual_phase3"]["training"]["num_workers"] = 0
    cfg["physdual_phase3"]["training"]["pin_memory"] = False
    return cfg, esm_dir


def test_run_phase3_smoke_end_to_end_kfold_and_paired_comparison(tmp_path: Path, monkeypatch):
    """Executa 2 dobras em modo smoke e verifica agregação, OOF e comparação pareada."""
    from run_phase3_physdual import aggregate_phase3_kfold_runs, run_single_unit

    monkeypatch.setenv("PHYSDUAL_ROOT", str(tmp_path))
    cfg, esm_dir = _create_synthetic_phase3_workspace(tmp_path)

    for fold in (0, 1):
        res = run_single_unit(
            dataset_name="davis",
            split_type="cold_target_seqid30",
            protocol="kfold",
            fold=fold,
            config=cfg,
            esm_cache_dir=esm_dir,
            device="cpu",
            smoke=True,
        )
        assert res["test_isolation_verified"] is True
        assert res["optimizer_name"] == "adamw"
        assert "history" in res and len(res["history"]) == 1
        assert "train_val_mse_gap_at_best" in res
        assert "early_memorization_warn" in res
        assert "marginal_baselines" in res
        assert "knn_baselines" in res

    agg = aggregate_phase3_kfold_runs(
        dataset="davis",
        split="cold_target_seqid30",
        folds=[0, 1],
        config=cfg,
        root=tmp_path,
    )
    assert agg["protocol"] == "kfold"
    assert agg["num_folds"] == 2
    assert agg["test_isolation_verified"] is True
    assert Path(agg["oof_csv_path"]).exists()
    paired = agg.get("paired_vs_baselines", {})
    for b_key in ("drug_mean_predictor", "knn_kmer_target", "knn_esm_target", "ridge_esm", "graphdta_cnn3"):
        assert b_key in paired
        assert "delta_mse_mean" in paired[b_key]
        assert "delta_ci_mean" in paired[b_key]
        assert "wins_mse" in paired[b_key]


def test_verify_phase3_smoke_and_full_gates(tmp_path: Path, monkeypatch):
    """Valida `verify_phase3.py` nos modos `--smoke` e `--full`."""
    from run_phase3_physdual import aggregate_phase3_kfold_runs, run_single_unit
    from verify_phase3 import run_phase3_full_verification, run_phase3_smoke_verification

    smoke_report = run_phase3_smoke_verification()
    assert smoke_report["all_checks_passed"] is True

    monkeypatch.setenv("PHYSDUAL_ROOT", str(tmp_path))
    cfg, esm_dir = _create_synthetic_phase3_workspace(tmp_path)
    for fold in (0, 1):
        run_single_unit(
            dataset_name="davis",
            split_type="cold_target_seqid30",
            protocol="kfold",
            fold=fold,
            config=cfg,
            esm_cache_dir=esm_dir,
            device="cpu",
            smoke=True,
        )
    aggregate_phase3_kfold_runs(
        dataset="davis",
        split="cold_target_seqid30",
        folds=[0, 1],
        config=cfg,
        root=tmp_path,
    )

    full_report = run_phase3_full_verification(root=tmp_path, require_paired_win=False)
    assert full_report["all_gates_passed"] is True
