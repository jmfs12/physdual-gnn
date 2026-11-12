"""Testes de validação da Fase 2 e do orquestrador de baselines (`test_verify_phase2.py`)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def test_verify_phase2_smoke_mode():
    """Valida que scripts/verify_phase2.py --smoke executa e retorna sucesso."""
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "verify_phase2.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "--smoke"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Script falhou com stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    assert "RESULTADO FINAL: APROVADO" in result.stdout


def test_run_phase2_baseline_aggregation_and_idempotence(tmp_path: Path):
    """Valida a agregação estatística de 5 seeds e a lógica de idempotência por run."""
    from run_phase2_baseline import aggregate_runs, is_run_completed, run_single_seed

    runs_dir = tmp_path / "outputs" / "results" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    model = "graphdta_gin"
    dataset = "davis"
    split = "random"
    seeds = [0, 1, 2, 3, 4]

    ci_values = [0.880, 0.885, 0.890, 0.882, 0.888]
    mse_values = [0.235, 0.230, 0.225, 0.232, 0.228]

    for seed, ci_val, mse_val in zip(seeds, ci_values, mse_values):
        run_data = {
            "model": model,
            "dataset": dataset,
            "split": split,
            "seed": seed,
            "featurizer_version": "v2_onehot78_l1norm",
            "normalize_atom_features": True,
            "drug_in_dim": 78,
            "effective_config": {"batch_size": 512, "lr": 0.0005},
            "best_epoch": 140 + seed,
            "total_epochs_run": 190 + seed,
            "near_epoch_ceiling_warn": False,
            "elapsed_sec": 200.0 + seed * 5,
            "sec_per_epoch": 1.10,
            "val_metrics_at_best": {
                "ci": ci_val + 0.005,
                "mse": mse_val - 0.01,
                "rmse": float(np.sqrt(mse_val - 0.01)),
                "pearson": 0.84,
                "spearman": 0.70,
                "rm2": 0.69,
            },
            "test_metrics_full": {
                "ci": ci_val,
                "mse": mse_val,
                "rmse": float(np.sqrt(mse_val)),
                "pearson": 0.83,
                "spearman": 0.69,
                "rm2": 0.68,
            },
            "test_metrics_uncensored": {
                "num_uncensored_pairs": 912,
                "ci": 0.795,
                "mse": 0.410,
                "rmse": 0.640,
                "pearson": 0.760,
                "spearman": 0.745,
                "rm2": 0.540,
            },
            "test_metrics_by_seq_len": {
                "le_1000": {"count": 2650, "ci": 0.887, "mse": 0.228},
                "gt_1000": {"count": 356, "ci": 0.871, "mse": 0.262},
            },
            "trivial_baselines": {
                "mean_predictor": {"ci": 0.5, "mse": 0.815},
                "ridge_aa_composition": {"ci": 0.612, "mse": 0.710},
            },
        }
        run_file = runs_dir / f"{model}_{dataset}_{split}_seed{seed}.json"
        run_file.write_text(json.dumps(run_data, indent=2), encoding="utf-8")

    out_file = tmp_path / "outputs" / "results" / f"baseline_{model}_{dataset}_{split}.json"
    aggregated = aggregate_runs(
        model=model,
        dataset=dataset,
        split=split,
        runs_dir=runs_dir,
        output_file=out_file,
    )

    assert out_file.exists()
    assert len(aggregated["per_seed_results"]) == 5
    for s in seeds:
        assert str(s) in aggregated["per_seed_results"] or s in aggregated["per_seed_results"]

    expected_ci_mean = float(np.mean(ci_values))
    expected_ci_std = float(np.std(ci_values, ddof=1))
    actual_ci_mean = aggregated["aggregated"]["test_metrics_full"]["ci"]["mean"]
    actual_ci_std = aggregated["aggregated"]["test_metrics_full"]["ci"]["std"]
    assert np.isclose(actual_ci_mean, expected_ci_mean, atol=1e-4)
    assert np.isclose(actual_ci_std, expected_ci_std, atol=1e-4)

    # Verificação de Idempotência
    target_seed = 0
    run_file_0 = runs_dir / f"{model}_{dataset}_{split}_seed{target_seed}.json"
    assert is_run_completed(model, dataset, split, target_seed, runs_dir=runs_dir)

    original_content = run_file_0.read_text(encoding="utf-8")

    # Ao chamar run_single_seed com force=False em run já concluída, deve retornar pulada sem sobrescrever
    res_skip = run_single_seed(
        model_name=model,
        dataset_name=dataset,
        split_type=split,
        seed=target_seed,
        runs_dir=runs_dir,
        force=False,
    )
    assert res_skip.get("skipped", False) is True or res_skip.get("seed") == target_seed
    assert run_file_0.read_text(encoding="utf-8") == original_content


def test_run_phase2_baseline_kfold_aggregation(tmp_path: Path):
    """Valida a agregação K-Fold, consolidação de OOF e análise por bootstrap."""
    from run_phase2_baseline import aggregate_kfold_runs, is_run_completed

    runs_dir = tmp_path / "outputs" / "results" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    oof_dir = tmp_path / "outputs" / "results" / "oof"

    model = "graphdta_gin"
    dataset = "davis"
    split = "cold_target"
    folds = [0, 1, 2, 3, 4]

    ci_per_fold = [0.720, 0.735, 0.710, 0.740, 0.725]
    mse_per_fold = [0.750, 0.730, 0.760, 0.720, 0.745]

    for f_idx, (ci_val, mse_val) in enumerate(zip(ci_per_fold, mse_per_fold)):
        oof_preds = []
        for i in range(20):
            oof_preds.append({
                "fold": f_idx,
                "drug_id": f"D_{i % 5}",
                "target_id": f"T_{(f_idx * 20 + i) % 10}",
                "smiles": f"C{i}CC(=O)N",
                "sequence": f"MVLSP{f_idx}_{i % 4}",
                "y_true": float(5.0 + (i % 5)),
                "y_pred": float(5.1 + (i % 5) + 0.05 * (f_idx % 2)),
                "y_pred_drug_mean": float(5.5),
                "y_pred_target_mean": float(5.2),
            })

        run_data = {
            "model": model,
            "dataset": dataset,
            "split": split,
            "protocol": "kfold",
            "unit_id": f_idx,
            "fold": f_idx,
            "seed": None,
            "effective_config": {"epochs": 500, "patience": 80, "batch_size": 512, "lr": 0.0005},
            "best_epoch": 5,
            "total_epochs_run": 85,
            "near_epoch_ceiling_warn": False,
            "elapsed_sec": 120.0,
            "sec_per_epoch": 1.41,
            "test_metrics_full": {
                "ci": ci_val,
                "mse": mse_val,
                "rmse": float(np.sqrt(mse_val)),
                "pearson": 0.50,
                "spearman": 0.48,
                "rm2": 0.25,
            },
            "oof_predictions": oof_preds,
        }
        run_file = runs_dir / f"{model}_{dataset}_{split}_fold{f_idx}.json"
        run_file.write_text(json.dumps(run_data, indent=2), encoding="utf-8")

    out_file = tmp_path / "outputs" / "results" / f"baseline_{model}_{dataset}_{split}_kfold.json"
    agg = aggregate_kfold_runs(
        model=model,
        dataset=dataset,
        split=split,
        runs_dir=runs_dir,
        folds=folds,
        oof_dir=oof_dir,
        output_file=out_file,
    )

    assert out_file.exists()
    assert agg["protocol"] == "kfold"
    assert agg["num_folds"] == 5
    assert len(agg["per_fold_results"]) == 5

    # Verificação de per_fold_mean_std
    expected_ci_mean = float(np.mean(ci_per_fold))
    expected_ci_std = float(np.std(ci_per_fold, ddof=1))
    assert np.isclose(agg["per_fold_mean_std"]["ci"]["mean"], expected_ci_mean, atol=1e-4)
    assert np.isclose(agg["per_fold_mean_std"]["ci"]["std"], expected_ci_std, atol=1e-4)

    # Verificação do arquivo CSV de OOF
    oof_csv_path = Path(agg["oof_csv_path"])
    assert oof_csv_path.exists()
    df_oof = pd.read_csv(oof_csv_path)
    assert len(df_oof) == 100  # 5 dobras x 20 predições

    # Verificação de métricas globais e bootstrap
    assert "ci" in agg["global_oof_metrics"]
    assert "mse" in agg["global_oof_metrics"]
    assert "threshold_5.0" in agg["binary_classification_metrics"]
    assert "threshold_7.0" in agg["binary_classification_metrics"]
    assert "vs_drug_mean" in agg["bootstrap_analysis"]
    assert "delta_mse" in agg["bootstrap_analysis"]["vs_drug_mean"]


def test_run_phase2_baseline_extend_epochs_and_kfold_idempotence(tmp_path: Path):
    """Valida a detecção de completude com extend_epochs e no protocolo kfold."""
    from run_phase2_baseline import is_run_completed

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Cria run concluída com 500 épocas no protocolo seeds
    run_seed = {
        "model": "graphdta_gin",
        "dataset": "kiba",
        "split": "random",
        "seed": 1,
        "effective_config": {"epochs": 500},
        "test_metrics_full": {"ci": 0.854, "mse": 0.181},
    }
    (runs_dir / "graphdta_gin_kiba_random_seed1.json").write_text(
        json.dumps(run_seed, indent=2), encoding="utf-8"
    )

    # Sem extend_epochs: considera concluída
    assert is_run_completed(
        "graphdta_gin", "kiba", "random", 1, runs_dir=runs_dir, protocol="seeds"
    )

    # Com extend_epochs=True e target_epochs=1000: detecta que precisa estender
    assert not is_run_completed(
        "graphdta_gin",
        "kiba",
        "random",
        1,
        runs_dir=runs_dir,
        protocol="seeds",
        target_epochs=1000,
        extend_epochs=True,
    )

    # Com extend_epochs=True e target_epochs=500: já alcançou
    assert is_run_completed(
        "graphdta_gin",
        "kiba",
        "random",
        1,
        runs_dir=runs_dir,
        protocol="seeds",
        target_epochs=500,
        extend_epochs=True,
    )

    # Protocolo kfold
    run_fold = {
        "model": "graphdta_cnn3",
        "dataset": "davis",
        "split": "cold_drug",
        "fold": 0,
        "effective_config": {"epochs": 500},
        "test_metrics_full": {"ci": 0.695, "mse": 0.510},
    }
    (runs_dir / "graphdta_cnn3_davis_cold_drug_fold0.json").write_text(
        json.dumps(run_fold, indent=2), encoding="utf-8"
    )
    assert is_run_completed(
        "graphdta_cnn3", "davis", "cold_drug", 0, runs_dir=runs_dir, protocol="kfold"
    )

