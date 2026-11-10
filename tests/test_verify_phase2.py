"""Testes de validação da Fase 2 e do orquestrador de baselines (`test_verify_phase2.py`)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
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
