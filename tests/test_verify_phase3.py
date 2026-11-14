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

    full_report = run_phase3_full_verification(root=tmp_path, require_paired_win=False, expected_units=2)
    assert full_report["all_gates_passed"] is True


def test_run_phase3_seeds_cli_yaml_overrides_root_and_shared_esm_cache(tmp_path: Path, monkeypatch):
    """Valida `protocol='seeds'`, reutilização de `esm_cache`, `root=` em `run_single_unit` e CLI `--config`."""
    import copy
    import yaml
    from run_phase3_physdual import aggregate_phase3_seed_runs, main as phase3_main, run_single_unit

    monkeypatch.setenv("PHYSDUAL_ROOT", str(tmp_path))
    cfg, esm_dir = _create_synthetic_phase3_workspace(tmp_path)
    splits_dir = tmp_path / "data" / "processed" / "splits"
    split_payload = json.loads((splits_dir / "davis_cold_target_seqid30_fold0.json").read_text(encoding="utf-8"))
    (splits_dir / "davis_random_seed0.json").write_text(json.dumps(split_payload), encoding="utf-8")
    (splits_dir / "davis_random_seed1.json").write_text(json.dumps(split_payload), encoding="utf-8")

    # 1. Config customizado passado via CLI --config não deve ser sobrescrito por defaults do argparse
    custom_cfg = copy.deepcopy(cfg)
    custom_cfg["paths"] = {
        "data_dir": "data",
        "raw_data_dir": "data/raw",
        "processed_data_dir": "data/processed",
        "cache_dir": "cache",
        "esm_cache_dir": "cache/esm_embeddings",
        "outputs_dir": "outputs",
        "checkpoints_dir": "outputs/checkpoints",
        "results_dir": "outputs/results",
        "logs_dir": "outputs/logs",
    }
    custom_cfg["physdual_phase3"]["training"]["lr"] = 0.00025
    custom_cfg["physdual_phase3"]["training"]["weight_decay"] = 0.02
    custom_cfg_path = tmp_path / "custom_phase3.yaml"
    custom_cfg_path.write_text(yaml.safe_dump(custom_cfg), encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_phase3_physdual.py",
            "--protocol",
            "seeds",
            "--dataset",
            "davis",
            "--split",
            "random",
            "--seeds",
            "0,1",
            "--config",
            str(custom_cfg_path),
            "--root",
            str(tmp_path),
            "--esm-cache-dir",
            str(esm_dir),
            "--device",
            "cpu",
            "--smoke",
        ],
    )
    phase3_main()

    seed0_path = tmp_path / "outputs" / "results" / "runs" / "physdual_gin_esm_mlp_davis_random_seed0.json"
    agg_path = tmp_path / "outputs" / "results" / "phase3_physdual_gin_esm_mlp_davis_random.json"
    assert seed0_path.exists()
    assert agg_path.exists()
    seed0_data = json.loads(seed0_path.read_text(encoding="utf-8"))
    assert seed0_data["effective_config"]["lr"] == pytest.approx(0.00025)
    assert seed0_data["effective_config"]["weight_decay"] == pytest.approx(0.02)

    agg_data = aggregate_phase3_seed_runs(dataset="davis", split="random", seeds=[0, 1], config=cfg, root=tmp_path)
    assert agg_data["protocol"] == "seeds"
    assert agg_data["num_seeds"] == 2
    assert "ci" in agg_data["aggregated"]

    # 2. run_single_unit com root= explicitamente diferente de cfg['project_root'] + shared_cache
    alt_root = tmp_path / "alt_ws"
    _create_synthetic_phase3_workspace(alt_root)
    (alt_root / "data" / "processed" / "splits" / "davis_random_seed0.json").write_text(
        json.dumps(split_payload), encoding="utf-8"
    )
    shared_cache: dict = {}
    run_single_unit(
        dataset_name="davis",
        split_type="random",
        protocol="seeds",
        seed=0,
        config=cfg,  # project_root ainda aponta para tmp_path
        root=alt_root,
        esm_cache_dir=alt_root / "cache" / "esm_embeddings",
        esm_cache=shared_cache,
        device="cpu",
        smoke=True,
    )
    assert len(shared_cache) == 4
    assert (alt_root / "outputs" / "results" / "runs" / "physdual_gin_esm_mlp_davis_random_seed0.json").exists()


def test_verify_phase3_full_rejects_nan_incomplete_and_missing_baselines(tmp_path: Path, monkeypatch):
    """Garante que Gate 1 (NaN), Gate 2 (incompleto) e Gate 6 (baseline ausente) falham quando violados."""
    from run_phase3_physdual import aggregate_phase3_kfold_runs, run_single_unit
    from verify_phase3 import run_phase3_full_verification

    monkeypatch.setenv("PHYSDUAL_ROOT", str(tmp_path))
    cfg, esm_dir = _create_synthetic_phase3_workspace(tmp_path)
    for fold in (0, 1):
        run_single_unit(
            dataset_name="davis",
            split_type="cold_target_seqid30",
            protocol="kfold",
            fold=fold,
            config=cfg,
            root=tmp_path,
            esm_cache_dir=esm_dir,
            device="cpu",
            smoke=True,
        )
    aggregate_phase3_kfold_runs(dataset="davis", split="cold_target_seqid30", folds=[0, 1], config=cfg, root=tmp_path)

    # Gate 2 falha se esperar 5 dobras quando só existem 2
    rep_incomplete = run_phase3_full_verification(root=tmp_path, require_paired_win=False, expected_units=5)
    assert rep_incomplete["gates"]["gate2_aggregates_present"]["passed"] is False

    # Gate 6 falha quando baseline obrigatório é removido e require_paired_win=True
    agg_file = tmp_path / "outputs" / "results" / "phase3_physdual_gin_esm_mlp_davis_cold_target_seqid30_kfold.json"
    agg_payload = json.loads(agg_file.read_text(encoding="utf-8"))
    agg_payload["paired_vs_baselines"].pop("ridge_esm", None)
    agg_file.write_text(json.dumps(agg_payload), encoding="utf-8")
    rep_missing_b = run_phase3_full_verification(root=tmp_path, require_paired_win=True, expected_units=2)
    assert rep_missing_b["gates"]["gate6_paired_superiority"]["passed"] is False
    assert any("ridge_esm" in e for e in rep_missing_b["gates"]["gate6_paired_superiority"]["errors"])

    # Gate 1 falha se test_metrics_full contiver NaN
    run0_file = tmp_path / "outputs" / "results" / "runs" / "physdual_gin_esm_mlp_davis_cold_target_seqid30_fold0.json"
    run0_payload = json.loads(run0_file.read_text(encoding="utf-8"))
    run0_payload["test_metrics_full"]["mse"] = float("nan")
    run0_file.write_text(json.dumps(run0_payload), encoding="utf-8")
    rep_nan = run_phase3_full_verification(root=tmp_path, require_paired_win=False, expected_units=2)
    assert rep_nan["gates"]["gate1_runs_integrity"]["passed"] is False


def test_run_phase3_extend_epochs_resumes_only_unconverged_runs(tmp_path: Path, monkeypatch):
    """Garante que `--extend-epochs` retoma runs no teto via checkpoint e pula runs já convergidos."""
    from run_phase3_physdual import run_single_unit

    monkeypatch.setenv("PHYSDUAL_ROOT", str(tmp_path))
    cfg, esm_dir = _create_synthetic_phase3_workspace(tmp_path)

    # Fold 0 treina até teto=2 épocas (salva checkpoint_last.pt na época 2)
    res0 = run_single_unit(
        dataset_name="davis",
        split_type="cold_target_seqid30",
        protocol="kfold",
        fold=0,
        config=cfg,
        root=tmp_path,
        esm_cache_dir=esm_dir,
        device="cpu",
        epochs=2,
        patience=5,
        davis_cold_patience=5,
        batch_size=4,
        num_workers=0,
        smoke=False,
    )
    assert res0["total_epochs_run"] == 2
    assert len(res0["history"]) == 2

    # Fold 1 simula run que já convergiu antes do teto (2 < 5, sem near_epoch_ceiling_warn)
    run1_file = tmp_path / "outputs" / "results" / "runs" / "physdual_gin_esm_mlp_davis_cold_target_seqid30_fold1.json"
    res1 = json.loads(json.dumps(res0))
    res1["fold"] = 1
    res1["unit_id"] = 1
    res1["effective_config"]["epochs"] = 5
    res1["total_epochs_run"] = 2
    res1["near_epoch_ceiling_warn"] = False
    run1_file.write_text(json.dumps(res1), encoding="utf-8")

    # Estendendo para 4 épocas: fold 0 deve retomar de 3..4; fold 1 deve ser pulado
    res0_ext = run_single_unit(
        dataset_name="davis",
        split_type="cold_target_seqid30",
        protocol="kfold",
        fold=0,
        config=cfg,
        root=tmp_path,
        esm_cache_dir=esm_dir,
        device="cpu",
        epochs=4,
        patience=10,
        davis_cold_patience=10,
        batch_size=4,
        num_workers=0,
        smoke=False,
        extend_epochs=True,
    )
    assert res0_ext["total_epochs_run"] == 4
    assert len(res0_ext["history"]) == 4

    res1_ext = run_single_unit(
        dataset_name="davis",
        split_type="cold_target_seqid30",
        protocol="kfold",
        fold=1,
        config=cfg,
        root=tmp_path,
        esm_cache_dir=esm_dir,
        device="cpu",
        epochs=4,
        patience=10,
        davis_cold_patience=10,
        batch_size=4,
        num_workers=0,
        smoke=False,
        extend_epochs=True,
    )
    assert res1_ext["total_epochs_run"] == 2
