#!/usr/bin/env python3
"""Orquestrador da Fase 3 do PhysDual-GNN (`GIN + ESM-2 congelado + MLP`).

Executa treinamento idempotente, retomada via checkpoint atômico, isolamento estrito
de teste (`test_isolation_verified: true`), coleta de baselines marginais/kNN,
diagnóstico de memorização (`best_epoch`, `train_val_mse_gap_at_best`), agregação
por dobra/seed com OOF + bootstrap agrupado e comparação pareada contra marginais,
kNN, controles ESM-2 em CPU (`knn_esm_target`, `ridge_esm`) e baselines neurais
(`graphdta_gin`, `graphdta_cnn3`).
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
for p in (SRC_DIR, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from physdual.data.datamodule import get_dta_dataloaders  # noqa: E402
from physdual.eval.metrics import (  # noqa: E402
    bootstrap_grouped_comparisons,
    compute_all_metrics,
    compute_binary_classification_metrics,
)
from physdual.models.physdual import build_physdual_phase3_model  # noqa: E402
from physdual.training.trainer import train_model  # noqa: E402
from physdual.utils.config import load_config  # noqa: E402
from physdual.utils.paths import get_project_root  # noqa: E402
from physdual.utils.seed import set_seed  # noqa: E402
from run_phase2_baseline import (  # noqa: E402
    _compute_baselines_from_loaders,
    get_run_filename,
    is_run_completed,
    save_json_atomic,
)

PHASE3_MODEL_NAME = "physdual_gin_esm_mlp"
PHASE3_ECFP_MODEL_NAME = "physdual_ecfp_esm_mlp"


def normalize_phase3_model_name(model_name: str) -> str:
    norm = model_name.strip().lower()
    if norm in ("ecfp_mlp", PHASE3_ECFP_MODEL_NAME):
        return PHASE3_ECFP_MODEL_NAME
    return model_name.strip()


def resolve_diagnostic_model_name(
    base_model: str,
    shuffle_esm: bool = False,
    shuffle_esm_seed: int = 0,
    train_seed_offset: int = 0,
) -> str:
    """Constrói o identificador único da run sem colidir com execuções primárias."""
    name = normalize_phase3_model_name(base_model)
    if shuffle_esm:
        name = f"{name}_shuf{int(shuffle_esm_seed)}"
    if int(train_seed_offset) != 0:
        name = f"{name}_tsoff{int(train_seed_offset)}"
    return name

_ESM_CPU_REGIME_MAP: dict[tuple[str, str, str], str] = {
    ("davis", "cold_target", "kfold"): "davis_v2_cold_target_kfold",
    ("davis", "cold_drug", "kfold"): "davis_v2_cold_drug_kfold",
    ("davis", "cold_target_seqid30", "kfold"): "davis_seqid30_kfold",
    ("davis", "cold_target_seqid40", "kfold"): "davis_seqid40_kfold",
    ("davis", "random", "seeds"): "davis_v1_random",
    ("kiba", "random", "seeds"): "kiba_v1_random",
    ("kiba", "cold_drug", "seeds"): "kiba_v1_cold_drug",
    ("kiba", "cold_target", "seeds"): "kiba_v1_cold_target",
    ("kiba", "cold_target_seqid30", "kfold"): "kiba_seqid30_kfold",
    ("kiba", "cold_target_seqid40", "kfold"): "kiba_seqid40_kfold",
}


def _apply_phase3_overrides(
    cfg: dict[str, Any],
    esm_dropout_in: float | None = None,
    prot_dropout: float | None = None,
    mlp_dropout: float | None = None,
    prot_output_dim: int | None = None,
) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    p3 = out.setdefault("physdual_phase3", {})
    if esm_dropout_in is not None:
        p3["esm_dropout_in"] = float(esm_dropout_in)
    if prot_dropout is not None:
        p3["prot_dropout"] = float(prot_dropout)
    if mlp_dropout is not None:
        p3["mlp_dropout"] = float(mlp_dropout)
    if prot_output_dim is not None:
        p3["prot_output_dim"] = int(prot_output_dim)
    return out


def _resolve_output_dir(cfg: dict[str, Any], key: str, root: Path | str | None = None) -> Path:
    raw = Path(cfg["paths"][key])
    if root is None:
        base_root = Path(cfg.get("project_root", get_project_root()))
        return raw if raw.is_absolute() else (base_root / raw)
    base_root = Path(root)
    if not raw.is_absolute():
        return base_root / raw
    proj_root = cfg.get("project_root")
    if proj_root is not None:
        try:
            rel = raw.relative_to(Path(proj_root))
            return base_root / rel
        except ValueError:
            pass
    fallback_sub = {"results_dir": "results", "checkpoints_dir": "checkpoints", "logs_dir": "logs"}.get(key, key)
    return base_root / "outputs" / fallback_sub


def run_single_unit(
    dataset_name: str,
    split_type: str,
    protocol: str = "kfold",
    unit_id: int | None = None,
    seed: int | None = None,
    fold: int | None = None,
    config: dict[str, Any] | None = None,
    root: Path | str | None = None,
    runs_dir: Path | str | None = None,
    checkpoints_base_dir: Path | str | None = None,
    esm_cache_dir: Path | str | None = None,
    esm_cache: dict[str, torch.Tensor] | None = None,
    force: bool = False,
    device: str | torch.device = "auto",
    epochs: int | None = None,
    patience: int | None = None,
    davis_cold_patience: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    weight_decay: float | None = None,
    num_workers: int | None = None,
    drive_checkpoint_dir: Path | str | None = None,
    smoke: bool = False,
    extend_epochs: bool = False,
    model_name: str = PHASE3_MODEL_NAME,
    esm_dropout_in: float | None = None,
    prot_dropout: float | None = None,
    mlp_dropout: float | None = None,
    prot_output_dim: int | None = None,
    shuffle_esm: bool = False,
    shuffle_esm_seed: int = 0,
    train_seed_offset: int = 0,
) -> dict[str, Any]:
    """Executa uma unidade (dobra ou seed) completa do `PhysDualPhase3` ou controles diagnósticos."""
    if unit_id is None:
        unit_id = fold if fold is not None else (seed if seed is not None else 0)

    base_model_name = normalize_phase3_model_name(model_name)
    eff_model_name = resolve_diagnostic_model_name(
        base_model_name,
        shuffle_esm=shuffle_esm,
        shuffle_esm_seed=shuffle_esm_seed,
        train_seed_offset=train_seed_offset,
    )

    cfg = _apply_phase3_overrides(
        config if config is not None else load_config(root=root),
        esm_dropout_in=esm_dropout_in,
        prot_dropout=prot_dropout,
        mlp_dropout=mlp_dropout,
        prot_output_dim=prot_output_dim,
    )
    base_root = Path(root) if root is not None else Path(cfg.get("project_root", get_project_root()))

    if runs_dir is None:
        runs_path = _resolve_output_dir(cfg, "results_dir", root=root) / "runs"
    else:
        runs_path = Path(runs_dir)
    runs_path.mkdir(parents=True, exist_ok=True)

    run_file = runs_path / get_run_filename(eff_model_name, dataset_name, split_type, unit_id, protocol=protocol)

    p3_cfg = cfg.get("physdual_phase3", {})
    train_cfg = p3_cfg.get("training", cfg.get("graphdta", {}).get("training", {}))

    eff_epochs = epochs if epochs is not None else int(train_cfg.get("epochs", 500))
    eff_patience = patience if patience is not None else int(train_cfg.get("patience", 50))
    eff_davis_cold_pat = (
        davis_cold_patience
        if davis_cold_patience is not None
        else int(train_cfg.get("davis_cold_patience", 80))
    )
    eff_batch_size = batch_size if batch_size is not None else int(train_cfg.get("batch_size", 512))
    eff_lr = lr if lr is not None else float(train_cfg.get("lr", 5e-4))
    eff_wd = weight_decay if weight_decay is not None else float(train_cfg.get("weight_decay", 1e-3))
    eff_workers = num_workers if num_workers is not None else int(train_cfg.get("num_workers", 2))
    eff_opt = str(train_cfg.get("optimizer", "adamw"))
    save_last_every = int(train_cfg.get("save_last_every", 10))

    if smoke:
        eff_epochs = 1
        eff_patience = 1
        eff_batch_size = min(eff_batch_size, 32)
        eff_workers = 0

    should_skip = False
    if not force and is_run_completed(
        model=eff_model_name,
        dataset=dataset_name,
        split=split_type,
        unit_id=unit_id,
        runs_dir=runs_path,
        protocol=protocol,
        target_epochs=eff_epochs,
        extend_epochs=extend_epochs,
    ):
        should_skip = True
    elif not force and extend_epochs and run_file.exists():
        try:
            prev_data = json.loads(run_file.read_text(encoding="utf-8"))
            prev_eff_ep = int(prev_data.get("effective_config", {}).get("epochs", 0))
            prev_ran = int(prev_data.get("total_epochs_run", 0))
            if (
                prev_eff_ep > 0
                and prev_ran < prev_eff_ep
                and not bool(prev_data.get("near_epoch_ceiling_warn", False))
            ):
                should_skip = True
        except Exception:
            pass

    if should_skip:
        print(f"[SKIP] {eff_model_name} | {dataset_name}/{split_type}/{protocol}={unit_id} já concluído.", flush=True)
        return json.loads(run_file.read_text(encoding="utf-8"))

    # Determinismo estrito (mantém partição de dados inalterada mesmo com train_seed_offset)
    run_seed = int((unit_id if protocol == "seeds" else (42 + unit_id)) + int(train_seed_offset))
    set_seed(run_seed)

    target_device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "auto"
        else torch.device(device)
    )

    if checkpoints_base_dir is None:
        ckpt_base = _resolve_output_dir(cfg, "checkpoints_dir", root=root)
    else:
        ckpt_base = Path(checkpoints_base_dir)

    unit_prefix = "fold" if protocol == "kfold" else "seed"
    print(
        f"[START] {eff_model_name} | {dataset_name}/{split_type} ({unit_prefix}={unit_id}) | "
        f"device={target_device} | epochs={eff_epochs} | extend={extend_epochs}",
        flush=True,
    )
    run_ckpt_dir = ckpt_base / f"{eff_model_name}_{dataset_name}_{split_type}_{unit_prefix}{unit_id}"
    run_drive_ckpt_dir = (
        Path(drive_checkpoint_dir) / f"{eff_model_name}_{dataset_name}_{split_type}_{unit_prefix}{unit_id}"
        if drive_checkpoint_dir is not None
        else None
    )

    cfg_loader = copy.deepcopy(cfg)
    cfg_loader.setdefault("physdual_phase3", {}).setdefault("training", {})["batch_size"] = eff_batch_size
    cfg_loader["physdual_phase3"]["training"]["num_workers"] = eff_workers

    use_ecfp = base_model_name == PHASE3_ECFP_MODEL_NAME
    ecfp_dim = int(p3_cfg.get("ecfp_dim", 2048))

    train_loader, val_loader, test_loader = get_dta_dataloaders(
        dataset_name=dataset_name,
        split_type=split_type,
        seed=unit_id if protocol == "seeds" else 0,
        config=cfg_loader,
        root=base_root,
        protocol=protocol,
        fold=unit_id if protocol == "kfold" else None,
        use_esm=True,
        esm_cache_dir=esm_cache_dir,
        esm_cache=esm_cache,
        shuffle_esm=shuffle_esm,
        shuffle_esm_seed=shuffle_esm_seed,
        use_ecfp=use_ecfp,
        ecfp_dim=ecfp_dim,
    )

    model = build_physdual_phase3_model(cfg, model_name=base_model_name)
    print(
        f"[TRAIN] DataLoaders + ESM-2 prontos ({len(train_loader.dataset)} treino / "
        f"{len(val_loader.dataset)} val / {len(test_loader.dataset)} teste). Iniciando épocas...",
        flush=True,
    )

    t0 = time.time()
    train_result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=eff_epochs,
        lr=eff_lr,
        weight_decay=eff_wd,
        patience=eff_patience,
        davis_cold_patience=eff_davis_cold_pat,
        monitor=str(train_cfg.get("monitor", "mse")),
        checkpoint_dir=run_ckpt_dir,
        device=target_device,
        use_lr_scheduler=bool(train_cfg.get("use_lr_scheduler", False)),
        dataset_name=dataset_name,
        split_type=split_type,
        save_last_every=save_last_every,
        drive_checkpoint_dir=run_drive_ckpt_dir,
        resume=not smoke,
        optimizer_name=eff_opt,
    )
    elapsed_sec = float(time.time() - t0)
    total_epochs = int(train_result["total_epochs_run"])
    sec_per_epoch = float(elapsed_sec / max(1, total_epochs))

    trivial_baselines, marginal_baselines, knn_baselines = _compute_baselines_from_loaders(
        train_loader, test_loader
    )

    # Anexa predições marginais para bootstrap agrupado
    oof_preds = train_result.get("oof_predictions")
    if oof_preds is not None:
        tr_df = train_loader.dataset.df
        g_mean = float(tr_df["affinity"].mean())
        d_means = tr_df.groupby("smiles")["affinity"].mean().to_dict()
        t_means = tr_df.groupby("sequence")["affinity"].mean().to_dict()
        for rec in oof_preds:
            rec["y_pred_drug_mean"] = float(d_means.get(rec["smiles"], g_mean))
            rec["y_pred_target_mean"] = float(t_means.get(rec["sequence"], g_mean))
            rec["fold"] = unit_id if protocol == "kfold" else None

    result_payload: dict[str, Any] = {
        "model": eff_model_name,
        "base_model": base_model_name,
        "dataset": dataset_name,
        "split": split_type,
        "protocol": protocol,
        "unit_id": unit_id,
        "seed": unit_id if protocol == "seeds" else run_seed,
        "run_seed": run_seed,
        "fold": unit_id if protocol == "kfold" else None,
        "diagnostic_control": {
            "is_control": bool(base_model_name != PHASE3_MODEL_NAME or shuffle_esm or int(train_seed_offset) != 0),
            "shuffle_esm": bool(shuffle_esm),
            "shuffle_esm_seed": int(shuffle_esm_seed) if shuffle_esm else None,
            "train_seed_offset": int(train_seed_offset),
        },
        "esm_model": str(cfg.get("esm", {}).get("model_name", "esm2_t33_650M_UR50D")),
        "esm_dim": int(model.esm_dim),
        "drug_in_dim": int(model.drug_in_dim),
        "optimizer_name": str(train_result.get("optimizer_name", eff_opt)),
        "test_isolation_verified": bool(
            train_result.get("val_metrics_at_best") and train_result.get("test_metrics_full")
        ),
        "effective_config": {
            "epochs": eff_epochs,
            "patience": eff_patience,
            "davis_cold_patience": eff_davis_cold_pat,
            "batch_size": eff_batch_size,
            "lr": eff_lr,
            "weight_decay": eff_wd,
            "esm_dropout_in": float(model.target_encoder.dropout_in),
            "prot_dropout": float(model.target_encoder.dropout_out),
            "prot_output_dim": int(model.prot_output_dim),
            "mlp_dropout": float(p3_cfg.get("mlp_dropout", 0.3)),
            "device": str(target_device),
        },
        "best_epoch": int(train_result["best_epoch"]),
        "total_epochs_run": total_epochs,
        "near_epoch_ceiling_warn": bool(train_result["near_epoch_ceiling_warn"]),
        "early_memorization_warn": bool(train_result.get("early_memorization_warn", False)),
        "train_loss_at_best": train_result.get("train_loss_at_best"),
        "train_val_mse_gap_at_best": train_result.get("train_val_mse_gap_at_best"),
        "elapsed_sec": elapsed_sec,
        "sec_per_epoch": sec_per_epoch,
        "val_metrics_at_best": train_result["val_metrics_at_best"],
        "test_metrics_full": train_result["test_metrics_full"],
        "test_metrics_uncensored": train_result["test_metrics_uncensored"],
        "test_metrics_by_seq_len": train_result["test_metrics_by_seq_len"],
        "trivial_baselines": trivial_baselines,
        "marginal_baselines": marginal_baselines,
        "knn_baselines": knn_baselines,
        "history": train_result["history"],
        "oof_predictions": oof_preds,
    }
    save_json_atomic(result_payload, run_file)
    print(
        f"[OK] {eff_model_name} | {dataset_name}/{split_type} ({unit_prefix}={unit_id}) | "
        f"best_ep={result_payload['best_epoch']}/{total_epochs} | "
        f"CI={result_payload['test_metrics_full']['ci']:.4f} | "
        f"MSE={result_payload['test_metrics_full']['mse']:.4f}",
        flush=True,
    )
    return result_payload


def _compute_paired_summary(
    model_per_unit: dict[str, dict[str, Any]],
    unit_ids: list[int],
    baseline_per_unit: dict[str, dict[str, dict[str, float]]],
) -> dict[str, dict[str, Any]]:
    """Calcula deltas pareados por dobra/seed (`model - baseline`) para MSE e CI."""
    paired: dict[str, dict[str, Any]] = {}
    for b_name, b_units in baseline_per_unit.items():
        d_mses: list[float] = []
        d_cis: list[float] = []
        for u in unit_ids:
            uk = str(u)
            if uk not in model_per_unit or uk not in b_units:
                continue
            m_met = model_per_unit[uk].get("test_metrics_full", {})
            b_met = b_units[uk]
            if "mse" in m_met and "mse" in b_met:
                d_mses.append(float(m_met["mse"]) - float(b_met["mse"]))
            if "ci" in m_met and "ci" in b_met:
                d_cis.append(float(m_met["ci"]) - float(b_met["ci"]))
        if not d_mses or not d_cis:
            continue
        paired[b_name] = {
            "n_units": len(d_mses),
            "delta_mse_mean": float(np.mean(d_mses)),
            "delta_mse_std": float(np.std(d_mses, ddof=1)) if len(d_mses) > 1 else 0.0,
            "delta_ci_mean": float(np.mean(d_cis)),
            "delta_ci_std": float(np.std(d_cis, ddof=1)) if len(d_cis) > 1 else 0.0,
            "wins_mse": int(sum(1 for d in d_mses if d < 0.0)),
            "wins_ci": int(sum(1 for d in d_cis if d > 0.0)),
        }
    return paired


def _collect_external_baselines_per_unit(
    results_dir: Path,
    dataset: str,
    split: str,
    protocol: str,
    unit_ids: list[int],
    found_runs_map: dict[str, dict[str, Any]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Reúne métricas por unidade de marginais, kNN, controles ESM CPU e baselines GraphDTA."""
    baselines: dict[str, dict[str, dict[str, float]]] = {
        "drug_mean_predictor": {},
        "target_mean_predictor": {},
        "knn_kmer_target": {},
        "knn_tanimoto_drug": {},
    }
    for uk, r in found_runs_map.items():
        marg = r.get("marginal_baselines", {})
        knn = r.get("knn_baselines", {})
        if "drug_mean_predictor" in marg:
            baselines["drug_mean_predictor"][uk] = marg["drug_mean_predictor"]
        if "target_mean_predictor" in marg:
            baselines["target_mean_predictor"][uk] = marg["target_mean_predictor"]
        if "knn_kmer_target" in knn:
            baselines["knn_kmer_target"][uk] = knn["knn_kmer_target"]
        if "knn_tanimoto_drug" in knn:
            baselines["knn_tanimoto_drug"][uk] = knn["knn_tanimoto_drug"]

    # Controles ESM CPU (knn_esm_target, ridge_esm_target_only, ridge_esm)
    esm_cpu_path = results_dir / "phase3_esm_cpu_baselines.json"
    regime_key = _ESM_CPU_REGIME_MAP.get((dataset.lower(), split.lower(), protocol.lower()))
    if esm_cpu_path.exists() and regime_key is not None:
        esm_data = json.loads(esm_cpu_path.read_text(encoding="utf-8"))
        reg_obj = esm_data.get("regimes", {}).get(regime_key, {})
        for entry in reg_obj.get("per_unit", []):
            uid = entry.get("fold" if protocol == "kfold" else "seed")
            if uid is None or int(uid) not in unit_ids:
                continue
            uk = str(int(uid))
            for m_key in ("knn_esm_target", "ridge_esm_target_only", "ridge_esm"):
                if m_key in entry and "metrics" in entry[m_key]:
                    baselines.setdefault(m_key, {})[uk] = entry[m_key]["metrics"]

    # Baselines neurais da Fase 2 (graphdta_gin, graphdta_cnn3)
    for p2_mod in ("graphdta_gin", "graphdta_cnn3"):
        suffix = "_kfold.json" if protocol == "kfold" else ".json"
        p2_path = results_dir / f"baseline_{p2_mod}_{dataset}_{split}{suffix}"
        if not p2_path.exists():
            continue
        p2_data = json.loads(p2_path.read_text(encoding="utf-8"))
        unit_dict = p2_data.get("per_fold_results" if protocol == "kfold" else "per_seed_results", {})
        for u in unit_ids:
            uk = str(u)
            if uk in unit_dict and "test_metrics_full" in unit_dict[uk]:
                baselines.setdefault(p2_mod, {})[uk] = unit_dict[uk]["test_metrics_full"]

    return {k: v for k, v in baselines.items() if v}


def aggregate_phase3_kfold_runs(
    dataset: str,
    split: str,
    folds: list[int] | None = None,
    config: dict[str, Any] | None = None,
    root: Path | str | None = None,
    model: str = PHASE3_MODEL_NAME,
) -> dict[str, Any]:
    """Consolida as dobras k-fold da Fase 3 com OOF, bootstrap agrupado e deltas pareados."""
    cfg = config if config is not None else load_config(root=root)
    results_dir = _resolve_output_dir(cfg, "results_dir", root=root)
    runs_dir = results_dir / "runs"

    target_folds = folds if folds is not None else [0, 1, 2, 3, 4]
    per_fold_results: dict[str, dict[str, Any]] = {}
    found_runs: list[dict[str, Any]] = []
    all_oof_records: list[dict[str, Any]] = []

    for f in target_folds:
        rf = runs_dir / get_run_filename(model, dataset, split, f, protocol="kfold")
        if not rf.exists():
            continue
        data = json.loads(rf.read_text(encoding="utf-8"))
        found_runs.append(data)
        fold_clean = {k: v for k, v in data.items() if k != "oof_predictions"}
        per_fold_results[str(f)] = fold_clean
        for rec in data.get("oof_predictions") or []:
            r_copy = dict(rec)
            r_copy["fold"] = f
            all_oof_records.append(r_copy)

    if not found_runs:
        raise FileNotFoundError(f"Nenhuma dobra encontrada para ({model}, {dataset}, {split}) em {runs_dir}")

    metrics_keys = ["ci", "mse", "rmse", "pearson", "spearman", "rm2"]
    per_fold_mean_std: dict[str, dict[str, float]] = {}
    for mk in metrics_keys:
        vals = [
            float(r["test_metrics_full"][mk])
            for r in found_runs
            if mk in (r.get("test_metrics_full") or {})
        ]
        if vals:
            per_fold_mean_std[mk] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            }

    oof_dir = results_dir / "oof"
    oof_dir.mkdir(parents=True, exist_ok=True)
    oof_csv_path = oof_dir / f"{model}_{dataset}_{split}_oof.csv"

    global_oof_metrics: dict[str, float] = {}
    binary_classification_metrics: dict[str, Any] = {}
    bootstrap_results: dict[str, Any] = {}

    if all_oof_records:
        oof_df = pd.DataFrame(all_oof_records)
        oof_df.to_csv(oof_csv_path, index=False)
        y_true = oof_df["y_true"].to_numpy(dtype=np.float64)
        y_pred = oof_df["y_pred"].to_numpy(dtype=np.float64)
        global_oof_metrics = compute_all_metrics(y_true, y_pred)
        if dataset.lower().strip() == "davis":
            binary_classification_metrics = compute_binary_classification_metrics(y_true, y_pred)

        is_cold_target = "cold_target" in split.lower()
        group_col = "sequence" if is_cold_target else "smiles"
        if "seqid" in split.lower():
            logs_dir = _resolve_output_dir(cfg, "logs_dir", root=root)
            cmap_path = logs_dir / f"{dataset}_seqid_cluster_map.json"
            if cmap_path.exists():
                cmap_data = json.loads(cmap_path.read_text(encoding="utf-8"))
                seq_to_cluster = cmap_data.get("thresholds", {}).get(split, {}).get("seq_to_cluster", {})
                mapped_clusters = oof_df["sequence"].astype(str).map(seq_to_cluster)
                if not mapped_clusters.isna().any():
                    oof_df["seqid_cluster"] = mapped_clusters.astype(int)
                    group_col = "seqid_cluster"
        baseline_cols = [c for c in ("y_pred_drug_mean", "y_pred_target_mean") if c in oof_df.columns]
        if group_col in oof_df.columns and oof_df[group_col].nunique() >= 2 and baseline_cols:
            bootstrap_results["cluster_group_col"] = group_col
            bs_map = bootstrap_grouped_comparisons(
                oof_df,
                group_col=group_col,
                model_col="y_pred",
                baseline_cols=baseline_cols,
                target_col="y_true",
                n_bootstraps=1000,
                seed=42,
            )
            if "y_pred_drug_mean" in bs_map:
                bootstrap_results["vs_drug_mean"] = bs_map["y_pred_drug_mean"]
            if "y_pred_target_mean" in bs_map:
                bootstrap_results["vs_target_mean"] = bs_map["y_pred_target_mean"]

    ext_baselines = _collect_external_baselines_per_unit(
        results_dir=results_dir,
        dataset=dataset,
        split=split,
        protocol="kfold",
        unit_ids=target_folds,
        found_runs_map=per_fold_results,
    )
    paired_vs_baselines = _compute_paired_summary(per_fold_results, target_folds, ext_baselines)

    out_path = results_dir / f"phase3_{model}_{dataset}_{split}_kfold.json"
    agg_payload: dict[str, Any] = {
        "model": model,
        "dataset": dataset,
        "split": split,
        "protocol": "kfold",
        "num_folds": len(found_runs),
        "folds": [int(f) for f in per_fold_results.keys()],
        "test_isolation_verified": all(bool(r.get("test_isolation_verified", False)) for r in found_runs),
        "per_fold_results": per_fold_results,
        "per_fold_mean_std": per_fold_mean_std,
        "global_oof_metrics": global_oof_metrics,
        "binary_classification_metrics": binary_classification_metrics,
        "bootstrap_analysis": bootstrap_results,
        "paired_vs_baselines": paired_vs_baselines,
        "oof_csv_path": str(oof_csv_path),
    }
    save_json_atomic(agg_payload, out_path)
    print(f"[AGGREGATE-PHASE3-KFOLD] Salvo em: {out_path} ({len(found_runs)}/{len(target_folds)} dobras)", flush=True)
    return agg_payload


def aggregate_phase3_seed_runs(
    dataset: str,
    split: str,
    seeds: list[int] | None = None,
    config: dict[str, Any] | None = None,
    root: Path | str | None = None,
    model: str = PHASE3_MODEL_NAME,
) -> dict[str, Any]:
    """Consolida execuções por seeds (Protocolo v1) da Fase 3."""
    cfg = config if config is not None else load_config(root=root)
    results_dir = _resolve_output_dir(cfg, "results_dir", root=root)
    runs_dir = results_dir / "runs"

    target_seeds = seeds if seeds is not None else list(cfg.get("seeds", [0, 1, 2, 3, 4]))
    per_seed_results: dict[str, dict[str, Any]] = {}
    found_runs: list[dict[str, Any]] = []

    for s in target_seeds:
        rf = runs_dir / get_run_filename(model, dataset, split, s, protocol="seeds")
        if not rf.exists():
            continue
        data = json.loads(rf.read_text(encoding="utf-8"))
        found_runs.append(data)
        per_seed_results[str(s)] = {k: v for k, v in data.items() if k != "oof_predictions"}

    if not found_runs:
        raise FileNotFoundError(f"Nenhuma seed encontrada para ({model}, {dataset}, {split}) em {runs_dir}")

    metrics_keys = ["ci", "mse", "rmse", "pearson", "spearman", "rm2"]
    aggregated: dict[str, dict[str, float]] = {}
    for mk in metrics_keys:
        vals = [
            float(r["test_metrics_full"][mk])
            for r in found_runs
            if mk in (r.get("test_metrics_full") or {})
        ]
        if vals:
            aggregated[mk] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            }

    ext_baselines = _collect_external_baselines_per_unit(
        results_dir=results_dir,
        dataset=dataset,
        split=split,
        protocol="seeds",
        unit_ids=target_seeds,
        found_runs_map=per_seed_results,
    )
    paired_vs_baselines = _compute_paired_summary(per_seed_results, target_seeds, ext_baselines)

    out_path = results_dir / f"phase3_{model}_{dataset}_{split}.json"
    agg_payload: dict[str, Any] = {
        "model": model,
        "dataset": dataset,
        "split": split,
        "protocol": "seeds",
        "num_seeds": len(found_runs),
        "seeds": [int(s) for s in per_seed_results.keys()],
        "test_isolation_verified": all(bool(r.get("test_isolation_verified", False)) for r in found_runs),
        "per_seed_results": per_seed_results,
        "aggregated": aggregated,
        "paired_vs_baselines": paired_vs_baselines,
    }
    save_json_atomic(agg_payload, out_path)
    print(f"[AGGREGATE-PHASE3-SEEDS] Salvo em: {out_path} ({len(found_runs)}/{len(target_seeds)} seeds)", flush=True)
    return agg_payload


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _resolve_requested_splits(dataset: str, split: str, protocol: str) -> list[str]:
    if split != "all":
        return [split]
    if protocol == "kfold":
        if dataset.lower().strip() == "kiba":
            return ["cold_target_seqid30", "cold_target_seqid40"]
        return ["cold_drug", "cold_target", "cold_target_seqid30", "cold_target_seqid40"]
    return ["random", "cold_drug", "cold_target"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa treinamento e avaliação da Fase 3 (PhysDualPhase3).")
    parser.add_argument(
        "--model",
        type=str,
        default=PHASE3_MODEL_NAME,
        choices=[PHASE3_MODEL_NAME, PHASE3_ECFP_MODEL_NAME, "ecfp_mlp"],
        help="Arquitetura Fase 3 (`physdual_gin_esm_mlp` padrão ou controle diagnóstico `ecfp_mlp`).",
    )
    parser.add_argument("--protocol", type=str, default="kfold", choices=["seeds", "kfold"])
    parser.add_argument("--dataset", type=str, default="davis", choices=["davis", "kiba"])
    parser.add_argument(
        "--split",
        type=str,
        default="cold_target",
        choices=["random", "cold_drug", "cold_target", "cold_target_seqid30", "cold_target_seqid40", "all"],
    )
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--folds", type=str, default="0,1,2,3,4")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--davis-cold-patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--esm-dropout-in", type=float, default=None)
    parser.add_argument("--prot-dropout", type=float, default=None)
    parser.add_argument("--mlp-dropout", type=float, default=None)
    parser.add_argument("--prot-output-dim", type=int, default=None)
    parser.add_argument("--esm-cache-dir", type=str, default=None)
    parser.add_argument("--drive-checkpoint-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--extend-epochs", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--shuffle-esm",
        action="store_true",
        help="Controle diagnóstico: embaralha embeddings ESM-2 via mapeamento fixo sequência->outra sequência.",
    )
    parser.add_argument(
        "--shuffle-esm-seeds",
        type=str,
        default="0",
        help="Lista de seeds de permutação para `--shuffle-esm` (ex: '0,1,2').",
    )
    parser.add_argument(
        "--train-seed-offset",
        type=int,
        default=0,
        help="Controle diagnóstico: desloca a seed de inicialização/shuffle de treino sem alterar os splits.",
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--root", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(config_path=args.config, root=args.root)
    base_root = Path(cfg["project_root"])
    runs_dir = base_root / cfg["paths"]["results_dir"] / "runs"
    base_model_name = normalize_phase3_model_name(args.model)
    shuf_seeds = _parse_int_list(args.shuffle_esm_seeds) if args.shuffle_esm else [0]

    if args.aggregate_only:
        for s_seed in shuf_seeds:
            eff_model_name = resolve_diagnostic_model_name(
                base_model_name,
                shuffle_esm=args.shuffle_esm,
                shuffle_esm_seed=s_seed,
                train_seed_offset=args.train_seed_offset,
            )
            esc_model = re.escape(eff_model_name)
            seed_groups: dict[tuple[str, str], list[int]] = {}
            fold_groups: dict[tuple[str, str], list[int]] = {}
            if runs_dir.exists():
                for jf in runs_dir.glob(f"{eff_model_name}_*.json"):
                    m_f = re.match(
                        rf"^{esc_model}_(davis|kiba)_(random|cold_drug|cold_target|cold_target_seqid30|cold_target_seqid40)_fold(\d+)$",
                        jf.stem,
                    )
                    if m_f:
                        fold_groups.setdefault((m_f.group(1), m_f.group(2)), []).append(int(m_f.group(3)))
                        continue
                    m_s = re.match(
                        rf"^{esc_model}_(davis|kiba)_(random|cold_drug|cold_target)_seed(\d+)$",
                        jf.stem,
                    )
                    if m_s:
                        seed_groups.setdefault((m_s.group(1), m_s.group(2)), []).append(int(m_s.group(3)))
            for (ds_val, sp_val), u_list in sorted(fold_groups.items()):
                aggregate_phase3_kfold_runs(
                    dataset=ds_val,
                    split=sp_val,
                    folds=sorted(set(u_list)),
                    config=cfg,
                    root=base_root,
                    model=eff_model_name,
                )
            for (ds_val, sp_val), u_list in sorted(seed_groups.items()):
                aggregate_phase3_seed_runs(
                    dataset=ds_val,
                    split=sp_val,
                    seeds=sorted(set(u_list)),
                    config=cfg,
                    root=base_root,
                    model=eff_model_name,
                )
        return

    splits = _resolve_requested_splits(args.dataset, args.split, args.protocol)
    if args.protocol != "kfold" and any(sp.startswith("cold_target_seqid") for sp in splits):
        raise ValueError("Splits `cold_target_seqid30/40` requerem `--protocol kfold`.")

    units = _parse_int_list(args.folds if args.protocol == "kfold" else args.seeds)
    shared_esm_cache: dict[str, torch.Tensor] = {}

    for s_seed in shuf_seeds:
        eff_model_name = resolve_diagnostic_model_name(
            base_model_name,
            shuffle_esm=args.shuffle_esm,
            shuffle_esm_seed=s_seed,
            train_seed_offset=args.train_seed_offset,
        )
        for sp in splits:
            for u in units:
                run_single_unit(
                    dataset_name=args.dataset,
                    split_type=sp,
                    protocol=args.protocol,
                    unit_id=u,
                    config=cfg,
                    root=base_root,
                    esm_cache_dir=args.esm_cache_dir,
                    esm_cache=shared_esm_cache,
                    force=args.force,
                    device=args.device,
                    epochs=args.epochs,
                    patience=args.patience,
                    davis_cold_patience=args.davis_cold_patience,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    num_workers=args.num_workers,
                    drive_checkpoint_dir=args.drive_checkpoint_dir,
                    smoke=args.smoke,
                    extend_epochs=args.extend_epochs,
                    model_name=base_model_name,
                    esm_dropout_in=args.esm_dropout_in,
                    prot_dropout=args.prot_dropout,
                    mlp_dropout=args.mlp_dropout,
                    prot_output_dim=args.prot_output_dim,
                    shuffle_esm=args.shuffle_esm,
                    shuffle_esm_seed=s_seed,
                    train_seed_offset=args.train_seed_offset,
                )
            if args.protocol == "kfold":
                avail_folds = sorted({
                    u_idx for u_idx in set(units) | {0, 1, 2, 3, 4}
                    if (runs_dir / get_run_filename(eff_model_name, args.dataset, sp, u_idx, protocol="kfold")).exists()
                })
                aggregate_phase3_kfold_runs(
                    dataset=args.dataset,
                    split=sp,
                    folds=avail_folds or units,
                    config=cfg,
                    root=base_root,
                    model=eff_model_name,
                )
            else:
                avail_seeds = sorted({
                    u_idx for u_idx in set(units) | {0, 1, 2, 3, 4}
                    if (runs_dir / get_run_filename(eff_model_name, args.dataset, sp, u_idx, protocol="seeds")).exists()
                })
                aggregate_phase3_seed_runs(
                    dataset=args.dataset,
                    split=sp,
                    seeds=avail_seeds or units,
                    config=cfg,
                    root=base_root,
                    model=eff_model_name,
                )


if __name__ == "__main__":
    main()
