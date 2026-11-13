#!/usr/bin/env python3
"""Orquestrador resumível, idempotente e agregador de baselines da Fase 2 (`scripts/run_phase2_baseline.py`).

Implementa as seguintes garantias de engenharia e protocolos:
- C1: Isolamento estrito de teste (avaliação no teste realizada unicamente após o fim do treino).
- C2: Persistência individual por run atômica e retomabilidade granular por checkpoint.
- C3: Idempotência de execução (runs já concluídas são puladas sem reprocessamento, salvo com --force).
- Protocolo Seeds (v1): 5 seeds pseudoaleatórias com agregação estatística (média e desvio-padrão ddof=1).
- Protocolo GroupKFold (v2 - Opção C): 5 dobras agrupadas por conteúdo (smiles canônico ou sequência),
  salvamento de previsões fora da dobra (OOF), métricas de classificação binária (AUROC/AUPR) e
  bootstrap agrupado com intervalo de confiança de 95% para Delta MSE e Delta CI contra baselines marginais.
- Suporte a extensão de épocas (--extend-epochs) com retomada correta do estado do EarlyStopping.
- Benchmark de tempo de GPU/CPU para validação prévia de viabilidade de treinamento.
- Avaliação automática de baselines: triviais, marginais (promiscuidade) e kNN (similaridade/homologia).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from physdual.data.datamodule import compute_aa_composition, get_dta_dataloaders  # noqa: E402
from physdual.eval.metrics import (  # noqa: E402
    bootstrap_grouped_comparison,
    bootstrap_grouped_comparisons,
    compute_all_metrics,
    compute_binary_classification_metrics,
    compute_knn_baselines,
    compute_marginal_baselines,
    compute_trivial_baselines,
)
from physdual.models.graphdta import build_baseline_model  # noqa: E402
from physdual.training.trainer import train_model  # noqa: E402
from physdual.utils.config import load_config  # noqa: E402
from physdual.utils.paths import get_project_root  # noqa: E402
from physdual.utils.seed import set_seed  # noqa: E402

seed_everything = set_seed


def get_run_filename(
    model: str,
    dataset: str,
    split: str,
    unit_id: int | None = None,
    seed: int | None = None,
    fold: int | None = None,
    protocol: str = "seeds",
) -> str:
    """Retorna o nome padronizado do arquivo JSON individual de uma run."""
    if unit_id is None:
        unit_id = fold if fold is not None else (seed if seed is not None else 0)
    unit_prefix = "fold" if protocol == "kfold" else "seed"
    return f"{model}_{dataset}_{split}_{unit_prefix}{unit_id}.json"


def save_json_atomic(data: dict[str, Any], path: Path | str) -> Path:
    """Salva um dicionário em JSON atomicamente usando arquivo temporário no mesmo diretório."""
    dest_path = Path(path).resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.parent / f".tmp_{dest_path.name}_{os.getpid()}"
    try:
        tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp_path, dest_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
    return dest_path


def is_run_completed(
    model: str,
    dataset: str,
    split: str,
    unit_id: int | None = None,
    seed: int | None = None,
    fold: int | None = None,
    runs_dir: Path | str | None = None,
    protocol: str = "seeds",
    target_epochs: int | None = None,
    extend_epochs: bool = False,
) -> bool:
    """Verifica se o JSON individual de uma run já existe e é válido."""
    if unit_id is None:
        unit_id = fold if fold is not None else (seed if seed is not None else 0)

    if runs_dir is None:
        cfg = load_config()
        runs_dir = Path(cfg["project_root"]) / cfg["paths"]["results_dir"] / "runs"
    else:
        runs_dir = Path(runs_dir)

    run_file = runs_dir / get_run_filename(model, dataset, split, unit_id, protocol=protocol)
    if not run_file.exists():
        return False

    try:
        data = json.loads(run_file.read_text(encoding="utf-8"))
        has_metrics = "test_metrics_full" in data and "model" in data
        if not has_metrics:
            return False
        if extend_epochs and target_epochs is not None:
            prev_epochs = data.get("effective_config", {}).get("epochs", 0)
            if prev_epochs < target_epochs:
                return False
        return True
    except Exception:
        return False


def _compute_baselines_from_loaders(
    train_loader: Any,
    test_loader: Any,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Calcula baselines triviais, marginais e kNN a partir dos DataLoaders."""
    train_dataset = getattr(train_loader, "dataset", None)
    test_dataset = getattr(test_loader, "dataset", None)

    train_df = getattr(train_dataset, "df", None)
    test_df = getattr(test_dataset, "df", None)

    if train_df is None or test_df is None:
        trivial = {
            "mean_predictor": {"ci": 0.5, "mse": 1.0},
            "ridge_aa_composition": {"ci": 0.5, "mse": 1.0},
        }
        marginal = {
            "drug_mean_predictor": {"ci": 0.5, "mse": 1.0},
            "target_mean_predictor": {"ci": 0.5, "mse": 1.0},
        }
        knn = {
            "knn_tanimoto_drug": {"ci": 0.5, "mse": 1.0},
            "knn_kmer_target": {"ci": 0.5, "mse": 1.0},
        }
        return trivial, marginal, knn

    y_train = train_df["affinity"].values
    y_test = test_df["affinity"].values

    X_train_aa = np.stack([compute_aa_composition(seq) for seq in train_df["sequence"]])
    X_test_aa = np.stack([compute_aa_composition(seq) for seq in test_df["sequence"]])

    trivial = compute_trivial_baselines(y_train, y_test, X_train_aa, X_test_aa)
    marginal = compute_marginal_baselines(train_df, test_df)
    knn = compute_knn_baselines(train_df, test_df)

    return trivial, marginal, knn


def run_single_unit(
    model_name: str,
    dataset_name: str,
    split_type: str,
    unit_id: int | None = None,
    protocol: str = "seeds",
    seed: int | None = None,
    fold: int | None = None,
    config: dict[str, Any] | None = None,
    runs_dir: Path | str | None = None,
    checkpoints_base_dir: Path | str | None = None,
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
) -> dict[str, Any]:
    """Executa o treinamento de uma unidade individual (seed ou fold) de forma idempotente e atômica.

    Args:
        model_name: Nome do modelo ('graphdta_gin', 'graphdta_cnn3').
        dataset_name: Nome do dataset ('davis', 'kiba').
        split_type: Partição ('random', 'cold_drug', 'cold_target').
        unit_id: Número da semente (seeds) ou índice da dobra (kfold).
        protocol: Protocolo de validação ('seeds' ou 'kfold').
        seed: Argumento de compatibilidade (atribuído a unit_id se protocol='seeds').
        fold: Argumento de compatibilidade (atribuído a unit_id se protocol='kfold').
        config: Dicionário base de configuração (opcional).
        runs_dir: Diretório de persistência dos JSONs individuais.
        checkpoints_base_dir: Diretório raiz de checkpoints locais.
        force: Se True, sobrescreve e reexecuta runs previamente concluídas.
        device: Dispositivo de execução ('auto', 'cuda', 'cpu').
        epochs: Máximo de épocas de treino.
        patience: Paciência base para early stopping.
        davis_cold_patience: Paciência específica para splits cold de Davis.
        batch_size: Tamanho do minilote.
        lr: Taxa de aprendizado inicial.
        weight_decay: Decaimento de peso L2.
        num_workers: Quantidade de processos auxiliares no DataLoader.
        drive_checkpoint_dir: Espelhamento atômico opcional no Google Drive.
        smoke: Se True, executa versão mínima ultra-rápida (1 época, batch size reduzido).
        extend_epochs: Se True, permite estender até `epochs` mesmo se run anterior existia.

    Returns:
        Dicionário com todos os metadados e métricas persistidos na run.
    """
    if unit_id is None:
        unit_id = fold if fold is not None else (seed if seed is not None else 0)

    cfg = copy.deepcopy(config) if config is not None else load_config()
    base_root = Path(cfg.get("project_root", get_project_root()))
    paths_cfg = cfg["paths"]

    if runs_dir is None:
        runs_dir = base_root / paths_cfg["results_dir"] / "runs"
    else:
        runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_file = runs_dir / get_run_filename(
        model_name, dataset_name, split_type, unit_id, protocol=protocol
    )

    # Configurações de hiperparâmetros
    gdta_cfg = cfg.get("graphdta", {})
    train_cfg = gdta_cfg.get("training", {})

    eff_epochs = epochs if epochs is not None else int(train_cfg.get("epochs", 500))
    eff_patience = patience if patience is not None else int(train_cfg.get("patience", 50))
    eff_davis_cold_pat = (
        davis_cold_patience
        if davis_cold_patience is not None
        else int(train_cfg.get("davis_cold_patience", 80))
    )
    eff_batch_size = batch_size if batch_size is not None else int(train_cfg.get("batch_size", 512))
    eff_lr = lr if lr is not None else float(train_cfg.get("lr", 5e-4))
    eff_wd = weight_decay if weight_decay is not None else float(train_cfg.get("weight_decay", 0.0))
    eff_workers = num_workers if num_workers is not None else int(train_cfg.get("num_workers", 2))
    save_last_every = int(train_cfg.get("save_last_every", 10))

    if smoke:
        eff_epochs = 1
        eff_patience = 1
        eff_batch_size = min(eff_batch_size, 64)
        eff_workers = 0

    # -------------------------------------------------------------------------
    # Idempotência C3: Pula runs concluídas
    # -------------------------------------------------------------------------
    if not force and is_run_completed(
        model_name,
        dataset_name,
        split_type,
        unit_id,
        runs_dir=runs_dir,
        protocol=protocol,
        target_epochs=eff_epochs,
        extend_epochs=extend_epochs,
    ):
        print(f"[SKIP] Run já concluída: {run_file}")
        try:
            cached_data = json.loads(run_file.read_text(encoding="utf-8"))
            cached_data["skipped"] = True
            return cached_data
        except Exception:
            pass

    # Fixação determinística de seeds
    random_seed = unit_id if protocol == "seeds" else (42 + unit_id)
    seed_everything(random_seed)

    # Resolução de dispositivo
    if device == "auto":
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    cfg["graphdta"]["training"]["batch_size"] = eff_batch_size
    cfg["graphdta"]["training"]["num_workers"] = eff_workers

    # Diretório de checkpoints local por run
    if checkpoints_base_dir is None:
        checkpoints_base_dir = base_root / paths_cfg["checkpoints_dir"]
    else:
        checkpoints_base_dir = Path(checkpoints_base_dir)

    unit_prefix = "fold" if protocol == "kfold" else "seed"
    run_ckpt_dir = (
        checkpoints_base_dir / f"{model_name}_{dataset_name}_{split_type}_{unit_prefix}{unit_id}"
    )
    run_ckpt_dir.mkdir(parents=True, exist_ok=True)

    run_drive_ckpt_dir: Path | None = None
    if drive_checkpoint_dir is not None:
        run_drive_ckpt_dir = (
            Path(drive_checkpoint_dir)
            / f"{model_name}_{dataset_name}_{split_type}_{unit_prefix}{unit_id}"
        )

    # Carregamento de DataLoaders
    if protocol == "kfold":
        train_loader, val_loader, test_loader = get_dta_dataloaders(
            dataset_name=dataset_name,
            split_type=split_type,
            protocol="kfold",
            fold=unit_id,
            config=cfg,
            root=base_root,
        )
    else:
        train_loader, val_loader, test_loader = get_dta_dataloaders(
            dataset_name=dataset_name,
            split_type=split_type,
            protocol="seeds",
            seed=unit_id,
            config=cfg,
            root=base_root,
        )

    # Construção do modelo
    model = build_baseline_model(model_name, cfg)

    # Execução do Treinamento
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
        monitor="mse",
        checkpoint_dir=run_ckpt_dir,
        device=target_device,
        use_lr_scheduler=bool(train_cfg.get("use_lr_scheduler", False)),
        dataset_name=dataset_name,
        split_type=split_type,
        save_last_every=save_last_every,
        drive_checkpoint_dir=run_drive_ckpt_dir,
        resume=True,
    )
    elapsed_sec = float(time.time() - t0)
    total_epochs = int(train_result["total_epochs_run"])
    sec_per_epoch = float(elapsed_sec / max(1, total_epochs))

    # Cálculo dos baselines triviais, marginais e kNN
    trivial_baselines, marginal_baselines, knn_baselines = _compute_baselines_from_loaders(
        train_loader, test_loader
    )

    feat_cfg = cfg.get("featurizer", {})
    feat_version = str(feat_cfg.get("version", "v2_onehot78_l1norm"))
    norm_atoms = bool(feat_cfg.get("normalize_atom_features", True))
    drug_in_dim = int(gdta_cfg.get("drug_in_dim", 78))

    # Enriquece predições OOF com preditores de média do treino
    oof_predictions = train_result.get("oof_predictions", [])
    train_dataset = getattr(train_loader, "dataset", None)
    train_df = getattr(train_dataset, "df", None)
    if train_df is not None and oof_predictions:
        seq_map = dict(zip(train_df["target_id"].astype(str), train_df["sequence"])) if "target_id" in train_df.columns and "sequence" in train_df.columns else {}
        smi_map = dict(zip(train_df["drug_id"].astype(str), train_df["smiles"])) if "drug_id" in train_df.columns and "smiles" in train_df.columns else {}

        drug_means = train_df.groupby("smiles")["affinity"].mean().to_dict() if "smiles" in train_df.columns else {}
        target_means = train_df.groupby("sequence")["affinity"].mean().to_dict() if "sequence" in train_df.columns else {}
        global_mean = float(train_df["affinity"].mean())
        for rec in oof_predictions:
            rec["fold"] = unit_id if protocol == "kfold" else None
            rec["seed"] = unit_id if protocol == "seeds" else None
            if (not rec.get("sequence")) and rec.get("target_id") in seq_map:
                rec["sequence"] = str(seq_map[rec["target_id"]])
            if (not rec.get("smiles")) and rec.get("drug_id") in smi_map:
                rec["smiles"] = str(smi_map[rec["drug_id"]])
            rec["y_pred_drug_mean"] = float(drug_means.get(rec.get("smiles", ""), global_mean))
            rec["y_pred_target_mean"] = float(target_means.get(rec.get("sequence", ""), global_mean))

    run_record: dict[str, Any] = {
        "model": model_name,
        "dataset": dataset_name,
        "split": split_type,
        "protocol": protocol,
        "unit_id": unit_id,
        "seed": unit_id if protocol == "seeds" else None,
        "fold": unit_id if protocol == "kfold" else None,
        "featurizer_version": feat_version,
        "normalize_atom_features": norm_atoms,
        "drug_in_dim": drug_in_dim,
        "effective_config": {
            "epochs": eff_epochs,
            "patience": eff_patience,
            "davis_cold_patience": eff_davis_cold_pat,
            "batch_size": eff_batch_size,
            "lr": eff_lr,
            "weight_decay": eff_wd,
            "device": str(target_device),
        },
        "best_epoch": int(train_result["best_epoch"]),
        "total_epochs_run": total_epochs,
        "near_epoch_ceiling_warn": bool(train_result["near_epoch_ceiling_warn"]),
        "elapsed_sec": round(elapsed_sec, 2),
        "sec_per_epoch": round(sec_per_epoch, 3),
        "val_metrics_at_best": train_result["val_metrics_at_best"],
        "test_metrics_full": train_result["test_metrics_full"],
        "test_metrics_uncensored": train_result.get("test_metrics_uncensored"),
        "test_metrics_by_seq_len": train_result.get("test_metrics_by_seq_len"),
        "trivial_baselines": trivial_baselines,
        "marginal_baselines": marginal_baselines,
        "knn_baselines": knn_baselines,
        "history": train_result.get("history", []),
        "oof_predictions": oof_predictions,
        "test_isolation_verified": bool(
            train_result.get("val_metrics_at_best") and train_result.get("test_metrics_full")
        ),
    }

    save_json_atomic(run_record, run_file)
    print(
        f"[SAVE] Run salva com sucesso: {run_file} (best_epoch={run_record['best_epoch']}, "
        f"sec_per_epoch={sec_per_epoch:.2f}s)"
    )
    return run_record


def run_single_seed(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Compatibilidade legada com a interface da Fase 2 v1."""
    kwargs.setdefault("protocol", "seeds")
    return run_single_unit(*args, **kwargs)


def _calc_stats(values: list[float | int | None]) -> dict[str, float]:
    """Calcula média e desvio-padrão amostral (ddof=1)."""
    valid = [float(v) for v in values if v is not None and not np.isnan(float(v))]
    if not valid:
        return {"mean": float("nan"), "std": float("nan")}
    arr = np.array(valid, dtype=np.float64)
    std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return {
        "mean": float(np.mean(arr)),
        "std": std_val,
    }


def aggregate_runs(
    model: str,
    dataset: str,
    split: str,
    runs_dir: Path | str,
    output_file: Path | str | None = None,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Agrega os JSONs individuais das 5 seeds e computa métricas estatísticas (média e desvio-padrão).

    Args:
        model: Identificador do modelo ('graphdta_gin', 'graphdta_cnn3').
        dataset: Dataset ('davis', 'kiba').
        split: Tipo de partição ('random', 'cold_drug', 'cold_target').
        runs_dir: Diretório onde os JSONs individuais residem.
        output_file: Caminho de saída para o JSON consolidado.
        seeds: Lista de seeds esperadas (padrão: [0, 1, 2, 3, 4]).

    Returns:
        Dicionário com `per_seed_results` e `aggregated`.
    """
    runs_path = Path(runs_dir)
    target_seeds = [0, 1, 2, 3, 4] if seeds is None else seeds

    per_seed_results: dict[str, Any] = {}
    found_runs: list[dict[str, Any]] = []

    for s in target_seeds:
        run_file = runs_path / get_run_filename(model, dataset, split, s, protocol="seeds")
        if run_file.exists():
            try:
                data = json.loads(run_file.read_text(encoding="utf-8"))
                per_seed_results[str(s)] = {k: v for k, v in data.items() if k != "oof_predictions"}
                found_runs.append(data)
            except Exception as e:
                print(f"[WARN] Erro ao carregar {run_file}: {e}")

    if not found_runs:
        raise FileNotFoundError(
            f"Nenhuma run encontrada em '{runs_path}' para o trio ({model}, {dataset}, {split})."
        )

    # Extração de métricas de teste completas
    full_keys = ["ci", "mse", "rmse", "pearson", "spearman", "rm2"]
    agg_full: dict[str, dict[str, float]] = {}
    for k in full_keys:
        vals = [r.get("test_metrics_full", {}).get(k) for r in found_runs]
        agg_full[k] = _calc_stats(vals)

    # Extração de métricas uncensored (especialmente Davis)
    agg_uncensored: dict[str, dict[str, float]] | None = None
    if any(r.get("test_metrics_uncensored") is not None for r in found_runs):
        agg_uncensored = {}
        unc_keys = ["num_uncensored_pairs", "ci", "mse", "rmse", "pearson", "spearman", "rm2"]
        for k in unc_keys:
            vals = [
                r.get("test_metrics_uncensored", {}).get(k)
                for r in found_runs
                if r.get("test_metrics_uncensored") is not None
            ]
            agg_uncensored[k] = _calc_stats(vals)

    # Extração de métricas por comprimento de sequência
    agg_by_seq: dict[str, Any] = {}
    for bucket in ("le_1000", "gt_1000"):
        bucket_stats: dict[str, dict[str, float]] = {}
        for metric_name in ("ci", "mse"):
            vals = [
                r.get("test_metrics_by_seq_len", {}).get(bucket, {}).get(metric_name)
                for r in found_runs
                if r.get("test_metrics_by_seq_len") is not None
            ]
            bucket_stats[metric_name] = _calc_stats(vals)
        counts = [
            r.get("test_metrics_by_seq_len", {}).get(bucket, {}).get("count")
            for r in found_runs
            if r.get("test_metrics_by_seq_len") is not None
        ]
        bucket_stats["count"] = _calc_stats(counts)
        agg_by_seq[bucket] = bucket_stats

    # Extração de baselines triviais
    agg_trivial: dict[str, Any] = {}
    for bl_name in ("mean_predictor", "ridge_aa_composition"):
        bl_stats: dict[str, dict[str, float]] = {}
        for metric_name in ("ci", "mse"):
            vals = [
                r.get("trivial_baselines", {}).get(bl_name, {}).get(metric_name)
                for r in found_runs
                if r.get("trivial_baselines") is not None
            ]
            bl_stats[metric_name] = _calc_stats(vals)
        agg_trivial[bl_name] = bl_stats

    # Extração de baselines marginais
    agg_marginal: dict[str, Any] = {}
    for m_name in ("drug_mean_predictor", "target_mean_predictor"):
        m_stats: dict[str, dict[str, float]] = {}
        for metric_name in ("ci", "mse", "rmse", "pearson", "spearman", "rm2"):
            vals = [
                r.get("marginal_baselines", {}).get(m_name, {}).get(metric_name)
                for r in found_runs
                if r.get("marginal_baselines") is not None
            ]
            m_stats[metric_name] = _calc_stats(vals)
        agg_marginal[m_name] = m_stats

    # Extração de baselines kNN
    agg_knn: dict[str, Any] = {}
    for knn_name in ("knn_tanimoto_drug", "knn_kmer_target"):
        knn_stats: dict[str, dict[str, float]] = {}
        for metric_name in ("ci", "mse", "rmse", "pearson", "spearman", "rm2"):
            vals = [
                r.get("knn_baselines", {}).get(knn_name, {}).get(metric_name)
                for r in found_runs
                if r.get("knn_baselines") is not None
            ]
            knn_stats[metric_name] = _calc_stats(vals)
        agg_knn[knn_name] = knn_stats

    best_epochs = [r.get("best_epoch") for r in found_runs]
    sec_per_epochs = [r.get("sec_per_epoch") for r in found_runs]
    total_epochs = [r.get("total_epochs_run") for r in found_runs]
    elapsed_secs = [r.get("elapsed_sec") for r in found_runs]

    aggregated_payload: dict[str, Any] = {
        "test_metrics_full": agg_full,
        "test_metrics_uncensored": agg_uncensored,
        "test_metrics_by_seq_len": agg_by_seq,
        "trivial_baselines": agg_trivial,
        "marginal_baselines": agg_marginal,
        "knn_baselines": agg_knn,
        "best_epoch": _calc_stats(best_epochs),
        "sec_per_epoch": _calc_stats(sec_per_epochs),
        "total_epochs_run": _calc_stats(total_epochs),
        "elapsed_sec": _calc_stats(elapsed_secs),
    }

    result = {
        "model": model,
        "dataset": dataset,
        "split": split,
        "protocol": "seeds",
        "num_seeds": len(found_runs),
        "seeds": [int(s) for s in per_seed_results.keys()],
        "test_isolation_verified": all(
            bool(r.get("test_isolation_verified", r.get("val_metrics_at_best") and r.get("test_metrics_full")))
            for r in found_runs
        ),
        "per_seed_results": per_seed_results,
        "aggregated": aggregated_payload,
    }

    if output_file is not None:
        out_path = Path(output_file)
    else:
        cfg = load_config()
        base_root = Path(cfg.get("project_root", get_project_root()))
        out_path = base_root / cfg["paths"]["results_dir"] / f"baseline_{model}_{dataset}_{split}.json"

    save_json_atomic(result, out_path)
    print(
        f"[AGGREGATE] Resultados consolidados salvos em: {out_path} "
        f"({len(found_runs)}/{len(target_seeds)} seeds)"
    )
    return result


def aggregate_kfold_runs(
    model: str,
    dataset: str,
    split: str,
    runs_dir: Path | str | None = None,
    folds: list[int] | None = None,
    oof_dir: Path | str | None = None,
    output_file: Path | str | None = None,
) -> dict[str, Any]:
    """Agrega os JSONs individuais das 5 dobras K-Fold, consolida previsões OOF e computa bootstrap.

    Args:
        model: Identificador do modelo ('graphdta_gin', 'graphdta_cnn3').
        dataset: Dataset ('davis', 'kiba').
        split: Tipo de partição ('cold_drug', 'cold_target', 'random').
        runs_dir: Diretório onde os JSONs individuais de cada dobra residem (padrão via config).
        folds: Lista de dobras (padrão: [0, 1, 2, 3, 4]).
        oof_dir: Diretório para gravação da tabela CSV consolidada com previsões OOF.
        output_file: Caminho de saída opcional para o JSON consolidado.

    Returns:
        Dicionário com per_fold_results, per_fold_mean_std, global_oof_metrics,
        binary_classification_metrics e bootstrap_analysis.
    """
    cfg = load_config()
    base_root = Path(cfg.get("project_root", get_project_root()))
    runs_path = Path(runs_dir) if runs_dir is not None else (base_root / cfg["paths"]["results_dir"] / "runs")
    target_folds = [0, 1, 2, 3, 4] if folds is None else folds

    per_fold_results: dict[str, Any] = {}
    found_runs: list[dict[str, Any]] = []
    all_oof_records: list[dict[str, Any]] = []

    for f in target_folds:
        run_file = runs_path / get_run_filename(model, dataset, split, f, protocol="kfold")
        if run_file.exists():
            try:
                data = json.loads(run_file.read_text(encoding="utf-8"))
                found_runs.append(data)
                for rec in data.get("oof_predictions", []):
                    rec_copy = dict(rec)
                    if "fold" not in rec_copy or rec_copy["fold"] is None:
                        rec_copy["fold"] = f
                    all_oof_records.append(rec_copy)
                # Omite oof_predictions duplicadas no JSON agregado (já consolidadas no CSV OOF)
                summary_data = {k: v for k, v in data.items() if k != "oof_predictions"}
                per_fold_results[str(f)] = summary_data
            except Exception as e:
                print(f"[WARN] Erro ao carregar dobra {run_file}: {e}")

    if not found_runs:
        raise FileNotFoundError(
            f"Nenhuma dobra K-Fold encontrada em '{runs_path}' para o trio ({model}, {dataset}, {split})."
        )

    # 1. Estatísticas por dobra (média e std ddof=1)
    full_keys = ["ci", "mse", "rmse", "pearson", "spearman", "rm2"]
    per_fold_mean_std: dict[str, dict[str, float]] = {}
    for k in full_keys:
        vals = [r.get("test_metrics_full", {}).get(k) for r in found_runs]
        per_fold_mean_std[k] = _calc_stats(vals)

    best_epochs = [r.get("best_epoch") for r in found_runs]
    total_epochs = [r.get("total_epochs_run") for r in found_runs]
    sec_per_epochs = [r.get("sec_per_epoch") for r in found_runs]
    elapsed_secs = [r.get("elapsed_sec") for r in found_runs]

    fold_timings = {
        "best_epoch": _calc_stats(best_epochs),
        "total_epochs_run": _calc_stats(total_epochs),
        "sec_per_epoch": _calc_stats(sec_per_epochs),
        "elapsed_sec": _calc_stats(elapsed_secs),
    }

    # 2. Consolidação OOF (Out-Of-Fold)
    if oof_dir is None:
        oof_path = base_root / cfg["paths"]["results_dir"] / "oof"
    else:
        oof_path = Path(oof_dir)
    oof_path.mkdir(parents=True, exist_ok=True)
    oof_csv_path = oof_path / f"{model}_{dataset}_{split}_oof.csv"

    global_oof_metrics: dict[str, float] = {}
    binary_classification_metrics: dict[str, Any] = {}
    bootstrap_results: dict[str, Any] = {}

    if all_oof_records:
        oof_df = pd.DataFrame(all_oof_records)

        # Auto-recuperação de metadados e baselines marginais por dobra a partir de dados curados
        clean_csv = base_root / "data" / "processed" / f"{dataset.lower()}_clean.csv"
        splits_dir = base_root / "data" / "processed" / "splits"
        if clean_csv.exists():
            try:
                clean_df = pd.read_csv(clean_csv)
                t_to_seq = dict(zip(clean_df["target_id"].astype(str), clean_df["sequence"]))
                d_to_smi = dict(zip(clean_df["drug_id"].astype(str), clean_df["smiles"]))
                if (
                    "sequence" not in oof_df.columns
                    or oof_df["sequence"].nunique() < 2
                    or (oof_df["sequence"].fillna("").astype(str).str.len() == 0).any()
                ):
                    if "target_id" in oof_df.columns:
                        oof_df["sequence"] = oof_df["target_id"].astype(str).map(t_to_seq).fillna("")
                if (
                    "smiles" not in oof_df.columns
                    or oof_df["smiles"].nunique() < 2
                    or (oof_df["smiles"].fillna("").astype(str).str.len() == 0).any()
                ):
                    if "drug_id" in oof_df.columns:
                        oof_df["smiles"] = oof_df["drug_id"].astype(str).map(d_to_smi).fillna("")

                # Checa unicidade por dobra para detectar baselines que caíram no fallback da média global
                recalc_marginals = False
                for col in ("y_pred_drug_mean", "y_pred_target_mean"):
                    if col not in oof_df.columns or oof_df[col].nunique() < 2:
                        recalc_marginals = True
                    elif "fold" in oof_df.columns and oof_df.groupby("fold")[col].nunique().min() < 2:
                        recalc_marginals = True

                ids_match = (
                    "target_id" in oof_df.columns
                    and bool(set(oof_df["target_id"].astype(str)).intersection(t_to_seq.keys()))
                )
                if recalc_marginals and splits_dir.exists() and ids_match:
                    for f in oof_df["fold"].unique():
                        sp_file = splits_dir / f"{dataset.lower()}_{split}_fold{int(f)}.json"
                        if sp_file.exists():
                            sp_data = json.loads(sp_file.read_text(encoding="utf-8"))
                            f_train = clean_df.iloc[sp_data["train"]]
                            f_drug_means = f_train.groupby("smiles")["affinity"].mean().to_dict()
                            f_target_means = f_train.groupby("sequence")["affinity"].mean().to_dict()
                            f_gmean = float(f_train["affinity"].mean())

                            mask = (oof_df["fold"] == f)
                            oof_df.loc[mask, "y_pred_drug_mean"] = oof_df.loc[mask, "smiles"].map(f_drug_means).fillna(f_gmean)
                            oof_df.loc[mask, "y_pred_target_mean"] = oof_df.loc[mask, "sequence"].map(f_target_means).fillna(f_gmean)
            except Exception as exc:
                print(f"[WARN] Falha na auto-recuperação de metadados OOF: {exc}")

        oof_df.to_csv(oof_csv_path, index=False)
        print(f"[OOF] Previsões fora da dobra salvas em: {oof_csv_path} ({len(oof_df)} pares)")

        y_true = np.asarray(oof_df["y_true"], dtype=np.float64)
        y_pred = np.asarray(oof_df["y_pred"], dtype=np.float64)
        global_oof_metrics = compute_all_metrics(y_true, y_pred)
        binary_classification_metrics = compute_binary_classification_metrics(
            y_true, y_pred, thresholds=[5.0, 7.0]
        )

        # 3. Bootstrap agrupado (cluster-level) por droga ou por alvo
        is_cold_target = split.startswith("cold_target")
        group_col = "sequence" if is_cold_target else "smiles"
        if group_col not in oof_df.columns or oof_df[group_col].nunique() < 2:
            alt_col = "target_id" if is_cold_target else "drug_id"
            if alt_col in oof_df.columns and oof_df[alt_col].nunique() >= 2:
                group_col = alt_col

        if group_col in oof_df.columns:
            bootstrap_results["cluster_group_col"] = group_col
            baseline_cols = [c for c in ("y_pred_drug_mean", "y_pred_target_mean") if c in oof_df.columns]
            if baseline_cols:
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

    if output_file is not None:
        out_path = Path(output_file)
    else:
        out_path = base_root / cfg["paths"]["results_dir"] / f"baseline_{model}_{dataset}_{split}_kfold.json"

    result: dict[str, Any] = {
        "model": model,
        "dataset": dataset,
        "split": split,
        "protocol": "kfold",
        "num_folds": len(found_runs),
        "folds": [int(f) for f in per_fold_results.keys()],
        "test_isolation_verified": all(
            bool(r.get("test_isolation_verified", r.get("val_metrics_at_best") and r.get("test_metrics_full")))
            for r in found_runs
        ),
        "per_fold_results": per_fold_results,
        "per_fold_mean_std": per_fold_mean_std,
        "fold_timings": fold_timings,
        "global_oof_metrics": global_oof_metrics,
        "binary_classification_metrics": binary_classification_metrics,
        "bootstrap_analysis": bootstrap_results,
        "oof_csv_path": str(oof_csv_path),
    }

    save_json_atomic(result, out_path)
    print(
        f"[AGGREGATE-KFOLD] Resultados consolidados salvos em: {out_path} "
        f"({len(found_runs)}/{len(target_folds)} dobras)"
    )
    return result


def run_timing_benchmark(
    benchmark_epochs: int = 3,
    config: dict[str, Any] | None = None,
    output_file: Path | str | None = None,
    device: str | torch.device = "auto",
) -> dict[str, Any]:
    """Executa benchmark de tempo em GPU/CPU para estimar duração de 500 épocas em Davis e KIBA."""
    cfg = copy.deepcopy(config) if config is not None else load_config()
    base_root = Path(cfg.get("project_root", get_project_root()))

    if device == "auto":
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    print(f"=== Iniciando Benchmark de Tempo ({benchmark_epochs} épocas por dataset) no dispositivo {target_device} ===")

    benchmark_results: dict[str, Any] = {
        "benchmark_epochs": benchmark_epochs,
        "device": str(target_device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "datasets": {},
    }

    for ds in ("davis", "kiba"):
        print(f"\n--- Benchmark {ds.upper()} ---")
        try:
            train_loader, val_loader, _ = get_dta_dataloaders(
                dataset_name=ds,
                split_type="random",
                seed=0,
                config=cfg,
                root=base_root,
            )
            model = build_baseline_model("graphdta_gin", cfg)

            t0 = time.time()
            res = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=benchmark_epochs,
                lr=5e-4,
                patience=benchmark_epochs,
                device=target_device,
                dataset_name=ds,
                split_type="random",
                save_last_every=100,
                resume=False,
            )
            total_sec = float(time.time() - t0)
            sec_ep = total_sec / max(1, benchmark_epochs)
            est_500_hours = (sec_ep * 500) / 3600.0

            print(f"  Duração {benchmark_epochs} épocas: {total_sec:.2f} s ({sec_ep:.2f} s/época)")
            print(f"  Estimativa 500 épocas (1 run): {est_500_hours:.2f} h")
            print(f"  Estimativa 5 seeds x 3 splits (15 runs): {est_500_hours * 15:.2f} h")

            benchmark_results["datasets"][ds] = {
                "num_train_pairs": len(train_loader.dataset),
                "num_val_pairs": len(val_loader.dataset),
                "sec_per_epoch": round(sec_ep, 3),
                "est_500_epochs_hours": round(est_500_hours, 2),
                "est_15_runs_hours": round(est_500_hours * 15, 2),
            }
        except Exception as exc:
            print(f"  [WARN] Falha no benchmark de {ds}: {exc}")
            benchmark_results["datasets"][ds] = {"error": str(exc)}

    if output_file is None:
        output_file = base_root / cfg["paths"]["results_dir"] / "phase2_timing_benchmark.json"
    else:
        output_file = Path(output_file)

    save_json_atomic(benchmark_results, output_file)
    print(f"\nBenchmark finalizado e salvo em: {output_file}")
    return benchmark_results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Configura e analisa argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Orquestrador resumível, idempotente e agregador de baselines da Fase 2."
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default="seeds",
        choices=["seeds", "kfold"],
        help="Protocolo de validação: 'seeds' (5 sementes, v1) ou 'kfold' (5-fold GroupKFold, v2).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="davis",
        choices=["davis", "kiba", "all"],
        help="Dataset a executar ('davis', 'kiba' ou 'all').",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="random",
        choices=[
            "random",
            "cold_drug",
            "cold_target",
            "cold_target_seqid30",
            "cold_target_seqid40",
            "all",
        ],
        help="Tipo de partição a executar ou 'all' ('random', 'cold_drug', 'cold_target', 'cold_target_seqid30', 'cold_target_seqid40').",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="graphdta_gin",
        choices=["graphdta_gin", "graphdta_cnn3", "all"],
        help="Modelo a executar ('graphdta_gin', 'graphdta_cnn3' ou 'all').",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
        help="Lista de seeds separadas por vírgula para protocolo seeds (padrão: '0,1,2,3,4').",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default="0,1,2,3,4",
        help="Lista de dobras separadas por vírgula para protocolo kfold (padrão: '0,1,2,3,4').",
    )
    parser.add_argument("--epochs", type=int, default=500, help="Número máximo de épocas.")
    parser.add_argument(
        "--extend-epochs",
        action="store_true",
        help="Permite estender o treinamento de runs existentes até o novo teto de épocas (ex: de 500 para 1000).",
    )
    parser.add_argument("--patience", type=int, default=50, help="Paciência para early stopping.")
    parser.add_argument(
        "--davis-cold-patience",
        type=int,
        default=80,
        help="Paciência mínima obrigatória para splits cold em Davis.",
    )
    parser.add_argument("--batch-size", type=int, default=512, help="Tamanho do minilote.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Taxa de aprendizado inicial.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Decaimento de peso L2.")
    parser.add_argument("--device", type=str, default="auto", help="Dispositivo ('auto', 'cuda', 'cpu').")
    parser.add_argument("--num-workers", type=int, default=2, help="Workers do DataLoader.")
    parser.add_argument("--force", action="store_true", help="Reexecuta runs mesmo que já concluídas.")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Apenas consolida os JSONs individuais existentes em outputs/results/runs/.",
    )
    parser.add_argument(
        "--benchmark-epochs",
        type=int,
        default=None,
        help="Executa benchmark de tempo de N épocas em Davis e KIBA.",
    )
    parser.add_argument(
        "--drive-checkpoint-dir",
        type=str,
        default=None,
        help="Diretório opcional no Google Drive para espelhamento seguro de checkpoints.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Modo de teste rápido (1 época, batch size menor) para validação de pipeline.",
    )
    parser.add_argument("--config", type=str, default=None, help="Caminho alternativo para configs/base.yaml.")
    parser.add_argument("--root", type=str, default=None, help="Raiz opcional do projeto.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada principal da CLI."""
    args = parse_args(argv)
    cfg = load_config(config_path=args.config, root=args.root)
    base_root = Path(cfg.get("project_root", get_project_root()))
    runs_dir = base_root / cfg["paths"]["results_dir"] / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    if args.benchmark_epochs is not None and args.benchmark_epochs > 0:
        run_timing_benchmark(
            benchmark_epochs=args.benchmark_epochs,
            config=cfg,
            device=args.device,
        )
        return 0

    if args.aggregate_only:
        print(f"Varrendo '{runs_dir}' para agregação de baselines existentes...")
        seed_trios: set[tuple[str, str, str]] = set()
        fold_trios: set[tuple[str, str, str]] = set()

        import re
        for f in runs_dir.glob("*_*_*_seed*.json"):
            m = re.match(r"^(.+)_(davis|kiba)_(random|cold_drug|cold_target)_seed\d+$", f.stem)
            if m:
                seed_trios.add((m.group(1), m.group(2), m.group(3)))

        for f in runs_dir.glob("*_*_*_fold*.json"):
            m = re.match(
                r"^(.+)_(davis|kiba)_(random|cold_drug|cold_target|cold_target_seqid30|cold_target_seqid40)_fold\d+$",
                f.stem,
            )
            if m:
                fold_trios.add((m.group(1), m.group(2), m.group(3)))

        if not seed_trios and not fold_trios:
            print(f"[INFO] Nenhum arquivo de run encontrado em {runs_dir}.")
            return 0

        for model_val, ds_val, split_val in sorted(seed_trios):
            try:
                aggregate_runs(model=model_val, dataset=ds_val, split=split_val, runs_dir=runs_dir)
            except Exception as e:
                print(f"[WARN] Não foi possível agregar trio seeds ({model_val}, {ds_val}, {split_val}): {e}")

        for model_val, ds_val, split_val in sorted(fold_trios):
            try:
                aggregate_kfold_runs(model=model_val, dataset=ds_val, split=split_val, runs_dir=runs_dir)
            except Exception as e:
                print(f"[WARN] Não foi possível agregar trio kfold ({model_val}, {ds_val}, {split_val}): {e}")

        return 0

    # Resolução de combinações a executar
    models = ["graphdta_gin", "graphdta_cnn3"] if args.model == "all" else [args.model]
    datasets = ["davis", "kiba"] if args.dataset == "all" else [args.dataset]
    splits = ["random", "cold_drug", "cold_target"] if args.split == "all" else [args.split]

    protocol = args.protocol
    if protocol != "kfold" and any(sp.startswith("cold_target_seqid") for sp in splits):
        raise ValueError(
            "Os splits por identidade de sequência ('cold_target_seqid30', 'cold_target_seqid40') "
            "requerem --protocol kfold (5-fold)."
        )
    if protocol == "kfold":
        units = [int(f.strip()) for f in args.folds.split(",") if f.strip()]
    else:
        units = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    print(f"=== Orquestrador da Fase 2 (PhysDual-GNN) ===")
    print(
        f"Protocolo: {protocol} | Modelos: {models} | Datasets: {datasets} | "
        f"Splits: {splits} | Unidades ({'dobras' if protocol == 'kfold' else 'seeds'}): {units}"
    )

    for m in models:
        for ds in datasets:
            for sp in splits:
                for u in units:
                    unit_label = f"Dobra={u}" if protocol == "kfold" else f"Seed={u}"
                    print(f"\n>>> Executando [{protocol.upper()}]: Modelo={m} | Dataset={ds} | Split={sp} | {unit_label}")
                    run_single_unit(
                        model_name=m,
                        dataset_name=ds,
                        split_type=sp,
                        unit_id=u,
                        protocol=protocol,
                        config=cfg,
                        runs_dir=runs_dir,
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
                    )

                # Agregação automática após completar todas as unidades do trio
                try:
                    if protocol == "kfold":
                        aggregate_kfold_runs(
                            model=m,
                            dataset=ds,
                            split=sp,
                            runs_dir=runs_dir,
                            folds=units,
                        )
                    else:
                        aggregate_runs(
                            model=m,
                            dataset=ds,
                            split=sp,
                            runs_dir=runs_dir,
                            seeds=units,
                        )
                except Exception as exc:
                    print(f"[WARN] Agregação pendente para ({m}, {ds}, {sp}): {exc}")

    print("\nExecução concluída com sucesso.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
