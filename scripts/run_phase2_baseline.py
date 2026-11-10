#!/usr/bin/env python3
"""Orquestrador resumível, idempotente e agregador de baselines da Fase 2 (`scripts/run_phase2_baseline.py`).

Implementa as seguintes garantias de engenharia:
- C1: Isolamento estrito de teste (avaliação no teste realizada unicamente após o fim do treino).
- C2: Persistência individual por run atômica e retomabilidade granular por checkpoint.
- C3: Idempotência de execução (runs já concluídas são puladas sem reprocessamento, salvo com --force).
- Agregação desacoplada com cálculo estatístico de média e desvio-padrão (ddof=1).
- Benchmark de tempo de GPU/CPU para validação prévia de viabilidade de treinamento.
- Avaliação automática de baselines de referência triviais (mean_predictor e ridge_aa_composition).
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
from physdual.eval.metrics import compute_trivial_baselines  # noqa: E402
from physdual.models.graphdta import build_baseline_model  # noqa: E402
from physdual.training.trainer import train_model  # noqa: E402
from physdual.utils.config import load_config  # noqa: E402
from physdual.utils.paths import get_project_root  # noqa: E402
from physdual.utils.seed import set_seed  # noqa: E402

seed_everything = set_seed


def get_run_filename(model: str, dataset: str, split: str, seed: int) -> str:
    """Retorna o nome padronizado do arquivo JSON individual de uma run."""
    return f"{model}_{dataset}_{split}_seed{seed}.json"


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
    seed: int,
    runs_dir: Path | str | None = None,
) -> bool:
    """Verifica se o JSON individual de uma run já existe e é válido."""
    if runs_dir is None:
        cfg = load_config()
        runs_dir = Path(cfg["project_root"]) / cfg["paths"]["results_dir"] / "runs"
    else:
        runs_dir = Path(runs_dir)

    run_file = runs_dir / get_run_filename(model, dataset, split, seed)
    if not run_file.exists():
        return False

    try:
        data = json.loads(run_file.read_text(encoding="utf-8"))
        return "test_metrics_full" in data and "model" in data
    except Exception:
        return False


def _compute_baselines_from_loaders(
    train_loader: Any,
    test_loader: Any,
) -> dict[str, dict[str, float]]:
    """Calcula baselines triviais (mean_predictor e ridge_aa_composition) a partir dos DataLoaders."""
    train_dataset = getattr(train_loader, "dataset", None)
    test_dataset = getattr(test_loader, "dataset", None)

    train_df = getattr(train_dataset, "df", None)
    test_df = getattr(test_dataset, "df", None)

    if train_df is None or test_df is None:
        return {
            "mean_predictor": {"ci": 0.5, "mse": 1.0},
            "ridge_aa_composition": {"ci": 0.5, "mse": 1.0},
        }

    y_train = train_df["affinity"].values
    y_test = test_df["affinity"].values

    X_train_aa = np.stack([compute_aa_composition(seq) for seq in train_df["sequence"]])
    X_test_aa = np.stack([compute_aa_composition(seq) for seq in test_df["sequence"]])

    return compute_trivial_baselines(y_train, y_test, X_train_aa, X_test_aa)


def run_single_seed(
    model_name: str,
    dataset_name: str,
    split_type: str,
    seed: int,
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
) -> dict[str, Any]:
    """Executa o treinamento de uma seed individual de forma idempotente e atômica.

    Args:
        model_name: Nome do modelo ('graphdta_gin', 'graphdta_cnn3').
        dataset_name: Nome do dataset ('davis', 'kiba').
        split_type: Partição ('random', 'cold_drug', 'cold_target').
        seed: Número da semente pseudoaleatória (0 a 4).
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

    Returns:
        Dicionário com todos os metadados e métricas persistidos na run.
    """
    cfg = copy.deepcopy(config) if config is not None else load_config()
    base_root = Path(cfg.get("project_root", get_project_root()))
    paths_cfg = cfg["paths"]

    if runs_dir is None:
        runs_dir = base_root / paths_cfg["results_dir"] / "runs"
    else:
        runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_file = runs_dir / get_run_filename(model_name, dataset_name, split_type, seed)

    # -------------------------------------------------------------------------
    # Idempotência C3: Pula runs concluídas
    # -------------------------------------------------------------------------
    if not force and is_run_completed(model_name, dataset_name, split_type, seed, runs_dir=runs_dir):
        print(f"[SKIP] Run já concluída: {run_file}")
        try:
            cached_data = json.loads(run_file.read_text(encoding="utf-8"))
            cached_data["skipped"] = True
            return cached_data
        except Exception:
            pass

    # Fixação determinística de seeds
    seed_everything(seed)

    # Resolução de dispositivo
    if device == "auto":
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    # Configurações de hiperparâmetros
    gdta_cfg = cfg.get("graphdta", {})
    train_cfg = gdta_cfg.get("training", {})

    eff_epochs = epochs if epochs is not None else int(train_cfg.get("epochs", 500))
    eff_patience = patience if patience is not None else int(train_cfg.get("patience", 50))
    eff_davis_cold_pat = davis_cold_patience if davis_cold_patience is not None else int(train_cfg.get("davis_cold_patience", 80))
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

    cfg["graphdta"]["training"]["batch_size"] = eff_batch_size
    cfg["graphdta"]["training"]["num_workers"] = eff_workers

    # Diretório de checkpoints local por run
    if checkpoints_base_dir is None:
        checkpoints_base_dir = base_root / paths_cfg["checkpoints_dir"]
    else:
        checkpoints_base_dir = Path(checkpoints_base_dir)

    run_ckpt_dir = checkpoints_base_dir / f"{model_name}_{dataset_name}_{split_type}_seed{seed}"
    run_ckpt_dir.mkdir(parents=True, exist_ok=True)

    run_drive_ckpt_dir: Path | None = None
    if drive_checkpoint_dir is not None:
        run_drive_ckpt_dir = Path(drive_checkpoint_dir) / f"{model_name}_{dataset_name}_{split_type}_seed{seed}"

    # Carregamento de DataLoaders
    train_loader, val_loader, test_loader = get_dta_dataloaders(
        dataset_name=dataset_name,
        split_type=split_type,
        seed=seed,
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

    # Cálculo dos baselines triviais
    trivial_baselines = _compute_baselines_from_loaders(train_loader, test_loader)

    feat_cfg = cfg.get("featurizer", {})
    feat_version = str(feat_cfg.get("version", "v2_onehot78_l1norm"))
    norm_atoms = bool(feat_cfg.get("normalize_atom_features", True))
    drug_in_dim = int(gdta_cfg.get("drug_in_dim", 78))

    run_record: dict[str, Any] = {
        "model": model_name,
        "dataset": dataset_name,
        "split": split_type,
        "seed": seed,
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
    }

    save_json_atomic(run_record, run_file)
    print(f"[SAVE] Run salva com sucesso: {run_file} (best_epoch={run_record['best_epoch']}, sec_per_epoch={sec_per_epoch:.2f}s)")
    return run_record


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
        run_file = runs_path / get_run_filename(model, dataset, split, s)
        if run_file.exists():
            try:
                data = json.loads(run_file.read_text(encoding="utf-8"))
                per_seed_results[str(s)] = data
                found_runs.append(data)
            except Exception as e:
                print(f"[WARN] Erro ao carregar {run_file}: {e}")

    if not found_runs:
        raise FileNotFoundError(
            f"Nenhuma run encontrada em '{runs_path}' para o trio ({model}, {dataset}, {split})."
        )

    def _calc_stats(values: list[float | int | None]) -> dict[str, float]:
        valid = [float(v) for v in values if v is not None and not np.isnan(float(v))]
        if not valid:
            return {"mean": float("nan"), "std": float("nan")}
        arr = np.array(valid, dtype=np.float64)
        std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        return {
            "mean": float(np.mean(arr)),
            "std": std_val,
        }

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

    best_epochs = [r.get("best_epoch") for r in found_runs]
    sec_per_epochs = [r.get("sec_per_epoch") for r in found_runs]
    total_epochs = [r.get("total_epochs_run") for r in found_runs]
    elapsed_secs = [r.get("elapsed_sec") for r in found_runs]

    aggregated_payload: dict[str, Any] = {
        "test_metrics_full": agg_full,
        "test_metrics_uncensored": agg_uncensored,
        "test_metrics_by_seq_len": agg_by_seq,
        "trivial_baselines": agg_trivial,
        "best_epoch": _calc_stats(best_epochs),
        "sec_per_epoch": _calc_stats(sec_per_epochs),
        "total_epochs_run": _calc_stats(total_epochs),
        "elapsed_sec": _calc_stats(elapsed_secs),
    }

    result = {
        "model": model,
        "dataset": dataset,
        "split": split,
        "num_seeds": len(found_runs),
        "seeds": [int(s) for s in per_seed_results.keys()],
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
    print(f"[AGGREGATE] Resultados consolidados salvos em: {out_path} ({len(found_runs)}/{len(target_seeds)} seeds)")
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
            model.to(target_device)
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
            criterion = torch.nn.MSELoss()

            times = []
            for ep in range(1, benchmark_epochs + 1):
                t0 = time.time()
                model.train()
                for batch in train_loader:
                    drug_batch = batch["drug_batch"].to(target_device)
                    target_tokens = batch["target_tokens"].to(target_device)
                    targets = batch["affinity"].to(target_device)
                    optimizer.zero_grad()
                    out = model(drug_batch, target_tokens)
                    loss = criterion(out.view(-1), targets.view(-1))
                    loss.backward()
                    optimizer.step()
                ep_sec = time.time() - t0
                times.append(ep_sec)
                print(f"  Época {ep}/{benchmark_epochs}: {ep_sec:.2f}s")

            sec_per_epoch = float(np.mean(times))
            est_hours_per_seed = (sec_per_epoch * 500) / 3600.0
            est_hours_15_runs = est_hours_per_seed * 15.0  # 3 splits x 5 seeds
            eligible_for_cnn3 = bool(sec_per_epoch <= 15.0)

            benchmark_results["datasets"][ds] = {
                "sec_per_epoch": round(sec_per_epoch, 3),
                "est_hours_500_epochs_single_run": round(est_hours_per_seed, 2),
                "est_hours_15_runs": round(est_hours_15_runs, 2),
                "eligible_for_cnn3_secondary": eligible_for_cnn3,
            }
            print(f"  Média: {sec_per_epoch:.2f}s/época | 15 runs: ~{est_hours_15_runs:.1f}h")
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
        choices=["random", "cold_drug", "cold_target", "all"],
        help="Tipo de partição a executar ou 'all'.",
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
        help="Lista de seeds separadas por vírgula (padrão: '0,1,2,3,4').",
    )
    parser.add_argument("--epochs", type=int, default=500, help="Número máximo de épocas.")
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
        print(f"Varrendo '{runs_dir}' para agregação de todos os baselines existentes...")
        # Localiza todos os trios únicos (model, dataset, split) em runs_dir
        found_trios: set[tuple[str, str, str]] = set()
        for f in runs_dir.glob("*_*_*_seed*.json"):
            name = f.stem
            parts = name.split("_")
            # modelo pode ser graphdta_gin ou graphdta_cnn3
            if len(parts) >= 4:
                split_val = parts[-2]
                ds_val = parts[-3]
                model_val = "_".join(parts[:-3])
                found_trios.add((model_val, ds_val, split_val))

        if not found_trios:
            print(f"[INFO] Nenhum arquivo de run encontrado em {runs_dir}.")
            return 0

        for model_val, ds_val, split_val in sorted(found_trios):
            try:
                aggregate_runs(
                    model=model_val,
                    dataset=ds_val,
                    split=split_val,
                    runs_dir=runs_dir,
                )
            except Exception as e:
                print(f"[WARN] Não foi possível agregar trio ({model_val}, {ds_val}, {split_val}): {e}")
        return 0

    # Resolução de combinações a executar
    models = ["graphdta_gin", "graphdta_cnn3"] if args.model == "all" else [args.model]
    datasets = ["davis", "kiba"] if args.dataset == "all" else [args.dataset]
    splits = ["random", "cold_drug", "cold_target"] if args.split == "all" else [args.split]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    print(f"=== Orquestrador da Fase 2 (PhysDual-GNN) ===")
    print(f"Modelos: {models} | Datasets: {datasets} | Splits: {splits} | Seeds: {seeds}")

    for m in models:
        for ds in datasets:
            for sp in splits:
                for s in seeds:
                    print(f"\n>>> Executando: Modelo={m} | Dataset={ds} | Split={sp} | Seed={s}")
                    run_single_seed(
                        model_name=m,
                        dataset_name=ds,
                        split_type=sp,
                        seed=s,
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
                    )
                # Agrega automaticamente o trio após iterar as seeds
                try:
                    aggregate_runs(
                        model=m,
                        dataset=ds,
                        split=sp,
                        runs_dir=runs_dir,
                        seeds=seeds,
                    )
                except Exception as exc:
                    print(f"[WARN] Agregação pendente para ({m}, {ds}, {sp}): {exc}")

    print("\nExecução concluída com sucesso.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
