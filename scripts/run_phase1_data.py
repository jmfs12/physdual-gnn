#!/usr/bin/env python3
"""Script orquestrador CLI da Fase 1 (Dados, Splits, Grafos PyG e Cache ESM-2)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from physdual.data.datasets import prepare_dataset  # noqa: E402
from physdual.data.esm_cache import (  # noqa: E402
    create_cache_tarball,
    extract_esm_embeddings_batch,
)
from physdual.data.featurizer_drug import (  # noqa: E402
    build_drug_graph_cache,
    get_drug_graph_cache_filename,
)
from physdual.data.splits import (  # noqa: E402
    create_group_kfold_splits,
    create_splits,
    save_splits,
    verify_group_kfold_coverage,
    verify_split_leakage,
)
from physdual.utils.config import load_config  # noqa: E402
from physdual.utils.paths import ensure_project_dirs  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa o pipeline completo da Fase 1 (Davis/KIBA, Splits, Grafos PyG e Cache ESM-2)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Dataset(s) a processar: 'davis', 'kiba', 'all' ou lista separada por vírgula (padrão: all).",
    )
    parser.add_argument(
        "--affinity-unit",
        type=str,
        default="auto",
        choices=["auto", "nM", "pKd", "kiba_score"],
        help="Unidade de afinidade do arquivo de entrada (padrão: auto, detectado no nível do dataset inteiro).",
    )
    parser.add_argument(
        "--data-source",
        type=str,
        default="auto",
        choices=["auto", "canonical", "kaggle"],
        help="Fonte dos dados brutos: 'canonical' (DeepDTA oficial, sem login), 'kaggle' (kagglehub) ou 'auto' (padrão: canonical com fallback).",
    )
    parser.add_argument(
        "--duplicate-policy",
        type=str,
        default="keep_both",
        choices=["keep_both", "mean", "median"],
        help="Política para pares com mesmo (smiles, sequence) vindos de IDs distintos (padrão: keep_both).",
    )
    parser.add_argument(
        "--esm-model",
        type=str,
        default=None,
        help="Nome do modelo ESM-2 (sobrepõe configs/base.yaml, padrão: esm2_t33_650M_UR50D).",
    )
    parser.add_argument(
        "--batch-tokens",
        type=int,
        default=None,
        help="Orçamento máximo de tokens por lote na extração do ESM-2 (padrão: 2048).",
    )
    parser.add_argument(
        "--skip-esm",
        action="store_true",
        help="Pula a extração em GPU dos embeddings do ESM-2 (útil para validação rápida de dados/splits).",
    )
    parser.add_argument(
        "--only-drug-graphs",
        action="store_true",
        help="Gera exclusivamente o cache de grafos de fármacos (v2) sem reexecutar splits nem extração ESM-2.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Caminho opcional para arquivo YAML de configuração (padrão: configs/base.yaml).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Dispositivo PyTorch para extração do ESM-2 ('cuda' ou 'cpu').",
    )
    return parser.parse_args(argv)


def _resolve_datasets_arg(ds_arg: str) -> list[str]:
    val = ds_arg.strip().lower()
    if val == "all":
        return ["davis", "kiba"]
    items = [x.strip().lower() for x in val.split(",") if x.strip()]
    for item in items:
        if item not in ("davis", "kiba"):
            raise ValueError(f"Dataset inválido: '{item}'. Use 'davis', 'kiba' ou 'all'.")
    return items


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    root = Path(cfg["project_root"])
    ensure_project_dirs(root)

    paths_cfg = cfg["paths"]
    raw_dir = Path(paths_cfg["raw_data_dir"])
    processed_dir = Path(paths_cfg["processed_data_dir"])
    esm_cache_dir = Path(paths_cfg["esm_cache_dir"])
    logs_dir = Path(paths_cfg["logs_dir"])
    logs_dir.mkdir(parents=True, exist_ok=True)

    esm_cfg = cfg.get("esm", {})
    model_name = args.esm_model or str(esm_cfg.get("model_name", "esm2_t33_650M_UR50D"))
    batch_tokens = args.batch_tokens or int(esm_cfg.get("batch_tokens", 2048))
    max_len = int(esm_cfg.get("max_len", 1022))
    overlap = int(esm_cfg.get("overlap", 256))

    seeds: list[int] = [int(s) for s in cfg.get("seeds", [0, 1, 2, 3, 4])]
    split_types: list[str] = list(cfg.get("splits", {}).get("available", ["random", "cold_drug", "cold_target"]))
    ratios_cfg = cfg.get("splits", {}).get("ratios", {"train": 0.8, "val": 0.1, "test": 0.1})
    ratios = (float(ratios_cfg["train"]), float(ratios_cfg["val"]), float(ratios_cfg["test"]))

    feat_cfg = cfg.get("featurizer", {})
    feat_version = str(feat_cfg.get("version", "v2_onehot78_l1norm"))
    norm_atoms = bool(feat_cfg.get("normalize_atom_features", True))

    datasets_to_run = _resolve_datasets_arg(args.dataset)

    if args.only_drug_graphs:
        import pandas as pd

        print(f"[Fase 1] Modo --only-drug-graphs ativo: gerando cache de grafos {feat_version} (norm={norm_atoms}) ...")
        for ds in datasets_to_run:
            clean_path = processed_dir / f"{ds}_clean.csv"
            if clean_path.exists():
                df = pd.read_csv(clean_path)
            else:
                unit_arg = "kiba_score" if (ds == "kiba" and args.affinity_unit == "auto") else args.affinity_unit
                df, _, _ = prepare_dataset(
                    dataset_name=ds,
                    affinity_unit=unit_arg,  # type: ignore[arg-type]
                    save_clean=True,
                    duplicate_policy=args.duplicate_policy,  # type: ignore[arg-type]
                    raw_dir=raw_dir,
                    processed_dir=processed_dir,
                    source=args.data_source,  # type: ignore[arg-type]
                    use_kaggle=(args.data_source == "kaggle"),
                    enforce_expected_counts=True,
                )
            unique_drugs_df = df[["drug_id", "smiles"]].drop_duplicates(subset=["smiles"])
            cache_name = get_drug_graph_cache_filename(ds, feat_version)
            graph_cache_path = processed_dir / cache_name
            _, report_versioned = build_drug_graph_cache(
                smiles_list=unique_drugs_df["smiles"].tolist(),
                output_path=graph_cache_path,
                drug_ids=unique_drugs_df["drug_id"].tolist(),
                featurizer_version=feat_version,  # type: ignore[arg-type]
                normalize_atom_features=norm_atoms,
            )
            legacy_cache_path = processed_dir / f"{ds}_drug_graphs.pt"
            build_drug_graph_cache(
                smiles_list=unique_drugs_df["smiles"].tolist(),
                output_path=legacy_cache_path,
                drug_ids=unique_drugs_df["drug_id"].tolist(),
                featurizer_version=feat_version,  # type: ignore[arg-type]
                normalize_atom_features=norm_atoms,
            )
            print(
                f"  -> Cache de grafos salvo em {graph_cache_path} "
                f"(sucesso={report_versioned['num_succeeded']}, falhas={report_versioned['num_failed']})"
            )
        return 0

    summary_path = logs_dir / "phase1_summary_report.json"
    existing_datasets: dict[str, Any] = {}
    if summary_path.exists():
        try:
            prev = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(prev.get("datasets"), dict):
                existing_datasets = dict(prev["datasets"])
        except Exception:
            existing_datasets = {}

    summary_report: dict[str, Any] = {
        "model_name": model_name,
        "esm_dim": int(esm_cfg.get("esm_dim", 1280)),
        "max_len": max_len,
        "overlap": overlap,
        "datasets": existing_datasets,
        "windowed_sequences": {"count": 0, "target_ids": []},
        "davis_censoring": {"censored_count": 0, "censored_pct": 0.0},
        "smiles_audit": {"modified_count": 0, "failed_count": 0, "failures": []},
    }

    all_protein_sequences: list[tuple[str, str]] = []
    windowed_target_ids: list[str] = []

    for ds in datasets_to_run:
        print(f"[Fase 1] Processando dataset: {ds.upper()} ...")
        unit_arg = "kiba_score" if (ds == "kiba" and args.affinity_unit == "auto") else args.affinity_unit
        df, clean_path, ing_report = prepare_dataset(
            dataset_name=ds,
            affinity_unit=unit_arg,  # type: ignore[arg-type]
            save_clean=True,
            duplicate_policy=args.duplicate_policy,  # type: ignore[arg-type]
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            source=args.data_source,  # type: ignore[arg-type]
            use_kaggle=(args.data_source == "kaggle"),
            enforce_expected_counts=True,
        )
        print(
            f"  -> Dataset padronizado salvo em {clean_path} "
            f"(pares={ing_report['clean_num_pairs']}, "
            f"fármacos ID/SMILES={ing_report['num_drugs_by_id']}/{ing_report['num_drugs_by_canonical_smiles']}, "
            f"alvos ID/Seq={ing_report['num_targets_by_id']}/{ing_report['num_targets_by_sequence']})"
        )

        if ds == "davis":
            summary_report["davis_censoring"] = {
                "censored_count": ing_report["davis_censored_count"],
                "censored_pct": ing_report["davis_censored_pct"],
            }

        summary_report["smiles_audit"]["modified_count"] += int(ing_report["smiles_modified_count"])
        summary_report["smiles_audit"]["failed_count"] += int(ing_report["smiles_failed_count"])
        summary_report["smiles_audit"]["failures"].extend(ing_report["smiles_failures"])

        # Geração e auditoria dos splits (3 tipos x 5 seeds)
        splits_dir = processed_dir / "splits"
        split_stats: dict[str, Any] = {}
        for st in split_types:
            split_stats[st] = {}
            for seed in seeds:
                sp = create_splits(df, split_type=st, seed=seed, ratios=ratios)
                if not verify_split_leakage(df, sp, split_type=st):
                    raise RuntimeError(f"Vazamento detectado no split {ds} / {st} / seed {seed}!")
                sp_path = save_splits(sp, splits_dir / f"{ds}_{st}_seed{seed}.json")
                split_stats[st][f"seed{seed}"] = {
                    "path": str(sp_path.relative_to(root)),
                    "train_pairs": len(sp["train"]),
                    "val_pairs": len(sp["val"]),
                    "test_pairs": len(sp["test"]),
                    "train_unique_drugs": int(df.iloc[sp["train"]]["smiles"].nunique()),
                    "val_unique_drugs": int(df.iloc[sp["val"]]["smiles"].nunique()),
                    "test_unique_drugs": int(df.iloc[sp["test"]]["smiles"].nunique()),
                    "train_unique_targets": int(df.iloc[sp["train"]]["sequence"].nunique()),
                    "val_unique_targets": int(df.iloc[sp["val"]]["sequence"].nunique()),
                    "test_unique_targets": int(df.iloc[sp["test"]]["sequence"].nunique()),
                }
        print(f"  -> {len(split_types) * len(seeds)} splits gerados e validados sem vazamento por conteúdo.")

        # Geração de splits GroupKFold (5 dobras) para regimes cold
        kfold_stats: dict[str, Any] = {}
        for st in ["cold_drug", "cold_target"]:
            folds = create_group_kfold_splits(
                df,
                split_type=st,
                n_splits=5,
                val_ratio=0.1,
                assignment_seed=42,
            )
            if not verify_group_kfold_coverage(df, folds, split_type=st):
                raise RuntimeError(f"Falha de cobertura ou disjunção no GroupKFold {ds} / {st}!")
            kfold_stats[st] = {}
            for fold in folds:
                f_id = fold["fold_id"]
                f_path = save_splits(fold, splits_dir / f"{ds}_{st}_fold{f_id}.json")
                kfold_stats[st][f"fold{f_id}"] = {
                    "path": str(f_path.relative_to(root)),
                    "train_pairs": len(fold["train"]),
                    "val_pairs": len(fold["val"]),
                    "test_pairs": len(fold["test"]),
                    "num_test_groups": fold["num_test_groups"],
                }
        summary_report["splits_kfold"] = kfold_stats
        print(f"  -> 10 splits GroupKFold (5 dobras para cold_drug e cold_target) gerados e validados.")

        # Cache de grafos de fármacos (weights_only=True)
        unique_drugs_df = df[["drug_id", "smiles"]].drop_duplicates(subset=["smiles"])
        cache_name = get_drug_graph_cache_filename(ds, feat_version)
        graph_cache_path = processed_dir / cache_name
        _, graph_report = build_drug_graph_cache(
            smiles_list=unique_drugs_df["smiles"].tolist(),
            output_path=graph_cache_path,
            drug_ids=unique_drugs_df["drug_id"].tolist(),
            featurizer_version=feat_version,  # type: ignore[arg-type]
            normalize_atom_features=norm_atoms,
        )
        legacy_cache_path = processed_dir / f"{ds}_drug_graphs.pt"
        build_drug_graph_cache(
            smiles_list=unique_drugs_df["smiles"].tolist(),
            output_path=legacy_cache_path,
            drug_ids=unique_drugs_df["drug_id"].tolist(),
            featurizer_version=feat_version,  # type: ignore[arg-type]
            normalize_atom_features=norm_atoms,
        )
        print(
            f"  -> Cache de grafos moleculares salvo em {graph_cache_path} "
            f"(versão={feat_version}, sucesso={graph_report['num_succeeded']}, falhas={graph_report['num_failed']})"
        )

        # Coleta sequências únicas para extração do ESM-2 e contabiliza janelas (> max_len)
        unique_targets_df = df[["target_id", "sequence"]].drop_duplicates(subset=["sequence"])
        for _, row in unique_targets_df.iterrows():
            tid = str(row["target_id"])
            seq = str(row["sequence"])
            all_protein_sequences.append((tid, seq))
            if len(seq) > max_len:
                windowed_target_ids.append(tid)

        summary_report["datasets"][ds] = {
            "ingestion": ing_report,
            "drug_graphs": graph_report,
            "splits": split_stats,
        }

    summary_report["windowed_sequences"] = {
        "count": len(windowed_target_ids),
        "target_ids": windowed_target_ids,
    }

    if not args.skip_esm and all_protein_sequences:
        print(f"[Fase 1] Extraindo embeddings do ESM-2 ({model_name}) para sequências únicas ...")
        _, esm_report = extract_esm_embeddings_batch(
            sequences=all_protein_sequences,
            model_name=model_name,
            cache_dir=esm_cache_dir,
            batch_tokens=batch_tokens,
            max_len=max_len,
            overlap=overlap,
            device=args.device,
        )
        tar_path = create_cache_tarball(esm_cache_dir, model_name=model_name)
        summary_report["esm_extraction"] = esm_report
        summary_report["esm_tarball"] = str(tar_path.relative_to(root))
        summary_report["windowed_sequences"] = {
            "count": esm_report["num_windowed_sequences"],
            "target_ids": esm_report["windowed_target_ids"],
        }
        print(
            f"  -> Extração ESM-2 concluída (total={esm_report['total_sequences']}, "
            f"janeladas={esm_report['num_windowed_sequences']}). Tarball: {tar_path}"
        )

    summary_path = logs_dir / "phase1_summary_report.json"
    summary_path.write_text(json.dumps(summary_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Fase 1] Relatório consolidado salvo em: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
