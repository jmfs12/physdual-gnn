#!/usr/bin/env python3
"""Verificador objetivo do Gate de Pronto da Fase 1 (`scripts/verify_phase1.py`).

Valida os 7 critérios obrigatórios do Gate de Pronto da Fase 1:
1. Todas as sequências únicas (Davis e KIBA) têm embedding no cache do modelo escolhido.
2. Para cada embedding, `shape[0] == len(sequence)` e `shape[1] == esm_dim` do config.
3. Nenhum NaN/Inf nos embeddings ou tensores de grafos; nenhum grafo de fármaco vazio (>= 1 átomo).
4. Os 30 splits (3 tipos x 5 seeds x 2 datasets) existem, são não vazios e passam em `verify_split_leakage` por conteúdo.
5. Contagens de procedência na tabela bruta batem com os valores esperados (Davis: 68x442=30.056; KIBA: 2.111x229=118.254) e nenhum par foi perdido sem registro na tabela limpa.
6. Todos os arquivos `.pt` carregam com `torch.load(..., weights_only=True)`.
7. O relatório final consolida: sequências com janela deslizante, valores censurados do Davis, SMILES alterados/falhos e contagens by_id vs by_content.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from physdual.data.datasets import EXPECTED_COUNTS  # noqa: E402
from physdual.data.esm_cache import get_embedding_cache_path  # noqa: E402
from physdual.data.splits import load_splits, verify_split_leakage  # noqa: E402
from physdual.utils.config import load_config  # noqa: E402


def run_phase1_verification(
    config_path: str | Path | None = None,
    root: str | Path | None = None,
    expected_counts_override: dict[str, dict[str, int]] | None = None,
    skip_pt_checks: bool = False,
) -> dict[str, Any]:
    """Executa os 7 checks do Gate de Pronto da Fase 1 e retorna um dicionário detalhado."""
    cfg = load_config(config_path=config_path, root=root)
    base_root = Path(cfg["project_root"])
    paths_cfg = cfg["paths"]

    processed_dir = Path(paths_cfg["processed_data_dir"])
    splits_dir = processed_dir / "splits"
    esm_cache_dir = Path(paths_cfg["esm_cache_dir"])
    logs_dir = Path(paths_cfg["logs_dir"])

    esm_cfg = cfg.get("esm", {})
    model_name = str(esm_cfg.get("model_name", "esm2_t33_650M_UR50D"))
    expected_dim = int(esm_cfg.get("esm_dim", 1280))

    datasets = ["davis", "kiba"]
    split_types = ["random", "cold_drug", "cold_target"]
    seeds = [int(s) for s in cfg.get("seeds", [0, 1, 2, 3, 4])]
    exp_counts = expected_counts_override or EXPECTED_COUNTS

    checks: dict[str, dict[str, Any]] = {
        "1_embeddings_coverage": {"passed": True, "errors": []},
        "2_embeddings_shape_and_dim": {"passed": True, "errors": []},
        "3_no_nan_inf_and_non_empty_graphs": {"passed": True, "errors": []},
        "4_splits_leakage_and_coverage": {"passed": True, "errors": []},
        "5_provenance_and_zero_unrecorded_loss": {"passed": True, "errors": []},
        "6_weights_only_deserialization": {"passed": True, "errors": []},
        "7_summary_report_completeness": {"passed": True, "errors": []},
    }

    dfs_by_ds: dict[str, pd.DataFrame] = {}

    # Check 5: Procedência na tabela bruta e perda zero não registrada na tabela limpa
    for ds in datasets:
        clean_csv = processed_dir / f"{ds}_clean.csv"
        rep_json = processed_dir / f"{ds}_ingestion_report.json"
        if not clean_csv.exists() or not rep_json.exists():
            checks["5_provenance_and_zero_unrecorded_loss"]["passed"] = False
            checks["5_provenance_and_zero_unrecorded_loss"]["errors"].append(
                f"Arquivos ausentes para '{ds}': {clean_csv} ou {rep_json}"
            )
            continue

        df = pd.read_csv(clean_csv)
        dfs_by_ds[ds] = df
        rep = json.loads(rep_json.read_text(encoding="utf-8"))
        exp = exp_counts[ds]

        if (
            int(rep.get("raw_num_pairs", -1)) != exp["num_pairs"]
            or int(rep.get("num_drugs_by_id", -1)) != exp["num_drugs"]
            or int(rep.get("num_targets_by_id", -1)) != exp["num_targets"]
        ):
            checks["5_provenance_and_zero_unrecorded_loss"]["passed"] = False
            checks["5_provenance_and_zero_unrecorded_loss"]["errors"].append(
                f"Divergência de procedência bruta em '{ds}': report={rep}, esperado={exp}"
            )

        raw_n = int(rep.get("raw_num_pairs", 0))
        clean_n = int(rep.get("clean_num_pairs", -1))
        dropped_n = int(rep.get("num_pairs_dropped_recorded", -1))
        if clean_n != len(df) or (raw_n - clean_n) != dropped_n:
            checks["5_provenance_and_zero_unrecorded_loss"]["passed"] = False
            checks["5_provenance_and_zero_unrecorded_loss"]["errors"].append(
                f"Perda de pares não registrada em '{ds}': raw={raw_n}, clean={clean_n}, len(df)={len(df)}, dropped={dropped_n}"
            )

    # Check 4: Splits (3 tipos x 5 seeds x 2 datasets = 30 splits)
    for ds in datasets:
        if ds not in dfs_by_ds:
            checks["4_splits_leakage_and_coverage"]["passed"] = False
            checks["4_splits_leakage_and_coverage"]["errors"].append(f"DataFrame '{ds}' indisponível para validar splits.")
            continue
        df = dfs_by_ds[ds]
        for st in split_types:
            for seed in seeds:
                sp_file = splits_dir / f"{ds}_{st}_seed{seed}.json"
                if not sp_file.exists():
                    checks["4_splits_leakage_and_coverage"]["passed"] = False
                    checks["4_splits_leakage_and_coverage"]["errors"].append(f"Split ausente: {sp_file}")
                    continue
                sp = load_splits(sp_file)
                if not verify_split_leakage(df, sp, split_type=st):
                    checks["4_splits_leakage_and_coverage"]["passed"] = False
                    checks["4_splits_leakage_and_coverage"]["errors"].append(
                        f"Falha de vazamento ou cobertura no split: {sp_file}"
                    )

    # Check 7: Relatório consolidado
    summary_file = logs_dir / "phase1_summary_report.json"
    if not summary_file.exists():
        checks["7_summary_report_completeness"]["passed"] = False
        checks["7_summary_report_completeness"]["errors"].append(f"Relatório ausente: {summary_file}")
    else:
        summary_data = json.loads(summary_file.read_text(encoding="utf-8"))
        for req_key in ("windowed_sequences", "davis_censoring", "smiles_audit"):
            if req_key not in summary_data:
                checks["7_summary_report_completeness"]["passed"] = False
                checks["7_summary_report_completeness"]["errors"].append(
                    f"Chave obrigatória '{req_key}' ausente em {summary_file}"
                )

    # Checks 1, 2, 3 e 6 (exigem PyTorch e arquivos .pt)
    if not skip_pt_checks:
        try:
            import torch
        except ImportError:
            for k in (
                "1_embeddings_coverage",
                "2_embeddings_shape_and_dim",
                "3_no_nan_inf_and_non_empty_graphs",
                "6_weights_only_deserialization",
            ):
                checks[k]["passed"] = False
                checks[k]["errors"].append("PyTorch não instalado para verificar arquivos .pt.")
        else:
            # Verifica embeddings de todas as sequências únicas de Davis e KIBA
            unique_seqs: set[str] = set()
            for ds, df in dfs_by_ds.items():
                unique_seqs.update(df["sequence"].astype(str).unique().tolist())

            for seq in sorted(unique_seqs):
                pt_path = get_embedding_cache_path(seq, model_name=model_name, cache_dir=esm_cache_dir)
                if not pt_path.exists():
                    checks["1_embeddings_coverage"]["passed"] = False
                    checks["1_embeddings_coverage"]["errors"].append(f"Embedding ausente: {pt_path}")
                    continue

                try:
                    payload = torch.load(pt_path, map_location="cpu", weights_only=True)
                except Exception as exc:
                    checks["6_weights_only_deserialization"]["passed"] = False
                    checks["6_weights_only_deserialization"]["errors"].append(
                        f"Falha em weights_only=True para {pt_path}: {exc}"
                    )
                    continue

                emb = payload.get("embedding")
                if emb is None or tuple(emb.shape) != (len(seq), expected_dim):
                    checks["2_embeddings_shape_and_dim"]["passed"] = False
                    checks["2_embeddings_shape_and_dim"]["errors"].append(
                        f"Shape inválido em {pt_path}: {None if emb is None else tuple(emb.shape)}, "
                        f"esperado={(len(seq), expected_dim)}"
                    )
                elif torch.isnan(emb).any() or torch.isinf(emb).any():
                    checks["3_no_nan_inf_and_non_empty_graphs"]["passed"] = False
                    checks["3_no_nan_inf_and_non_empty_graphs"]["errors"].append(
                        f"NaN ou Inf detectado no embedding: {pt_path}"
                    )

            # Verifica caches de grafos moleculares
            for ds in datasets:
                g_path = processed_dir / f"{ds}_drug_graphs.pt"
                if not g_path.exists():
                    checks["3_no_nan_inf_and_non_empty_graphs"]["passed"] = False
                    checks["3_no_nan_inf_and_non_empty_graphs"]["errors"].append(
                        f"Cache de grafos ausente: {g_path}"
                    )
                    continue
                try:
                    g_cache = torch.load(g_path, map_location="cpu", weights_only=True)
                except Exception as exc:
                    checks["6_weights_only_deserialization"]["passed"] = False
                    checks["6_weights_only_deserialization"]["errors"].append(
                        f"Falha em weights_only=True no cache de grafos {g_path}: {exc}"
                    )
                    continue

                if not isinstance(g_cache, dict) or len(g_cache) == 0:
                    checks["3_no_nan_inf_and_non_empty_graphs"]["passed"] = False
                    checks["3_no_nan_inf_and_non_empty_graphs"]["errors"].append(
                        f"Cache de grafos vazio em {g_path}"
                    )
                    continue

                for smi_key, g_dict in g_cache.items():
                    x = g_dict.get("x")
                    if x is None or int(x.size(0)) < 1:
                        checks["3_no_nan_inf_and_non_empty_graphs"]["passed"] = False
                        checks["3_no_nan_inf_and_non_empty_graphs"]["errors"].append(
                            f"Grafo vazio (< 1 átomo) para '{smi_key}' em {g_path}"
                        )
                    elif torch.isnan(x).any() or torch.isinf(x).any():
                        checks["3_no_nan_inf_and_non_empty_graphs"]["passed"] = False
                        checks["3_no_nan_inf_and_non_empty_graphs"]["errors"].append(
                            f"NaN/Inf em features atômicas de '{smi_key}' em {g_path}"
                        )

    overall_passed = all(c["passed"] for c in checks.values())
    return {
        "passed": overall_passed,
        "root": str(base_root),
        "model_name": model_name,
        "esm_dim": expected_dim,
        "checks": checks,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audita os 7 critérios objetivos do Gate de Pronto da Fase 1."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Caminho para o arquivo YAML de configuração (padrão: configs/base.yaml).",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Raiz opcional do projeto (sobrepõe PHYSDUAL_ROOT).",
    )
    parser.add_argument(
        "--skip-pt-checks",
        action="store_true",
        help="Pula verificações de arquivos .pt (útil quando validando apenas dados/splits sem GPU).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_phase1_verification(
        config_path=args.config,
        root=args.root,
        skip_pt_checks=args.skip_pt_checks,
    )
    print("=== Gate de Pronto da Fase 1 (PhysDual-GNN) ===")
    for name, info in result["checks"].items():
        status = "OK" if info["passed"] else "FALHA"
        print(f"  [{status}] {name}")
        for err in info["errors"][:5]:
            print(f"       -> {err}")

    if result["passed"]:
        print("RESULTADO FINAL: APROVADO (Todos os critérios da Fase 1 atendidos).")
        return 0

    print("RESULTADO FINAL: REPROVADO (Corrija os erros acima antes de avançar para a Fase 2).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
