"""Script diagnóstico para Google Colab: `ecfp_esm_mlp` em `kiba/cold_drug` (v1_seeds).

AVISO METODOLÓGICO (ADR #15):
  Este script tem caráter ESTRITAMENTE DIAGNÓSTICO (não confirmatório / pós-registro).
  Na Fase 3, os 2 regimes primários (`kiba/cold_drug`, `kiba/cold_target`) e 2 secundários
  (`kiba/cold_target_seqid40`, `kiba/cold_target_seqid30`) pré-registrados no ADR #15 já
  obtiveram veredito negativo com `gin_esm_mlp`. Este controle diagnostica se a perda de
  `gin_esm_mlp` para `knn_tanimoto_drug` e `graphdta_gin` em `kiba/cold_drug` decorre do
  codificador GIN de fármacos (comparando contra `physdual_ecfp_esm_mlp`, que substitui o
  ramo GIN por ECFP4 1024d) ou do acoplamento MLP/cabeça de interação.

Uso típico no Colab (GPU):
  python scripts/paper/colab_kiba_cold_drug_ecfp.py \\
    --root /content/drive/MyDrive/physdual-gnn \\
    --esm-cache-dir /content/drive/MyDrive/physdual-gnn/cache/esm_embeddings \\
    --device cuda --epochs 1000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PAPER_SCRIPTS_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = PAPER_SCRIPTS_DIR.parent
REPO_ROOT = PAPER_SCRIPTS_DIR.parents[1]
if str(PAPER_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_SCRIPTS_DIR))

from common import load_json_strict, resolve_roots, sample_mean_std  # noqa: E402


def build_train_command(args: argparse.Namespace, data_root: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_phase3_physdual.py"),
        "--model",
        "physdual_ecfp_esm_mlp",
        "--protocol",
        "seeds",
        "--dataset",
        "kiba",
        "--split",
        "cold_drug",
        "--seeds",
        args.seeds,
        "--epochs",
        str(args.epochs),
        "--device",
        args.device,
        "--root",
        str(data_root),
    ]
    if args.esm_cache_dir:
        cmd.extend(["--esm-cache-dir", str(args.esm_cache_dir)])
    if args.force:
        cmd.append("--force")
    if args.extend_epochs:
        cmd.append("--extend-epochs")
    return cmd


def _extract_test_metrics(run_payload: dict[str, Any]) -> dict[str, Any]:
    metrics = run_payload.get("test_metrics_full")
    if isinstance(metrics, dict):
        return metrics
    metrics_alt = run_payload.get("test_metrics")
    if isinstance(metrics_alt, dict):
        return metrics_alt
    raise KeyError("Run JSON não contém 'test_metrics_full' nem 'test_metrics'.")


def _load_knn_tanimoto_by_seed(res_dir: Path) -> dict[int, dict[str, Any]]:
    kiba_cls_path = res_dir / "kiba_cpu_classical_baselines.json"
    if kiba_cls_path.exists():
        cls_agg = load_json_strict(kiba_cls_path)
        per_unit = cls_agg["regimes"]["cold_drug"]["per_unit"]
        return {int(r["seed"]): r["knn_tanimoto_drug"]["metrics"] for r in per_unit}
    legacy_path = res_dir / "baseline_classical_kiba_cold_drug_seeds.json"
    if legacy_path.exists():
        cls_agg = load_json_strict(legacy_path)
        return {int(r["seed"]): r["test_metrics"] for r in cls_agg["per_seed_results"]["knn_tanimoto_drug"]}
    raise FileNotFoundError(
        f"Arquivo de baselines clássicos do KIBA ausente: {kiba_cls_path}"
    )


def summarize_diagnostic_comparison(data_root: Path, seeds: list[int]) -> dict[str, Any]:
    """Lê os artefatos existentes em `outputs/results/` e compara `ecfp_esm_mlp` vs referências."""
    runs_dir = data_root / "outputs" / "results" / "runs"
    res_dir = data_root / "outputs" / "results"

    ecfp_by_seed: dict[int, dict[str, Any]] = {}
    gin_by_seed: dict[int, dict[str, Any]] = {}
    gdta_by_seed: dict[int, dict[str, Any]] = {}

    for s in seeds:
        ecfp_p = runs_dir / f"physdual_ecfp_esm_mlp_kiba_cold_drug_seed{s}.json"
        if not ecfp_p.exists():
            raise FileNotFoundError(f"Run diagnóstico ausente: {ecfp_p}")
        ecfp_by_seed[s] = load_json_strict(ecfp_p)

        gin_p = runs_dir / f"physdual_gin_esm_mlp_kiba_cold_drug_seed{s}.json"
        if gin_p.exists():
            gin_by_seed[s] = load_json_strict(gin_p)

        gdta_p = runs_dir / f"graphdta_gin_kiba_cold_drug_seed{s}.json"
        if gdta_p.exists():
            gdta_by_seed[s] = load_json_strict(gdta_p)

    knn_seeds = _load_knn_tanimoto_by_seed(res_dir)

    per_seed_table: list[dict[str, Any]] = []
    for s in seeds:
        er = ecfp_by_seed[s]
        em = _extract_test_metrics(er)
        ep_max = int(er.get("effective_config", {}).get("epochs", 1000))
        b_ep = int(er["best_epoch"])
        row: dict[str, Any] = {
            "seed": s,
            "ecfp_best_epoch": b_ep,
            "ecfp_epochs_ceiling": ep_max,
            "ecfp_over_80pct": bool(b_ep > 0.8 * ep_max),
            "ecfp_early_collapse_lt15": bool(b_ep < 15),
            "ecfp_ci": float(em["ci"]),
            "ecfp_mse": float(em["mse"]),
            "ecfp_rm2": float(em["rm2"]),
            "knn_tanimoto_ci": float(knn_seeds[s]["ci"]),
            "knn_tanimoto_mse": float(knn_seeds[s]["mse"]),
        }
        if s in gin_by_seed:
            gm = _extract_test_metrics(gin_by_seed[s])
            row["gin_ci"] = float(gm["ci"])
            row["gin_mse"] = float(gm["mse"])
            row["delta_ecfp_minus_gin_ci"] = float(em["ci"]) - float(gm["ci"])
            row["delta_ecfp_minus_gin_mse"] = float(em["mse"]) - float(gm["mse"])
        if s in gdta_by_seed:
            dm = _extract_test_metrics(gdta_by_seed[s])
            row["graphdta_gin_ci"] = float(dm["ci"])
            row["graphdta_gin_mse"] = float(dm["mse"])
        per_seed_table.append(row)

    summary = {
        "diagnostic_only": True,
        "adr15_status": "NÃO CONFIRMATÓRIO (diagnóstico pós-ADR #15 do ramo de fármaco em kiba/cold_drug)",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": "kiba",
        "split": "cold_drug",
        "protocol": "v1_seeds",
        "seeds": seeds,
        "ecfp_summary": {
            "ci": sample_mean_std([r["ecfp_ci"] for r in per_seed_table]),
            "mse": sample_mean_std([r["ecfp_mse"] for r in per_seed_table]),
            "rm2": sample_mean_std([r["ecfp_rm2"] for r in per_seed_table]),
            "collapses_lt15": int(sum(1 for r in per_seed_table if r["ecfp_early_collapse_lt15"])),
            "over_80pct_ceiling": int(sum(1 for r in per_seed_table if r["ecfp_over_80pct"])),
        },
        "wins_ecfp_vs_gin_ci": int(sum(1 for r in per_seed_table if r.get("delta_ecfp_minus_gin_ci", 0.0) > 0.0)),
        "wins_ecfp_vs_gin_mse": int(sum(1 for r in per_seed_table if r.get("delta_ecfp_minus_gin_mse", 0.0) < 0.0)),
        "wins_ecfp_vs_knn_tanimoto_ci": int(sum(1 for r in per_seed_table if r["ecfp_ci"] > r["knn_tanimoto_ci"])),
        "wins_ecfp_vs_knn_tanimoto_mse": int(sum(1 for r in per_seed_table if r["ecfp_mse"] < r["knn_tanimoto_mse"])),
        "per_seed": per_seed_table,
    }
    out_path = data_root / "outputs" / "results" / "phase3_colab_kiba_cold_drug_ecfp_diagnostic.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Resumo diagnóstico salvo em: {out_path}", flush=True)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnóstico Colab: physdual_ecfp_esm_mlp em kiba/cold_drug (pós-ADR #15)"
    )
    parser.add_argument("--root", type=Path, default=None, help="Raiz de dados/outputs no Drive")
    parser.add_argument("--esm-cache-dir", type=Path, default=None)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--extend-epochs", action="store_true")
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Não executa treino; apenas agrega e compara JSONs já existentes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Apenas imprime o comando exato de execução sem treinar.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _, default_data_root = resolve_roots()
    data_root = args.root.resolve() if args.root else default_data_root
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    cmd = build_train_command(args, data_root)
    if args.dry_run:
        print("[DRY-RUN] Comando diagnóstico (NÃO confirmatório sob ADR #15):")
        print(" ".join(cmd))
        return 0

    if not args.compare_only:
        print("[DIAGNOSTIC-ONLY] Executando physdual_ecfp_esm_mlp em kiba/cold_drug...", flush=True)
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
        if proc.returncode != 0:
            return int(proc.returncode)

    summary = summarize_diagnostic_comparison(data_root, seeds)
    ci_m, ci_s = summary["ecfp_summary"]["ci"]
    mse_m, mse_s = summary["ecfp_summary"]["mse"]
    print(
        f"[DIAG-RESULT] ecfp_esm_mlp kiba/cold_drug: CI={ci_m:.4f}+-{ci_s:.4f} | "
        f"MSE={mse_m:.4f}+-{mse_s:.4f} | vs_gin_wins={summary['wins_ecfp_vs_gin_ci']}/{len(seeds)} CI"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
