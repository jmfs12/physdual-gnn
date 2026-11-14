#!/usr/bin/env python3
"""Audita OOFs locais, verifica reprodução dos baselines e executa bootstrap agrupado pareado (Davis/KIBA Fase 3)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from physdual.eval.metrics import bootstrap_grouped_comparisons, compute_all_metrics  # noqa: E402
from physdual.utils.paths import get_project_root  # noqa: E402


DAVIS_KFOLD_SPLITS = (
    "cold_target",
    "cold_drug",
    "cold_target_seqid30",
    "cold_target_seqid40",
)

KIBA_KFOLD_SPLITS = (
    "cold_target_seqid30",
    "cold_target_seqid40",
)


def _mean_std_median(vals: list[float]) -> dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)),
    }


def _evaluate_adr15_outcome(
    paired_entry: dict[str, Any] | None,
    bs_entry: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Classifica critério operacional pré-registrado no ADR #15 para um comparador."""
    if not paired_entry or not bs_entry:
        return None
    wins_mse = int(paired_entry.get("wins_mse", 0))
    wins_ci = int(paired_entry.get("wins_ci", 0))
    ci_mse = bs_entry.get("mse_delta", {}).get("ci_95", [0.0, 0.0])
    ci_ci = bs_entry.get("ci_delta", {}).get("ci_95", [0.0, 0.0])
    mse_excl_zero = bool(float(ci_mse[1]) < 0.0)
    ci_excl_zero = bool(float(ci_ci[0]) > 0.0)
    if wins_mse >= 4 and wins_ci >= 4 and mse_excl_zero and ci_excl_zero:
        verdict = "primary_win"
    elif wins_mse <= 2 or wins_ci <= 2 or not mse_excl_zero or not ci_excl_zero:
        verdict = "formal_negative"
    else:
        verdict = "inconclusive"
    return {
        "verdict": verdict,
        "wins_mse": wins_mse,
        "wins_ci": wins_ci,
        "mse_ic95_excludes_zero_favorable": mse_excl_zero,
        "ci_ic95_excludes_zero_favorable": ci_excl_zero,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Auditoria OOF e bootstrap agrupado pareado (Davis ou KIBA Fase 3).")
    parser.add_argument("--dataset", type=str, default="davis", choices=["davis", "kiba"])
    args = parser.parse_args(argv)
    dataset = args.dataset.lower().strip()

    root = get_project_root()
    results_dir = root / "outputs" / "results"
    oof_dir = results_dir / "oof"
    cmap_path = root / "outputs" / "logs" / f"{dataset}_seqid_cluster_map.json"
    cmap = json.loads(cmap_path.read_text(encoding="utf-8")) if cmap_path.exists() else {}

    esm_cpu = json.loads((results_dir / "phase3_esm_cpu_baselines.json").read_text(encoding="utf-8"))
    if dataset == "davis":
        splits = DAVIS_KFOLD_SPLITS
        esm_key_map = {
            "cold_target": "davis_v2_cold_target_kfold",
            "cold_drug": "davis_v2_cold_drug_kfold",
            "cold_target_seqid30": "davis_seqid30_kfold",
            "cold_target_seqid40": "davis_seqid40_kfold",
        }
    else:
        splits = KIBA_KFOLD_SPLITS
        esm_key_map = {
            "cold_target_seqid30": "kiba_seqid30_kfold",
            "cold_target_seqid40": "kiba_seqid40_kfold",
        }

    report: dict[str, Any] = {
        "dataset": dataset,
        "protocol": "v2_kfold",
        "exploratory": dataset == "davis",
        "pre_registered_adr15": dataset == "kiba",
        "n_bootstraps": 1000,
        "seed": 42,
        "reproduction_checks": {},
        "regimes": {},
    }
    if dataset == "davis":
        report["cold_drug_descriptive_all_models"] = {}

    for sp in splits:
        cpu_df = pd.read_csv(oof_dir / f"cpu_baselines_{dataset}_{sp}_oof.csv")
        p3_df = pd.read_csv(oof_dir / f"physdual_gin_esm_mlp_{dataset}_{sp}_oof.csv")
        model_dfs: list[tuple[str, pd.DataFrame]] = [("gin_esm_mlp", p3_df)]
        gin_df: pd.DataFrame | None = None
        cnn_df: pd.DataFrame | None = None
        if dataset == "davis":
            gin_df = pd.read_csv(oof_dir / f"graphdta_gin_{dataset}_{sp}_oof.csv")
            cnn_df = pd.read_csv(oof_dir / f"graphdta_cnn3_{dataset}_{sp}_oof.csv")
            model_dfs.extend([("graphdta_gin", gin_df), ("graphdta_cnn3", cnn_df)])

        for name, mdf in model_dfs:
            if len(mdf) != len(cpu_df):
                raise ValueError(f"{sp}/{name}: tamanho OOF divergente ({len(mdf)} vs {len(cpu_df)})")
            if not np.array_equal(mdf["fold"].to_numpy(), cpu_df["fold"].to_numpy()):
                raise ValueError(f"{sp}/{name}: folds desalinhados")
            if not np.allclose(mdf["y_true"].to_numpy(), cpu_df["y_true"].to_numpy(), atol=1e-6):
                raise ValueError(f"{sp}/{name}: y_true desalinhado")
            if mdf["drug_id"].astype(str).tolist() != cpu_df["drug_id"].astype(str).tolist():
                raise ValueError(f"{sp}/{name}: drug_id desalinhado")
            if mdf["target_id"].astype(str).tolist() != cpu_df["target_id"].astype(str).tolist():
                raise ValueError(f"{sp}/{name}: target_id desalinhado")
            if not mdf["sequence"].isna().all() and (
                mdf["sequence"].astype(str).tolist() != cpu_df["sequence"].astype(str).tolist()
            ):
                raise ValueError(f"{sp}/{name}: sequence desalinhado")
            if mdf["smiles"].astype(str).tolist() != cpu_df["smiles"].astype(str).tolist():
                raise ValueError(f"{sp}/{name}: smiles desalinhado")

        # Verifica que drug_mean e target_mean recomputados reproduzem o salvo no OOF do Fase 3 e nos JSONs
        max_diff_dm = float(
            np.max(np.abs(p3_df["y_pred_drug_mean"].to_numpy() - cpu_df["y_pred_drug_mean_predictor"].to_numpy()))
        )
        max_diff_tm = float(
            np.max(np.abs(p3_df["y_pred_target_mean"].to_numpy() - cpu_df["y_pred_target_mean_predictor"].to_numpy()))
        )

        # Verifica métricas OOF salvas contra recomputadas
        p3_agg = json.loads((results_dir / f"phase3_physdual_gin_esm_mlp_{dataset}_{sp}_kfold.json").read_text(encoding="utf-8"))
        esm_reg = esm_cpu["regimes"][esm_key_map[sp]]

        yt_all = cpu_df["y_true"].to_numpy(dtype=np.float64)
        repro_diffs: dict[str, dict[str, float]] = {
            "oof_y_pred_drug_mean_max_abs_diff": {"val": max_diff_dm},
            "oof_y_pred_target_mean_max_abs_diff": {"val": max_diff_tm},
        }
        repro_targets: list[tuple[str, np.ndarray, dict[str, Any]]] = [
            ("gin_esm_mlp", p3_df["y_pred"].to_numpy(dtype=np.float64), p3_agg["global_oof_metrics"]),
            ("ridge_esm", cpu_df["y_pred_ridge_esm"].to_numpy(dtype=np.float64), esm_reg["global_oof_metrics"]["ridge_esm"]),
            ("knn_esm_target", cpu_df["y_pred_knn_esm_target"].to_numpy(dtype=np.float64), esm_reg["global_oof_metrics"]["knn_esm_target"]),
        ]
        if dataset == "davis" and gin_df is not None and cnn_df is not None:
            gin_agg = json.loads((results_dir / f"baseline_graphdta_gin_{dataset}_{sp}_kfold.json").read_text(encoding="utf-8"))
            cnn_agg = json.loads((results_dir / f"baseline_graphdta_cnn3_{dataset}_{sp}_kfold.json").read_text(encoding="utf-8"))
            repro_targets.extend([
                ("graphdta_gin", gin_df["y_pred"].to_numpy(dtype=np.float64), gin_agg["global_oof_metrics"]),
                ("graphdta_cnn3", cnn_df["y_pred"].to_numpy(dtype=np.float64), cnn_agg["global_oof_metrics"]),
            ])

        for label, pred_arr, saved_oof in repro_targets:
            rec = compute_all_metrics(yt_all, pred_arr)
            repro_diffs[label] = {
                "ci_abs_diff": abs(rec["ci"] - float(saved_oof["ci"])),
                "mse_abs_diff": abs(rec["mse"] - float(saved_oof["mse"])),
            }

        # Verifica reproduçao de marginais e kNN contra per_fold_results do JSON da Fase 3
        max_fold_diffs = {k: 0.0 for k in ("drug_mean_predictor", "target_mean_predictor", "knn_tanimoto_drug", "knn_kmer_target")}
        for f_id in range(5):
            mask = cpu_df["fold"].to_numpy() == f_id
            yf = yt_all[mask]
            pf_json = p3_agg["per_fold_results"][str(f_id)]
            for mk in ("drug_mean_predictor", "target_mean_predictor"):
                rec_m = compute_all_metrics(yf, cpu_df.loc[mask, f"y_pred_{mk}"].to_numpy(dtype=np.float64))
                saved_m = pf_json["marginal_baselines"][mk]
                max_fold_diffs[mk] = max(max_fold_diffs[mk], abs(rec_m["mse"] - saved_m["mse"]), abs(rec_m["ci"] - saved_m["ci"]))
            for kk in ("knn_tanimoto_drug", "knn_kmer_target"):
                rec_k = compute_all_metrics(yf, cpu_df.loc[mask, f"y_pred_{kk}"].to_numpy(dtype=np.float64))
                saved_k = pf_json["knn_baselines"][kk]
                max_fold_diffs[kk] = max(max_fold_diffs[kk], abs(rec_k["mse"] - saved_k["mse"]), abs(rec_k["ci"] - saved_k["ci"]))

        repro_diffs["per_fold_max_abs_diff"] = max_fold_diffs
        report["reproduction_checks"][sp] = repro_diffs

        # Constrói coluna de agrupamento exata para bootstrap agrupado
        if sp == "cold_drug":
            group_col = "bootstrap_group_smiles"
            group_vals = cpu_df["smiles"].astype(str)
            group_desc = "smiles (fármaco único)"
        elif sp == "cold_target":
            group_col = "bootstrap_group_sequence"
            group_vals = cpu_df["sequence"].astype(str)
            group_desc = "sequence (sequência proteica idêntica)"
        else:
            group_col = "bootstrap_group_seqid_cluster"
            s2c = cmap["thresholds"][sp]["seq_to_cluster"]
            group_vals = cpu_df["sequence"].astype(str).map(s2c).astype(int)
            group_desc = f"cluster BLOSUM62 single-linkage ({sp})"

        oof_bs_dict: dict[str, Any] = {
            "y_true": yt_all,
            group_col: group_vals,
            "y_pred_gin_esm_mlp": p3_df["y_pred"].to_numpy(dtype=np.float64),
            "y_pred_ridge_esm": cpu_df["y_pred_ridge_esm"].to_numpy(dtype=np.float64),
            "y_pred_ridge_esm_target_only": cpu_df["y_pred_ridge_esm_target_only"].to_numpy(dtype=np.float64),
            "y_pred_knn_esm_target": cpu_df["y_pred_knn_esm_target"].to_numpy(dtype=np.float64),
            "y_pred_knn_kmer_target": cpu_df["y_pred_knn_kmer_target"].to_numpy(dtype=np.float64),
            "y_pred_knn_tanimoto_drug": cpu_df["y_pred_knn_tanimoto_drug"].to_numpy(dtype=np.float64),
            "y_pred_drug_mean_predictor": cpu_df["y_pred_drug_mean_predictor"].to_numpy(dtype=np.float64),
            "y_pred_target_mean_predictor": cpu_df["y_pred_target_mean_predictor"].to_numpy(dtype=np.float64),
        }
        if dataset == "davis" and cnn_df is not None and gin_df is not None:
            oof_bs_dict["y_pred_graphdta_cnn3"] = cnn_df["y_pred"].to_numpy(dtype=np.float64)
            oof_bs_dict["y_pred_graphdta_gin"] = gin_df["y_pred"].to_numpy(dtype=np.float64)
        oof_bs = pd.DataFrame(oof_bs_dict)

        baseline_cols = [c for c in oof_bs.columns if c.startswith("y_pred_") and c != "y_pred_gin_esm_mlp"]
        bs_res = bootstrap_grouped_comparisons(
            oof_df=oof_bs,
            group_col=group_col,
            model_col="y_pred_gin_esm_mlp",
            baseline_cols=baseline_cols,
            target_col="y_true",
            n_bootstraps=1000,
            seed=42,
        )

        # Identifica melhor baseline do regime por MSE OOF e por média por dobra
        per_model_oof = {
            c.replace("y_pred_", ""): compute_all_metrics(yt_all, oof_bs[c].to_numpy(dtype=np.float64))
            for c in baseline_cols
        }
        best_by_oof_mse = min(per_model_oof.items(), key=lambda kv: kv[1]["mse"])[0]
        best_by_oof_ci = max(per_model_oof.items(), key=lambda kv: kv[1]["ci"])[0]
        non_esm_cols = [
            c.replace("y_pred_", "")
            for c in baseline_cols
            if "esm" not in c
        ]
        best_non_esm_mse = min(non_esm_cols, key=lambda name: per_model_oof[name]["mse"]) if non_esm_cols else None

        # Média por dobra vs OOF para gin_esm_mlp e todos os baselines
        per_fold_table: dict[str, list[dict[str, float]]] = {}
        all_models_for_fold = ["gin_esm_mlp"] + [c.replace("y_pred_", "") for c in baseline_cols]
        for mname in all_models_for_fold:
            col = f"y_pred_{mname}"
            rows_f = []
            for f_id in range(5):
                mask = cpu_df["fold"].to_numpy() == f_id
                mf = compute_all_metrics(yt_all[mask], oof_bs.loc[mask, col].to_numpy(dtype=np.float64))
                rows_f.append({"fold": f_id, "ci": mf["ci"], "mse": mf["mse"]})
            per_fold_table[mname] = rows_f

        per_fold_summary = {
            mname: {
                "ci_per_fold": _mean_std_median([r["ci"] for r in rows_f]),
                "mse_per_fold": _mean_std_median([r["mse"] for r in rows_f]),
                "oof": compute_all_metrics(yt_all, oof_bs[f"y_pred_{mname}"].to_numpy(dtype=np.float64)),
            }
            for mname, rows_f in per_fold_table.items()
        }

        paired_map = p3_agg.get("paired_vs_baselines", {})
        bs_clean = {
            k.replace("y_pred_", ""): {**v, "exploratory": dataset == "davis"}
            for k, v in bs_res.items()
        }
        adr15_eval: dict[str, Any] = {
            "vs_ridge_esm": _evaluate_adr15_outcome(paired_map.get("ridge_esm"), bs_clean.get("ridge_esm")),
        }
        if best_non_esm_mse is not None:
            adr15_eval["best_non_esm_baseline"] = best_non_esm_mse
            adr15_eval["vs_best_non_esm"] = _evaluate_adr15_outcome(
                paired_map.get(best_non_esm_mse), bs_clean.get(best_non_esm_mse)
            )

        report["regimes"][sp] = {
            "exploratory": dataset == "davis",
            "group_col": group_col,
            "group_description": group_desc,
            "n_clusters": int(group_vals.nunique()),
            "best_baseline_oof_mse": best_by_oof_mse,
            "best_baseline_oof_ci": best_by_oof_ci,
            "adr15_operational_check": adr15_eval,
            "per_model_summary": per_fold_summary,
            "per_fold_by_model": per_fold_table,
            "bootstrap_vs_baselines": bs_clean,
        }

        if dataset == "davis" and sp == "cold_drug":
            desc_models = {}
            for mname, rows_f in per_fold_table.items():
                rows_no2 = [r for r in rows_f if r["fold"] != 2]
                desc_models[mname] = {
                    "per_fold": rows_f,
                    "all_5_folds": {
                        "ci": _mean_std_median([r["ci"] for r in rows_f]),
                        "mse": _mean_std_median([r["mse"] for r in rows_f]),
                    },
                    "without_fold2_4_folds_descriptive": {
                        "ci": _mean_std_median([r["ci"] for r in rows_no2]),
                        "mse": _mean_std_median([r["mse"] for r in rows_no2]),
                    },
                }
            # Adiciona mean_predictor constante no cold_drug
            mp_rows = []
            for f_id in range(5):
                mask = cpu_df["fold"].to_numpy() == f_id
                mf = compute_all_metrics(yt_all[mask], cpu_df.loc[mask, "y_pred_mean_predictor"].to_numpy(dtype=np.float64))
                mp_rows.append({"fold": f_id, "ci": mf["ci"], "mse": mf["mse"]})
            mp_no2 = [r for r in mp_rows if r["fold"] != 2]
            desc_models["mean_predictor"] = {
                "per_fold": mp_rows,
                "all_5_folds": {"ci": _mean_std_median([r["ci"] for r in mp_rows]), "mse": _mean_std_median([r["mse"] for r in mp_rows])},
                "without_fold2_4_folds_descriptive": {"ci": _mean_std_median([r["ci"] for r in mp_no2]), "mse": _mean_std_median([r["mse"] for r in mp_no2])},
            }
            report["cold_drug_descriptive_all_models"] = {
                "descriptive_post_hoc": True,
                "folds_included": [0, 1, 3, 4],
                "excluded_fold": 2,
                "models": desc_models,
            }

    out_path = results_dir / f"phase3_{dataset}_paired_bootstrap.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Bootstrap pareado {dataset.upper()} salvo em: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
