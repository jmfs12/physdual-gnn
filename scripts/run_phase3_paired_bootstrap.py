#!/usr/bin/env python3
"""Audita OOFs locais, verifica reprodução dos baselines e executa bootstrap agrupado pareado (Davis Fase 3)."""

from __future__ import annotations

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


def _mean_std_median(vals: list[float]) -> dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)),
    }


def main() -> int:
    root = get_project_root()
    results_dir = root / "outputs" / "results"
    oof_dir = results_dir / "oof"
    cmap_path = root / "outputs" / "logs" / "davis_seqid_cluster_map.json"
    cmap = json.loads(cmap_path.read_text(encoding="utf-8")) if cmap_path.exists() else {}

    esm_cpu = json.loads((results_dir / "phase3_esm_cpu_baselines.json").read_text(encoding="utf-8"))
    esm_key_map = {
        "cold_target": "davis_v2_cold_target_kfold",
        "cold_drug": "davis_v2_cold_drug_kfold",
        "cold_target_seqid30": "davis_seqid30_kfold",
        "cold_target_seqid40": "davis_seqid40_kfold",
    }

    report: dict[str, Any] = {
        "dataset": "davis",
        "protocol": "v2_kfold",
        "exploratory": True,
        "n_bootstraps": 1000,
        "seed": 42,
        "reproduction_checks": {},
        "regimes": {},
        "cold_drug_descriptive_all_models": {},
    }

    for sp in DAVIS_KFOLD_SPLITS:
        cpu_df = pd.read_csv(oof_dir / f"cpu_baselines_davis_{sp}_oof.csv")
        p3_df = pd.read_csv(oof_dir / f"physdual_gin_esm_mlp_davis_{sp}_oof.csv")
        gin_df = pd.read_csv(oof_dir / f"graphdta_gin_davis_{sp}_oof.csv")
        cnn_df = pd.read_csv(oof_dir / f"graphdta_cnn3_davis_{sp}_oof.csv")

        for name, mdf in (("gin_esm_mlp", p3_df), ("graphdta_gin", gin_df), ("graphdta_cnn3", cnn_df)):
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
        p3_agg = json.loads((results_dir / f"phase3_physdual_gin_esm_mlp_davis_{sp}_kfold.json").read_text(encoding="utf-8"))
        gin_agg = json.loads((results_dir / f"baseline_graphdta_gin_davis_{sp}_kfold.json").read_text(encoding="utf-8"))
        cnn_agg = json.loads((results_dir / f"baseline_graphdta_cnn3_davis_{sp}_kfold.json").read_text(encoding="utf-8"))
        esm_reg = esm_cpu["regimes"][esm_key_map[sp]]

        yt_all = cpu_df["y_true"].to_numpy(dtype=np.float64)
        repro_diffs: dict[str, dict[str, float]] = {
            "oof_y_pred_drug_mean_max_abs_diff": {"val": max_diff_dm},
            "oof_y_pred_target_mean_max_abs_diff": {"val": max_diff_tm},
        }
        for label, pred_arr, saved_oof in (
            ("gin_esm_mlp", p3_df["y_pred"].to_numpy(dtype=np.float64), p3_agg["global_oof_metrics"]),
            ("graphdta_gin", gin_df["y_pred"].to_numpy(dtype=np.float64), gin_agg["global_oof_metrics"]),
            ("graphdta_cnn3", cnn_df["y_pred"].to_numpy(dtype=np.float64), cnn_agg["global_oof_metrics"]),
            ("ridge_esm", cpu_df["y_pred_ridge_esm"].to_numpy(dtype=np.float64), esm_reg["global_oof_metrics"]["ridge_esm"]),
            ("knn_esm_target", cpu_df["y_pred_knn_esm_target"].to_numpy(dtype=np.float64), esm_reg["global_oof_metrics"]["knn_esm_target"]),
        ):
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

        oof_bs = pd.DataFrame({
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
            "y_pred_graphdta_cnn3": cnn_df["y_pred"].to_numpy(dtype=np.float64),
            "y_pred_graphdta_gin": gin_df["y_pred"].to_numpy(dtype=np.float64),
        })

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

        report["regimes"][sp] = {
            "exploratory": True,
            "group_col": group_col,
            "group_description": group_desc,
            "n_clusters": int(group_vals.nunique()),
            "best_baseline_oof_mse": best_by_oof_mse,
            "best_baseline_oof_ci": best_by_oof_ci,
            "per_model_summary": per_fold_summary,
            "per_fold_by_model": per_fold_table,
            "bootstrap_vs_baselines": {
                k.replace("y_pred_", ""): {**v, "exploratory": True}
                for k, v in bs_res.items()
            },
        }

        if sp == "cold_drug":
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

    out_path = results_dir / "phase3_davis_paired_bootstrap.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Bootstrap pareado Davis salvo em: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
