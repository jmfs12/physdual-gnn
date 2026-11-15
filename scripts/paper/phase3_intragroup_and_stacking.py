"""Análises pós-auditoria em CPU (somente leitura sobre outputs/cache/splits):

1. CI e Spearman INTRAGRUPO:
   - Nos regimes `cold_target*` (Davis e KIBA `v2_kfold` + `kiba/cold_target` `v1_seeds`):
     ranqueamento intragrupo por fármaco (`smiles`), avaliando se os modelos realmente
     ordenam alvos inéditos para um mesmo fármaco ou apenas reproduzem o viés médio do fármaco.
   - Nos regimes `cold_drug` (Davis `v2_kfold` + KIBA `v1_seeds`):
     ranqueamento intragrupo por alvo (`sequence`), avaliando se os modelos ordenam
     fármacos inéditos contra um mesmo alvo.
2. Baseline de empilhamento residual em CPU (zero vazamento de teste):
   - Ajusta `y_res = y_true - drug_mean_train` em `train` sobre:
       * `stack_drug_mean_only` (`y_res_hat = 0`)
       * `stack_res_esm` (`ESM mean-pool 1280d`)
       * `stack_res_esm_ecfp` (`ESM 1280d + ECFP4 1024d`)
       * `stack_res_esm_ecfp_inter` (`ESM 1280d + ECFP4 1024d + interações bilinear SVD 16x16`)
   - Seleciona `alpha` exclusivamente em `val`, soma `drug_mean_train` de volta e mede
     `CI`, `MSE`, `RM2` globais/OOF e `CI`/`Spearman` intragrupo no conjunto `test`.

Saída: `docs/paper/phase3_intragroup_and_stacking.json`
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import cho_factor, cho_solve, eigh as sp_eigh
from sklearn.decomposition import TruncatedSVD

PAPER_SCRIPTS_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = PAPER_SCRIPTS_DIR.parent
REPO_ROOT = PAPER_SCRIPTS_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
for p in (str(PAPER_SCRIPTS_DIR), str(SCRIPTS_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common import load_json_strict, resolve_roots, sample_mean_std  # noqa: E402
from physdual.data.splits import load_splits  # noqa: E402
from physdual.eval.esm_baselines import (  # noqa: E402
    DEFAULT_RIDGE_ALPHAS,
    compute_morgan_ecfp4_map,
    load_mean_pooled_esm_embeddings,
)
from physdual.eval.metrics import (  # noqa: E402
    compute_ci,
    compute_mse,
    compute_pearson,
    compute_rm2,
    compute_rmse,
    compute_spearman,
)
from run_phase3_esm_cpu_baselines import (  # noqa: E402
    _build_3mer_norm_matrix,
    _build_tanimoto_sim_matrix,
    _compute_classical_unit,
)

RESIDUAL_ALPHAS: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0, 10000.0, 50000.0)


def _fast_ci(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Índice de concordância exato em O(N log N) via Fenwick Tree."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.size < 2:
        return 0.0
    order = np.argsort(yt, kind="mergesort")
    yt_s = yt[order]
    yp_s = yp[order]
    _, p_inv = np.unique(yp_s, return_inverse=True)
    m = int(p_inv.max()) + 1
    tree = np.zeros(m + 1, dtype=np.int64)
    num = 0.0
    den = 0.0
    n = len(yt_s)
    i = 0
    while i < n:
        j = i + 1
        while j < n and yt_s[j] == yt_s[i]:
            j += 1
        for k in range(i, j):
            idx = int(p_inv[k])
            # prefix sum at idx and idx+1
            s_less = 0
            p1 = idx
            while p1 > 0:
                s_less += int(tree[p1])
                p1 -= p1 & -p1
            s_le = 0
            p2 = idx + 1
            while p2 > 0:
                s_le += int(tree[p2])
                p2 -= p2 & -p2
            num += s_less + 0.5 * (s_le - s_less)
            den += i
        for k in range(i, j):
            u = int(p_inv[k]) + 1
            while u <= m:
                tree[u] += 1
                u += u & -u
        i = j
    return float(num / den) if den > 0.0 else 0.0


def _fast_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "ci": float(_fast_ci(y_true, y_pred)),
        "mse": float(compute_mse(y_true, y_pred)),
        "rmse": float(compute_rmse(y_true, y_pred)),
        "pearson": float(compute_pearson(y_true, y_pred)),
        "spearman": float(compute_spearman(y_true, y_pred)),
        "rm2": float(compute_rm2(y_true, y_pred)),
    }


def _prepare_group_indices(
    groups: np.ndarray,
    y_true: np.ndarray,
    min_pairs: int = 2,
) -> tuple[int, list[np.ndarray]]:
    """Pré-agrupa índices por grupo elegível (>= `min_pairs` e variância em `y_true`)."""
    raw_groups = pd.Series(np.arange(len(groups))).groupby(groups, sort=False).indices
    total_groups = len(raw_groups)
    eligible: list[np.ndarray] = []
    for idx in raw_groups.values():
        if len(idx) < min_pairs:
            continue
        yt_g = y_true[idx]
        if np.ptp(yt_g) <= 0.0 and np.unique(yt_g).size < 2:
            continue
        if np.unique(yt_g).size < 2:
            continue
        eligible.append(np.asarray(idx, dtype=np.intp))
    return total_groups, eligible


def _intragroup_from_prepared(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    total_groups: int,
    eligible_indices: list[np.ndarray],
) -> dict[str, Any]:
    """Calcula CI e Spearman intragrupo usando índices pré-agrupados (alta velocidade)."""
    if not eligible_indices:
        return {
            "n_groups_total": int(total_groups),
            "n_groups_evaluable": 0,
            "n_groups_const_pred": 0,
            "n_pairs_covered": 0,
            "ci_mean": None,
            "ci_std": None,
            "ci_median": None,
            "spearman_mean": None,
            "spearman_std": None,
            "spearman_median": None,
        }

    yt_all = np.asarray(y_true, dtype=np.float64)
    yp_all = np.asarray(y_pred, dtype=np.float64)
    ci_vals: list[float] = []
    sp_vals: list[float] = []
    const_pred_groups = 0
    pairs_covered = 0

    for idx in eligible_indices:
        yt_g = yt_all[idx]
        yp_g = yp_all[idx]
        pairs_covered += len(idx)
        if len(idx) <= 256:
            dy = yt_g[:, None] - yt_g[None, :]
            mask = dy > 0.0
            den_g = int(np.count_nonzero(mask))
            if den_g == 0:
                ci_vals.append(0.0)
            else:
                dp = yp_g[:, None] - yp_g[None, :]
                dp_m = dp[mask]
                concord = float(np.count_nonzero(dp_m > 0.0)) + 0.5 * float(np.count_nonzero(dp_m == 0.0))
                ci_vals.append(concord / float(den_g))
        else:
            ci_vals.append(float(_fast_ci(yt_g, yp_g)))
        if np.ptp(yp_g) <= 1e-12:
            const_pred_groups += 1
            sp_vals.append(0.0)
        else:
            rx = stats.rankdata(yt_g)
            ry = stats.rankdata(yp_g)
            rx_c = rx - rx.mean()
            ry_c = ry - ry.mean()
            denom = float(np.sqrt(np.dot(rx_c, rx_c) * np.dot(ry_c, ry_c)))
            rho = float(np.dot(rx_c, ry_c) / denom) if denom > 0.0 else 0.0
            sp_vals.append(0.0 if not np.isfinite(rho) else rho)

    ci_arr = np.asarray(ci_vals, dtype=np.float64)
    sp_arr = np.asarray(sp_vals, dtype=np.float64)
    return {
        "n_groups_total": int(total_groups),
        "n_groups_evaluable": int(len(ci_vals)),
        "n_groups_const_pred": int(const_pred_groups),
        "n_pairs_covered": int(pairs_covered),
        "ci_mean": float(np.mean(ci_arr)),
        "ci_std": float(np.std(ci_arr, ddof=1)) if len(ci_arr) > 1 else 0.0,
        "ci_median": float(np.median(ci_arr)),
        "spearman_mean": float(np.mean(sp_arr)),
        "spearman_std": float(np.std(sp_arr, ddof=1)) if len(sp_arr) > 1 else 0.0,
        "spearman_median": float(np.median(sp_arr)),
    }


def _intragroup_ci_spearman(
    groups: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    min_pairs: int = 2,
) -> dict[str, Any]:
    """Calcula CI e Spearman intragrupo para grupos com >= `min_pairs` e variância em `y_true`."""
    total_groups, eligible = _prepare_group_indices(groups, np.asarray(y_true, dtype=np.float64), min_pairs=min_pairs)
    return _intragroup_from_prepared(y_true, y_pred, total_groups, eligible)


def compute_intragroup_v2_kfold(data_root: Path) -> dict[str, Any]:
    oof_dir = data_root / "outputs" / "results" / "oof"
    specs = [
        ("davis", "cold_target", "smiles"),
        ("davis", "cold_target_seqid40", "smiles"),
        ("davis", "cold_target_seqid30", "smiles"),
        ("davis", "cold_drug", "sequence"),
        ("kiba", "cold_target_seqid40", "smiles"),
        ("kiba", "cold_target_seqid30", "smiles"),
    ]
    out: dict[str, Any] = {}
    for ds, sp, gcol in specs:
        reg_key = f"{ds}/{sp}"
        cpu_df = pd.read_csv(oof_dir / f"cpu_baselines_{ds}_{sp}_oof.csv")
        p3_df = pd.read_csv(oof_dir / f"physdual_gin_esm_mlp_{ds}_{sp}_oof.csv", usecols=["y_pred"])

        preds: dict[str, np.ndarray] = {
            "gin_esm_mlp": p3_df["y_pred"].to_numpy(dtype=np.float64),
            "ridge_esm": cpu_df["y_pred_ridge_esm"].to_numpy(dtype=np.float64),
            "knn_esm_target": cpu_df["y_pred_knn_esm_target"].to_numpy(dtype=np.float64),
            "knn_kmer_target": cpu_df["y_pred_knn_kmer_target"].to_numpy(dtype=np.float64),
            "knn_tanimoto_drug": cpu_df["y_pred_knn_tanimoto_drug"].to_numpy(dtype=np.float64),
            "drug_mean_predictor": cpu_df["y_pred_drug_mean_predictor"].to_numpy(dtype=np.float64),
            "target_mean_predictor": cpu_df["y_pred_target_mean_predictor"].to_numpy(dtype=np.float64),
        }
        for gm in ("graphdta_gin", "graphdta_cnn3"):
            gp = oof_dir / f"{gm}_{ds}_{sp}_oof.csv"
            if gp.exists():
                gdf = pd.read_csv(gp, usecols=["y_pred"])
                preds[gm] = gdf["y_pred"].to_numpy(dtype=np.float64)

        ecfp_p = oof_dir / f"physdual_ecfp_esm_mlp_{ds}_{sp}_oof.csv"
        if ecfp_p.exists():
            edf = pd.read_csv(ecfp_p, usecols=["y_pred"])
            preds["ecfp_esm_mlp"] = edf["y_pred"].to_numpy(dtype=np.float64)

        yt = cpu_df["y_true"].to_numpy(dtype=np.float64)
        grp = cpu_df[gcol].astype(str).to_numpy()
        folds_arr = cpu_df["fold"].to_numpy(dtype=np.int64)
        tot2, elig2 = _prepare_group_indices(grp, yt, min_pairs=2)
        tot5, elig5 = _prepare_group_indices(grp, yt, min_pairs=5)
        fold_preps = []
        for f in range(5):
            f_mask = folds_arr == f
            f_idx = np.flatnonzero(f_mask)
            f_tot2, f_elig2 = _prepare_group_indices(grp[f_idx], yt[f_idx], min_pairs=2)
            fold_preps.append((f_idx, f_tot2, f_elig2))
        models_out: dict[str, Any] = {}
        for mname, yp in preds.items():
            f_res = [
                _intragroup_from_prepared(yt[f_idx], yp[f_idx], f_tot2, f_elig2)
                for f_idx, f_tot2, f_elig2 in fold_preps
            ]
            models_out[mname] = {
                "min_pairs_2": _intragroup_from_prepared(yt, yp, tot2, elig2),
                "min_pairs_5": _intragroup_from_prepared(yt, yp, tot5, elig5),
                "per_fold_min2": {
                    "ci_mean": sample_mean_std([x["ci_mean"] for x in f_res if x["ci_mean"] is not None]),
                    "spearman_mean": sample_mean_std([x["spearman_mean"] for x in f_res if x["spearman_mean"] is not None]),
                    "n_groups_evaluable_per_fold": [x["n_groups_evaluable"] for x in f_res],
                    "n_groups_const_pred_per_fold": [x["n_groups_const_pred"] for x in f_res],
                },
                "oof_global": _fast_all_metrics(yt, yp),
            }
        out[reg_key] = {
            "protocol": "v2_kfold",
            "group_axis": gcol,
            "n_oof_pairs": int(len(yt)),
            "models": models_out,
        }
    return out


def compute_v1_seeds_intragroup(
    data_root: Path,
    seq_to_vec: dict[str, np.ndarray],
) -> dict[str, Any]:
    splits_dir = data_root / "data" / "processed" / "splits"
    runs_dir = data_root / "outputs" / "results" / "runs"
    kiba_df = pd.read_csv(data_root / "data" / "processed" / "kiba_clean.csv")

    km_k, s_idx_k = _build_3mer_norm_matrix(kiba_df["sequence"].astype(str).tolist())
    km_sim_k = km_k @ km_k.T
    tan_sim_k, d_idx_k = _build_tanimoto_sim_matrix(kiba_df["smiles"].astype(str).tolist())

    v1_intra: dict[str, Any] = {}
    for sp, gcol in (("cold_target", "smiles"), ("cold_drug", "sequence")):
        reg_key = f"kiba/{sp}"
        per_seed_m2: dict[str, list[dict[str, Any]]] = {}
        per_seed_m5: dict[str, list[dict[str, Any]]] = {}
        for seed in range(5):
            split_idx = load_splits(splits_dir / f"kiba_{sp}_seed{seed}.json")
            tr_df = kiba_df.iloc[split_idx["train"]]
            va_df = kiba_df.iloc[split_idx["val"]]
            te_df = kiba_df.iloc[split_idx["test"]]
            yt = te_df["affinity"].to_numpy(dtype=np.float64)
            grp = te_df[gcol].astype(str).to_numpy()

            run_d = load_json_strict(runs_dir / f"physdual_gin_esm_mlp_kiba_{sp}_seed{seed}.json")
            yp_gin = np.array([r["y_pred"] for r in run_d["oof_predictions"]], dtype=np.float64)

            # Fast Cholesky Ridge on ESM-2 (equivalent to DEFAULT_RIDGE_ALPHAS, fit on train, alpha on val)
            P_tr_v1 = np.vstack([seq_to_vec[s] for s in tr_df["sequence"].astype(str)]).astype(np.float64)
            P_va_v1 = np.vstack([seq_to_vec[s] for s in va_df["sequence"].astype(str)]).astype(np.float64)
            P_te_v1 = np.vstack([seq_to_vec[s] for s in te_df["sequence"].astype(str)]).astype(np.float64)
            y_tr_v1 = tr_df["affinity"].to_numpy(dtype=np.float64)
            y_va_v1 = va_df["affinity"].to_numpy(dtype=np.float64)
            mu_v1 = np.mean(P_tr_v1, axis=0)
            sd_v1 = np.std(P_tr_v1, axis=0)
            sd_v1[sd_v1 < 1e-12] = 1.0
            P_tr_v1 -= mu_v1
            P_tr_v1 /= sd_v1
            P_va_v1 -= mu_v1
            P_va_v1 /= sd_v1
            P_te_v1 -= mu_v1
            P_te_v1 /= sd_v1
            y_tr_mean_v1 = float(np.mean(y_tr_v1))
            y_tr_c = y_tr_v1 - y_tr_mean_v1
            G_v1 = P_tr_v1.T @ P_tr_v1
            gy_v1 = P_tr_v1.T @ y_tr_c
            diag_v1 = np.arange(G_v1.shape[0])
            best_va_mse_v1 = float("inf")
            best_w_v1 = None
            for alpha in DEFAULT_RIDGE_ALPHAS:
                A_v1 = G_v1.copy()
                A_v1[diag_v1, diag_v1] += float(alpha)
                w_v1 = cho_solve(cho_factor(A_v1, check_finite=False, overwrite_a=True), gy_v1, check_finite=False)
                vp_v1 = y_tr_mean_v1 + (P_va_v1 @ w_v1)
                vmse = float(np.mean((y_va_v1 - vp_v1) ** 2))
                if vmse < best_va_mse_v1:
                    best_va_mse_v1 = vmse
                    best_w_v1 = w_v1
            assert best_w_v1 is not None
            yp_ridge_esm = y_tr_mean_v1 + (P_te_v1 @ best_w_v1)

            # Fast cosine 5-NN target ESM baseline (identical to esm_baselines.predict_knn_esm_target)
            tr_seqs_u = tr_df["sequence"].astype(str).unique().tolist()
            te_seqs_u = te_df["sequence"].astype(str).unique().tolist()
            U_tr = np.vstack([seq_to_vec[s] for s in tr_seqs_u]).astype(np.float64)
            U_te = np.vstack([seq_to_vec[s] for s in te_seqs_u]).astype(np.float64)
            U_tr /= np.maximum(np.linalg.norm(U_tr, axis=1, keepdims=True), 1e-12)
            U_te /= np.maximum(np.linalg.norm(U_te, axis=1, keepdims=True), 1e-12)
            sim_esm = U_te @ U_tr.T
            tr_seq_to_idx = {s: idx for idx, s in enumerate(tr_seqs_u)}
            te_seq_to_idx = {s: idx for idx, s in enumerate(te_seqs_u)}
            tr_drug_seq_aff: dict[str, tuple[list[int], list[float]]] = {}
            for sm, sq, af in zip(tr_df["smiles"].astype(str), tr_df["sequence"].astype(str), y_tr_v1):
                entry = tr_drug_seq_aff.setdefault(sm, ([], []))
                entry[0].append(tr_seq_to_idx[sq])
                entry[1].append(float(af))
            tr_drug_arrays = {
                sm: (np.asarray(idxs, dtype=np.intp), np.asarray(affs, dtype=np.float64))
                for sm, (idxs, affs) in tr_drug_seq_aff.items()
            }
            target_means = tr_df.groupby("sequence")["affinity"].mean().to_dict()
            yp_knn_esm = np.empty(len(te_df), dtype=np.float64)
            for row_i, (sm, sq) in enumerate(zip(te_df["smiles"].astype(str), te_df["sequence"].astype(str))):
                entry_arr = tr_drug_arrays.get(sm)
                if entry_arr is None:
                    yp_knn_esm[row_i] = float(target_means.get(sq, y_tr_mean_v1))
                    continue
                c_indices, c_affs = entry_arr
                ti = te_seq_to_idx[sq]
                sims = sim_esm[ti, c_indices]
                order = np.argsort(-sims, kind="stable")[:5]
                top_w = np.maximum(sims[order], 0.0)
                top_a = c_affs[order]
                w_sum = float(np.sum(top_w))
                if w_sum > 1e-6:
                    yp_knn_esm[row_i] = float(np.sum(top_w * top_a) / w_sum)
                else:
                    yp_knn_esm[row_i] = float(np.mean(top_a)) if len(top_a) > 0 else float(target_means.get(sq, y_tr_mean_v1))

            res_cls = _compute_classical_unit(
                tr_df, te_df, km_sim_k, s_idx_k, {}, k=5, tanimoto_sim_mat=tan_sim_k, drug_to_idx=d_idx_k
            )
            preds_seed = {
                "gin_esm_mlp": yp_gin,
                "ridge_esm": yp_ridge_esm,
                "knn_esm_target": yp_knn_esm,
                "knn_kmer_target": res_cls["knn_kmer_target"]["y_pred"],
                "knn_tanimoto_drug": res_cls["knn_tanimoto_drug"]["y_pred"],
                "drug_mean_predictor": res_cls["drug_mean_predictor"]["y_pred"],
                "target_mean_predictor": res_cls["target_mean_predictor"]["y_pred"],
            }
            tot2_s, elig2_s = _prepare_group_indices(grp, yt, min_pairs=2)
            tot5_s, elig5_s = _prepare_group_indices(grp, yt, min_pairs=5)
            for mname, yp in preds_seed.items():
                per_seed_m2.setdefault(mname, []).append(_intragroup_from_prepared(yt, yp, tot2_s, elig2_s))
                per_seed_m5.setdefault(mname, []).append(_intragroup_from_prepared(yt, yp, tot5_s, elig5_s))

        models_summary: dict[str, Any] = {}
        for mname in per_seed_m2:
            m2_list = per_seed_m2[mname]
            m5_list = per_seed_m5[mname]
            models_summary[mname] = {
                "min_pairs_2": {
                    "ci_mean": sample_mean_std([x["ci_mean"] for x in m2_list if x["ci_mean"] is not None]),
                    "spearman_mean": sample_mean_std([x["spearman_mean"] for x in m2_list if x["spearman_mean"] is not None]),
                    "n_groups_evaluable_per_seed": [x["n_groups_evaluable"] for x in m2_list],
                    "n_groups_const_pred_per_seed": [x["n_groups_const_pred"] for x in m2_list],
                },
                "min_pairs_5": {
                    "ci_mean": sample_mean_std([x["ci_mean"] for x in m5_list if x["ci_mean"] is not None]),
                    "spearman_mean": sample_mean_std([x["spearman_mean"] for x in m5_list if x["spearman_mean"] is not None]),
                    "n_groups_evaluable_per_seed": [x["n_groups_evaluable"] for x in m5_list],
                },
            }
        v1_intra[reg_key] = {
            "protocol": "v1_seeds",
            "group_axis": gcol,
            "note_graphdta_gin": "não disponível (runs v1_seeds de graphdta_gin não salvaram oof_predictions)",
            "models": models_summary,
        }
    return v1_intra


def compute_residual_stacking(
    data_root: Path,
    seq_to_vec: dict[str, np.ndarray],
    drug_to_fp: dict[str, np.ndarray],
    stack_datasets: tuple[str, ...] = ("davis", "kiba"),
    stack_protos: tuple[str, ...] = ("kfold", "seeds"),
) -> dict[str, Any]:
    splits_dir = data_root / "data" / "processed" / "splits"
    davis_df = pd.read_csv(data_root / "data" / "processed" / "davis_clean.csv")
    kiba_df = pd.read_csv(data_root / "data" / "processed" / "kiba_clean.csv")
    df_map = {"davis": davis_df, "kiba": kiba_df}

    # 2. Residual stacking baseline on CPU (strict train-only fit, val alpha selection, test evaluation)
    stack_specs = [
        ("davis", "cold_target", "kfold", "smiles"),
        ("davis", "cold_target_seqid40", "kfold", "smiles"),
        ("davis", "cold_target_seqid30", "kfold", "smiles"),
        ("kiba", "cold_target_seqid40", "kfold", "smiles"),
        ("kiba", "cold_target_seqid30", "kfold", "smiles"),
        ("kiba", "cold_target", "seeds", "smiles"),
        ("davis", "cold_drug", "kfold", "sequence"),
        ("kiba", "cold_drug", "seeds", "sequence"),
    ]
    stack_out: dict[str, Any] = {}
    for ds, sp, proto, bias_col in stack_specs:
        if ds not in stack_datasets or proto not in stack_protos:
            continue
        reg_key = f"{ds}/{sp}"
        print(f"  [stacking] {reg_key} ({proto})...", flush=True)
        df = df_map[ds]
        intra_gcol = "smiles" if "cold_target" in sp else "sequence"
        unit_prefix = "fold" if proto == "kfold" else "seed"

        oof_y_true_all: list[np.ndarray] = []
        oof_grp_all: list[np.ndarray] = []
        oof_preds_by_variant: dict[str, list[np.ndarray]] = {
            "bias_only": [],
            "res_esm": [],
            "res_ecfp": [],
            "res_esm_ecfp": [],
            "res_esm_ecfp_inter": [],
        }
        per_unit_metrics: dict[str, list[dict[str, float]]] = {k: [] for k in oof_preds_by_variant}
        chosen_alphas: dict[str, list[float]] = {k: [] for k in oof_preds_by_variant if k != "bias_only"}

        for u in range(5):
            split_idx = load_splits(splits_dir / f"{ds}_{sp}_{unit_prefix}{u}.json")
            tr_df = df.iloc[split_idx["train"]]
            va_df = df.iloc[split_idx["val"]]
            te_df = df.iloc[split_idx["test"]]

            y_tr = tr_df["affinity"].to_numpy(dtype=np.float64)
            y_va = va_df["affinity"].to_numpy(dtype=np.float64)
            y_te = te_df["affinity"].to_numpy(dtype=np.float64)
            global_mean = float(np.mean(y_tr))
            bias_map = tr_df.groupby(bias_col)["affinity"].mean().to_dict()

            b_tr = tr_df[bias_col].astype(str).map(bias_map).fillna(global_mean).to_numpy(dtype=np.float64)
            b_va = va_df[bias_col].astype(str).map(bias_map).fillna(global_mean).to_numpy(dtype=np.float64)
            b_te = te_df[bias_col].astype(str).map(bias_map).fillna(global_mean).to_numpy(dtype=np.float64)

            r_tr = y_tr - b_tr
            r_va = y_va - b_va

            P_tr = np.vstack([seq_to_vec[s] for s in tr_df["sequence"].astype(str)]).astype(np.float32)
            P_va = np.vstack([seq_to_vec[s] for s in va_df["sequence"].astype(str)]).astype(np.float32)
            P_te = np.vstack([seq_to_vec[s] for s in te_df["sequence"].astype(str)]).astype(np.float32)

            D_tr = np.vstack([drug_to_fp[s] for s in tr_df["smiles"].astype(str)]).astype(np.float32)
            D_va = np.vstack([drug_to_fp[s] for s in va_df["smiles"].astype(str)]).astype(np.float32)
            D_te = np.vstack([drug_to_fp[s] for s in te_df["smiles"].astype(str)]).astype(np.float32)

            # Bilinear interactions via TruncatedSVD fit strictly on unique train entities (zero leakage)
            uniq_P_tr = np.vstack([seq_to_vec[s] for s in sorted(set(tr_df["sequence"].astype(str)))])
            uniq_D_tr = np.vstack([drug_to_fp[s] for s in sorted(set(tr_df["smiles"].astype(str)))])
            n_comp_p = min(16, max(2, uniq_P_tr.shape[0] - 1))
            n_comp_d = min(16, max(2, uniq_D_tr.shape[0] - 1))
            svd_p = TruncatedSVD(n_components=n_comp_p, random_state=42).fit(uniq_P_tr)
            svd_d = TruncatedSVD(n_components=n_comp_d, random_state=42).fit(uniq_D_tr)

            def _inter(P_mat: np.ndarray, D_mat: np.ndarray) -> np.ndarray:
                pz = svd_p.transform(P_mat).astype(np.float32)
                dz = svd_d.transform(D_mat).astype(np.float32)
                return (pz[:, :, None] * dz[:, None, :]).reshape(P_mat.shape[0], -1)

            I_tr = _inter(P_tr, D_tr)
            I_va = _inter(P_va, D_va)
            I_te = _inter(P_te, D_te)

            Z_tr = np.hstack([P_tr, D_tr, I_tr])
            Z_va = np.hstack([P_va, D_va, I_va])
            Z_te = np.hstack([P_te, D_te, I_te])
            mu = np.mean(Z_tr, axis=0)
            sd = np.std(Z_tr, axis=0)
            sd[sd < 1e-8] = 1.0
            Z_tr -= mu
            Z_tr /= sd
            Z_va -= mu
            Z_va /= sd
            Z_te -= mu
            Z_te /= sd

            oof_y_true_all.append(y_te)
            oof_grp_all.append(te_df[intra_gcol].astype(str).to_numpy())

            # bias_only
            oof_preds_by_variant["bias_only"].append(b_te)
            per_unit_metrics["bias_only"].append(_fast_all_metrics(y_te, b_te))

            r_tr_mean = float(np.mean(r_tr))
            r_tr_c32 = (r_tr - r_tr_mean).astype(np.float32)

            # Fast float32 Gram matrix (symmetrized in float64) + PSD eigendecomp fallback
            G_full = (Z_tr.T @ Z_tr).astype(np.float64)
            G_full = 0.5 * (G_full + G_full.T)
            g_y_full = (Z_tr.T @ r_tr_c32).astype(np.float64)
            dP = P_tr.shape[1]
            dD = D_tr.shape[1]
            dTot = Z_tr.shape[1]
            slices_map = {
                "res_esm": slice(0, dP),
                "res_ecfp": slice(dP, dP + dD),
                "res_esm_ecfp": slice(0, dP + dD),
                "res_esm_ecfp_inter": slice(0, dTot),
            }

            for vname, sl in slices_map.items():
                G_sub = G_full[sl, sl]
                g_sub = g_y_full[sl]
                dim = G_sub.shape[0]
                diag_idx = np.arange(dim)

                best_alpha = float(RESIDUAL_ALPHAS[0])
                best_val_mse = float("inf")
                best_w: np.ndarray | None = None
                eig_cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
                for alpha in RESIDUAL_ALPHAS:
                    if eig_cache is not None:
                        ev, evec, proj = eig_cache
                        w = evec @ (proj / (ev + float(alpha)))
                    else:
                        A = G_sub.copy()
                        A[diag_idx, diag_idx] += float(alpha)
                        try:
                            c_fac = cho_factor(A, check_finite=False, overwrite_a=True)
                            w = cho_solve(c_fac, g_sub, check_finite=False)
                        except np.linalg.LinAlgError:
                            ev, evec = sp_eigh(G_sub, check_finite=False, driver="evd")
                            ev = np.maximum(ev, 0.0)
                            proj = evec.T @ g_sub
                            eig_cache = (ev, evec, proj)
                            w = evec @ (proj / (ev + float(alpha)))
                    val_pred_r = r_tr_mean + (Z_va[:, sl] @ w.astype(np.float32)).astype(np.float64)
                    val_mse = float(np.mean((r_va - val_pred_r) ** 2))
                    if val_mse < best_val_mse:
                        best_val_mse = val_mse
                        best_alpha = float(alpha)
                        best_w = w

                assert best_w is not None
                yp_te = b_te + r_tr_mean + (Z_te[:, sl] @ best_w.astype(np.float32)).astype(np.float64)
                chosen_alphas[vname].append(best_alpha)
                oof_preds_by_variant[vname].append(yp_te)
                per_unit_metrics[vname].append(_fast_all_metrics(y_te, yp_te))

        yt_oof = np.concatenate(oof_y_true_all)
        grp_oof = np.concatenate(oof_grp_all)
        tot2_oof, elig2_oof = _prepare_group_indices(grp_oof, yt_oof, min_pairs=2) if proto == "kfold" else (0, [])
        tot5_oof, elig5_oof = _prepare_group_indices(grp_oof, yt_oof, min_pairs=5) if proto == "kfold" else (0, [])
        unit_preps = [_prepare_group_indices(oof_grp_all[i], oof_y_true_all[i], min_pairs=2) for i in range(5)]
        variant_summaries: dict[str, Any] = {}
        for vname, ulist in per_unit_metrics.items():
            yp_oof = np.concatenate(oof_preds_by_variant[vname])
            intra2 = _intragroup_from_prepared(yt_oof, yp_oof, tot2_oof, elig2_oof) if proto == "kfold" else None
            intra5 = _intragroup_from_prepared(yt_oof, yp_oof, tot5_oof, elig5_oof) if proto == "kfold" else None
            u_intra2 = [
                _intragroup_from_prepared(oof_y_true_all[i], oof_preds_by_variant[vname][i], unit_preps[i][0], unit_preps[i][1])
                for i in range(5)
            ]
            per_unit_intra2 = {
                "ci_mean": sample_mean_std([x["ci_mean"] for x in u_intra2 if x["ci_mean"] is not None]),
                "spearman_mean": sample_mean_std([x["spearman_mean"] for x in u_intra2 if x["spearman_mean"] is not None]),
            }
            if proto == "seeds":
                intra2 = per_unit_intra2
            variant_summaries[vname] = {
                "per_unit_mean_std": {
                    "ci": sample_mean_std([x["ci"] for x in ulist]),
                    "mse": sample_mean_std([x["mse"] for x in ulist]),
                    "rm2": sample_mean_std([x["rm2"] for x in ulist]),
                },
                "oof_global": _fast_all_metrics(yt_oof, yp_oof) if proto == "kfold" else None,
                "intragroup_min2": intra2,
                "intragroup_per_unit_min2": per_unit_intra2,
                "intragroup_min5": intra5,
                "chosen_alphas": chosen_alphas.get(vname, []),
            }

        stack_out[reg_key] = {
            "protocol": "v2_kfold" if proto == "kfold" else "v1_seeds",
            "residual_target": f"y_true - {bias_col}_mean_train",
            "intragroup_axis": intra_gcol,
            "variants": variant_summaries,
        }

    return stack_out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Análises intragrupo e stacking residual em CPU")
    parser.add_argument(
        "--part",
        choices=("all", "intra", "stack_davis", "stack_kiba", "stack_kiba_kfold", "stack_kiba_seeds"),
        default="all",
        help="Subconjunto a executar (mescla incrementalmente no JSON de saída)",
    )
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    repo_root, data_root = resolve_roots()
    out_path = repo_root / "docs" / "paper" / "phase3_intragroup_and_stacking.json"

    existing: dict[str, Any] = {}
    if out_path.exists() and args.part != "all":
        existing = json.loads(out_path.read_text(encoding="utf-8"))

    davis_df = pd.read_csv(data_root / "data" / "processed" / "davis_clean.csv")
    kiba_df = pd.read_csv(data_root / "data" / "processed" / "kiba_clean.csv")
    all_seqs = sorted(set(davis_df["sequence"].astype(str)).union(set(kiba_df["sequence"].astype(str))))
    all_smiles = sorted(set(davis_df["smiles"].astype(str)).union(set(kiba_df["smiles"].astype(str))))

    seq_to_vec = load_mean_pooled_esm_embeddings(
        cache_dir=data_root / "cache" / "esm_embeddings",
        sequences=all_seqs,
        model_name="esm2_t33_650M_UR50D",
        expected_dim=1280,
    )
    drug_to_fp = compute_morgan_ecfp4_map(all_smiles, radius=2, n_bits=1024)

    intra_v2 = existing.get("intragroup_v2_kfold", {})
    intra_v1 = existing.get("intragroup_v1_seeds", {})
    stacking = existing.get("residual_stacking", {})

    if args.part in ("all", "intra"):
        print("[1/3] Intragroup v2_kfold...", flush=True)
        intra_v2 = compute_intragroup_v2_kfold(data_root)
        print(f"[2/3] Intragroup v1_seeds ({time.perf_counter()-t0:.1f}s)...", flush=True)
        intra_v1 = compute_v1_seeds_intragroup(data_root, seq_to_vec)

    if args.part in ("all", "stack_davis", "stack_kiba", "stack_kiba_kfold", "stack_kiba_seeds"):
        ds_tuple = ("davis", "kiba") if args.part == "all" else ("davis",) if args.part == "stack_davis" else ("kiba",)
        proto_tuple = (
            ("kfold",) if args.part == "stack_kiba_kfold"
            else ("seeds",) if args.part == "stack_kiba_seeds"
            else ("kfold", "seeds")
        )
        print(f"[stacking] datasets={ds_tuple} protos={proto_tuple} ({time.perf_counter()-t0:.1f}s)...", flush=True)
        new_stack = compute_residual_stacking(
            data_root, seq_to_vec, drug_to_fp, stack_datasets=ds_tuple, stack_protos=proto_tuple
        )
        stacking.update(new_stack)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "intragroup_definition": {
            "cold_target_group": "smiles (por fármaco; mede ordenação de alvos inéditos para o mesmo fármaco)",
            "cold_drug_group": "sequence (por alvo; mede ordenação de fármacos inéditos contra o mesmo alvo)",
            "evaluable_filter": "grupos com n >= 2 (e sensibilidade n >= 5) pares de teste e >= 2 valores distintos em y_true",
            "constant_prediction_convention": "CI = 0.5000 (compute_ci), Spearman = 0.0 (contagem registrada em n_groups_const_pred)",
        },
        "intragroup_v2_kfold": intra_v2,
        "intragroup_v1_seeds": intra_v1,
        "residual_stacking": stacking,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] phase3_intragroup_and_stacking.json ({args.part}) salvo em {payload['elapsed_sec']}s: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
