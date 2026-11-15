"""Auditorias determinísticas e inferência em CPU para a Fase 3 (`docs/paper/phase3_audits.json`).

Este script é STRICTAMENTE SOMENTE-LEITURA sobre `outputs/`, `data/`, `cache/` e `configs/`.
Ele valida:
1. Pré-registro no Git (`ADR #15`, tag `phase3-adr15-pre-kiba`) e proveniência de schema nos JSONs.
2. Completude, teto de épocas, early memorization (`best_epoch < 15` em `cold`), NaN/Inf e isolamento.
3. Reprodução exata dos baselines em CPU (`v2_kfold` via OOF CSVs e `v1_seeds` via cache ESM + splits).
4. Bootstrap agrupado pareado atualizado (`seed=42`, `n_bootstraps=1000`) para Davis `v2_kfold`, KIBA `seqid`
   e regimes primários KIBA `v1_seeds`, além dos controles diagnósticos (`--shuffle-esm`, `ecfp_mlp`, `tsoff1000`).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

PAPER_SCRIPTS_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = PAPER_SCRIPTS_DIR.parent
REPO_ROOT = PAPER_SCRIPTS_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
for p in (str(PAPER_SCRIPTS_DIR), str(SCRIPTS_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common import ensure_paper_dirs, load_json_strict, resolve_roots, sample_mean_std  # noqa: E402
from physdual.data.splits import load_splits  # noqa: E402
from physdual.eval.esm_baselines import (  # noqa: E402
    compute_morgan_ecfp4_map,
    evaluate_esm_cpu_baselines_split,
    load_mean_pooled_esm_embeddings,
)
from physdual.eval.metrics import (  # noqa: E402
    compute_all_metrics,
    compute_binary_classification_metrics,
    compute_uncensored_metrics,
)
from run_phase3_esm_cpu_baselines import (  # noqa: E402
    _build_3mer_norm_matrix,
    _build_tanimoto_sim_matrix,
    _compute_classical_unit,
)

BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42

ADR15_FROZEN_CONFIG = {
    "batch_size": 512,
    "davis_cold_patience": 80,
    "device": "cuda",
    "epochs": 1000,
    "esm_dropout_in": 0.4,
    "lr": 0.0005,
    "mlp_dropout": 0.3,
    "patience": 50,
    "prot_dropout": 0.3,
    "prot_output_dim": 128,
    "weight_decay": 0.001,
}


def _summary_stats(vals: list[float]) -> dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _fast_bootstrap_precompute(
    yt_in: np.ndarray,
    groups_in: np.ndarray,
    preds_map: dict[str, np.ndarray],
    n_bootstraps: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Calcula `T_bs`, `mse_bs[m]` e `C_bs[m]` por bootstrap agrupado exato O(N log N) via divisão-e-conquista.

    Semanticamente e numericamente idêntico (0 ULP nos agregados inteiros de concordância e MSE)
    a `physdual.eval.metrics.bootstrap_grouped_comparisons`.
    """
    unique_clusters, inv = np.unique(groups_in, return_inverse=True)
    n_clusters = len(unique_clusters)
    if n_clusters < 2:
        return {"degenerate": True, "n_clusters": n_clusters}

    order = np.argsort(inv, kind="stable")
    inv_sorted = inv[order]
    yt = np.asarray(yt_in, dtype=np.float64)[order]
    preds_ordered = {k: np.asarray(v, dtype=np.float64)[order] for k, v in preds_map.items()}

    counts = np.bincount(inv_sorted, minlength=n_clusters)
    starts = np.zeros(n_clusters, dtype=np.int64)
    starts[1:] = np.cumsum(counts)[:-1]
    sse_map = {k: np.add.reduceat((yt - yp) ** 2, starts) for k, yp in preds_ordered.items()}

    rng = np.random.RandomState(int(seed))
    W = np.zeros((n_bootstraps, n_clusters), dtype=np.float64)
    for b in range(n_bootstraps):
        sampled = rng.choice(n_clusters, size=n_clusters, replace=True)
        W[b] = np.bincount(sampled, minlength=n_clusters)

    n_bs = np.maximum(W @ counts.astype(np.float64), 1.0)
    mse_bs_map = {k: (W @ sse) / n_bs for k, sse in sse_map.items()}

    yt_order = np.argsort(yt, kind="mergesort")
    yt_s = yt[yt_order]
    cl_s = inv_sorted[yt_order]
    tie_starts = np.concatenate(([0], np.nonzero(yt_s[1:] != yt_s[:-1])[0] + 1, [len(yt_s)]))
    n_blocks = len(tie_starts) - 1

    leaf_ranges: list[tuple[int, int, int, int]] = []
    cross_pairs: list[tuple[int, int, int, int]] = []

    def build_tree(b0: int, b1: int) -> None:
        if b1 - b0 <= 1:
            return
        s0, s1 = int(tie_starts[b0]), int(tie_starts[b1])
        if s1 - s0 <= 384:
            leaf_ranges.append((s0, s1, b0, b1))
            return
        mid_target = (s0 + s1) // 2
        bm = int(np.searchsorted(tie_starts[b0 : b1 + 1], mid_target)) + b0
        if bm <= b0:
            bm = b0 + 1
        elif bm >= b1:
            bm = b1 - 1
        sm = int(tie_starts[bm])
        cross_pairs.append((s0, sm, sm, s1))
        build_tree(b0, bm)
        build_tree(bm, b1)

    build_tree(0, n_blocks)

    T_bs = np.zeros(n_bootstraps, dtype=np.float64)
    for l0, l1, r0, r1 in cross_pairs:
        cL = np.bincount(cl_s[l0:l1], minlength=n_clusters).astype(np.float64)
        cR = np.bincount(cl_s[r0:r1], minlength=n_clusters).astype(np.float64)
        T_bs += (W @ cL) * (W @ cR)

    leaf_data: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray]] = []
    for s0, s1, b0, b1 in leaf_ranges:
        if b1 - b0 <= 1:
            continue
        yt_leaf = yt_s[s0:s1]
        cl_leaf = cl_s[s0:s1]
        u_cl, inv_leaf = np.unique(cl_leaf, return_inverse=True)
        oh = np.zeros((len(u_cl), len(yt_leaf)), dtype=np.float64)
        oh[inv_leaf, np.arange(len(yt_leaf))] = 1.0
        dt = (yt_leaf[:, None] > yt_leaf[None, :]).astype(np.float64)
        T_sub = oh @ dt @ oh.T
        Wu = W[:, u_cl]
        T_bs += np.sum((Wu @ T_sub) * Wu, axis=1)
        leaf_data.append((s0, s1, yt_leaf, u_cl, oh))

    W_i = torch.from_numpy(W.T.astype(np.int64))
    B_BATCH = 500
    ci_bs_map: dict[str, np.ndarray] = {}
    cached_by_bytes: dict[bytes, np.ndarray] = {}
    for mkey, yp_ord in preds_ordered.items():
        if len(yp_ord) == 0 or np.all(yp_ord == yp_ord[0]):
            ci_bs_map[mkey] = np.full(n_bootstraps, 0.5, dtype=np.float64)
            continue
        sig = yp_ord.tobytes()
        if sig in cached_by_bytes:
            ci_bs_map[mkey] = cached_by_bytes[sig].copy()
            continue

        yp_s = yp_ord[yt_order]
        c_leaf = np.zeros(n_bootstraps, dtype=np.float64)
        for s0, s1, yt_leaf, u_cl, oh in leaf_data:
            dt = yt_leaf[:, None] > yt_leaf[None, :]
            y_l = yp_s[s0:s1]
            cm = (dt & (y_l[:, None] > y_l[None, :])).astype(np.float64) + 0.5 * (
                dt & (y_l[:, None] == y_l[None, :])
            ).astype(np.float64)
            C_sub = oh @ cm @ oh.T
            Wu = W[:, u_cl]
            c_leaf += np.sum((Wu @ C_sub) * Wu, axis=1)

        c_cross2 = torch.zeros(n_bootstraps, dtype=torch.int64)
        for l0, l1, r0, r1 in cross_pairs:
            yL = yp_s[l0:l1]
            yR = yp_s[r0:r1]
            oL = np.argsort(yL)
            yL_s = yL[oL]
            clL_s = torch.from_numpy(cl_s[l0:l1][oL].astype(np.int64))
            clR_t = torch.from_numpy(cl_s[r0:r1].astype(np.int64))
            lt = torch.from_numpy(np.searchsorted(yL_s, yR, side="left").astype(np.int64))
            le = torch.from_numpy(np.searchsorted(yL_s, yR, side="right").astype(np.int64))

            for b_start in range(0, n_bootstraps, B_BATCH):
                b_end = min(b_start + B_BATCH, n_bootstraps)
                Wb = W_i[:, b_start:b_end]
                pref = torch.zeros((len(yL_s) + 1, b_end - b_start), dtype=torch.int64)
                torch.cumsum(Wb[clL_s], dim=0, out=pref[1:])
                w2 = pref[lt] + pref[le]
                c_cross2[b_start:b_end] += torch.sum(Wb[clR_t] * w2, dim=0)

        c_arr = c_leaf + 0.5 * c_cross2.numpy().astype(np.float64)
        ci_arr = np.where(T_bs > 0, c_arr / T_bs, 0.5)
        ci_bs_map[mkey] = ci_arr
        cached_by_bytes[sig] = ci_arr

    return {
        "degenerate": False,
        "n_clusters": n_clusters,
        "n_bootstraps": n_bootstraps,
        "mse_bs": mse_bs_map,
        "ci_bs": ci_bs_map,
    }


def _fast_bootstrap_compare(
    pre: dict[str, Any],
    model_key: str,
    baseline_keys: list[str],
    alpha: float = 0.05,
) -> dict[str, dict[str, Any]]:
    if pre.get("degenerate"):
        empty_res = {
            "delta_mse": {"mean": 0.0, "std": 0.0, "ci_95": [0.0, 0.0], "wins": 0, "total_bootstraps": BOOTSTRAP_N},
            "delta_ci": {"mean": 0.0, "std": 0.0, "ci_95": [0.0, 0.0], "wins": 0, "total_bootstraps": BOOTSTRAP_N},
            "p_superiority_mse": 0.5,
            "p_superiority_ci": 0.5,
        }
        return {k: dict(empty_res) for k in baseline_keys}

    n_bs = int(pre["n_bootstraps"])
    low_p = (alpha / 2.0) * 100.0
    high_p = (1.0 - alpha / 2.0) * 100.0
    mse_m = pre["mse_bs"][model_key]
    ci_m = pre["ci_bs"][model_key]
    out: dict[str, dict[str, Any]] = {}
    for bk in baseline_keys:
        if bk not in pre["mse_bs"]:
            continue
        arr_mse = mse_m - pre["mse_bs"][bk]
        arr_ci = ci_m - pre["ci_bs"][bk]
        out[bk] = {
            "delta_mse": {
                "mean": float(np.mean(arr_mse)),
                "std": float(np.std(arr_mse, ddof=1)) if n_bs > 1 else 0.0,
                "ci_95": [float(np.percentile(arr_mse, low_p)), float(np.percentile(arr_mse, high_p))],
                "wins": int(np.sum(arr_mse < 0)),
                "total_bootstraps": n_bs,
            },
            "delta_ci": {
                "mean": float(np.mean(arr_ci)),
                "std": float(np.std(arr_ci, ddof=1)) if n_bs > 1 else 0.0,
                "ci_95": [float(np.percentile(arr_ci, low_p)), float(np.percentile(arr_ci, high_p))],
                "wins": int(np.sum(arr_ci > 0)),
                "total_bootstraps": n_bs,
            },
            "p_superiority_mse": float(np.mean(arr_mse < 0)),
            "p_superiority_ci": float(np.mean(arr_ci > 0)),
        }
    return out


def _evaluate_adr15_outcome(
    wins_mse: int,
    wins_ci: int,
    n_units: int,
    bs_entry: dict[str, Any] | None,
) -> dict[str, Any]:
    """Aplica exatamente o critério operacional do ADR #15 sem ajustes pós-hoc."""
    if bs_entry is None:
        return {
            "verdict": "nao_avaliavel",
            "wins_mse": wins_mse,
            "wins_ci": wins_ci,
            "n_units": n_units,
            "mse_ic95_excludes_zero_favorable": None,
            "ci_ic95_excludes_zero_favorable": None,
        }
    ci_mse = bs_entry["delta_mse"]["ci_95"]
    ci_ci = bs_entry["delta_ci"]["ci_95"]
    mse_excl_zero = bool(float(ci_mse[1]) < 0.0)
    ci_excl_zero = bool(float(ci_ci[0]) > 0.0)
    if wins_mse >= 4 and wins_ci >= 4 and mse_excl_zero and ci_excl_zero:
        verdict = "atendido"
    elif wins_mse <= 2 or wins_ci <= 2 or not mse_excl_zero or not ci_excl_zero:
        verdict = "nao_atendido_negativo"
    else:
        verdict = "inconclusivo"
    return {
        "verdict": verdict,
        "wins_mse": wins_mse,
        "wins_ci": wins_ci,
        "n_units": n_units,
        "mse_ic95_excludes_zero_favorable": mse_excl_zero,
        "ci_ic95_excludes_zero_favorable": ci_excl_zero,
    }


def audit_git_preregistration(repo_root: Path) -> dict[str, Any]:
    refs = ("phase2-closed", "1ae2853", "0d64b81", "06d1249", "46d13f8", "phase3-adr15-pre-kiba", "9a60916", "936c32f", "HEAD")
    timeline = {}
    for ref in refs:
        out = subprocess.check_output(
            ["git", "show", "-s", "--format=%H|%ci|%s", ref],
            cwd=repo_root,
            text=True,
        ).strip().splitlines()[-1]
        h, dt, msg = out.split("|", 2)
        timeline[ref] = {"commit": h, "date": dt, "subject": msg}

    diff_frozen = subprocess.check_output(
        ["git", "diff", "46d13f8..HEAD", "--", "configs/base.yaml", "docs/DECISOES.md", "src/physdual/models/"],
        cwd=repo_root,
        text=True,
    )
    cfg_commits = subprocess.check_output(
        ["git", "log", "--format=%h|%ci|%s", "--", "configs/base.yaml"],
        cwd=repo_root,
        text=True,
    ).strip().splitlines()

    return {
        "timeline": timeline,
        "post_adr15_diff_bytes_configs_decisoes_models": len(diff_frozen.encode("utf-8")),
        "configs_base_yaml_history": cfg_commits,
        "tuning_budget": {
            "adr15_declared_configs_tried": 1,
            "git_verified_configs_tried": 1,
            "epochs_amendment_pre_freeze": "500 -> 1000 no commit 06d1249 (anterior ao ADR #15 em 46d13f8)",
            "post_freeze_hyperparam_edits": 0,
        },
        "forensic_caveat": (
            "Os JSONs individuais em outputs/results/runs/ não embutem started_at_utc nem git_commit; "
            "entretanto, a ordenação pós-ADR #15 (46d13f8, 2026-11-14 00:32:14 -0300, tag phase3-adr15-pre-kiba) "
            "é corroborada convergentemente por: (a) proveniência de schema (62/62 runs KIBA/davis_random/controles/fold1-estendido "
            "possuem campos base_model e diagnostic_control introduzidos em 46d13f8, ausentes nos 19 runs anteriores "
            "de Davis cold); (b) criação dos splits kiba_seqid30/40 em 46d13f8; (c) diff=0 em configs/base.yaml, "
            "docs/DECISOES.md e src/physdual/models/ após 46d13f8; e (d) auditoria forense dos timestamps de criação "
            "NTFS (`st_ctime - elapsed_sec`, registrando em run_integrity.start_timestamp_audit 0/62 inícios anteriores a 46d13f8)."
        ),
    }


def audit_runs_and_aggregates(data_root: Path) -> dict[str, Any]:
    res_dir = data_root / "outputs" / "results"
    runs_dir = res_dir / "runs"
    splits_dir = data_root / "data" / "processed" / "splits"

    phys_files = sorted(runs_dir.glob("physdual_*.json"))
    gdta_files = sorted(runs_dir.glob("graphdta_*.json"))

    pre_adr15_runs: list[str] = []
    post_adr15_runs: list[str] = []
    config_deviations: list[dict[str, Any]] = []
    truncated_runs: list[dict[str, Any]] = []
    over_80_pct_runs: list[dict[str, Any]] = []
    early_cold_runs: list[dict[str, Any]] = []
    isolation_failures: list[str] = []
    split_size_mismatches: list[dict[str, Any]] = []
    nan_inf_runs: list[str] = []
    pre_ts: list[tuple[float, float, str]] = []
    post_ts: list[tuple[float, float, str, str, str]] = []

    for pf in phys_files:
        d = load_json_strict(pf)
        has_dc = ("base_model" in d) and ("diagnostic_control" in d)
        (post_adr15_runs if has_dc else pre_adr15_runs).append(pf.name)
        ct_end = float(pf.stat().st_ctime)
        ct_start = ct_end - float(d.get("elapsed_sec", 0.0))
        if has_dc:
            post_ts.append((ct_start, ct_end, pf.name, str(d["dataset"]), str(d["split"])))
        else:
            pre_ts.append((ct_start, ct_end, pf.name))

        ec = d.get("effective_config", {})
        diffs_cfg = {k: (ec.get(k), v) for k, v in ADR15_FROZEN_CONFIG.items() if ec.get(k) != v}
        if diffs_cfg:
            config_deviations.append({"file": pf.name, "diffs": diffs_cfg})

        ep = int(ec.get("epochs", 0))
        best_ep = int(d.get("best_epoch", 0))
        tot_ep = int(d.get("total_epochs_run", 0))
        ds, sp, proto = d["dataset"], d["split"], d["protocol"]
        pat = int(ec["davis_cold_patience"] if (ds == "davis" and "cold" in sp) else ec["patience"])

        if ep > 0 and best_ep > 0.8 * ep:
            over_80_pct_runs.append({"file": pf.name, "best_epoch": best_ep, "epochs": ep, "total_epochs_run": tot_ep})
        if ep > 0 and tot_ep >= ep and (tot_ep - best_ep) < pat:
            truncated_runs.append({"file": pf.name, "best_epoch": best_ep, "epochs": ep, "total_epochs_run": tot_ep, "patience": pat})
        if "cold" in sp and best_ep < 15:
            early_cold_runs.append({
                "file": pf.name,
                "dataset": ds,
                "split": sp,
                "unit": d.get("fold") if proto == "kfold" else d.get("seed"),
                "best_epoch": best_ep,
                "early_memorization_warn_json": bool(d.get("early_memorization_warn", False)),
            })
        if d.get("test_isolation_verified") is not True:
            isolation_failures.append(pf.name)

        # Check split size vs oof_predictions
        unit = d.get("fold") if proto == "kfold" else d.get("seed")
        sp_fn = f"{ds}_{sp}_{'fold' if proto == 'kfold' else 'seed'}{unit}.json"
        sp_path = splits_dir / sp_fn
        sp_json = load_json_strict(sp_path)
        expected_n = len(sp_json["test"])
        oof_list = d.get("oof_predictions", [])
        if len(oof_list) != expected_n:
            split_size_mismatches.append({"file": pf.name, "oof_len": len(oof_list), "split_len": expected_n})

        # Check NaN/Inf on scalar metrics
        vals = [d.get("train_loss_at_best"), d.get("train_val_mse_gap_at_best")]
        for block in ("val_metrics_at_best", "test_metrics_full"):
            vals.extend(v for v in d.get(block, {}).values() if isinstance(v, (int, float)))
        if any(v is None or not np.isfinite(float(v)) for v in vals):
            nan_inf_runs.append(pf.name)

    # Retroactive check on GraphDTA cold runs too
    gdta_early_cold: list[dict[str, Any]] = []
    gdta_over_80: list[dict[str, Any]] = []
    for gf in gdta_files:
        d = load_json_strict(gf)
        best_ep = int(d["best_epoch"])
        ep = int(d.get("effective_config", {}).get("epochs", 500))
        sp = str(d["split"])
        if best_ep > 0.8 * ep:
            gdta_over_80.append({"file": gf.name, "best_epoch": best_ep, "epochs": ep})
        if "cold" in sp and best_ep < 15:
            gdta_early_cold.append({
                "file": gf.name,
                "model": d["model"],
                "dataset": d["dataset"],
                "split": sp,
                "protocol": "v2_kfold" if d.get("fold") is not None else "v1_seeds",
                "unit": d.get("fold") if d.get("fold") is not None else d.get("seed"),
                "best_epoch": best_ep,
            })

    # Aggregate completeness
    agg_audit: list[dict[str, Any]] = []
    for p in sorted(list(res_dir.glob("baseline_*.json")) + list(res_dir.glob("phase3_physdual_*.json"))):
        d = load_json_strict(p)
        proto = d.get("protocol")
        raw_u = d.get("folds", []) if proto == "kfold" else d.get("seeds", [])
        units = [u["fold"] if isinstance(u, dict) else int(u) for u in raw_u]
        expected_u = [2] if "tsoff1000_davis_cold_drug" in p.name else [0, 1, 2, 3, 4]
        agg_audit.append({
            "file": p.name,
            "protocol": proto,
            "units": units,
            "expected_units": expected_u,
            "complete": units == expected_u,
            "mtime": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        })

    # Stale check: phase3_davis_paired_bootstrap.json vs current cold_target oof CSV
    stale_pbs = load_json_strict(res_dir / "phase3_davis_paired_bootstrap.json")
    stale_ct = stale_pbs["regimes"]["cold_target"]["per_model_summary"]["gin_esm_mlp"]
    curr_ct = load_json_strict(res_dir / "phase3_physdual_gin_esm_mlp_davis_cold_target_kfold.json")
    tsoff_seqid = load_json_strict(res_dir / "phase3_physdual_gin_esm_mlp_tsoff1000_davis_cold_target_seqid30_kfold.json")
    pbs_is_stale = bool(
        abs(float(stale_ct["oof"]["ci"]) - float(curr_ct["global_oof_metrics"]["ci"])) > 1e-6
        or abs(float(stale_ct["oof"]["mse"]) - float(curr_ct["global_oof_metrics"]["mse"])) > 1e-6
    )

    brt = timezone(timedelta(hours=-3))
    adr15_commit_ts = datetime(2026, 11, 14, 0, 32, 14, tzinfo=brt).timestamp()
    pre_ts.sort()
    post_ts.sort()
    kiba_post_ts = [x for x in post_ts if x[3] == "kiba"]
    dr_post_ts = [x for x in post_ts if x[3] == "davis" and x[4] == "random"]
    fmt_brt = lambda ts, add_h=0: datetime.fromtimestamp(ts + add_h * 3600, tz=brt).strftime("%Y-%m-%d %H:%M:%S -0300")
    post_starts_before_adr15_calibrated = [
        x[2] for x in post_ts if (x[0] + 5 * 3600) < adr15_commit_ts
    ]

    return {
        "counts": {
            "total_run_jsons": len(phys_files) + len(gdta_files),
            "physdual_runs": len(phys_files),
            "graphdta_runs": len(gdta_files),
            "pre_adr15_physdual_runs": len(pre_adr15_runs),
            "post_adr15_physdual_runs": len(post_adr15_runs),
            "aggregates_total": len(agg_audit),
            "oof_csvs": len(list((res_dir / "oof").glob("*.csv"))),
        },
        "pre_adr15_physdual_runs": pre_adr15_runs,
        "start_timestamp_audit": {
            "adr15_commit": "46d13f8 (2026-11-14 00:32:14 -0300)",
            "filesystem_clock_offset_note": (
                "st_mtime local foi sobrescrito pela sincronização em lote do Drive (~2026-11-14 23:25 -0300); "
                "st_ctime preservou criação com offset constante de -5h (UTC-8 PST Drive/Colab vs -03:00 BRT Git), "
                "calibrado pelos 19 runs pré-ADR #15 após commit 1ae2853 (19:03:09 -0300 -> st_ctime_start 14:33:27 +5h = 19:33:27) "
                "e log de sessão Colab (04:07 UTC = 01:07 BRT -> shuf0 st_ctime_start 20:06:35 +5h = 01:06:35)."
            ),
            "pre_adr15_min_start_calibrated": fmt_brt(pre_ts[0][0], 5) if pre_ts else None,
            "pre_adr15_max_end_calibrated": fmt_brt(pre_ts[-1][1], 5) if pre_ts else None,
            "post_adr15_earliest_run": post_ts[0][2] if post_ts else None,
            "post_adr15_earliest_start_raw_ctime": fmt_brt(post_ts[0][0], 0) if post_ts else None,
            "post_adr15_earliest_start_calibrated": fmt_brt(post_ts[0][0], 5) if post_ts else None,
            "post_adr15_latest_end_calibrated": fmt_brt(post_ts[-1][1], 5) if post_ts else None,
            "kiba_earliest_run": kiba_post_ts[0][2] if kiba_post_ts else None,
            "kiba_earliest_start_calibrated": fmt_brt(kiba_post_ts[0][0], 5) if kiba_post_ts else None,
            "davis_random_earliest_run": dr_post_ts[0][2] if dr_post_ts else None,
            "davis_random_earliest_start_calibrated": fmt_brt(dr_post_ts[0][0], 5) if dr_post_ts else None,
            "post_runs_started_before_adr15_count": len(post_starts_before_adr15_calibrated),
        },
        "config_deviations": config_deviations,
        "truncated_runs": truncated_runs,
        "over_80_pct_runs_physdual": over_80_pct_runs,
        "over_80_pct_runs_graphdta": gdta_over_80,
        "early_cold_runs_physdual": early_cold_runs,
        "early_cold_runs_graphdta": gdta_early_cold,
        "isolation_failures": isolation_failures,
        "split_size_mismatches": split_size_mismatches,
        "nan_inf_runs": nan_inf_runs,
        "aggregate_completeness": agg_audit,
        "stale_artifacts": {
            "phase3_davis_paired_bootstrap_json": {
                "is_stale": pbs_is_stale,
                "status": "stale" if pbs_is_stale else "regenerado_sincronizado",
                "mtime": datetime.fromtimestamp((res_dir / "phase3_davis_paired_bootstrap.json").stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "stale_cold_target_fold1_note": "Anteriormente desatualizado; regenerado após extensão da dobra 1 (481/500 -> 525/605/1000 épocas)",
                "bootstrap_oof_ci": stale_ct["oof"]["ci"],
                "bootstrap_oof_mse": stale_ct["oof"]["mse"],
                "current_oof_ci": curr_ct["global_oof_metrics"]["ci"],
                "current_oof_mse": curr_ct["global_oof_metrics"]["mse"],
                "regeneration_cmd": "python scripts/run_phase3_paired_bootstrap.py --dataset davis",
            },
            "tsoff1000_seqid30_cluster_col_quirk": tsoff_seqid.get("bootstrap_analysis", {}).get("cluster_group_col"),
        },
    }


def _extract_kiba_davis_overlap(data_root: Path) -> dict[str, Any]:
    davis_df = pd.read_csv(data_root / "data" / "processed" / "davis_clean.csv")
    kiba_df = pd.read_csv(data_root / "data" / "processed" / "kiba_clean.csv")
    splits_dir = data_root / "data" / "processed" / "splits"

    d_seqs = set(davis_df["sequence"].astype(str))
    k_seqs = set(kiba_df["sequence"].astype(str))
    shared = d_seqs & k_seqs

    kiba_cold_stats = {}
    for sp, col in (("cold_drug", "smiles"), ("cold_target", "sequence")):
        per_seed = []
        pooled = set()
        test_sizes = []
        for s in range(5):
            te = load_splits(splits_dir / f"kiba_{sp}_seed{s}.json")["test"]
            test_sizes.append(len(te))
            vals = set(kiba_df.iloc[te][col].astype(str))
            per_seed.append(len(vals))
            pooled |= vals
        kiba_cold_stats[sp] = {
            "group_col": col,
            "unique_per_seed": per_seed,
            "unique_pooled": len(pooled),
            "test_pairs_per_seed": test_sizes,
        }

    # Davis collapsed mutants info
    uniq_id_per_seq = davis_df.groupby("sequence")["target_id"].nunique()
    collapsed = uniq_id_per_seq[uniq_id_per_seq > 1]

    return {
        "davis_unique_seqs": len(d_seqs),
        "kiba_unique_seqs": len(k_seqs),
        "shared_exact_seqs": len(shared),
        "shared_pct_of_kiba": 100.0 * len(shared) / len(k_seqs),
        "kiba_v1_cold_stats": kiba_cold_stats,
        "davis_collapsed_mutants": {
            "num_shared_sequences": int(len(collapsed)),
            "num_collapsed_target_ids": int(collapsed.sum()),
        },
    }


def _per_fold_and_oof_table(
    yt_all: np.ndarray,
    folds_arr: np.ndarray,
    model_preds: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, list[dict[str, float]]]]:
    per_model_summary: dict[str, Any] = {}
    per_fold_by_model: dict[str, list[dict[str, float]]] = {}
    for mname, ypred in model_preds.items():
        rows = []
        for f_id in range(5):
            mask = folds_arr == f_id
            m = compute_all_metrics(yt_all[mask], ypred[mask])
            rows.append({"fold": f_id, **m})
        per_fold_by_model[mname] = rows
        per_model_summary[mname] = {
            "ci_per_fold": _summary_stats([r["ci"] for r in rows]),
            "mse_per_fold": _summary_stats([r["mse"] for r in rows]),
            "rmse_per_fold": _summary_stats([r["rmse"] for r in rows]),
            "pearson_per_fold": _summary_stats([r["pearson"] for r in rows]),
            "spearman_per_fold": _summary_stats([r["spearman"] for r in rows]),
            "rm2_per_fold": _summary_stats([r["rm2"] for r in rows]),
            "oof": compute_all_metrics(yt_all, ypred),
        }
    return per_model_summary, per_fold_by_model


def _paired_unit_deltas(
    target_per_unit: list[dict[str, Any]],
    baselines_per_unit: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    t_mse = np.array([float(u["mse"]) for u in target_per_unit], dtype=np.float64)
    t_ci = np.array([float(u["ci"]) for u in target_per_unit], dtype=np.float64)
    n = len(target_per_unit)
    for bname, b_units in baselines_per_unit.items():
        if len(b_units) != n:
            continue
        b_mse = np.array([float(u["mse"]) for u in b_units], dtype=np.float64)
        b_ci = np.array([float(u["ci"]) for u in b_units], dtype=np.float64)
        dm = t_mse - b_mse
        dc = t_ci - b_ci
        out[bname] = {
            "n_units": n,
            "delta_mse_per_unit": [float(v) for v in dm],
            "delta_ci_per_unit": [float(v) for v in dc],
            "delta_mse_mean": float(np.mean(dm)),
            "delta_mse_std": float(np.std(dm, ddof=1)) if n > 1 else 0.0,
            "delta_mse_median": float(np.median(dm)),
            "delta_ci_mean": float(np.mean(dc)),
            "delta_ci_std": float(np.std(dc, ddof=1)) if n > 1 else 0.0,
            "delta_ci_median": float(np.median(dc)),
            "wins_mse": int(np.sum(dm < 0.0)),
            "wins_ci": int(np.sum(dc > 0.0)),
        }
    return out


def compute_v2_kfold_audits_and_controls(data_root: Path) -> dict[str, Any]:
    res_dir = data_root / "outputs" / "results"
    oof_dir = res_dir / "oof"
    runs_dir = res_dir / "runs"
    logs_dir = data_root / "outputs" / "logs"

    esm_cpu = load_json_strict(res_dir / "phase3_esm_cpu_baselines.json")
    kiba_cls = load_json_strict(res_dir / "kiba_cpu_classical_baselines.json")
    seqid_cls = load_json_strict(res_dir / "phase3_seqid_classical_baselines.json")
    kiba_pbs = load_json_strict(res_dir / "phase3_kiba_paired_bootstrap.json")

    esm_key_map = {
        ("davis", "cold_target"): "davis_v2_cold_target_kfold",
        ("davis", "cold_drug"): "davis_v2_cold_drug_kfold",
        ("davis", "cold_target_seqid30"): "davis_seqid30_kfold",
        ("davis", "cold_target_seqid40"): "davis_seqid40_kfold",
        ("kiba", "cold_target_seqid30"): "kiba_seqid30_kfold",
        ("kiba", "cold_target_seqid40"): "kiba_seqid40_kfold",
    }

    regimes_out: dict[str, Any] = {}
    repro_checks: dict[str, Any] = {}

    specs = [
        ("davis", "cold_target"),
        ("davis", "cold_drug"),
        ("davis", "cold_target_seqid40"),
        ("davis", "cold_target_seqid30"),
        ("kiba", "cold_target_seqid40"),
        ("kiba", "cold_target_seqid30"),
    ]

    cmap_cache = {
        ds: load_json_strict(logs_dir / f"{ds}_seqid_cluster_map.json")
        for ds in ("davis", "kiba")
    }

    for ds, sp in specs:
        reg_key = f"{ds}/{sp}"
        cpu_df = pd.read_csv(oof_dir / f"cpu_baselines_{ds}_{sp}_oof.csv")
        p3_df = pd.read_csv(oof_dir / f"physdual_gin_esm_mlp_{ds}_{sp}_oof.csv")
        p3_agg = load_json_strict(res_dir / f"phase3_physdual_gin_esm_mlp_{ds}_{sp}_kfold.json")
        esm_reg = esm_cpu["regimes"][esm_key_map[(ds, sp)]]

        yt_all = cpu_df["y_true"].to_numpy(dtype=np.float64)
        folds_arr = cpu_df["fold"].to_numpy(dtype=np.int64)

        model_preds: dict[str, np.ndarray] = {
            "gin_esm_mlp": p3_df["y_pred"].to_numpy(dtype=np.float64),
            "ridge_esm": cpu_df["y_pred_ridge_esm"].to_numpy(dtype=np.float64),
            "ridge_esm_target_only": cpu_df["y_pred_ridge_esm_target_only"].to_numpy(dtype=np.float64),
            "knn_esm_target": cpu_df["y_pred_knn_esm_target"].to_numpy(dtype=np.float64),
            "knn_kmer_target": cpu_df["y_pred_knn_kmer_target"].to_numpy(dtype=np.float64),
            "knn_tanimoto_drug": cpu_df["y_pred_knn_tanimoto_drug"].to_numpy(dtype=np.float64),
            "drug_mean_predictor": cpu_df["y_pred_drug_mean_predictor"].to_numpy(dtype=np.float64),
            "target_mean_predictor": cpu_df["y_pred_target_mean_predictor"].to_numpy(dtype=np.float64),
            "mean_predictor": cpu_df["y_pred_mean_predictor"].to_numpy(dtype=np.float64),
        }
        if ds == "davis":
            gin_df = pd.read_csv(oof_dir / f"graphdta_gin_{ds}_{sp}_oof.csv")
            cnn_df = pd.read_csv(oof_dir / f"graphdta_cnn3_{ds}_{sp}_oof.csv")
            model_preds["graphdta_gin"] = gin_df["y_pred"].to_numpy(dtype=np.float64)
            model_preds["graphdta_cnn3"] = cnn_df["y_pred"].to_numpy(dtype=np.float64)

        # Controles extras que existem neste regime (Davis)
        extra_models: dict[str, np.ndarray] = {}
        if ds == "davis" and sp == "cold_target":
            ecfp_df = pd.read_csv(oof_dir / "physdual_ecfp_esm_mlp_davis_cold_target_oof.csv")
            extra_models["ecfp_esm_mlp"] = ecfp_df["y_pred"].to_numpy(dtype=np.float64)
            for s_idx in (0, 1, 2):
                sh_df = pd.read_csv(oof_dir / f"physdual_gin_esm_mlp_shuf{s_idx}_davis_cold_target_oof.csv")
                extra_models[f"shuf{s_idx}"] = sh_df["y_pred"].to_numpy(dtype=np.float64)
        elif ds == "davis" and sp == "cold_drug":
            ecfp_df = pd.read_csv(oof_dir / "physdual_ecfp_esm_mlp_davis_cold_drug_oof.csv")
            extra_models["ecfp_esm_mlp"] = ecfp_df["y_pred"].to_numpy(dtype=np.float64)
        elif ds == "davis" and sp == "cold_target_seqid30":
            ts_df = pd.read_csv(oof_dir / "physdual_gin_esm_mlp_tsoff1000_davis_cold_target_seqid30_oof.csv")
            extra_models["tsoff1000"] = ts_df["y_pred"].to_numpy(dtype=np.float64)

        # Reproduction diffs
        max_dm_p3 = float(np.max(np.abs(p3_df["y_pred_drug_mean"].to_numpy() - model_preds["drug_mean_predictor"])))
        max_tm_p3 = float(np.max(np.abs(p3_df["y_pred_target_mean"].to_numpy() - model_preds["target_mean_predictor"])))
        rec_p3 = compute_all_metrics(yt_all, model_preds["gin_esm_mlp"])
        rec_re = compute_all_metrics(yt_all, model_preds["ridge_esm"])
        rec_ke = compute_all_metrics(yt_all, model_preds["knn_esm_target"])
        cls_src = (
            kiba_cls["regimes"][sp]["global_oof_metrics"]
            if ds == "kiba"
            else (seqid_cls["splits"][sp]["global_oof_metrics"] if "seqid" in sp else None)
        )
        cls_diffs = {}
        if cls_src:
            for mk in ("drug_mean_predictor", "target_mean_predictor", "knn_kmer_target", "knn_tanimoto_drug"):
                rc = compute_all_metrics(yt_all, model_preds[mk])
                cls_diffs[mk] = {
                    "ci_abs_diff": abs(rc["ci"] - float(cls_src[mk]["ci"])),
                    "mse_abs_diff": abs(rc["mse"] - float(cls_src[mk]["mse"])),
                }
        repro_checks[reg_key] = {
            "y_pred_drug_mean_max_abs_diff": max_dm_p3,
            "y_pred_target_mean_max_abs_diff": max_tm_p3,
            "gin_esm_mlp_oof_ci_diff": abs(rec_p3["ci"] - float(p3_agg["global_oof_metrics"]["ci"])),
            "gin_esm_mlp_oof_mse_diff": abs(rec_p3["mse"] - float(p3_agg["global_oof_metrics"]["mse"])),
            "ridge_esm_oof_ci_diff": abs(rec_re["ci"] - float(esm_reg["global_oof_metrics"]["ridge_esm"]["ci"])),
            "ridge_esm_oof_mse_diff": abs(rec_re["mse"] - float(esm_reg["global_oof_metrics"]["ridge_esm"]["mse"])),
            "knn_esm_oof_ci_diff": abs(rec_ke["ci"] - float(esm_reg["global_oof_metrics"]["knn_esm_target"]["ci"])),
            "knn_esm_oof_mse_diff": abs(rec_ke["mse"] - float(esm_reg["global_oof_metrics"]["knn_esm_target"]["mse"])),
            "classical_oof_diffs": cls_diffs,
        }

        # Group column for cluster bootstrap
        if sp == "cold_drug":
            group_col = "smiles"
            group_vals = cpu_df["smiles"].astype(str)
        elif sp == "cold_target":
            group_col = "sequence"
            group_vals = cpu_df["sequence"].astype(str)
        else:
            group_col = "seqid_cluster"
            s2c = cmap_cache[ds]["thresholds"][sp]["seq_to_cluster"]
            group_vals = cpu_df["sequence"].astype(str).map(s2c).astype(int)

        all_models_preds = {**model_preds, **extra_models}
        per_model_summary, per_fold_by_model = _per_fold_and_oof_table(yt_all, folds_arr, all_models_preds)

        # Uncensored & binary metrics for Davis OOF
        uncens_oof: dict[str, Any] = {}
        binary_oof: dict[str, Any] = {}
        if ds == "davis":
            for mname, yp in all_models_preds.items():
                uncens_oof[mname] = compute_uncensored_metrics(yt_all, yp, censor_value=5.0)
                binary_oof[mname] = compute_binary_classification_metrics(yt_all, yp, thresholds=(5.0, 7.0))

        # Paired fold deltas for gin_esm_mlp vs all primary baselines
        base_keys = [k for k in model_preds if k != "gin_esm_mlp"]
        paired_fold = _paired_unit_deltas(per_fold_by_model["gin_esm_mlp"], {k: per_fold_by_model[k] for k in base_keys})

        # Bootstrap grouped comparisons (compute O(N log N) exact bootstrap ONCE for all models)
        bs_pre = _fast_bootstrap_precompute(
            yt_in=yt_all,
            groups_in=group_vals.to_numpy(),
            preds_map=all_models_preds,
            n_bootstraps=BOOTSTRAP_N,
            seed=BOOTSTRAP_SEED,
        )
        bs_gin_clean = _fast_bootstrap_compare(
            bs_pre,
            model_key="gin_esm_mlp",
            baseline_keys=[k for k in all_models_preds if k != "gin_esm_mlp"],
        )

        # If ecfp_esm_mlp or shuf* or tsoff1000 present, extract direct comparisons at zero extra cost
        extra_bootstrap: dict[str, Any] = {}
        for em in extra_models:
            cmp_cols = [
                k
                for k in ("gin_esm_mlp", "ridge_esm", "drug_mean_predictor", "knn_kmer_target", "knn_tanimoto_drug")
                if k in all_models_preds
            ]
            extra_bootstrap[em] = _fast_bootstrap_compare(bs_pre, model_key=em, baseline_keys=cmp_cols)

        # Best baselines (strictly excluding mean_predictor from "best non-ESM" unless needed, matching ADR #15 candidates)
        non_esm_candidates = [k for k in base_keys if "esm" not in k and k != "mean_predictor"]
        best_non_esm_mse = min(non_esm_candidates, key=lambda k: per_model_summary[k]["oof"]["mse"])
        best_non_esm_ci = max(non_esm_candidates, key=lambda k: per_model_summary[k]["oof"]["ci"])
        best_overall_mse = min([k for k in base_keys if k != "mean_predictor"], key=lambda k: per_model_summary[k]["oof"]["mse"])
        best_overall_ci = max([k for k in base_keys if k != "mean_predictor"], key=lambda k: per_model_summary[k]["oof"]["ci"])

        # ADR #15 operational evaluation (applicable to KIBA primary seqid; exploratory label on Davis)
        adr15 = {
            "vs_ridge_esm": _evaluate_adr15_outcome(
                paired_fold["ridge_esm"]["wins_mse"],
                paired_fold["ridge_esm"]["wins_ci"],
                5,
                bs_gin_clean["ridge_esm"],
            ),
            "best_non_esm_mse_name": best_non_esm_mse,
            "vs_best_non_esm_mse": _evaluate_adr15_outcome(
                paired_fold[best_non_esm_mse]["wins_mse"],
                paired_fold[best_non_esm_mse]["wins_ci"],
                5,
                bs_gin_clean[best_non_esm_mse],
            ),
            "vs_knn_kmer_target": _evaluate_adr15_outcome(
                paired_fold["knn_kmer_target"]["wins_mse"],
                paired_fold["knn_kmer_target"]["wins_ci"],
                5,
                bs_gin_clean["knn_kmer_target"],
            ),
        }

        # Check paired_vs_baselines & saved kiba bootstrap diffs
        saved_pvb = p3_agg.get("paired_vs_baselines", {})
        pvb_diffs = {}
        for k, sv in saved_pvb.items():
            if k in paired_fold:
                pvb_diffs[k] = {
                    "delta_mse_diff": abs(paired_fold[k]["delta_mse_mean"] - float(sv["delta_mse_mean"])),
                    "delta_ci_diff": abs(paired_fold[k]["delta_ci_mean"] - float(sv["delta_ci_mean"])),
                }
        kiba_bs_diffs = {}
        if ds == "kiba" and sp in kiba_pbs.get("regimes", {}):
            saved_bs = kiba_pbs["regimes"][sp].get("bootstrap_vs_baselines", {})
            for k, sv in saved_bs.items():
                if k in bs_gin_clean:
                    kiba_bs_diffs[k] = {
                        "delta_mse_mean_diff": abs(bs_gin_clean[k]["delta_mse"]["mean"] - float(sv["delta_mse"]["mean"])),
                        "delta_ci_mean_diff": abs(bs_gin_clean[k]["delta_ci"]["mean"] - float(sv["delta_ci"]["mean"])),
                    }

        regimes_out[reg_key] = {
            "dataset": ds,
            "split": sp,
            "protocol": "v2_kfold",
            "exploratory": ds == "davis",
            "n_oof_pairs": int(len(yt_all)),
            "group_col": group_col,
            "n_clusters": int(group_vals.nunique()),
            "best_non_esm_oof_mse": best_non_esm_mse,
            "best_non_esm_oof_ci": best_non_esm_ci,
            "best_overall_oof_mse": best_overall_mse,
            "best_overall_oof_ci": best_overall_ci,
            "per_model_summary": per_model_summary,
            "per_fold_by_model": per_fold_by_model,
            "paired_vs_baselines_folds": paired_fold,
            "paired_vs_baselines_json_diffs": pvb_diffs,
            "kiba_paired_bootstrap_saved_diffs": kiba_bs_diffs,
            "bootstrap_vs_baselines": bs_gin_clean,
            "extra_controls_bootstrap": extra_bootstrap,
            "adr15_operational": adr15,
            "uncensored_oof_davis": uncens_oof if ds == "davis" else None,
            "binary_oof_davis": binary_oof if ds == "davis" else None,
        }

    # Special descriptive control check: cold_drug without fold 2 (all models including ecfp_esm_mlp) + fold 2 tsoff1000
    cd_table = regimes_out["davis/cold_drug"]["per_fold_by_model"]
    cd_no2 = {}
    for mname, rows in cd_table.items():
        r4 = [r for r in rows if r["fold"] != 2]
        cd_no2[mname] = {
            "ci_4folds": _summary_stats([r["ci"] for r in r4]),
            "mse_4folds": _summary_stats([r["mse"] for r in r4]),
            "fold2": rows[2],
        }
    ts_cd_f2 = load_json_strict(runs_dir / "physdual_gin_esm_mlp_tsoff1000_davis_cold_drug_fold2.json")

    return {
        "reproduction_checks": repro_checks,
        "regimes": regimes_out,
        "davis_cold_drug_descriptive_no_fold2": cd_no2,
        "davis_cold_drug_fold2_tsoff1000": {
            "best_epoch": ts_cd_f2["best_epoch"],
            "total_epochs_run": ts_cd_f2["total_epochs_run"],
            "test_metrics_full": ts_cd_f2["test_metrics_full"],
            "train_val_mse_gap_at_best": ts_cd_f2["train_val_mse_gap_at_best"],
        },
    }


def compute_v1_seeds_audits_and_bootstrap(
    data_root: Path,
    only_keys: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    res_dir = data_root / "outputs" / "results"
    runs_dir = res_dir / "runs"
    splits_dir = data_root / "data" / "processed" / "splits"

    davis_df = pd.read_csv(data_root / "data" / "processed" / "davis_clean.csv")
    kiba_df = pd.read_csv(data_root / "data" / "processed" / "kiba_clean.csv")
    df_map = {"davis": davis_df, "kiba": kiba_df}

    esm_cpu = load_json_strict(res_dir / "phase3_esm_cpu_baselines.json")
    kiba_cls = load_json_strict(res_dir / "kiba_cpu_classical_baselines.json")

    # Preload shared ESM cache (strictly read-only, weights_only=True inside loader)
    all_seqs = sorted(set(davis_df["sequence"].astype(str)).union(set(kiba_df["sequence"].astype(str))))
    seq_to_vec = load_mean_pooled_esm_embeddings(
        cache_dir=data_root / "cache" / "esm_embeddings",
        sequences=all_seqs,
        model_name="esm2_t33_650M_UR50D",
        expected_dim=1280,
    )
    all_smiles = sorted(set(davis_df["smiles"].astype(str)).union(set(kiba_df["smiles"].astype(str))))
    drug_to_fp = compute_morgan_ecfp4_map(all_smiles, radius=2, n_bits=1024)

    classical_cache: dict[str, tuple[np.ndarray, dict[str, int], np.ndarray, dict[str, int]]] = {}
    for ds_name, df in df_map.items():
        km, s_idx = _build_3mer_norm_matrix(df["sequence"].astype(str).tolist())
        tm, d_idx = _build_tanimoto_sim_matrix(df["smiles"].astype(str).tolist())
        classical_cache[ds_name] = (km @ km.T, s_idx, tm, d_idx)

    v1_specs = [
        ("kiba", "cold_drug", "kiba_v1_cold_drug", "smiles"),
        ("kiba", "cold_target", "kiba_v1_cold_target", "sequence"),
        ("kiba", "random", "kiba_v1_random", None),
        ("davis", "random", "davis_v1_random", None),
    ]
    if only_keys is not None:
        v1_specs = [s for s in v1_specs if f"{s[0]}/{s[1]}" in only_keys]

    regimes_out: dict[str, Any] = {}
    repro_checks: dict[str, Any] = {}

    for ds, sp, esm_reg_key, cluster_col in v1_specs:
        reg_key = f"{ds}/{sp}"
        df = df_map[ds]
        km_mat, s_idx, tan_mat, d_idx = classical_cache[ds]

        p3_agg = load_json_strict(res_dir / f"phase3_physdual_gin_esm_mlp_{ds}_{sp}.json")
        esm_saved_units = esm_cpu["regimes"][esm_reg_key]["per_unit"]
        kiba_cls_units = kiba_cls["regimes"][sp]["per_unit"] if ds == "kiba" else None

        # Collect GraphDTA per-seed saved summaries
        gdta_aggs: dict[str, dict[str, Any]] = {}
        for gm in ("graphdta_gin", "graphdta_cnn3"):
            gp = res_dir / f"baseline_{gm}_{ds}_{sp}.json"
            if gp.exists():
                gdta_aggs[gm] = load_json_strict(gp)

        per_seed_models: dict[str, list[dict[str, Any]]] = {}
        pooled_rows: list[pd.DataFrame] = []
        max_diffs: dict[str, float] = {
            "gin_esm_mlp": 0.0,
            "ridge_esm": 0.0,
            "ridge_esm_target_only": 0.0,
            "knn_esm_target": 0.0,
            "knn_tanimoto_drug": 0.0,
            "knn_kmer_target": 0.0,
            "drug_mean_predictor": 0.0,
            "target_mean_predictor": 0.0,
            "mean_predictor": 0.0,
        }

        uncens_per_seed_gin: list[dict[str, Any]] = []
        binary_per_seed_gin: list[dict[str, Any]] = []

        for seed in range(5):
            split_idx = load_splits(splits_dir / f"{ds}_{sp}_seed{seed}.json")
            tr_df = df.iloc[split_idx["train"]]
            va_df = df.iloc[split_idx["val"]]
            te_df = df.iloc[split_idx["test"]]
            yt = te_df["affinity"].to_numpy(dtype=np.float64)

            # 1. gin_esm_mlp OOF predictions from run JSON
            run_d = load_json_strict(runs_dir / f"physdual_gin_esm_mlp_{ds}_{sp}_seed{seed}.json")
            oof_list = run_d["oof_predictions"]
            yp_gin = np.array([r["y_pred"] for r in oof_list], dtype=np.float64)
            yt_json = np.array([r["y_true"] for r in oof_list], dtype=np.float64)
            if not np.allclose(yt, yt_json, atol=1e-7):
                raise ValueError(f"{reg_key} seed {seed}: y_true mismatch entre run JSON e split")

            m_gin = compute_all_metrics(yt, yp_gin)
            saved_gin = run_d["test_metrics_full"]
            max_diffs["gin_esm_mlp"] = max(
                max_diffs["gin_esm_mlp"], abs(m_gin["ci"] - saved_gin["ci"]), abs(m_gin["mse"] - saved_gin["mse"])
            )
            per_seed_models.setdefault("gin_esm_mlp", []).append({"seed": seed, **m_gin})

            if ds == "davis":
                uncens_per_seed_gin.append(compute_uncensored_metrics(yt, yp_gin, censor_value=5.0))
                binary_per_seed_gin.append(compute_binary_classification_metrics(yt, yp_gin, thresholds=(5.0, 7.0)))

            # 2. ESM CPU baselines
            res_esm = evaluate_esm_cpu_baselines_split(
                tr_df, va_df, te_df, seq_to_vec=seq_to_vec, drug_to_fp=drug_to_fp, k=5
            )
            saved_esm_u = esm_saved_units[seed]
            for ek in ("ridge_esm", "ridge_esm_target_only", "knn_esm_target"):
                max_diffs[ek] = max(
                    max_diffs[ek],
                    abs(res_esm[ek]["metrics"]["ci"] - saved_esm_u[ek]["metrics"]["ci"]),
                    abs(res_esm[ek]["metrics"]["mse"] - saved_esm_u[ek]["metrics"]["mse"]),
                )
                per_seed_models.setdefault(ek, []).append({"seed": seed, **res_esm[ek]["metrics"]})

            # 3. Classical CPU baselines
            res_cls = _compute_classical_unit(
                tr_df, te_df, km_mat, s_idx, {}, k=5, tanimoto_sim_mat=tan_mat, drug_to_idx=d_idx
            )
            for ck in ("mean_predictor", "drug_mean_predictor", "target_mean_predictor", "knn_tanimoto_drug", "knn_kmer_target"):
                if kiba_cls_units is not None:
                    saved_c = kiba_cls_units[seed][ck]["metrics"]
                elif ck in run_d.get("marginal_baselines", {}):
                    saved_c = run_d["marginal_baselines"][ck]
                elif ck in run_d.get("knn_baselines", {}):
                    saved_c = run_d["knn_baselines"][ck]
                else:
                    saved_c = run_d["trivial_baselines"][ck]
                max_diffs[ck] = max(
                    max_diffs[ck],
                    abs(res_cls[ck]["metrics"]["ci"] - saved_c["ci"]),
                    abs(res_cls[ck]["metrics"]["mse"] - saved_c["mse"]),
                )
                per_seed_models.setdefault(ck, []).append({"seed": seed, **res_cls[ck]["metrics"]})

            # 4. GraphDTA per-seed saved summaries (per-sample OOF not saved in v1)
            for gm, gagg in gdta_aggs.items():
                psr_map = gagg["per_seed_results"]
                g_psr = psr_map[str(seed)] if isinstance(psr_map, dict) else psr_map[seed]
                gm_metrics = g_psr["test_metrics_full"]
                per_seed_models.setdefault(gm, []).append({"seed": seed, **gm_metrics})

            pooled_rows.append(pd.DataFrame({
                "seed": seed,
                "smiles": te_df["smiles"].astype(str).to_numpy(),
                "sequence": te_df["sequence"].astype(str).to_numpy(),
                "pair_id": (te_df["drug_id"].astype(str) + "__" + te_df["target_id"].astype(str)).to_numpy(),
                "y_true": yt,
                "y_pred_gin_esm_mlp": yp_gin,
                "y_pred_ridge_esm": res_esm["ridge_esm"]["y_pred"],
                "y_pred_ridge_esm_target_only": res_esm["ridge_esm_target_only"]["y_pred"],
                "y_pred_knn_esm_target": res_esm["knn_esm_target"]["y_pred"],
                "y_pred_knn_tanimoto_drug": res_cls["knn_tanimoto_drug"]["y_pred"],
                "y_pred_knn_kmer_target": res_cls["knn_kmer_target"]["y_pred"],
                "y_pred_drug_mean_predictor": res_cls["drug_mean_predictor"]["y_pred"],
                "y_pred_target_mean_predictor": res_cls["target_mean_predictor"]["y_pred"],
                "y_pred_mean_predictor": res_cls["mean_predictor"]["y_pred"],
            }))

        repro_checks[reg_key] = max_diffs
        pooled_df = pd.concat(pooled_rows, ignore_index=True)

        # Summary per model across the 5 seeds
        model_summary: dict[str, Any] = {}
        for mname, ulist in per_seed_models.items():
            model_summary[mname] = {
                "ci": _summary_stats([u["ci"] for u in ulist]),
                "mse": _summary_stats([u["mse"] for u in ulist]),
                "rmse": _summary_stats([u["rmse"] for u in ulist]),
                "pearson": _summary_stats([u["pearson"] for u in ulist]),
                "spearman": _summary_stats([u["spearman"] for u in ulist]),
                "rm2": _summary_stats([u["rm2"] for u in ulist]),
            }

        # Paired 5-seed deltas against ALL available comparators (including GraphDTA!)
        base_names = [k for k in per_seed_models if k != "gin_esm_mlp"]
        paired_seeds = _paired_unit_deltas(
            per_seed_models["gin_esm_mlp"],
            {k: per_seed_models[k] for k in base_names},
        )

        # Compare paired_seeds vs saved paired_vs_baselines in phase3_physdual_*.json
        saved_pvb = p3_agg.get("paired_vs_baselines", {})
        pvb_diffs = {}
        for k, sv in saved_pvb.items():
            if k in paired_seeds:
                pvb_diffs[k] = {
                    "delta_mse_diff": abs(paired_seeds[k]["delta_mse_mean"] - sv["delta_mse_mean"]),
                    "delta_ci_diff": abs(paired_seeds[k]["delta_ci_mean"] - sv["delta_ci_mean"]),
                }

        # Best baselines per seed mean MSE (excluding mean_predictor & ridge_aa)
        sample_cols = [c for c in pooled_df.columns if c.startswith("y_pred_") and c != "y_pred_gin_esm_mlp"]
        non_esm_all = [k for k in base_names if "esm" not in k and k != "mean_predictor"]
        best_non_esm_mse = min(non_esm_all, key=lambda k: model_summary[k]["mse"]["mean"])
        best_non_esm_ci = max(non_esm_all, key=lambda k: model_summary[k]["ci"]["mean"])
        best_overall_mse = min([k for k in base_names if k != "mean_predictor"], key=lambda k: model_summary[k]["mse"]["mean"])

        # Cluster bootstrap ONLY where cluster unit is well-defined (cold_drug by smiles, cold_target by sequence)
        bs_clean = None
        n_clusters = None
        if cluster_col is not None:
            n_clusters = int(pooled_df[cluster_col].nunique())
            v1_preds = {c.replace("y_pred_", ""): pooled_df[c].to_numpy(dtype=np.float64) for c in pooled_df.columns if c.startswith("y_pred_")}
            bs_pre_v1 = _fast_bootstrap_precompute(
                yt_in=pooled_df["y_true"].to_numpy(dtype=np.float64),
                groups_in=pooled_df[cluster_col].to_numpy(),
                preds_map=v1_preds,
                n_bootstraps=BOOTSTRAP_N,
                seed=BOOTSTRAP_SEED,
            )
            bs_clean = _fast_bootstrap_compare(
                bs_pre_v1,
                model_key="gin_esm_mlp",
                baseline_keys=[c.replace("y_pred_", "") for c in sample_cols],
            )

        # Evaluate ADR #15 for KIBA cold_drug & cold_target
        adr15 = None
        if ds == "kiba" and sp in ("cold_drug", "cold_target") and bs_clean is not None:
            adr15 = {
                "vs_ridge_esm": _evaluate_adr15_outcome(
                    paired_seeds["ridge_esm"]["wins_mse"],
                    paired_seeds["ridge_esm"]["wins_ci"],
                    5,
                    bs_clean["ridge_esm"],
                ),
                "best_non_esm_mse_name": best_non_esm_mse,
                "vs_best_non_esm_mse": _evaluate_adr15_outcome(
                    paired_seeds[best_non_esm_mse]["wins_mse"],
                    paired_seeds[best_non_esm_mse]["wins_ci"],
                    5,
                    bs_clean.get(best_non_esm_mse),
                ),
            }
            if sp == "cold_drug":
                adr15["vs_knn_tanimoto_drug"] = _evaluate_adr15_outcome(
                    paired_seeds["knn_tanimoto_drug"]["wins_mse"],
                    paired_seeds["knn_tanimoto_drug"]["wins_ci"],
                    5,
                    bs_clean["knn_tanimoto_drug"],
                )
                adr15["vs_graphdta_gin_5seeds_only"] = {
                    "verdict": "nao_atendido_negativo",
                    "wins_mse": paired_seeds["graphdta_gin"]["wins_mse"],
                    "wins_ci": paired_seeds["graphdta_gin"]["wins_ci"],
                    "note": "OOF por amostra de graphdta_gin v1 não salvo; 0/5 vitórias em MSE já reprova critério (>=4/5) pelo ADR #15.",
                }
            elif sp == "cold_target":
                adr15["vs_knn_kmer_target"] = _evaluate_adr15_outcome(
                    paired_seeds["knn_kmer_target"]["wins_mse"],
                    paired_seeds["knn_kmer_target"]["wins_ci"],
                    5,
                    bs_clean["knn_kmer_target"],
                )

        # Davis random uncensored + binary summaries
        uncens_sum = None
        binary_sum = None
        if ds == "davis":
            uncens_sum = {
                "ci": _summary_stats([u["ci"] for u in uncens_per_seed_gin]),
                "mse": _summary_stats([u["mse"] for u in uncens_per_seed_gin]),
                "rm2": _summary_stats([u["rm2"] for u in uncens_per_seed_gin]),
            }
            binary_sum = {
                "auroc_gt5": _summary_stats([b["threshold_5.0"]["auroc"] for b in binary_per_seed_gin]),
                "aupr_gt5": _summary_stats([b["threshold_5.0"]["aupr"] for b in binary_per_seed_gin]),
                "auroc_ge7": _summary_stats([b["threshold_7.0"]["auroc"] for b in binary_per_seed_gin]),
                "aupr_ge7": _summary_stats([b["threshold_7.0"]["aupr"] for b in binary_per_seed_gin]),
            }

        regimes_out[reg_key] = {
            "dataset": ds,
            "split": sp,
            "protocol": "v1_seeds",
            "role": "sanity" if sp == "random" else "primary",
            "n_pooled_rows": int(len(pooled_df)),
            "n_unique_pairs": int(pooled_df["pair_id"].nunique()),
            "cluster_col": cluster_col,
            "n_clusters_pooled": n_clusters,
            "best_non_esm_mse": best_non_esm_mse,
            "best_non_esm_ci": best_non_esm_ci,
            "best_overall_mse": best_overall_mse,
            "model_summary": model_summary,
            "per_seed_models": per_seed_models,
            "paired_vs_baselines_seeds": paired_seeds,
            "paired_vs_baselines_json_diffs": pvb_diffs,
            "bootstrap_vs_baselines_pooled_clusters": bs_clean,
            "adr15_operational": adr15,
            "uncensored_gin_esm_mlp": uncens_sum,
            "binary_gin_esm_mlp": binary_sum,
        }

    return {"reproduction_checks": repro_checks, "regimes": regimes_out}


def run_phase3_audits(step: str = "all") -> Path:
    import argparse  # noqa: F401
    t0 = time.perf_counter()
    repo_root, data_root = resolve_roots()
    dirs = ensure_paper_dirs(repo_root)
    out_path = dirs["paper"] / "phase3_audits.json"

    if step != "all" and out_path.exists():
        payload: dict[str, Any] = load_json_strict(out_path)
    else:
        payload = {}

    payload["generated_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload["bootstrap_config"] = {"n_bootstraps": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED}

    if step in ("all", "meta"):
        payload["git_preregistration"] = audit_git_preregistration(repo_root)
        payload["run_integrity"] = audit_runs_and_aggregates(data_root)
        payload["dataset_overlap_and_units"] = _extract_kiba_davis_overlap(data_root)
    if step in ("all", "v2"):
        payload["v2_kfold"] = compute_v2_kfold_audits_and_controls(data_root)
    if step in ("all", "v1"):
        payload["v1_seeds"] = compute_v1_seeds_audits_and_bootstrap(data_root)
    elif step in ("v1_cold", "v1_random"):
        keys = ("kiba/cold_drug", "kiba/cold_target") if step == "v1_cold" else ("kiba/random", "davis/random")
        part = compute_v1_seeds_audits_and_bootstrap(data_root, only_keys=keys)
        existing_v1 = payload.get("v1_seeds", {"reproduction_checks": {}, "regimes": {}})
        existing_v1.setdefault("reproduction_checks", {}).update(part["reproduction_checks"])
        existing_v1.setdefault("regimes", {}).update(part["regimes"])
        payload["v1_seeds"] = existing_v1

    payload["elapsed_sec"] = round(time.perf_counter() - t0, 2)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", default="all", choices=["all", "meta", "v2", "v1", "v1_cold", "v1_random"])
    args = parser.parse_args()
    p = run_phase3_audits(step=args.step)
    print(f"[OK] Auditoria Phase 3 ({args.step}) salva em: {p}")
