"""Curva contínua identidade-margem estilo MIRAGE e faixas intragrupo em CPU.

Executa estritamente em modo somente leitura sobre `data/`, `outputs/` e (quando
completo localmente) `cache/esm_embeddings/`.

Calcula para Davis (`v2_kfold`: `cold_target`, `cold_target_seqid40`,
`cold_target_seqid30`) e KIBA (`v1_seeds`: `cold_target`; `v2_kfold`:
`cold_target_seqid40`, `cold_target_seqid30`):
1. Matrizes de identidade global (`global_id_max`) e local Smith-Waterman
   (`sw_id_aln_len_ge150aa`) BLOSUM62 (`open_gap=-10.0`, `extend_gap=-0.5`) via
   `ProcessPoolExecutor`.
2. Métricas por proteína de teste `(u, p)`: `mse` e margens pareadas
   `margin_vs_drug_mean = MSE(drug_mean) - MSE(modelo)` e `margin_vs_ridge_esm`.
3. Tendência contínua estilo MIRAGE (Spearman `rho` e inclinação OLS por +10%
   de identidade) com IC 95% via cluster bootstrap agrupado por sequência
   proteica única (`B = 2000`, `seed = 42`) e curva suavizada em 5 quantis.
4. Faixas fixas interpretáveis (`<0.30`, `[0.30, 0.40)`, `[0.40, 0.55)`, `>=0.55`)
   calculando `MSE`, margens, `intra_CI` e `intra_Spearman` por fármaco (`smiles`,
   `min_pairs=2`, `>= 2` valores distintos de `y_true`) **estritamente dentro de
   cada unidade `u` e faixa `b`**, sem concatenação entre dobras/seeds.

Saída padrão: `docs/paper/continuous_identity_margin.json`
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

PAPER_SCRIPTS_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = PAPER_SCRIPTS_DIR.parent
REPO_ROOT = PAPER_SCRIPTS_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
for p in (str(PAPER_SCRIPTS_DIR), str(SCRIPTS_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common import ensure_paper_dirs, load_json_strict, resolve_roots, sample_mean_std  # noqa: E402
from phase3_intragroup_and_stacking import (  # noqa: E402
    _intragroup_from_prepared,
    _prepare_group_indices,
)
from physdual.data.esm_cache import get_embedding_cache_path  # noqa: E402
from physdual.data.splits import load_splits  # noqa: E402
from physdual.eval.esm_baselines import (  # noqa: E402
    compute_morgan_ecfp4_map,
    evaluate_esm_cpu_baselines_split,
    load_mean_pooled_esm_embeddings,
)
from physdual.eval.metrics import compute_mse  # noqa: E402
from run_phase3_esm_cpu_baselines import (  # noqa: E402
    _build_3mer_norm_matrix,
    _compute_classical_unit,
)

IDENTITY_AXES: tuple[str, ...] = ("global_id_max", "sw_id_aln_len_ge150aa")

BAND_DEFINITIONS: tuple[tuple[str, float | None, float | None], ...] = (
    ("<0.30", None, 0.30),
    ("[0.30, 0.40)", 0.30, 0.40),
    ("[0.40, 0.55)", 0.40, 0.55),
    (">=0.55", 0.55, None),
)

CANONICAL_MODELS: tuple[str, ...] = (
    "gin_esm_mlp",
    "graphdta_gin",
    "ridge_esm",
    "knn_kmer_target",
    "drug_mean_predictor",
)

REGIME_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("davis", "cold_target", "v2_kfold", "fold"),
    ("davis", "cold_target_seqid40", "v2_kfold", "fold"),
    ("davis", "cold_target_seqid30", "v2_kfold", "fold"),
    ("kiba", "cold_target", "v1_seeds", "seed"),
    ("kiba", "cold_target_seqid40", "v2_kfold", "fold"),
    ("kiba", "cold_target_seqid30", "v2_kfold", "fold"),
)

_WORKER_SEQS: list[str] = []
_GL_ALIGNER: Any = None
_SW_ALIGNER: Any = None


def _init_worker(seqs: list[str]) -> None:
    global _WORKER_SEQS, _GL_ALIGNER, _SW_ALIGNER
    from Bio.Align import PairwiseAligner, substitution_matrices

    _WORKER_SEQS = seqs
    blosum62 = substitution_matrices.load("BLOSUM62")
    _GL_ALIGNER = PairwiseAligner(
        mode="global",
        substitution_matrix=blosum62,
        open_gap_score=-10.0,
        extend_gap_score=-0.5,
    )
    _SW_ALIGNER = PairwiseAligner(
        mode="local",
        substitution_matrix=blosum62,
        open_gap_score=-10.0,
        extend_gap_score=-0.5,
    )


def _align_pair_worker(pair: tuple[int, int]) -> tuple[int, int, float, float, int]:
    i, j = pair
    si = _WORKER_SEQS[i]
    sj = _WORKER_SEQS[j]
    li = len(si)
    lj = len(sj)
    max_l = li if li >= lj else lj

    aln_gl = next(iter(_GL_ALIGNER.align(si, sj)))
    gl_ident = int(aln_gl.counts().identities)
    gl_id_max = gl_ident / float(max_l) if max_l > 0 else 0.0

    aln_sw = next(iter(_SW_ALIGNER.align(si, sj)))
    sw_ident = int(aln_sw.counts().identities)
    sw_len = int(aln_sw.length)
    sw_id_aln = sw_ident / float(sw_len) if sw_len > 0 else 0.0
    return i, j, gl_id_max, sw_id_aln, sw_len


def compute_identity_matrices(seqs: list[str], n_workers: int) -> dict[str, np.ndarray]:
    """Calcula matrizes simétricas `global_id_max`, `sw_id_aln_len` e `sw_aln_len`."""
    n = len(seqs)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    chunksize = max(64, min(1024, len(pairs) // max(1, n_workers * 4)))

    if n_workers <= 1:
        _init_worker(seqs)
        results = [_align_pair_worker(p) for p in pairs]
    else:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(seqs,),
        ) as pool:
            results = list(pool.map(_align_pair_worker, pairs, chunksize=chunksize))

    gl_max = np.eye(n, dtype=np.float32)
    sw_aln = np.eye(n, dtype=np.float32)
    sw_len = np.zeros((n, n), dtype=np.int32)
    for k in range(n):
        sw_len[k, k] = len(seqs[k])

    for i, j, gmax, saln, slen in results:
        gl_max[i, j] = gl_max[j, i] = gmax
        sw_aln[i, j] = sw_aln[j, i] = saln
        sw_len[i, j] = sw_len[j, i] = slen

    return {
        "global_id_max": gl_max,
        "sw_id_aln_len": sw_aln,
        "sw_aln_len": sw_len,
    }


def _safe_spearman_and_ols_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Calcula Spearman rho e inclinação OLS (por +1.0 de x) de forma finita."""
    if x.size < 2 or float(np.ptp(x)) <= 1e-12 or float(np.ptp(y)) <= 1e-12:
        return 0.0, 0.0
    xc = x - float(np.mean(x))
    yc = y - float(np.mean(y))
    denom_x = float(np.dot(xc, xc))
    slope = float(np.dot(xc, yc) / denom_x) if denom_x > 1e-15 else 0.0
    if not np.isfinite(slope):
        slope = 0.0

    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rxc = rx - float(np.mean(rx))
    ryc = ry - float(np.mean(ry))
    denom_r = float(np.sqrt(np.dot(rxc, rxc) * np.dot(ryc, ryc)))
    rho = float(np.dot(rxc, ryc) / denom_r) if denom_r > 1e-15 else 0.0
    if not np.isfinite(rho):
        rho = 0.0
    return rho, slope


def _cluster_bootstrap_trend(
    seqs: list[str],
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """Cluster bootstrap agrupado por sequência proteica única para Spearman rho e slope."""
    rho_pt, slope_pt = _safe_spearman_and_ols_slope(x_vals, y_vals)
    slope10_pt = float(slope_pt * 0.10)

    cluster_map: dict[str, list[int]] = {}
    for idx, seq in enumerate(seqs):
        cluster_map.setdefault(seq, []).append(idx)
    uniq_seqs = sorted(cluster_map.keys())
    n_clusters = len(uniq_seqs)

    cluster_x = [x_vals[np.asarray(cluster_map[s], dtype=np.intp)] for s in uniq_seqs]
    cluster_y = [y_vals[np.asarray(cluster_map[s], dtype=np.intp)] for s in uniq_seqs]

    rng = np.random.default_rng(seed)
    boot_rhos = np.empty(n_bootstrap, dtype=np.float64)
    boot_slopes10 = np.empty(n_bootstrap, dtype=np.float64)

    for b in range(n_bootstrap):
        chosen = rng.integers(0, n_clusters, size=n_clusters)
        xb = np.concatenate([cluster_x[int(c)] for c in chosen])
        yb = np.concatenate([cluster_y[int(c)] for c in chosen])
        rho_b, slope_b = _safe_spearman_and_ols_slope(xb, yb)
        boot_rhos[b] = rho_b
        boot_slopes10[b] = slope_b * 0.10

    rho_lo, rho_hi = np.percentile(boot_rhos, [2.5, 97.5])
    sl_lo, sl_hi = np.percentile(boot_slopes10, [2.5, 97.5])
    return {
        "n_observations": int(len(x_vals)),
        "n_unique_proteins": int(n_clusters),
        "n_bootstrap": int(n_bootstrap),
        "spearman_rho": float(rho_pt),
        "spearman_ci95": [float(rho_lo), float(rho_hi)],
        "spearman_count_gt_zero": int(np.sum(boot_rhos > 0.0)),
        "spearman_frac_gt_zero": float(np.mean(boot_rhos > 0.0)),
        "slope_per_10pct_id": float(slope10_pt),
        "slope_per_10pct_ci95": [float(sl_lo), float(sl_hi)],
        "slope_count_gt_zero": int(np.sum(boot_slopes10 > 0.0)),
        "slope_frac_gt_zero": float(np.mean(boot_slopes10 > 0.0)),
    }


def _cluster_bootstrap_mean(
    seqs: list[str],
    vals: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, list[float]]:
    """Média pontual e IC 95% via cluster bootstrap por sequência proteica única."""
    mean_pt = float(np.mean(vals))
    cluster_map: dict[str, list[int]] = {}
    for idx, seq in enumerate(seqs):
        cluster_map.setdefault(seq, []).append(idx)
    uniq_seqs = sorted(cluster_map.keys())
    n_clusters = len(uniq_seqs)
    if n_clusters <= 1:
        return mean_pt, [mean_pt, mean_pt]

    cluster_vals = [vals[np.asarray(cluster_map[s], dtype=np.intp)] for s in uniq_seqs]
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        chosen = rng.integers(0, n_clusters, size=n_clusters)
        yb = np.concatenate([cluster_vals[int(c)] for c in chosen])
        boot_means[b] = float(np.mean(yb))
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return mean_pt, [float(lo), float(hi)]


def _in_band(val: float, low: float | None, high: float | None) -> bool:
    if low is not None and val < low:
        return False
    if high is not None and val >= high:
        return False
    return True


def _load_v2_kfold_units(
    data_root: Path,
    ds: str,
    sp: str,
    clean_df: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[str], list[str], dict[str, str]]:
    oof_dir = data_root / "outputs" / "results" / "oof"
    splits_dir = data_root / "data" / "processed" / "splits"

    cpu_df = pd.read_csv(oof_dir / f"cpu_baselines_{ds}_{sp}_oof.csv")
    gin_df = pd.read_csv(oof_dir / f"physdual_gin_esm_mlp_{ds}_{sp}_oof.csv")

    if len(cpu_df) != len(gin_df):
        raise ValueError(f"{ds}/{sp}: OOF len mismatch cpu={len(cpu_df)} gin={len(gin_df)}")
    if not (
        np.array_equal(cpu_df["fold"].to_numpy(), gin_df["fold"].to_numpy())
        and (cpu_df["drug_id"].astype(str) == gin_df["drug_id"].astype(str)).all()
        and (cpu_df["target_id"].astype(str) == gin_df["target_id"].astype(str)).all()
        and np.allclose(cpu_df["y_true"].to_numpy(), gin_df["y_true"].to_numpy(), atol=1e-5)
    ):
        raise ValueError(f"{ds}/{sp}: OOF desalinhado entre cpu_baselines e physdual_gin_esm_mlp")

    preds_all: dict[str, np.ndarray] = {
        "gin_esm_mlp": gin_df["y_pred"].to_numpy(dtype=np.float64),
        "ridge_esm": cpu_df["y_pred_ridge_esm"].to_numpy(dtype=np.float64),
        "knn_kmer_target": cpu_df["y_pred_knn_kmer_target"].to_numpy(dtype=np.float64),
        "drug_mean_predictor": cpu_df["y_pred_drug_mean_predictor"].to_numpy(dtype=np.float64),
    }
    sources = {
        "gin_esm_mlp": f"outputs/results/oof/physdual_gin_esm_mlp_{ds}_{sp}_oof.csv",
        "ridge_esm": f"outputs/results/oof/cpu_baselines_{ds}_{sp}_oof.csv (y_pred_ridge_esm)",
        "knn_kmer_target": f"outputs/results/oof/cpu_baselines_{ds}_{sp}_oof.csv (y_pred_knn_kmer_target)",
        "drug_mean_predictor": f"outputs/results/oof/cpu_baselines_{ds}_{sp}_oof.csv (y_pred_drug_mean_predictor)",
    }
    unavailable: list[str] = []

    gp = oof_dir / f"graphdta_gin_{ds}_{sp}_oof.csv"
    if gp.exists():
        gdf = pd.read_csv(gp)
        if not (
            len(gdf) == len(cpu_df)
            and np.array_equal(cpu_df["fold"].to_numpy(), gdf["fold"].to_numpy())
            and (cpu_df["drug_id"].astype(str) == gdf["drug_id"].astype(str)).all()
            and (cpu_df["target_id"].astype(str) == gdf["target_id"].astype(str)).all()
            and np.allclose(cpu_df["y_true"].to_numpy(), gdf["y_true"].to_numpy(), atol=1e-5)
        ):
            raise ValueError(f"{ds}/{sp}: OOF desalinhado em graphdta_gin")
        preds_all["graphdta_gin"] = gdf["y_pred"].to_numpy(dtype=np.float64)
        sources["graphdta_gin"] = f"outputs/results/oof/graphdta_gin_{ds}_{sp}_oof.csv"
    else:
        unavailable.append("graphdta_gin")
        sources["graphdta_gin"] = f"não disponível (OOF CSV ausente para {ds}/{sp})"

    available = [m for m in CANONICAL_MODELS if m in preds_all]
    folds_col = cpu_df["fold"].to_numpy(dtype=np.int64)
    smiles_col = cpu_df["smiles"].astype(str).to_numpy()
    seqs_col = cpu_df["sequence"].astype(str).to_numpy()
    yt_col = cpu_df["y_true"].to_numpy(dtype=np.float64)

    units: list[dict[str, Any]] = []
    for u in range(5):
        sp_idx = load_splits(splits_dir / f"{ds}_{sp}_fold{u}.json")
        tr_seqs = sorted(set(clean_df.iloc[sp_idx["train"]]["sequence"].astype(str)))
        mask = folds_col == u
        te_seqs = sorted(set(seqs_col[mask]))
        unit_preds = {m: preds_all[m][mask] for m in available}
        units.append({
            "unit": u,
            "train_seqs": tr_seqs,
            "test_seqs": te_seqs,
            "smiles": smiles_col[mask],
            "sequence": seqs_col[mask],
            "y_true": yt_col[mask],
            "preds": unit_preds,
        })
    return units, available, unavailable, sources


def _load_kiba_cold_target_v1_units(
    data_root: Path,
    kiba_df: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[str], list[str], dict[str, str], dict[str, float]]:
    res_dir = data_root / "outputs" / "results"
    runs_dir = res_dir / "runs"
    splits_dir = data_root / "data" / "processed" / "splits"

    km_norm, s_idx_k = _build_3mer_norm_matrix(kiba_df["sequence"].astype(str).tolist())
    km_sim_k = km_norm @ km_norm.T

    kiba_cls = load_json_strict(res_dir / "kiba_cpu_classical_baselines.json")
    cls_units = kiba_cls["regimes"]["cold_target"]["per_unit"]

    kiba_seqs = sorted(set(kiba_df["sequence"].astype(str)))
    esm_cache_dir = data_root / "cache" / "esm_embeddings"
    all_pts_present = all(
        get_embedding_cache_path(s, model_name="esm2_t33_650M_UR50D", cache_dir=esm_cache_dir).exists()
        for s in kiba_seqs
    )

    seq_to_vec: dict[str, np.ndarray] | None = None
    drug_to_fp: dict[str, np.ndarray] | None = None
    esm_units: list[dict[str, Any]] | None = None
    if all_pts_present:
        seq_to_vec = load_mean_pooled_esm_embeddings(
            cache_dir=esm_cache_dir,
            sequences=kiba_seqs,
            model_name="esm2_t33_650M_UR50D",
            expected_dim=1280,
        )
        all_smiles = sorted(set(kiba_df["smiles"].astype(str)))
        drug_to_fp = compute_morgan_ecfp4_map(all_smiles, radius=2, n_bits=1024)
        esm_cpu = load_json_strict(res_dir / "phase3_esm_cpu_baselines.json")
        esm_units = esm_cpu["regimes"]["kiba_v1_cold_target"]["per_unit"]

    available = ["gin_esm_mlp", "knn_kmer_target", "drug_mean_predictor"]
    unavailable = ["graphdta_gin"]
    sources: dict[str, str] = {
        "gin_esm_mlp": "outputs/results/runs/physdual_gin_esm_mlp_kiba_cold_target_seed{0..4}.json (oof_predictions)",
        "graphdta_gin": "não disponível (runs v1_seeds de graphdta_gin não salvaram oof_predictions)",
        "knn_kmer_target": "reconstrução analítica determinística em CPU via kiba_clean.csv + splits (3-mer)",
        "drug_mean_predictor": "oof_predictions.y_pred_drug_mean (verificado idêntico à média por fármaco do treino)",
    }
    if seq_to_vec is not None:
        available.insert(1, "ridge_esm")
        sources["ridge_esm"] = (
            "reconstrução determinística em CPU via cache local esm2_t33_650M_UR50D (429/429 .pt) + ECFP4 + splits"
        )
    else:
        unavailable.append("ridge_esm")
        sources["ridge_esm"] = "não disponível localmente (cache .pt ESM-2 ausente)"

    max_repro_diff: dict[str, float] = {
        "drug_mean_vs_saved_oof": 0.0,
        "knn_kmer_mse_vs_saved": 0.0,
    }
    if seq_to_vec is not None:
        max_repro_diff["ridge_esm_mse_vs_saved"] = 0.0

    units: list[dict[str, Any]] = []
    for seed in range(5):
        sp_idx = load_splits(splits_dir / f"kiba_cold_target_seed{seed}.json")
        tr_df = kiba_df.iloc[sp_idx["train"]]
        va_df = kiba_df.iloc[sp_idx["val"]]
        te_df = kiba_df.iloc[sp_idx["test"]]

        yt = te_df["affinity"].to_numpy(dtype=np.float64)
        te_smiles = te_df["smiles"].astype(str).to_numpy()
        te_seqs = te_df["sequence"].astype(str).to_numpy()

        run_d = load_json_strict(runs_dir / f"physdual_gin_esm_mlp_kiba_cold_target_seed{seed}.json")
        oof_list = run_d.get("oof_predictions") or []
        if len(oof_list) != len(te_df):
            raise ValueError(f"kiba/cold_target seed={seed}: len(oof_predictions)={len(oof_list)} != {len(te_df)}")

        yp_gin = np.asarray([r["y_pred"] for r in oof_list], dtype=np.float64)
        yp_dm_oof = np.asarray([r["y_pred_drug_mean"] for r in oof_list], dtype=np.float64)
        yt_oof = np.asarray([r["y_true"] for r in oof_list], dtype=np.float64)
        if not np.allclose(yt, yt_oof, atol=1e-5):
            raise ValueError(f"kiba/cold_target seed={seed}: y_true mismatch entre JSON e split")

        res_cls = _compute_classical_unit(
            tr_df, te_df, km_sim_k, s_idx_k, {}, k=5, tanimoto_sim_mat=None, drug_to_idx=None
        )
        max_repro_diff["drug_mean_vs_saved_oof"] = max(
            max_repro_diff["drug_mean_vs_saved_oof"],
            float(np.max(np.abs(yp_dm_oof - res_cls["drug_mean_predictor"]["y_pred"]))),
        )
        max_repro_diff["knn_kmer_mse_vs_saved"] = max(
            max_repro_diff["knn_kmer_mse_vs_saved"],
            abs(res_cls["knn_kmer_target"]["metrics"]["mse"] - float(cls_units[seed]["knn_kmer_target"]["metrics"]["mse"])),
        )

        unit_preds: dict[str, np.ndarray] = {
            "gin_esm_mlp": yp_gin,
            "knn_kmer_target": res_cls["knn_kmer_target"]["y_pred"],
            "drug_mean_predictor": res_cls["drug_mean_predictor"]["y_pred"],
        }

        if seq_to_vec is not None and drug_to_fp is not None and esm_units is not None:
            res_esm = evaluate_esm_cpu_baselines_split(
                tr_df, va_df, te_df, seq_to_vec=seq_to_vec, drug_to_fp=drug_to_fp, k=5
            )
            max_repro_diff["ridge_esm_mse_vs_saved"] = max(
                max_repro_diff["ridge_esm_mse_vs_saved"],
                abs(res_esm["ridge_esm"]["metrics"]["mse"] - float(esm_units[seed]["ridge_esm"]["metrics"]["mse"])),
            )
            unit_preds["ridge_esm"] = res_esm["ridge_esm"]["y_pred"]

        units.append({
            "unit": seed,
            "train_seqs": sorted(set(tr_df["sequence"].astype(str))),
            "test_seqs": sorted(set(te_seqs.tolist())),
            "smiles": te_smiles,
            "sequence": te_seqs,
            "y_true": yt,
            "preds": unit_preds,
        })

    return units, available, unavailable, sources, max_repro_diff


def _analyze_regime(
    ds: str,
    sp: str,
    proto: str,
    unit_prefix: str,
    units: list[dict[str, Any]],
    available_models: list[str],
    unavailable_models: list[str],
    model_sources: dict[str, str],
    mats: dict[str, np.ndarray],
    seq_to_idx: dict[str, int],
    n_bootstrap: int,
    seed: int,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    has_ridge = "ridge_esm" in available_models
    per_protein_records: list[dict[str, Any]] = []

    # 1. Identidade e MSE/margens por proteína de teste (u, p)
    unit_prot_identity: list[dict[str, dict[str, float]]] = []
    for u_data in units:
        u = int(u_data["unit"])
        tr_idx = np.asarray([seq_to_idx[s] for s in u_data["train_seqs"]], dtype=np.intp)
        te_seqs_u = u_data["test_seqs"]
        te_idx = np.asarray([seq_to_idx[s] for s in te_seqs_u], dtype=np.intp)

        sub_gl = mats["global_id_max"][np.ix_(te_idx, tr_idx)]
        sub_sw_aln = mats["sw_id_aln_len"][np.ix_(te_idx, tr_idx)]
        sub_sw_len = mats["sw_aln_len"][np.ix_(te_idx, tr_idx)]

        v_gl = np.max(sub_gl, axis=1)
        v_sw150 = np.max(np.where(sub_sw_len >= 150, sub_sw_aln, 0.0), axis=1)

        id_map_u: dict[str, dict[str, float]] = {}
        for pos, seq in enumerate(te_seqs_u):
            id_map_u[seq] = {
                "global_id_max": float(v_gl[pos]),
                "sw_id_aln_len_ge150aa": float(v_sw150[pos]),
            }
        unit_prot_identity.append(id_map_u)

        seq_arr = u_data["sequence"]
        yt = u_data["y_true"]
        for seq in te_seqs_u:
            pmask = seq_arr == seq
            yt_p = yt[pmask]
            mse_by_mod = {
                m: float(compute_mse(yt_p, u_data["preds"][m][pmask]))
                for m in available_models
            }
            mse_dm = mse_by_mod["drug_mean_predictor"]
            margin_dm = {m: float(mse_dm - mse_by_mod[m]) for m in available_models}
            margin_ridge = (
                {m: float(mse_by_mod["ridge_esm"] - mse_by_mod[m]) for m in available_models}
                if has_ridge
                else None
            )
            per_protein_records.append({
                "unit": u,
                "sequence": seq,
                "n_test_pairs": int(np.count_nonzero(pmask)),
                "global_id_max": id_map_u[seq]["global_id_max"],
                "sw_id_aln_len_ge150aa": id_map_u[seq]["sw_id_aln_len_ge150aa"],
                "mse_by_model": mse_by_mod,
                "margin_vs_drug_mean": margin_dm,
                "margin_vs_ridge_esm": margin_ridge,
            })

    # 2. Tendências contínuas e curvas por 5 quantis (cluster bootstrap por proteína única)
    rec_seqs = [r["sequence"] for r in per_protein_records]
    continuous_trends: dict[str, Any] = {}
    quantile_curves: dict[str, Any] = {}

    dm_trend_models = [m for m in available_models if m != "drug_mean_predictor"]
    ridge_trend_models = [m for m in available_models if m != "ridge_esm"] if has_ridge else []

    for axis in IDENTITY_AXES:
        x_all = np.asarray([r[axis] for r in per_protein_records], dtype=np.float64)

        axis_trends_dm: dict[str, Any] = {}
        for m in dm_trend_models:
            y_all = np.asarray([r["margin_vs_drug_mean"][m] for r in per_protein_records], dtype=np.float64)
            axis_trends_dm[m] = _cluster_bootstrap_trend(rec_seqs, x_all, y_all, n_bootstrap=n_bootstrap, seed=seed)

        axis_trends_ridge: dict[str, Any] = {}
        if has_ridge:
            for m in ridge_trend_models:
                y_all = np.asarray([r["margin_vs_ridge_esm"][m] for r in per_protein_records], dtype=np.float64)
                axis_trends_ridge[m] = _cluster_bootstrap_trend(rec_seqs, x_all, y_all, n_bootstrap=n_bootstrap, seed=seed)

        continuous_trends[axis] = {
            "margin_vs_drug_mean": axis_trends_dm,
            "margin_vs_ridge_esm": axis_trends_ridge if has_ridge else None,
        }

        # 5 quantis de igual contagem sobre observações (u, p) ordenadas deterministicamente
        order = sorted(
            range(len(per_protein_records)),
            key=lambda idx: (per_protein_records[idx][axis], per_protein_records[idx]["unit"], per_protein_records[idx]["sequence"]),
        )
        chunks = np.array_split(np.asarray(order, dtype=np.intp), 5)
        bins_out: list[dict[str, Any]] = []
        for q_idx, idx_arr in enumerate(chunks):
            if idx_arr.size == 0:
                continue
            q_recs = [per_protein_records[int(i)] for i in idx_arr]
            q_seqs = [r["sequence"] for r in q_recs]
            q_x = np.asarray([r[axis] for r in q_recs], dtype=np.float64)
            q_models: dict[str, Any] = {}
            for m in available_models:
                ym_dm = np.asarray([r["margin_vs_drug_mean"][m] for r in q_recs], dtype=np.float64)
                m_mean, m_ci = _cluster_bootstrap_mean(q_seqs, ym_dm, n_bootstrap=n_bootstrap, seed=seed + q_idx + 1)
                entry_m: dict[str, Any] = {
                    "margin_vs_drug_mean_mean": m_mean,
                    "margin_vs_drug_mean_ci95": m_ci,
                }
                if has_ridge:
                    ym_rg = np.asarray([r["margin_vs_ridge_esm"][m] for r in q_recs], dtype=np.float64)
                    rg_mean, rg_ci = _cluster_bootstrap_mean(q_seqs, ym_rg, n_bootstrap=n_bootstrap, seed=seed + q_idx + 101)
                    entry_m["margin_vs_ridge_esm_mean"] = rg_mean
                    entry_m["margin_vs_ridge_esm_ci95"] = rg_ci
                q_models[m] = entry_m

            bins_out.append({
                "quantile_index": int(q_idx),
                "n_obs": int(len(q_recs)),
                "n_unique_proteins": int(len(set(q_seqs))),
                "n_test_pairs": int(sum(r["n_test_pairs"] for r in q_recs)),
                "id_min": float(np.min(q_x)),
                "id_median": float(np.median(q_x)),
                "id_mean": float(np.mean(q_x)),
                "id_max": float(np.max(q_x)),
                "models": q_models,
            })

        quantile_curves[axis] = {
            "n_quantiles": len(bins_out),
            "bins": bins_out,
        }

    # 3. Faixas fixas interpretáveis ESTRITAMENTE dentro de cada unidade u (sem concatenação)
    bands_out: dict[str, Any] = {}
    for axis in IDENTITY_AXES:
        axis_bands: dict[str, Any] = {}
        for band_lbl, low, high in BAND_DEFINITIONS:
            per_unit_band: list[dict[str, Any]] = []
            prot_counts: list[float] = []
            pair_counts: list[float] = []
            eval_drug_counts: list[float] = []

            mod_mses: dict[str, list[float]] = {m: [] for m in available_models}
            mod_marg_dm: dict[str, list[float]] = {m: [] for m in available_models}
            mod_marg_rg: dict[str, list[float]] = {m: [] for m in available_models}
            mod_intra_ci: dict[str, list[float]] = {m: [] for m in available_models}
            mod_intra_sp: dict[str, list[float]] = {m: [] for m in available_models}

            for u_idx, u_data in enumerate(units):
                id_map_u = unit_prot_identity[u_idx]
                band_seqs = {
                    s for s, vals in id_map_u.items()
                    if _in_band(vals[axis], low, high)
                }
                if not band_seqs:
                    continue

                seq_arr = u_data["sequence"]
                bmask = np.isin(seq_arr, list(band_seqs))
                n_pairs_ub = int(np.count_nonzero(bmask))
                if n_pairs_ub == 0:
                    continue

                yt_ub = u_data["y_true"][bmask]
                sm_ub = u_data["smiles"][bmask]
                tot_g, elig_g = _prepare_group_indices(sm_ub, yt_ub, min_pairs=2)

                prot_counts.append(float(len(band_seqs)))
                pair_counts.append(float(n_pairs_ub))
                eval_drug_counts.append(float(len(elig_g)))

                mse_dm_ub = float(compute_mse(yt_ub, u_data["preds"]["drug_mean_predictor"][bmask]))
                mse_rg_ub = (
                    float(compute_mse(yt_ub, u_data["preds"]["ridge_esm"][bmask]))
                    if has_ridge
                    else None
                )

                unit_models_ub: dict[str, Any] = {}
                for m in available_models:
                    yp_ub = u_data["preds"][m][bmask]
                    mse_ub = float(compute_mse(yt_ub, yp_ub))
                    marg_dm_ub = float(mse_dm_ub - mse_ub)
                    marg_rg_ub = float(mse_rg_ub - mse_ub) if mse_rg_ub is not None else None
                    intra_ub = _intragroup_from_prepared(yt_ub, yp_ub, tot_g, elig_g)

                    mod_mses[m].append(mse_ub)
                    mod_marg_dm[m].append(marg_dm_ub)
                    if marg_rg_ub is not None:
                        mod_marg_rg[m].append(marg_rg_ub)
                    if intra_ub["ci_mean"] is not None:
                        mod_intra_ci[m].append(float(intra_ub["ci_mean"]))
                    if intra_ub["spearman_mean"] is not None:
                        mod_intra_sp[m].append(float(intra_ub["spearman_mean"]))

                    unit_models_ub[m] = {
                        "mse": mse_ub,
                        "margin_vs_drug_mean": marg_dm_ub,
                        "margin_vs_ridge_esm": marg_rg_ub,
                        "intra_ci_mean": intra_ub["ci_mean"],
                        "intra_spearman_mean": intra_ub["spearman_mean"],
                        "n_drugs_evaluable": int(intra_ub["n_groups_evaluable"]),
                    }

                per_unit_band.append({
                    "unit": int(u_data["unit"]),
                    "n_test_proteins": int(len(band_seqs)),
                    "n_test_pairs": n_pairs_ub,
                    "n_drugs_total": int(tot_g),
                    "n_drugs_evaluable": int(len(elig_g)),
                    "models": unit_models_ub,
                })

            n_active = len(per_unit_band)
            models_summary_b: dict[str, Any] = {}
            for m in available_models:
                if n_active == 0:
                    models_summary_b[m] = {
                        "n_units_active": 0,
                        "n_units_intra_evaluable": 0,
                        "mse_mean": None,
                        "mse_std": None,
                        "margin_vs_drug_mean_mean": None,
                        "margin_vs_drug_mean_std": None,
                        "margin_vs_ridge_esm_mean": None,
                        "margin_vs_ridge_esm_std": None,
                        "intra_ci_mean": None,
                        "intra_ci_std": None,
                        "intra_spearman_mean": None,
                        "intra_spearman_std": None,
                    }
                    continue

                mse_m, mse_s = sample_mean_std(mod_mses[m])
                mdm_m, mdm_s = sample_mean_std(mod_marg_dm[m])
                if has_ridge and mod_marg_rg[m]:
                    mrg_m, mrg_s = sample_mean_std(mod_marg_rg[m])
                else:
                    mrg_m, mrg_s = None, None

                n_intra = len(mod_intra_ci[m])
                if n_intra > 0:
                    ici_m, ici_s = sample_mean_std(mod_intra_ci[m])
                    isp_m, isp_s = sample_mean_std(mod_intra_sp[m])
                else:
                    ici_m, ici_s, isp_m, isp_s = None, None, None, None

                models_summary_b[m] = {
                    "n_units_active": int(n_active),
                    "n_units_intra_evaluable": int(n_intra),
                    "mse_mean": mse_m,
                    "mse_std": mse_s,
                    "margin_vs_drug_mean_mean": mdm_m,
                    "margin_vs_drug_mean_std": mdm_s,
                    "margin_vs_ridge_esm_mean": mrg_m,
                    "margin_vs_ridge_esm_std": mrg_s,
                    "intra_ci_mean": ici_m,
                    "intra_ci_std": ici_s,
                    "intra_spearman_mean": isp_m,
                    "intra_spearman_std": isp_s,
                }

            axis_bands[band_lbl] = {
                "n_units_nonempty": int(n_active),
                "mean_test_proteins_per_unit": float(np.mean(prot_counts)) if prot_counts else 0.0,
                "mean_test_pairs_per_unit": float(np.mean(pair_counts)) if pair_counts else 0.0,
                "mean_evaluable_drugs_per_unit": float(np.mean(eval_drug_counts)) if eval_drug_counts else 0.0,
                "per_unit": per_unit_band,
                "models": models_summary_b,
            }
        bands_out[axis] = axis_bands

    out: dict[str, Any] = {
        "dataset": ds,
        "split": sp,
        "protocol": proto,
        "unit_type": unit_prefix,
        "available_models": available_models,
        "unavailable_models": unavailable_models,
        "model_sources": model_sources,
        "n_units": len(units),
        "n_protein_observations": len(per_protein_records),
        "n_unique_test_proteins": len(set(rec_seqs)),
        "continuous_trends": continuous_trends,
        "quantile_curves": quantile_curves,
        "bands": bands_out,
        "per_protein_records": per_protein_records,
    }
    if extra_meta:
        out["reproduction_checks"] = extra_meta
    return out


def compute_continuous_identity_margin(
    n_workers: int = 4,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    repo_root, data_root = resolve_roots()

    df_map = {
        "davis": pd.read_csv(data_root / "data" / "processed" / "davis_clean.csv"),
        "kiba": pd.read_csv(data_root / "data" / "processed" / "kiba_clean.csv"),
    }

    mats_by_ds: dict[str, dict[str, np.ndarray]] = {}
    seq_idx_by_ds: dict[str, dict[str, int]] = {}
    for ds in ("davis", "kiba"):
        t_ds = time.perf_counter()
        seqs = sorted(set(df_map[ds]["sequence"].astype(str)))
        print(
            f"[{ds}] Calculando alinhamentos globais + Smith-Waterman para {len(seqs)} sequências ({len(seqs)*(len(seqs)-1)//2} pares) com {n_workers} workers...",
            flush=True,
        )
        mats_by_ds[ds] = compute_identity_matrices(seqs, n_workers=n_workers)
        seq_idx_by_ds[ds] = {s: i for i, s in enumerate(seqs)}
        print(f"[{ds}] Matrizes de identidade concluídas em {time.perf_counter() - t_ds:.1f}s.", flush=True)

    regimes_out: dict[str, Any] = {}
    for ds, sp, proto, uprefix in REGIME_SPECS:
        reg_key = f"{ds}/{sp}"
        print(f"[regime] {reg_key} ({proto})...", flush=True)
        extra_meta: dict[str, Any] | None = None
        if proto == "v2_kfold":
            units, avail, unavail, sources = _load_v2_kfold_units(data_root, ds, sp, df_map[ds])
        else:
            units, avail, unavail, sources, repro_diff = _load_kiba_cold_target_v1_units(data_root, df_map["kiba"])
            extra_meta = repro_diff

        regimes_out[reg_key] = _analyze_regime(
            ds=ds,
            sp=sp,
            proto=proto,
            unit_prefix=uprefix,
            units=units,
            available_models=avail,
            unavailable_models=unavail,
            model_sources=sources,
            mats=mats_by_ds[ds],
            seq_to_idx=seq_idx_by_ds[ds],
            n_bootstrap=n_bootstrap,
            seed=seed,
            extra_meta=extra_meta,
        )

    import Bio

    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "biopython_version": Bio.__version__,
        "alignment_specs": {
            "global_id_max": "PairwiseAligner(mode='global', matrix='BLOSUM62', open_gap=-10.0, extend_gap=-0.5); exact_matches_global / max(len_a, len_b)",
            "sw_id_aln_len_ge150aa": "PairwiseAligner(mode='local', matrix='BLOSUM62', open_gap=-10.0, extend_gap=-0.5); max_{train}(exact_matches_local / aln_length) com aln_length >= 150 aa (0.0 se nenhum)",
        },
        "bootstrap_config": {
            "B": int(n_bootstrap),
            "seed": int(seed),
            "cluster_unit": "unique_protein_sequence",
            "method": "percentile_95",
            "notes": "Cluster bootstrap agrupado por sequência proteica única p; nunca emite p-valores 0.000 ou 1.000.",
        },
        "band_definitions": [
            {"label": lbl, "low": low, "high": high}
            for lbl, low, high in BAND_DEFINITIONS
        ],
        "statistical_guardrails": {
            "cross_fold_ci_spearman_concatenation": False,
            "protocols_separated": ["v2_kfold", "v1_seeds"],
            "intragroup_rule": "CI e Spearman intragrupo (smiles, min_pairs=2, >=2 valores distintos de y_true) calculados estritamente dentro de cada unidade u e faixa b, agregados sobre unidades não vazias (ddof=1 para n>=2).",
        },
        "regimes": regimes_out,
    }


def main(argv: list[str] | None = None) -> int:
    default_workers = max(1, min(20, os.cpu_count() or 4))
    parser = argparse.ArgumentParser(
        description="Curva contínua identidade-margem estilo MIRAGE e faixas intragrupo (somente leitura em CPU)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Caminho de saída do JSON (padrão: docs/paper/continuous_identity_margin.json)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help=f"Número de processos no ProcessPoolExecutor (padrão: {default_workers})",
    )
    parser.add_argument(
        "--bootstrap-b",
        type=int,
        default=2000,
        help="Número de réplicas do cluster bootstrap por proteína (padrão: 2000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente do gerador aleatório do bootstrap (padrão: 42)",
    )
    args = parser.parse_args(argv)

    repo_root, _ = resolve_roots()
    out_path = (
        Path(args.output)
        if args.output is not None
        else ensure_paper_dirs(repo_root)["paper"] / "continuous_identity_margin.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = compute_continuous_identity_margin(
        n_workers=max(1, int(args.workers)),
        n_bootstrap=int(args.bootstrap_b),
        seed=int(args.seed),
    )
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] continuous_identity_margin.json salvo em {payload['elapsed_sec']}s: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
