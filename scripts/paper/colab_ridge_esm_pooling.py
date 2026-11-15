"""Script diagnóstico para Google Colab: Pooling de embeddings ESM-2 `[L, 1280]` no Ridge.

Objetivo (item 7b pós-auditoria da Fase 3):
  Investigar, sem treinar redes neurais pesadas, se o colapso de ordenação intragrupo
  nos regimes `cold_target*` decorre especificamente do `mean pooling` sobre a dimensão
  de sequência (`[L, 1280] -> [1280]`) nos caches `.pt` do ESM-2 (presentes no Drive/cache),
  comparando três estratégias de agregação de sequência em `Ridge`:
    1. `mean`     (`1280d`): média aritmética sobre os `L` resíduos.
    2. `max`      (`1280d`): máximo por canal ao longo dos `L` resíduos.
    3. `mean_std` (`2560d`): concatenação `[mean ; std]` por canal sobre os `L` resíduos.

Cada modo é avaliado em `Ridge` sob dois alvos (com padronização no treino e seleção de
`alpha` exclusivamente em `val`, zero vazamento de teste):
  - `raw_y`: regressão direta `y ~ pooling(ESM)` e `y ~ [pooling(ESM) ; ECFP4]`.
  - `residual_drug_mean`: empilhamento `y - drug_mean_train ~ pooling(ESM)` somando
    `drug_mean_train` de volta na predição.

Reporta `CI`, `MSE`, `RM2` globais/por unidade e `CI`/`Spearman` intragrupo (por fármaco
`smiles`, média intradobra/intrastatus sem viés de leave-one-fold-out).
Uso típico no Colab:
  python scripts/paper/colab_ridge_esm_pooling.py \\
    --data-root /content/drive/MyDrive/physdual-gnn \\
    --esm-cache-dir /content/drive/MyDrive/physdual-gnn/cache/esm_embeddings \\
    --output /content/drive/MyDrive/physdual-gnn/outputs/results/phase3_colab_ridge_pooling.json
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
import torch
from scipy import stats
from scipy.linalg import cho_factor, cho_solve, eigh as sp_eigh

PAPER_SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAPER_SCRIPTS_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
for p in (str(PAPER_SCRIPTS_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common import load_json_strict, resolve_roots, sample_mean_std  # noqa: E402
from physdual.data.esm_cache import compute_seq_hash, get_embedding_cache_path  # noqa: E402
from physdual.data.splits import load_splits  # noqa: E402
from physdual.eval.esm_baselines import compute_morgan_ecfp4_map  # noqa: E402
from physdual.eval.metrics import (  # noqa: E402
    compute_ci,
    compute_mse,
    compute_pearson,
    compute_rm2,
    compute_rmse,
    compute_spearman,
)

POOLING_MODES: tuple[str, ...] = ("mean", "max", "mean_std")
RIDGE_ALPHAS: tuple[float, ...] = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 50000.0)


def load_esm_pooling_variants(
    cache_dir: Path,
    sequences: list[str],
    model_name: str = "esm2_t33_650M_UR50D",
    expected_dim: int = 1280,
) -> dict[str, dict[str, np.ndarray]]:
    """Carrega cada `[L, 1280]` `.pt` uma única vez e extrai `mean`, `max` e `mean_std`."""
    out: dict[str, dict[str, np.ndarray]] = {mode: {} for mode in POOLING_MODES}
    uniq = sorted(set(str(s) for s in sequences))
    for i, seq in enumerate(uniq, 1):
        pt_path = get_embedding_cache_path(seq, model_name=model_name, cache_dir=cache_dir)
        if not pt_path.exists():
            raise FileNotFoundError(
                f"Embedding ESM-2 ausente ({compute_seq_hash(seq)}): {pt_path}"
            )
        payload = torch.load(pt_path, map_location="cpu", weights_only=True)
        emb = payload["embedding"] if isinstance(payload, dict) else payload
        if emb.ndim != 2 or int(emb.size(1)) != int(expected_dim):
            raise ValueError(f"Shape inválido em {pt_path}: {tuple(emb.shape)}")
        if not bool(torch.isfinite(emb).all()):
            raise ValueError(f"NaN/Inf em {pt_path}")
        emb_f = emb.to(dtype=torch.float32)
        v_mean = emb_f.mean(dim=0).numpy().astype(np.float64)
        v_max = emb_f.max(dim=0).values.numpy().astype(np.float64)
        v_std = emb_f.std(dim=0, unbiased=False).numpy().astype(np.float64)
        out["mean"][seq] = v_mean
        out["max"][seq] = v_max
        out["mean_std"][seq] = np.concatenate([v_mean, v_std], axis=0)
        if i % 100 == 0 or i == len(uniq):
            print(f"  [ESM-cache] {i}/{len(uniq)} embeddings carregados", flush=True)
    return out


def _fit_ridge_val_alpha(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    X_te: np.ndarray,
    alphas: tuple[float, ...] = RIDGE_ALPHAS,
) -> tuple[float, np.ndarray]:
    mu = np.mean(X_tr, axis=0)
    sd = np.std(X_tr, axis=0)
    sd[sd < 1e-12] = 1.0
    Z_tr = (X_tr - mu) / sd
    Z_va = (X_va - mu) / sd
    Z_te = (X_te - mu) / sd

    y_mean = float(np.mean(y_tr))
    y_c = y_tr - y_mean
    G = 0.5 * (Z_tr.T @ Z_tr + (Z_tr.T @ Z_tr).T)
    g_y = Z_tr.T @ y_c
    diag_idx = np.arange(G.shape[0])

    best_alpha = float(alphas[0])
    best_val_mse = float("inf")
    best_w: np.ndarray | None = None
    eig_cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    for alpha in alphas:
        a = float(alpha)
        if eig_cache is not None:
            ev, evec, proj = eig_cache
            w = evec @ (proj / (ev + a))
        else:
            A = G.copy()
            A[diag_idx, diag_idx] += a
            try:
                cf = cho_factor(A, check_finite=False, overwrite_a=True)
                w = cho_solve(cf, g_y, check_finite=False)
            except np.linalg.LinAlgError:
                ev, evec = sp_eigh(G, check_finite=False, driver="evd")
                ev = np.maximum(ev, 0.0)
                proj = evec.T @ g_y
                eig_cache = (ev, evec, proj)
                w = evec @ (proj / (ev + a))
        val_pred = y_mean + Z_va @ w
        val_mse = float(np.mean((y_va - val_pred) ** 2))
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_alpha = a
            best_w = w

    assert best_w is not None
    return best_alpha, y_mean + Z_te @ best_w


def _intragroup_ci_spearman_unit(
    groups: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    min_pairs: int = 2,
) -> dict[str, Any]:
    raw_groups = pd.Series(np.arange(len(groups))).groupby(groups, sort=False).indices
    ci_vals: list[float] = []
    sp_vals: list[float] = []
    for idx in raw_groups.values():
        if len(idx) < min_pairs:
            continue
        yt_g = y_true[idx]
        if np.unique(yt_g).size < 2:
            continue
        yp_g = y_pred[idx]
        ci_vals.append(float(compute_ci(yt_g, yp_g)))
        if np.ptp(yp_g) <= 1e-12:
            sp_vals.append(0.0)
        else:
            rho = float(stats.spearmanr(yt_g, yp_g).statistic)
            sp_vals.append(0.0 if not np.isfinite(rho) else rho)
    if not ci_vals:
        return {"n_groups_evaluable": 0, "ci_mean": None, "spearman_mean": None}
    return {
        "n_groups_evaluable": int(len(ci_vals)),
        "ci_mean": float(np.mean(ci_vals)),
        "spearman_mean": float(np.mean(sp_vals)),
    }


def _unit_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "ci": float(compute_ci(y_true, y_pred)),
        "mse": float(compute_mse(y_true, y_pred)),
        "rmse": float(compute_rmse(y_true, y_pred)),
        "pearson": float(compute_pearson(y_true, y_pred)),
        "spearman": float(compute_spearman(y_true, y_pred)),
        "rm2": float(compute_rm2(y_true, y_pred)),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnóstico Colab: pooling ESM-2 (mean, max, mean_std) no Ridge em cold_target*"
    )
    parser.add_argument("--data-root", type=Path, default=None, help="Raiz de data/ e outputs/")
    parser.add_argument(
        "--esm-cache-dir",
        type=Path,
        default=None,
        help="Diretório base de caches ESM (contendo subpasta esm2_t33_650M_UR50D)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["davis", "kiba"],
        choices=["davis", "kiba"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON de saída (padrão: <data_root>/outputs/results/phase3_colab_ridge_pooling.json)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    t0 = time.perf_counter()
    args = parse_args(argv)
    _, default_data_root = resolve_roots()
    data_root = args.data_root.resolve() if args.data_root else default_data_root
    esm_cache_dir = (
        args.esm_cache_dir.resolve()
        if args.esm_cache_dir
        else (data_root / "cache" / "esm_embeddings")
    )
    out_path = (
        args.output.resolve()
        if args.output
        else (data_root / "outputs" / "results" / "phase3_colab_ridge_pooling.json")
    )

    regimes: list[tuple[str, str, str]] = []
    if "davis" in args.datasets:
        regimes.extend([
            ("davis", "cold_target", "kfold"),
            ("davis", "cold_target_seqid40", "kfold"),
            ("davis", "cold_target_seqid30", "kfold"),
        ])
    if "kiba" in args.datasets:
        regimes.extend([
            ("kiba", "cold_target", "seeds"),
            ("kiba", "cold_target_seqid40", "kfold"),
            ("kiba", "cold_target_seqid30", "kfold"),
        ])

    df_map = {
        ds: pd.read_csv(data_root / "data" / "processed" / f"{ds}_clean.csv")
        for ds in sorted(set(args.datasets))
    }
    all_seqs = sorted(set().union(*(df["sequence"].astype(str) for df in df_map.values())))
    all_smiles = sorted(set().union(*(df["smiles"].astype(str) for df in df_map.values())))

    print(f"[INIT] Carregando caches ESM de {esm_cache_dir}...", flush=True)
    pooled_maps = load_esm_pooling_variants(esm_cache_dir, all_seqs)
    drug_to_fp = compute_morgan_ecfp4_map(all_smiles, radius=2, n_bits=1024)

    splits_dir = data_root / "data" / "processed" / "splits"
    results: dict[str, Any] = {}

    for ds, sp, proto in regimes:
        reg_key = f"{ds}/{sp}"
        unit_prefix = "fold" if proto == "kfold" else "seed"
        df = df_map[ds]
        print(f"[RUN] {reg_key} ({proto})...", flush=True)

        regime_res: dict[str, Any] = {"protocol": "v2_kfold" if proto == "kfold" else "v1_seeds", "modes": {}}
        for mode in POOLING_MODES:
            seq_vec = pooled_maps[mode]
            per_task: dict[str, list[dict[str, Any]]] = {
                "raw_y_esm_only": [],
                "raw_y_esm_ecfp": [],
                "residual_drug_mean_esm": [],
            }
            for u in range(5):
                sp_idx = load_splits(splits_dir / f"{ds}_{sp}_{unit_prefix}{u}.json")
                tr_df = df.iloc[sp_idx["train"]]
                va_df = df.iloc[sp_idx["val"]]
                te_df = df.iloc[sp_idx["test"]]

                y_tr = tr_df["affinity"].to_numpy(dtype=np.float64)
                y_va = va_df["affinity"].to_numpy(dtype=np.float64)
                y_te = te_df["affinity"].to_numpy(dtype=np.float64)
                grp_te = te_df["smiles"].astype(str).to_numpy()

                P_tr = np.vstack([seq_vec[s] for s in tr_df["sequence"].astype(str)])
                P_va = np.vstack([seq_vec[s] for s in va_df["sequence"].astype(str)])
                P_te = np.vstack([seq_vec[s] for s in te_df["sequence"].astype(str)])

                D_tr = np.vstack([drug_to_fp[s] for s in tr_df["smiles"].astype(str)])
                D_va = np.vstack([drug_to_fp[s] for s in va_df["smiles"].astype(str)])
                D_te = np.vstack([drug_to_fp[s] for s in te_df["smiles"].astype(str)])

                a1, p_raw_esm = _fit_ridge_val_alpha(P_tr, y_tr, P_va, y_va, P_te)
                a2, p_raw_both = _fit_ridge_val_alpha(
                    np.hstack([P_tr, D_tr]), y_tr,
                    np.hstack([P_va, D_va]), y_va,
                    np.hstack([P_te, D_te]),
                )

                gmean = float(np.mean(y_tr))
                dmap = tr_df.groupby("smiles")["affinity"].mean().to_dict()
                b_tr = tr_df["smiles"].astype(str).map(dmap).fillna(gmean).to_numpy(dtype=np.float64)
                b_va = va_df["smiles"].astype(str).map(dmap).fillna(gmean).to_numpy(dtype=np.float64)
                b_te = te_df["smiles"].astype(str).map(dmap).fillna(gmean).to_numpy(dtype=np.float64)
                a3, r_hat = _fit_ridge_val_alpha(P_tr, y_tr - b_tr, P_va, y_va - b_va, P_te)
                p_res = b_te + r_hat

                for tname, yp, alpha in (
                    ("raw_y_esm_only", p_raw_esm, a1),
                    ("raw_y_esm_ecfp", p_raw_both, a2),
                    ("residual_drug_mean_esm", p_res, a3),
                ):
                    m = _unit_metrics(y_te, yp)
                    intra = _intragroup_ci_spearman_unit(grp_te, y_te, yp, min_pairs=2)
                    per_task[tname].append({
                        "unit": u,
                        "alpha": alpha,
                        **m,
                        "intra_ci_mean": intra["ci_mean"],
                        "intra_spearman_mean": intra["spearman_mean"],
                    })

            regime_res["modes"][mode] = {
                tname: {
                    "ci": sample_mean_std([x["ci"] for x in items]),
                    "mse": sample_mean_std([x["mse"] for x in items]),
                    "rm2": sample_mean_std([x["rm2"] for x in items]),
                    "intra_ci": sample_mean_std([x["intra_ci_mean"] for x in items if x["intra_ci_mean"] is not None]),
                    "intra_spearman": sample_mean_std([x["intra_spearman_mean"] for x in items if x["intra_spearman_mean"] is not None]),
                    "per_unit": items,
                }
                for tname, items in per_task.items()
            }
        results[reg_key] = regime_res

    payload = {
        "script": "colab_ridge_esm_pooling.py",
        "diagnostic_only": True,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "pooling_modes": list(POOLING_MODES),
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Relatório salvo em {out_path} ({payload['elapsed_sec']}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
