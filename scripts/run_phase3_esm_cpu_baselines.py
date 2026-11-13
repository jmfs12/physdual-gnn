#!/usr/bin/env python3
"""Executa os controles baratos em CPU da Fase 3 (Etapa 3.0):

1. Baselines clássicos marginais (`drug_mean_predictor`, `target_mean_predictor`) e kNN
   (`knn_kmer_target`, `knn_tanimoto_drug`) nos novos splits por identidade de sequência do Davis
   (`cold_target_seqid30`, `cold_target_seqid40`) — executável localmente sem `torch`.
2. Controles sobre embeddings ESM-2 congelados (`knn_esm_target`, `ridge_esm_target_only`, `ridge_esm`)
   sobre `Davis v2_kfold`, `Davis seqid30/40`, `Davis v1_seeds` e `KIBA v1_seeds` (requer `torch` + cache `.pt`).

Saída principal:
- `--classical-seqid-only`: grava `outputs/results/phase3_seqid_classical_baselines.json`
- Padrão (completo): grava `outputs/results/phase3_esm_cpu_baselines.json`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from physdual.data.splits import load_splits  # noqa: E402
from physdual.eval.esm_baselines import (  # noqa: E402
    compute_morgan_ecfp4_map,
    evaluate_esm_cpu_baselines_split,
    load_mean_pooled_esm_embeddings,
)
from physdual.eval.metrics import compute_all_metrics  # noqa: E402
from physdual.utils.paths import get_project_root  # noqa: E402

try:
    from rdkit import DataStructs
except ImportError:
    DataStructs = None


def _compute_classical_unit(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    kmer_sim_mat: np.ndarray,
    seq_to_idx: dict[str, int],
    drug_fps: dict[str, Any],
    k: int = 5,
) -> dict[str, Any]:
    """Calcula `mean_predictor`, `drug_mean`, `target_mean`, `knn_tanimoto` e `knn_kmer` determinísticos."""
    y_tr = train_df["affinity"].to_numpy(dtype=np.float64)
    y_te = test_df["affinity"].to_numpy(dtype=np.float64)
    global_mean = float(np.mean(y_tr))
    drug_means = train_df.groupby("smiles")["affinity"].mean().to_dict()
    target_means = train_df.groupby("sequence")["affinity"].mean().to_dict()

    pred_mean = np.full(len(test_df), global_mean, dtype=np.float64)
    pred_drug = np.array([drug_means.get(s, global_mean) for s in test_df["smiles"]], dtype=np.float64)
    pred_target = np.array([target_means.get(s, global_mean) for s in test_df["sequence"]], dtype=np.float64)

    # knn_tanimoto_drug
    train_by_target: dict[str, list[tuple[str, float]]] = {}
    for seq, smi, aff in zip(train_df["sequence"], train_df["smiles"], y_tr):
        train_by_target.setdefault(str(seq), []).append((str(smi), float(aff)))
    pred_knn_drug = np.full(len(test_df), global_mean, dtype=np.float64)
    for idx, (seq, smi) in enumerate(zip(test_df["sequence"], test_df["smiles"])):
        cands = train_by_target.get(str(seq), [])
        fp_d = drug_fps.get(str(smi))
        if cands and fp_d is not None and DataStructs is not None:
            sims = np.array(
                [DataStructs.TanimotoSimilarity(fp_d, drug_fps[cs]) if cs in drug_fps else 0.0 for cs, _ in cands],
                dtype=np.float64,
            )
            affs = np.array([ca for _, ca in cands], dtype=np.float64)
            order = np.argsort(-sims, kind="stable")[:k]
            w = sims[order]
            a = affs[order]
            ws = float(np.sum(w))
            pred_knn_drug[idx] = float(np.sum(w * a) / ws) if ws > 1e-6 else float(np.mean(a))
        else:
            pred_knn_drug[idx] = float(drug_means.get(str(smi), global_mean))

    # knn_kmer_target
    train_by_drug: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    tmp_by_drug: dict[str, list[tuple[int, float]]] = {}
    for smi, seq, aff in zip(train_df["smiles"], train_df["sequence"], y_tr):
        tmp_by_drug.setdefault(str(smi), []).append((seq_to_idx[str(seq)], float(aff)))
    for smi, lst in tmp_by_drug.items():
        train_by_drug[smi] = (
            np.array([x[0] for x in lst], dtype=np.int64),
            np.array([x[1] for x in lst], dtype=np.float64),
        )
    pred_knn_target = np.full(len(test_df), global_mean, dtype=np.float64)
    for idx, (smi, seq) in enumerate(zip(test_df["smiles"], test_df["sequence"])):
        cand = train_by_drug.get(str(smi))
        if cand is not None:
            c_idx, c_aff = cand
            sims = kmer_sim_mat[c_idx, seq_to_idx[str(seq)]]
            order = np.argsort(-sims, kind="stable")[:k]
            w = sims[order]
            a = c_aff[order]
            ws = float(np.sum(w))
            pred_knn_target[idx] = float(np.sum(w * a) / ws) if ws > 1e-6 else float(np.mean(a))
        else:
            pred_knn_target[idx] = float(target_means.get(str(seq), global_mean))

    return {
        "mean_predictor": {"metrics": compute_all_metrics(y_te, pred_mean), "y_pred": pred_mean},
        "drug_mean_predictor": {"metrics": compute_all_metrics(y_te, pred_drug), "y_pred": pred_drug},
        "target_mean_predictor": {"metrics": compute_all_metrics(y_te, pred_target), "y_pred": pred_target},
        "knn_tanimoto_drug": {"metrics": compute_all_metrics(y_te, pred_knn_drug), "y_pred": pred_knn_drug},
        "knn_kmer_target": {"metrics": compute_all_metrics(y_te, pred_knn_target), "y_pred": pred_knn_target},
    }


def _build_3mer_norm_matrix(seqs: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    from collections import Counter

    uniq = sorted(set(seqs))
    vocab: dict[str, int] = {}
    rows_counts: list[dict[int, int]] = []
    for s in uniq:
        cnt: dict[int, int] = {}
        for pos in range(max(0, len(s) - 2)):
            tri = s[pos : pos + 3]
            idx = vocab.setdefault(tri, len(vocab))
            cnt[idx] = cnt.get(idx, 0) + 1
        rows_counts.append(cnt)

    mat = np.zeros((len(uniq), len(vocab)), dtype=np.float64)
    for r_i, cnt in enumerate(rows_counts):
        for c_i, v in cnt.items():
            mat[r_i, c_i] = float(v)
        norm = float(np.linalg.norm(mat[r_i]))
        if norm > 1e-12:
            mat[r_i] /= norm
    return mat, {s: i for i, s in enumerate(uniq)}


def _build_morgan_bitvects(smiles_list: list[str]) -> dict[str, Any]:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    try:
        from rdkit.Chem import rdFingerprintGenerator

        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    except Exception:
        gen = None
    out: dict[str, Any] = {}
    for smi in set(str(s) for s in smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        out[smi] = gen.GetFingerprint(mol) if gen is not None else AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    return out


def _aggregate_units(per_unit: list[dict[str, dict[str, Any]]], key_names: list[str]) -> dict[str, Any]:
    agg: dict[str, Any] = {}
    for model_key in key_names:
        agg[model_key] = {}
        for metric_name in ("ci", "mse", "rmse", "pearson", "spearman", "rm2"):
            vals = [float(u[model_key]["metrics"][metric_name]) for u in per_unit]
            agg[model_key][metric_name] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            }
    return agg


def main() -> int:
    parser = argparse.ArgumentParser(description="Controles baratos em CPU da Fase 3 (Etapa 3.0).")
    parser.add_argument("--classical-seqid-only", action="store_true", help="Roda apenas clássicos (sem ESM/torch) nos splits seqid30/40 do Davis.")
    parser.add_argument("--esm-model", default="esm2_t33_650M_UR50D")
    parser.add_argument("--esm-dim", type=int, default=1280)
    args = parser.parse_args()

    t0 = time.perf_counter()
    root = get_project_root()
    splits_dir = root / "data" / "processed" / "splits"

    davis_df = pd.read_csv(root / "data" / "processed" / "davis_clean.csv")
    kiba_df = pd.read_csv(root / "data" / "processed" / "kiba_clean.csv")

    if args.classical_seqid_only:
        kmer_mat, seq_to_idx = _build_3mer_norm_matrix(davis_df["sequence"].astype(str).tolist())
        kmer_sim_mat = kmer_mat @ kmer_mat.T
        fps = _build_morgan_bitvects(davis_df["smiles"].astype(str).tolist())
        out: dict[str, Any] = {"dataset": "davis", "splits": {}}
        for split_name in ("cold_target_seqid30", "cold_target_seqid40"):
            per_fold = []
            oof_y_true = []
            oof_preds: dict[str, list[np.ndarray]] = {}
            for k in range(5):
                sp = load_splits(splits_dir / f"davis_{split_name}_fold{k}.json")
                tr_df = davis_df.iloc[sp["train"]]
                te_df = davis_df.iloc[sp["test"]]
                res_k = _compute_classical_unit(tr_df, te_df, kmer_sim_mat, seq_to_idx, fps, k=5)
                per_fold.append({"fold": k, **{m: {"metrics": d["metrics"]} for m, d in res_k.items()}})
                oof_y_true.append(te_df["affinity"].to_numpy(dtype=np.float64))
                for m, d in res_k.items():
                    oof_preds.setdefault(m, []).append(d["y_pred"])
            model_keys = ["mean_predictor", "drug_mean_predictor", "target_mean_predictor", "knn_tanimoto_drug", "knn_kmer_target"]
            yt_all = np.concatenate(oof_y_true)
            oof_metrics = {m: compute_all_metrics(yt_all, np.concatenate(oof_preds[m])) for m in model_keys}
            out["splits"][split_name] = {
                "per_fold": per_fold,
                "per_fold_mean_std": _aggregate_units(per_fold, model_keys),
                "global_oof_metrics": oof_metrics,
            }
        out["elapsed_sec"] = round(time.perf_counter() - t0, 2)
        out_path = root / "outputs" / "results" / "phase3_seqid_classical_baselines.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out["splits"], indent=2))
        return 0

    # Modo completo: carrega mean_pool(ESM-2) para todas as sequências de Davis e KIBA
    all_seqs = sorted(set(davis_df["sequence"].astype(str)).union(set(kiba_df["sequence"].astype(str))))
    cache_dir = root / "cache" / "esm_embeddings"
    seq_to_vec = load_mean_pooled_esm_embeddings(
        cache_dir=cache_dir, sequences=all_seqs, model_name=args.esm_model, expected_dim=args.esm_dim
    )
    all_smiles = sorted(set(davis_df["smiles"].astype(str)).union(set(kiba_df["smiles"].astype(str))))
    drug_to_fp = compute_morgan_ecfp4_map(all_smiles, radius=2, n_bits=1024)

    regimes: list[tuple[str, pd.DataFrame, str, str, int]] = [
        ("davis_v2_cold_target_kfold", davis_df, "davis", "cold_target", 5),
        ("davis_v2_cold_drug_kfold", davis_df, "davis", "cold_drug", 5),
        ("davis_seqid30_kfold", davis_df, "davis", "cold_target_seqid30", 5),
        ("davis_seqid40_kfold", davis_df, "davis", "cold_target_seqid40", 5),
        ("davis_v1_random", davis_df, "davis", "random", 5),
        ("kiba_v1_random", kiba_df, "kiba", "random", 5),
        ("kiba_v1_cold_drug", kiba_df, "kiba", "cold_drug", 5),
        ("kiba_v1_cold_target", kiba_df, "kiba", "cold_target", 5),
    ]

    full_out: dict[str, Any] = {
        "esm_model": args.esm_model,
        "esm_dim": args.esm_dim,
        "regimes": {},
    }
    for label, df, ds_name, sp_name, n_units in regimes:
        is_kfold = "kfold" in label
        unit_prefix = "fold" if is_kfold else "seed"
        per_unit = []
        oof_yt = []
        oof_preds: dict[str, list[np.ndarray]] = {}
        for u in range(n_units):
            sp = load_splits(splits_dir / f"{ds_name}_{sp_name}_{unit_prefix}{u}.json")
            res_u = evaluate_esm_cpu_baselines_split(
                df.iloc[sp["train"]],
                df.iloc[sp["val"]],
                df.iloc[sp["test"]],
                seq_to_vec=seq_to_vec,
                drug_to_fp=drug_to_fp,
                k=5,
            )
            entry = {
                unit_prefix: u,
                **{
                    m: {
                        "metrics": d["metrics"],
                        **({k_: d[k_] for k_ in ("best_alpha", "val_mse") if k_ in d}),
                    }
                    for m, d in res_u.items()
                },
            }
            per_unit.append(entry)
            if is_kfold:
                oof_yt.append(df.iloc[sp["test"]]["affinity"].to_numpy(dtype=np.float64))
                for m, d in res_u.items():
                    oof_preds.setdefault(m, []).append(d["y_pred"])
        m_keys = ["knn_esm_target", "ridge_esm_target_only", "ridge_esm"]
        reg_entry: dict[str, Any] = {
            "per_unit": per_unit,
            "mean_std": _aggregate_units(per_unit, m_keys),
        }
        if is_kfold and oof_yt:
            yt_all = np.concatenate(oof_yt)
            reg_entry["global_oof_metrics"] = {
                m: compute_all_metrics(yt_all, np.concatenate(oof_preds[m])) for m in m_keys
            }
        full_out["regimes"][label] = reg_entry

    full_out["elapsed_sec"] = round(time.perf_counter() - t0, 2)
    out_path = root / "outputs" / "results" / "phase3_esm_cpu_baselines.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(full_out, indent=2), encoding="utf-8")
    print(f"[OK] Controles ESM CPU salvos em: {out_path} ({full_out['elapsed_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
