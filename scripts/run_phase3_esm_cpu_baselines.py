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


def _build_tanimoto_sim_matrix(smiles_list: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    """Pré-computa matriz simétrica de similaridade de Tanimoto (ECFP4) via BulkTanimotoSimilarity."""
    uniq = sorted(set(str(s) for s in smiles_list))
    fps = _build_morgan_bitvects(uniq)
    n = len(uniq)
    mat = np.zeros((n, n), dtype=np.float64)
    if DataStructs is not None:
        fp_list = [fps.get(s) for s in uniq]
        valid_idx = [i for i, fp in enumerate(fp_list) if fp is not None]
        valid_fps = [fp_list[i] for i in valid_idx]
        for row_pos, i in enumerate(valid_idx):
            sims = DataStructs.BulkTanimotoSimilarity(valid_fps[row_pos], valid_fps)
            mat[i, valid_idx] = np.asarray(sims, dtype=np.float64)
    return mat, {s: i for i, s in enumerate(uniq)}


def _compute_classical_unit(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    kmer_sim_mat: np.ndarray,
    seq_to_idx: dict[str, int],
    drug_fps: dict[str, Any],
    k: int = 5,
    tanimoto_sim_mat: np.ndarray | None = None,
    drug_to_idx: dict[str, int] | None = None,
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
    pred_knn_drug = np.full(len(test_df), global_mean, dtype=np.float64)
    if tanimoto_sim_mat is not None and drug_to_idx is not None:
        tmp_by_target: dict[str, list[tuple[int, float]]] = {}
        for seq, smi, aff in zip(train_df["sequence"], train_df["smiles"], y_tr):
            d_i = drug_to_idx.get(str(smi))
            if d_i is not None:
                tmp_by_target.setdefault(str(seq), []).append((d_i, float(aff)))
        train_by_target_idx = {
            seq: (np.array([x[0] for x in lst], dtype=np.int64), np.array([x[1] for x in lst], dtype=np.float64))
            for seq, lst in tmp_by_target.items()
        }
        for idx, (seq, smi) in enumerate(zip(test_df["sequence"], test_df["smiles"])):
            cand = train_by_target_idx.get(str(seq))
            d_idx = drug_to_idx.get(str(smi))
            if cand is not None and d_idx is not None:
                c_idx, c_aff = cand
                sims = tanimoto_sim_mat[c_idx, d_idx]
                order = np.argsort(-sims, kind="stable")[:k]
                w = sims[order]
                a = c_aff[order]
                ws = float(np.sum(w))
                pred_knn_drug[idx] = float(np.sum(w * a) / ws) if ws > 1e-6 else float(np.mean(a))
            else:
                pred_knn_drug[idx] = float(drug_means.get(str(smi), global_mean))
    else:
        train_by_target: dict[str, list[tuple[str, float]]] = {}
        for seq, smi, aff in zip(train_df["sequence"], train_df["smiles"], y_tr):
            train_by_target.setdefault(str(seq), []).append((str(smi), float(aff)))
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


def _build_oof_unit_df(
    fold_idx: int,
    test_indices: list[int],
    te_df: pd.DataFrame,
    res_u: dict[str, dict[str, Any]],
    cl_u: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    return pd.DataFrame({
        "row_index": test_indices,
        "fold": fold_idx,
        "drug_id": te_df["drug_id"].astype(str).tolist(),
        "target_id": te_df["target_id"].astype(str).tolist(),
        "smiles": te_df["smiles"].astype(str).tolist(),
        "sequence": te_df["sequence"].astype(str).tolist(),
        "y_true": te_df["affinity"].to_numpy(dtype=np.float64),
        "y_pred_knn_esm_target": res_u["knn_esm_target"]["y_pred"],
        "y_pred_ridge_esm_target_only": res_u["ridge_esm_target_only"]["y_pred"],
        "y_pred_ridge_esm": res_u["ridge_esm"]["y_pred"],
        "y_pred_mean_predictor": cl_u["mean_predictor"]["y_pred"],
        "y_pred_drug_mean_predictor": cl_u["drug_mean_predictor"]["y_pred"],
        "y_pred_target_mean_predictor": cl_u["target_mean_predictor"]["y_pred"],
        "y_pred_knn_tanimoto_drug": cl_u["knn_tanimoto_drug"]["y_pred"],
        "y_pred_knn_kmer_target": cl_u["knn_kmer_target"]["y_pred"],
    })


def main() -> int:
    parser = argparse.ArgumentParser(description="Controles baratos em CPU da Fase 3 (Etapa 3.0).")
    parser.add_argument("--classical-seqid-only", action="store_true", help="Roda apenas clássicos (sem ESM/torch) nos splits seqid30/40 do Davis.")
    parser.add_argument("--kiba-classical", action="store_true", help="Calcula marginais e kNN em CPU para KIBA (v1_seeds e seqid30/40).")
    parser.add_argument("--export-davis-oof", action="store_true", help="Exporta OOFs alinhados dos baselines CPU (ESM + clássicos) para Davis kfold.")
    parser.add_argument("--export-kiba-oof", action="store_true", help="Exporta OOFs alinhados dos baselines CPU (ESM + clássicos) para KIBA seqid30/40 kfold.")
    parser.add_argument("--regimes", type=str, default=None, help="Lista separada por vírgula de regimes a executar e mesclar no JSON existente.")
    parser.add_argument("--esm-model", default="esm2_t33_650M_UR50D")
    parser.add_argument("--esm-dim", type=int, default=1280)
    args = parser.parse_args()

    t0 = time.perf_counter()
    root = get_project_root()
    splits_dir = root / "data" / "processed" / "splits"

    davis_df = pd.read_csv(root / "data" / "processed" / "davis_clean.csv")
    kiba_df = pd.read_csv(root / "data" / "processed" / "kiba_clean.csv")

    if args.kiba_classical:
        kmer_mat, seq_to_idx = _build_3mer_norm_matrix(kiba_df["sequence"].astype(str).tolist())
        kmer_sim_mat = kmer_mat @ kmer_mat.T
        tan_sim_mat, drug_to_idx = _build_tanimoto_sim_matrix(kiba_df["smiles"].astype(str).tolist())
        fps: dict[str, Any] = {}
        kiba_regimes = [
            ("random", "seeds"),
            ("cold_drug", "seeds"),
            ("cold_target", "seeds"),
            ("cold_target_seqid30", "kfold"),
            ("cold_target_seqid40", "kfold"),
        ]
        model_keys = ["mean_predictor", "drug_mean_predictor", "target_mean_predictor", "knn_tanimoto_drug", "knn_kmer_target"]
        kiba_out: dict[str, Any] = {"dataset": "kiba", "regimes": {}}
        for sp_name, proto in kiba_regimes:
            u_prefix = "fold" if proto == "kfold" else "seed"
            per_unit = []
            oof_yt: list[np.ndarray] = []
            oof_preds: dict[str, list[np.ndarray]] = {}
            for u in range(5):
                sp = load_splits(splits_dir / f"kiba_{sp_name}_{u_prefix}{u}.json")
                tr_df = kiba_df.iloc[sp["train"]]
                te_df = kiba_df.iloc[sp["test"]]
                res_u = _compute_classical_unit(
                    tr_df, te_df, kmer_sim_mat, seq_to_idx, fps, k=5,
                    tanimoto_sim_mat=tan_sim_mat, drug_to_idx=drug_to_idx,
                )
                per_unit.append({u_prefix: u, **{m: {"metrics": d["metrics"]} for m, d in res_u.items()}})
                if proto == "kfold":
                    oof_yt.append(te_df["affinity"].to_numpy(dtype=np.float64))
                    for m, d in res_u.items():
                        oof_preds.setdefault(m, []).append(d["y_pred"])
            entry: dict[str, Any] = {
                "protocol": proto,
                "per_unit": per_unit,
                "mean_std": _aggregate_units(per_unit, model_keys),
            }
            if proto == "kfold" and oof_yt:
                yt_all = np.concatenate(oof_yt)
                entry["global_oof_metrics"] = {
                    m: compute_all_metrics(yt_all, np.concatenate(oof_preds[m])) for m in model_keys
                }
            kiba_out["regimes"][sp_name] = entry
        kiba_out["elapsed_sec"] = round(time.perf_counter() - t0, 2)
        out_path = root / "outputs" / "results" / "kiba_cpu_classical_baselines.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(kiba_out, indent=2), encoding="utf-8")
        print(f"[OK] Baselines clássicos KIBA salvos em: {out_path} ({kiba_out['elapsed_sec']}s)")
        return 0

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
        ("kiba_seqid30_kfold", kiba_df, "kiba", "cold_target_seqid30", 5),
        ("kiba_seqid40_kfold", kiba_df, "kiba", "cold_target_seqid40", 5),
    ]

    oof_dir = root / "outputs" / "results" / "oof"
    oof_dir.mkdir(parents=True, exist_ok=True)
    classical_cache: dict[str, tuple[np.ndarray, dict[str, int], np.ndarray, dict[str, int]]] = {}
    if args.export_davis_oof:
        km, s_idx = _build_3mer_norm_matrix(davis_df["sequence"].astype(str).tolist())
        tm, d_idx = _build_tanimoto_sim_matrix(davis_df["smiles"].astype(str).tolist())
        classical_cache["davis"] = (km @ km.T, s_idx, tm, d_idx)
    if args.export_kiba_oof:
        km, s_idx = _build_3mer_norm_matrix(kiba_df["sequence"].astype(str).tolist())
        tm, d_idx = _build_tanimoto_sim_matrix(kiba_df["smiles"].astype(str).tolist())
        classical_cache["kiba"] = (km @ km.T, s_idx, tm, d_idx)

    out_path = root / "outputs" / "results" / "phase3_esm_cpu_baselines.json"
    selected_regimes = {r.strip() for r in args.regimes.split(",") if r.strip()} if args.regimes else None
    if selected_regimes:
        regimes = [r for r in regimes if r[0] in selected_regimes]
        if out_path.exists():
            full_out = json.loads(out_path.read_text(encoding="utf-8"))
        else:
            full_out = {"esm_model": args.esm_model, "esm_dim": args.esm_dim, "regimes": {}}
    else:
        full_out = {
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
        oof_rows: list[pd.DataFrame] = []
        for u in range(n_units):
            sp = load_splits(splits_dir / f"{ds_name}_{sp_name}_{unit_prefix}{u}.json")
            tr_df = df.iloc[sp["train"]]
            va_df = df.iloc[sp["val"]]
            te_df = df.iloc[sp["test"]]
            res_u = evaluate_esm_cpu_baselines_split(
                tr_df,
                va_df,
                te_df,
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
                oof_yt.append(te_df["affinity"].to_numpy(dtype=np.float64))
                for m, d in res_u.items():
                    oof_preds.setdefault(m, []).append(d["y_pred"])
                if ds_name in classical_cache:
                    km_mat, s_idx, tan_mat, d_idx = classical_cache[ds_name]
                    cl_u = _compute_classical_unit(
                        tr_df, te_df, km_mat, s_idx, {}, k=5,
                        tanimoto_sim_mat=tan_mat, drug_to_idx=d_idx,
                    )
                    oof_rows.append(_build_oof_unit_df(u, sp["test"], te_df, res_u, cl_u))
        if ds_name in classical_cache and oof_rows:
            pd.concat(oof_rows, ignore_index=True).to_csv(
                oof_dir / f"cpu_baselines_{ds_name}_{sp_name}_oof.csv",
                index=False,
            )
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(full_out, indent=2), encoding="utf-8")
    print(f"[OK] Controles ESM CPU salvos em: {out_path} ({full_out['elapsed_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
