"""Auditorias determinísticas em CPU para o pacote científico (Fases 1 e 2).

Computa exclusivamente a partir de `data/processed/*.csv`, `data/processed/splits/*.json`
e `outputs/results/runs/*.json`:
1. Piso de ruído por colisão de conteúdo `(smiles, sequence)` no Davis e KIBA.
2. Baselines `mean_predictor` e marginais no subconjunto não censurado `pKd > 5.0` (Davis v1 e v2).
3. Baselines marginais (`drug_mean`, `target_mean`) e `kNN` (`Tanimoto ECFP4`, `3-mer cosine`, k=5)
   para todas as 45 partições `v1_seeds`, com deltas pareados por seed contra os JSONs de run.
4. Estatísticas recalculadas de `kiba_random` apenas com execuções não retomadas (`seeds 0, 2, 4`).
5. Clusterização por identidade de sequência das 379 proteínas únicas do Davis via `biopython`
   (`PairwiseAligner` global), registrando tamanhos de clusters e partição 5-fold proposta.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from concurrent.futures import ProcessPoolExecutor
import os

import Bio
import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner, substitution_matrices
from lifelines.utils import concordance_index
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy import stats

from common import resolve_roots


def _compute_metrics_lite(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if len(yt) == 0:
        return {k: float("nan") for k in ("ci", "mse", "rmse", "pearson", "spearman", "rm2")}

    mse = float(np.mean((yt - yp) ** 2))
    rmse = float(math.sqrt(mse))
    if np.unique(yt).size < 2 or np.unique(yp).size < 2:
        ci = 0.5
        pearson = 0.0
        spearman = 0.0
        rm2 = 0.0
    else:
        ci = float(concordance_index(yt, yp))
        pearson = float(stats.pearsonr(yt, yp)[0])
        spearman = float(stats.spearmanr(yt, yp)[0])
        r2 = pearson ** 2 if pearson > 0 else 0.0
        # rm2 com guarda r <= 0 -> 0.0 (mesma regra de physdual.eval.metrics.compute_rm2)
        if pearson <= 0:
            rm2 = 0.0
        else:
            yp0 = yp - np.mean(yp)
            yt0 = yt - np.mean(yt)
            denom = float(np.sum(yp0 ** 2))
            k = float(np.sum(yt0 * yp0) / denom) if denom > 0 else 0.0
            num0 = float(np.sum((yt0 - k * yp0) ** 2))
            den0 = float(np.sum(yt0 ** 2))
            r02 = 1.0 - (num0 / den0) if den0 > 0 else 0.0
            rm2 = float(r2 * (1.0 - math.sqrt(abs(r2 - r02))))
    return {
        "ci": ci,
        "mse": mse,
        "rmse": rmse,
        "pearson": pearson,
        "spearman": spearman,
        "rm2": rm2,
    }


def _mean_std_dict(vals: list[float]) -> dict[str, float | int]:
    clean = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
    n = len(clean)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    mean_v = float(statistics.mean(clean))
    std_v = float(statistics.stdev(clean)) if n > 1 else 0.0
    return {"mean": mean_v, "std": std_v, "n": n}


def compute_noise_floor(data_root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for ds in ("davis", "kiba"):
        df = pd.read_csv(data_root / "data" / "processed" / f"{ds}_clean.csv")
        grp = df.groupby(["smiles", "sequence"])["affinity"]
        sizes = grp.size()
        nunq = grp.nunique()
        dup_groups = sizes[sizes > 1]
        diff_groups = nunq[nunq > 1]
        g_mean = grp.transform("mean")
        sq_err = (df["affinity"] - g_mean) ** 2
        dup_mask = grp.transform("size") > 1
        diff_mask = grp.transform("nunique") > 1

        seq_targets = df.groupby("sequence")["target_id"].unique()
        shared_seq = seq_targets[seq_targets.apply(len) > 1]
        out[ds] = {
            "total_pairs": int(len(df)),
            "unique_drugs_by_id": int(df["drug_id"].nunique()),
            "unique_drugs_by_smiles": int(df["smiles"].nunique()),
            "unique_targets_by_id": int(df["target_id"].nunique()),
            "unique_targets_by_sequence": int(df["sequence"].nunique()),
            "shared_sequences_count": int(len(shared_seq)),
            "shared_sequence_target_ids_count": int(sum(len(v) for v in shared_seq)),
            "duplicate_groups_count": int(len(dup_groups)),
            "duplicate_rows_count": int(dup_mask.sum()),
            "differing_affinity_groups_count": int(len(diff_groups)),
            "differing_affinity_rows_count": int(diff_mask.sum()),
            "mse_floor_all_pairs": float(sq_err.mean()),
            "rmse_floor_all_pairs": float(math.sqrt(float(sq_err.mean()))),
            "mse_floor_duplicate_rows": float(sq_err[dup_mask].mean()) if dup_mask.any() else 0.0,
            "mse_floor_differing_rows": float(sq_err[diff_mask].mean()) if diff_mask.any() else 0.0,
        }
    return out


def compute_davis_uncensored_baselines(data_root: Path) -> dict[str, Any]:
    df = pd.read_csv(data_root / "data" / "processed" / "davis_clean.csv")
    splits_dir = data_root / "data" / "processed" / "splits"
    results: dict[str, Any] = {}

    specs = [
        ("v1_seeds", "random", [f"davis_random_seed{i}.json" for i in range(5)]),
        ("v1_seeds", "cold_drug", [f"davis_cold_drug_seed{i}.json" for i in range(5)]),
        ("v1_seeds", "cold_target", [f"davis_cold_target_seed{i}.json" for i in range(5)]),
        ("v2_kfold", "cold_drug", [f"davis_cold_drug_fold{i}.json" for i in range(5)]),
        ("v2_kfold", "cold_target", [f"davis_cold_target_fold{i}.json" for i in range(5)]),
    ]
    for proto, split_name, files in specs:
        per_unit: list[dict[str, Any]] = []
        for fname in files:
            sp = json.loads((splits_dir / fname).read_text(encoding="utf-8"))
            tr_df = df.iloc[sp["train"]]
            te_df = df.iloc[sp["test"]]
            te_gt5 = te_df[np.abs(te_df["affinity"] - 5.0) > 1e-4]
            y_te_gt5 = te_gt5["affinity"].to_numpy(dtype=np.float64)

            global_mean = float(tr_df["affinity"].mean())
            tr_gt5 = tr_df[np.abs(tr_df["affinity"] - 5.0) > 1e-4]
            active_mean = float(tr_gt5["affinity"].mean()) if len(tr_gt5) > 0 else global_mean

            drug_means = tr_df.groupby("smiles")["affinity"].mean().to_dict()
            target_means = tr_df.groupby("sequence")["affinity"].mean().to_dict()

            pred_global = np.full(len(te_gt5), global_mean, dtype=np.float64)
            pred_active = np.full(len(te_gt5), active_mean, dtype=np.float64)
            pred_drug = np.array([drug_means.get(s, global_mean) for s in te_gt5["smiles"]], dtype=np.float64)
            pred_target = np.array([target_means.get(s, global_mean) for s in te_gt5["sequence"]], dtype=np.float64)

            per_unit.append(
                {
                    "split_file": fname,
                    "num_uncensored_pairs": int(len(te_gt5)),
                    "mean_predictor_full_train": _compute_metrics_lite(y_te_gt5, pred_global),
                    "mean_predictor_active_train": _compute_metrics_lite(y_te_gt5, pred_active),
                    "drug_mean_full_train": _compute_metrics_lite(y_te_gt5, pred_drug),
                    "target_mean_full_train": _compute_metrics_lite(y_te_gt5, pred_target),
                }
            )
        summary: dict[str, Any] = {}
        for bl in (
            "mean_predictor_full_train",
            "mean_predictor_active_train",
            "drug_mean_full_train",
            "target_mean_full_train",
        ):
            summary[bl] = {
                m: _mean_std_dict([u[bl][m] for u in per_unit])
                for m in ("ci", "mse", "rmse", "pearson")
            }
        results[f"{proto}:{split_name}"] = {"per_unit": per_unit, "aggregated": summary}
    return results


def _precompute_similarity_matrices(df: pd.DataFrame) -> dict[str, Any]:
    uniq_smiles = sorted(df["smiles"].unique())
    smi_to_idx = {s: i for i, s in enumerate(uniq_smiles)}
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in uniq_smiles]
    n_s = len(uniq_smiles)
    tanimoto_mat = np.zeros((n_s, n_s), dtype=np.float64)
    for i in range(n_s):
        tanimoto_mat[i] = DataStructs.BulkTanimotoSimilarity(fps[i], fps)

    uniq_seqs = sorted(df["sequence"].unique())
    seq_to_idx = {s: i for i, s in enumerate(uniq_seqs)}
    vocab: dict[str, int] = {}
    rows: list[Counter[str]] = []
    for s in uniq_seqs:
        sl = s.lower()
        c = Counter(sl[i : i + 3] for i in range(len(sl) - 2))
        rows.append(c)
        for k in c:
            if k not in vocab:
                vocab[k] = len(vocab)
    mat = np.zeros((len(uniq_seqs), len(vocab)), dtype=np.float64)
    for r_idx, c in enumerate(rows):
        for k, v in c.items():
            mat[r_idx, vocab[k]] = v
    norms = np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-12)
    mat_norm = mat / norms
    kmer_cos_mat = mat_norm @ mat_norm.T

    return {
        "smi_to_idx": smi_to_idx,
        "tanimoto_mat": tanimoto_mat,
        "seq_to_idx": seq_to_idx,
        "kmer_cos_mat": kmer_cos_mat,
    }


def _eval_marginals_and_knn_split(
    df: pd.DataFrame,
    train_idx: list[int],
    test_idx: list[int],
    sim_cache: dict[str, Any],
    k: int = 5,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    tr_df = df.iloc[train_idx]
    te_df = df.iloc[test_idx]
    y_train = tr_df["affinity"].to_numpy(dtype=np.float64)
    y_test = te_df["affinity"].to_numpy(dtype=np.float64)
    global_mean = float(np.mean(y_train))

    drug_means = tr_df.groupby("smiles")["affinity"].mean().to_dict()
    target_means = tr_df.groupby("sequence")["affinity"].mean().to_dict()

    pred_drug = np.array([drug_means.get(s, global_mean) for s in te_df["smiles"]], dtype=np.float64)
    pred_target = np.array([target_means.get(s, global_mean) for s in te_df["sequence"]], dtype=np.float64)

    smi_to_idx = sim_cache["smi_to_idx"]
    seq_to_idx = sim_cache["seq_to_idx"]
    tanimoto_mat = sim_cache["tanimoto_mat"]
    kmer_cos_mat = sim_cache["kmer_cos_mat"]

    # Indexação rápida para candidatos de treino
    tr_by_target: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    tmp_t: dict[int, tuple[list[int], list[float]]] = defaultdict(lambda: ([], []))
    tr_by_drug: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    tmp_d: dict[int, tuple[list[int], list[float]]] = defaultdict(lambda: ([], []))

    tr_smi_idx = [smi_to_idx[s] for s in tr_df["smiles"]]
    tr_seq_idx = [seq_to_idx[s] for s in tr_df["sequence"]]
    for d_i, t_i, aff in zip(tr_smi_idx, tr_seq_idx, y_train):
        tmp_t[t_i][0].append(d_i)
        tmp_t[t_i][1].append(float(aff))
        tmp_d[d_i][0].append(t_i)
        tmp_d[d_i][1].append(float(aff))

    for t_i, (ds, afs) in tmp_t.items():
        tr_by_target[t_i] = (np.asarray(ds, dtype=np.int64), np.asarray(afs, dtype=np.float64))
    for d_i, (ts, afs) in tmp_d.items():
        tr_by_drug[d_i] = (np.asarray(ts, dtype=np.int64), np.asarray(afs, dtype=np.float64))

    pred_knn_drug = np.empty(len(te_df), dtype=np.float64)
    pred_knn_target = np.empty(len(te_df), dtype=np.float64)

    te_smis = list(te_df["smiles"])
    te_seqs = list(te_df["sequence"])
    for idx, (d_s, t_s) in enumerate(zip(te_smis, te_seqs)):
        d_i = smi_to_idx[d_s]
        t_i = seq_to_idx[t_s]

        # 1. kNN Tanimoto drug
        if t_i in tr_by_target:
            cand_d, cand_a = tr_by_target[t_i]
            sims = tanimoto_mat[d_i, cand_d]
            top = np.argsort(sims, kind="stable")[::-1][:k]
            w = sims[top]
            a = cand_a[top]
            ws = float(np.sum(w))
            pred_knn_drug[idx] = float(np.sum(w * a) / ws) if ws > 1e-6 else float(np.mean(a))
        else:
            pred_knn_drug[idx] = drug_means.get(d_s, global_mean)

        # 2. kNN 3-mer target
        if d_i in tr_by_drug:
            cand_t, cand_a = tr_by_drug[d_i]
            sims = kmer_cos_mat[t_i, cand_t]
            top = np.argsort(sims, kind="stable")[::-1][:k]
            w = sims[top]
            a = cand_a[top]
            ws = float(np.sum(w))
            pred_knn_target[idx] = float(np.sum(w * a) / ws) if ws > 1e-6 else float(np.mean(a))
        else:
            pred_knn_target[idx] = target_means.get(t_s, global_mean)

    marginals = {
        "drug_mean_predictor": _compute_metrics_lite(y_test, pred_drug),
        "target_mean_predictor": _compute_metrics_lite(y_test, pred_target),
    }
    knn = {
        "knn_tanimoto_drug": _compute_metrics_lite(y_test, pred_knn_drug),
        "knn_kmer_target": _compute_metrics_lite(y_test, pred_knn_target),
    }
    return marginals, knn


def compute_v1_marginals_knn_and_paired(data_root: Path) -> dict[str, Any]:
    dfs = {
        ds: pd.read_csv(data_root / "data" / "processed" / f"{ds}_clean.csv")
        for ds in ("davis", "kiba")
    }
    sim_caches = {ds: _precompute_similarity_matrices(df) for ds, df in dfs.items()}
    splits_dir = data_root / "data" / "processed" / "splits"
    runs_dir = data_root / "outputs" / "results" / "runs"

    # Primeiro calcula baselines de split (independem do modelo neural)
    split_baselines: dict[str, list[dict[str, Any]]] = {}
    for ds in ("davis", "kiba"):
        for sp_name in ("random", "cold_drug", "cold_target"):
            units: list[dict[str, Any]] = []
            for seed in range(5):
                sp_file = splits_dir / f"{ds}_{sp_name}_seed{seed}.json"
                sp_data = json.loads(sp_file.read_text(encoding="utf-8"))
                marg, knn = _eval_marginals_and_knn_split(
                    dfs[ds], sp_data["train"], sp_data["test"], sim_caches[ds], k=5
                )
                units.append({"seed": seed, "marginal_baselines": marg, "knn_baselines": knn})
            split_baselines[f"{ds}:{sp_name}"] = units

    # Agora calcula comparação pareada por seed contra cada modelo executado em v1_seeds
    groups = [
        ("graphdta_cnn3", "davis", "cold_drug"),
        ("graphdta_cnn3", "davis", "cold_target"),
        ("graphdta_cnn3", "davis", "random"),
        ("graphdta_gin", "davis", "cold_drug"),
        ("graphdta_gin", "davis", "cold_target"),
        ("graphdta_gin", "davis", "random"),
        ("graphdta_gin", "kiba", "cold_drug"),
        ("graphdta_gin", "kiba", "cold_target"),
        ("graphdta_gin", "kiba", "random"),
    ]
    paired_groups: dict[str, Any] = {}
    for model, ds, sp_name in groups:
        key = f"{model}:{ds}:{sp_name}"
        sb_units = split_baselines[f"{ds}:{sp_name}"]

        # Escolhe baseline marginal relevante
        if sp_name == "cold_drug":
            rel_name = "target_mean_predictor"
        elif sp_name == "cold_target":
            rel_name = "drug_mean_predictor"
        else:
            # Em random, seleciona o menor MSE médio entre drug_mean e target_mean
            dm_mse = float(np.mean([u["marginal_baselines"]["drug_mean_predictor"]["mse"] for u in sb_units]))
            tm_mse = float(np.mean([u["marginal_baselines"]["target_mean_predictor"]["mse"] for u in sb_units]))
            rel_name = "drug_mean_predictor" if dm_mse <= tm_mse else "target_mean_predictor"

        per_seed: list[dict[str, Any]] = []
        d_mse_list: list[float] = []
        d_ci_list: list[float] = []
        wins_mse = 0
        wins_ci = 0
        for seed in range(5):
            run_data = json.loads((runs_dir / f"{model}_{ds}_{sp_name}_seed{seed}.json").read_text(encoding="utf-8"))
            tm = run_data["test_metrics_full"]
            bl_rel = sb_units[seed]["marginal_baselines"][rel_name]
            d_mse = float(tm["mse"] - bl_rel["mse"])
            d_ci = float(tm["ci"] - bl_rel["ci"])
            if d_mse < 0:
                wins_mse += 1
            if d_ci > 0:
                wins_ci += 1
            d_mse_list.append(d_mse)
            d_ci_list.append(d_ci)
            per_seed.append(
                {
                    "seed": seed,
                    "model_ci": float(tm["ci"]),
                    "model_mse": float(tm["mse"]),
                    "relevant_baseline": rel_name,
                    "baseline_ci": float(bl_rel["ci"]),
                    "baseline_mse": float(bl_rel["mse"]),
                    "delta_ci": d_ci,
                    "delta_mse": d_mse,
                }
            )

        agg_marg = {
            mname: {
                met: _mean_std_dict([u["marginal_baselines"][mname][met] for u in sb_units])
                for met in ("ci", "mse", "rmse", "pearson", "spearman", "rm2")
            }
            for mname in ("drug_mean_predictor", "target_mean_predictor")
        }
        agg_knn = {
            kname: {
                met: _mean_std_dict([u["knn_baselines"][kname][met] for u in sb_units])
                for met in ("ci", "mse", "rmse", "pearson", "spearman", "rm2")
            }
            for kname in ("knn_tanimoto_drug", "knn_kmer_target")
        }
        paired_groups[key] = {
            "relevant_baseline": rel_name,
            "n_seeds": 5,
            "marginal_baselines": agg_marg,
            "knn_baselines": agg_knn,
            "delta_mse_paired": _mean_std_dict(d_mse_list),
            "delta_ci_paired": _mean_std_dict(d_ci_list),
            "wins_mse": wins_mse,
            "wins_ci": wins_ci,
            "per_seed": per_seed,
        }

    return {"split_baselines": split_baselines, "paired_groups": paired_groups}


_WORKER_SEQS: list[str] = []
_WORKER_ALIGNER: PairwiseAligner | None = None


def _init_align_worker(seqs: list[str]) -> None:
    global _WORKER_SEQS, _WORKER_ALIGNER
    _WORKER_SEQS = seqs
    blosum62 = substitution_matrices.load("BLOSUM62")
    _WORKER_ALIGNER = PairwiseAligner(
        mode="global",
        substitution_matrix=blosum62,
        open_gap_score=-10.0,
        extend_gap_score=-0.5,
    )


def _align_pair_worker(pair: tuple[int, int]) -> tuple[int, int, float]:
    i, j = pair
    si = _WORKER_SEQS[i]
    sj = _WORKER_SEQS[j]
    max_l = len(si) if len(si) >= len(sj) else len(sj)
    aln = _WORKER_ALIGNER.align(si, sj)[0]  # type: ignore[union-attr]
    ident = float(aln.counts().identities) / float(max_l)
    return i, j, ident


def compute_davis_seqid_clusters(data_root: Path) -> dict[str, Any]:
    df = pd.read_csv(data_root / "data" / "processed" / "davis_clean.csv")
    uniq_seqs = sorted(df["sequence"].unique())
    n = len(uniq_seqs)
    lens = [len(s) for s in uniq_seqs]

    # Pré-filtro exato matemático: nem razão de comprimentos nem interseção multiset de AA pode ser < 0.30
    aa_counters = [Counter(s) for s in uniq_seqs]
    candidate_pairs: list[tuple[int, int]] = []
    for i in range(n):
        li = lens[i]
        ci = aa_counters[i]
        for j in range(i + 1, n):
            lj = lens[j]
            max_l = li if li >= lj else lj
            min_l = lj if li >= lj else li
            if (min_l / max_l) < 0.30:
                continue
            max_matches = sum(min(cnt, aa_counters[j][aa]) for aa, cnt in ci.items() if aa in aa_counters[j])
            if (max_matches / max_l) < 0.30:
                continue
            candidate_pairs.append((i, j))

    t0 = time.perf_counter()
    adj_30: list[list[int]] = [[] for _ in range(n)]
    adj_40: list[list[int]] = [[] for _ in range(n)]
    adj_50: list[list[int]] = [[] for _ in range(n)]
    n_workers = max(1, min(os.cpu_count() or 4, 8))
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_align_worker, initargs=(uniq_seqs,)) as pool:
        for i, j, ident in pool.map(_align_pair_worker, candidate_pairs, chunksize=256):
            if ident >= 0.30:
                adj_30[i].append(j)
                adj_30[j].append(i)
            if ident >= 0.40:
                adj_40[i].append(j)
                adj_40[j].append(i)
            if ident >= 0.50:
                adj_50[i].append(j)
                adj_50[j].append(i)
    aligned_pairs = len(candidate_pairs)
    elapsed = time.perf_counter() - t0

    seq_pair_counts = df.groupby("sequence").size().to_dict()
    seq_tid_counts = df.groupby("sequence")["target_id"].nunique().to_dict()

    thresholds_out: dict[str, Any] = {}
    for label, adj in (("id_ge_30pct", adj_30), ("id_ge_40pct", adj_40), ("id_ge_50pct", adj_50)):
        visited = [False] * n
        clusters: list[list[int]] = []
        for i in range(n):
            if visited[i]:
                continue
            comp: list[int] = []
            stack = [i]
            visited[i] = True
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            clusters.append(comp)

        # Ordena clusters do maior para menor em número de pares para balanceamento greedy 5-fold (GroupKFold)
        cluster_infos = []
        for cid, comp in enumerate(clusters):
            seqs = [uniq_seqs[idx] for idx in comp]
            n_pairs = sum(int(seq_pair_counts[s]) for s in seqs)
            n_tids = sum(int(seq_tid_counts[s]) for s in seqs)
            cluster_infos.append((n_pairs, len(comp), n_tids, cid, comp))
        cluster_infos.sort(key=lambda x: (-x[0], -x[1], x[3]))

        folds: list[dict[str, int]] = [
            {"pairs": 0, "unique_seqs": 0, "target_ids": 0, "clusters": 0} for _ in range(5)
        ]
        for n_pairs, n_seqs, n_tids, _, _ in cluster_infos:
            best_f = min(range(5), key=lambda f_i: (folds[f_i]["pairs"], folds[f_i]["unique_seqs"], f_i))
            folds[best_f]["pairs"] += n_pairs
            folds[best_f]["unique_seqs"] += n_seqs
            folds[best_f]["target_ids"] += n_tids
            folds[best_f]["clusters"] += 1

        # Monta partição train (3 folds) / val (1 fold) / test (1 fold) equivalente ao Protocolo v2
        partitions = []
        total_pairs = len(df)
        for test_f in range(5):
            val_f = (test_f + 1) % 5
            tr_fs = [f for f in range(5) if f not in (test_f, val_f)]
            partitions.append(
                {
                    "fold": test_f,
                    "train_pairs": sum(folds[f]["pairs"] for f in tr_fs),
                    "val_pairs": folds[val_f]["pairs"],
                    "test_pairs": folds[test_f]["pairs"],
                    "train_unique_seqs": sum(folds[f]["unique_seqs"] for f in tr_fs),
                    "val_unique_seqs": folds[val_f]["unique_seqs"],
                    "test_unique_seqs": folds[test_f]["unique_seqs"],
                    "train_target_ids": sum(folds[f]["target_ids"] for f in tr_fs),
                    "val_target_ids": folds[val_f]["target_ids"],
                    "test_target_ids": folds[test_f]["target_ids"],
                }
            )
        assert sum(p["test_pairs"] for p in partitions) == total_pairs

        thresholds_out[label] = {
            "num_clusters": len(clusters),
            "singleton_clusters": int(sum(1 for c in clusters if len(c) == 1)),
            "largest_cluster_unique_seqs": int(max(len(c) for c in clusters)),
            "largest_cluster_pairs": int(max(ci[0] for ci in cluster_infos)),
            "fold_cluster_totals": folds,
            "proposed_5fold_partitions": partitions,
        }

    return {
        "tool": "biopython",
        "version": str(Bio.__version__),
        "method": "PairwiseAligner(mode='global', matrix='BLOSUM62', open_gap=-10.0, extend_gap=-0.5) exact_matches / max(len_a, len_b) + connected components + greedy LPT 5-fold balancing",
        "num_unique_sequences": n,
        "aligned_pairs_computed": aligned_pairs,
        "elapsed_sec": round(elapsed, 2),
        "thresholds": thresholds_out,
    }


def compute_kiba_random_clean_timings(data_root: Path) -> dict[str, Any]:
    runs_dir = data_root / "outputs" / "results" / "runs"
    non_resumed = [0, 2, 4]
    sec_ep = []
    tot_ep = []
    elap = []
    for s in non_resumed:
        d = json.loads((runs_dir / f"graphdta_gin_kiba_random_seed{s}.json").read_text(encoding="utf-8"))
        sec_ep.append(float(d["sec_per_epoch"]))
        tot_ep.append(float(d["total_epochs_run"]))
        elap.append(float(d["elapsed_sec"]))
    resumed_sec_ep = []
    for s in (1, 3):
        d = json.loads((runs_dir / f"graphdta_gin_kiba_random_seed{s}.json").read_text(encoding="utf-8"))
        resumed_sec_ep.append(float(d["sec_per_epoch"]))
    return {
        "non_resumed_seeds": non_resumed,
        "resumed_seeds": [1, 3],
        "non_resumed_sec_per_epoch": _mean_std_dict(sec_ep),
        "non_resumed_total_epochs": _mean_std_dict(tot_ep),
        "non_resumed_elapsed_sec_sum": float(sum(elap)),
        "resumed_sec_per_epoch_artifacts_distorted": resumed_sec_ep,
    }


def run_cpu_audits() -> Path:
    repo_root, data_root = resolve_roots()
    out_path = repo_root / "docs" / "paper" / "cpu_audits.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "noise_floor": compute_noise_floor(data_root),
        "davis_uncensored_baselines": compute_davis_uncensored_baselines(data_root),
        "v1_marginals_knn_and_paired": compute_v1_marginals_knn_and_paired(data_root),
        "kiba_random_clean_timings": compute_kiba_random_clean_timings(data_root),
        "davis_seqid_clusters": compute_davis_seqid_clusters(data_root),
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    path = run_cpu_audits()
    print(f"OK: {path} ({path.stat().st_size} bytes)")
