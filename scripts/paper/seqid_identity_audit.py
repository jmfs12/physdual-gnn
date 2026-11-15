"""Auditoria CPU das métricas de identidade de sequência nos splits `seqid` (Davis e KIBA).

Executa estritamente em modo somente leitura sobre `data/processed/*_clean.csv`,
`data/processed/splits/*.json` e `outputs/logs/*_seqid_*.json`.

Calcula para cada par `(seq_a, seq_b)` dentro de cada dataset:
  1. Identidade global BLOSUM62 (`exact_matches / max(len_a, len_b)`) — métrica do split original.
  2. Identidade global normalizada pela sequência menor (`exact_matches / min(len_a, len_b)`).
  3. Identidade local máxima Smith-Waterman (`mode='local'`, `BLOSUM62`, `open_gap=-10.0`, `extend_gap=-0.5`):
     - `sw_id_min_len = exact_matches_local / min(len_a, len_b)`
     - `sw_id_aln_len = exact_matches_local / aln_length` (reportada também com filtros de comprimento alinhado `>= 50` aa e `>= 150` aa para distinguir homologia de domínio de fragmentos curtos).

Para cada dobra `0..4` de `cold_target_seqid30`, `cold_target_seqid40` (e `cold_target` como referência),
computa para cada proteína de teste `q` o máximo de identidade contra todas as proteínas de treino `r`,
reportando distribuição completa e fração/contagem de proteínas de teste com homólogo em treino `>= 30%`, `>= 40%`, `>= 50%` e `>= 70%`.

Saída: `docs/paper/seqid_identity_audit.json`
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PAPER_SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAPER_SCRIPTS_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
for p in (str(PAPER_SCRIPTS_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common import resolve_roots  # noqa: E402
from physdual.data.splits import load_splits  # noqa: E402

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


def _align_pair_worker(pair: tuple[int, int]) -> tuple[int, int, float, float, float, float, int]:
    i, j = pair
    si = _WORKER_SEQS[i]
    sj = _WORKER_SEQS[j]
    li = len(si)
    lj = len(sj)
    max_l = li if li >= lj else lj
    min_l = lj if li >= lj else li

    aln_gl = next(iter(_GL_ALIGNER.align(si, sj)))
    gl_ident = int(aln_gl.counts().identities)
    gl_id_max = gl_ident / float(max_l) if max_l > 0 else 0.0
    gl_id_min = gl_ident / float(min_l) if min_l > 0 else 0.0

    aln_sw = next(iter(_SW_ALIGNER.align(si, sj)))
    sw_ident = int(aln_sw.counts().identities)
    sw_len = int(aln_sw.length)
    sw_id_min = sw_ident / float(min_l) if min_l > 0 else 0.0
    sw_id_aln = sw_ident / float(sw_len) if sw_len > 0 else 0.0
    return i, j, gl_id_max, gl_id_min, sw_id_min, sw_id_aln, sw_len


def _compute_pairwise_matrices(
    seqs: list[str],
    n_workers: int,
    part: str = "all",
    cache_tag: str = "ds",
    cache_dir: Path | None = None,
) -> dict[str, np.ndarray] | None:
    n = len(seqs)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    t1 = len(pairs) // 3
    t2 = (2 * len(pairs)) // 3
    cdir = cache_dir if cache_dir is not None else Path(tempfile.gettempdir())
    cdir.mkdir(parents=True, exist_ok=True)
    cache1 = cdir / f"physdual_{cache_tag}_part1.npz"
    cache2 = cdir / f"physdual_{cache_tag}_part2.npz"

    if part == "part1":
        sub_pairs = pairs[:t1]
    elif part == "part2":
        sub_pairs = pairs[t1:t2]
    elif part == "part3":
        sub_pairs = pairs[t2:]
    else:
        sub_pairs = pairs

    chunksize = max(64, min(1024, len(sub_pairs) // max(1, n_workers * 4)))
    if n_workers <= 1:
        _init_worker(seqs)
        results = [_align_pair_worker(p) for p in sub_pairs]
    else:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(seqs,),
        ) as pool:
            results = list(pool.map(_align_pair_worker, sub_pairs, chunksize=chunksize))

    if part == "part1":
        np.savez(cache1, data=np.asarray(results, dtype=np.float64))
        print(f"  [cache] part1 salvo em {cache1} ({len(results)} pares)", flush=True)
        return None
    if part == "part2":
        np.savez(cache2, data=np.asarray(results, dtype=np.float64))
        print(f"  [cache] part2 salvo em {cache2} ({len(results)} pares)", flush=True)
        return None

    if part == "part3":
        if not cache1.exists() or not cache2.exists():
            raise FileNotFoundError(f"Cache part1/part2 ausente: {cache1}, {cache2}")
        p1 = np.load(cache1)["data"]
        p2 = np.load(cache2)["data"]
        prev = np.vstack([p1, p2])
        all_rows = [(int(r[0]), int(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]), int(r[6])) for r in prev]
        all_rows.extend(results)
        results = all_rows

    gl_max = np.eye(n, dtype=np.float32)
    gl_min = np.eye(n, dtype=np.float32)
    sw_min = np.eye(n, dtype=np.float32)
    sw_aln = np.eye(n, dtype=np.float32)
    sw_len = np.zeros((n, n), dtype=np.int32)
    for k in range(n):
        sw_len[k, k] = len(seqs[k])

    for i, j, gmax, gmin, smin, saln, slen in results:
        gl_max[i, j] = gl_max[j, i] = gmax
        gl_min[i, j] = gl_min[j, i] = gmin
        sw_min[i, j] = sw_min[j, i] = smin
        sw_aln[i, j] = sw_aln[j, i] = saln
        sw_len[i, j] = sw_len[j, i] = slen

    return {
        "global_id_max": gl_max,
        "global_id_min": gl_min,
        "sw_id_min_len": sw_min,
        "sw_id_aln_len": sw_aln,
        "sw_aln_len": sw_len,
    }


def _summarize_max_vals(vals: np.ndarray) -> dict[str, Any]:
    if vals.size == 0:
        return {"n": 0}
    arr = vals.astype(np.float64)
    n = int(arr.size)
    return {
        "n": n,
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if n > 1 else 0.0,
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
        "count_gt_30": int(np.sum(arr > 0.30)),
        "pct_gt_30": float(100.0 * np.mean(arr > 0.30)),
        "count_ge_30": int(np.sum(arr >= 0.30)),
        "pct_ge_30": float(100.0 * np.mean(arr >= 0.30)),
        "count_gt_40": int(np.sum(arr > 0.40)),
        "pct_gt_40": float(100.0 * np.mean(arr > 0.40)),
        "count_ge_40": int(np.sum(arr >= 0.40)),
        "pct_ge_40": float(100.0 * np.mean(arr >= 0.40)),
        "count_ge_50": int(np.sum(arr >= 0.50)),
        "pct_ge_50": float(100.0 * np.mean(arr >= 0.50)),
        "count_ge_70": int(np.sum(arr >= 0.70)),
        "pct_ge_70": float(100.0 * np.mean(arr >= 0.70)),
    }


def _audit_dataset(
    data_root: Path,
    ds: str,
    n_workers: int,
    part: str = "all",
    cache_dir: Path | None = None,
) -> dict[str, Any] | None:
    t0 = time.perf_counter()
    df = pd.read_csv(data_root / "data" / "processed" / f"{ds}_clean.csv")
    seqs = sorted(set(df["sequence"].astype(str)))
    seq_to_idx = {s: i for i, s in enumerate(seqs)}
    seq_lens = [len(s) for s in seqs]

    print(f"[{ds}] Computing pairwise global + Smith-Waterman alignments ({part}) for {len(seqs)} unique seqs ({len(seqs)*(len(seqs)-1)//2} total pairs) with {n_workers} workers...", flush=True)
    mats = _compute_pairwise_matrices(seqs, n_workers=n_workers, part=part, cache_tag=ds, cache_dir=cache_dir)
    if mats is None:
        return None
    print(f"[{ds}] Pairwise matrices done in {time.perf_counter()-t0:.1f}s.", flush=True)

    cmap = json.loads((data_root / "outputs" / "logs" / f"{ds}_seqid_cluster_map.json").read_text(encoding="utf-8"))
    summ = json.loads((data_root / "outputs" / "logs" / f"{ds}_seqid_splits_summary.json").read_text(encoding="utf-8"))

    cluster_stats: dict[str, Any] = {}
    for th_name, c_obj in cmap.get("thresholds", {}).items():
        if "seq_to_cluster" in c_obj:
            counts = sorted(Counter(c_obj["seq_to_cluster"].values()).values(), reverse=True)
        else:
            counts = sorted([len(c) for c in c_obj.get("clusters", [])], reverse=True)
        arr_c = np.asarray(counts, dtype=np.float64)
        s_folds = summ.get("thresholds", {}).get(th_name, {})
        cluster_stats[th_name] = {
            "num_clusters": int(len(counts)),
            "singleton_clusters": int(sum(1 for c in counts if c == 1)),
            "min_seqs": int(min(counts)),
            "p25_seqs": float(np.percentile(arr_c, 25)),
            "median_seqs": float(np.median(arr_c)),
            "mean_seqs": float(np.mean(arr_c)),
            "p75_seqs": float(np.percentile(arr_c, 75)),
            "p90_seqs": float(np.percentile(arr_c, 90)),
            "max_seqs": int(max(counts)),
            "top10_cluster_sizes_seqs": counts[:10],
            "largest_cluster_pairs": s_folds.get("largest_cluster_pairs"),
        }

    splits_dir = data_root / "data" / "processed" / "splits"
    regimes_out: dict[str, Any] = {}
    split_specs = [("cold_target_seqid30", "fold"), ("cold_target_seqid40", "fold")]
    if ds == "davis":
        split_specs.append(("cold_target", "fold"))
    else:
        split_specs.append(("cold_target", "seed"))

    for sp, uprefix in split_specs:
        folds_report: list[dict[str, Any]] = []
        pool_vals: dict[str, list[np.ndarray]] = {
            "global_id_max": [],
            "global_id_min": [],
            "sw_id_min_len": [],
            "sw_id_aln_len_ge50aa": [],
            "sw_id_aln_len_ge150aa": [],
        }
        for u in range(5):
            sp_path = splits_dir / f"{ds}_{sp}_{uprefix}{u}.json"
            if not sp_path.exists():
                continue
            sp_idx = load_splits(sp_path)
            tr_seqs = sorted(set(df.iloc[sp_idx["train"]]["sequence"].astype(str)))
            te_seqs = sorted(set(df.iloc[sp_idx["test"]]["sequence"].astype(str)))
            tr_idx = np.asarray([seq_to_idx[s] for s in tr_seqs], dtype=np.intp)
            te_idx = np.asarray([seq_to_idx[s] for s in te_seqs], dtype=np.intp)

            sub_gl_max = mats["global_id_max"][np.ix_(te_idx, tr_idx)]
            sub_gl_min = mats["global_id_min"][np.ix_(te_idx, tr_idx)]
            sub_sw_min = mats["sw_id_min_len"][np.ix_(te_idx, tr_idx)]
            sub_sw_aln = mats["sw_id_aln_len"][np.ix_(te_idx, tr_idx)]
            sub_sw_len = mats["sw_aln_len"][np.ix_(te_idx, tr_idx)]

            v_gl_max = np.max(sub_gl_max, axis=1)
            v_gl_min = np.max(sub_gl_min, axis=1)
            v_sw_min = np.max(sub_sw_min, axis=1)

            masked_50 = np.where(sub_sw_len >= 50, sub_sw_aln, 0.0)
            v_sw_aln50 = np.max(masked_50, axis=1)
            masked_150 = np.where(sub_sw_len >= 150, sub_sw_aln, 0.0)
            v_sw_aln150 = np.max(masked_150, axis=1)

            best_sw_min_j = np.argmax(sub_sw_min, axis=1)
            best_sw_min_alnlen = sub_sw_len[np.arange(len(te_idx)), best_sw_min_j]

            pool_vals["global_id_max"].append(v_gl_max)
            pool_vals["global_id_min"].append(v_gl_min)
            pool_vals["sw_id_min_len"].append(v_sw_min)
            pool_vals["sw_id_aln_len_ge50aa"].append(v_sw_aln50)
            pool_vals["sw_id_aln_len_ge150aa"].append(v_sw_aln150)

            folds_report.append({
                "unit": u,
                "n_train_seqs": int(len(tr_seqs)),
                "n_test_seqs": int(len(te_seqs)),
                "global_id_max": _summarize_max_vals(v_gl_max),
                "global_id_min": _summarize_max_vals(v_gl_min),
                "sw_id_min_len": _summarize_max_vals(v_sw_min),
                "sw_id_aln_len_ge50aa": _summarize_max_vals(v_sw_aln50),
                "sw_id_aln_len_ge150aa": _summarize_max_vals(v_sw_aln150),
                "best_sw_min_aln_len_median_aa": float(np.median(best_sw_min_alnlen)),
            })

        pooled_summary = {
            k: _summarize_max_vals(np.concatenate(vlist)) if vlist else {}
            for k, vlist in pool_vals.items()
        }
        regimes_out[sp] = {
            "unit_type": uprefix,
            "folds": folds_report,
            "across_all_units_test_proteins": pooled_summary,
        }

    return {
        "n_unique_seqs": len(seqs),
        "seq_len_min": int(min(seq_lens)),
        "seq_len_median": float(np.median(seq_lens)),
        "seq_len_max": int(max(seq_lens)),
        "cluster_size_distributions": cluster_stats,
        "regimes": regimes_out,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Auditoria CPU de identidade global/local (Smith-Waterman) nos splits seqid")
    parser.add_argument("--workers", type=int, default=max(1, min(os.cpu_count() or 4, 20)))
    parser.add_argument("--step", choices=("all", "kiba", "davis_part1", "davis_part2", "davis_part3"), default="all")
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args(argv)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    t0 = time.perf_counter()
    repo_root, data_root = resolve_roots()
    out_path = repo_root / "docs" / "paper" / "seqid_identity_audit.json"

    existing: dict[str, Any] = {}
    if out_path.exists() and args.step != "all":
        existing = json.loads(out_path.read_text(encoding="utf-8"))
    datasets_out: dict[str, Any] = existing.get("datasets", {})

    if args.step in ("all", "kiba"):
        datasets_out["kiba"] = _audit_dataset(data_root, "kiba", n_workers=args.workers, part="all", cache_dir=cache_dir)
    if args.step == "all":
        datasets_out["davis"] = _audit_dataset(data_root, "davis", n_workers=args.workers, part="all", cache_dir=cache_dir)
    elif args.step == "davis_part1":
        _audit_dataset(data_root, "davis", n_workers=args.workers, part="part1", cache_dir=cache_dir)
        print(f"[OK] davis_part1 concluído em {time.perf_counter() - t0:.1f}s")
        return 0
    elif args.step == "davis_part2":
        _audit_dataset(data_root, "davis", n_workers=args.workers, part="part2", cache_dir=cache_dir)
        print(f"[OK] davis_part2 concluído em {time.perf_counter() - t0:.1f}s")
        return 0
    elif args.step == "davis_part3":
        datasets_out["davis"] = _audit_dataset(data_root, "davis", n_workers=args.workers, part="part3", cache_dir=cache_dir)

    import Bio
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "biopython_version": Bio.__version__,
        "alignment_specs": {
            "split_original_metric": "PairwiseAligner(mode='global', matrix='BLOSUM62', open_gap=-10.0, extend_gap=-0.5); identidade = exact_matches / max(len_a, len_b)",
            "clustering_algorithm": "single_linkage_connected_components sobre pares >= tau + greedy_LPT_balancing (maiores clusters em pares primeiro em 5 dobras)",
            "global_id_min_metric": "PairwiseAligner(mode='global', matrix='BLOSUM62', open_gap=-10.0, extend_gap=-0.5); exact_matches / min(len_a, len_b)",
            "smith_waterman_local_metrics": {
                "sw_id_min_len": "PairwiseAligner(mode='local', matrix='BLOSUM62', open_gap=-10.0, extend_gap=-0.5); exact_matches_local / min(len_a, len_b)",
                "sw_id_aln_len_ge50aa": "max_{train} (exact_matches_local / aln_length) entre alinhamentos locais com aln_length >= 50 aa (0.0 se nenhum)",
                "sw_id_aln_len_ge150aa": "max_{train} (exact_matches_local / aln_length) entre alinhamentos locais com aln_length >= 150 aa (escala de domínio quinase; 0.0 se nenhum)",
            },
        },
        "datasets": datasets_out,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] seqid_identity_audit.json ({args.step}) salvo em {payload['elapsed_sec']}s: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
