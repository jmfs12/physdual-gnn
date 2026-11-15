"""Passo 1: Consolida as tabelas T1..T15 em CSV, Markdown e LaTeX em `docs/paper/tables/`."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from common import (
    NA_TEXT,
    ensure_paper_dirs,
    fmt_float,
    fmt_mean_std,
    load_json_strict,
    resolve_roots,
    sample_mean_std,
    write_table,
)


def _normalize_model_name(name: str) -> str:
    """Remove prefixo 'physdual_' para preservar nomenclatura canônica da Fase 3."""
    s = str(name)
    if s.startswith("physdual_"):
        return s[len("physdual_") :]
    return s


def _load_all_runs(data_root: Path) -> list[dict[str, Any]]:
    runs_dir = data_root / "outputs" / "results" / "runs"
    files = sorted(runs_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"Nenhum JSON de run encontrado em {runs_dir}")
    runs = []
    for f in files:
        d = load_json_strict(f)
        if not str(d.get("model", "")).startswith("graphdta_"):
            continue
        if "seqid" in str(d.get("split", "")):
            continue
        d["_filename"] = f.name
        d["_proto"] = "v2_kfold" if d.get("fold") is not None else "v1_seeds"
        d["_unit"] = int(d["fold"]) if d["_proto"] == "v2_kfold" else int(d["seed"])
        runs.append(d)
    return runs


def _load_all_runs_all_phases(data_root: Path) -> list[dict[str, Any]]:
    """Carrega todos os 166 JSONs de execução das Fases 1, 2 e 3 (sem filtrar modelo/seqid)."""
    runs_dir = data_root / "outputs" / "results" / "runs"
    files = sorted(runs_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"Nenhum JSON de run encontrado em {runs_dir}")
    runs = []
    for f in files:
        d = load_json_strict(f)
        d["_filename"] = f.name
        d["_model_norm"] = _normalize_model_name(str(d.get("model", "")))
        d["_proto"] = "v2_kfold" if d.get("fold") is not None else "v1_seeds"
        d["_unit"] = int(d["fold"]) if d["_proto"] == "v2_kfold" else int(d["seed"])
        runs.append(d)
    return runs


def build_t1(repo_root: Path, data_root: Path, tables_dir: Path) -> dict[str, Path]:
    cpu_audits = load_json_strict(repo_root / "docs" / "paper" / "cpu_audits.json")
    rows = []
    for ds in ("davis", "kiba"):
        rep = load_json_strict(data_root / "data" / "processed" / f"{ds}_ingestion_report.json")
        df = pd.read_csv(data_root / "data" / "processed" / f"{ds}_clean.csv")
        nf = cpu_audits["noise_floor"][ds]

        uniq_seqs = df["sequence"].drop_duplicates()
        seq_lens = uniq_seqs.str.len()
        gt1000_seqs = int((seq_lens > 1000).sum())
        gt1000_pairs = int((df["sequence"].str.len() > 1000).sum())
        gt1022_seqs = int((seq_lens > 1022).sum())
        gt1022_pairs = int((df["sequence"].str.len() > 1022).sum())

        grp = df.groupby(["smiles", "sequence"])["affinity"].agg(["count", "nunique"])
        dups = grp[grp["count"] > 1]
        diff_dups = int((dups["nunique"] > 1).sum())

        amb = rep["ambiguous_aa_char_counts"]
        amb_str = ", ".join(f"{k}:{amb[k]}" for k in ("X", "B", "U", "Z", "O"))

        cens_cnt = int(rep["davis_censored_count"])
        cens_pct = f"{float(rep['davis_censored_pct']):.2f}%" if ds == "davis" else "0.00% (N/A)"
        shared_str = (
            f"{nf['shared_sequences_count']} seqs -> {nf['shared_sequence_target_ids_count']} target_ids"
            if nf["shared_sequences_count"] > 0
            else "0"
        )

        rows.append({
            "dataset": ds.upper(),
            "pares_brutos": int(rep["raw_num_pairs"]),
            "pares_limpos": int(rep["clean_num_pairs"]),
            "farmacos_id": int(rep["num_drugs_by_id"]),
            "farmacos_smiles_canonico": int(rep["num_drugs_by_canonical_smiles"]),
            "alvos_id": int(rep["num_targets_by_id"]),
            "alvos_sequencia_unica": int(rep["num_targets_by_sequence"]),
            "alvos_seq_compartilhadas": shared_str,
            "seqs_gt_1000aa_unicas_e_pares": f"{gt1000_seqs} ({gt1000_pairs} pares)",
            "seqs_janela_gt_1022aa_unicas_e_pares": f"{gt1022_seqs} ({gt1022_pairs} pares)",
            "smiles_alterados_rdkit": int(rep["smiles_modified_count"]),
            "smiles_falhos": int(rep["smiles_failed_count"]),
            "aa_estendidos_seqs": int(rep["ambiguous_aa_seq_count"]),
            "aa_estendidos_contagem": amb_str,
            "grupos_duplicados_conteudo": int(rep["duplicate_content_pairs_count"]),
            "duplicatas_afinidade_distinta": diff_dups,
            "piso_ruido_mse_global": fmt_float(nf["mse_floor_all_pairs"]),
            "piso_ruido_mse_linhas_distintas": fmt_float(nf["mse_floor_differing_rows"]),
            "pares_censurados_pkd_5_0": cens_cnt if ds == "davis" else 0,
            "pct_censurados": cens_pct,
        })

    df_t1 = pd.DataFrame(rows)
    return write_table(
        "T1_dataset_stats",
        "T1 — Estatísticas, Duplicatas e Piso de Ruído dos Datasets (Davis e KIBA)",
        df_t1,
        "data/processed/{davis,kiba}_ingestion_report.json, data/processed/{davis,kiba}_clean.csv e docs/paper/cpu_audits.json",
        tables_dir,
        extra_notes=[
            "Sequências com janela deslizante ESM-2 (>1022 aa): `70` únicas (`109` `target_id`s) no Davis + `42` únicas no KIBA = `112` por soma de dataset (`83` sequências únicas globalmente deduplicadas), sincronizadas em `outputs/logs/phase1_summary_report.json`.",
            "No Davis, `81` `target_id`s colapsam em apenas `18` sequências únicas idênticas em `proteins.txt` (`shared_sequences_count=18`), gerando `1.224` grupos duplicados (`5.508` linhas) e `546` grupos com afinidades distintas (`2.894` linhas); o piso irreduzível de ruído (MSE da média do grupo) é `0.0322` global (`RMSE=0.1795`) e `0.3347` nas linhas divergentes — e indica que o Davis atual não testa resiliência a mutações.",
            "No KIBA, `524` grupos duplicados (`1.172` linhas; `310` grupos/744 linhas divergentes) resultam de `2.111` `drug_id`s que colapsam em `2.065` SMILES canônicos; piso global `MSE = 0.0011` (`0.1780` nas linhas divergentes).",
            "No Davis, `smiles_modified_count = 68` decorre de recanonização sintática RDKit; `num_modified_fragments = 0` em `phase1_summary_report.json` (0 sais multifragmento removidos).",
        ],
    )


def build_t2(data_root: Path, tables_dir: Path) -> dict[str, Path]:
    repo_root = resolve_roots()[0]
    cpu_audits = load_json_strict(repo_root / "docs" / "paper" / "cpu_audits.json")
    seqid = cpu_audits["davis_seqid_clusters"]
    t30 = seqid["thresholds"]["id_ge_30pct"]
    t40 = seqid["thresholds"]["id_ge_40pct"]
    t50 = seqid["thresholds"]["id_ge_50pct"]
    f30 = ", ".join(f"{x['pairs']}/{x['unique_seqs']}" for x in t30["fold_cluster_totals"])
    f40 = ", ".join(f"{x['pairs']}/{x['unique_seqs']}" for x in t40["fold_cluster_totals"])
    f50 = ", ".join(f"{x['pairs']}/{x['unique_seqs']}" for x in t50["fold_cluster_totals"])

    dfs = {
        "davis": pd.read_csv(data_root / "data" / "processed" / "davis_clean.csv"),
        "kiba": pd.read_csv(data_root / "data" / "processed" / "kiba_clean.csv"),
    }
    rows = []
    for sp_file in sorted(
        p for p in (data_root / "data" / "processed" / "splits").glob("*.json")
        if "seqid" not in p.stem
    ):
        sp = load_json_strict(sp_file)
        name = sp_file.stem
        ds = sp.get("dataset") or ("davis" if name.startswith("davis") else "kiba")
        df = dfs[ds]
        if "fold" in name:
            proto = "v2_kfold"
            split_type = sp["split_type"]
            unit = f"fold_{sp['fold_id']}"
        else:
            proto = "v1_seeds"
            if "_random_" in name:
                split_type = "random"
            elif "_cold_drug_" in name:
                split_type = "cold_drug"
            else:
                split_type = "cold_target"
            unit = name.split("_")[-1]

        tr = df.iloc[sp["train"]]
        va = df.iloc[sp["val"]]
        te = df.iloc[sp["test"]]

        rows.append({
            "dataset": ds,
            "protocolo": proto,
            "split": split_type,
            "unidade": unit,
            "treino_pares": len(tr),
            "val_pares": len(va),
            "teste_pares": len(te),
            "treino_smiles_canon": int(tr["smiles"].nunique()),
            "val_smiles_canon": int(va["smiles"].nunique()),
            "teste_smiles_canon": int(te["smiles"].nunique()),
            "treino_drug_ids": int(tr["drug_id"].nunique()),
            "val_drug_ids": int(va["drug_id"].nunique()),
            "teste_drug_ids": int(te["drug_id"].nunique()),
            "treino_seq_unicas": int(tr["sequence"].nunique()),
            "val_seq_unicas": int(va["sequence"].nunique()),
            "teste_seq_unicas": int(te["sequence"].nunique()),
            "treino_target_ids": int(tr["target_id"].nunique()),
            "val_target_ids": int(va["target_id"].nunique()),
            "teste_target_ids": int(te["target_id"].nunique()),
        })

    df_t2 = pd.DataFrame(rows)
    return write_table(
        "T2_splits",
        "T2 — Tamanhos e Grupos Efetivos por Conteúdo (SMILES Canônico e Sequência) nos Splits",
        df_t2,
        "data/processed/splits/*.json, data/processed/{davis,kiba}_clean.csv e docs/paper/cpu_audits.json",
        tables_dir,
        extra_notes=[
            "Em `davis` (`cold_drug` e `cold_target`), o agrupamento estrito é por conteúdo (`smiles` canônico para drogas; `sequence` para alvos). Note que `379` sequências únicas correspondem a `442` `target_id`s no Davis (`81` `target_id`s colapsam em `18` sequências), enquanto no KIBA `2065` SMILES canônicos correspondem a `2111` `drug_id`s.",
            "Verificação de vazamento por conteúdo em todos os 30 splits cold (`v1_seeds` + `v2_kfold`): `0` sobreposições entre treino, validação e teste.",
            f"Proposta P0 de split por identidade de sequência em `davis_cold_target` (`{seqid['tool']}=={seqid['version']}`, `{seqid['method']}` em `379` sequências únicas, `{seqid['aligned_pairs_computed']:,}` alinhamentos em `{seqid['elapsed_sec']:.2f} s`): (i) `id >= 30%`: `{t30['num_clusters']}` clusters (`{t30['singleton_clusters']}` singletons, maior cluster `{t30['largest_cluster_unique_seqs']}` seqs / `{t30['largest_cluster_pairs']}` pares), 5 folds `[pares/seqs_unicas] = [{f30}]`; (ii) `id >= 40%`: `{t40['num_clusters']}` clusters (`{t40['singleton_clusters']}` singletons, maior cluster `{t40['largest_cluster_unique_seqs']}` seqs / `{t40['largest_cluster_pairs']}` pares), 5 folds `[{f40}]`; (iii) `id >= 50%`: `{t50['num_clusters']}` clusters (`{t50['singleton_clusters']}` singletons, maior cluster `{t50['largest_cluster_unique_seqs']}` seqs / `{t50['largest_cluster_pairs']}` pares), 5 folds `[{f50}]`.",
        ],
    )


def build_t3(data_root: Path, runs: list[dict[str, Any]], tables_dir: Path) -> dict[str, Path]:
    res_dir = data_root / "outputs" / "results"
    rows = []

    # 1. Protocolo v1 (seeds)
    for agg_path in sorted(res_dir.glob("baseline_*.json")):
        if agg_path.name.endswith("_kfold.json") or "seqid" in agg_path.stem:
            continue
        ad = load_json_strict(agg_path)
        agg = ad["aggregated"]
        tm = agg["test_metrics_full"]
        tb = agg.get("trivial_baselines", {})
        mp = tb.get("mean_predictor", {})
        rg = tb.get("ridge_aa_composition", {})

        rows.append({
            "protocolo": "v1_seeds",
            "modelo": ad["model"],
            "dataset": ad["dataset"],
            "split": ad["split"],
            "n_unidades": int(ad["num_seeds"]),
            "ci": fmt_mean_std(tm["ci"]["mean"], tm["ci"]["std"]),
            "mse": fmt_mean_std(tm["mse"]["mean"], tm["mse"]["std"]),
            "rmse": fmt_mean_std(tm["rmse"]["mean"], tm["rmse"]["std"]),
            "pearson": fmt_mean_std(tm["pearson"]["mean"], tm["pearson"]["std"]),
            "spearman": fmt_mean_std(tm["spearman"]["mean"], tm["spearman"]["std"]),
            "rm2": fmt_mean_std(tm["rm2"]["mean"], tm["rm2"]["std"]),
            "oof_pooled_ci": NA_TEXT,
            "oof_pooled_mse": NA_TEXT,
            "mean_predictor_mse": fmt_mean_std(mp.get("mse", {}).get("mean"), mp.get("mse", {}).get("std")) if mp else NA_TEXT,
            "ridge_aa_ci": fmt_mean_std(rg.get("ci", {}).get("mean"), rg.get("ci", {}).get("std")) if rg else NA_TEXT,
            "ridge_aa_mse": fmt_mean_std(rg.get("mse", {}).get("mean"), rg.get("mse", {}).get("std")) if rg else NA_TEXT,
            "auroc_gt5": NA_TEXT,
            "auroc_ge7": NA_TEXT,
            "fonte": f"outputs/results/{agg_path.name}",
        })

    # 2. Protocolo v2 (kfold) — estritamente separado
    for agg_path in sorted(res_dir.glob("baseline_*_kfold.json")):
        if "seqid" in agg_path.stem:
            continue
        ad = load_json_strict(agg_path)
        pf = ad["per_fold_mean_std"]
        oof = ad["global_oof_metrics"]
        bc = ad.get("binary_classification_metrics", {})

        # Trivial baselines das 5 dobras (lidos de per_fold_results)
        pfr = [v for _, v in sorted(ad["per_fold_results"].items(), key=lambda x: int(x[0]))]
        mp_mses = [f["trivial_baselines"]["mean_predictor"]["mse"] for f in pfr if f.get("trivial_baselines")]
        rg_cis = [f["trivial_baselines"]["ridge_aa_composition"]["ci"] for f in pfr if f.get("trivial_baselines")]
        rg_mses = [f["trivial_baselines"]["ridge_aa_composition"]["mse"] for f in pfr if f.get("trivial_baselines")]

        mp_m, mp_s = sample_mean_std(mp_mses)
        rc_m, rc_s = sample_mean_std(rg_cis)
        rm_m, rm_s = sample_mean_std(rg_mses)

        rows.append({
            "protocolo": "v2_kfold",
            "modelo": ad["model"],
            "dataset": ad["dataset"],
            "split": ad["split"],
            "n_unidades": int(ad["num_folds"]),
            "ci": fmt_mean_std(pf["ci"]["mean"], pf["ci"]["std"]),
            "mse": fmt_mean_std(pf["mse"]["mean"], pf["mse"]["std"]),
            "rmse": fmt_mean_std(pf["rmse"]["mean"], pf["rmse"]["std"]),
            "pearson": fmt_mean_std(pf["pearson"]["mean"], pf["pearson"]["std"]),
            "spearman": fmt_mean_std(pf["spearman"]["mean"], pf["spearman"]["std"]),
            "rm2": fmt_mean_std(pf["rm2"]["mean"], pf["rm2"]["std"]),
            "oof_pooled_ci": fmt_float(oof["ci"]),
            "oof_pooled_mse": fmt_float(oof["mse"]),
            "mean_predictor_mse": fmt_mean_std(mp_m, mp_s),
            "ridge_aa_ci": fmt_mean_std(rc_m, rc_s),
            "ridge_aa_mse": fmt_mean_std(rm_m, rm_s),
            "auroc_gt5": fmt_float(bc.get("threshold_5.0", {}).get("auroc")),
            "auroc_ge7": fmt_float(bc.get("threshold_7.0", {}).get("auroc")),
            "fonte": f"outputs/results/{agg_path.name}",
        })

    df_t3 = pd.DataFrame(rows)
    return write_table(
        "T3_main_results",
        "T3 — Resultados Principais dos Baselines (Protocolos v1 Seeds e v2 GroupKFold Separados)",
        df_t3,
        "outputs/results/baseline_*.json",
        tables_dir,
        extra_notes=[
            "Regra de separação estrita: `v1_seeds` e `v2_kfold` aparecem em blocos distintos e jamais são combinados na mesma média.",
            "`kiba_random` (`v1_seeds`) possui teto efetivo de `1000` épocas para todas as 5 seeds (seeds `0, 2, 4` encerraram por early stopping `< 500`; seeds `1, 3` foram estendidas `500 -> 1000`); `graphdta_cnn3` no KIBA foi adiado para após os resultados iniciais da Fase 3.",
            "Para `mean_predictor`, o CI teórico/empírico de predição constante é `0.5000 ± 0.0000` em todos os regimes.",
        ],
    )


def build_t4(data_root: Path, runs: list[dict[str, Any]], tables_dir: Path) -> dict[str, Path]:
    repo_root = resolve_roots()[0]
    cpu_audits = load_json_strict(repo_root / "docs" / "paper" / "cpu_audits.json")
    v1_paired = cpu_audits["v1_marginals_knn_and_paired"]["paired_groups"]
    res_dir = data_root / "outputs" / "results"
    rows = []

    # Protocolo v2_kfold: completo com marginais por dobra, pareados, wins e bootstrap exploratorio
    for agg_path in sorted(res_dir.glob("baseline_*_kfold.json")):
        if "seqid" in agg_path.stem:
            continue
        ad = load_json_strict(agg_path)
        split = ad["split"]
        rel_key = "target_mean_predictor" if split == "cold_drug" else "drug_mean_predictor"
        boot_key = "vs_target_mean" if split == "cold_drug" else "vs_drug_mean"

        pfr = [v for _, v in sorted(ad["per_fold_results"].items(), key=lambda x: int(x[0]))]
        dm_ci, dm_mse, tm_ci, tm_mse = [], [], [], []
        d_mse_paired, d_ci_paired = [], []
        wins_mse, wins_ci = 0, 0
        knn_d_ci, knn_d_mse, knn_t_ci, knn_t_mse = [], [], [], []

        for f in pfr:
            mb = f["marginal_baselines"]
            tm = f["test_metrics_full"]
            dm_ci.append(mb["drug_mean_predictor"]["ci"])
            dm_mse.append(mb["drug_mean_predictor"]["mse"])
            tm_ci.append(mb["target_mean_predictor"]["ci"])
            tm_mse.append(mb["target_mean_predictor"]["mse"])

            delta_mse = tm["mse"] - mb[rel_key]["mse"]
            delta_ci = tm["ci"] - mb[rel_key]["ci"]
            d_mse_paired.append(delta_mse)
            d_ci_paired.append(delta_ci)
            if delta_mse < 0:
                wins_mse += 1
            if delta_ci > 0:
                wins_ci += 1

            kb = f.get("knn_baselines", {})
            knn_d_ci.append(kb["knn_tanimoto_drug"]["ci"])
            knn_d_mse.append(kb["knn_tanimoto_drug"]["mse"])
            knn_t_ci.append(kb["knn_kmer_target"]["ci"])
            knn_t_mse.append(kb["knn_kmer_target"]["mse"])

        bs = ad["bootstrap_analysis"][boot_key]
        bs_mse = bs["delta_mse"]
        bs_ci = bs["delta_ci"]
        mse_win_str = (
            f"{bs_mse['wins']}/{bs_mse['total_bootstraps']} fav."
            if ("wins" in bs_mse and "total_bootstraps" in bs_mse)
            else "N/A fav."
        )
        ci_win_str = (
            f"{bs_ci['wins']}/{bs_ci['total_bootstraps']} fav."
            if ("wins" in bs_ci and "total_bootstraps" in bs_ci)
            else "N/A fav."
        )

        rows.append({
            "protocolo": "v2_kfold",
            "modelo": ad["model"],
            "dataset": ad["dataset"],
            "split": split,
            "baseline_relevante": rel_key,
            "n_avaliado": f"{len(pfr)}/{ad['num_folds']}",
            "drug_mean_ci": fmt_mean_std(*sample_mean_std(dm_ci)),
            "drug_mean_mse": fmt_mean_std(*sample_mean_std(dm_mse)),
            "target_mean_ci": fmt_mean_std(*sample_mean_std(tm_ci)),
            "target_mean_mse": fmt_mean_std(*sample_mean_std(tm_mse)),
            "delta_mse_pareado": fmt_mean_std(*sample_mean_std(d_mse_paired), signed=True),
            "delta_ci_pareado": fmt_mean_std(*sample_mean_std(d_ci_paired), signed=True),
            "vitorias_mse_ci": f"{wins_mse}/{len(pfr)} MSE | {wins_ci}/{len(pfr)} CI",
            "bootstrap_delta_mse_ic95_frac_expl": f"{bs_mse['mean']:+.4f} [{bs_mse['ci_95'][0]:.4f}, {bs_mse['ci_95'][1]:.4f}] | {mse_win_str} (exploratório)",
            "bootstrap_delta_ci_ic95_frac_expl": f"{bs_ci['mean']:+.4f} [{bs_ci['ci_95'][0]:.4f}, {bs_ci['ci_95'][1]:.4f}] | {ci_win_str} (exploratório)",
            "knn_tanimoto_ci_mse": f"{sample_mean_std(knn_d_ci)[0]:.4f} / {sample_mean_std(knn_d_mse)[0]:.4f}",
            "knn_kmer_ci_mse": f"{sample_mean_std(knn_t_ci)[0]:.4f} / {sample_mean_std(knn_t_mse)[0]:.4f}",
            "fonte": f"outputs/results/{agg_path.name}",
        })

    # Protocolo v1_seeds: marginais, kNN e deltas pareados por seed recomputados em CPU para todos os 45 runs
    by_grp: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in runs:
        if r["_proto"] == "v1_seeds":
            by_grp[(r["model"], r["dataset"], r["split"])].append(r)

    for (mod, ds, sp), grp_runs in sorted(by_grp.items()):
        gk = f"{mod}:{ds}:{sp}"
        pg = v1_paired[gk]
        mb = pg["marginal_baselines"]
        kb = pg["knn_baselines"]
        dm = mb["drug_mean_predictor"]
        tm = mb["target_mean_predictor"]
        kd = kb["knn_tanimoto_drug"]
        kt = kb["knn_kmer_target"]
        dm_p = pg["delta_mse_paired"]
        dc_p = pg["delta_ci_paired"]
        n_s = int(pg["n_seeds"])

        rows.append({
            "protocolo": "v1_seeds",
            "modelo": mod,
            "dataset": ds,
            "split": sp,
            "baseline_relevante": pg["relevant_baseline"],
            "n_avaliado": f"{n_s}/{len(grp_runs)}",
            "drug_mean_ci": fmt_mean_std(dm["ci"]["mean"], dm["ci"]["std"]),
            "drug_mean_mse": fmt_mean_std(dm["mse"]["mean"], dm["mse"]["std"]),
            "target_mean_ci": fmt_mean_std(tm["ci"]["mean"], tm["ci"]["std"]),
            "target_mean_mse": fmt_mean_std(tm["mse"]["mean"], tm["mse"]["std"]),
            "delta_mse_pareado": fmt_mean_std(dm_p["mean"], dm_p["std"], signed=True),
            "delta_ci_pareado": fmt_mean_std(dc_p["mean"], dc_p["std"], signed=True),
            "vitorias_mse_ci": f"{pg['wins_mse']}/{n_s} MSE | {pg['wins_ci']}/{n_s} CI",
            "bootstrap_delta_mse_ic95_frac_expl": "N/A (v1: Δ por seed; bootstrap só em v2 OOF)",
            "bootstrap_delta_ci_ic95_frac_expl": "N/A (v1: Δ por seed; bootstrap só em v2 OOF)",
            "knn_tanimoto_ci_mse": f"{kd['ci']['mean']:.4f} / {kd['mse']['mean']:.4f}",
            "knn_kmer_ci_mse": f"{kt['ci']['mean']:.4f} / {kt['mse']['mean']:.4f}",
            "fonte": f"docs/paper/cpu_audits.json (splits + CSVs + runs/{mod}_{ds}_{sp}_seed*.json)",
        })

    df_t4 = pd.DataFrame(rows)
    return write_table(
        "T4_marginal_and_paired",
        "T4 — Baselines Marginais de Promiscuidade, kNN e Comparação Pareada (Δ MSE e Δ CI)",
        df_t4,
        "outputs/results/baseline_*_kfold.json, outputs/results/runs/*.json e docs/paper/cpu_audits.json",
        tables_dir,
        extra_notes=[
            "Convenção de sinal: `Δ MSE < 0` e `Δ CI > 0` indicam superioridade do modelo sobre o baseline marginal relevante no mesmo conjunto de teste.",
            "Para `v1_seeds` (todos os 45 runs), os baselines marginais, `kNN` (`k=5` determinístico) e os deltas pareados por seed foram recomputados em CPU a partir dos CSVs limpos e splits oficiais em `docs/paper/cpu_audits.json` (sem necessidade de checkpoints); no regime `random`, o baseline relevante exibido é `drug_mean_predictor` (melhor marginal em MSE e CI).",
            "Para `v2_kfold`, o bootstrap agrupado por cluster (`1.000` reamostras) reporta IC 95% e a fração de reamostras favoráveis (`wins/1000`), marcados como **comparações exploratórias** (substituindo valores-p).",
        ],
    )


def build_t5(data_root: Path, tables_dir: Path) -> dict[str, Path]:
    repo_root = resolve_roots()[0]
    cpu_audits = load_json_strict(repo_root / "docs" / "paper" / "cpu_audits.json")
    unc_base = cpu_audits["davis_uncensored_baselines"]
    res_dir = data_root / "outputs" / "results"
    rows = []

    # v1_seeds (Davis)
    for agg_path in sorted(res_dir.glob("baseline_*_davis_*.json")):
        if agg_path.name.endswith("_kfold.json"):
            continue
        ad = load_json_strict(agg_path)
        unc = ad["aggregated"]["test_metrics_uncensored"]
        ub = unc_base[f"v1_seeds:{ad['split']}"]["aggregated"]
        rows.append({
            "protocolo": "v1_seeds",
            "modelo": ad["model"],
            "split": ad["split"],
            "n_unidades": int(ad["num_seeds"]),
            "pares_nao_censurados_por_unidade": fmt_mean_std(unc["num_uncensored_pairs"]["mean"], unc["num_uncensored_pairs"]["std"], decimals=1),
            "ci_pkd_gt_5": fmt_mean_std(unc["ci"]["mean"], unc["ci"]["std"]),
            "mse_pkd_gt_5": fmt_mean_std(unc["mse"]["mean"], unc["mse"]["std"]),
            "pearson_pkd_gt_5": fmt_mean_std(unc["pearson"]["mean"], unc["pearson"]["std"]),
            "spearman_pkd_gt_5": fmt_mean_std(unc["spearman"]["mean"], unc["spearman"]["std"]),
            "rm2_pkd_gt_5": fmt_mean_std(unc["rm2"]["mean"], unc["rm2"]["std"]),
            "mean_predictor_mse_pkd_gt_5": fmt_mean_std(ub["mean_predictor_full_train"]["mse"]["mean"], ub["mean_predictor_full_train"]["mse"]["std"]),
            "mean_predictor_ativo_mse_pkd_gt_5": fmt_mean_std(ub["mean_predictor_active_train"]["mse"]["mean"], ub["mean_predictor_active_train"]["mse"]["std"]),
            "drug_mean_mse_pkd_gt_5": fmt_mean_std(ub["drug_mean_full_train"]["mse"]["mean"], ub["drug_mean_full_train"]["mse"]["std"]),
            "target_mean_mse_pkd_gt_5": fmt_mean_std(ub["target_mean_full_train"]["mse"]["mean"], ub["target_mean_full_train"]["mse"]["std"]),
            "fonte": f"outputs/results/{agg_path.name}; docs/paper/cpu_audits.json",
        })

    # v2_kfold (Davis)
    for agg_path in sorted(res_dir.glob("baseline_*_davis_*_kfold.json")):
        if "seqid" in agg_path.stem:
            continue
        ad = load_json_strict(agg_path)
        pfr = [v for _, v in sorted(ad["per_fold_results"].items(), key=lambda x: int(x[0]))]
        uncs = [f["test_metrics_uncensored"] for f in pfr]
        ub = unc_base[f"v2_kfold:{ad['split']}"]["aggregated"]
        rows.append({
            "protocolo": "v2_kfold",
            "modelo": ad["model"],
            "split": ad["split"],
            "n_unidades": int(ad["num_folds"]),
            "pares_nao_censurados_por_unidade": fmt_mean_std(*sample_mean_std([u["num_uncensored_pairs"] for u in uncs]), decimals=1) + f" (total={sum(u['num_uncensored_pairs'] for u in uncs)})",
            "ci_pkd_gt_5": fmt_mean_std(*sample_mean_std([u["ci"] for u in uncs])),
            "mse_pkd_gt_5": fmt_mean_std(*sample_mean_std([u["mse"] for u in uncs])),
            "pearson_pkd_gt_5": fmt_mean_std(*sample_mean_std([u["pearson"] for u in uncs])),
            "spearman_pkd_gt_5": fmt_mean_std(*sample_mean_std([u["spearman"] for u in uncs])),
            "rm2_pkd_gt_5": fmt_mean_std(*sample_mean_std([u["rm2"] for u in uncs])),
            "mean_predictor_mse_pkd_gt_5": fmt_mean_std(ub["mean_predictor_full_train"]["mse"]["mean"], ub["mean_predictor_full_train"]["mse"]["std"]),
            "mean_predictor_ativo_mse_pkd_gt_5": fmt_mean_std(ub["mean_predictor_active_train"]["mse"]["mean"], ub["mean_predictor_active_train"]["mse"]["std"]),
            "drug_mean_mse_pkd_gt_5": fmt_mean_std(ub["drug_mean_full_train"]["mse"]["mean"], ub["drug_mean_full_train"]["mse"]["std"]),
            "target_mean_mse_pkd_gt_5": fmt_mean_std(ub["target_mean_full_train"]["mse"]["mean"], ub["target_mean_full_train"]["mse"]["std"]),
            "fonte": f"outputs/results/{agg_path.name}; docs/paper/cpu_audits.json",
        })

    df_t5 = pd.DataFrame(rows)
    return write_table(
        "T5_uncensored_davis",
        "T5 — Desempenho no Subconjunto Não Censurado do Davis (pKd > 5.0)",
        df_t5,
        "outputs/results/baseline_*_davis_*.json e docs/paper/cpu_audits.json",
        tables_dir,
        extra_notes=[
            "AVISO METODOLÓGICO: O filtro `|y_true - 5.0| > 1e-4` seleciona pelo valor do rótulo verdadeiro (`pKd > 5.0`), induzindo viés de seleção para cima no MSE em relação ao conjunto completo (onde 69,64% dos rótulos valem 5,0).",
            "Baselines recomputados no mesmo subconjunto `pKd > 5.0` em `docs/paper/cpu_audits.json`: `mean_predictor_mse_pkd_gt_5` usa a média do treino completo (comparável diretamente ao modelo treinado sem filtro), enquanto `mean_predictor_ativo_mse_pkd_gt_5` usa apenas pares de treino com `pKd > 5.0`; incluem-se também `drug_mean` e `target_mean` (treino completo) no mesmo subconjunto.",
        ],
    )


def build_t6(data_root: Path, tables_dir: Path) -> dict[str, Path]:
    res_dir = data_root / "outputs" / "results"
    rows = []

    # v1_seeds
    for agg_path in sorted(res_dir.glob("baseline_*.json")):
        if agg_path.name.endswith("_kfold.json") or "seqid" in agg_path.stem:
            continue
        ad = load_json_strict(agg_path)
        sl = ad["aggregated"]["test_metrics_by_seq_len"]
        for faixa, label in (("le_1000", "<= 1000 aa"), ("gt_1000", "> 1000 aa")):
            blk = sl[faixa]
            rows.append({
                "protocolo": "v1_seeds",
                "modelo": ad["model"],
                "dataset": ad["dataset"],
                "split": ad["split"],
                "faixa_comprimento": label,
                "n_pares_medio_teste": fmt_mean_std(blk["count"]["mean"], blk["count"]["std"], decimals=1),
                "ci": fmt_mean_std(blk["ci"]["mean"], blk["ci"]["std"]),
                "mse": fmt_mean_std(blk["mse"]["mean"], blk["mse"]["std"]),
                "fonte": f"outputs/results/{agg_path.name}",
            })

    # v2_kfold
    for agg_path in sorted(res_dir.glob("baseline_*_kfold.json")):
        if "seqid" in agg_path.stem:
            continue
        ad = load_json_strict(agg_path)
        pfr = [v for _, v in sorted(ad["per_fold_results"].items(), key=lambda x: int(x[0]))]
        for faixa, label in (("le_1000", "<= 1000 aa"), ("gt_1000", "> 1000 aa")):
            cnts = [f["test_metrics_by_seq_len"][faixa]["count"] for f in pfr]
            cis = [f["test_metrics_by_seq_len"][faixa]["ci"] for f in pfr]
            mses = [f["test_metrics_by_seq_len"][faixa]["mse"] for f in pfr]
            rows.append({
                "protocolo": "v2_kfold",
                "modelo": ad["model"],
                "dataset": ad["dataset"],
                "split": ad["split"],
                "faixa_comprimento": label,
                "n_pares_medio_teste": fmt_mean_std(*sample_mean_std(cnts), decimals=1) + f" (total={sum(cnts)})",
                "ci": fmt_mean_std(*sample_mean_std(cis)),
                "mse": fmt_mean_std(*sample_mean_std(mses)),
                "fonte": f"outputs/results/{agg_path.name} (per_fold_results[*].test_metrics_by_seq_len)",
            })

    df_t6 = pd.DataFrame(rows)
    return write_table(
        "T6_seq_length",
        "T6 — Desempenho Estratificado por Comprimento da Sequência Proteica (<=1000 vs. >1000 aa)",
        df_t6,
        "outputs/results/baseline_*.json",
        tables_dir,
        extra_notes=[
            "Total de sequências únicas no dataset completo (T1): Davis = `303` (`<=1000` aa) vs. `76` (`>1000` aa); KIBA = `185` (`<=1000` aa) vs. `44` (`>1000` aa).",
        ],
    )


def build_t7(runs_all: list[dict[str, Any]], tables_dir: Path) -> dict[str, Path]:
    rows = []
    for r in sorted(runs_all, key=lambda x: (x["_proto"], x["dataset"], x["_model_norm"], x["split"], x["_unit"])):
        ec = r.get("effective_config", {})
        ep_cfg = int(ec.get("epochs", 500))
        best_ep = int(r["best_epoch"])
        tot_ep = int(r["total_epochs_run"])
        alerta_08 = best_ep > 0.8 * ep_cfg
        is_cold = "cold" in str(r["split"])
        flag_lt15 = bool(is_cold and best_ep < 15)
        mod_norm = r["_model_norm"]
        raw_mod = str(r["model"])
        dc = r.get("diagnostic_control")
        is_ctrl = bool((dc or {}).get("is_control", False) or mod_norm in ("ecfp_esm_mlp",) or "_shuf" in mod_norm or "_tsoff" in mod_norm)
        emw = r.get("early_memorization_warn")
        emw_str = NA_TEXT if emw is None else str(bool(emw)).lower()

        if raw_mod == "graphdta_gin" and r["dataset"] == "davis" and r["split"] == "cold_target":
            obs = "colapso precoce (best_epoch <= 13)" if best_ep <= 15 else "outlier tardio no cold_target"
        elif raw_mod.startswith("graphdta_") and flag_lt15:
            obs = f"cold best_epoch<15 retroativo ({r['dataset']}/{r['split']})"
        elif raw_mod.startswith("graphdta_") and r["dataset"] == "kiba" and r["split"] == "random" and ep_cfg == 1000:
            obs = "retomado 500->1000 ep.; sec/ep distorcido (só retomada)"
        elif raw_mod.startswith("graphdta_") and r["dataset"] == "kiba" and r["split"] == "random" and ep_cfg == 500:
            obs = "ES <500 ep. (equiv. teto efetivo 1000)"
        elif raw_mod == "physdual_gin_esm_mlp" and r["dataset"] == "davis" and r["split"] == "cold_target" and r["_unit"] == 1:
            obs = "estendido 500->1000 ep.; sec/ep=0.260 distorcido (só trecho retomado)"
        elif raw_mod == "physdual_gin_esm_mlp" and r["dataset"] == "davis" and r["split"] == "cold_target":
            obs = "pré-ADR #15 (configs epochs=500; ES <500)"
        elif raw_mod == "physdual_gin_esm_mlp" and r["dataset"] == "davis" and r["split"] == "cold_drug" and r["_unit"] == 2:
            obs = "fold 2 extremo (best_ep=5; cold<15 pré-ADR #15 sem flag histórica)"
        elif "_shuf" in mod_norm:
            obs = "controle --shuffle-esm; memorização precoce" if flag_lt15 else "controle --shuffle-esm"
        elif "_tsoff" in mod_norm:
            obs = "controle --train-seed-offset 1000"
        elif mod_norm == "ecfp_esm_mlp":
            obs = "controle ligand ECFP4; cold best_ep<15" if flag_lt15 else "controle ligand ECFP4"
        elif alerta_08:
            obs = "proximo ao teto de epocas"
        elif flag_lt15:
            obs = "cold best_epoch<15 (alerta memorização precoce)"
        else:
            obs = "normal"

        rows.append({
            "protocolo": r["_proto"],
            "modelo": mod_norm,
            "dataset": r["dataset"],
            "split": r["split"],
            "unidade": f"{'fold' if r['_proto']=='v2_kfold' else 'seed'}_{r['_unit']}",
            "best_epoch": best_ep,
            "total_epochs_run": tot_ep,
            "epochs_cfg": ep_cfg,
            "train_loss_at_best": fmt_float(r.get("train_loss_at_best"), decimals=4),
            "train_val_mse_gap_at_best": fmt_float(r.get("train_val_mse_gap_at_best"), decimals=4),
            "sec_per_epoch": fmt_float(r.get("sec_per_epoch"), decimals=3),
            "elapsed_sec": fmt_float(r.get("elapsed_sec"), decimals=2),
            "flag_gt_0_8_epochs": str(bool(alerta_08)).lower(),
            "near_epoch_ceiling_warn_json": str(bool(r.get("near_epoch_ceiling_warn", False))).lower(),
            "flag_cold_best_ep_lt_15": str(flag_lt15).lower(),
            "early_memorization_warn_json": emw_str,
            "is_diagnostic_control": str(is_ctrl).lower(),
            "observacao": obs,
            "fonte": f"outputs/results/runs/{r['_filename']}",
        })

    df_t7 = pd.DataFrame(rows)
    return write_table(
        "T7_training_diagnostics",
        "T7 — Diagnóstico de Treinamento por Run (Fases 1–3: 166 Execuções)",
        df_t7,
        "outputs/results/runs/*.json",
        tables_dir,
        extra_notes=[
            "Abrange todas as 166 execuções: 85 runs GraphDTA (Fases 1–2, incluindo 20 runs em `seqid30/40`) e 81 runs da Fase 3 (`gin_esm_mlp`, `ecfp_esm_mlp`, `--shuffle-esm` e `--train-seed-offset 1000`).",
            "Nenhum dos 166 runs atingiu truncamento (`total_epochs_run >= epochs_cfg` sem `early_stopping`) nem excedeu `0.8 * epochs_cfg` (`best_epoch > 0.8 * epochs`).",
            "Em regimes `cold`, 10 runs da Fase 3 e 19 runs retroativos de `graphdta_gin` no Davis tiveram `best_epoch < 15`; `gin_esm_mlp` no `davis/cold_drug` fold 2 (`best_epoch=5`) antecedeu o ADR #15 e traz `early_memorization_warn=false` no JSON bruto.",
            "Em `davis/cold_target` (`gin_esm_mlp` fold 1) e `kiba/random` (`graphdta_gin` seeds 1, 3), os runs foram estendidos de `500` para `1000` épocas a partir de checkpoint e exibem `sec_per_epoch` subestimado (`elapsed_sec` apenas do trecho retomado dividido pelo total acumulado de épocas).",
        ],
    )


def build_t8(repo_root: Path, data_root: Path, runs: list[dict[str, Any]], tables_dir: Path) -> dict[str, Path]:
    cfg = yaml.safe_load((repo_root / "configs" / "base.yaml").read_text(encoding="utf-8"))
    cpu_audits = load_json_strict(repo_root / "docs" / "paper" / "cpu_audits.json")
    clean_kr = cpu_audits["kiba_random_clean_timings"]
    bench = load_json_strict(data_root / "outputs" / "results" / "phase2_timing_benchmark.json")
    snap = (repo_root / "docs" / "colab_env_snapshot.txt").read_text(encoding="utf-8").splitlines()

    gpu_line = next((l.split(":", 1)[1].strip() for l in snap if "GPU [0]" in l), NA_TEXT)
    py_ver = next((l.split(":", 1)[1].strip() for l in snap if l.startswith("Python")), NA_TEXT)
    torch_ver = next((l.split("==", 1)[1].strip() for l in snap if l.startswith("torch==")), NA_TEXT)
    pyg_ver = next((l.split("==", 1)[1].strip() for l in snap if l.startswith("torch-geometric==")), NA_TEXT)
    rdkit_ver = next((l.split("==", 1)[1].strip() for l in snap if l.startswith("rdkit==")), NA_TEXT)
    esm_ver = next((l.split("==", 1)[1].strip() for l in snap if l.startswith("fair-esm==")), NA_TEXT)

    env_summary = f"Python {py_ver}; torch {torch_ver}; PyG {pyg_ver}; RDKit {rdkit_ver}; fair-esm {esm_ver}"

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in runs:
        grouped[(r["_proto"], r["model"], r["dataset"], r["split"])].append(r)

    rows = []
    for (proto, mod, ds, sp), grp_runs in sorted(grouped.items()):
        ec0 = grp_runs[0]["effective_config"]
        is_kr = (ds == "kiba" and sp == "random")
        ep_str = "1000 efetivo (s0,2,4 ES<500)" if is_kr else "/".join(str(e) for e in sorted({int(r["effective_config"]["epochs"]) for r in grp_runs}))
        pat_eff = int(ec0["davis_cold_patience"]) if (ds == "davis" and sp != "random") else int(ec0["patience"])

        if is_kr:
            sec_ep_str = fmt_mean_std(
                clean_kr["non_resumed_sec_per_epoch"]["mean"],
                clean_kr["non_resumed_sec_per_epoch"]["std"],
                decimals=3,
            ) + " (s0,2,4 limpos)"
            tot_sec = sum(float(r["elapsed_sec"]) for r in grp_runs)
            tot_hr = tot_sec / 3600.0
            caveat = " (subestimado: seeds 1,3 contam só retomada pós-500)"
        else:
            secs_ep = [float(r["sec_per_epoch"]) for r in grp_runs]
            sec_ep_str = fmt_mean_std(*sample_mean_std(secs_ep), decimals=3)
            tot_sec = sum(float(r["elapsed_sec"]) for r in grp_runs)
            tot_hr = tot_sec / 3600.0
            caveat = ""

        rows.append({
            "protocolo": proto,
            "modelo": mod,
            "dataset": ds,
            "split": sp,
            "n_runs": len(grp_runs),
            "featurizer": grp_runs[0].get("featurizer_version", cfg["featurizer"]["version"]),
            "lr_batch_dropout": f"lr={ec0['lr']}, bs={ec0['batch_size']}, do={cfg['graphdta']['dropout']}",
            "epochs_cfg": ep_str,
            "patience_efetiva": pat_eff,
            "gpu_snapshot": f"{gpu_line} (benchmark={bench['gpu_name']})",
            "versoes_snapshot": env_summary,
            "sec_per_epoch_medio": sec_ep_str,
            "tempo_total_grupo_s": f"{tot_sec:.1f}",
            "tempo_total_grupo_h": f"{tot_hr:.2f} h{caveat}",
            "fonte": f"configs/base.yaml; docs/colab_env_snapshot.txt; outputs/results/runs/{mod}_{ds}_{sp}_*.json" + ("; docs/paper/cpu_audits.json" if is_kr else ""),
        })

    df_t8 = pd.DataFrame(rows)
    return write_table(
        "T8_configuration",
        "T8 — Hiperparâmetros Efetivos, Versões de Ambiente e Tempo Computacional por Grupo",
        df_t8,
        "configs/base.yaml, docs/colab_env_snapshot.txt, outputs/results/phase2_timing_benchmark.json, outputs/results/runs/*.json e docs/paper/cpu_audits.json",
        tables_dir,
        extra_notes=[
            f"Benchmark preliminar (3 épocas, `{bench['gpu_name']}`): Davis = `{bench['datasets']['davis']['sec_per_epoch']} s/época`; KIBA = `{bench['datasets']['kiba']['sec_per_epoch']} s/época` (`eligible_for_cnn3_secondary={bench['datasets']['kiba']['eligible_for_cnn3_secondary']}`).",
            "Para `kiba_random`, `sec_per_epoch_medio` foi recalculado apenas sobre os 3 runs não retomados (`seeds 0, 2, 4`: `22.338 ± 0.014 s/época`), excluindo os artefatos distorcidos de retomada (`seeds 1, 3`: `0.646 s` e `2.347 s`). O tempo total somado não inclui as ~500 épocas iniciais prévias à retomada nas seeds 1 e 3.",
        ],
    )


def _fmt_boot_ic_frac(boot_entry: dict[str, Any] | None) -> tuple[str, str, str, str]:
    """Formata ICs 95% de ΔMSE e ΔCI + frações favoráveis `[w/1000]` sem p-valor extremo."""
    if not boot_entry:
        return NA_TEXT, NA_TEXT, NA_TEXT, NA_TEXT
    dm = boot_entry["delta_mse"]
    dc = boot_entry["delta_ci"]
    dm_ic = f"[{fmt_float(dm['ci_95'][0])}, {fmt_float(dm['ci_95'][1])}]"
    dm_frac = f"{int(dm['wins'])}/{int(dm['total_bootstraps'])}"
    dc_ic = f"[{fmt_float(dc['ci_95'][0])}, {fmt_float(dc['ci_95'][1])}]"
    dc_frac = f"{int(dc['wins'])}/{int(dc['total_bootstraps'])}"
    return dm_ic, dm_frac, dc_ic, dc_frac


_REGIME_SPECS = [
    ("v1_seeds", "kiba/cold_drug", "primario_confirmativo_kiba"),
    ("v1_seeds", "kiba/cold_target", "primario_confirmativo_kiba"),
    ("v1_seeds", "kiba/random", "sanidade_nao_confirmativo"),
    ("v1_seeds", "davis/random", "sanidade_nao_confirmativo"),
    ("v2_kfold", "kiba/cold_target_seqid40", "secundario_confirmativo_kiba_seqid"),
    ("v2_kfold", "kiba/cold_target_seqid30", "secundario_confirmativo_kiba_seqid"),
    ("v2_kfold", "davis/cold_target", "exploratorio_davis"),
    ("v2_kfold", "davis/cold_drug", "exploratorio_davis"),
    ("v2_kfold", "davis/cold_target_seqid40", "exploratorio_davis"),
    ("v2_kfold", "davis/cold_target_seqid30", "exploratorio_davis"),
]

_MAIN_MODEL_ORDER = [
    "gin_esm_mlp",
    "graphdta_gin",
    "graphdta_cnn3",
    "ridge_esm",
    "ridge_esm_target_only",
    "knn_esm_target",
    "knn_kmer_target",
    "knn_tanimoto_drug",
    "drug_mean_predictor",
    "target_mean_predictor",
    "mean_predictor",
]


def _get_regime(pa: dict[str, Any], proto: str, rkey: str) -> dict[str, Any]:
    return pa[proto]["regimes"][rkey]


def build_t9(pa: dict[str, Any], tables_dir: Path) -> dict[str, Path]:
    """T9: Aplicação pré-registrada do ADR #15 por regime (primário/secundário/sanidade/exploratório)."""
    rows = []
    for proto, rkey, papel in _REGIME_SPECS:
        reg = _get_regime(pa, proto, rkey)
        ds, sp = reg["dataset"], reg["split"]
        if proto == "v1_seeds":
            g_sum = reg["model_summary"]["gin_esm_mlp"]
            gin_ci = fmt_mean_std(g_sum["ci"]["mean"], g_sum["ci"]["std"])
            gin_mse = fmt_mean_std(g_sum["mse"]["mean"], g_sum["mse"]["std"])
            paired_map = reg["paired_vs_baselines_seeds"]
            boot_map = reg.get("bootstrap_vs_baselines_pooled_clusters") or {}
        else:
            g_sum = reg["per_model_summary"]["gin_esm_mlp"]
            gin_ci = fmt_mean_std(g_sum["ci_per_fold"]["mean"], g_sum["ci_per_fold"]["std"])
            gin_mse = fmt_mean_std(g_sum["mse_per_fold"]["mean"], g_sum["mse_per_fold"]["std"])
            paired_map = reg["paired_vs_baselines_folds"]
            boot_map = reg.get("bootstrap_vs_baselines") or {}

        adr = reg.get("adr15_operational")
        if adr is None:
            obs_map = {
                "kiba/random": "sanidade: dMSE=-0.0217, dCI=+0.0170 vs graphdta_gin (4/5 seeds); graphdta_cnn3 ausente no KIBA",
                "davis/random": "sanidade: dMSE=-0.0295 (5/5) vs graphdta_cnn3 e -0.0427 (5/5) vs graphdta_gin; inflado por 81 target_ids em 18 seqs",
            }
            rows.append({
                "protocolo": proto,
                "papel_no_estudo": papel,
                "dataset": ds,
                "split": sp,
                "gin_ci_media_dp": gin_ci,
                "gin_mse_media_dp": gin_mse,
                "delta_vs_ridge_esm_mse": fmt_mean_std(paired_map["ridge_esm"]["delta_mse_mean"], paired_map["ridge_esm"]["delta_mse_std"], signed=True),
                "delta_vs_ridge_esm_ci": fmt_mean_std(paired_map["ridge_esm"]["delta_ci_mean"], paired_map["ridge_esm"]["delta_ci_std"], signed=True),
                "vitorias_ridge_esm": f"MSE {paired_map['ridge_esm']['wins_mse']}/5; CI {paired_map['ridge_esm']['wins_ci']}/5",
                "bootstrap_ridge_ic95_frac": "não aplicável (regime random de sanidade)",
                "veredito_vs_ridge_esm": "não aplicável (sanidade)",
                "melhor_baseline_nao_esm": NA_TEXT,
                "delta_vs_melhor_nao_esm_mse": NA_TEXT,
                "delta_vs_melhor_nao_esm_ci": NA_TEXT,
                "vitorias_melhor_nao_esm": NA_TEXT,
                "bootstrap_nao_esm_ic95_frac": "não aplicável (regime random de sanidade)",
                "veredito_vs_melhor_nao_esm": "não aplicável (sanidade)",
                "veredito_combinado_adr15": "não aplicável (sanidade)",
                "observacao": obs_map.get(rkey, "sanidade"),
                "fonte": f"docs/paper/phase3_audits.json ({proto}.regimes.{rkey})",
            })
            continue

        vr = adr["vs_ridge_esm"]
        best_non = adr["best_non_esm_mse_name"]
        vb = adr["vs_best_non_esm_mse"]
        pr = paired_map["ridge_esm"]
        pb = paired_map[best_non]
        br_dm_ic, br_dm_f, br_dc_ic, br_dc_f = _fmt_boot_ic_frac(boot_map.get("ridge_esm"))
        bb_dm_ic, bb_dm_f, bb_dc_ic, bb_dc_f = _fmt_boot_ic_frac(boot_map.get(best_non))

        if vr["verdict"] == "atendido" and vb["verdict"] == "atendido":
            comb = "atendido"
        elif vr["verdict"] == "nao_atendido_negativo" or vb["verdict"] == "nao_atendido_negativo":
            comb = "nao_atendido_negativo"
        else:
            comb = "inconclusivo"

        ver_comb_str = comb if "confirmativo" in papel else f"exploratorio ({comb})"

        obs_notes = {
            "kiba/cold_drug": "vs graphdta_gin (5 seeds): nao_atendido_negativo (dMSE=+0.0305, 0/5; dCI=-0.0025, 3/5); superado por knn_tanimoto_drug e graphdta_gin em MSE",
            "kiba/cold_target": "graphdta_gin é adversário fraco no KIBA cold_target (CI=0.6579); graphdta_cnn3 ausente; knn_kmer_target empata/supera em MSE (0.3730 vs 0.3735)",
            "kiba/cold_target_seqid40": "nao supera sequer ridge_esm (ICs cruzam zero); knn_tanimoto_drug == drug_mean_predictor (MSE=0.4553, CI=0.6986); sem GraphDTA",
            "kiba/cold_target_seqid30": "ridge_esm colapsa (MSE=0.6097) gerando atendimento isolado vs ridge_esm, mas gin_esm_mlp (0.5069) perde para drug_mean_predictor == knn_tanimoto_drug (0.4929)",
            "davis/cold_target": "exploratório (pré-ADR #15 + extensão fold 1); atende vs ridge_esm, mas empata em bootstrap com knn_kmer_target",
            "davis/cold_drug": "exploratório (~68 fármacos + fold 2 extremo MSE=1.3670); perde em 4/5 dobras para knn_kmer_target e knn_tanimoto_drug",
            "davis/cold_target_seqid40": "exploratório: vantagem discreta sobre ridge_esm (IC exclui 0, wins MSE 3/5), mas empata em MSE com knn_tanimoto_drug == drug_mean_predictor",
            "davis/cold_target_seqid30": "exploratório: colapso de margem sobre ridge_esm (IC [-0.0449, +0.0518]) e sobre drug_mean_predictor (IC [-0.0596, +0.0369])",
        }

        rows.append({
            "protocolo": proto,
            "papel_no_estudo": papel,
            "dataset": ds,
            "split": sp,
            "gin_ci_media_dp": gin_ci,
            "gin_mse_media_dp": gin_mse,
            "delta_vs_ridge_esm_mse": fmt_mean_std(pr["delta_mse_mean"], pr["delta_mse_std"], signed=True),
            "delta_vs_ridge_esm_ci": fmt_mean_std(pr["delta_ci_mean"], pr["delta_ci_std"], signed=True),
            "vitorias_ridge_esm": f"MSE {vr['wins_mse']}/{vr['n_units']}; CI {vr['wins_ci']}/{vr['n_units']}",
            "bootstrap_ridge_ic95_frac": f"dMSE {br_dm_ic} ({br_dm_f}); dCI {br_dc_ic} ({br_dc_f})",
            "veredito_vs_ridge_esm": vr["verdict"],
            "melhor_baseline_nao_esm": best_non,
            "delta_vs_melhor_nao_esm_mse": fmt_mean_std(pb["delta_mse_mean"], pb["delta_mse_std"], signed=True),
            "delta_vs_melhor_nao_esm_ci": fmt_mean_std(pb["delta_ci_mean"], pb["delta_ci_std"], signed=True),
            "vitorias_melhor_nao_esm": f"MSE {vb['wins_mse']}/{vb['n_units']}; CI {vb['wins_ci']}/{vb['n_units']}",
            "bootstrap_nao_esm_ic95_frac": f"dMSE {bb_dm_ic} ({bb_dm_f}); dCI {bb_dc_ic} ({bb_dc_f})",
            "veredito_vs_melhor_nao_esm": vb["verdict"],
            "veredito_combinado_adr15": ver_comb_str,
            "observacao": obs_notes.get(rkey, "auditado"),
            "fonte": f"docs/paper/phase3_audits.json ({proto}.regimes.{rkey}.adr15_operational)",
        })

    df_t9 = pd.DataFrame(rows)
    return write_table(
        "T9_phase3_decision_adr15",
        "T9 — Decisão Pré-Registrada da Fase 3 Segundo o ADR #15 (Modelo `gin_esm_mlp`)",
        df_t9,
        "docs/DECISOES.md (ADR #15, commit 46d13f8 / tag phase3-adr15-pre-kiba) e docs/paper/phase3_audits.json",
        tables_dir,
        extra_notes=[
            "Critério ADR #15 para progressão positiva: superar `ridge_esm` E o melhor baseline não-ESM em `>= 4/5` sementes/dobras E com IC 95% de bootstrap em clusters excluindo zero na direção favorável (`dMSE < 0` e `dCI > 0`).",
            "Nenhum regime confirmatório do KIBA atingiu ambos os braços do critério pré-registrado (`veredito_combinado_adr15 = nao_atendido_negativo` nos 4 regimes frios do KIBA).",
            "Regimes do Davis permanecem estritamente exploratórios (`v2_kfold` pré-ADR #15 e dobra 1 do `cold_target` estendida pós-ADR #15).",
        ],
    )


def build_t10(pa: dict[str, Any], tables_dir: Path) -> dict[str, Path]:
    """T10: Sumário por unidade (média ± DP e mediana) de todos os modelos e regimes (v1 e v2 separados)."""
    rows = []
    for proto, rkey, papel in _REGIME_SPECS:
        reg = _get_regime(pa, proto, rkey)
        ds, sp = reg["dataset"], reg["split"]
        src_map = reg["model_summary"] if proto == "v1_seeds" else reg["per_model_summary"]
        for mod in _MAIN_MODEL_ORDER:
            if mod not in src_map:
                continue
            ms = src_map[mod]
            if proto == "v1_seeds":
                ci_m, ci_s, ci_med = ms["ci"]["mean"], ms["ci"]["std"], ms["ci"]["median"]
                mse_m, mse_s, mse_med = ms["mse"]["mean"], ms["mse"]["std"], ms["mse"]["median"]
                rm_m, rm_s = ms["rmse"]["mean"], ms["rmse"]["std"]
                pe_m, pe_s = ms["pearson"]["mean"], ms["pearson"]["std"]
                sp_m, sp_s = ms["spearman"]["mean"], ms["spearman"]["std"]
                r2_m, r2_s = ms["rm2"]["mean"], ms["rm2"]["std"]
                n_u = len(reg["per_seed_models"][mod])
            else:
                ci_m, ci_s, ci_med = ms["ci_per_fold"]["mean"], ms["ci_per_fold"]["std"], ms["ci_per_fold"]["median"]
                mse_m, mse_s, mse_med = ms["mse_per_fold"]["mean"], ms["mse_per_fold"]["std"], ms["mse_per_fold"]["median"]
                rm_m, rm_s = ms["rmse_per_fold"]["mean"], ms["rmse_per_fold"]["std"]
                pe_m, pe_s = ms["pearson_per_fold"]["mean"], ms["pearson_per_fold"]["std"]
                sp_m, sp_s = ms["spearman_per_fold"]["mean"], ms["spearman_per_fold"]["std"]
                r2_m, r2_s = ms["rm2_per_fold"]["mean"], ms["rm2_per_fold"]["std"]
                n_u = len(reg["per_fold_by_model"][mod])

            obs = "modelo principal Fase 3" if mod == "gin_esm_mlp" else "baseline"
            if mod == "graphdta_gin" and ds == "kiba" and sp == "cold_target":
                obs = "adversário fraco no KIBA cold_target (CNN sem atenção/ESM)"
            elif mod == "knn_tanimoto_drug" and "cold_target" in sp:
                obs = "idêntico a drug_mean_predictor em alvos inéditos (k=5 Tanimoto=1.0)"
            elif mod == "knn_esm_target" and sp == "cold_drug":
                obs = "idêntico ou próximo a target_mean_predictor em fármacos inéditos"

            rows.append({
                "protocolo": proto,
                "papel_no_estudo": papel,
                "dataset": ds,
                "split": sp,
                "modelo": mod,
                "n_unidades": n_u,
                "ci_media_dp": fmt_mean_std(ci_m, ci_s),
                "ci_mediana": fmt_float(ci_med),
                "mse_media_dp": fmt_mean_std(mse_m, mse_s),
                "mse_mediana": fmt_float(mse_med),
                "rmse_media_dp": fmt_mean_std(rm_m, rm_s),
                "pearson_media_dp": fmt_mean_std(pe_m, pe_s),
                "spearman_media_dp": fmt_mean_std(sp_m, sp_s),
                "rm2_media_dp": fmt_mean_std(r2_m, r2_s),
                "observacao": obs,
                "fonte": f"docs/paper/phase3_audits.json ({proto}.regimes.{rkey})",
            })

    df_t10 = pd.DataFrame(rows)
    return write_table(
        "T10_phase3_results_summary",
        "T10 — Resultados da Fase 3 por Regime e Modelo (Protocolos v1 Seeds e v2 GroupKFold Separados)",
        df_t10,
        "docs/paper/phase3_audits.json e CSVs OOF em outputs/results/",
        tables_dir,
        extra_notes=[
            "`v1_seeds` (média em 5 sementes no KIBA padrão e `random`) e `v2_kfold` (média em 5 dobras nos regimes `seqid` do KIBA e nos 4 regimes exploratórios do Davis) são apresentados em blocos disjuntos.",
            "Observação sobre o KIBA: `graphdta_cnn3` não existe no KIBA (adiado na Fase 2); os splits `seqid40/30` do KIBA não possuem execuções GraphDTA; `graphdta_gin` no KIBA `cold_target` (`CI 0.6579 ± 0.0437`) é um adversário fraco conhecido.",
            "Para controles diagnósticos (`ecfp_esm_mlp`, `--shuffle-esm`, `--train-seed-offset 1000`), ver T14.",
        ],
    )


def build_t11(pa: dict[str, Any], tables_dir: Path) -> dict[str, Path]:
    """T11: OOF global estritamente no v2_kfold + subconjunto não censurado (pKd > 5.0) e binário no Davis."""
    rows = []
    # Bloco v2_kfold: OOF global (+ Davis uncensored & binary)
    for proto, rkey, _papel in _REGIME_SPECS:
        if proto != "v2_kfold":
            continue
        reg = _get_regime(pa, "v2_kfold", rkey)
        ds, sp = reg["dataset"], reg["split"]
        unc_map = reg.get("uncensored_oof_davis") or {}
        bin_map = reg.get("binary_oof_davis") or {}
        for mod in _MAIN_MODEL_ORDER:
            if mod not in reg["per_model_summary"]:
                continue
            oof = reg["per_model_summary"][mod]["oof"]
            u = unc_map.get(mod)
            b = bin_map.get(mod)
            b5 = b.get("threshold_5.0", {}) if b else {}
            b7 = b.get("threshold_7.0", {}) if b else {}
            rows.append({
                "protocolo": "v2_kfold",
                "dataset": ds,
                "split": sp,
                "modelo": mod,
                "n_pares_oof": int(reg["n_oof_pairs"]),
                "oof_ci": fmt_float(oof["ci"]),
                "oof_mse": fmt_float(oof["mse"]),
                "oof_rmse": fmt_float(oof["rmse"]),
                "oof_rm2": fmt_float(oof["rm2"]),
                "n_nao_censurados_pkd_gt5": str(int(u["num_uncensored_pairs"])) if u else "não aplicável (KIBA)",
                "nao_cens_ci": fmt_float(u["ci"]) if u else "não aplicável (KIBA)",
                "nao_cens_mse": fmt_float(u["mse"]) if u else "não aplicável (KIBA)",
                "nao_cens_rm2": fmt_float(u["rm2"]) if u else "não aplicável (KIBA)",
                "auroc_gt5": fmt_float(b5.get("auroc")) if b else "não aplicável (KIBA)",
                "aupr_gt5": fmt_float(b5.get("aupr")) if b else "não aplicável (KIBA)",
                "auroc_ge7": fmt_float(b7.get("auroc")) if b else "não aplicável (KIBA)",
                "aupr_ge7": fmt_float(b7.get("aupr")) if b else "não aplicável (KIBA)",
                "status_limiar_binario": "a verificar (Davis)" if ds == "davis" else "não aplicável (KIBA)",
                "fonte": f"docs/paper/phase3_audits.json (v2_kfold.regimes.{rkey})",
            })

    # Bloco v1_seeds: apenas sanidade Davis random para gin_esm_mlp (média ± DP nas 5 seeds; sem OOF pooled)
    dr = _get_regime(pa, "v1_seeds", "davis/random")
    u_dr = dr.get("uncensored_gin_esm_mlp")
    b_dr = dr.get("binary_gin_esm_mlp")
    if u_dr and b_dr:
        rows.append({
            "protocolo": "v1_seeds (média 5 seeds)",
            "dataset": "davis",
            "split": "random",
            "modelo": "gin_esm_mlp",
            "n_pares_oof": "não aplicável (v1_seeds)",
            "oof_ci": "não aplicável (v1_seeds)",
            "oof_mse": "não aplicável (v1_seeds)",
            "oof_rmse": "não aplicável (v1_seeds)",
            "oof_rm2": "não aplicável (v1_seeds)",
            "n_nao_censurados_pkd_gt5": "por seed (v1_seeds)",
            "nao_cens_ci": fmt_mean_std(u_dr["ci"]["mean"], u_dr["ci"]["std"]),
            "nao_cens_mse": fmt_mean_std(u_dr["mse"]["mean"], u_dr["mse"]["std"]),
            "nao_cens_rm2": fmt_mean_std(u_dr["rm2"]["mean"], u_dr["rm2"]["std"]),
            "auroc_gt5": fmt_mean_std(b_dr["auroc_gt5"]["mean"], b_dr["auroc_gt5"]["std"]),
            "aupr_gt5": fmt_mean_std(b_dr["aupr_gt5"]["mean"], b_dr["aupr_gt5"]["std"]),
            "auroc_ge7": fmt_mean_std(b_dr["auroc_ge7"]["mean"], b_dr["auroc_ge7"]["std"]),
            "aupr_ge7": fmt_mean_std(b_dr["aupr_ge7"]["mean"], b_dr["aupr_ge7"]["std"]),
            "status_limiar_binario": "a verificar (Davis random sanidade)",
            "fonte": "docs/paper/phase3_audits.json (v1_seeds.regimes.davis/random)",
        })

    df_t11 = pd.DataFrame(rows)
    return write_table(
        "T11_phase3_global_oof_and_binary",
        "T11 — Métricas OOF Globais (`v2_kfold`), Subconjunto Não Censurado (`pKd > 5.0`) e Classificação Binária (Davis)",
        df_t11,
        "docs/paper/phase3_audits.json e OOF CSVs",
        tables_dir,
        extra_notes=[
            "Métricas `OOF globais` são calculadas exclusivamente no protocolo `v2_kfold` (partição exata e única sem duplicatas). Em `v1_seeds` (`davis/random`), reportam-se apenas médias ± DP nas 5 seeds.",
            "Em Davis, `9.125` de `30.056` pares têm `pKd > 5.0` (não censurados); ~`69.64%` dos pares estão censurados em `pKd = 5.0` (Kd = 10 µM).",
            "Os limiares binários `pKd > 5.0` e `pKd >= 7.0` para AUROC/AUPR permanecem marcados como `a verificar` quanto a referências externas da literatura não confirmadas nesta auditoria.",
        ],
    )


def build_t12(pa: dict[str, Any], tables_dir: Path) -> dict[str, Path]:
    """T12: Desempenho por dobra/seed (0..4) lado a lado para todos os modelos nas mesmas unidades."""
    model_order = [
        "gin_esm_mlp",
        "ecfp_esm_mlp",
        "shuf0",
        "shuf1",
        "shuf2",
        "tsoff1000",
        "graphdta_gin",
        "graphdta_cnn3",
        "ridge_esm",
        "ridge_esm_target_only",
        "knn_esm_target",
        "knn_kmer_target",
        "knn_tanimoto_drug",
        "drug_mean_predictor",
        "target_mean_predictor",
        "mean_predictor",
    ]
    rows = []
    for proto, rkey, _papel in _REGIME_SPECS:
        reg = _get_regime(pa, proto, rkey)
        ds, sp = reg["dataset"], reg["split"]
        unit_map = reg["per_seed_models"] if proto == "v1_seeds" else reg["per_fold_by_model"]
        unit_key = "seed" if proto == "v1_seeds" else "fold"
        for mod in model_order:
            if mod not in unit_map:
                continue
            entries = sorted(unit_map[mod], key=lambda x: int(x[unit_key]))
            by_u = {int(e[unit_key]): e for e in entries}
            cis = [float(by_u[u]["ci"]) for u in range(5) if u in by_u]
            mses = [float(by_u[u]["mse"]) for u in range(5) if u in by_u]
            m_ci, _ = sample_mean_std(cis)
            m_mse, _ = sample_mean_std(mses)
            obs = "todos os 5 blocos completos"
            if mod == "gin_esm_mlp" and ds == "davis" and sp == "cold_drug":
                obs = "dobra 2 extrema (MSE=1.3670, CI=0.5720)"
            elif mod == "gin_esm_mlp" and ds == "davis" and sp == "cold_target":
                obs = "dobra 1 estendida 500->1000 ep."
            row: dict[str, Any] = {
                "protocolo": proto,
                "dataset": ds,
                "split": sp,
                "modelo": mod,
            }
            for u in range(5):
                eu = by_u.get(u)
                row[f"u{u}_ci"] = fmt_float(eu["ci"]) if eu else NA_TEXT
                row[f"u{u}_mse"] = fmt_float(eu["mse"]) if eu else NA_TEXT
            row["media_ci"] = fmt_float(m_ci)
            row["media_mse"] = fmt_float(m_mse)
            row["observacao"] = obs
            row["fonte"] = f"docs/paper/phase3_audits.json ({proto}.regimes.{rkey})"
            rows.append(row)

    df_t12 = pd.DataFrame(rows)
    return write_table(
        "T12_phase3_per_fold_or_seed",
        "T12 — Métricas Por Unidade (Seeds `0..4` em `v1_seeds`; Dobras `0..4` em `v2_kfold`) para Inspeção Lado a Lado",
        df_t12,
        "docs/paper/phase3_audits.json",
        tables_dir,
        extra_notes=[
            "Colunas `u0..u4` representam `seed_0..seed_4` no bloco `v1_seeds` e `fold_0..fold_4` no bloco `v2_kfold`.",
            "Permite inspecionar diretamente se uma unidade (ex.: dobra 2 de `davis/cold_drug`) foi difícil para todos os modelos ou seletivamente para `gin_esm_mlp`.",
        ],
    )


def build_t13(pa: dict[str, Any], tables_dir: Path) -> dict[str, Path]:
    """T13: Curva de generalização, diferenças pareadas por unidade e bootstrap em clusters (IC 95% + fração favorável)."""
    comp_order = [
        "ridge_esm",
        "knn_kmer_target",
        "knn_tanimoto_drug",
        "drug_mean_predictor",
        "target_mean_predictor",
        "ridge_esm_target_only",
        "knn_esm_target",
        "graphdta_gin",
        "graphdta_cnn3",
    ]
    rows = []
    for proto, rkey, papel in _REGIME_SPECS:
        reg = _get_regime(pa, proto, rkey)
        ds, sp = reg["dataset"], reg["split"]
        if proto == "v1_seeds":
            paired_map = reg["paired_vs_baselines_seeds"]
            boot_map = reg.get("bootstrap_vs_baselines_pooled_clusters") or {}
            best_non = reg.get("best_non_esm_mse") or NA_TEXT
            u_boot = reg.get("cluster_col") or "não aplicável (random)"
            n_cl = reg.get("n_clusters_pooled")
            g_sum = reg["model_summary"]["gin_esm_mlp"]
            gin_ci_str = fmt_mean_std(g_sum["ci"]["mean"], g_sum["ci"]["std"])
            gin_mse_str = fmt_mean_std(g_sum["mse"]["mean"], g_sum["mse"]["std"])
        else:
            paired_map = reg["paired_vs_baselines_folds"]
            boot_map = reg.get("bootstrap_vs_baselines") or {}
            best_non = reg.get("best_non_esm_oof_mse") or NA_TEXT
            u_boot = reg.get("group_col") or NA_TEXT
            n_cl = reg.get("n_clusters")
            g_sum = reg["per_model_summary"]["gin_esm_mlp"]
            gin_ci_str = fmt_mean_std(g_sum["ci_per_fold"]["mean"], g_sum["ci_per_fold"]["std"])
            gin_mse_str = fmt_mean_std(g_sum["mse_per_fold"]["mean"], g_sum["mse_per_fold"]["std"])

        n_cl_str = str(int(n_cl)) if n_cl is not None else "não aplicável (random)"

        for base in comp_order:
            if base not in paired_map:
                continue
            pr = paired_map[base]
            bt = boot_map.get(base)
            dm_ic, dm_f, dc_ic, dc_f = _fmt_boot_ic_frac(bt)
            if bt:
                dm_lo, dm_hi = bt["delta_mse"]["ci_95"]
                dc_lo, dc_hi = bt["delta_ci"]["ci_95"]
                excl = f"dMSE={'sim' if dm_hi < 0 else 'não'}; dCI={'sim' if dc_lo > 0 else 'não'}"
            else:
                excl = "não aplicável (sem bootstrap)"

            rows.append({
                "protocolo": proto,
                "papel_no_estudo": papel,
                "dataset": ds,
                "split": sp,
                "gin_ci_media_dp": gin_ci_str,
                "gin_mse_media_dp": gin_mse_str,
                "baseline_comparado": base,
                "e_melhor_nao_esm": "sim" if base == best_non else "não",
                "unidade_bootstrap": u_boot,
                "n_unidades_pareadas": int(pr["n_units"]),
                "n_clusters_bootstrap": n_cl_str,
                "delta_mse_pareado_media_dp": fmt_mean_std(pr["delta_mse_mean"], pr["delta_mse_std"], signed=True),
                "delta_ci_pareado_media_dp": fmt_mean_std(pr["delta_ci_mean"], pr["delta_ci_std"], signed=True),
                "vitorias_mse": f"{int(pr['wins_mse'])}/{int(pr['n_units'])}",
                "vitorias_ci": f"{int(pr['wins_ci'])}/{int(pr['n_units'])}",
                "boot_delta_mse_ic95": dm_ic,
                "boot_frac_favoravel_mse": dm_f,
                "boot_delta_ci_ic95": dc_ic,
                "boot_frac_favoravel_ci": dc_f,
                "ic95_exclui_zero_favoravel": excl,
                "fonte": f"docs/paper/phase3_audits.json ({proto}.regimes.{rkey})",
            })

    df_t13 = pd.DataFrame(rows)
    return write_table(
        "T13_phase3_paired_and_generalization",
        "T13 — Curva de Generalização, Diferenças Pareadas (`gin_esm_mlp` − Baseline) e Bootstrap por Clusters (`n=1000`, `seed=42`)",
        df_t13,
        "docs/paper/phase3_audits.json",
        tables_dir,
        extra_notes=[
            "Convenção de sinal: `ΔMSE = MSE(gin_esm_mlp) − MSE(baseline)` (negativo favorece `gin_esm_mlp`); `ΔCI = CI(gin_esm_mlp) − CI(baseline)` (positivo favorece `gin_esm_mlp`).",
            "Reamostragem em bloco (`n=1000`, `seed=42`) por cluster de sequência nos regimes `seqid`, por SMILES canônico no `cold_drug`, e por sequência única nos `cold_target` padrão; em `v1_seeds`, os clusters são empilhados das 5 seeds.",
            "Frações favoráveis são reportadas como contagem exata `[w/1000]`; nenhuma p-value extrema é utilizada.",
        ],
    )


def build_t14(pa: dict[str, Any], tables_dir: Path) -> dict[str, Path]:
    """T14: Controles diagnósticos exploratórios no Davis (--shuffle-esm, ecfp_esm_mlp, tsoff1000, post-hoc sem dobra 2)."""
    rows = []

    def _add_v2_ctrl_row(
        reg_key: str,
        label: str,
        tipo: str,
        n_u: Any,
        ci_str: str,
        mse_str: str,
        vs_gin_boot: dict[str, Any] | None,
        vs_dm_boot: dict[str, Any] | None,
        interp: str,
        src: str,
    ) -> None:
        ds, sp = reg_key.split("/", 1)
        dm_g_ic, dm_g_f, dc_g_ic, dc_g_f = _fmt_boot_ic_frac(vs_gin_boot)
        dm_d_ic, dm_d_f, dc_d_ic, dc_d_f = _fmt_boot_ic_frac(vs_dm_boot)
        rows.append({
            "dataset": ds,
            "split": sp,
            "controle_ou_modelo": label,
            "tipo_analise": tipo,
            "n_unidades": str(n_u),
            "ci_media_dp": ci_str,
            "mse_media_dp": mse_str,
            "boot_dMSE_vs_gin_ic95": dm_g_ic,
            "boot_dCI_vs_gin_ic95": dc_g_ic,
            "boot_frac_vs_gin_mse_ci": f"MSE {dm_g_f}; CI {dc_g_f}" if vs_gin_boot else NA_TEXT,
            "boot_dMSE_vs_drug_mean_ic95": dm_d_ic,
            "boot_dCI_vs_drug_mean_ic95": dc_d_ic,
            "boot_frac_vs_drug_mean_mse_ci": f"MSE {dm_d_f}; CI {dc_d_f}" if vs_dm_boot else NA_TEXT,
            "interpretacao_exploratoria": interp,
            "fonte": src,
        })

    # 1. Davis cold_target: gin_esm_mlp, ecfp_esm_mlp, shuf0/1/2, ridge_esm, drug_mean_predictor, knn_kmer_target
    r_ct = _get_regime(pa, "v2_kfold", "davis/cold_target")
    ec_ct = r_ct["extra_controls_bootstrap"]
    bt_ct = r_ct["bootstrap_vs_baselines"]
    pms_ct = r_ct["per_model_summary"]
    ct_defs = [
        ("gin_esm_mlp", "referencia_principal", None, bt_ct.get("drug_mean_predictor"), "referência com ESM real no Davis cold_target"),
        ("ecfp_esm_mlp", "controle_arquitetura_ligante", ec_ct["ecfp_esm_mlp"]["gin_esm_mlp"], ec_ct["ecfp_esm_mlp"]["drug_mean_predictor"], "equivalente ao GIN+ESM (dMSE=-0.0121, IC cruza 0); sinal não depende do grafo molecular"),
        ("shuf0", "controle_permutacao_esm", ec_ct["shuf0"]["gin_esm_mlp"], ec_ct["shuf0"]["drug_mean_predictor"], "ESM embaralhado derruba abaixo do gin_esm_mlp e do drug_mean_predictor (IC exclui 0)"),
        ("shuf1", "controle_permutacao_esm", ec_ct["shuf1"]["gin_esm_mlp"], ec_ct["shuf1"]["drug_mean_predictor"], "ESM embaralhado derruba abaixo do gin_esm_mlp e do drug_mean_predictor (IC exclui 0)"),
        ("shuf2", "controle_permutacao_esm", ec_ct["shuf2"]["gin_esm_mlp"], ec_ct["shuf2"]["drug_mean_predictor"], "ESM embaralhado derruba abaixo do gin_esm_mlp e do drug_mean_predictor (IC exclui 0)"),
        ("ridge_esm", "baseline_linear_esm", None, None, "baseline linear com mesma representação ESM"),
        ("drug_mean_predictor", "baseline_marginal", None, None, "teto sem uso real da proteína; superado pelo modelo real mas não pelos permutados"),
        ("knn_kmer_target", "baseline_nao_esm", None, None, "melhor baseline não-ESM no regime"),
    ]
    for mk, tipo, bg, bdm, interp in ct_defs:
        ms = pms_ct[mk]
        lbl = f"gin_esm_mlp_{mk}" if mk.startswith("shuf") else mk
        _add_v2_ctrl_row(
            "davis/cold_target",
            lbl,
            tipo,
            5,
            fmt_mean_std(ms["ci_per_fold"]["mean"], ms["ci_per_fold"]["std"]),
            fmt_mean_std(ms["mse_per_fold"]["mean"], ms["mse_per_fold"]["std"]),
            bg,
            bdm,
            interp,
            "docs/paper/phase3_audits.json (v2_kfold.regimes.davis/cold_target)",
        )

    # 2. Davis cold_drug: 5 folds + ecfp + post-hoc sem dobra 2 + dobra 2 tsoff1000
    r_cd = _get_regime(pa, "v2_kfold", "davis/cold_drug")
    ec_cd = r_cd["extra_controls_bootstrap"]
    pms_cd = r_cd["per_model_summary"]
    cd_defs = [
        ("gin_esm_mlp", "referencia_5_dobras", None, r_cd["bootstrap_vs_baselines"].get("drug_mean_predictor"), "inclui dobra 2 extrema (MSE=1.3670, best_ep=5)"),
        ("ecfp_esm_mlp", "controle_arquitetura_ligante", ec_cd["ecfp_esm_mlp"]["gin_esm_mlp"], ec_cd["ecfp_esm_mlp"]["drug_mean_predictor"], "dobra 2 melhor que gin_esm_mlp (MSE=1.0419 vs 1.3670), mas empata na média em 5 dobras; folds 3,4 com best_ep=3"),
        ("ridge_esm", "baseline_linear_esm", None, None, "referência linear em 5 dobras"),
        ("knn_tanimoto_drug", "baseline_nao_esm_ci", None, None, "maior CI médio em 5 dobras (0.7065)"),
        ("knn_kmer_target", "baseline_nao_esm_mse", None, None, "menor MSE médio em 5 dobras (0.7310)"),
    ]
    for mk, tipo, bg, bdm, interp in cd_defs:
        ms = pms_cd[mk]
        _add_v2_ctrl_row(
            "davis/cold_drug",
            mk,
            tipo,
            5,
            fmt_mean_std(ms["ci_per_fold"]["mean"], ms["ci_per_fold"]["std"]),
            fmt_mean_std(ms["mse_per_fold"]["mean"], ms["mse_per_fold"]["std"]),
            bg,
            bdm,
            interp,
            "docs/paper/phase3_audits.json (v2_kfold.regimes.davis/cold_drug)",
        )

    # Post-hoc 4 dobras (sem dobra 2) e dobra 2 com tsoff1000
    no_f2 = pa["v2_kfold"]["davis_cold_drug_descriptive_no_fold2"]
    for mk in ("gin_esm_mlp", "ecfp_esm_mlp", "ridge_esm", "knn_kmer_target", "knn_tanimoto_drug"):
        e4 = no_f2[mk]
        _add_v2_ctrl_row(
            "davis/cold_drug",
            f"{mk}_sem_fold2",
            "analise_post_hoc_nao_confirmativa",
            "4 (excl. fold 2)",
            fmt_mean_std(e4["ci_4folds"]["mean"], e4["ci_4folds"]["std"]),
            fmt_mean_std(e4["mse_4folds"]["mean"], e4["mse_4folds"]["std"]),
            None,
            None,
            "análise descritiva post-hoc removendo dobra 2 (~13-14 fármacos/dobra)",
            "docs/paper/phase3_audits.json (v2_kfold.davis_cold_drug_descriptive_no_fold2)",
        )

    f2_orig = next(e for e in r_cd["per_fold_by_model"]["gin_esm_mlp"] if int(e["fold"]) == 2)
    f2_ts = pa["v2_kfold"]["davis_cold_drug_fold2_tsoff1000"]["test_metrics_full"]
    _add_v2_ctrl_row(
        "davis/cold_drug",
        "gin_esm_mlp_fold2_orig",
        "diagnostico_semente_dobra2",
        "1 (fold 2)",
        fmt_float(f2_orig["ci"]),
        fmt_float(f2_orig["mse"]),
        None,
        None,
        "execução original com parada em best_epoch=5",
        "docs/paper/phase3_audits.json (v2_kfold.regimes.davis/cold_drug)",
    )
    _add_v2_ctrl_row(
        "davis/cold_drug",
        "gin_esm_mlp_tsoff1000_fold2",
        "controle_semente_tsoff1000",
        "1 (fold 2)",
        fmt_float(f2_ts["ci"]),
        fmt_float(f2_ts["mse"]),
        None,
        None,
        "mesma partição dobra 2 com seed+1000 (best_epoch=31): reduz parte da perda, sugerindo instabilidade de inicialização em ~14 fármacos",
        "docs/paper/phase3_audits.json (v2_kfold.davis_cold_drug_fold2_tsoff1000)",
    )

    # 3. Davis cold_target_seqid30: gin_esm_mlp vs tsoff1000
    r_s30 = _get_regime(pa, "v2_kfold", "davis/cold_target_seqid30")
    ec_s30 = r_s30["extra_controls_bootstrap"]
    pms_s30 = r_s30["per_model_summary"]
    for mk, tipo, bg, bdm, interp in [
        ("gin_esm_mlp", "referencia_principal", None, r_s30["bootstrap_vs_baselines"].get("drug_mean_predictor"), "referência seed padrão em seqid30"),
        ("tsoff1000", "controle_semente_tsoff1000", ec_s30["tsoff1000"]["gin_esm_mlp"], ec_s30["tsoff1000"]["drug_mean_predictor"], "seed+1000 é equivalente em bootstrap ao original (dMSE=+0.0065, dCI=-0.0029)"),
    ]:
        ms = pms_s30[mk]
        lbl = f"gin_esm_mlp_{mk}" if mk == "tsoff1000" else mk
        _add_v2_ctrl_row(
            "davis/cold_target_seqid30",
            lbl,
            tipo,
            5,
            fmt_mean_std(ms["ci_per_fold"]["mean"], ms["ci_per_fold"]["std"]),
            fmt_mean_std(ms["mse_per_fold"]["mean"], ms["mse_per_fold"]["std"]),
            bg,
            bdm,
            interp,
            "docs/paper/phase3_audits.json (v2_kfold.regimes.davis/cold_target_seqid30)",
        )

    df_t14 = pd.DataFrame(rows)
    return write_table(
        "T14_phase3_diagnostic_controls",
        "T14 — Controles Diagnósticos Exploratórios no Davis (`--shuffle-esm`, `ecfp_esm_mlp`, `--train-seed-offset 1000` e Post-hoc sem Dobra 2)",
        df_t14,
        "docs/paper/phase3_audits.json e outputs/results/runs/*.json",
        tables_dir,
        extra_notes=[
            "Convenção de delta no bootstrap vs `gin_esm_mlp`: `dMSE = MSE(controle) − MSE(gin_esm_mlp)` (positivo indica piora do controle); `dCI = CI(controle) − CI(gin_esm_mlp)` (negativo indica queda no controle).",
            "Os 3 controles com ESM embaralhado (`shuf0..2`) caem abaixo de `gin_esm_mlp` e também abaixo de `drug_mean_predictor`, indicando dependência de alinhamento real proteína-fármaco, porém com memorização precoce (`best_epoch <= 13` em 7 de 15 execuções).",
            "A substituição do GNN por ECFP4 (`ecfp_esm_mlp`) não produz diferença com significância em bootstrap nem no `cold_target` nem no `cold_drug`, mantendo o papel do GIN sobre ECFP4 inconclusivo sob mean-pooling.",
            "Resultados sem a dobra 2 (`_sem_fold2`) são puramente descritivos/post-hoc e não substituem a média das 5 dobras pré-especificadas.",
        ],
    )


def build_t15(repo_root: Path, data_root: Path, runs_all: list[dict[str, Any]], tables_dir: Path) -> dict[str, Path]:
    """T15: Configuração efetiva e tempo computacional dos 17 grupos da Fase 3 (81 execuções)."""
    cfg = yaml.safe_load((repo_root / "configs" / "base.yaml").read_text(encoding="utf-8"))
    snap = (repo_root / "docs" / "colab_env_snapshot.txt").read_text(encoding="utf-8").splitlines()
    gpu_line = next((l.split(":", 1)[1].strip() for l in snap if "GPU [0]" in l), NA_TEXT)
    py_ver = next((l.split(":", 1)[1].strip() for l in snap if l.startswith("Python")), NA_TEXT)
    torch_ver = next((l.split("==", 1)[1].strip() for l in snap if l.startswith("torch==")), NA_TEXT)
    esm_ver = next((l.split("==", 1)[1].strip() for l in snap if l.startswith("fair-esm==")), NA_TEXT)
    env_summary = f"Python {py_ver}; torch {torch_ver}; fair-esm {esm_ver}; snapshot={gpu_line}"

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in runs_all:
        if str(r.get("model", "")).startswith("graphdta_"):
            continue
        grouped[(r["_proto"], r["_model_norm"], r["dataset"], r["split"])].append(r)

    rows = []
    for (proto, mod_norm, ds, sp), grp_runs in sorted(grouped.items()):
        ec0 = grp_runs[0]["effective_config"]
        eps = sorted({int(r["effective_config"]["epochs"]) for r in grp_runs})
        ep_str = "/".join(str(e) for e in eps)
        pat_eff = int(ec0["davis_cold_patience"]) if (ds == "davis" and sp != "random") else int(ec0["patience"])
        is_davis_ct_main = (mod_norm == "gin_esm_mlp" and ds == "davis" and sp == "cold_target")

        if is_davis_ct_main:
            clean_secs = [float(r["sec_per_epoch"]) for r in grp_runs if int(r["_unit"]) != 1]
            sec_ep_str = fmt_mean_std(*sample_mean_std(clean_secs), decimals=3) + " (folds 0,2,3,4)"
            obs = "folds 0,2,3,4 em epochs=500; fold 1 estendido para 1000 ep. (sec/ep=0.260 excluído por retomada)"
        else:
            secs = [float(r["sec_per_epoch"]) for r in grp_runs]
            sec_ep_str = fmt_mean_std(*sample_mean_std(secs), decimals=3)
            obs = "configuração padrão ADR #15" if len(grp_runs) == 5 else "execução diagnóstica única (fold 2)"

        tot_sec = sum(float(r["elapsed_sec"]) for r in grp_runs)
        tot_hr = tot_sec / 3600.0
        drug_dim = grp_runs[0].get("drug_in_dim", 78)
        drug_rep = f"GIN ({drug_dim}d)" if int(drug_dim) == 78 else f"ECFP4 ({drug_dim}d)"
        esm_m = grp_runs[0].get("esm_model", "esm2_t33_650M_UR50D")
        esm_d = grp_runs[0].get("esm_dim", 1280)
        hp_str = (
            f"lr={ec0['lr']}, wd={ec0['weight_decay']}, bs={ec0['batch_size']}, "
            f"esm_do={ec0['esm_dropout_in']}, prot_do={ec0['prot_dropout']}, mlp_do={ec0['mlp_dropout']}, "
            f"prot_dim={ec0['prot_output_dim']}"
        )

        rows.append({
            "protocolo": proto,
            "modelo": mod_norm,
            "dataset": ds,
            "split": sp,
            "n_runs": len(grp_runs),
            "drug_esm_featurizer": f"{drug_rep} + {esm_m} ({esm_d}d) v={cfg['featurizer']['version']}",
            "hiperparametros_efetivos": hp_str,
            "epochs_cfg": ep_str,
            "patience_efetiva": pat_eff,
            "ambiente_snapshot": env_summary,
            "sec_per_epoch_medio": sec_ep_str,
            "tempo_total_grupo_s": f"{tot_sec:.1f}",
            "tempo_total_grupo_h": f"{tot_hr:.2f} h",
            "observacao": obs,
            "fonte": f"configs/base.yaml; docs/colab_env_snapshot.txt; outputs/results/runs/{grp_runs[0]['model']}_{ds}_{sp}_*.json",
        })

    df_t15 = pd.DataFrame(rows)
    return write_table(
        "T15_phase3_configuration",
        "T15 — Configuração Efetiva, Ambiente e Tempo Computacional da Fase 3 (17 Grupos / 81 Runs)",
        df_t15,
        "configs/base.yaml, docs/colab_env_snapshot.txt e outputs/results/runs/physdual_*.json",
        tables_dir,
        extra_notes=[
            "Todas as 62 execuções pós-ADR #15 (KIBA, `davis/random` e controles) seguiram integralmente `epochs=1000`, `lr=0.0005`, `weight_decay=0.001`, `batch_size=512`, `esm_dropout_in=0.4`, `prot_dropout=0.3`, `mlp_dropout=0.3`, `prot_output_dim=128`.",
            "Para `gin_esm_mlp` em `davis/cold_target`, a dobra 1 (`0.260 s/época`) foi excluída da média `sec_per_epoch` por representar apenas o trecho retomado pós-época 500 (`157.24 s / 605 épocas`).",
        ],
    )


def build_all_tables() -> dict[str, dict[str, Path]]:
    repo_root, data_root = resolve_roots()
    dirs = ensure_paper_dirs(repo_root)
    tables_dir = dirs["tables"]
    runs_phase12 = _load_all_runs(data_root)
    runs_all = _load_all_runs_all_phases(data_root)
    phase3_audits = load_json_strict(repo_root / "docs" / "paper" / "phase3_audits.json")

    outputs = {
        "T1": build_t1(repo_root, data_root, tables_dir),
        "T2": build_t2(data_root, tables_dir),
        "T3": build_t3(data_root, runs_phase12, tables_dir),
        "T4": build_t4(data_root, runs_phase12, tables_dir),
        "T5": build_t5(data_root, tables_dir),
        "T6": build_t6(data_root, tables_dir),
        "T7": build_t7(runs_all, tables_dir),
        "T8": build_t8(repo_root, data_root, runs_phase12, tables_dir),
        "T9": build_t9(phase3_audits, tables_dir),
        "T10": build_t10(phase3_audits, tables_dir),
        "T11": build_t11(phase3_audits, tables_dir),
        "T12": build_t12(phase3_audits, tables_dir),
        "T13": build_t13(phase3_audits, tables_dir),
        "T14": build_t14(phase3_audits, tables_dir),
        "T15": build_t15(repo_root, data_root, runs_all, tables_dir),
    }

    # Verificacao estrita de ausencia de NaN em todos os CSVs gerados
    for tname, paths in outputs.items():
        df_chk = pd.read_csv(paths["csv"], keep_default_na=False)
        if df_chk.isna().any().any():
            raise ValueError(f"Tabela {tname} contem valores NaN em {paths['csv']}")
        for col in df_chk.columns:
            if df_chk[col].astype(str).str.lower().isin(["nan", "none", ""]).any():
                raise ValueError(f"Tabela {tname} contem string vazia/nan/none na coluna {col}")

    return outputs


if __name__ == "__main__":
    out = build_all_tables()
    for k, paths in out.items():
        print(f"[OK] {k}: {paths['csv'].name}, {paths['md'].name}, {paths['tex'].name}")
