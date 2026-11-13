"""Passo 1: Consolida as tabelas T1..T8 em CSV, Markdown e LaTeX em `docs/paper/tables/`."""

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


def _load_all_runs(data_root: Path) -> list[dict[str, Any]]:
    runs_dir = data_root / "outputs" / "results" / "runs"
    files = sorted(runs_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"Nenhum JSON de run encontrado em {runs_dir}")
    runs = []
    for f in files:
        d = load_json_strict(f)
        d["_filename"] = f.name
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
        if agg_path.name.endswith("_kfold.json"):
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
            "bootstrap_delta_mse_ic95_frac_expl": f"{bs_mse['mean']:+.4f} [{bs_mse['ci_95'][0]:.4f}, {bs_mse['ci_95'][1]:.4f}] | {bs_mse['wins']}/{bs_mse['total_bootstraps']} fav. (exploratório)",
            "bootstrap_delta_ci_ic95_frac_expl": f"{bs_ci['mean']:+.4f} [{bs_ci['ci_95'][0]:.4f}, {bs_ci['ci_95'][1]:.4f}] | {bs_ci['wins']}/{bs_ci['total_bootstraps']} fav. (exploratório)",
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
        if agg_path.name.endswith("_kfold.json"):
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


def build_t7(runs: list[dict[str, Any]], tables_dir: Path) -> dict[str, Path]:
    rows = []
    for r in sorted(runs, key=lambda x: (x["_proto"], x["dataset"], x["model"], x["split"], x["_unit"])):
        ec = r.get("effective_config", {})
        ep_cfg = int(ec.get("epochs", 500))
        best_ep = int(r["best_epoch"])
        tot_ep = int(r["total_epochs_run"])
        alerta_08 = best_ep > 0.8 * ep_cfg
        obs = ""
        if r["model"] == "graphdta_gin" and r["dataset"] == "davis" and r["split"] == "cold_target":
            if best_ep <= 15:
                obs = "colapso precoce (best_epoch <= 13)"
            else:
                obs = "outlier tardio no cold_target"
        elif r["dataset"] == "kiba" and r["split"] == "random" and ep_cfg == 1000:
            obs = "retomado 500->1000 ep.; sec/ep distorcido (só retomada)"
        elif r["dataset"] == "kiba" and r["split"] == "random" and ep_cfg == 500:
            obs = "ES <500 ep. (equiv. teto efetivo 1000)"
        elif alerta_08:
            obs = "proximo ao teto de epocas"
        else:
            obs = "normal"

        rows.append({
            "protocolo": r["_proto"],
            "modelo": r["model"],
            "dataset": r["dataset"],
            "split": r["split"],
            "unidade": f"{'fold' if r['_proto']=='v2_kfold' else 'seed'}_{r['_unit']}",
            "best_epoch": best_ep,
            "total_epochs_run": tot_ep,
            "epochs_cfg": ep_cfg,
            "sec_per_epoch": fmt_float(r.get("sec_per_epoch"), decimals=3),
            "elapsed_sec": fmt_float(r.get("elapsed_sec"), decimals=2),
            "flag_gt_0_8_epochs": str(bool(alerta_08)).lower(),
            "near_epoch_ceiling_warn_json": str(bool(r.get("near_epoch_ceiling_warn", False))).lower(),
            "observacao": obs,
            "fonte": f"outputs/results/runs/{r['_filename']}",
        })

    df_t7 = pd.DataFrame(rows)
    return write_table(
        "T7_training_diagnostics",
        "T7 — Diagnóstico de Treinamento por Run (65 Execuções)",
        df_t7,
        "outputs/results/runs/*.json",
        tables_dir,
        extra_notes=[
            "Em Davis `cold_target`, `graphdta_gin` apresentou `best_epoch` entre `2` e `7` em todas as 5 seeds de `v1_seeds`, e `[236, 13, 3, 6, 9]` nas 5 dobras de `v2_kfold`.",
            "Em `kiba_random`, o teto efetivo do protocolo é `1000` épocas para todas as 5 seeds: `seeds 0, 2, 4` encerraram por early stopping (`patience=50`) em `344`, `336` e `393` épocas (muito antes de 500, não requerendo re-execução), enquanto `seeds 1` e `3` foram retomadas até 1000 (`best_epoch = 774` e `509`). Nas seeds `1` e `3`, `elapsed_sec` (`532.43 s` e `1311.70 s`) e `sec_per_epoch` (`0.646 s` e `2.347 s`) refletem apenas o trecho retomado pós-500 dividido pelo total acumulado.",
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


def build_all_tables() -> dict[str, dict[str, Path]]:
    repo_root, data_root = resolve_roots()
    dirs = ensure_paper_dirs(repo_root)
    tables_dir = dirs["tables"]
    runs = _load_all_runs(data_root)

    outputs = {
        "T1": build_t1(repo_root, data_root, tables_dir),
        "T2": build_t2(data_root, tables_dir),
        "T3": build_t3(data_root, runs, tables_dir),
        "T4": build_t4(data_root, runs, tables_dir),
        "T5": build_t5(data_root, tables_dir),
        "T6": build_t6(data_root, tables_dir),
        "T7": build_t7(runs, tables_dir),
        "T8": build_t8(repo_root, data_root, runs, tables_dir),
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
