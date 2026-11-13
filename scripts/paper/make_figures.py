"""Passo 2: Gera figuras F1..F6 em PDF e PNG (paleta Okabe-Ito) + `CAPTIONS.md`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ensure_paper_dirs, load_json_strict, resolve_roots

# Paleta Okabe-Ito (acessivel a daltonicos)
BLUE = "#0072B2"
VERMILION = "#D55E00"
GREEN = "#009E73"
ORANGE = "#E69F00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
GRAY = "#555555"
BLACK = "#111111"


def _setup_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.35,
        "figure.dpi": 150,
    })


def _save_fig(fig: plt.Figure, figures_dir: Path, stem: str) -> dict[str, Path]:
    pdf = figures_dir / f"{stem}.pdf"
    png = figures_dir / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": pdf, "png": png}


def make_f1(data_root: Path, figures_dir: Path) -> dict[str, Path]:
    davis = pd.read_csv(data_root / "data" / "processed" / "davis_clean.csv")
    kiba = pd.read_csv(data_root / "data" / "processed" / "kiba_clean.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    # Davis
    ax = axes[0]
    vals_d = davis["affinity"].to_numpy()
    bins_d = np.linspace(5.0, 11.0, 31)
    ax.hist(vals_d, bins=bins_d, color=BLUE, edgecolor="white", alpha=0.88)
    n_cens = int((vals_d == 5.0).sum())
    pct_cens = 100.0 * n_cens / len(vals_d)
    ax.axvline(5.0, color=VERMILION, linestyle="--", linewidth=1.4)
    ax.annotate(
        f"Censura em pKd = 5,0\n{n_cens:,} pares ({pct_cens:.2f}%)",
        xy=(5.12, n_cens * 0.88),
        xytext=(6.2, n_cens * 0.78),
        arrowprops={"arrowstyle": "->", "color": VERMILION, "lw": 1.2},
        fontsize=8.5,
        color=VERMILION,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": VERMILION, "alpha": 0.9},
    )
    ax.set_title(f"Davis (N = {len(vals_d):,} pares)")
    ax.set_xlabel("Afinidade (pKd)")
    ax.set_ylabel("Contagem de pares")

    # KIBA
    ax = axes[1]
    vals_k = kiba["affinity"].to_numpy()
    ax.hist(vals_k, bins=45, color=GREEN, edgecolor="white", alpha=0.88)
    ax.axvline(float(np.mean(vals_k)), color=BLACK, linestyle="--", linewidth=1.2, label=f"Média = {np.mean(vals_k):.2f}")
    ax.axvline(float(np.median(vals_k)), color=ORANGE, linestyle=":", linewidth=1.4, label=f"Mediana = {np.median(vals_k):.2f}")
    ax.set_title(f"KIBA (N = {len(vals_k):,} pares)")
    ax.set_xlabel("KIBA score")
    ax.set_ylabel("Contagem de pares")
    ax.legend(frameon=True)

    fig.suptitle("F1 — Distribuição das Afinidades Experimentais (Davis e KIBA)", fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save_fig(fig, figures_dir, "F1_affinity_distributions")


def make_f2(data_root: Path, figures_dir: Path) -> dict[str, Path]:
    repo_root = resolve_roots()[0]
    cpu_audits = load_json_strict(repo_root / "docs" / "paper" / "cpu_audits.json")
    v1_pg = cpu_audits["v1_marginals_knn_and_paired"]["paired_groups"]
    res_dir = data_root / "outputs" / "results"

    # Carregar v1_seeds
    v1_specs = [
        ("davis", "random", "Davis\nrandom"),
        ("davis", "cold_drug", "Davis\ncold_drug"),
        ("davis", "cold_target", "Davis\ncold_target"),
        ("kiba", "random", "KIBA\nrandom*"),
        ("kiba", "cold_drug", "KIBA\ncold_drug"),
        ("kiba", "cold_target", "KIBA\ncold_target"),
    ]
    v1_data = {}
    for mod in ("graphdta_gin", "graphdta_cnn3"):
        for ds, sp, _ in v1_specs:
            p = res_dir / f"baseline_{mod}_{ds}_{sp}.json"
            if p.exists():
                v1_data[(mod, ds, sp)] = load_json_strict(p)

    # Carregar v2_kfold
    v2_specs = [("davis", "cold_drug", "Davis\ncold_drug (K-Fold)"), ("davis", "cold_target", "Davis\ncold_target (K-Fold)")]
    v2_data = {}
    for mod in ("graphdta_gin", "graphdta_cnn3"):
        for ds, sp, _ in v2_specs:
            p = res_dir / f"baseline_{mod}_{ds}_{sp}_kfold.json"
            if p.exists():
                v2_data[(mod, ds, sp)] = load_json_strict(p)

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.4), gridspec_kw={"height_ratios": [1.1, 1.0]})

    # Painel Superior: Protocolo v1 (seeds)
    x_v1 = np.arange(len(v1_specs))
    w = 0.35
    for metric_idx, (metric, ylabel, title_suffix) in enumerate([("ci", "Concordance Index (CI ↑)", "CI"), ("mse", "Erro Quadrático Médio (MSE ↓)", "MSE")]):
        ax = axes[0, metric_idx]
        gin_m, gin_s, cnn_m, cnn_s, mean_ref, marg_ref, knn_ref = [], [], [], [], [], [], []
        for ds, sp, _ in v1_specs:
            dg = v1_data[("graphdta_gin", ds, sp)]["aggregated"]
            gin_m.append(dg["test_metrics_full"][metric]["mean"])
            gin_s.append(dg["test_metrics_full"][metric]["std"])
            mean_ref.append(dg["trivial_baselines"]["mean_predictor"][metric]["mean"])

            pg = v1_pg[f"graphdta_gin:{ds}:{sp}"]
            rel_m = pg["relevant_baseline"]
            marg_ref.append(pg["marginal_baselines"][rel_m][metric]["mean"])
            rel_k = "knn_tanimoto_drug" if sp in ("cold_drug", "random") and ds == "kiba" else ("knn_kmer_target" if sp == "cold_target" or ds == "davis" else "knn_tanimoto_drug")
            if sp == "cold_drug":
                rel_k = "knn_tanimoto_drug"
            elif sp == "cold_target":
                rel_k = "knn_kmer_target"
            else:
                # No random, melhor kNN por metrica/split
                kt = pg["knn_baselines"]["knn_tanimoto_drug"][metric]["mean"]
                kk = pg["knn_baselines"]["knn_kmer_target"][metric]["mean"]
                rel_k = "knn_tanimoto_drug" if ((metric == "ci" and kt >= kk) or (metric == "mse" and kt <= kk)) else "knn_kmer_target"
            knn_ref.append(pg["knn_baselines"][rel_k][metric]["mean"])

            if ("graphdta_cnn3", ds, sp) in v1_data:
                dc = v1_data[("graphdta_cnn3", ds, sp)]["aggregated"]
                cnn_m.append(dc["test_metrics_full"][metric]["mean"])
                cnn_s.append(dc["test_metrics_full"][metric]["std"])
            else:
                cnn_m.append(np.nan)
                cnn_s.append(0.0)

        ax.bar(x_v1 - w / 2, gin_m, w, yerr=gin_s, capsize=3, color=BLUE, label="graphdta_gin (oficial)")
        mask_cnn = ~np.isnan(cnn_m)
        ax.bar(x_v1[mask_cnn] + w / 2, np.array(cnn_m)[mask_cnn], w, yerr=np.array(cnn_s)[mask_cnn], capsize=3, color=VERMILION, label="graphdta_cnn3 (variante)")
        ax.scatter(x_v1, mean_ref, marker="_", s=340, linewidths=1.8, color=GRAY, zorder=5, label="mean_predictor (média treino)")
        ax.scatter(x_v1, marg_ref, marker="_", s=460, linewidths=2.5, color=BLACK, zorder=6, label="Baseline marginal relevante")
        ax.scatter(x_v1, knn_ref, marker="^", s=55, color=GREEN, zorder=7, label="kNN similaridade (Tanimoto / 3-mer)")

        ax.set_xticks(x_v1)
        ax.set_xticklabels([lbl for _, _, lbl in v1_specs])
        ax.set_ylabel(ylabel)
        ax.set_title(f"Protocolo v1 (5 Seeds) — {title_suffix}")
        if metric_idx == 0:
            ax.set_ylim(0.42, 0.98)
        ax.legend(loc="best", fontsize=7.5)

    # Painel Inferior: Protocolo v2 (5-Fold GroupKFold no Davis Cold)
    x_v2 = np.arange(len(v2_specs))
    for metric_idx, (metric, ylabel, title_suffix) in enumerate([("ci", "Concordance Index (CI ↑)", "CI"), ("mse", "Erro Quadrático Médio (MSE ↓)", "MSE")]):
        ax = axes[1, metric_idx]
        gin_m, gin_s, cnn_m, cnn_s, marg_m, knn_m = [], [], [], [], [], []
        for ds, sp, _ in v2_specs:
            dg = v2_data[("graphdta_gin", ds, sp)]
            dc = v2_data[("graphdta_cnn3", ds, sp)]
            gin_m.append(dg["per_fold_mean_std"][metric]["mean"])
            gin_s.append(dg["per_fold_mean_std"][metric]["std"])
            cnn_m.append(dc["per_fold_mean_std"][metric]["mean"])
            cnn_s.append(dc["per_fold_mean_std"][metric]["std"])

            pfr = list(dg["per_fold_results"].values())
            rel_m = "target_mean_predictor" if sp == "cold_drug" else "drug_mean_predictor"
            rel_k = "knn_tanimoto_drug" if sp == "cold_drug" else "knn_kmer_target"
            marg_m.append(np.mean([f["marginal_baselines"][rel_m][metric] for f in pfr]))
            knn_m.append(np.mean([f["knn_baselines"][rel_k][metric] for f in pfr]))

        ax.bar(x_v2 - w / 2, gin_m, w, yerr=gin_s, capsize=3, color=BLUE, label="graphdta_gin (oficial)")
        ax.bar(x_v2 + w / 2, cnn_m, w, yerr=cnn_s, capsize=3, color=VERMILION, label="graphdta_cnn3 (variante)")
        ax.scatter(x_v2, marg_m, marker="_", s=550, linewidths=2.5, color=BLACK, zorder=5, label="Baseline marginal relevante (alvo/droga)")
        ax.scatter(x_v2, knn_m, marker="^", s=65, color=GREEN, zorder=6, label="kNN similaridade (Tanimoto / 3-mer)")

        ax.set_xticks(x_v2)
        ax.set_xticklabels([lbl for _, _, lbl in v2_specs])
        ax.set_ylabel(ylabel)
        ax.set_title(f"Protocolo v2 (5-Fold GroupKFold) — {title_suffix}")
        if metric_idx == 0:
            ax.set_ylim(0.50, 0.90)
        ax.legend(loc="best", fontsize=7.8)

    fig.suptitle("F2 — CI e MSE por Modelo, Dataset e Split (Protocolos v1 e v2 Estritamente Separados)", fontweight="bold", y=1.01)
    fig.tight_layout()
    return _save_fig(fig, figures_dir, "F2_ci_mse_by_model_split")


def make_f3(data_root: Path, figures_dir: Path) -> dict[str, Path]:
    repo_root = resolve_roots()[0]
    cpu_audits = load_json_strict(repo_root / "docs" / "paper" / "cpu_audits.json")
    v1_pg = cpu_audits["v1_marginals_knn_and_paired"]["paired_groups"]
    res_dir = data_root / "outputs" / "results"

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), gridspec_kw={"width_ratios": [1.55, 1.0]}, sharey=False)
    offsets = {"graphdta_gin": -0.17, "graphdta_cnn3": 0.17}
    colors = {"graphdta_gin": BLUE, "graphdta_cnn3": VERMILION}
    labels_nice = {"graphdta_gin": "graphdta_gin (oficial)", "graphdta_cnn3": "graphdta_cnn3 (variante)"}

    # Painel A: Protocolo v1_seeds por seed
    ax = axes[0]
    v1_specs = [
        ("davis", "cold_drug", "Davis\ncold_drug"),
        ("davis", "cold_target", "Davis\ncold_target"),
        ("kiba", "cold_drug", "KIBA\ncold_drug"),
        ("kiba", "cold_target", "KIBA\ncold_target"),
    ]
    x_v1 = np.arange(len(v1_specs))
    for mod in ("graphdta_gin", "graphdta_cnn3"):
        labeled = False
        for idx, (ds, sp, _) in enumerate(v1_specs):
            gk = f"{mod}:{ds}:{sp}"
            if gk not in v1_pg:
                continue
            pg = v1_pg[gk]
            deltas = [s["delta_mse"] for s in pg["per_seed"]]
            xpos = x_v1[idx] + (offsets[mod] if ds == "davis" else 0.0)
            jitter = np.linspace(-0.04, 0.04, len(deltas))
            ax.scatter(
                np.full(len(deltas), xpos) + jitter,
                deltas,
                color=colors[mod],
                alpha=0.72,
                s=40,
                zorder=4,
                label=labels_nice[mod] if not labeled else None,
            )
            labeled = True
            mean_v = pg["delta_mse_paired"]["mean"]
            std_v = pg["delta_mse_paired"]["std"]
            ax.errorbar(
                [xpos],
                [mean_v],
                yerr=[[std_v], [std_v]],
                fmt="D",
                color=BLACK,
                ecolor=colors[mod],
                elinewidth=2.0,
                capsize=4.5,
                markersize=6.0,
                zorder=6,
            )
    ax.axhline(0.0, color=BLACK, linestyle="--", linewidth=1.2)
    ax.set_xticks(x_v1)
    ax.set_xticklabels([s[2] for s in v1_specs])
    ax.set_ylabel("Δ MSE Pareado (Modelo − Marginal)")
    ax.set_title("Protocolo v1 (5 Seeds): Δ MSE por Seed (± DP)")
    ax.legend(loc="upper left", frameon=True, fontsize=8.0)

    # Painel B: Protocolo v2_kfold por dobra (IC 95% bootstrap exploratorio)
    ax = axes[1]
    v2_specs = [
        ("davis", "cold_drug", "target_mean_predictor", "vs_target_mean", "Davis cold_drug\n(vs. target_mean)"),
        ("davis", "cold_target", "drug_mean_predictor", "vs_drug_mean", "Davis cold_target\n(vs. drug_mean)"),
    ]
    x_v2 = np.arange(len(v2_specs))
    for mod in ("graphdta_gin", "graphdta_cnn3"):
        for idx, (ds, sp, marg_key, boot_key, _) in enumerate(v2_specs):
            ad = load_json_strict(res_dir / f"baseline_{mod}_{ds}_{sp}_kfold.json")
            pfr = [v for _, v in sorted(ad["per_fold_results"].items(), key=lambda x: int(x[0]))]
            deltas = [f["test_metrics_full"]["mse"] - f["marginal_baselines"][marg_key]["mse"] for f in pfr]
            bs = ad["bootstrap_analysis"][boot_key]["delta_mse"]
            xpos = x_v2[idx] + offsets[mod]
            jitter = np.linspace(-0.04, 0.04, len(deltas))
            ax.scatter(
                np.full(len(deltas), xpos) + jitter,
                deltas,
                color=colors[mod],
                alpha=0.72,
                s=40,
                zorder=4,
                label=labels_nice[mod] if idx == 0 else None,
            )
            mean_v = bs["mean"]
            err_low = mean_v - bs["ci_95"][0]
            err_high = bs["ci_95"][1] - mean_v
            ax.errorbar(
                [xpos],
                [mean_v],
                yerr=[[err_low], [err_high]],
                fmt="D",
                color=BLACK,
                ecolor=colors[mod],
                elinewidth=2.2,
                capsize=5,
                markersize=6.2,
                zorder=6,
            )
    ax.axhline(0.0, color=BLACK, linestyle="--", linewidth=1.2)
    ax.set_xticks(x_v2)
    ax.set_xticklabels([s[4] for s in v2_specs])
    ax.set_title("Protocolo v2 (5-Fold GroupKFold): Δ MSE (IC 95% Expl.)")
    ax.legend(loc="upper left", frameon=True, fontsize=8.0)

    fig.suptitle("F3 — Δ MSE Pareado vs. Baseline Marginal nos Regimes Cold (< 0 indica ganho do modelo)", fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save_fig(fig, figures_dir, "F3_paired_delta_mse_cold")


def make_f4(data_root: Path, figures_dir: Path) -> dict[str, Path]:
    runs_dir = data_root / "outputs" / "results" / "runs"
    gin = load_json_strict(runs_dir / "graphdta_gin_davis_cold_target_fold3.json")
    cnn = load_json_strict(runs_dir / "graphdta_cnn3_davis_cold_target_fold2.json")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), sharey=True)

    for ax, d, title, color in [
        (axes[0], gin, "graphdta_gin (Davis cold_target, fold 3)", BLUE),
        (axes[1], cnn, "graphdta_cnn3 (Davis cold_target, fold 2)", VERMILION),
    ]:
        h = pd.DataFrame(d["history"])
        best_ep = int(d["best_epoch"])
        ax.plot(h["epoch"], h["train_loss"], color=SKY if color == BLUE else ORANGE, linewidth=1.8, label="Treino (train_loss MSE)")
        ax.plot(h["epoch"], h["val_mse"], color=color, linewidth=2.0, label="Validação (val_mse)")
        ax.axvline(best_ep, color=BLACK, linestyle="--", linewidth=1.3, label=f"best_epoch = {best_ep}")
        ax.set_title(title)
        ax.set_xlabel("Época")
        ax.set_ylim(0.0, min(4.0, max(float(h["val_mse"].quantile(0.95)) * 1.3, 1.8)))
        ax.legend(frameon=True, loc="upper right")

    axes[0].set_ylabel("MSE (Treino / Validação)")
    fig.suptitle("F4 — Curvas de Treino e Validação em Davis cold_target (Protocolo v2 GroupKFold)", fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save_fig(fig, figures_dir, "F4_learning_curves_davis_cold_target")


def make_f5(data_root: Path, figures_dir: Path) -> dict[str, Path]:
    runs_dir = data_root / "outputs" / "results" / "runs"
    runs = [load_json_strict(p) for p in sorted(runs_dir.glob("*.json"))]

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), gridspec_kw={"width_ratios": [2.2, 1.0]})

    # Painel A: v1_seeds
    v1_groups = [
        ("davis", "random", "Davis\nrandom"),
        ("davis", "cold_drug", "Davis\ncold_drug"),
        ("davis", "cold_target", "Davis\ncold_target"),
        ("kiba", "random", "KIBA\nrandom*"),
        ("kiba", "cold_drug", "KIBA\ncold_drug"),
        ("kiba", "cold_target", "KIBA\ncold_target"),
    ]
    ax = axes[0]
    for g_idx, (ds, sp, _) in enumerate(v1_groups):
        for mod, off, col, mk in [("graphdta_gin", -0.15, BLUE, "o"), ("graphdta_cnn3", 0.15, VERMILION, "s")]:
            vals = [r["best_epoch"] for r in runs if r.get("fold") is None and r["dataset"] == ds and r["split"] == sp and r["model"] == mod]
            if not vals:
                continue
            jit = np.linspace(-0.05, 0.05, len(vals))
            ax.scatter(
                np.full(len(vals), g_idx + off) + jit,
                vals,
                color=col,
                marker=mk,
                s=45,
                alpha=0.85,
                label=mod if (g_idx == 0) else None,
            )
    ax.axhline(400, color=GRAY, linestyle="--", linewidth=1.1, label="0.8 × 500 = 400 (alerta)")
    ax.axhline(800, color=PURPLE, linestyle=":", linewidth=1.2, label="0.8 × 1000 = 800 (KIBA s1,3)")
    ax.set_xticks(range(len(v1_groups)))
    ax.set_xticklabels([g[2] for g in v1_groups])
    ax.set_ylabel("best_epoch")
    ax.set_title("Protocolo v1 (Seeds 0–4)")
    ax.legend(loc="upper left", fontsize=7.8)

    # Painel B: v2_kfold
    v2_groups = [("davis", "cold_drug", "Davis\ncold_drug"), ("davis", "cold_target", "Davis\ncold_target")]
    ax = axes[1]
    for g_idx, (ds, sp, _) in enumerate(v2_groups):
        for mod, off, col, mk in [("graphdta_gin", -0.15, BLUE, "o"), ("graphdta_cnn3", 0.15, VERMILION, "s")]:
            vals = [r["best_epoch"] for r in runs if r.get("fold") is not None and r["dataset"] == ds and r["split"] == sp and r["model"] == mod]
            jit = np.linspace(-0.05, 0.05, len(vals))
            ax.scatter(np.full(len(vals), g_idx + off) + jit, vals, color=col, marker=mk, s=45, alpha=0.85, label=mod if g_idx == 0 else None)
    ax.axhline(400, color=GRAY, linestyle="--", linewidth=1.1)
    ax.set_xticks(range(len(v2_groups)))
    ax.set_xticklabels([g[2] for g in v2_groups])
    ax.set_title("Protocolo v2 (5-Fold GroupKFold)")
    ax.set_ylim(-15, 430)

    fig.suptitle("F5 — Distribuição de best_epoch por Run e Modelo (Protocolos v1 e v2 Separados)", fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save_fig(fig, figures_dir, "F5_best_epoch_distribution")


def make_f6(data_root: Path, figures_dir: Path) -> dict[str, Path]:
    res_dir = data_root / "outputs" / "results"
    df_t6 = pd.read_csv(ensure_paper_dirs()["tables"] / "T6_seq_length.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))

    for ax, proto, title in [
        (axes[0], "v1_seeds", "Protocolo v1 (Seeds): MSE por Comprimento"),
        (axes[1], "v2_kfold", "Protocolo v2 (5-Fold GroupKFold): MSE por Comprimento"),
    ]:
        sub = df_t6[df_t6["protocolo"] == proto].copy()
        groups = []
        for _, row in sub[["modelo", "dataset", "split"]].drop_duplicates().iterrows():
            groups.append((row["modelo"], row["dataset"], row["split"]))
        x = np.arange(len(groups))
        w = 0.35
        le_m, le_s, gt_m, gt_s, labels = [], [], [], [], []
        for mod, ds, sp in groups:
            r_le = sub[(sub["modelo"] == mod) & (sub["dataset"] == ds) & (sub["split"] == sp) & (sub["faixa_comprimento"] == "<= 1000 aa")].iloc[0]
            r_gt = sub[(sub["modelo"] == mod) & (sub["dataset"] == ds) & (sub["split"] == sp) & (sub["faixa_comprimento"] == "> 1000 aa")].iloc[0]
            m1, s1 = [float(v) for v in r_le["mse"].split(" ± ")]
            m2, s2 = [float(v) for v in r_gt["mse"].split(" ± ")]
            le_m.append(m1); le_s.append(s1)
            gt_m.append(m2); gt_s.append(s2)
            mshort = "GIN" if "gin" in mod else "CNN3"
            labels.append(f"{ds.upper()}\n{sp}\n({mshort})")

        ax.bar(x - w / 2, le_m, w, yerr=le_s, capsize=2.5, color=BLUE, label="<= 1000 aa")
        ax.bar(x + w / 2, gt_m, w, yerr=gt_s, capsize=2.5, color=VERMILION, label="> 1000 aa (truncado)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylabel("MSE ↓")
        ax.set_title(title)
        ax.legend(frameon=True)

    fig.suptitle("F6 — Degradação do MSE em Proteínas Longas (> 1000 aa) vs. <= 1000 aa", fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save_fig(fig, figures_dir, "F6_seq_length_stratification")


def make_all_figures() -> dict[str, dict[str, Path]]:
    _setup_style()
    repo_root, data_root = resolve_roots()
    dirs = ensure_paper_dirs(repo_root)
    figures_dir = dirs["figures"]

    figs = {
        "F1": make_f1(data_root, figures_dir),
        "F2": make_f2(data_root, figures_dir),
        "F3": make_f3(data_root, figures_dir),
        "F4": make_f4(data_root, figures_dir),
        "F5": make_f5(data_root, figures_dir),
        "F6": make_f6(data_root, figures_dir),
    }

    captions = [
        "# Legendas das Figuras (`docs/paper/figures/`)",
        "",
        "- **F1 (`F1_affinity_distributions.{pdf,png}`):** Distribuição das afinidades experimentais nos conjuntos limpos de Davis ($pK_d$, $N = 30.056$) e KIBA ($KIBA\\ score$, $N = 118.254$). Destaca-se o pico de censura à esquerda em $pK_d = 5{,}0$ no Davis (`20.931` pares; `69,64%`). **Tabela de referência:** `T1_dataset_stats`. **Fontes:** `data/processed/davis_clean.csv`, `data/processed/kiba_clean.csv`, `data/processed/davis_ingestion_report.json`.",
        "- **F2 (`F2_ci_mse_by_model_split.{pdf,png}`):** Concordance Index (CI) e MSE (média $\\pm$ desvio amostral) de `graphdta_gin` e `graphdta_cnn3` nos protocolos **v1 (5 seeds)** (painel superior) e **v2 (5-Fold GroupKFold)** (painel inferior), mantidos estritamente separados. Linhas/marcadores indicam `mean_predictor` (v1), o baseline marginal relevante (`drug_mean` ou `target_mean`) e o `kNN` de similaridade relevante (`Tanimoto` / `3-mer`) para todos os grupos (via `cpu_audits.json` no v1 e JSONs no v2). `*` indica teto efetivo de `1000` épocas em `kiba_random` (seeds 0, 2, 4 pararam por *early stopping* antes de 500; seeds 1 e 3 estendidas até 1000). **Tabelas de referência:** `T3_main_results`, `T4_marginal_and_paired`. **Fontes:** `outputs/results/baseline_*.json`, `docs/paper/cpu_audits.json`.",
        "- **F3 (`F3_paired_delta_mse_cold.{pdf,png}`):** Diferença pareada $\\Delta\\text{MSE}$ (`modelo - baseline marginal relevante`) nos regimes cold em **dois painéis separados**: **Painel A (Protocolo v1, 5 seeds)** mostrando cada seed individual e média $\\pm$ desvio padrão (`Davis cold_drug`, `Davis cold_target`, `KIBA cold_drug`, `KIBA cold_target`); **Painel B (Protocolo v2, 5-Fold GroupKFold)** exibindo cada dobra, média e IC 95% via bootstrap agrupado no nível de cluster (**comparação exploratória**). Valores negativos (`< 0`) indicam superioridade do modelo sobre o baseline marginal. **Tabela de referência:** `T4_marginal_and_paired`. **Fontes:** `docs/paper/cpu_audits.json`, `outputs/results/baseline_*_kfold.json`.",
        "- **F4 (`F4_learning_curves_davis_cold_target.{pdf,png}`):** Curvas de perda de treino (`train_loss`) e validação (`val_mse`) ao longo das épocas para execuções representativas no Davis `cold_target` sob Protocolo v2 (`graphdta_gin` fold 3 com `best_epoch = 6` vs. `graphdta_cnn3` fold 2 com `best_epoch = 59`). Nota-se que os 43 runs originais de `v1_seeds` não gravaram `history` por época (apenas `v2_kfold` e a retomada das seeds 1 e 3 de `kiba_random` possuem `history`). **Tabela de referência:** `T7_training_diagnostics`. **Fontes:** `outputs/results/runs/graphdta_gin_davis_cold_target_fold3.json`, `outputs/results/runs/graphdta_cnn3_davis_cold_target_fold2.json`.",
        "- **F5 (`F5_best_epoch_distribution.{pdf,png}`):** Distribuição de `best_epoch` para todas as 65 execuções, separando Protocolo v1 (painel esquerdo) e Protocolo v2 (painel direito), com linhas de alerta em $0{,}8 \\times 500 = 400$ e $0{,}8 \\times 1000 = 800$. Evidencia o colapso precoce de `graphdta_gin` em Davis `cold_target` (`2..7` em v1; `3..13` em 4 de 5 dobras de v2, com `fold0=236`) e `kiba_random` com teto efetivo `1000` (seeds 0, 2, 4 em `286..343`; seed 1 em `774`; seed 3 em `509`). **Tabela de referência:** `T7_training_diagnostics`. **Fontes:** `outputs/results/runs/*.json`.",
        "- **F6 (`F6_seq_length_stratification.{pdf,png}`):** Comparação de MSE entre proteínas com comprimento $\\le 1000$ aa e $> 1000$ aa (truncadas no GraphDTA) nos protocolos v1 e v2 separados. **Tabela de referência:** `T6_seq_length`. **Fontes:** `outputs/results/baseline_*.json`.",
    ]
    cap_path = figures_dir / "CAPTIONS.md"
    cap_path.write_text("\n".join(captions) + "\n", encoding="utf-8")
    figs["CAPTIONS"] = {"md": cap_path}
    return figs


if __name__ == "__main__":
    out = make_all_figures()
    for k, paths in out.items():
        print(f"[OK] {k}: {[p.name for p in paths.values()]}")
