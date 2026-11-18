"""Passo 2: Gera figuras F1..F12 em PDF e PNG (paleta Okabe-Ito) + `CAPTIONS.md`."""

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


def _normalize_phase3_model(name: str) -> str:
    s = str(name)
    return s[len("physdual_") :] if s.startswith("physdual_") else s


def make_f7(repo_root: Path, figures_dir: Path) -> dict[str, Path]:
    """F7: Curva de generalização (Δ vs ridge_esm e melhor não-ESM) com IC bootstrap; v1 e v2 separados."""
    pa = load_json_strict(repo_root / "docs" / "paper" / "phase3_audits.json")
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.2))

    # Configuracao por coluna: Davis (esquerda) e KIBA (direita), mantendo v1_seeds e v2_kfold disjuntos
    davis_ticks = [
        ("v1_seeds", "davis/random", "random\n(v1, ±DP)"),
        ("v2_kfold", "davis/cold_target", "cold_target\n(v2)"),
        ("v2_kfold", "davis/cold_target_seqid40", "seqid40\n(v2)"),
        ("v2_kfold", "davis/cold_target_seqid30", "seqid30\n(v2)"),
        ("v2_kfold", "davis/cold_drug", "cold_drug\n(v2)"),
    ]
    kiba_ticks = [
        ("v1_seeds", "kiba/random", "random\n(v1, ±DP)"),
        ("v1_seeds", "kiba/cold_target", "cold_target\n(v1)"),
        ("v1_seeds", "kiba/cold_drug", "cold_drug\n(v1)"),
        ("v2_kfold", "kiba/cold_target_seqid40", "seqid40\n(v2)"),
        ("v2_kfold", "kiba/cold_target_seqid30", "seqid30\n(v2)"),
    ]

    for row_idx, (metric_key, boot_field, ylabel, better_note) in enumerate([
        ("delta_mse", "delta_mse", "Δ MSE (gin_esm_mlp − Baseline) ↓", "Δ MSE (< 0 favorece modelo)"),
        ("delta_ci", "delta_ci", "Δ CI (gin_esm_mlp − Baseline) ↑", "Δ CI (> 0 favorece modelo)"),
    ]):
        for col_idx, (ds_title, specs, grad_indices, v1_sep_x, extra_sep_x) in enumerate([
            ("Davis (v1 sanidade | v2 exploratório)", davis_ticks, [1, 2, 3], 0.5, 3.5),
            ("KIBA (v1 primário | v2 seqid secundário)", kiba_ticks, [3, 4], 2.5, None),
        ]):
            ax = axes[row_idx, col_idx]
            x_vals = np.arange(len(specs))

            for comp_kind, off, col, mk, ls, comp_label in [
                ("ridge", -0.11, BLUE, "o", "-", "vs. ridge_esm"),
                ("best_non", 0.11, VERMILION, "s", "--", "vs. melhor baseline não-ESM"),
            ]:
                pts_x, pts_y, err_lo, err_hi = [], [], [], []
                is_boot = []
                for xi, (proto, rkey, _lbl) in enumerate(specs):
                    reg = pa[proto]["regimes"][rkey]
                    if comp_kind == "ridge":
                        comp_name = "ridge_esm"
                    else:
                        if proto == "v1_seeds":
                            comp_name = reg.get("best_non_esm_mse")
                        else:
                            comp_name = reg.get("best_non_esm_oof_mse")
                        if not comp_name:
                            continue

                    paired = reg["paired_vs_baselines_seeds" if proto == "v1_seeds" else "paired_vs_baselines_folds"][comp_name]
                    boot_map = reg.get("bootstrap_vs_baselines_pooled_clusters" if proto == "v1_seeds" else "bootstrap_vs_baselines") or {}
                    bt = boot_map.get(comp_name)
                    if bt is not None:
                        val = bt[boot_field]["mean"]
                        ci95 = bt[boot_field]["ci_95"]
                        pts_x.append(xi + off)
                        pts_y.append(val)
                        err_lo.append(max(0.0, val - ci95[0]))
                        err_hi.append(max(0.0, ci95[1] - val))
                        is_boot.append(True)
                    else:
                        val = paired[f"{metric_key}_mean"]
                        sd = paired[f"{metric_key}_std"]
                        pts_x.append(xi + off)
                        pts_y.append(val)
                        err_lo.append(sd)
                        err_hi.append(sd)
                        is_boot.append(False)

                # Plotar pontos individuais com seus respectivos ICs/DPs
                for px, py, elo, ehi, bflag, idx_pt in zip(pts_x, pts_y, err_lo, err_hi, is_boot, range(len(pts_x))):
                    ax.errorbar(
                        [px], [py], yerr=[[elo], [ehi]],
                        fmt=mk, color=col, ecolor=col, elinewidth=1.8 if bflag else 1.2,
                        capsize=4.0, markersize=6.0, alpha=0.95 if bflag else 0.72,
                        label=comp_label if idx_pt == 0 else None, zorder=5,
                    )

                # Conectar exclusivamente a curva seqid contínua do protocolo v2_kfold (sem misturar v1 e v2)
                if comp_kind == "ridge":
                    grad_map = {int(round(px - off)): (px, py) for px, py in zip(pts_x, pts_y)}
                    gx = [grad_map[g][0] for g in grad_indices if g in grad_map]
                    gy = [grad_map[g][1] for g in grad_indices if g in grad_map]
                    if len(gx) >= 2:
                        ax.plot(gx, gy, color=col, linestyle="-", linewidth=1.7, alpha=0.85, zorder=4)
                else:
                    grad_map = {int(round(px - off)): (px, py) for px, py in zip(pts_x, pts_y)}
                    gx = [grad_map[g][0] for g in grad_indices if g in grad_map]
                    gy = [grad_map[g][1] for g in grad_indices if g in grad_map]
                    if len(gx) >= 2:
                        ax.plot(gx, gy, color=col, linestyle="--", linewidth=1.5, alpha=0.85, zorder=4)

            ax.axhline(0.0, color=BLACK, linestyle="--", linewidth=1.1, zorder=2)
            ax.axvline(v1_sep_x, color=GRAY, linestyle=":", linewidth=1.3, zorder=2)
            if extra_sep_x is not None:
                ax.axvline(extra_sep_x, color=GRAY, linestyle=":", linewidth=1.1, zorder=2)

            ax.set_xticks(x_vals)
            ax.set_xticklabels([s[2] for s in specs], fontsize=8.2)
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            ax.set_title(f"{ds_title} — {better_note}", fontsize=9.6)
            ax.legend(loc="best", fontsize=7.8, frameon=True)

    fig.suptitle(
        "F7 — Curva de Generalização (gin_esm_mlp vs. ridge_esm e Melhor Não-ESM; v1 e v2 Estritamente Separados)",
        fontweight="bold", y=0.98,
    )
    fig.subplots_adjust(hspace=0.36, wspace=0.22, top=0.90)
    return _save_fig(fig, figures_dir, "F7_generalization_curve")


def make_f8(repo_root: Path, figures_dir: Path) -> dict[str, Path]:
    """F8: Δ pareado por unidade (seed/dobra) contra ridge_esm e melhor baseline não-ESM em painéis v1 vs v2."""
    pa = load_json_strict(repo_root / "docs" / "paper" / "phase3_audits.json")
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.0), gridspec_kw={"width_ratios": [1.0, 2.35]})

    v1_specs = [("kiba/cold_drug", "KIBA\ncold_drug"), ("kiba/cold_target", "KIBA\ncold_target")]
    v2_specs = [
        ("kiba/cold_target_seqid40", "KIBA\nseqid40"),
        ("kiba/cold_target_seqid30", "KIBA\nseqid30"),
        ("davis/cold_target", "Davis\ncold_target"),
        ("davis/cold_target_seqid40", "Davis\nseqid40"),
        ("davis/cold_target_seqid30", "Davis\nseqid30"),
        ("davis/cold_drug", "Davis\ncold_drug"),
    ]

    for row_idx, (field, ylabel, subtitle) in enumerate([
        ("delta_mse", "Δ MSE Pareado (gin_esm_mlp − Baseline)", "Δ MSE (< 0 favorece modelo)"),
        ("delta_ci", "Δ CI Pareado (gin_esm_mlp − Baseline)", "Δ CI (> 0 favorece modelo)"),
    ]):
        for col_idx, (proto, specs, panel_title) in enumerate([
            ("v1_seeds", v1_specs, "v1_seeds (5 Seeds — KIBA Primário)"),
            ("v2_kfold", v2_specs, "v2_kfold (5-Fold GroupKFold — KIBA seqid & Davis Expl.)"),
        ]):
            ax = axes[row_idx, col_idx]
            x_vals = np.arange(len(specs))
            for comp_kind, off, col, mk, lbl in [
                ("ridge_esm", -0.17, BLUE, "o", "vs. ridge_esm"),
                ("best_non", 0.17, VERMILION, "s", "vs. melhor baseline não-ESM"),
            ]:
                for idx, (rkey, _) in enumerate(specs):
                    reg = pa[proto]["regimes"][rkey]
                    comp = "ridge_esm" if comp_kind == "ridge_esm" else (
                        reg["best_non_esm_mse"] if proto == "v1_seeds" else reg["best_non_esm_oof_mse"]
                    )
                    pm = reg["paired_vs_baselines_seeds" if proto == "v1_seeds" else "paired_vs_baselines_folds"][comp]
                    vals = pm[f"{field}_per_unit"]
                    jit = np.linspace(-0.045, 0.045, len(vals))
                    xpos = x_vals[idx] + off
                    ax.scatter(
                        np.full(len(vals), xpos) + jit, vals,
                        color=col, marker=mk, alpha=0.72, s=34, zorder=4,
                        label=lbl if idx == 0 else None,
                    )
                    m = pm[f"{field}_mean"]
                    s = pm[f"{field}_std"]
                    ax.errorbar(
                        [xpos], [m], yerr=[[s], [s]], fmt="D", color=BLACK,
                        ecolor=col, elinewidth=1.8, capsize=4.0, markersize=5.2, zorder=6,
                    )
            ax.axhline(0.0, color=BLACK, linestyle="--", linewidth=1.1, zorder=2)
            ax.set_xticks(x_vals)
            ax.set_xticklabels([lbl for _, lbl in specs], fontsize=8.2)
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            ax.set_title(f"{panel_title} | {subtitle}", fontsize=9.3)
            ax.legend(loc="best", fontsize=7.6, frameon=True)

    fig.suptitle(
        "F8 — Δ Pareado por Seed/Dobra vs. ridge_esm e Melhor Não-ESM (Pontos Individuais e Média ± DP)",
        fontweight="bold", y=0.98,
    )
    fig.subplots_adjust(hspace=0.36, wspace=0.22, top=0.90)
    return _save_fig(fig, figures_dir, "F8_paired_deltas_by_unit")


def make_f9(repo_root: Path, figures_dir: Path) -> dict[str, Path]:
    """F9: Controles diagnósticos no Davis cold_target (modelo real vs shuffle-esm vs ecfp_esm_mlp vs baselines)."""
    pa = load_json_strict(repo_root / "docs" / "paper" / "phase3_audits.json")
    reg = pa["v2_kfold"]["regimes"]["davis/cold_target"]
    pms = reg["per_model_summary"]
    pfm = reg["per_fold_by_model"]

    items = [
        ("gin_esm_mlp", "gin_esm_mlp (real)", BLUE),
        ("ecfp_esm_mlp", "ecfp_esm_mlp (ECFP4)", GREEN),
        ("ridge_esm", "ridge_esm (linear)", SKY),
        ("knn_kmer_target", "knn_kmer (não-ESM)", GRAY),
        ("drug_mean_predictor", "drug_mean (marginal)", BLACK),
        ("shuf0", "shuf0 (ESM perm.)", VERMILION),
        ("shuf1", "shuf1 (ESM perm.)", ORANGE),
        ("shuf2", "shuf2 (ESM perm.)", PURPLE),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.2))
    x = np.arange(len(items))

    for ax, metric_k, pf_k, ylabel, title in [
        (axes[0], "ci_per_fold", "ci", "Concordance Index (CI ↑)", "CI Médio (± DP) e Dobras em Davis cold_target"),
        (axes[1], "mse_per_fold", "mse", "MSE (↓)", "MSE Médio (± DP) e Dobras em Davis cold_target"),
    ]:
        means = [pms[m][metric_k]["mean"] for m, _, _ in items]
        stds = [pms[m][metric_k]["std"] for m, _, _ in items]
        cols = [c for _, _, c in items]

        ax.bar(x, means, yerr=stds, capsize=3.5, color=cols, alpha=0.78, edgecolor="white", zorder=3)
        dm_ref = pms["drug_mean_predictor"][metric_k]["mean"]
        ax.axhline(dm_ref, color=BLACK, linestyle="--", linewidth=1.3, label=f"drug_mean ({dm_ref:.4f})", zorder=4)

        for xi, (m_key, _, _col) in enumerate(items):
            vals = [float(e[pf_k]) for e in pfm[m_key]]
            jit = np.linspace(-0.14, 0.14, len(vals))
            ax.scatter(np.full(len(vals), xi) + jit, vals, color=BLACK, s=18, alpha=0.75, zorder=5)

        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl, _ in items], fontsize=8.0, rotation=28, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8.0, frameon=True)

    fig.suptitle(
        "F9 — Controles Diagnósticos em Davis cold_target: gin_esm_mlp vs. --shuffle-esm vs. ecfp_esm_mlp vs. drug_mean (Exploratório)",
        fontweight="bold", y=0.98,
    )
    fig.subplots_adjust(wspace=0.20, bottom=0.25, top=0.85)
    return _save_fig(fig, figures_dir, "F9_diagnostic_controls_cold_target")


def make_f10(repo_root: Path, figures_dir: Path) -> dict[str, Path]:
    """F10: Desempenho por dobra (0..4) em Davis cold_drug para todos os modelos com a dobra 2 destacada."""
    pa = load_json_strict(repo_root / "docs" / "paper" / "phase3_audits.json")
    reg = pa["v2_kfold"]["regimes"]["davis/cold_drug"]
    pfm = reg["per_fold_by_model"]
    f2_tsoff = pa["v2_kfold"]["davis_cold_drug_fold2_tsoff1000"]["test_metrics_full"]

    series = [
        ("gin_esm_mlp", "gin_esm_mlp (5 folds)", BLUE, "o", "-", 2.3),
        ("ecfp_esm_mlp", "ecfp_esm_mlp (ECFP4)", GREEN, "s", "--", 1.8),
        ("graphdta_cnn3", "graphdta_cnn3", ORANGE, "^", "-.", 1.5),
        ("graphdta_gin", "graphdta_gin", PURPLE, "v", ":", 1.5),
        ("ridge_esm", "ridge_esm", SKY, "D", "-", 1.6),
        ("knn_kmer_target", "knn_kmer_target", VERMILION, "P", "--", 1.6),
        ("knn_tanimoto_drug", "knn_tanimoto_drug", GRAY, "X", "-.", 1.6),
        ("target_mean_predictor", "target_mean_predictor", BLACK, "_", ":", 1.5),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.6))
    folds = np.arange(5)

    for ax_idx, (ax, field, ylabel, title) in enumerate([
        (axes[0], "ci", "Concordance Index (CI ↑)", "CI por Dobra em Davis cold_drug"),
        (axes[1], "mse", "MSE (↓)", "MSE por Dobra em Davis cold_drug"),
    ]):
        ax.axvspan(1.65, 2.35, color=VERMILION, alpha=0.12, zorder=1, label="Dobra 2 (extrema)" if ax_idx == 0 else None)
        for mod, lbl, col, mk, ls, lw in series:
            entries = sorted(pfm[mod], key=lambda e: int(e["fold"]))
            ys = [float(e[field]) for e in entries]
            ax.plot(folds, ys, color=col, marker=mk, linestyle=ls, linewidth=lw, markersize=6.0, label=lbl if ax_idx == 0 else None, zorder=4)

        ax.scatter(
            [2.0], [float(f2_tsoff[field])],
            marker="*", s=175, color=VERMILION, edgecolor=BLACK, linewidths=0.8,
            label="tsoff1000 fold 2 (CI=0.7090; MSE=1.0765)" if ax_idx == 0 else None, zorder=7,
        )
        ax.set_xticks(folds)
        ax.set_xticklabels([f"Fold {f}" for f in folds])
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8.0, frameon=True, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(
        "F10 — Desempenho Lado a Lado por Dobra em Davis cold_drug (Dobra 2 Destacada + Controle tsoff1000)",
        fontweight="bold", y=0.98,
    )
    fig.subplots_adjust(wspace=0.18, bottom=0.20, top=0.87)
    return _save_fig(fig, figures_dir, "F10_cold_drug_per_fold")


def make_f11(data_root: Path, figures_dir: Path) -> dict[str, Path]:
    """F11: Gap treino-validação no melhor checkpoint e best_epoch para todas as 81 execuções da Fase 3."""
    runs_dir = data_root / "outputs" / "results" / "runs"
    p3_runs = []
    for p in sorted(runs_dir.glob("*.json")):
        d = load_json_strict(p)
        if str(d.get("model", "")).startswith("graphdta_"):
            continue
        d["_proto"] = "v2_kfold" if d.get("fold") is not None else "v1_seeds"
        d["_mod_norm"] = _normalize_phase3_model(d["model"])
        p3_runs.append(d)

    v1_groups = [
        ("davis", "random", "gin_esm_mlp", "Davis\nrandom"),
        ("kiba", "random", "gin_esm_mlp", "KIBA\nrandom"),
        ("kiba", "cold_drug", "gin_esm_mlp", "KIBA\ncold_drug"),
        ("kiba", "cold_target", "gin_esm_mlp", "KIBA\ncold_target"),
    ]
    v2_groups = [
        ("davis", "cold_target", "gin_esm_mlp", "Davis\ncold_target"),
        ("davis", "cold_drug", "gin_esm_mlp", "Davis\ncold_drug"),
        ("davis", "cold_target_seqid40", "gin_esm_mlp", "Davis\nseqid40"),
        ("davis", "cold_target_seqid30", "gin_esm_mlp", "Davis\nseqid30"),
        ("davis", "cold_target", "ecfp_esm_mlp", "Davis CT\nECFP4"),
        ("davis", "cold_drug", "ecfp_esm_mlp", "Davis CD\nECFP4"),
        ("davis", "cold_target", "gin_esm_mlp_shuf0", "Davis CT\nshuf0"),
        ("davis", "cold_target", "gin_esm_mlp_shuf1", "Davis CT\nshuf1"),
        ("davis", "cold_target", "gin_esm_mlp_shuf2", "Davis CT\nshuf2"),
        ("davis", "cold_target_seqid30", "gin_esm_mlp_tsoff1000", "Davis s30\ntsoff1k"),
        ("kiba", "cold_target_seqid40", "gin_esm_mlp", "KIBA\nseqid40"),
        ("kiba", "cold_target_seqid30", "gin_esm_mlp", "KIBA\nseqid30"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14.6, 8.2), gridspec_kw={"width_ratios": [1.0, 2.7]})

    for row_idx, (field, ylabel, ref_lines, _kind) in enumerate([
        ("best_epoch", "best_epoch", [(15, VERMILION, "--", "best_epoch = 15 (alerta cold)"), (800, GRAY, ":", "0.8 × 1000 = 800")], "epoch"),
        ("train_val_mse_gap_at_best", "Gap (val_mse − train_loss) no best_epoch", [(0.0, BLACK, "--", "gap = 0 (val = treino)")], "gap"),
    ]):
        for col_idx, (proto, groups, col_title) in enumerate([
            ("v1_seeds", v1_groups, "v1_seeds (5 Seeds — Fase 3)"),
            ("v2_kfold", v2_groups, "v2_kfold (5-Fold GroupKFold — Principais e Controles)"),
        ]):
            ax = axes[row_idx, col_idx]
            x = np.arange(len(groups))
            for idx, (ds, sp, mod_norm, _) in enumerate(groups):
                vals = [
                    float(r[field])
                    for r in p3_runs
                    if r["_proto"] == proto and r["dataset"] == ds and r["split"] == sp and r["_mod_norm"] == mod_norm
                ]
                if not vals:
                    continue
                jit = np.linspace(-0.14, 0.14, len(vals))
                if "shuf" in mod_norm:
                    col, mk = VERMILION, "x"
                elif "ecfp" in mod_norm:
                    col, mk = GREEN, "s"
                elif "tsoff" in mod_norm:
                    col, mk = PURPLE, "^"
                else:
                    col, mk = BLUE, "o"
                ax.scatter(np.full(len(vals), idx) + jit, vals, color=col, marker=mk, s=40, alpha=0.88, zorder=4)
                ax.scatter([idx], [float(np.mean(vals))], color=BLACK, marker="_", s=180, linewidths=2.0, zorder=5)

            for ry, rcol, rls, rlbl in ref_lines:
                ax.axhline(ry, color=rcol, linestyle=rls, linewidth=1.2, label=rlbl, zorder=2)

            ax.set_xticks(x)
            ax.set_xticklabels([g[3] for g in groups], fontsize=7.6)
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            if row_idx == 0:
                ax.set_title(col_title, fontsize=9.8)
            ax.legend(loc="best", fontsize=7.5, frameon=True)

    fig.suptitle(
        "F11 — Diagnóstico de Treinamento da Fase 3: best_epoch e Gap val_mse − train_loss (v1 e v2 Separados)",
        fontweight="bold", y=0.98,
    )
    fig.subplots_adjust(hspace=0.35, wspace=0.18, top=0.90)
    return _save_fig(fig, figures_dir, "F11_train_val_gap_and_best_epoch")


def make_f12(repo_root: Path, figures_dir: Path) -> dict[str, Path]:
    """F12: Curva contínua identidade-margem estilo MIRAGE em painel 2x2 (Davis v2_kfold vs KIBA v1_seeds)."""
    cim_path = ensure_paper_dirs(repo_root)["paper"] / "continuous_identity_margin.json"
    if not cim_path.exists():
        cim_path = repo_root / "docs" / "paper" / "continuous_identity_margin.json"
    cim = load_json_strict(cim_path)

    col_specs = [
        ("davis/cold_target", "Davis cold_target (v2_kfold, 5 dobras)"),
        ("kiba/cold_target", "KIBA cold_target (v1_seeds, 5 seeds)"),
    ]
    row_specs = [
        ("global_id_max", "Identidade Global BLOSUM62 (max vs. treino)"),
        ("sw_id_aln_len_ge150aa", "Identidade Local Smith-Waterman ≥ 150 aa (max vs. treino)"),
    ]
    model_styles = [
        ("gin_esm_mlp", "gin_esm_mlp (Fase 3)", BLUE, "o", "-", -0.004),
        ("graphdta_gin", "graphdta_gin (Fase 2)", PURPLE, "^", "--", 0.0),
        ("ridge_esm", "ridge_esm (linear ESM+ECFP4)", ORANGE, "s", "-.", 0.004),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.8))

    for row_idx, (axis_key, x_label) in enumerate(row_specs):
        for col_idx, (rkey, col_title) in enumerate(col_specs):
            ax = axes[row_idx, col_idx]
            reg = cim["regimes"][rkey]
            bins = reg["quantile_curves"][axis_key]["bins"]

            ax.axhline(
                0.0,
                color=BLACK,
                linestyle="--",
                linewidth=1.2,
                label="drug_mean_predictor (margem = 0)" if (row_idx == 0 and col_idx == 0) else None,
                zorder=2,
            )

            for mod, lbl, col, mk, ls, x_jit in model_styles:
                if not bins or mod not in bins[0]["models"]:
                    continue
                xs = np.asarray([float(b["id_median"]) + x_jit for b in bins], dtype=np.float64)
                ys = np.asarray(
                    [float(b["models"][mod]["margin_vs_drug_mean_mean"]) for b in bins],
                    dtype=np.float64,
                )
                lo = np.asarray(
                    [float(b["models"][mod]["margin_vs_drug_mean_ci95"][0]) for b in bins],
                    dtype=np.float64,
                )
                hi = np.asarray(
                    [float(b["models"][mod]["margin_vs_drug_mean_ci95"][1]) for b in bins],
                    dtype=np.float64,
                )
                yerr_lo = np.maximum(0.0, ys - lo)
                yerr_hi = np.maximum(0.0, hi - ys)
                ax.errorbar(
                    xs,
                    ys,
                    yerr=[yerr_lo, yerr_hi],
                    color=col,
                    marker=mk,
                    linestyle=ls,
                    linewidth=2.0 if mod == "gin_esm_mlp" else 1.6,
                    markersize=6.0,
                    capsize=3.5,
                    alpha=0.92,
                    label=lbl if (row_idx == 0 and col_idx == 0) else None,
                    zorder=5 if mod == "gin_esm_mlp" else 4,
                )

            tr_gin = reg["continuous_trends"][axis_key]["margin_vs_drug_mean"].get("gin_esm_mlp")
            if tr_gin is not None:
                rho = float(tr_gin["spearman_rho"])
                ci_lo, ci_hi = tr_gin["spearman_ci95"]
                ax.text(
                    0.03,
                    0.95,
                    f"gin_esm_mlp Spearman ρ = {rho:+.3f}\nIC 95% [{ci_lo:+.3f}, {ci_hi:+.3f}]",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=8.0,
                    bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": BLUE, "alpha": 0.90},
                    zorder=6,
                )

            ax.set_xlabel(x_label)
            if col_idx == 0:
                ax.set_ylabel("Margem MSE(drug_mean) − MSE(modelo) (↑ melhor)")
            if row_idx == 0:
                ax.set_title(col_title, fontsize=10.2)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        fontsize=8.5,
        frameon=True,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(
        "F12 — Curva Contínua Identidade-Margem Estilo MIRAGE (5 Quantis e IC 95% via Cluster Bootstrap por Proteína)",
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(hspace=0.30, wspace=0.18, bottom=0.12, top=0.90)
    return _save_fig(fig, figures_dir, "F12_continuous_identity_margin")


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
        "F7": make_f7(repo_root, figures_dir),
        "F8": make_f8(repo_root, figures_dir),
        "F9": make_f9(repo_root, figures_dir),
        "F10": make_f10(repo_root, figures_dir),
        "F11": make_f11(data_root, figures_dir),
        "F12": make_f12(repo_root, figures_dir),
    }

    captions = [
        "# Legendas das Figuras (`docs/paper/figures/`)",
        "",
        "- **F1 (`F1_affinity_distributions.{pdf,png}`):** Distribuição das afinidades experimentais nos conjuntos limpos de Davis ($pK_d$, $N = 30.056$) e KIBA ($KIBA\\ score$, $N = 118.254$). Destaca-se o pico de censura à esquerda em $pK_d = 5{,}0$ no Davis (`20.931` pares; `69,64%`). **Tabela de referência:** `T1_dataset_stats`. **Fontes:** `data/processed/davis_clean.csv`, `data/processed/kiba_clean.csv`, `data/processed/davis_ingestion_report.json`.",
        "- **F2 (`F2_ci_mse_by_model_split.{pdf,png}`):** Concordance Index (CI) e MSE (média $\\pm$ desvio amostral) de `graphdta_gin` e `graphdta_cnn3` nos protocolos **v1 (5 seeds)** (painel superior) e **v2 (5-Fold GroupKFold)** (painel inferior), mantidos estritamente separados. Linhas/marcadores indicam `mean_predictor` (v1), o baseline marginal relevante (`drug_mean` ou `target_mean`) e o `kNN` de similaridade relevante (`Tanimoto` / `3-mer`) para todos os grupos (via `cpu_audits.json` no v1 e JSONs no v2). `*` indica teto efetivo de `1000` épocas em `kiba_random` (seeds 0, 2, 4 pararam por *early stopping* antes de 500; seeds 1 e 3 estendidas até 1000). **Tabelas de referência:** `T3_main_results`, `T4_marginal_and_paired`. **Fontes:** `outputs/results/baseline_*.json`, `docs/paper/cpu_audits.json`.",
        "- **F3 (`F3_paired_delta_mse_cold.{pdf,png}`):** Diferença pareada $\\Delta\\text{MSE}$ (`modelo - baseline marginal relevante`) nos regimes cold em **dois painéis separados**: **Painel A (Protocolo v1, 5 seeds)** mostrando cada seed individual e média $\\pm$ desvio padrão (`Davis cold_drug`, `Davis cold_target`, `KIBA cold_drug`, `KIBA cold_target`); **Painel B (Protocolo v2, 5-Fold GroupKFold)** exibindo cada dobra, média e IC 95% via bootstrap agrupado no nível de cluster (**comparação exploratória**). Valores negativos (`< 0`) indicam superioridade do modelo sobre o baseline marginal. **Tabela de referência:** `T4_marginal_and_paired`. **Fontes:** `docs/paper/cpu_audits.json`, `outputs/results/baseline_*_kfold.json`.",
        "- **F4 (`F4_learning_curves_davis_cold_target.{pdf,png}`):** Curvas de perda de treino (`train_loss`) e validação (`val_mse`) ao longo das épocas para execuções representativas no Davis `cold_target` sob Protocolo v2 (`graphdta_gin` fold 3 com `best_epoch = 6` vs. `graphdta_cnn3` fold 2 com `best_epoch = 59`). Nota-se que os 43 runs originais de `v1_seeds` não gravaram `history` por época (apenas `v2_kfold` e a retomada das seeds 1 e 3 de `kiba_random` possuem `history`). **Tabela de referência:** `T7_training_diagnostics`. **Fontes:** `outputs/results/runs/graphdta_gin_davis_cold_target_fold3.json`, `outputs/results/runs/graphdta_cnn3_davis_cold_target_fold2.json`.",
        "- **F5 (`F5_best_epoch_distribution.{pdf,png}`):** Distribuição de `best_epoch` para as 65 execuções GraphDTA das Fases 1–2, separando Protocolo v1 (painel esquerdo) e Protocolo v2 (painel direito), com linhas de alerta em $0{,}8 \\times 500 = 400$ e $0{,}8 \\times 1000 = 800$. Evidencia o colapso precoce de `graphdta_gin` em Davis `cold_target` (`2..7` em v1; `3..13` em 4 de 5 dobras de v2, com `fold0=236`) e `kiba_random` com teto efetivo `1000` (seeds 0, 2, 4 em `286..343`; seed 1 em `774`; seed 3 em `509`). **Tabela de referência:** `T7_training_diagnostics`. **Fontes:** `outputs/results/runs/graphdta_*.json`.",
        "- **F6 (`F6_seq_length_stratification.{pdf,png}`):** Comparação de MSE entre proteínas com comprimento $\\le 1000$ aa e $> 1000$ aa (truncadas no GraphDTA) nos protocolos v1 e v2 separados. **Tabela de referência:** `T6_seq_length`. **Fontes:** `outputs/results/baseline_*.json`.",
        "- **F7 (`F7_generalization_curve.{pdf,png}`):** Curva de generalização da Fase 3 (`gin_esm_mlp` menos `ridge_esm` e menos o melhor baseline não-ESM de cada regime) em Davis (esquerda) e KIBA (direita), mantendo `v1_seeds` (com DP em `random` e IC 95% de bootstrap em clusters empilhados nos regimes frios do KIBA) estritamente separado da sequência contínua `v2_kfold` (`cold_target` → `seqid40` → `seqid30`; `cold_drug` separado). Evidencia a erosão da margem em função da dissimilaridade proteica e a ausência de superação consistente do melhor baseline não-ESM. **Tabelas de referência:** `T9_phase3_decision_adr15`, `T13_phase3_paired_and_generalization`. **Fontes:** `docs/paper/phase3_audits.json`.",
        "- **F8 (`F8_paired_deltas_by_unit.{pdf,png}`):** Diferenças pareadas $\\Delta\\text{MSE}$ e $\\Delta\\text{CI}$ por seed/dobra (`gin_esm_mlp` − `ridge_esm` e − melhor baseline não-ESM) nos painéis separados `v1_seeds` (esquerda: KIBA `cold_drug` e `cold_target`) e `v2_kfold` (direita: KIBA `seqid40/30` e Davis exploratório), exibindo pontos individuais das 5 unidades e média $\\pm$ desvio-padrão amostral. **Tabelas de referência:** `T9_phase3_decision_adr15`, `T13_phase3_paired_and_generalization`. **Fontes:** `docs/paper/phase3_audits.json`.",
        "- **F9 (`F9_diagnostic_controls_cold_target.{pdf,png}`):** Controles diagnósticos exploratórios em Davis `cold_target` (`v2_kfold`, 5 dobras), comparando `gin_esm_mlp` (modelo real) contra `ecfp_esm_mlp` (ECFP4 + ESM), `ridge_esm`, `knn_kmer_target`, `drug_mean_predictor` e as três permutações de ESM (`--shuffle-esm` seeds `0, 1, 2`). Todos os controles embaralhados caem abaixo de `drug_mean_predictor` (ICs 95% de bootstrap excluem zero), enquanto `ecfp_esm_mlp` empata estatisticamente com `gin_esm_mlp`. **Tabelas de referência:** `T10_phase3_results_summary`, `T14_phase3_diagnostic_controls`. **Fontes:** `docs/paper/phase3_audits.json`.",
        "- **F10 (`F10_cold_drug_per_fold.{pdf,png}`):** Desempenho por dobra (`0..4`) em Davis `cold_drug` (`v2_kfold`) para `gin_esm_mlp`, controles e baselines, com destaque sombreado na **dobra 2** (ponto `gin_esm_mlp` extremo `CI=0.5720, MSE=1.3670`, acompanhado de controle `--train-seed-offset 1000` em estrela vermelha com `CI=0.7090, MSE=1.0765`). Evidencia se a dificuldade da dobra é generalizada ou amplificada por inicialização em um subconjunto pequeno (~14 fármacos). **Tabelas de referência:** `T12_phase3_per_fold_or_seed`, `T14_phase3_diagnostic_controls`. **Fontes:** `docs/paper/phase3_audits.json`.",
        "- **F11 (`F11_train_val_gap_and_best_epoch.{pdf,png}`):** Distribuição de `best_epoch` (linha superior, com alerta em `best_epoch = 15` para memorização precoce em regimes frios) e gap `val_mse − train_loss` no `best_epoch` (linha inferior) para as 81 execuções da Fase 3, mantendo `v1_seeds` (esquerda) e `v2_kfold` (direita) separados. **Tabelas de referência:** `T7_training_diagnostics`, `T15_phase3_configuration`. **Fontes:** `outputs/results/runs/physdual_*.json`.",
        "- **F12 (`F12_continuous_identity_margin.{pdf,png}`):** Curva contínua identidade-margem estilo MIRAGE em painel 2×2 separando estritamente `Davis cold_target` (`v2_kfold`, 5 dobras, coluna esquerda) e `KIBA cold_target` (`v1_seeds`, 5 seeds, coluna direita), nos eixos `global_id_max` (linha superior) e `sw_id_aln_len_ge150aa` Smith-Waterman $\\ge 150\\text{ aa}$ (linha inferior). Cada ponto representa um dos 5 quantis de igual contagem de observações `(u, p)` com a margem média $\\text{MSE}(\\text{drug\\_mean}) - \\text{MSE}(\\text{modelo})$ (valores positivos indicam superação do preditor médio por fármaco) e IC 95% via cluster bootstrap agrupado por sequência proteica única ($B = 2000$, `seed = 42`), acompanhado do coeficiente de Spearman $\\rho$ [IC 95%] de `gin_esm_mlp`. **Tabela de referência:** `T17_continuous_identity_margin`. **Fontes:** `docs/paper/continuous_identity_margin.json`.",
    ]
    cap_path = figures_dir / "CAPTIONS.md"
    cap_path.write_text("\n".join(captions) + "\n", encoding="utf-8")
    figs["CAPTIONS"] = {"md": cap_path}
    return figs


if __name__ == "__main__":
    out = make_all_figures()
    for k, paths in out.items():
        print(f"[OK] {k}: {[p.name for p in paths.values()]}")
