"""Testes de esquema, integridade e ausência de NaNs do pacote científico (Fases 1, 2 e 3)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = REPO_ROOT / "docs" / "paper"
TABLES_DIR = PAPER_DIR / "tables"
FIGURES_DIR = PAPER_DIR / "figures"

EXPECTED_DOCS = [
    PAPER_DIR / "00_INVENTARIO.md",
    PAPER_DIR / "01_METODOS_DADOS_E_BASELINES.md",
    PAPER_DIR / "02_RESULTADOS_PRELIMINARES.md",
    PAPER_DIR / "03_LIMITACOES_E_AMEACAS_A_VALIDADE.md",
    PAPER_DIR / "04_REGISTRO_DE_AFIRMACOES.md",
    PAPER_DIR / "05_LACUNAS.md",
    PAPER_DIR / "06_AUDITORIA_FASE3.md",
    PAPER_DIR / "REPRODUCAO.md",
    FIGURES_DIR / "CAPTIONS.md",
]

EXPECTED_TABLES = {
    "T1_dataset_stats": (2, ["dataset", "pares_brutos", "pares_limpos", "pares_censurados_pkd_5_0"]),
    "T2_splits": (40, ["dataset", "protocolo", "split", "unidade", "treino_pares", "teste_pares"]),
    "T3_main_results": (13, ["protocolo", "modelo", "dataset", "split", "n_unidades", "ci", "mse"]),
    "T4_marginal_and_paired": (
        13,
        ["protocolo", "modelo", "dataset", "split", "baseline_relevante", "delta_mse_pareado", "delta_ci_pareado"],
    ),
    "T5_uncensored_davis": (10, ["protocolo", "modelo", "split", "ci_pkd_gt_5", "mse_pkd_gt_5"]),
    "T6_seq_length": (26, ["protocolo", "modelo", "dataset", "split", "faixa_comprimento", "ci", "mse"]),
    "T7_training_diagnostics": (
        166,
        [
            "protocolo",
            "modelo",
            "dataset",
            "split",
            "unidade",
            "best_epoch",
            "train_loss_at_best",
            "train_val_mse_gap_at_best",
            "flag_cold_best_ep_lt_15",
            "is_diagnostic_control",
        ],
    ),
    "T8_configuration": (13, ["protocolo", "modelo", "dataset", "split", "n_runs", "featurizer"]),
    "T9_phase3_decision_adr15": (
        10,
        ["protocolo", "papel_no_estudo", "dataset", "split", "veredito_combinado_adr15", "fonte"],
    ),
    "T10_phase3_results_summary": (
        103,
        ["protocolo", "papel_no_estudo", "dataset", "split", "modelo", "n_unidades", "ci_media_dp", "mse_media_dp", "rm2_media_dp"],
    ),
    "T11_phase3_global_oof_and_binary": (
        63,
        ["protocolo", "dataset", "split", "modelo", "n_pares_oof", "oof_ci", "oof_mse", "oof_rm2"],
    ),
    "T12_phase3_per_fold_or_seed": (
        109,
        ["protocolo", "dataset", "split", "modelo", "u0_ci", "u0_mse", "media_ci", "media_mse"],
    ),
    "T13_phase3_paired_and_generalization": (
        83,
        [
            "protocolo",
            "papel_no_estudo",
            "dataset",
            "split",
            "baseline_comparado",
            "n_clusters_bootstrap",
            "boot_delta_mse_ic95",
            "boot_frac_favoravel_mse",
        ],
    ),
    "T14_phase3_diagnostic_controls": (
        22,
        [
            "dataset",
            "split",
            "controle_ou_modelo",
            "tipo_analise",
            "n_unidades",
            "ci_media_dp",
            "mse_media_dp",
            "boot_dMSE_vs_gin_ic95",
        ],
    ),
    "T15_phase3_configuration": (
        17,
        [
            "protocolo",
            "modelo",
            "dataset",
            "split",
            "n_runs",
            "drug_esm_featurizer",
            "epochs_cfg",
            "patience_efetiva",
            "sec_per_epoch_medio",
        ],
    ),
}

EXPECTED_FIGURES = [
    "F1_affinity_distributions",
    "F2_ci_mse_by_model_split",
    "F3_paired_delta_mse_cold",
    "F4_learning_curves_davis_cold_target",
    "F5_best_epoch_distribution",
    "F6_seq_length_stratification",
    "F7_generalization_curve",
    "F8_paired_deltas_by_unit",
    "F9_diagnostic_controls_cold_target",
    "F10_cold_drug_per_fold",
    "F11_train_val_gap_and_best_epoch",
]


def test_build_all_orchestrator_runs_cleanly(tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "paper" / "build_all.py"),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"build_all.py falhou ({proc.returncode}):\n{proc.stderr}\n{proc.stdout}"
    assert (tmp_path / "00_INVENTARIO.md").exists() and (tmp_path / "REPRODUCAO.md").exists()
    assert (tmp_path / "tables" / "T1_dataset_stats.csv").exists()
    assert (tmp_path / "tables" / "T15_phase3_configuration.csv").exists()
    assert (tmp_path / "figures" / "F11_train_val_gap_and_best_epoch.png").exists()


def test_all_expected_paper_markdown_docs_exist_and_nonempty() -> None:
    for doc_path in EXPECTED_DOCS:
        assert doc_path.exists(), f"Documento ausente: {doc_path}"
        text = doc_path.read_text(encoding="utf-8")
        assert len(text.strip()) > 100, f"Documento vazio ou curto demais: {doc_path}"


@pytest.mark.parametrize("stem,spec", list(EXPECTED_TABLES.items()))
def test_table_formats_schema_and_zero_nans(stem: str, spec: tuple[int, list[str]]) -> None:
    expected_rows, required_cols = spec
    for ext in ("csv", "md", "tex"):
        path = TABLES_DIR / f"{stem}.{ext}"
        assert path.exists() and path.stat().st_size > 0, f"Tabela ausente ou vazia: {path}"

    df = pd.read_csv(TABLES_DIR / f"{stem}.csv", dtype=str, keep_default_na=False)
    assert len(df) == expected_rows, f"{stem}.csv tem {len(df)} linhas, esperado {expected_rows}"
    for col in required_cols:
        assert col in df.columns, f"Coluna obrigatória '{col}' ausente em {stem}.csv"

    bad_tokens = {"", "nan", "NaN", "None", "null", "NULL", "N/A_NAN"}
    for col in df.columns:
        for idx, val in enumerate(df[col]):
            stripped = str(val).strip()
            assert stripped not in bad_tokens, f"Célula vazia ou NaN em {stem}.csv linha={idx} col={col}: {val!r}"
            assert "nan" not in stripped.lower().split(), f"Token NaN detectado em {stem}.csv: {val!r}"


def test_protocols_v1_and_v2_are_strictly_separated() -> None:
    for stem in (
        "T2_splits",
        "T3_main_results",
        "T4_marginal_and_paired",
        "T5_uncensored_davis",
        "T6_seq_length",
        "T7_training_diagnostics",
        "T8_configuration",
        "T9_phase3_decision_adr15",
        "T10_phase3_results_summary",
        "T12_phase3_per_fold_or_seed",
        "T13_phase3_paired_and_generalization",
        "T15_phase3_configuration",
    ):
        df = pd.read_csv(TABLES_DIR / f"{stem}.csv", dtype=str, keep_default_na=False)
        protocols = set(df["protocolo"].unique())
        assert protocols == {"v1_seeds", "v2_kfold"}, f"Protocolos inesperados em {stem}: {protocols}"

    df_t11 = pd.read_csv(TABLES_DIR / "T11_phase3_global_oof_and_binary.csv", dtype=str, keep_default_na=False)
    protocols_t11 = set(df_t11["protocolo"].unique())
    assert protocols_t11 == {"v1_seeds (média 5 seeds)", "v2_kfold"}, f"Protocolos inesperados em T11: {protocols_t11}"


@pytest.mark.parametrize("stem", EXPECTED_FIGURES)
def test_figures_pdf_and_png_exist_and_nonempty(stem: str) -> None:
    for ext in ("pdf", "png"):
        path = FIGURES_DIR / f"{stem}.{ext}"
        assert path.exists(), f"Figura ausente: {path}"
        assert path.stat().st_size > 1024, f"Figura suspeita/vazia: {path}"


def test_no_unverified_drug_repositioning_claims_in_paper_docs() -> None:
    forbidden = ("donepezil", "brexpiprazole", "dyrk2")
    for doc_path in EXPECTED_DOCS:
        lower = doc_path.read_text(encoding="utf-8").lower()
        for term in forbidden:
            assert term not in lower, f"Termo proibido '{term}' encontrado em {doc_path.name}"


def test_no_extreme_p_values_or_psup_in_paper_outputs() -> None:
    bad_p_tokens = ("p=0.000", "p=1.000", "p = 0.000", "p = 1.000", "p(sup)")
    targets = list(EXPECTED_DOCS) + [
        TABLES_DIR / f"{stem}.{ext}"
        for stem in EXPECTED_TABLES
        for ext in ("csv", "md", "tex")
    ]
    for path in targets:
        lower = path.read_text(encoding="utf-8").lower()
        for tok in bad_p_tokens:
            assert tok not in lower, f"Token '{tok}' proibido encontrado em {path.name}"


def test_t4_and_t5_have_full_cpu_audit_coverage() -> None:
    df_t4 = pd.read_csv(TABLES_DIR / "T4_marginal_and_paired.csv", dtype=str, keep_default_na=False)
    for col in (
        "drug_mean_ci",
        "target_mean_ci",
        "delta_mse_pareado",
        "delta_ci_pareado",
        "knn_tanimoto_ci_mse",
        "knn_kmer_ci_mse",
    ):
        assert not df_t4[col].str.contains("não disponível", case=False).any(), f"T4 coluna {col} ainda contém 'não disponível'"

    df_t5 = pd.read_csv(TABLES_DIR / "T5_uncensored_davis.csv", dtype=str, keep_default_na=False)
    for col in (
        "mean_predictor_mse_pkd_gt_5",
        "mean_predictor_ativo_mse_pkd_gt_5",
        "drug_mean_mse_pkd_gt_5",
        "target_mean_mse_pkd_gt_5",
    ):
        assert col in df_t5.columns, f"Coluna ausente em T5: {col}"
        assert not df_t5[col].str.contains("não disponível", case=False).any(), f"T5 coluna {col} ainda contém 'não disponível'"


def test_cpu_audits_json_and_esm_audit_script_exist() -> None:
    ca_path = PAPER_DIR / "cpu_audits.json"
    assert ca_path.exists() and ca_path.stat().st_size > 1000
    ca = json.loads(ca_path.read_text(encoding="utf-8"))
    for key in (
        "noise_floor",
        "davis_uncensored_baselines",
        "v1_marginals_knn_and_paired",
        "kiba_random_clean_timings",
        "davis_seqid_clusters",
    ):
        assert key in ca, f"Chave '{key}' ausente em cpu_audits.json"

    esm_audit_script = REPO_ROOT / "scripts" / "paper" / "audit_esm_cache_colab.py"
    assert esm_audit_script.exists() and esm_audit_script.stat().st_size > 200


def test_phase3_model_nomenclature_adr15() -> None:
    for stem, col in (
        ("T7_training_diagnostics", "modelo"),
        ("T10_phase3_results_summary", "modelo"),
        ("T11_phase3_global_oof_and_binary", "modelo"),
        ("T12_phase3_per_fold_or_seed", "modelo"),
        ("T14_phase3_diagnostic_controls", "controle_ou_modelo"),
        ("T15_phase3_configuration", "modelo"),
    ):
        df = pd.read_csv(TABLES_DIR / f"{stem}.csv", dtype=str, keep_default_na=False)
        bad = [v for v in df[col].unique() if "physdual" in str(v).lower()]
        assert not bad, f"Nomenclatura proibida por ADR #15 em {stem}.{col}: {bad}"


def test_phase3_audits_json_complete_aggregates() -> None:
    p3_path = PAPER_DIR / "phase3_audits.json"
    assert p3_path.exists() and p3_path.stat().st_size > 10_000
    p3 = json.loads(p3_path.read_text(encoding="utf-8"))
    for key in (
        "git_preregistration",
        "run_integrity",
        "dataset_overlap_and_units",
        "v2_kfold",
        "v1_seeds",
    ):
        assert key in p3, f"Chave '{key}' ausente em phase3_audits.json"

    ri = p3["run_integrity"]
    assert ri["isolation_failures"] == [], f"Falhas de isolamento detectadas: {ri['isolation_failures']}"
    assert ri["split_size_mismatches"] == [], f"Divergências de tamanho de split: {ri['split_size_mismatches']}"
    assert ri["nan_inf_runs"] == [], f"Runs com NaN/Inf detectados: {ri['nan_inf_runs']}"
    aggs = ri["aggregate_completeness"]
    assert len(aggs) == 34, f"Total esperado de agregados é 34, encontrado {len(aggs)}"
    incomplete = [a["file"] for a in aggs if not a.get("complete")]
    assert not incomplete, f"Agregados incompletos: {incomplete}"
    assert ri["start_timestamp_audit"]["post_runs_started_before_adr15_count"] == 0
    assert ri["stale_artifacts"]["phase3_davis_paired_bootstrap_json"]["is_stale"] is False


def test_phase3_intragroup_and_colab_scripts_present() -> None:
    import py_compile

    ig_path = PAPER_DIR / "phase3_intragroup_and_stacking.json"
    assert ig_path.exists() and ig_path.stat().st_size > 1000
    ig = json.loads(ig_path.read_text(encoding="utf-8"))
    for k in ("intragroup_v2_kfold", "intragroup_v1_seeds", "residual_stacking"):
        assert k in ig and len(ig[k]) > 0, f"Chave '{k}' ausente/vazia em phase3_intragroup_and_stacking.json"

    for reg_data in ig["intragroup_v2_kfold"].values():
        for m_data in reg_data["models"].values():
            assert "per_fold_min2" in m_data and "per_fold_min5" in m_data
            assert "min_pairs_2" not in m_data and "min_pairs_5" not in m_data

    for reg_data in ig["residual_stacking"].values():
        assert reg_data["alpha_grid"][-1] == 100_000_000.0
        assert "standardization" in reg_data
        assert len(reg_data["variants"]["res_esm"]["chosen_alphas"]) == 5
        for v_data in reg_data["variants"].values():
            assert "intragroup_per_unit_min2" in v_data and "intragroup_per_unit_min5" in v_data
            assert "intragroup_min2" not in v_data and "intragroup_min5" not in v_data

    sw_path = PAPER_DIR / "seqid_identity_audit.json"
    assert sw_path.exists() and sw_path.stat().st_size > 1000
    sw = json.loads(sw_path.read_text(encoding="utf-8"))
    for ds in ("davis", "kiba"):
        for reg in ("cold_target_seqid30", "cold_target_seqid40"):
            r = sw["datasets"][ds]["regimes"][reg]
            assert len(r["folds"]) == 5
            overall = r["across_all_units_test_proteins"]
            if reg.endswith("30"):
                assert overall["global_id_max"]["count_gt_30"] == 0
            else:
                assert overall["global_id_max"]["count_gt_40"] == 0
            assert "sw_id_aln_len_ge150aa" in overall

    for sname in ("colab_ridge_esm_pooling.py", "colab_kiba_cold_drug_ecfp.py"):
        sp = REPO_ROOT / "scripts" / "paper" / sname
        assert sp.exists() and sp.stat().st_size > 500, f"Script Colab ausente: {sp}"
        py_compile.compile(str(sp), doraise=True)

    pooling_src = (REPO_ROOT / "scripts" / "paper" / "colab_ridge_esm_pooling.py").read_text(encoding="utf-8")
    assert "PHASE4_DECISION_CRITERION" in pooling_src
    assert "100000000.0" in pooling_src
