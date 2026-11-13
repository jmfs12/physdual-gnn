"""Testes de esquema, integridade e ausência de NaNs do pacote científico (Fases 1 e 2)."""

from __future__ import annotations

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
    "T7_training_diagnostics": (65, ["protocolo", "modelo", "dataset", "split", "unidade", "best_epoch"]),
    "T8_configuration": (13, ["protocolo", "modelo", "dataset", "split", "n_runs", "featurizer"]),
}

EXPECTED_FIGURES = [
    "F1_affinity_distributions",
    "F2_ci_mse_by_model_split",
    "F3_paired_delta_mse_cold",
    "F4_learning_curves_davis_cold_target",
    "F5_best_epoch_distribution",
    "F6_seq_length_stratification",
]


def test_build_all_orchestrator_runs_cleanly() -> None:
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "paper" / "build_all.py")],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"build_all.py falhou ({proc.returncode}):\n{proc.stderr}\n{proc.stdout}"


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
    for stem in ("T2_splits", "T3_main_results", "T4_marginal_and_paired", "T5_uncensored_davis", "T6_seq_length", "T7_training_diagnostics", "T8_configuration"):
        df = pd.read_csv(TABLES_DIR / f"{stem}.csv", dtype=str, keep_default_na=False)
        protocols = set(df["protocolo"].unique())
        assert protocols == {"v1_seeds", "v2_kfold"}, f"Protocolos inesperados em {stem}: {protocols}"


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
    import json

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
