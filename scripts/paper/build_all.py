"""Orquestrador determinístico do pacote científico das Fases 1 e 2.

Executa em ordem:
1. Inventário e guia de reprodução (`00_INVENTARIO.md`, `REPRODUCAO.md`)
2. Tabelas consolidadas (`T1`–`T8` em `.csv`, `.md`, `.tex`)
3. Figuras com paleta daltônica (`F1`–`F6` em `.pdf`, `.png` e `CAPTIONS.md`)

Verifica que todos os artefatos de entrada obrigatórios e todas as saídas
esperadas existam e não estejam vazios, saindo com código != 0 em caso de falha.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PAPER_SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PAPER_SCRIPTS_DIR.parent.parent
SRC_DIR = PROJECT_ROOT / "src"

for p in (str(PAPER_SCRIPTS_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from build_inventory_and_repro import build_inventory_and_repro  # noqa: E402
from build_tables import build_all_tables  # noqa: E402
from common import resolve_roots  # noqa: E402
from make_figures import make_all_figures  # noqa: E402


REQUIRED_REPO_INPUTS = [
    "configs/base.yaml",
    "docs/DECISOES.md",
    "docs/FASE1_DADOS.md",
    "docs/FASE2_BASELINE.md",
    "docs/colab_env_snapshot.txt",
]

REQUIRED_DATA_INPUTS = [
    "data/processed/davis_clean.csv",
    "data/processed/kiba_clean.csv",
    "data/processed/davis_ingestion_report.json",
    "data/processed/kiba_ingestion_report.json",
    "outputs/logs/phase1_summary_report.json",
    "outputs/results/phase2_timing_benchmark.json",
]

MANUAL_PAPER_DOCS = [
    "docs/paper/01_METODOS_DADOS_E_BASELINES.md",
    "docs/paper/02_RESULTADOS_PRELIMINARES.md",
    "docs/paper/03_LIMITACOES_E_AMEACAS_A_VALIDADE.md",
    "docs/paper/04_REGISTRO_DE_AFIRMACOES.md",
    "docs/paper/05_LACUNAS.md",
]


def _assert_nonempty_files(paths: list[Path], label: str) -> list[str]:
    errors: list[str] = []
    for path in paths:
        if not path.exists():
            errors.append(f"[{label}] Ausente: {path}")
        elif not path.is_file():
            errors.append(f"[{label}] Não é arquivo: {path}")
        elif path.stat().st_size == 0:
            errors.append(f"[{label}] Arquivo vazio: {path}")
    return errors


def run_pipeline(check_manual_docs: bool = True) -> int:
    repo_root, data_root = resolve_roots()
    results_dir = data_root / "outputs" / "results"
    paper_dir = repo_root / "docs" / "paper"

    missing_inputs = _assert_nonempty_files(
        [repo_root / rel for rel in REQUIRED_REPO_INPUTS],
        label="REPO_INPUT",
    )
    missing_inputs.extend(
        _assert_nonempty_files(
            [data_root / rel for rel in REQUIRED_DATA_INPUTS],
            label="DATA_INPUT",
        )
    )
    runs_dir = results_dir / "runs"
    if not runs_dir.exists() or not any(runs_dir.glob("*.json")):
        missing_inputs.append(f"[INPUT] Nenhum JSON de run encontrado em {runs_dir}")
    if not any(results_dir.glob("baseline_*.json")):
        missing_inputs.append(
            f"[INPUT] Nenhum JSON agregado baseline_*.json em {results_dir}"
        )
    if missing_inputs:
        for msg in missing_inputs:
            print(msg, file=sys.stderr)
        return 2

    inv_path, repro_path = build_inventory_and_repro()
    table_outputs = build_all_tables()
    figure_outputs = make_all_figures()

    generated_paths: list[Path] = [inv_path, repro_path]
    for tid in [f"T{i}" for i in range(1, 9)]:
        if tid not in table_outputs:
            print(f"[OUTPUT] Tabela {tid} ausente no retorno", file=sys.stderr)
            return 3
        for ext in ("csv", "md", "tex"):
            if ext not in table_outputs[tid]:
                print(f"[OUTPUT] Formato {ext} ausente para {tid}", file=sys.stderr)
                return 3
            generated_paths.append(table_outputs[tid][ext])

    for fid in [f"F{i}" for i in range(1, 7)]:
        if fid not in figure_outputs:
            print(f"[OUTPUT] Figura {fid} ausente no retorno", file=sys.stderr)
            return 3
        for ext in ("pdf", "png"):
            if ext not in figure_outputs[fid]:
                print(f"[OUTPUT] Formato {ext} ausente para {fid}", file=sys.stderr)
                return 3
            generated_paths.append(figure_outputs[fid][ext])

    if "CAPTIONS" not in figure_outputs or "md" not in figure_outputs["CAPTIONS"]:
        print("[OUTPUT] CAPTIONS.md ausente no retorno", file=sys.stderr)
        return 3
    generated_paths.append(figure_outputs["CAPTIONS"]["md"])

    missing_outputs = _assert_nonempty_files(generated_paths, label="OUTPUT")
    if check_manual_docs:
        missing_outputs.extend(
            _assert_nonempty_files(
                [repo_root / rel for rel in MANUAL_PAPER_DOCS],
                label="PAPER_DOC",
            )
        )

    if missing_outputs:
        for msg in missing_outputs:
            print(msg, file=sys.stderr)
        return 4

    print(
        f"OK: {len(generated_paths)} artefatos gerados e validados em {paper_dir}"
        + (f" (+ {len(MANUAL_PAPER_DOCS)} docs verificados)" if check_manual_docs else "")
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build completo do pacote de dados do paper")
    parser.add_argument(
        "--generated-only",
        action="store_true",
        help="Verifica apenas artefatos gerados automaticamente (ignora 01..05 manuais)",
    )
    args = parser.parse_args(argv)
    return run_pipeline(check_manual_docs=not args.generated_only)


if __name__ == "__main__":
    raise SystemExit(main())
