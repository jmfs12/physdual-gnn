"""Funções compartilhadas, resolução de caminhos e exportadores CSV/MD/LaTeX (somente leitura)."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from physdual.utils.paths import get_project_root, get_repo_root  # noqa: E402

NA_TEXT = "não disponível"

NUMERIC_POLICY_NOTE = (
    "Política numérica padronizada: métricas contínuas (CI, MSE, RMSE, Pearson, Spearman, rm2, AUROC, AUPR, deltas) "
    "com 4 casas decimais (0.0000); percentuais com 2 casas decimais (0.00%); médias de épocas com 1 casa (0.0) "
    "e épocas individuais como inteiros; tempo por época com 3 casas (0.000 s) e horas totais com 2 casas (0.00 h). "
    "Dispersão expressa como média ± desvio-padrão amostral (ddof=1)."
)


def resolve_roots() -> tuple[Path, Path]:
    """Retorna `(repo_root, data_root)` respeitando `PHYSDUAL_ROOT` para dados/outputs."""
    repo_root = get_repo_root()
    data_root = get_project_root()
    return repo_root, data_root


def ensure_paper_dirs(repo_root: Path | None = None) -> dict[str, Path]:
    """Garante os diretórios de saída em `docs/paper/`."""
    base = (repo_root or REPO_ROOT) / "docs" / "paper"
    tables = base / "tables"
    figures = base / "figures"
    for d in (base, tables, figures):
        d.mkdir(parents=True, exist_ok=True)
    return {"paper": base, "tables": tables, "figures": figures}


def sha256_file(path: Path) -> str:
    """Calcula SHA-256 hexadecimal de um arquivo sem alterá-lo."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json_strict(path: Path) -> dict[str, Any]:
    """Lê um JSON obrigatório; falha explicitamente se ausente."""
    if not path.exists():
        raise FileNotFoundError(f"Artefato obrigatório ausente: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt_float(val: Any, decimals: int = 4) -> str:
    """Formata float com precisão fixa ou retorna 'não disponível'."""
    if val is None:
        return NA_TEXT
    try:
        fval = float(val)
    except (TypeError, ValueError):
        return NA_TEXT
    if math.isnan(fval) or math.isinf(fval):
        return NA_TEXT
    return f"{fval:.{decimals}f}"


def fmt_mean_std(mean_val: Any, std_val: Any, decimals: int = 4, signed: bool = False) -> str:
    """Formata `média ± desvio` ou retorna 'não disponível'."""
    if mean_val is None or std_val is None:
        return NA_TEXT
    try:
        m = float(mean_val)
        s = float(std_val)
    except (TypeError, ValueError):
        return NA_TEXT
    if math.isnan(m) or math.isnan(s):
        return NA_TEXT
    fmt = f"{{:{'+' if signed else ''}.{decimals}f}}"
    return f"{fmt.format(m)} ± {m_std(s, decimals)}"


def m_std(s: float, decimals: int) -> str:
    return f"{s:.{decimals}f}"


def sample_mean_std(values: list[float]) -> tuple[float, float]:
    """Calcula média e desvio amostral (ddof=1) para uma lista não vazia."""
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values))


def escape_latex(text: str) -> str:
    """Escapa caracteres especiais LaTeX preservando sinais matemáticos já formatados."""
    s = str(text)
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
        ("±", r"$\pm$"),
        ("≤", r"$\le$"),
        ("≥", r"$\ge$"),
        ("→", r"$\rightarrow$"),
        ("Δ", r"$\Delta$"),
    ]
    for old, new in replacements:
        s = s.replace(old, new)
    return s


def df_to_markdown(df: pd.DataFrame) -> str:
    """Converte DataFrame em tabela Markdown GFM sem dependências extras."""
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = [str(row[c]).replace("\n", " ").replace("|", r"\|") for c in df.columns]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def df_to_latex(df: pd.DataFrame, caption: str, label: str, source_note: str) -> str:
    """Converte DataFrame em ambiente `table` LaTeX (booktabs)."""
    ncols = len(df.columns)
    col_spec = "l" * ncols
    header = " & ".join(f"\\textbf{{{escape_latex(str(c))}}}" for c in df.columns) + r" \\"
    body_lines = []
    for _, row in df.iterrows():
        vals = [escape_latex(str(row[c])) for c in df.columns]
        body_lines.append(" & ".join(vals) + r" \\")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        f"\\caption{{{escape_latex(caption)}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
        *body_lines,
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{flushleft}",
        f"\\footnotesize \\textit{{Fonte:}} {escape_latex(source_note)}",
        r"\end{flushleft}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def write_table(
    stem: str,
    title: str,
    df: pd.DataFrame,
    source_note: str,
    tables_dir: Path,
    extra_notes: list[str] | None = None,
) -> dict[str, Path]:
    """Escreve a tabela em `.csv`, `.md` e `.tex` simultaneamente."""
    csv_path = tables_dir / f"{stem}.csv"
    md_path = tables_dir / f"{stem}.md"
    tex_path = tables_dir / f"{stem}.tex"

    df.to_csv(csv_path, index=False, encoding="utf-8")

    md_lines = [
        f"# {title}",
        "",
        f"**Fonte:** {source_note}",
        "",
        f"**Formato numérico:** {NUMERIC_POLICY_NOTE}",
        "",
        df_to_markdown(df),
    ]
    if extra_notes:
        md_lines.append("")
        for note in extra_notes:
            md_lines.append(f"- {note}")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    label = f"tab:{stem.lower()}"
    tex_caption = f"{title}. {NUMERIC_POLICY_NOTE}"
    tex_source = source_note + ((" " + " ".join(extra_notes)) if extra_notes else "")
    tex_path.write_text(df_to_latex(df, tex_caption, label, tex_source), encoding="utf-8")

    return {"csv": csv_path, "md": md_path, "tex": tex_path}
