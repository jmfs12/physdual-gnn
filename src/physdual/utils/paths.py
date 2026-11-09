"""Detecção de ambiente (Colab vs. local) e resolução da raiz do projeto."""

from __future__ import annotations

import os
import sys
from pathlib import Path

DEFAULT_COLAB_ROOT = Path("/content/drive/MyDrive/physdual-gnn")


def is_colab() -> bool:
    """Retorna True se o código estiver executando em um runtime do Google Colab."""
    return "google.colab" in sys.modules


def get_repo_root() -> Path:
    """Devolve o diretório raiz do repositório Git onde `src/` está localizado."""
    return Path(__file__).resolve().parents[3]


def get_project_root() -> Path:
    """Devolve a raiz de dados/artefatos do projeto.

    Ordem de precedência:
    1. Variável de ambiente `PHYSDUAL_ROOT`, se definida.
    2. `/content/drive/MyDrive/physdual-gnn`, se rodando no Google Colab.
    3. Raiz local do repositório Git.
    """
    env_root = os.environ.get("PHYSDUAL_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    if is_colab():
        return DEFAULT_COLAB_ROOT
    return get_repo_root()


def ensure_project_dirs(root: Path | None = None) -> dict[str, Path]:
    """Garante a existência da estrutura de diretórios de trabalho e devolve seus caminhos."""
    base = Path(root) if root is not None else get_project_root()
    dirs = {
        "root": base,
        "data_raw": base / "data" / "raw",
        "data_processed": base / "data" / "processed",
        "esm_cache": base / "cache" / "esm_embeddings",
        "checkpoints": base / "outputs" / "checkpoints",
        "results": base / "outputs" / "results",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs
