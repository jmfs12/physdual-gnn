"""Carregamento de configuração YAML e resolução de caminhos relativos à raiz do projeto."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from physdual.utils.paths import get_project_root, get_repo_root


def get_default_config_path() -> Path:
    """Retorna o caminho padrão para `configs/base.yaml` no repositório."""
    return get_repo_root() / "configs" / "base.yaml"


def load_config(
    config_path: str | Path | None = None,
    root: str | Path | None = None,
    resolve_paths: bool = True,
) -> dict[str, Any]:
    """Carrega um arquivo YAML de configuração e resolve os caminhos da seção `paths`.

    Args:
        config_path: Caminho para o arquivo YAML. Se omitido, usa `configs/base.yaml`.
        root: Raiz base para resolução dos caminhos relativos. Se omitido, usa `get_project_root()`.
        resolve_paths: Se True, converte cada caminho relativo em `paths` para `Path` absoluto.

    Returns:
        Dicionário com a configuração carregada.

    Raises:
        FileNotFoundError: Se o arquivo YAML não existir.
        ValueError: Se algum caminho em `paths` no YAML for absoluto.
    """
    cfg_file = Path(config_path) if config_path is not None else get_default_config_path()
    if not cfg_file.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {cfg_file}")

    with cfg_file.open("r", encoding="utf-8") as fp:
        cfg: dict[str, Any] = yaml.safe_load(fp) or {}

    cfg = deepcopy(cfg)
    base_root = Path(root).expanduser().resolve() if root is not None else get_project_root()

    raw_paths = cfg.get("paths", {})
    if not isinstance(raw_paths, dict):
        raise ValueError("A seção 'paths' do YAML deve ser um mapeamento chave-valor.")

    resolved_paths: dict[str, Path | str] = {}
    for key, rel_value in raw_paths.items():
        rel_str = str(rel_value)
        rel_path = Path(rel_str)
        if rel_path.is_absolute() or rel_str.startswith(("/", "\\")):
            raise ValueError(
                f"Caminho absoluto proibido em configs ('paths.{key}': '{rel_value}'). "
                "Use apenas caminhos relativos à raiz do projeto."
            )
        resolved_paths[key] = (base_root / rel_path) if resolve_paths else rel_str

    cfg["paths"] = resolved_paths
    cfg["project_root"] = base_root
    return cfg
