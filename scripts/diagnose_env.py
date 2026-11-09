#!/usr/bin/env python3
"""Script de diagnóstico do ambiente (Passo 0) e geração opcional de snapshot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from physdual.utils.env_info import collect_env_info, format_env_report, write_env_snapshot  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Processa argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Imprime diagnóstico completo do ambiente PhysDual-GNN."
    )
    parser.add_argument(
        "--snapshot",
        nargs="?",
        const=str(REPO_ROOT / "docs" / "colab_env_snapshot.txt"),
        default=None,
        help="Grava snapshot completo (diagnóstico + pip freeze) no arquivo indicado.",
    )
    return parser.parse_args()


def main() -> int:
    """Executa diagnóstico do ambiente e, opcionalmente, salva o snapshot."""
    args = parse_args()
    info = collect_env_info()
    print(format_env_report(info))

    if args.snapshot:
        out_path = write_env_snapshot(args.snapshot)
        print(f"\n[OK] Snapshot do ambiente gravado em: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
