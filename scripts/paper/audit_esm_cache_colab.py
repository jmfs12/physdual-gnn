"""Auditoria do cache de embeddings ESM-2 no Google Drive / Colab.

Verifica em cada arquivo `.pt` presente em `cache/esm_embeddings/`:
1. Existência da chave/tensor de embedding com shape `[len(seq), esm_dim]` (`esm_dim = 1280`).
2. Presença da flag booleana `windowed` e consistência da contagem total (`esperado = 112`:
   `70` sequências únicas `> 1022` aa no Davis + `42` no KIBA).

Uso no Colab (com Drive montado):
    PHYSDUAL_ROOT=/content/drive/MyDrive/physdual-gnn python scripts/paper/audit_esm_cache_colab.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch


def main() -> int:
    root = Path(os.environ.get("PHYSDUAL_ROOT", Path(__file__).resolve().parents[2]))
    cache_dir = root / "cache" / "esm_embeddings"
    if not cache_dir.exists():
        print(f"ERRO: diretório ausente: {cache_dir}", file=sys.stderr)
        return 2

    pt_files = sorted(cache_dir.rglob("*.pt"))
    if not pt_files:
        print(f"ERRO: nenhum arquivo .pt encontrado em {cache_dir}", file=sys.stderr)
        return 3

    expected_seqs: dict[str, int] = {}
    for ds in ("davis", "kiba"):
        csv_path = root / "data" / "processed" / f"{ds}_clean.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            for seq in df["sequence"].unique():
                expected_seqs[str(seq)] = len(str(seq))

    windowed_count = 0
    shape_errors: list[str] = []
    missing_windowed_field = 0

    for p in pt_files:
        obj: Any = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            emb = obj.get("embedding", obj.get("embeddings", obj.get("repr")))
            win = obj.get("windowed")
            if win is None:
                missing_windowed_field += 1
            elif bool(win):
                windowed_count += 1
            seq = obj.get("sequence")
            exp_len = len(seq) if isinstance(seq, str) else obj.get("seq_len")
        elif isinstance(obj, torch.Tensor):
            emb = obj
            missing_windowed_field += 1
            exp_len = None
        else:
            shape_errors.append(f"{p.name}: tipo inesperado {type(obj)}")
            continue

        if not isinstance(emb, torch.Tensor) or emb.ndim != 2 or emb.shape[1] != 1280:
            shape_errors.append(f"{p.name}: shape inválido {getattr(emb, 'shape', None)}")
            continue
        if exp_len is not None and int(emb.shape[0]) != int(exp_len):
            shape_errors.append(f"{p.name}: L={emb.shape[0]} != esperado {exp_len}")

    report = {
        "cache_dir": str(cache_dir),
        "total_pt_files": len(pt_files),
        "expected_unique_sequences": len(expected_seqs),
        "windowed_count": windowed_count,
        "expected_windowed_count": 112,
        "missing_windowed_field": missing_windowed_field,
        "shape_errors": shape_errors[:20],
        "num_shape_errors": len(shape_errors),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if shape_errors or missing_windowed_field > 0 or windowed_count != 112:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
