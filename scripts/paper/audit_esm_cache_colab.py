"""Auditoria do cache de embeddings ESM-2 no Google Drive / Colab.

Verifica cada arquivo `.pt` presente em `cache/esm_embeddings/{model_name}/`:
1. Cobertura completa das `429` sequências únicas globais (`Davis ∪ KIBA`).
2. Existência da chave/tensor de embedding com shape `[len(seq), esm_dim]` (`esm_dim = 1280`).
3. Presença da flag booleana `windowed` e consistência da contagem deduplicada global
   (`esperado = 83` sequências únicas `> 1022` aa em `Davis ∪ KIBA`, equivalente a
   `70` Davis + `42` KIBA − `29` compartilhadas = `83`; soma por dataset = `112`).
4. Reporte informativo de eventuais `.pt` residuais na raiz `cache/esm_embeddings/*.pt`
   (como `4637bffbde58d549.pt`, gerado por `scripts/smoke_test.py` com `esm2_t6_8M_UR50D`).

Uso no Colab (com Drive montado):
    PHYSDUAL_ROOT=/content/drive/MyDrive/physdual-gnn python scripts/paper/audit_esm_cache_colab.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch


def main() -> int:
    root = Path(os.environ.get("PHYSDUAL_ROOT", Path(__file__).resolve().parents[2]))
    model_name = os.environ.get("PHYSDUAL_ESM_MODEL", "esm2_t33_650M_UR50D").strip()
    cache_root = root / "cache" / "esm_embeddings"
    if not cache_root.exists():
        print(f"ERRO: diretório ausente: {cache_root}", file=sys.stderr)
        return 2

    model_dir = cache_root / model_name
    target_dir = model_dir if model_dir.exists() else cache_root
    pt_files = sorted(target_dir.glob("*.pt"))
    if not pt_files:
        print(f"ERRO: nenhum arquivo .pt encontrado em {target_dir}", file=sys.stderr)
        return 3

    root_pt_files = sorted(p.name for p in cache_root.glob("*.pt")) if target_dir == model_dir else []

    expected_seqs: dict[str, int] = {}
    expected_hashes: dict[str, int] = {}
    per_ds_windowed_sum = 0
    for ds in ("davis", "kiba"):
        csv_path = root / "data" / "processed" / f"{ds}_clean.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            for seq in df["sequence"].unique():
                s = str(seq)
                expected_seqs[s] = len(s)
                h = hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]
                expected_hashes[h] = len(s)
                if len(s) > 1022:
                    per_ds_windowed_sum += 1

    expected_windowed_global = sum(1 for length in expected_seqs.values() if length > 1022)
    found_hashes = {p.stem for p in pt_files}
    missing_expected_hashes = sorted(h for h in expected_hashes if h not in found_hashes)

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
        "cache_dir": str(target_dir),
        "model_name": model_name,
        "total_pt_files": len(pt_files),
        "expected_unique_sequences": len(expected_seqs),
        "missing_expected_files": len(missing_expected_hashes),
        "windowed_count": windowed_count,
        "expected_windowed_global_unique": expected_windowed_global,
        "expected_windowed_per_dataset_sum": per_ds_windowed_sum,
        "missing_windowed_field": missing_windowed_field,
        "shape_errors": shape_errors[:20],
        "num_shape_errors": len(shape_errors),
        "root_residual_pt_files": root_pt_files,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if (
        shape_errors
        or missing_windowed_field > 0
        or missing_expected_hashes
        or windowed_count != expected_windowed_global
    ):
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
