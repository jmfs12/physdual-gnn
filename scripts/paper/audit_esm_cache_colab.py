"""Auditoria do cache de embeddings ESM-2 no Google Drive / Colab.

Verifica cada arquivo `.pt` presente em `cache/esm_embeddings/{model_name}/`:
1. Cobertura completa das `429` sequências únicas globais (`Davis ∪ KIBA`).
2. Carregamento seguro com `weights_only=True`, `dtype == torch.float16` em disco e
   ausência total de `NaN`/`Inf` (`torch.isfinite(emb).all()`).
3. Tensor 2D com `shape == [len(seq), 1280]`, metadado `dim == 1280` e metadado
   `seq_len == len(seq)` validado contra as sequências únicas de `davis_clean.csv` e `kiba_clean.csv`.
4. Presença da flag booleana `windowed` e consistência da contagem deduplicada global
   (`esperado = 83` sequências únicas `> 1022` aa em `Davis ∪ KIBA`, equivalente a
   `70` Davis + `42` KIBA − `29` compartilhadas = `83`; soma por dataset = `112`).
5. Reporte informativo de eventuais `.pt` residuais na raiz `cache/esm_embeddings/*.pt`
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
    missing_windowed_field = 0
    shape_errors: list[str] = []
    dtype_errors: list[str] = []
    nan_inf_errors: list[str] = []
    seq_len_errors: list[str] = []
    dim_meta_errors: list[str] = []
    weights_only_errors: list[str] = []

    for p in pt_files:
        try:
            obj: Any = torch.load(p, map_location="cpu", weights_only=True)
        except Exception as exc:
            weights_only_errors.append(f"{p.name}: falha weights_only=True ({exc})")
            continue

        csv_seq_len = expected_hashes.get(p.stem)
        meta_seq_len = None
        meta_dim = None

        if isinstance(obj, dict):
            emb = obj.get("embedding", obj.get("embeddings", obj.get("repr")))
            win = obj.get("windowed")
            if win is None:
                missing_windowed_field += 1
            elif bool(win):
                windowed_count += 1
            meta_seq_len = obj.get("seq_len")
            meta_dim = obj.get("dim")
        elif isinstance(obj, torch.Tensor):
            emb = obj
            missing_windowed_field += 1
        else:
            shape_errors.append(f"{p.name}: tipo inesperado {type(obj)}")
            continue

        if not isinstance(emb, torch.Tensor):
            shape_errors.append(f"{p.name}: embedding ausente ou não é Tensor ({type(emb)})")
            continue

        if emb.dtype != torch.float16:
            dtype_errors.append(f"{p.name}: dtype={emb.dtype} != torch.float16")

        if not bool(torch.isfinite(emb).all()):
            n_nan = int(torch.isnan(emb).sum().item())
            n_inf = int(torch.isinf(emb).sum().item())
            nan_inf_errors.append(f"{p.name}: NaN={n_nan}, Inf={n_inf}")

        if emb.ndim != 2 or int(emb.shape[1]) != 1280:
            shape_errors.append(f"{p.name}: shape inválido {tuple(emb.shape)} (esperado [L, 1280])")

        if isinstance(obj, dict) and (meta_dim is None or int(meta_dim) != 1280):
            dim_meta_errors.append(f"{p.name}: metadado dim={meta_dim} != 1280")

        if csv_seq_len is None:
            seq_len_errors.append(f"{p.name}: hash ausente nos CSVs esperados (seq_len meta={meta_seq_len}, shape[0]={int(emb.shape[0]) if emb.ndim >= 1 else None})")
        else:
            if isinstance(obj, dict) and (meta_seq_len is None or int(meta_seq_len) != csv_seq_len):
                seq_len_errors.append(f"{p.name}: metadado seq_len={meta_seq_len} != len(seq)={csv_seq_len}")
            if emb.ndim < 1 or int(emb.shape[0]) != csv_seq_len:
                seq_len_errors.append(f"{p.name}: emb.shape[0]={emb.shape[0] if emb.ndim >= 1 else None} != len(seq)={csv_seq_len}")

    has_failures = any([
        weights_only_errors,
        shape_errors,
        dtype_errors,
        nan_inf_errors,
        seq_len_errors,
        dim_meta_errors,
        missing_windowed_field > 0,
        bool(missing_expected_hashes),
        windowed_count != expected_windowed_global,
    ])

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
        "num_weights_only_errors": len(weights_only_errors),
        "num_dtype_errors": len(dtype_errors),
        "num_nan_inf_errors": len(nan_inf_errors),
        "num_shape_errors": len(shape_errors),
        "num_dim_meta_errors": len(dim_meta_errors),
        "num_seq_len_errors": len(seq_len_errors),
        "weights_only_errors": weights_only_errors[:20],
        "dtype_errors": dtype_errors[:20],
        "nan_inf_errors": nan_inf_errors[:20],
        "shape_errors": shape_errors[:20],
        "dim_meta_errors": dim_meta_errors[:20],
        "seq_len_errors": seq_len_errors[:20],
        "root_residual_pt_files": root_pt_files,
        "all_checks_passed": not has_failures,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 4 if has_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
