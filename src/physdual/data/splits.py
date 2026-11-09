"""Particionamento determinístico de splits (`random`, `cold_drug`, `cold_target`) sem vazamento por conteúdo.

Agrupa entidades únicas pelo conteúdo químico/biológico:
- `cold_drug`: agrupa por `smiles` canônico (dois `drug_id` com o mesmo `smiles` caem juntos).
- `cold_target`: agrupa por `sequence` (dois `target_id` com a mesma `sequence` caem juntos).
- `random`: particiona pares aleatoriamente.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

VALID_SPLIT_TYPES = ("random", "cold_drug", "cold_target")


def _compute_partition_bounds(n_items: int, ratios: tuple[float, float, float]) -> tuple[int, int]:
    """Calcula índices de corte `(n_train, n_train + n_val)` garantindo partições não vazias se `n_items >= 3`."""
    if n_items < 3:
        raise ValueError(
            f"Número de entidades/itens únicos ({n_items}) insuficiente para dividir em train/val/test (>= 3 exigido)."
        )
    r_train, r_val, r_test = ratios
    total_r = r_train + r_val + r_test
    if abs(total_r - 1.0) > 1e-5 or any(r <= 0.0 for r in ratios):
        raise ValueError(f"As proporções de split devem ser positivas e somar 1.0, recebido: {ratios}")

    n_val = max(1, int(round(n_items * r_val)))
    n_test = max(1, int(round(n_items * r_test)))
    n_train = n_items - n_val - n_test
    if n_train < 1:
        n_train = 1
        rem = n_items - 1
        n_val = max(1, rem // 2)
        n_test = max(1, rem - n_val)
    return n_train, n_train + n_val


def create_splits(
    df: pd.DataFrame,
    split_type: str,
    seed: int,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, list[int]]:
    """Cria índices determinísticos de `train`, `val` e `test` para o DataFrame padronizado.

    Args:
        df: DataFrame contendo colunas `smiles` e `sequence`.
        split_type: `"random"`, `"cold_drug"` ou `"cold_target"`.
        seed: Semente inteira para `np.random.RandomState(seed)`.
        ratios: Tupla `(train_ratio, val_ratio, test_ratio)`.

    Returns:
        Dicionário `{"train": list[int], "val": list[int], "test": list[int]}` com índices posicionais (`0..len(df)-1`).
    """
    st = split_type.strip().lower()
    if st not in VALID_SPLIT_TYPES:
        raise ValueError(f"split_type inválido: '{split_type}'. Opções válidas: {VALID_SPLIT_TYPES}")

    rng = np.random.RandomState(int(seed))
    n_rows = len(df)

    if st == "random":
        cut1, cut2 = _compute_partition_bounds(n_rows, ratios)
        perm = rng.permutation(n_rows).tolist()
        return {
            "train": [int(i) for i in perm[:cut1]],
            "val": [int(i) for i in perm[cut1:cut2]],
            "test": [int(i) for i in perm[cut2:]],
        }

    group_col = "smiles" if st == "cold_drug" else "sequence"
    if group_col not in df.columns:
        raise ValueError(f"Coluna obrigatória '{group_col}' ausente no DataFrame para split '{st}'.")

    # Ordena chaves únicas antes do shuffle para determinismo absoluto entre plataformas
    unique_entities = sorted(df[group_col].astype(str).unique().tolist())
    cut1, cut2 = _compute_partition_bounds(len(unique_entities), ratios)

    perm_idx = rng.permutation(len(unique_entities))
    shuffled_entities = [unique_entities[i] for i in perm_idx]

    train_set = set(shuffled_entities[:cut1])
    val_set = set(shuffled_entities[cut1:cut2])
    test_set = set(shuffled_entities[cut2:])

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for idx, val in enumerate(df[group_col].astype(str).tolist()):
        if val in train_set:
            train_indices.append(idx)
        elif val in val_set:
            val_indices.append(idx)
        elif val in test_set:
            test_indices.append(idx)

    return {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }


def verify_split_leakage(
    df: pd.DataFrame,
    splits: dict[str, list[int]],
    split_type: str,
) -> bool:
    """Verifica ausência estrita de vazamento por conteúdo, não-vacuidade e cobertura total do DataFrame.

    Retorna `True` apenas se:
    1. `train`, `val` e `test` existirem e forem todos não vazios.
    2. Os conjuntos de índices de `train`, `val` e `test` forem mutuamente disjuntos e cobrirem `0..len(df)-1`.
    3. Em `cold_drug`: os conjuntos de `smiles` canônicos em `train`, `val` e `test` forem mutuamente disjuntos.
    4. Em `cold_target`: os conjuntos de `sequence` em `train`, `val` e `test` forem mutuamente disjuntos.
    """
    st = split_type.strip().lower()
    if st not in VALID_SPLIT_TYPES:
        raise ValueError(f"split_type inválido: '{split_type}'. Opções válidas: {VALID_SPLIT_TYPES}")

    for key in ("train", "val", "test"):
        if key not in splits or len(splits[key]) == 0:
            return False

    train_idx = set(splits["train"])
    val_idx = set(splits["val"])
    test_idx = set(splits["test"])

    # Verifica duplicatas internas na mesma partição
    if (
        len(train_idx) != len(splits["train"])
        or len(val_idx) != len(splits["val"])
        or len(test_idx) != len(splits["test"])
    ):
        return False

    # Verifica disjunção de índices e cobertura exata do DataFrame
    if not (train_idx.isdisjoint(val_idx) and train_idx.isdisjoint(test_idx) and val_idx.isdisjoint(test_idx)):
        return False

    union_idx = train_idx | val_idx | test_idx
    if union_idx != set(range(len(df))):
        return False

    if st == "random":
        return True

    content_col = "smiles" if st == "cold_drug" else "sequence"
    if content_col not in df.columns:
        raise ValueError(f"Coluna '{content_col}' ausente no DataFrame para verificação de '{st}'.")

    train_content = set(df.iloc[splits["train"]][content_col].astype(str))
    val_content = set(df.iloc[splits["val"]][content_col].astype(str))
    test_content = set(df.iloc[splits["test"]][content_col].astype(str))

    return (
        train_content.isdisjoint(val_content)
        and train_content.isdisjoint(test_content)
        and val_content.isdisjoint(test_content)
    )


def save_splits(splits: dict[str, list[int]], output_path: Path) -> Path:
    """Salva o dicionário de splits em formato JSON."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: [int(i) for i in v] for k, v in splits.items()}
    out.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return out


def load_splits(split_path: Path) -> dict[str, list[int]]:
    """Carrega um arquivo JSON de splits e devolve `{ "train": [...], "val": [...], "test": [...] }`."""
    path = Path(split_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): [int(i) for i in v] for k, v in data.items()}
