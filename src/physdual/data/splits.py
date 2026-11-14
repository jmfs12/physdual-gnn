"""Particionamento determinístico de splits (`random`, `cold_drug`, `cold_target`) sem vazamento por conteúdo.

Agrupa entidades únicas pelo conteúdo químico/biológico:
- `cold_drug`: agrupa por `smiles` canônico (dois `drug_id` com o mesmo `smiles` caem juntos).
- `cold_target`: agrupa por `sequence` (dois `target_id` com a mesma `sequence` caem juntos).
- `random`: particiona pares aleatoriamente.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

VALID_SPLIT_TYPES = ("random", "cold_drug", "cold_target")


def _normalize_split_type(split_type: str) -> str:
    """Normaliza variantes de `cold_target` (como `cold_target_seqid30`/`seqid40`) para o tipo base."""
    st = str(split_type).strip().lower()
    if st.startswith("cold_target_seqid") or st in ("seqid30", "seqid40", "seqid50"):
        return "cold_target"
    return st


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
    st = _normalize_split_type(split_type)
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
    st = _normalize_split_type(split_type)
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
    serializable = {k: [int(i) for i in v] if isinstance(v, (list, tuple, set)) else v for k, v in splits.items()}
    out.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return out


def load_splits(split_path: Path) -> dict[str, Any]:
    """Carrega um arquivo JSON de splits e devolve `{ "train": [...], "val": [...], "test": [...] }`."""
    path = Path(split_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): [int(i) for i in v] if isinstance(v, list) else v for k, v in data.items()}


def create_group_kfold_splits(
    df: pd.DataFrame,
    split_type: str = "cold_drug",
    n_splits: int = 5,
    val_ratio: float = 0.1,
    assignment_seed: int = 42,
    group_col: str | None = None,
    random_state: int | None = None,
) -> list[dict[str, Any]]:
    """Cria GroupKFold determinístico de K dobras disjuntas agrupadas por conteúdo (smiles ou sequence).

    Garante:
    - Cada grupo único (SMILES ou sequence) cai no teste de exatamente 1 dobra.
    - União de todos os conjuntos de teste cobre 100% dos pares e 100% dos grupos.
    - Validação de cada dobra é extraída dos grupos de treino restantes (~val_ratio),
      garantindo que train, val e test daquela dobra sejam mutuamente disjuntos por conteúdo.
    """
    if random_state is not None:
        assignment_seed = random_state

    if group_col is not None:
        gc = str(group_col).strip().lower()
        if gc in ("smiles", "drug", "drug_id"):
            st = "cold_drug"
            resolved_col = "smiles"
        elif gc in ("sequence", "target", "target_id"):
            st = "cold_target"
            resolved_col = "sequence"
        else:
            st = split_type.strip().lower() if split_type in ("cold_drug", "cold_target") else "cold_drug"
            resolved_col = group_col
    else:
        st = split_type.strip().lower()
        if st not in ("cold_drug", "cold_target"):
            raise ValueError(f"GroupKFold suportado para 'cold_drug' ou 'cold_target', recebido: '{split_type}'.")
        resolved_col = "smiles" if st == "cold_drug" else "sequence"

    if resolved_col not in df.columns:
        raise ValueError(f"Coluna obrigatória '{resolved_col}' ausente no DataFrame.")

    unique_groups = sorted(df[resolved_col].astype(str).unique().tolist())
    if len(unique_groups) < n_splits:
        raise ValueError(f"Quantidade de grupos únicos ({len(unique_groups)}) menor que n_splits ({n_splits}).")

    rng = np.random.RandomState(int(assignment_seed))
    perm = rng.permutation(len(unique_groups))
    shuffled_groups = [unique_groups[i] for i in perm]

    fold_assignments = {g: (idx % n_splits) for idx, g in enumerate(shuffled_groups)}

    col_values = df[resolved_col].astype(str).tolist()

    folds: list[dict[str, Any]] = []
    for k in range(n_splits):
        test_groups = set(g for g, f in fold_assignments.items() if f == k)
        train_candidates = [g for g in shuffled_groups if fold_assignments[g] != k]

        n_val = max(1, int(round(len(train_candidates) * float(val_ratio))))
        val_groups = set(train_candidates[:n_val])
        train_groups = set(train_candidates[n_val:])

        train_indices: list[int] = []
        val_indices: list[int] = []
        test_indices: list[int] = []

        for row_idx, val in enumerate(col_values):
            if val in test_groups:
                test_indices.append(row_idx)
            elif val in val_groups:
                val_indices.append(row_idx)
            elif val in train_groups:
                train_indices.append(row_idx)

        folds.append({
            "train": train_indices,
            "val": val_indices,
            "test": test_indices,
            "fold_id": k,
            "n_splits": n_splits,
            "assignment_seed": assignment_seed,
            "split_type": st,
            "group_col": resolved_col,
            "num_test_groups": len(test_groups),
            "num_val_groups": len(val_groups),
            "num_train_groups": len(train_groups),
        })

    return folds


def verify_group_kfold_coverage(
    df: pd.DataFrame,
    folds: list[dict[str, Any]],
    split_type: str = "cold_drug",
    group_col: str | None = None,
) -> bool:
    """Verifica disjunção, não-vazamento e cobertura de 100% de um esquema GroupKFold."""
    if len(folds) < 2:
        return False

    if group_col is not None:
        gc = str(group_col).strip().lower()
        if gc in ("smiles", "drug", "drug_id"):
            st = "cold_drug"
            resolved_col = "smiles"
        elif gc in ("sequence", "target", "target_id"):
            st = "cold_target"
            resolved_col = "sequence"
        else:
            st = _normalize_split_type(split_type)
            resolved_col = group_col
    else:
        st = _normalize_split_type(split_type)
        resolved_col = "smiles" if st == "cold_drug" else "sequence"

    if resolved_col not in df.columns:
        return False

    n_rows = len(df)
    test_sets = [set(f["test"]) for f in folds]

    # Cada dobra individual deve passar na verificação de vazamento
    for f in folds:
        if not verify_split_leakage(df, f, split_type=st):
            return False

    # Testes devem ser mutuamente disjuntos aos pares
    for i in range(len(folds)):
        for j in range(i + 1, len(folds)):
            if not test_sets[i].isdisjoint(test_sets[j]):
                return False

    # União de todos os testes deve cobrir exatamente todos os pares
    all_tested = set().union(*test_sets)
    if all_tested != set(range(n_rows)):
        return False

    # União de todos os grupos de teste deve cobrir exatamente todos os grupos únicos
    unique_groups = set(df[resolved_col].astype(str))
    all_tested_groups = set()
    for f in folds:
        all_tested_groups.update(df.iloc[f["test"]][resolved_col].astype(str))

    return all_tested_groups == unique_groups


_SEQID_ALIGNER: Any = None
_SEQID_LIST: list[str] = []


def _build_global_aligner() -> Any:
    from Bio.Align import PairwiseAligner, substitution_matrices

    blosum62 = substitution_matrices.load("BLOSUM62")
    return PairwiseAligner(
        mode="global",
        substitution_matrix=blosum62,
        open_gap_score=-10.0,
        extend_gap_score=-0.5,
    )


def _init_seqid_worker(seqs: list[str]) -> None:
    global _SEQID_LIST, _SEQID_ALIGNER
    _SEQID_LIST = seqs
    _SEQID_ALIGNER = _build_global_aligner()


def _align_seqid_pair_worker(pair: tuple[int, int]) -> tuple[int, int, float]:
    i, j = pair
    si = _SEQID_LIST[i]
    sj = _SEQID_LIST[j]
    max_l = len(si) if len(si) >= len(sj) else len(sj)
    aln = _SEQID_ALIGNER.align(si, sj)[0]
    ident = float(aln.counts().identities) / float(max_l)
    return i, j, ident


def compute_global_sequence_identity(seq_a: str, seq_b: str, aligner: Any | None = None) -> float:
    """Calcula identidade global BLOSUM62 definida por `exact_matches / max(len_a, len_b)`."""
    if not seq_a and not seq_b:
        return 1.0
    max_l = max(len(seq_a), len(seq_b))
    if max_l == 0:
        return 0.0
    if seq_a == seq_b:
        return 1.0
    aln_obj = aligner if aligner is not None else _build_global_aligner()
    aln = aln_obj.align(seq_a, seq_b)[0]
    return float(aln.counts().identities) / float(max_l)


def compute_sequence_identity_clusters(
    sequences: list[str] | pd.Series,
    identity_thresholds: tuple[float, ...] = (0.30, 0.40),
    n_workers: int | None = None,
) -> tuple[list[str], dict[float, list[list[int]]], str]:
    """Calcula clusters single-linkage por identidade global BLOSUM62 para sequências únicas.

    Retorna `(uniq_seqs, clusters_by_tau, alignment_desc)`, onde `clusters_by_tau[tau]` é uma lista
    de componentes conexos (listas de índices em `uniq_seqs`).
    """
    import Bio

    thresholds = sorted(float(t) for t in identity_thresholds)
    min_tau = thresholds[0]

    uniq_seqs = sorted(set(str(s) for s in sequences))
    n = len(uniq_seqs)
    lens = [len(s) for s in uniq_seqs]
    aa_counters = [Counter(s) for s in uniq_seqs]

    # Pré-filtro matemático exato: nem razão de comprimentos nem interseção multiset de AA pode ficar abaixo de min_tau
    candidate_pairs: list[tuple[int, int]] = []
    for i in range(n):
        li = lens[i]
        ci = aa_counters[i]
        for j in range(i + 1, n):
            lj = lens[j]
            max_l = li if li >= lj else lj
            min_l = lj if li >= lj else li
            if max_l == 0 or (min_l / max_l) < min_tau:
                continue
            cj = aa_counters[j]
            max_matches = sum(min(cnt, cj[aa]) for aa, cnt in ci.items() if aa in cj)
            if (max_matches / max_l) < min_tau:
                continue
            candidate_pairs.append((i, j))

    pair_identities: list[tuple[int, int, float]] = []
    use_workers = (
        n_workers
        if n_workers is not None
        else (1 if len(candidate_pairs) < 500 else max(1, min(os.cpu_count() or 4, 8)))
    )
    if use_workers <= 1:
        _init_seqid_worker(uniq_seqs)
        pair_identities = [_align_seqid_pair_worker(p) for p in candidate_pairs]
    else:
        with ProcessPoolExecutor(
            max_workers=use_workers,
            initializer=_init_seqid_worker,
            initargs=(uniq_seqs,),
        ) as pool:
            pair_identities = list(pool.map(_align_seqid_pair_worker, candidate_pairs, chunksize=256))

    alignment_desc = (
        f"biopython=={Bio.__version__} PairwiseAligner(mode='global', matrix='BLOSUM62', "
        f"open_gap=-10.0, extend_gap=-0.5)"
    )

    clusters_by_tau: dict[float, list[list[int]]] = {}
    for tau in thresholds:
        adj: list[list[int]] = [[] for _ in range(n)]
        for i, j, ident in pair_identities:
            if ident >= tau:
                adj[i].append(j)
                adj[j].append(i)

        visited = [False] * n
        clusters: list[list[int]] = []
        for i in range(n):
            if visited[i]:
                continue
            comp: list[int] = []
            stack = [i]
            visited[i] = True
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            comp.sort()
            clusters.append(comp)
        clusters_by_tau[tau] = clusters

    return uniq_seqs, clusters_by_tau, alignment_desc


def create_sequence_identity_kfold_splits(
    df: pd.DataFrame,
    identity_thresholds: tuple[float, ...] = (0.30, 0.40),
    n_splits: int = 5,
    group_col: str = "sequence",
    n_workers: int | None = None,
    precomputed_clusters: tuple[list[str], dict[float, list[list[int]]], str] | None = None,
) -> dict[float, list[dict[str, Any]]]:
    """Cria partições K-Fold agrupadas por clusters de identidade global de sequência.

    Especificação auditável:
    - Alinhamento: `Bio.Align.PairwiseAligner(mode='global', matrix='BLOSUM62', open_gap=-10.0, extend_gap=-0.5)`.
    - Identidade: `exact_matches / max(len_a, len_b)`.
    - Clustering: ligação simples (*single-linkage*) via componentes conexos sobre pares com `ident >= tau`.
    - Balanceamento: distribuição gulosa LPT dos clusters (maiores em pares primeiro) em `n_splits` dobras,
      rotacionando `test = dobra k`, `val = dobra (k + 1) % n_splits` e `train = demais dobras`.
    """
    if group_col not in df.columns:
        raise ValueError(f"Coluna obrigatória '{group_col}' ausente no DataFrame.")
    if n_splits < 3:
        raise ValueError(f"n_splits deve ser >= 3 para manter train/val/test disjuntos, recebido: {n_splits}")
    if not identity_thresholds:
        raise ValueError("identity_thresholds não pode estar vazio.")

    thresholds = sorted(float(t) for t in identity_thresholds)
    if precomputed_clusters is not None:
        uniq_seqs, clusters_by_tau, alignment_desc = precomputed_clusters
    else:
        uniq_seqs, clusters_by_tau, alignment_desc = compute_sequence_identity_clusters(
            df[group_col].astype(str).tolist(),
            identity_thresholds=tuple(thresholds),
            n_workers=n_workers,
        )

    seq_row_indices: dict[str, list[int]] = {}
    for row_idx, seq_val in enumerate(df[group_col].astype(str).tolist()):
        seq_row_indices.setdefault(seq_val, []).append(row_idx)

    has_target_id = "target_id" in df.columns
    seq_tid_counts = df.groupby(group_col)["target_id"].nunique().to_dict() if has_target_id else {}

    out_by_threshold: dict[float, list[dict[str, Any]]] = {}
    for tau in thresholds:
        clusters = clusters_by_tau[tau]

        if len(clusters) < n_splits:
            raise ValueError(
                f"Número de clusters ({len(clusters)}) em tau={tau:.2f} é menor que n_splits={n_splits}."
            )

        cluster_infos: list[tuple[int, int, int, int, list[int]]] = []
        for cid, comp in enumerate(clusters):
            seqs = [uniq_seqs[idx] for idx in comp]
            n_pairs = sum(len(seq_row_indices[s]) for s in seqs)
            n_tids = sum(int(seq_tid_counts.get(s, 1)) for s in seqs)
            cluster_infos.append((n_pairs, len(comp), n_tids, cid, comp))

        cluster_infos.sort(key=lambda x: (-x[0], -x[1], x[3]))

        fold_buckets: list[dict[str, Any]] = [
            {"pairs": 0, "unique_seqs": 0, "target_ids": 0, "clusters": 0, "seq_indices": []}
            for _ in range(n_splits)
        ]
        for n_pairs, n_seqs, n_tids, _, comp in cluster_infos:
            best_f = min(
                range(n_splits),
                key=lambda f_i: (fold_buckets[f_i]["pairs"], fold_buckets[f_i]["unique_seqs"], f_i),
            )
            fold_buckets[best_f]["pairs"] += n_pairs
            fold_buckets[best_f]["unique_seqs"] += n_seqs
            fold_buckets[best_f]["target_ids"] += n_tids
            fold_buckets[best_f]["clusters"] += 1
            fold_buckets[best_f]["seq_indices"].extend(comp)

        pct = int(round(tau * 100))
        num_singletons = int(sum(1 for c in clusters if len(c) == 1))
        largest_seqs = int(max(len(c) for c in clusters))
        largest_pairs = int(max(ci[0] for ci in cluster_infos))

        folds: list[dict[str, Any]] = []
        for k in range(n_splits):
            val_f = (k + 1) % n_splits
            train_fs = [f_i for f_i in range(n_splits) if f_i not in (k, val_f)]

            test_seqs = [uniq_seqs[idx] for idx in fold_buckets[k]["seq_indices"]]
            val_seqs = [uniq_seqs[idx] for idx in fold_buckets[val_f]["seq_indices"]]
            train_seqs = [
                uniq_seqs[idx]
                for tf in train_fs
                for idx in fold_buckets[tf]["seq_indices"]
            ]

            test_indices = sorted(r for s in test_seqs for r in seq_row_indices[s])
            val_indices = sorted(r for s in val_seqs for r in seq_row_indices[s])
            train_indices = sorted(r for s in train_seqs for r in seq_row_indices[s])

            folds.append({
                "train": train_indices,
                "val": val_indices,
                "test": test_indices,
                "fold_id": k,
                "n_splits": n_splits,
                "split_type": f"cold_target_seqid{pct}",
                "base_split_type": "cold_target",
                "group_col": group_col,
                "identity_threshold": tau,
                "identity_definition": "exact_matches / max(len_a, len_b)",
                "alignment_tool": alignment_desc,
                "clustering_algorithm": "single_linkage_connected_components + greedy_LPT_balancing",
                "num_clusters_total": len(clusters),
                "singleton_clusters": num_singletons,
                "largest_cluster_unique_seqs": largest_seqs,
                "largest_cluster_pairs": largest_pairs,
                "fold_test_clusters": int(fold_buckets[k]["clusters"]),
                "fold_val_clusters": int(fold_buckets[val_f]["clusters"]),
                "fold_train_clusters": int(sum(fold_buckets[tf]["clusters"] for tf in train_fs)),
                "num_train_groups": len(train_seqs),
                "num_val_groups": len(val_seqs),
                "num_test_groups": len(test_seqs),
                "num_train_target_ids": int(sum(fold_buckets[tf]["target_ids"] for tf in train_fs)),
                "num_val_target_ids": int(fold_buckets[val_f]["target_ids"]),
                "num_test_target_ids": int(fold_buckets[k]["target_ids"]),
            })

        out_by_threshold[tau] = folds

    return out_by_threshold


def verify_sequence_identity_splits(
    df: pd.DataFrame,
    folds: list[dict[str, Any]],
    identity_threshold: float,
    group_col: str = "sequence",
    n_workers: int | None = None,
) -> bool:
    """Verifica cobertura 100%, disjunção de índices/sequências e ausência de pares `>= threshold` entre partições."""
    if not verify_group_kfold_coverage(df, folds, split_type="cold_target", group_col=group_col):
        return False

    uniq_seqs = sorted(df[group_col].astype(str).unique().tolist())
    seq_to_idx = {s: i for i, s in enumerate(uniq_seqs)}
    lens = [len(s) for s in uniq_seqs]
    aa_counters = [Counter(s) for s in uniq_seqs]

    candidate_pairs_set: set[tuple[int, int]] = set()
    for fold in folds:
        tr_idx = [seq_to_idx[s] for s in df.iloc[fold["train"]][group_col].astype(str).unique()]
        va_idx = [seq_to_idx[s] for s in df.iloc[fold["val"]][group_col].astype(str).unique()]
        te_idx = [seq_to_idx[s] for s in df.iloc[fold["test"]][group_col].astype(str).unique()]
        for part_a, part_b in ((tr_idx, va_idx), (tr_idx, te_idx), (va_idx, te_idx)):
            for ia in part_a:
                la = lens[ia]
                ca = aa_counters[ia]
                for ib in part_b:
                    lb = lens[ib]
                    max_l = la if la >= lb else lb
                    min_l = lb if la >= lb else la
                    if max_l == 0 or (min_l / max_l) < identity_threshold:
                        continue
                    cb = aa_counters[ib]
                    max_matches = sum(min(cnt, cb[aa]) for aa, cnt in ca.items() if aa in cb)
                    if (max_matches / max_l) < identity_threshold:
                        continue
                    pair = (ia, ib) if ia < ib else (ib, ia)
                    candidate_pairs_set.add(pair)

    candidate_pairs = sorted(candidate_pairs_set)
    if not candidate_pairs:
        return True

    use_workers = (
        n_workers
        if n_workers is not None
        else (1 if len(candidate_pairs) < 500 else max(1, min(os.cpu_count() or 4, 8)))
    )
    if use_workers <= 1:
        _init_seqid_worker(uniq_seqs)
        for pair in candidate_pairs:
            _, _, ident = _align_seqid_pair_worker(pair)
            if ident >= identity_threshold:
                return False
    else:
        with ProcessPoolExecutor(
            max_workers=use_workers,
            initializer=_init_seqid_worker,
            initargs=(uniq_seqs,),
        ) as pool:
            for _, _, ident in pool.map(_align_seqid_pair_worker, candidate_pairs, chunksize=256):
                if ident >= identity_threshold:
                    return False
    return True

