"""Featurização molecular (RDKit -> PyG) e cache de grafos compatível com `weights_only=True`.

Implementa a resolução do bloqueador B3 e a emenda à Fase 1 (Task 0 / D2):
- Suporte a featurização atômica fiel ao GraphDTA oficial (78 dimensões, `thinng/GraphDTA`):
  44 símbolos atômicos, 11 graus, 11 H totais, 11 valências implícitas e 1 aromaticidade.
- Normalização L1 por átomo com guarda contra soma zero (`v2_onehot78_l1norm`), fazendo
  cada linha somar 1.0 (reproduzindo `create_data.py:L256`).
- Versão binária sem normalização (`v2_onehot78_raw`, com soma 4.0 ou 5.0) e versão legada
  escalar de 9 dimensões (`v1_int9`) mantidas para estudos de ablação.
- Cache versionado em disco (`{dataset}_drug_graphs_{featurizer_version}.pt`) com validação
  estrita de versão, dimensão e normalização ao carregar via `weights_only=True`.
- Nunca persiste objetos `torch_geometric.data.Data` diretamente em arquivos `.pt`.
- Reconstrói `torch_geometric.data.Data` sob demanda em `load_drug_graph` e `graph_dict_to_pyg_data`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from physdual.data.datasets import canonicalize_largest_fragment

FeaturizerVersion = Literal["v2_onehot78_l1norm", "v2_onehot78_raw", "v1_int9"]

# 44 símbolos atômicos exatos do repositório oficial GraphDTA (thinng/GraphDTA, create_data.py)
GRAPHDTA_ATOM_SYMBOLS: list[str] = [
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg",
    "Na", "Ca", "Fe", "As", "Al", "I", "B", "V", "K", "Tl",
    "Yb", "Sb", "Sn", "Ag", "Pd", "Co", "Se", "Ti", "Zn", "H",
    "Li", "Ge", "Cu", "Au", "Ni", "Cd", "In", "Mn", "Zr", "Cr",
    "Pt", "Hg", "Pb", "Unknown",
]

ATOM_FEATURE_DIM_V1 = 9
ATOM_FEATURE_DIM_V2 = 78
BOND_FEATURE_DIM = 4


def get_drug_graph_cache_filename(dataset_name: str, featurizer_version: str) -> str:
    """Retorna o nome padronizado do arquivo de cache versionado de grafos de fármacos."""
    return f"{dataset_name}_drug_graphs_{featurizer_version}.pt"


def _one_of_k_encoding(x: Any, allowable_set: list[Any]) -> list[float]:
    """Codificação one-hot estrita que lança exceção se x não pertencer ao allowable_set."""
    if x not in allowable_set:
        raise ValueError(f"Entrada {x} não está no conjunto permitido {allowable_set}")
    return [1.0 if x == s else 0.0 for s in allowable_set]


def _one_of_k_encoding_unk(x: Any, allowable_set: list[Any]) -> list[float]:
    """Codificação one-hot mapeando entradas desconhecidas para o último elemento."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [1.0 if x == s else 0.0 for s in allowable_set]


def _atom_features_graphdta_78d(atom: Any) -> list[float]:
    """Calcula o vetor de 78 dimensões one-hot exato de thinng/GraphDTA (create_data.py:L230-L256).

    Componentes:
        - Símbolo atômico (44 categorias, último='Unknown') -> 44 dims
        - Grau de ligação (0..10, 11 categorias) -> 11 dims
        - Número total de H (0..10 com unk) -> 11 dims
        - Valência implícita (0..10 com unk) -> 11 dims
        - Aromaticidade (binário 0.0 ou 1.0) -> 1 dim
    """
    sym = atom.GetSymbol()
    deg = atom.GetDegree()
    deg_set = list(range(11))
    deg_val = deg if deg in deg_set else deg_set[-1]

    tot_hs = atom.GetTotalNumHs()
    hs_set = list(range(11))

    imp_val = atom.GetImplicitValence()
    val_set = list(range(11))

    arom = 1.0 if atom.GetIsAromatic() else 0.0

    return (
        _one_of_k_encoding_unk(sym, GRAPHDTA_ATOM_SYMBOLS)
        + _one_of_k_encoding(deg_val, deg_set)
        + _one_of_k_encoding_unk(tot_hs, hs_set)
        + _one_of_k_encoding_unk(imp_val, val_set)
        + [arom]
    )


def _hybridization_to_float(hyb: Any) -> float:
    """Mapeia o estado de hibridização do RDKit para um índice numérico determinístico (legado v1)."""
    name = str(hyb).upper()
    mapping = {
        "S": 1.0,
        "SP": 2.0,
        "SP2": 3.0,
        "SP3": 4.0,
        "SP3D": 5.0,
        "SP3D2": 6.0,
    }
    for key, val in mapping.items():
        if name.endswith(key):
            return val
    return 0.0


def _chiral_tag_to_float(chiral_tag: Any) -> float:
    """Mapeia a quiralidade atômica do RDKit para um valor numérico (legado v1)."""
    name = str(chiral_tag).upper()
    if "CW" in name and "CCW" not in name:
        return 1.0
    if "CCW" in name:
        return 2.0
    if "UNSPECIFIED" not in name and "NONE" not in name:
        return 3.0
    return 0.0


def _bond_type_to_float(bond_type: Any) -> float:
    """Mapeia o tipo de ligação covalente do RDKit para um valor numérico."""
    name = str(bond_type).upper()
    if "SINGLE" in name:
        return 1.0
    if "DOUBLE" in name:
        return 2.0
    if "TRIPLE" in name:
        return 3.0
    if "AROMATIC" in name:
        return 1.5
    return 0.0


def _bond_stereo_to_float(stereo: Any) -> float:
    """Mapeia a estereoquímica de ligação do RDKit para um valor numérico."""
    name = str(stereo).upper()
    if "Z" in name or "CIS" in name:
        return 1.0
    if "E" in name or "TRANS" in name:
        return 2.0
    if "ANY" in name:
        return 3.0
    return 0.0


def smiles_to_graph_dict(
    smiles: str,
    drug_id: str | None = None,
    smiles_raw: str | None = None,
    featurizer_version: FeaturizerVersion = "v2_onehot78_l1norm",
    normalize_atom_features: bool | None = None,
) -> dict[str, Any]:
    """Converte um SMILES em um dicionário puro de tensores e tipos primitivos (`weights_only=True`).

    Suporta:
        - `v2_onehot78_l1norm`: 78 dims GraphDTA com normalização L1 por átomo (soma = 1.0).
        - `v2_onehot78_raw`: 78 dims GraphDTA binário sem normalização (soma = 4.0 ou 5.0).
        - `v1_int9`: 9 dims escalares inteiras legadas da Fase 1.

    Features por aresta dirigida (`edge_attr`, shape `[E, 4]`, `float32`):
        0: bond_type
        1: stereo
        2: is_conjugated
        3: is_in_ring
    """
    import torch
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    raw_input = smiles_raw if smiles_raw is not None else smiles
    canon_smiles, _, err = canonicalize_largest_fragment(smiles)
    if canon_smiles is None:
        raise ValueError(f"SMILES inválido para featurização: '{smiles}' ({err})")

    mol = Chem.MolFromSmiles(canon_smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Grafo molecular vazio ou inválido para SMILES: '{canon_smiles}'")

    if normalize_atom_features is not None:
        norm = bool(normalize_atom_features)
    else:
        norm = (featurizer_version == "v2_onehot78_l1norm")

    atom_rows: list[list[float]] = []
    is_v2 = featurizer_version in ("v2_onehot78_l1norm", "v2_onehot78_raw")

    for atom in mol.GetAtoms():
        if is_v2:
            feat = _atom_features_graphdta_78d(atom)
        else:
            feat = [
                float(atom.GetAtomicNum()),
                float(atom.GetFormalCharge()),
                float(atom.GetTotalDegree()),
                float(atom.GetExplicitValence()),
                float(atom.GetTotalNumHs()),
                _hybridization_to_float(atom.GetHybridization()),
                1.0 if atom.GetIsAromatic() else 0.0,
                1.0 if atom.IsInRing() else 0.0,
                _chiral_tag_to_float(atom.GetChiralTag()),
            ]

        if norm:
            row_sum = float(sum(feat))
            if row_sum > 0.0:
                feat = [v / row_sum for v in feat]

        atom_rows.append(feat)

    edge_indices: list[tuple[int, int]] = []
    edge_attrs: list[list[float]] = []
    for bond in mol.GetBonds():
        u = int(bond.GetBeginAtomIdx())
        v = int(bond.GetEndAtomIdx())
        b_feat = [
            _bond_type_to_float(bond.GetBondType()),
            _bond_stereo_to_float(bond.GetStereo()),
            1.0 if bond.GetIsConjugated() else 0.0,
            1.0 if bond.IsInRing() else 0.0,
        ]
        edge_indices.append((u, v))
        edge_attrs.append(b_feat)
        edge_indices.append((v, u))
        edge_attrs.append(b_feat)

    x = torch.tensor(atom_rows, dtype=torch.float32)
    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, BOND_FEATURE_DIM), dtype=torch.float32)

    return {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "smiles": str(canon_smiles),
        "smiles_raw": str(raw_input),
        "drug_id": str(drug_id) if drug_id is not None else "",
        "num_atoms": int(x.size(0)),
        "featurizer_version": str(featurizer_version),
        "atom_feature_dim": int(x.size(1)),
        "normalize_atom_features": bool(norm),
    }


def graph_dict_to_pyg_data(graph_dict: dict[str, Any]) -> Any:
    """Reconstrói um objeto `torch_geometric.data.Data` a partir de um `graph_dict`."""
    from torch_geometric.data import Data

    return Data(
        x=graph_dict["x"],
        edge_index=graph_dict["edge_index"],
        edge_attr=graph_dict["edge_attr"],
        smiles=graph_dict.get("smiles", ""),
        smiles_raw=graph_dict.get("smiles_raw", ""),
        drug_id=graph_dict.get("drug_id", ""),
        num_atoms=int(graph_dict.get("num_atoms", graph_dict["x"].size(0))),
    )


def smiles_to_pyg_graph(
    smiles: str,
    drug_id: str | None = None,
    featurizer_version: FeaturizerVersion = "v2_onehot78_l1norm",
    normalize_atom_features: bool | None = None,
) -> Any:
    """Converte diretamente um SMILES em um objeto `torch_geometric.data.Data`."""
    g_dict = smiles_to_graph_dict(
        smiles,
        drug_id=drug_id,
        smiles_raw=smiles,
        featurizer_version=featurizer_version,
        normalize_atom_features=normalize_atom_features,
    )
    return graph_dict_to_pyg_data(g_dict)


def build_drug_graph_cache(
    smiles_list: list[str],
    output_path: Path,
    drug_ids: list[str] | None = None,
    featurizer_version: FeaturizerVersion = "v2_onehot78_l1norm",
    normalize_atom_features: bool | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Constrói e salva em disco o cache versionado de grafos moleculares como dicionários puros (`weights_only=True`).

    Args:
        smiles_list: Lista de SMILES a featurizar.
        output_path: Caminho do arquivo `.pt` de saída.
        drug_ids: Lista opcional de `drug_id`s alinhada a `smiles_list`.
        featurizer_version: Versão da featurização ("v2_onehot78_l1norm", "v2_onehot78_raw", "v1_int9").
        normalize_atom_features: Se aplica normalização L1 por átomo (default infere da versão).

    Returns:
        Tupla `(saved_path, report)` onde `report` contém:
        `{"num_input", "num_succeeded", "num_failed", "num_modified_fragments", "failures", "featurizer_version"}`.
    """
    import torch

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    cache_dict: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []
    num_succeeded = 0
    num_modified_fragments = 0

    for idx, raw_smi in enumerate(smiles_list):
        d_id = drug_ids[idx] if (drug_ids is not None and idx < len(drug_ids)) else f"DRUG_{idx}"
        canon_smi, was_mod, err = canonicalize_largest_fragment(raw_smi)
        if canon_smi is None:
            failures.append({"drug_id": str(d_id), "smiles_raw": str(raw_smi), "error": err or "Inválido"})
            continue

        try:
            g_dict = smiles_to_graph_dict(
                canon_smi,
                drug_id=d_id,
                smiles_raw=raw_smi,
                featurizer_version=featurizer_version,
                normalize_atom_features=normalize_atom_features,
            )
            cache_dict[canon_smi] = g_dict
            if raw_smi != canon_smi:
                cache_dict[raw_smi] = g_dict
            num_succeeded += 1
            if was_mod:
                num_modified_fragments += 1
        except Exception as exc:
            failures.append({"drug_id": str(d_id), "smiles_raw": str(raw_smi), "error": str(exc)})

    torch.save(cache_dict, out)
    report: dict[str, Any] = {
        "num_input": len(smiles_list),
        "num_succeeded": num_succeeded,
        "num_failed": len(failures),
        "num_modified_fragments": num_modified_fragments,
        "featurizer_version": featurizer_version,
        "failures": failures,
    }
    return out, report


def load_drug_graph_cache(
    cache_path: Path,
    expected_version: str = "v2_onehot78_l1norm",
    expected_dim: int = 78,
    expected_normalize: bool = True,
) -> dict[str, dict[str, Any]]:
    """Carrega o dicionário de grafos do disco usando `weights_only=True` e valida metadados do cache.

    Lança `ValueError` se `featurizer_version`, `atom_feature_dim` ou `normalize_atom_features`
    divergirem do esperado.
    """
    import torch

    path = Path(cache_path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict):
        raise ValueError(f"Formato inválido no cache de grafos {path}: esperado dict, recebido {type(loaded)}")

    for key, g_dict in loaded.items():
        if not isinstance(g_dict, dict):
            continue

        actual_version = g_dict.get("featurizer_version")
        if actual_version is not None and actual_version != expected_version:
            raise ValueError(
                f"Incompatibilidade no cache {path} para '{key}': "
                f"featurizer_version='{actual_version}', esperado='{expected_version}'"
            )

        actual_norm = g_dict.get("normalize_atom_features")
        if actual_norm is not None and actual_norm != expected_normalize:
            raise ValueError(
                f"Incompatibilidade no cache {path} para '{key}': "
                f"normalize_atom_features={actual_norm}, esperado={expected_normalize}"
            )

        x = g_dict.get("x")
        actual_dim = g_dict.get("atom_feature_dim", x.size(1) if x is not None else None)
        if actual_dim is not None and actual_dim != expected_dim:
            raise ValueError(
                f"Incompatibilidade no cache {path} para '{key}': "
                f"atom_feature_dim={actual_dim}, esperado={expected_dim}"
            )

    return loaded


def load_drug_graph(smiles: str, cache_dict: dict[str, dict[str, Any]]) -> Any:
    """Recupera um grafo do `cache_dict` (por SMILES bruto ou canônico) e reconstrói `torch_geometric.data.Data`."""
    if smiles in cache_dict:
        return graph_dict_to_pyg_data(cache_dict[smiles])

    canon, _, _ = canonicalize_largest_fragment(smiles)
    if canon is not None and canon in cache_dict:
        return graph_dict_to_pyg_data(cache_dict[canon])

    raise KeyError(f"SMILES '{smiles}' (canônico: '{canon}') não encontrado no cache de grafos.")
