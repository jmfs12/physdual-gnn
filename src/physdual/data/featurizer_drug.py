"""Featurização molecular (RDKit -> PyG) e cache de grafos compatível com `weights_only=True`.

Implementa a resolução do bloqueador B3:
- Nunca persiste objetos `torch_geometric.data.Data` diretamente em arquivos `.pt`.
- Salva cada grafo como um dicionário puro de tensores (`x`, `edge_index`, `edge_attr`) e
  metadados primitivos (`smiles`, `smiles_raw`, `drug_id`, `num_atoms`), carregável com
  `torch.load(..., weights_only=True)`.
- Reconstrói `torch_geometric.data.Data` sob demanda em `load_drug_graph` e `graph_dict_to_pyg_data`.
- Seleciona o maior fragmento covalente quando o SMILES contém contra-íons/sais, preservando
  `smiles_raw` e registrando falhas em um relatório sem abortar o processamento em lote.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from physdual.data.datasets import canonicalize_largest_fragment

ATOM_FEATURE_DIM = 9
BOND_FEATURE_DIM = 4


def _hybridization_to_float(hyb: Any) -> float:
    """Mapeia o estado de hibridização do RDKit para um índice numérico determinístico."""
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
    """Mapeia a quiralidade atômica do RDKit para um valor numérico."""
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
) -> dict[str, Any]:
    """Converte um SMILES em um dicionário puro de tensores e tipos primitivos (`weights_only=True`).

    Features por átomo (`x`, shape `[N, 9]`, `float32`):
        0: atomic_num
        1: formal_charge
        2: total_degree
        3: explicit_valence
        4: total_num_hs
        5: hybridization
        6: is_aromatic
        7: is_in_ring
        8: chiral_tag

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

    atom_rows: list[list[float]] = []
    for atom in mol.GetAtoms():
        atom_rows.append(
            [
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
        )

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


def smiles_to_pyg_graph(smiles: str, drug_id: str | None = None) -> Any:
    """Converte diretamente um SMILES em um objeto `torch_geometric.data.Data`."""
    g_dict = smiles_to_graph_dict(smiles, drug_id=drug_id, smiles_raw=smiles)
    return graph_dict_to_pyg_data(g_dict)


def build_drug_graph_cache(
    smiles_list: list[str],
    output_path: Path,
    drug_ids: list[str] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Constrói e salva em disco o cache de grafos moleculares como dicionários puros (`weights_only=True`).

    Args:
        smiles_list: Lista de SMILES a featurizar.
        output_path: Caminho do arquivo `.pt` de saída.
        drug_ids: Lista opcional de `drug_id`s alinhada a `smiles_list`.

    Returns:
        Tupla `(saved_path, report)` onde `report` contém:
        `{"num_input", "num_succeeded", "num_failed", "num_modified_fragments", "failures"}`.
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
            g_dict = smiles_to_graph_dict(canon_smi, drug_id=d_id, smiles_raw=raw_smi)
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
        "failures": failures,
    }
    return out, report


def load_drug_graph_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    """Carrega o dicionário de grafos do disco usando estritamente `weights_only=True`."""
    import torch

    path = Path(cache_path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict):
        raise ValueError(f"Formato inválido no cache de grafos {path}: esperado dict, recebido {type(loaded)}")
    return loaded


def load_drug_graph(smiles: str, cache_dict: dict[str, dict[str, Any]]) -> Any:
    """Recupera um grafo do `cache_dict` (por SMILES bruto ou canônico) e reconstrói `torch_geometric.data.Data`."""
    if smiles in cache_dict:
        return graph_dict_to_pyg_data(cache_dict[smiles])

    canon, _, _ = canonicalize_largest_fragment(smiles)
    if canon is not None and canon in cache_dict:
        return graph_dict_to_pyg_data(cache_dict[canon])

    raise KeyError(f"SMILES '{smiles}' (canônico: '{canon}') não encontrado no cache de grafos.")
