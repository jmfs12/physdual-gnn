"""Testes unitários de featurização molecular e cache de grafos compatível com weights_only=True."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rdkit")
pytest.importorskip("torch_geometric")

from physdual.data.featurizer_drug import (  # noqa: E402
    build_drug_graph_cache,
    load_drug_graph,
    load_drug_graph_cache,
    smiles_to_pyg_graph,
)


def test_aspirin_graph_features():
    aspirin = "CC(=O)OC1=CC=CC=C1C(=O)O"
    g = smiles_to_pyg_graph(aspirin, drug_id="aspirin")
    assert g.x.size(0) == 13
    assert g.x.size(1) >= 9
    assert g.edge_index.size(0) == 2
    assert g.edge_index.size(1) == 26
    assert g.edge_attr is not None
    assert not torch.isnan(g.x).any()


def test_drug_graph_cache_roundtrip_weights_only(tmp_path):
    smiles_list = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",       # aspirina
        "CC(=O)NC1=CC=C(O)C=C1.[Cl-]",   # paracetamol com contra-íon (reduzido ao maior fragmento)
        "INVALID_SMILES_STRING",          # inválido: deve ir para o relatório de falhas sem quebrar o lote
    ]
    cache_file = tmp_path / "drug_graphs.pt"
    saved_path, report = build_drug_graph_cache(smiles_list, output_path=cache_file)

    assert saved_path.exists()
    assert report["num_succeeded"] == 2
    assert report["num_failed"] == 1
    assert report["num_modified_fragments"] == 1

    # Verifica carregamento estrito com weights_only=True diretamente via torch.load
    raw_cache = torch.load(saved_path, weights_only=True)
    assert isinstance(raw_cache, dict)

    loaded_cache = load_drug_graph_cache(saved_path)
    g_orig = smiles_to_pyg_graph("CC(=O)OC1=CC=CC=C1C(=O)O")
    g_loaded = load_drug_graph("CC(=O)OC1=CC=CC=C1C(=O)O", loaded_cache)

    assert torch.equal(g_orig.x, g_loaded.x)
    assert torch.equal(g_orig.edge_index, g_loaded.edge_index)
    assert torch.equal(g_orig.edge_attr, g_loaded.edge_attr)
