"""Testes unitários de featurização molecular e cache de grafos compatível com weights_only=True."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rdkit")
pytest.importorskip("torch_geometric")

from physdual.data.featurizer_drug import (  # noqa: E402
    build_drug_graph_cache,
    get_drug_graph_cache_filename,
    load_drug_graph,
    load_drug_graph_cache,
    smiles_to_graph_dict,
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


def test_smiles_to_graph_dict_v2_onehot78_l1norm_and_raw_aspirin():
    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    # Com normalização L1 (padrão oficial v2_onehot78_l1norm): linhas somam 1.0
    g_l1 = smiles_to_graph_dict(
        aspirin,
        drug_id="ASP",
        featurizer_version="v2_onehot78_l1norm",
        normalize_atom_features=True,
    )
    assert g_l1["x"].shape == (13, 78)
    assert g_l1["edge_index"].shape == (2, 26)
    assert g_l1["edge_attr"].shape == (26, 4)
    assert g_l1["featurizer_version"] == "v2_onehot78_l1norm"
    assert g_l1["atom_feature_dim"] == 78
    assert g_l1["normalize_atom_features"] is True
    row_sums_l1 = g_l1["x"].sum(dim=1)
    assert torch.allclose(row_sums_l1, torch.ones_like(row_sums_l1), atol=1e-5)

    # Sem normalização L1 (v2_onehot78_raw): linhas somam 4.0 (não-aromático) ou 5.0 (aromático)
    g_raw = smiles_to_graph_dict(
        aspirin,
        drug_id="ASP",
        featurizer_version="v2_onehot78_raw",
        normalize_atom_features=False,
    )
    assert g_raw["x"].shape == (13, 78)
    assert g_raw["normalize_atom_features"] is False
    row_sums_raw = g_raw["x"].sum(dim=1)
    assert all(val in (4.0, 5.0) for val in row_sums_raw.tolist())

    # Versão legada v1_int9 preservada para ablação
    g_v1 = smiles_to_graph_dict(aspirin, drug_id="ASP", featurizer_version="v1_int9", normalize_atom_features=False)
    assert g_v1["x"].shape == (13, 9)
    assert g_v1["featurizer_version"] == "v1_int9"


def test_get_drug_graph_cache_filename():
    assert get_drug_graph_cache_filename("davis", "v2_onehot78_l1norm") == "davis_drug_graphs_v2_onehot78_l1norm.pt"
    assert get_drug_graph_cache_filename("kiba", "v1_int9") == "kiba_drug_graphs_v1_int9.pt"


def test_versioned_cache_rejects_mismatched_version_dim_or_norm_flag(tmp_path):
    out_path = tmp_path / "davis_drug_graphs_v2_onehot78_raw.pt"
    build_drug_graph_cache(
        ["CCO"],
        out_path,
        drug_ids=["D1"],
        featurizer_version="v2_onehot78_raw",
        normalize_atom_features=False,
    )

    with pytest.raises(ValueError, match="featurizer_version|normalize_atom_features|dim"):
        load_drug_graph_cache(
            out_path,
            expected_version="v2_onehot78_l1norm",
            expected_dim=78,
            expected_normalize=True,
        )



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
