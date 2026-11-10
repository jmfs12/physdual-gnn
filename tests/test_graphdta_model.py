import inspect
from pathlib import Path
from typing import Any
import pandas as pd
import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from physdual.data.datamodule import DTADataset, dta_collate_fn
from physdual.data.featurizer_drug import smiles_to_graph_dict
from physdual.models.graphdta import (
    DeepDTAStyleCNN3TargetEncoder,
    GINEncoder,
    GraphDTACNN3,
    GraphDTAGIN,
    OfficialGraphDTATargetEncoder,
    build_baseline_model,
)


def test_drug_in_dim_has_no_magic_default():
    sig_gin = inspect.signature(GraphDTAGIN.__init__)
    sig_cnn3 = inspect.signature(GraphDTACNN3.__init__)
    sig_encoder = inspect.signature(GINEncoder.__init__)

    assert sig_gin.parameters["drug_in_dim"].default is inspect.Parameter.empty
    assert sig_cnn3.parameters["drug_in_dim"].default is inspect.Parameter.empty
    assert sig_encoder.parameters["drug_in_dim"].default is inspect.Parameter.empty


def test_target_encoders_output_shape():
    tokens = torch.randint(0, 26, (4, 1000))
    official_enc = OfficialGraphDTATargetEncoder(protein_vocab_size=25, max_seq_len=1000, embedding_dim=128, output_dim=128)
    out_official = official_enc(tokens)
    assert out_official.shape == (4, 128)

    cnn3_enc = DeepDTAStyleCNN3TargetEncoder(protein_vocab_size=25, embedding_dim=128, output_dim=128)
    out_cnn3 = cnn3_enc(tokens)
    assert out_cnn3.shape == (4, 128)


@pytest.mark.parametrize("model_cls", [GraphDTAGIN, GraphDTACNN3])
def test_both_models_forward_backward_and_small_batch_overfit(model_cls):
    torch.manual_seed(42)
    cache = {
        "CCO": smiles_to_graph_dict("CCO", drug_id="D1", featurizer_version="v2_onehot78_l1norm"),
        "CCN": smiles_to_graph_dict("CCN", drug_id="D2", featurizer_version="v2_onehot78_l1norm"),
    }
    df = pd.DataFrame([
        {"drug_id": "D1", "smiles": "CCO", "target_id": "T1", "sequence": "MKTVRQ", "affinity": 7.0},
        {"drug_id": "D2", "smiles": "CCN", "target_id": "T2", "sequence": "ACDEFG", "affinity": 9.0},
    ])
    ds = DTADataset(df, drug_graph_cache=cache, max_seq_len=1000)
    batch = dta_collate_fn([ds[0], ds[1]])

    model = model_cls(drug_in_dim=78, max_seq_len=1000)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    initial_loss = None
    final_loss = None
    for step in range(25):
        optimizer.zero_grad()
        preds = model(batch["drug_batch"], batch["target_tokens"])
        assert preds.shape == (2,)
        loss = torch.nn.functional.mse_loss(preds, batch["affinity"])
        if step == 0:
            initial_loss = loss.item()
            loss.backward()
            assert any(p.grad is not None and p.grad.norm().item() > 0 for p in model.drug_encoder.parameters())
            assert any(p.grad is not None and p.grad.norm().item() > 0 for p in model.target_encoder.parameters())
            optimizer.step()
        else:
            loss.backward()
            optimizer.step()
        final_loss = loss.item()

    assert final_loss < initial_loss * 0.2, f"Overfit falhou: {initial_loss} -> {final_loss}"


def test_build_baseline_model_factory():
    config_path = Path("configs/base.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_gin = build_baseline_model("graphdta_gin", config)
    assert isinstance(model_gin, GraphDTAGIN)
    assert model_gin.drug_encoder.drug_in_dim == 78

    model_cnn3 = build_baseline_model("graphdta_cnn3", config)
    assert isinstance(model_cnn3, GraphDTACNN3)
    assert model_cnn3.drug_encoder.drug_in_dim == 78

    with pytest.raises(ValueError, match="Modelo desconhecido"):
        build_baseline_model("unknown_model", config)

    with pytest.raises(ValueError, match="drug_in_dim"):
        build_baseline_model("graphdta_gin", {"graphdta": {}})
