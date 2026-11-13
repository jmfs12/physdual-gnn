"""Testes unitários para a arquitetura Fase 3 (`PhysDualPhase3`, `ESMTargetEncoder`),
carregamento de cache ESM-2 no `DTADataset` e otimizador `AdamW` + diagnóstico de memorização.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch.utils.data import DataLoader

from physdual.data.datamodule import (
    DTADataset,
    dta_collate_fn,
    get_dta_dataloaders,
    load_esm_mean_pool_dict,
)
from physdual.data.esm_cache import save_protein_embedding
from physdual.data.featurizer_drug import smiles_to_graph_dict
from physdual.models.physdual import (
    ESMTargetEncoder,
    PhysDualPhase3,
    build_physdual_phase3_model,
)
from physdual.training.trainer import train_model
from physdual.utils.config import load_config


def test_physdual_phase3_requires_explicit_dims():
    """Garante ausência de valores mágicos para `drug_in_dim` e `esm_dim`."""
    sig_enc = inspect.signature(ESMTargetEncoder.__init__)
    sig_mod = inspect.signature(PhysDualPhase3.__init__)
    assert sig_enc.parameters["esm_dim"].default is inspect.Parameter.empty
    assert sig_mod.parameters["drug_in_dim"].default is inspect.Parameter.empty
    assert sig_mod.parameters["esm_dim"].default is inspect.Parameter.empty


def test_esm_target_encoder_2d_and_masked_3d():
    """Valida projeção sobre vetor 2D [B, 1280] e pooling mascarado sobre 3D [B, L, 1280]."""
    torch.manual_seed(7)
    enc = ESMTargetEncoder(esm_dim=16, output_dim=8, dropout_in=0.0, dropout_out=0.0)
    enc.eval()

    x_2d = torch.randn(3, 16)
    out_2d = enc(x_2d)
    assert out_2d.shape == (3, 8)
    assert torch.isfinite(out_2d).all()

    # 3D com padding mascarado: posição mascarada extrema não deve contaminar a saída
    seq_a = torch.ones(1, 3, 16)
    seq_b = torch.cat([torch.ones(1, 3, 16), torch.full((1, 2, 16), 1e4)], dim=1)
    mask_a = torch.tensor([[1, 1, 1]], dtype=torch.bool)
    mask_b = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)

    out_a = enc(seq_a, esm_mask=mask_a)
    out_b = enc(seq_b, esm_mask=mask_b)
    assert out_a.shape == (1, 8)
    assert torch.allclose(out_a, out_b, atol=1e-5)


def test_physdual_phase3_forward_backward_and_small_batch_overfit():
    """Valida forward, backward e capacidade de memorizar mini-lote de 2 amostras com AdamW."""
    torch.manual_seed(42)
    drug_cache = {
        "CCO": smiles_to_graph_dict("CCO", drug_id="D1", featurizer_version="v2_onehot78_l1norm"),
        "CCN": smiles_to_graph_dict("CCN", drug_id="D2", featurizer_version="v2_onehot78_l1norm"),
    }
    esm_cache = {
        "MKTVRQ": torch.randn(32),
        "ACDEFG": torch.randn(32),
    }
    df = pd.DataFrame([
        {"drug_id": "D1", "smiles": "CCO", "target_id": "T1", "sequence": "MKTVRQ", "affinity": 5.2},
        {"drug_id": "D2", "smiles": "CCN", "target_id": "T2", "sequence": "ACDEFG", "affinity": 9.1},
    ])

    ds = DTADataset(df, drug_graph_cache=drug_cache, esm_cache=esm_cache, esm_dim=32)
    batch = dta_collate_fn([ds[0], ds[1]])
    assert "target_esm" in batch
    assert batch["target_esm"].shape == (2, 32)

    model = PhysDualPhase3(
        drug_in_dim=78,
        esm_dim=32,
        prot_output_dim=16,
        esm_dropout_in=0.0,
        prot_dropout=0.0,
        mlp_hidden_dims=(64, 32),
        mlp_dropout=0.0,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-3)
    loss_fn = torch.nn.MSELoss()

    losses = []
    for _ in range(35):
        opt.zero_grad()
        preds = model(batch["drug_batch"], batch["target_esm"])
        assert preds.shape == (2,)
        loss = loss_fn(preds, batch["affinity"])
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    assert losses[-1] < losses[0] * 0.25


def test_build_physdual_phase3_model_from_config():
    """Valida instanciação do PhysDualPhase3 a partir de `configs/base.yaml`."""
    cfg = load_config()
    model = build_physdual_phase3_model(cfg)
    assert isinstance(model, PhysDualPhase3)
    assert model.drug_in_dim == 78
    assert model.esm_dim == 1280


def test_datamodule_esm_cache_loading_and_get_dataloaders(tmp_path: Path):
    """Testa leitura de `.pt` fp16 no disco e integração com `get_dta_dataloaders(..., use_esm=True)`."""
    proc_dir = tmp_path / "data" / "processed"
    splits_dir = proc_dir / "splits"
    esm_dir = tmp_path / "cache" / "esm_embeddings"
    splits_dir.mkdir(parents=True)

    seqs = ["MKTVRQ", "ACDEFG", "GHIKLM"]
    smiles = ["CCO", "CCN", "CCC"]
    rows = []
    for di, smi in enumerate(smiles):
        for ti, seq in enumerate(seqs):
            rows.append({
                "drug_id": f"D{di}",
                "smiles": smi,
                "target_id": f"T{ti}",
                "sequence": seq,
                "affinity": float(5.0 + di + ti * 0.5),
            })
    df = pd.DataFrame(rows)
    df.to_csv(proc_dir / "davis_clean.csv", index=False)

    drug_cache = {
        smi: smiles_to_graph_dict(smi, drug_id=f"D{i}", featurizer_version="v2_onehot78_l1norm")
        for i, smi in enumerate(smiles)
    }
    torch.save(
        drug_cache,
        proc_dir / "davis_drug_graphs_v2_onehot78_l1norm.pt",
    )

    for idx, seq in enumerate(seqs):
        emb = torch.randn(len(seq), 16, dtype=torch.float32) + idx
        save_protein_embedding(seq, emb, model_name="esm2_test", cache_dir=esm_dir, windowed=False)

    split_payload = {
        "train": [0, 1, 3, 4],
        "val": [2, 5],
        "test": [6, 7, 8],
    }
    (splits_dir / "davis_cold_target_seqid30_fold0.json").write_text(json.dumps(split_payload), encoding="utf-8")

    # Teste direto de load_esm_mean_pool_dict
    pooled = load_esm_mean_pool_dict(seqs, model_name="esm2_test", cache_dir=esm_dir, expected_dim=16)
    assert set(pooled.keys()) == set(seqs)
    assert pooled["MKTVRQ"].dtype == torch.float32
    assert pooled["MKTVRQ"].shape == (16,)

    cfg = load_config()
    cfg["project_root"] = str(tmp_path)
    cfg["esm"]["model_name"] = "esm2_test"
    cfg["esm"]["esm_dim"] = 16
    cfg["physdual_phase3"]["esm_dim"] = 16
    cfg["physdual_phase3"]["training"]["batch_size"] = 2
    cfg["physdual_phase3"]["training"]["num_workers"] = 0
    cfg["physdual_phase3"]["training"]["pin_memory"] = False

    tr_loader, va_loader, te_loader = get_dta_dataloaders(
        dataset_name="davis",
        split_type="cold_target_seqid30",
        protocol="kfold",
        fold=0,
        config=cfg,
        root=tmp_path,
        use_esm=True,
        esm_cache_dir=esm_dir,
    )
    batch = next(iter(tr_loader))
    assert "target_esm" in batch
    assert batch["target_esm"].shape[1] == 16
    assert batch["target_esm"].dtype == torch.float32
    assert torch.isfinite(batch["target_esm"]).all()
    assert len(va_loader.dataset) == 2
    assert len(te_loader.dataset) == 3


def test_trainer_adamw_and_best_epoch_gap_diagnostics(tmp_path: Path):
    """Valida otimizador AdamW, `train_val_mse_gap_at_best` e alerta `early_memorization_warn`."""
    torch.manual_seed(0)
    drug_cache = {
        "CCO": smiles_to_graph_dict("CCO", drug_id="D1", featurizer_version="v2_onehot78_l1norm"),
        "CCN": smiles_to_graph_dict("CCN", drug_id="D2", featurizer_version="v2_onehot78_l1norm"),
    }
    esm_cache = {"MKTVRQ": torch.randn(16), "ACDEFG": torch.randn(16)}
    df = pd.DataFrame([
        {"drug_id": "D1", "smiles": "CCO", "target_id": "T1", "sequence": "MKTVRQ", "affinity": 5.0},
        {"drug_id": "D2", "smiles": "CCN", "target_id": "T2", "sequence": "ACDEFG", "affinity": 8.0},
    ])
    ds = DTADataset(df, drug_graph_cache=drug_cache, esm_cache=esm_cache, esm_dim=16)
    loader = DataLoader(ds, batch_size=2, collate_fn=dta_collate_fn)

    model = PhysDualPhase3(drug_in_dim=78, esm_dim=16, prot_output_dim=8, mlp_hidden_dims=(16, 8))
    res = train_model(
        model=model,
        train_loader=loader,
        val_loader=loader,
        test_loader=loader,
        epochs=3,
        patience=3,
        lr=1e-3,
        weight_decay=1e-3,
        optimizer_name="adamw",
        checkpoint_dir=tmp_path / "ckpt",
        dataset_name="davis",
        split_type="cold_target_seqid30",
        resume=False,
    )
    assert res["optimizer_name"] == "adamw"
    assert "train_loss_at_best" in res
    assert "train_val_mse_gap_at_best" in res
    assert isinstance(res["train_val_mse_gap_at_best"], float)
    assert res["early_memorization_warn"] is True  # best_epoch <= 3 < 15 em cold_target
