"""Testes unitários para o módulo datamodule (Task 2 da Fase 2).

Cobre:
- Vocabulário e alfabeto proteico (PROTEIN_ALPHABET, PROTEIN_CHAR_TO_IDX).
- Tokenização e padding de sequências de aminoácidos (tokenize_protein_sequence).
- Composição de aminoácidos (compute_aa_composition).
- DTADataset com pré-construção em memória e lookup O(1).
- dta_collate_fn com empacotamento PyG Batch.
- get_dta_dataloaders e descoberta de cache de grafos com fallback.
- prepare_local_cache para descompactação de tarball no SSD local.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from physdual.data.datamodule import (
    PROTEIN_ALPHABET,
    PROTEIN_CHAR_TO_IDX,
    DTADataset,
    compute_aa_composition,
    dta_collate_fn,
    get_dta_dataloaders,
    prepare_local_cache,
    tokenize_protein_sequence,
)
from physdual.data.featurizer_drug import smiles_to_graph_dict


def test_protein_alphabet_and_char_to_idx():
    """Valida o alfabeto de 25 caracteres IUPAC (sem 'J') e mapeamento 1..25."""
    assert len(PROTEIN_ALPHABET) == 25
    assert "J" not in PROTEIN_ALPHABET
    assert len(PROTEIN_CHAR_TO_IDX) == 25
    assert min(PROTEIN_CHAR_TO_IDX.values()) == 1
    assert max(PROTEIN_CHAR_TO_IDX.values()) == 25
    for idx, char in enumerate(PROTEIN_ALPHABET, start=1):
        assert PROTEIN_CHAR_TO_IDX[char] == idx


def test_tokenize_protein_sequence_length_and_padding():
    """Valida tamanho exato, padding à direita com zeros e mapeamento de caracteres."""
    seq = "MKTVRQ"
    tokens = tokenize_protein_sequence(seq, max_len=10)
    assert len(tokens) == 10
    assert tokens.dtype == np.int64
    assert (tokens[:6] > 0).all()
    assert (tokens[6:] == 0).all()

    # Truncamento quando sequência excede max_len
    tokens_trunc = tokenize_protein_sequence("MKTVRQ", max_len=4)
    assert len(tokens_trunc) == 4
    assert np.array_equal(tokens_trunc, tokens[:4])

    # Sequência vazia
    tokens_empty = tokenize_protein_sequence("", max_len=8)
    assert len(tokens_empty) == 8
    assert (tokens_empty == 0).all()

    # Caracteres minúsculos e fora do vocabulário ('J' e símbolos estranhos)
    tokens_special = tokenize_protein_sequence("mJ*a", max_len=5)
    assert tokens_special[0] == PROTEIN_CHAR_TO_IDX["M"]
    assert tokens_special[1] == 0  # 'J' fora do alfabeto
    assert tokens_special[2] == 0  # '*' fora do alfabeto
    assert tokens_special[3] == PROTEIN_CHAR_TO_IDX["A"]
    assert tokens_special[4] == 0  # padding


def test_compute_aa_composition():
    """Valida vetor de 25 frequências relativas para baseline trivial Ridge."""
    # Sequência vazia
    comp_empty = compute_aa_composition("")
    assert comp_empty.shape == (25,)
    assert comp_empty.dtype == np.float32
    assert (comp_empty == 0.0).all()

    # Resíduo único 'A'
    comp_a = compute_aa_composition("A")
    assert comp_a.shape == (25,)
    assert comp_a[PROTEIN_CHAR_TO_IDX["A"] - 1] == pytest.approx(1.0)
    assert comp_a.sum() == pytest.approx(1.0)

    # Sequência com repetições e minúsculas
    comp_mix = compute_aa_composition("acde")
    assert comp_mix.sum() == pytest.approx(1.0)
    for ch in "ACDE":
        assert comp_mix[PROTEIN_CHAR_TO_IDX[ch] - 1] == pytest.approx(0.25)


def test_dta_dataset_prebuilds_unique_entities_and_collates():
    """Valida pré-construção única em memória, lookup O(1) e empacotamento PyG Batch."""
    from torch_geometric.data import Batch

    cache = {
        "CCO": smiles_to_graph_dict("CCO", drug_id="D1", featurizer_version="v2_onehot78_l1norm"),
        "CCN": smiles_to_graph_dict("CCN", drug_id="D2", featurizer_version="v2_onehot78_l1norm"),
    }
    rows = [
        {"drug_id": "D1", "smiles": "CCO", "target_id": "T1", "sequence": "MKTV", "affinity": 7.5},
        {"drug_id": "D1", "smiles": "CCO", "target_id": "T1", "sequence": "MKTV", "affinity": 7.5},
        {"drug_id": "D2", "smiles": "CCN", "target_id": "T2", "sequence": "ACDE" * 300, "affinity": 6.2},
    ]
    df = pd.DataFrame(rows)
    ds = DTADataset(df, drug_graph_cache=cache, max_seq_len=1000)

    assert len(ds) == 3
    # Garante que Data e tokens foram pré-construídos apenas para entidades únicas
    assert len(ds.unique_graphs) == 2
    assert len(ds.unique_tokens) == 2
    assert len(ds.unique_seq_lens) == 2
    assert ds[0]["drug_graph"] is ds[1]["drug_graph"]
    assert ds[0]["target_tokens"] is ds[1]["target_tokens"]
    assert ds[2]["seq_len"] == 1200

    # Teste de __getitem__
    sample = ds[0]
    assert sample["smiles"] == "CCO"
    assert sample["drug_id"] == "D1"
    assert sample["target_id"] == "T1"
    assert sample["affinity"].dtype == torch.float32
    assert sample["affinity"].item() == pytest.approx(7.5)

    # Teste de collate
    batch = dta_collate_fn([ds[0], ds[2]])
    assert isinstance(batch["drug_batch"], Batch)
    assert batch["drug_batch"].x.shape[1] == 78
    assert batch["target_tokens"].shape == (2, 1000)
    assert batch["affinity"].shape == (2,)
    assert batch["affinity"].dtype == torch.float32
    assert batch["seq_lengths"].tolist() == [4, 1200]
    assert batch["smiles"] == ["CCO", "CCN"]
    assert batch["drug_id"] == ["D1", "D2"]
    assert batch["target_id"] == ["T1", "T2"]


def test_dta_dataset_missing_smiles_raises_key_error():
    """Valida erro explícito se algum SMILES do DataFrame faltar no cache."""
    cache = {
        "CCO": smiles_to_graph_dict("CCO", drug_id="D1", featurizer_version="v2_onehot78_l1norm"),
    }
    df = pd.DataFrame([
        {"drug_id": "D1", "smiles": "CCO", "target_id": "T1", "sequence": "MKTV", "affinity": 7.5},
        {"drug_id": "D2", "smiles": "CCN", "target_id": "T2", "sequence": "ACDE", "affinity": 6.2},
    ])
    with pytest.raises(KeyError, match="CCN"):
        DTADataset(df, drug_graph_cache=cache, max_seq_len=1000)


def test_dta_dataset_loads_from_path(tmp_path: Path):
    """Valida inicialização do DTADataset fornecendo caminho Path para o cache .pt."""
    cache = {
        "CCO": smiles_to_graph_dict("CCO", drug_id="D1", featurizer_version="v2_onehot78_l1norm"),
    }
    cache_path = tmp_path / "davis_drug_graphs_v2_onehot78_l1norm.pt"
    torch.save(cache, cache_path)

    df = pd.DataFrame([
        {"drug_id": "D1", "smiles": "CCO", "target_id": "T1", "sequence": "MKTV", "affinity": 7.5},
    ])
    ds = DTADataset(df, drug_graph_cache=cache_path, max_seq_len=500)
    assert len(ds) == 1
    assert ds[0]["smiles"] == "CCO"
    assert ds[0]["target_tokens"].shape == (500,)


def test_get_dta_dataloaders(tmp_path: Path):
    """Valida carregamento completo com split JSON, clean.csv e descoberta de cache com fallback."""
    root = tmp_path
    data_processed = root / "data" / "processed"
    splits_dir = data_processed / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cria DataFrame limpo com 4 registros
    df = pd.DataFrame([
        {"drug_id": "D1", "smiles": "CCO", "target_id": "T1", "sequence": "MKTV", "affinity": 7.0},
        {"drug_id": "D2", "smiles": "CCN", "target_id": "T2", "sequence": "ACDE", "affinity": 6.5},
        {"drug_id": "D3", "smiles": "CCC", "target_id": "T3", "sequence": "WXYZ", "affinity": 5.5},
        {"drug_id": "D4", "smiles": "CCCC", "target_id": "T4", "sequence": "GGGG", "affinity": 8.0},
    ])
    clean_csv = data_processed / "davis_clean.csv"
    df.to_csv(clean_csv, index=False)

    # 2. Cria cache de grafos
    cache = {
        row["smiles"]: smiles_to_graph_dict(row["smiles"], drug_id=row["drug_id"], featurizer_version="v2_onehot78_l1norm")
        for _, row in df.iterrows()
    }
    cache_path = data_processed / "davis_drug_graphs_v2_onehot78_l1norm.pt"
    torch.save(cache, cache_path)

    # 3. Cria split determinístico
    split_data = {
        "train": [0, 1],
        "val": [2],
        "test": [3],
    }
    split_path = splits_dir / "davis_random_seed42.json"
    split_path.write_text(json.dumps(split_data), encoding="utf-8")

    # Configuração de treino
    config = {
        "featurizer": {"version": "v2_onehot78_l1norm"},
        "graphdta": {
            "max_seq_len": 800,
            "training": {
                "batch_size": 2,
                "num_workers": 0,
                "pin_memory": False,
                "persistent_workers": False,
            },
        },
    }

    train_loader, val_loader, test_loader = get_dta_dataloaders(
        dataset_name="davis",
        split_type="random",
        seed=42,
        config=config,
        root=root,
    )

    assert len(train_loader.dataset) == 2
    assert len(val_loader.dataset) == 1
    assert len(test_loader.dataset) == 1

    # Itera sobre train_loader e valida lote PyG
    for batch in train_loader:
        assert "drug_batch" in batch
        assert "target_tokens" in batch
        assert batch["target_tokens"].shape == (2, 800)
        assert batch["affinity"].shape == (2,)
        break

    # Testa fallback quando o arquivo versionado não existe, mas o legado .pt existe
    cache_path.unlink()
    fallback_cache_path = data_processed / "davis_drug_graphs.pt"
    torch.save(cache, fallback_cache_path)

    train_loader_fb, _, _ = get_dta_dataloaders(
        dataset_name="davis",
        split_type="random",
        seed=42,
        config=config,
        root=root,
    )
    assert len(train_loader_fb.dataset) == 2


def test_prepare_local_cache(tmp_path: Path):
    """Valida descompactação segura de tarball no diretório local de cache."""
    # Cria diretório de origem com arquivos dummy
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "file1.txt").write_text("conteúdo 1", encoding="utf-8")
    (source_dir / "file2.pt").write_bytes(b"dados binarios")

    # Empacota em tar.gz
    tar_path = tmp_path / "cache.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(source_dir / "file1.txt", arcname="file1.txt")
        tar.add(source_dir / "file2.pt", arcname="file2.pt")

    # Descompacta usando prepare_local_cache
    dest_dir = tmp_path / "local_cache"
    result = prepare_local_cache(tar_path, dest_dir)
    assert result == dest_dir
    assert (dest_dir / "file1.txt").is_file()
    assert (dest_dir / "file1.txt").read_text(encoding="utf-8") == "conteúdo 1"
    assert (dest_dir / "file2.pt").is_file()

    # Se arquivo não existe, deve lançar FileNotFoundError
    with pytest.raises(FileNotFoundError):
        prepare_local_cache(tmp_path / "inexistente.tar.gz", dest_dir)
