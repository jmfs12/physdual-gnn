"""Módulo de dados, tokenizador de proteínas, DTADataset e DataLoaders PyG (Fase 2).

Implementa:
- Constantes de alfabeto e vocabulário de proteínas (`PROTEIN_ALPHABET`, `PROTEIN_CHAR_TO_IDX`).
- Tokenização rápida de sequências com padding e truncamento (`tokenize_protein_sequence`).
- Composição de 25 aminoácidos para baseline trivial Ridge (`compute_aa_composition`).
- `DTADataset`: Dataset PyTorch com pré-construção em memória de grafos e sequências
  únicas, permitindo indexação O(1) e reuso eficiente de tensores sem sobrecarga de I/O.
- `dta_collate_fn`: empacotamento PyG Batch de grafos moleculares e tensores de sequência/afinidade.
- `get_dta_dataloaders`: carregamento automatizado de dados limpos, splits e caches com fallback.
- `prepare_local_cache`: descompactação segura de arquivos tarball no SSD local da VM.
"""

from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data

from physdual.data.esm_cache import load_cached_protein_embedding
from physdual.data.featurizer_drug import load_drug_graph, load_drug_graph_cache
from physdual.data.splits import load_splits
from physdual.utils.paths import get_project_root

# 25 caracteres IUPAC, excluindo 'J' (conforme ADR #8 e especificação GraphDTA/DeepDTA)
PROTEIN_ALPHABET = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
PROTEIN_CHAR_TO_IDX: dict[str, int] = {char: idx for idx, char in enumerate(PROTEIN_ALPHABET, start=1)}


def tokenize_protein_sequence(seq: str, max_len: int = 1000) -> np.ndarray:
    """Converte uma sequência de aminoácidos em array 1D de inteiros com padding à direita.

    Args:
        seq: Sequência primária da proteína.
        max_len: Comprimento máximo fixo do vetor de saída.

    Returns:
        Array 1D de `np.int64` com tamanho exatamente igual a `max_len`. Caracteres fora
        do vocabulário ou posições de preenchimento são codificados com `0`.
    """
    tokens = np.zeros(max_len, dtype=np.int64)
    if not seq:
        return tokens

    clean_seq = seq.strip().upper()[:max_len]
    for i, ch in enumerate(clean_seq):
        tokens[i] = PROTEIN_CHAR_TO_IDX.get(ch, 0)
    return tokens


def compute_aa_composition(seq: str) -> np.ndarray:
    """Calcula as frequências relativas dos 25 aminoácidos de `PROTEIN_ALPHABET`.

    Usado como vetor de características de alvos para baselines triviais (Ridge).

    Args:
        seq: Sequência primária da proteína.

    Returns:
        Array 1D de `np.float32` de dimensão 25 somando 1.0 (ou vetor de zeros se a
        sequência for vazia).
    """
    comp = np.zeros(len(PROTEIN_ALPHABET), dtype=np.float32)
    if not seq:
        return comp

    clean_seq = seq.strip().upper()
    total_len = len(clean_seq)
    if total_len == 0:
        return comp

    for ch in clean_seq:
        idx = PROTEIN_CHAR_TO_IDX.get(ch, 0)
        if idx > 0:
            comp[idx - 1] += 1.0

    return comp / float(total_len)


def resolve_esm_cache_dir(
    config: dict[str, Any] | None = None,
    root: Path | None = None,
    esm_cache_dir: Path | str | None = None,
) -> Path:
    """Resolve o diretório do cache ESM-2 priorizando argumento explícito e depois `configs/base.yaml`."""
    if esm_cache_dir is not None:
        return Path(esm_cache_dir)
    if config is None:
        from physdual.utils.config import load_config
        config = load_config()
    base_root = Path(root) if root is not None else Path(config.get("project_root", get_project_root()))
    rel_cache = config.get("paths", {}).get("esm_cache_dir", "cache/esm_embeddings")
    p = Path(rel_cache)
    return p if p.is_absolute() else (base_root / p)


def load_esm_mean_pool_dict(
    sequences: Iterable[str],
    model_name: str = "esm2_t33_650M_UR50D",
    cache_dir: Path | str = "cache/esm_embeddings",
    expected_dim: int = 1280,
) -> dict[str, torch.Tensor]:
    """Carrega embeddings `.pt` (`weights_only=True`) e retorna `seq -> mean_pool(fp32)`."""
    cache_path = Path(cache_dir)
    out: dict[str, torch.Tensor] = {}
    for seq in sorted({str(s) for s in sequences}):
        emb = load_cached_protein_embedding(
            sequence=seq,
            model_name=model_name,
            cache_dir=cache_path,
            expected_dim=expected_dim,
        )
        vec = emb.mean(dim=0).to(dtype=torch.float32).contiguous()
        if not torch.isfinite(vec).all():
            raise ValueError(f"Embedding ESM-2 contém NaN/Inf para sequência de len={len(seq)}")
        out[seq] = vec
    return out


class DTADataset(Dataset):
    """Dataset de afinidade fármaco-alvo com pré-construção em memória para máxima eficiência de CPU/IO.

    Pré-constrói os objetos `torch_geometric.data.Data` para cada SMILES único e os tensores
    de tokens (ou vetores mean-pooled do ESM-2 quando `esm_cache` é fornecido) durante a inicialização.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        drug_graph_cache: dict[str, Any] | Path | str,
        max_seq_len: int = 1000,
        esm_cache: dict[str, torch.Tensor] | Path | str | None = None,
        esm_model_name: str = "esm2_t33_650M_UR50D",
        esm_dim: int = 1280,
    ) -> None:
        """Inicializa o DTADataset.

        Args:
            df: DataFrame com colunas obrigatórias `smiles`, `sequence` e `affinity`.
            drug_graph_cache: Dicionário contendo os grafos featurizados ou caminho do arquivo `.pt`.
            max_seq_len: Comprimento máximo de truncamento/padding para tokens de proteínas.
            esm_cache: Dicionário `seq -> Tensor[esm_dim]` (ou `[L, esm_dim]`) ou caminho do diretório cache.
            esm_model_name: Nome do modelo ESM-2 caso `esm_cache` seja caminho em disco.
            esm_dim: Dimensão esperada do embedding ESM-2.
        """
        for req_col in ("smiles", "sequence", "affinity"):
            if req_col not in df.columns:
                raise ValueError(f"Coluna obrigatória '{req_col}' ausente no DataFrame.")

        self.df = df.reset_index(drop=True)
        self.max_seq_len = int(max_seq_len)

        if isinstance(drug_graph_cache, (str, Path)):
            cache_dict = load_drug_graph_cache(Path(drug_graph_cache))
        elif isinstance(drug_graph_cache, dict):
            cache_dict = drug_graph_cache
        else:
            raise TypeError(
                f"`drug_graph_cache` deve ser um dict ou caminho Path/str, recebido: {type(drug_graph_cache)}"
            )

        # Pré-construção de grafos moleculares para SMILES únicos
        unique_smiles = self.df["smiles"].astype(str).unique()
        self.unique_graphs: dict[str, Data] = {}
        for smi in unique_smiles:
            if smi in cache_dict and isinstance(cache_dict[smi], Data):
                self.unique_graphs[smi] = cache_dict[smi]
            else:
                try:
                    self.unique_graphs[smi] = load_drug_graph(smi, cache_dict)
                except KeyError as e:
                    raise KeyError(f"SMILES '{smi}' ausente no cache de grafos de fármacos: {e}") from e

        # Pré-construção de tensores de tokens e comprimentos para sequências únicas
        unique_seqs = self.df["sequence"].astype(str).unique()
        self.unique_tokens: dict[str, torch.Tensor] = {}
        self.unique_seq_lens: dict[str, int] = {}
        for seq in unique_seqs:
            tokens_np = tokenize_protein_sequence(seq, max_len=self.max_seq_len)
            self.unique_tokens[seq] = torch.tensor(tokens_np, dtype=torch.long)
            self.unique_seq_lens[seq] = len(seq)

        # Cache opcional de embeddings ESM-2 (mean-pooled em float32)
        self.unique_esm_vecs: dict[str, torch.Tensor] | None = None
        if esm_cache is not None:
            if isinstance(esm_cache, (str, Path)):
                self.unique_esm_vecs = load_esm_mean_pool_dict(
                    unique_seqs,
                    model_name=esm_model_name,
                    cache_dir=Path(esm_cache),
                    expected_dim=int(esm_dim),
                )
            elif isinstance(esm_cache, dict):
                self.unique_esm_vecs = {}
                for seq in unique_seqs:
                    if seq not in esm_cache:
                        raise KeyError(f"Sequência ausente no `esm_cache` (len={len(seq)}).")
                    t = torch.as_tensor(esm_cache[seq], dtype=torch.float32)
                    if t.ndim == 2:
                        t = t.mean(dim=0)
                    if t.ndim != 1 or int(t.size(0)) != int(esm_dim):
                        raise ValueError(
                            f"Shape ESM inválido para seq len={len(seq)}: {tuple(t.shape)}, esperado ({esm_dim},)"
                        )
                    self.unique_esm_vecs[seq] = t.contiguous()
            else:
                raise TypeError(
                    f"`esm_cache` deve ser dict, Path/str ou None, recebido: {type(esm_cache)}"
                )

        # Vetores indexados para acesso O(1) de alta velocidade
        self._smiles_list: list[str] = self.df["smiles"].astype(str).tolist()
        self._seq_list: list[str] = self.df["sequence"].astype(str).tolist()
        self._affinity_list: np.ndarray = self.df["affinity"].to_numpy(dtype=np.float32)

        if "drug_id" in self.df.columns:
            self._drug_id_list: list[str] = self.df["drug_id"].astype(str).tolist()
        else:
            self._drug_id_list = [""] * len(self.df)

        if "target_id" in self.df.columns:
            self._target_id_list: list[str] = self.df["target_id"].astype(str).tolist()
        else:
            self._target_id_list = [""] * len(self.df)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        smiles = self._smiles_list[idx]
        seq = self._seq_list[idx]
        item: dict[str, Any] = {
            "drug_graph": self.unique_graphs[smiles],
            "target_tokens": self.unique_tokens[seq],
            "seq_len": int(self.unique_seq_lens[seq]),
            "smiles": str(smiles),
            "sequence": str(seq),
            "target_id": str(self._target_id_list[idx]),
            "drug_id": str(self._drug_id_list[idx]),
            "affinity": torch.tensor(float(self._affinity_list[idx]), dtype=torch.float32),
        }
        if self.unique_esm_vecs is not None:
            item["target_esm"] = self.unique_esm_vecs[seq]
        return item


def dta_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Agrupa amostras individuais em um lote PyG para GraphDTA / PhysDual-GNN.

    Empacota `drug_graph` via `torch_geometric.data.Batch.from_data_list`,
    empilha os tensores de sequência (e `target_esm` se presente) e as afinidades em `[B]`.
    """
    drug_batch = Batch.from_data_list([item["drug_graph"] for item in batch])
    target_tokens = torch.stack([item["target_tokens"] for item in batch])
    affinity = torch.stack([item["affinity"] for item in batch])
    seq_lengths = torch.tensor([item["seq_len"] for item in batch], dtype=torch.long)

    collated: dict[str, Any] = {
        "drug_batch": drug_batch,
        "target_tokens": target_tokens,
        "affinity": affinity,
        "seq_lengths": seq_lengths,
        "smiles": [item["smiles"] for item in batch],
        "sequence": [item["sequence"] for item in batch],
        "target_id": [item["target_id"] for item in batch],
        "drug_id": [item["drug_id"] for item in batch],
    }
    if "target_esm" in batch[0]:
        collated["target_esm"] = torch.stack([item["target_esm"] for item in batch])
    return collated


def get_dta_dataloaders(
    dataset_name: str,
    split_type: str,
    seed: int = 0,
    config: dict[str, Any] | None = None,
    root: Path | None = None,
    protocol: str = "seeds",
    fold: int | None = None,
    use_esm: bool = False,
    esm_cache_dir: Path | str | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Cria e retorna os DataLoaders `(train_loader, val_loader, test_loader)` para um dataset e split.

    Args:
        dataset_name: Nome do dataset (`"davis"` ou `"kiba"`).
        split_type: Tipo de partição (`"random"`, `"cold_drug"` ou `"cold_target"`).
        seed: Semente do particionamento determinístico (quando protocol="seeds").
        config: Dicionário global de configuração (`configs/base.yaml`).
        root: Raiz do projeto (opcional, resolvida via `get_project_root()` se `None`).
        protocol: `"seeds"` (padrão) ou `"kfold"`.
        fold: Índice da dobra quando protocol="kfold" (0 a K-1).

    Returns:
        Tupla `(train_loader, val_loader, test_loader)` configurada conforme `config["graphdta"]["training"]`.
    """
    if config is None:
        from physdual.utils.config import load_config
        config = load_config()

    base_root = Path(root) if root is not None else get_project_root()
    clean_csv_path = base_root / "data" / "processed" / f"{dataset_name.lower()}_clean.csv"
    if not clean_csv_path.exists():
        raise FileNotFoundError(f"Arquivo CSV limpo do dataset não encontrado: {clean_csv_path}")

    df = pd.read_csv(clean_csv_path)

    # Descoberta do cache de grafos de fármacos (versão prioritária com fallback legado)
    feat_cfg = config.get("featurizer", {})
    featurizer_version = feat_cfg.get("version", "v2_onehot78_l1norm")
    expected_dim = int(feat_cfg.get("atom_dim", feat_cfg.get("atom_feature_dim", 78)))
    expected_norm = bool(feat_cfg.get("normalize_atom_features", True))

    cache_versioned = base_root / "data" / "processed" / f"{dataset_name.lower()}_drug_graphs_{featurizer_version}.pt"
    cache_fallback = base_root / "data" / "processed" / f"{dataset_name.lower()}_drug_graphs.pt"

    if cache_versioned.exists():
        chosen_cache_path = cache_versioned
    elif cache_fallback.exists():
        chosen_cache_path = cache_fallback
    else:
        raise FileNotFoundError(
            f"Cache de grafos para '{dataset_name}' não encontrado em '{cache_versioned}' nem em '{cache_fallback}'."
        )

    # Carrega cache em memória uma única vez e compartilha com as 3 partições
    drug_cache = load_drug_graph_cache(
        chosen_cache_path,
        expected_version=featurizer_version,
        expected_dim=expected_dim,
        expected_normalize=expected_norm,
    )

    # Carregamento do arquivo de splits (suporta protocolo seeds e protocolo kfold)
    if protocol == "kfold" or fold is not None:
        target_fold = fold if fold is not None else seed
        split_path = base_root / "data" / "processed" / "splits" / f"{dataset_name.lower()}_{split_type}_fold{target_fold}.json"
    else:
        split_path = base_root / "data" / "processed" / "splits" / f"{dataset_name.lower()}_{split_type}_seed{seed}.json"

    if not split_path.exists():
        raise FileNotFoundError(f"Arquivo de splits não encontrado: {split_path}")

    splits = load_splits(split_path)

    # Sub-dataframes por partição
    train_df = df.iloc[splits["train"]].copy()
    val_df = df.iloc[splits["val"]].copy()
    test_df = df.iloc[splits["test"]].copy()

    graphdta_cfg = config.get("graphdta", {})
    max_seq_len = int(graphdta_cfg.get("max_seq_len", 1000))

    shared_esm_cache: dict[str, torch.Tensor] | None = None
    esm_model_name = str(config.get("esm", {}).get("model_name", "esm2_t33_650M_UR50D"))
    esm_dim = int(config.get("esm", {}).get("esm_dim", 1280))
    if use_esm:
        resolved_esm_dir = resolve_esm_cache_dir(config=config, root=base_root, esm_cache_dir=esm_cache_dir)
        all_unique_seqs = pd.concat(
            [train_df["sequence"], val_df["sequence"], test_df["sequence"]],
            ignore_index=True,
        ).astype(str).unique()
        shared_esm_cache = load_esm_mean_pool_dict(
            all_unique_seqs,
            model_name=esm_model_name,
            cache_dir=resolved_esm_dir,
            expected_dim=esm_dim,
        )

    train_ds = DTADataset(
        train_df,
        drug_graph_cache=drug_cache,
        max_seq_len=max_seq_len,
        esm_cache=shared_esm_cache,
        esm_model_name=esm_model_name,
        esm_dim=esm_dim,
    )
    val_ds = DTADataset(
        val_df,
        drug_graph_cache=drug_cache,
        max_seq_len=max_seq_len,
        esm_cache=shared_esm_cache,
        esm_model_name=esm_model_name,
        esm_dim=esm_dim,
    )
    test_ds = DTADataset(
        test_df,
        drug_graph_cache=drug_cache,
        max_seq_len=max_seq_len,
        esm_cache=shared_esm_cache,
        esm_model_name=esm_model_name,
        esm_dim=esm_dim,
    )

    train_cfg = (
        config.get("physdual_phase3", {}).get("training", graphdta_cfg.get("training", {}))
        if use_esm
        else graphdta_cfg.get("training", {})
    )
    batch_size = int(train_cfg.get("batch_size", 512))
    num_workers = int(train_cfg.get("num_workers", 0))
    pin_memory = bool(train_cfg.get("pin_memory", True))
    persistent_workers = (num_workers > 0) and bool(train_cfg.get("persistent_workers", True))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=dta_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=dta_collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=dta_collate_fn,
    )

    return train_loader, val_loader, test_loader


def prepare_local_cache(tar_path: Path | str, local_cache_dir: Path | str) -> Path:
    """Descompacta um arquivo tarball no diretório local de cache rápido (ex: SSD local do Colab).

    Args:
        tar_path: Caminho do arquivo compactado (.tar, .tar.gz, .tgz).
        local_cache_dir: Diretório de destino local onde os arquivos serão extraídos.

    Returns:
        Caminho do diretório de destino (`Path`).
    """
    tar_p = Path(tar_path)
    if not tar_p.is_file():
        raise FileNotFoundError(f"Arquivo tarball não encontrado: {tar_p}")

    dest = Path(local_cache_dir)
    dest.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_p, "r:*") as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(path=dest, filter="data")
        else:
            tar.extractall(path=dest)

    return dest
