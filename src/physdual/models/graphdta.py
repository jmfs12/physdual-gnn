"""Modelos e encoders do baseline GraphDTA (Fase 2).

Implementa:
- `GINEncoder`: Encoder de grafos moleculares via Graph Isomorphism Network (5 camadas).
- `OfficialGraphDTATargetEncoder`: Encoder convolucional 1D de sequências de proteínas fiel ao
  repositório oficial thinng/GraphDTA (commit 6162245c6776684b391d30610850fe7077478435).
- `DeepDTAStyleCNN3TargetEncoder`: Variante secundária com CNN de 3 camadas estilo DeepDTA.
- `GraphDTAGIN`: Baseline oficial combinando GINEncoder + OfficialGraphDTATargetEncoder + MLP.
- `GraphDTACNN3`: Variante secundária combinando GINEncoder + DeepDTAStyleCNN3TargetEncoder + MLP.
- `build_baseline_model`: Factory para instanciação a partir do dicionário de configuração.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GINConv, global_add_pool


class GINEncoder(nn.Module):
    """Encoder de grafo molecular baseado em Graph Isomorphism Network (GIN).

    Reproduz o extrator de características de fármacos do GraphDTA com 5 camadas GINConv,
    normalização por lote (BatchNorm1d) e agregação por soma global (global_add_pool).
    Restrição D2: `drug_in_dim` não possui valor default mágico.
    """

    def __init__(
        self,
        drug_in_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 5,
        output_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.drug_in_dim = drug_in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_dim = drug_in_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.fc1_xd = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor | Batch | Data,
        edge_index: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Avanço feedforward do encoder GIN.

        Suporta tanto chamada direta com `(drug_batch)` PyG quanto `(x, edge_index, batch)`.
        """
        if hasattr(x, "edge_index"):
            batch = getattr(x, "batch", None)
            edge_index = x.edge_index
            x = x.x

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(conv(x, edge_index))
            x = bn(x)

        x = global_add_pool(x, batch)
        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class OfficialGraphDTATargetEncoder(nn.Module):
    """Encoder convolucional 1D oficial do GraphDTA para sequências de proteínas.

    Reproduz fielmente a arquitetura de models/ginconv.py (GINConvNet) do repositório
    oficial thinng/GraphDTA (commit 6162245c6776684b391d30610850fe7077478435).

    Aplica uma única Conv1d tratando o comprimento máximo da sequência (max_seq_len=1000)
    como número de canais de entrada sobre o vetor embutido (embedding_dim=128),
    seguido de linearização (flatten) e projeção densa linear para output_dim=128.
    """

    def __init__(
        self,
        protein_vocab_size: int = 25,
        max_seq_len: int = 1000,
        embedding_dim: int = 128,
        n_filters: int = 32,
        kernel_size: int = 8,
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        self.protein_vocab_size = protein_vocab_size
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.output_dim = output_dim

        self.embedding_xt = nn.Embedding(protein_vocab_size + 1, embedding_dim)
        self.conv_xt_1 = nn.Conv1d(
            in_channels=max_seq_len,
            out_channels=n_filters,
            kernel_size=kernel_size,
        )
        conv_out_len = embedding_dim - kernel_size + 1
        self.fc1_xt = nn.Linear(n_filters * conv_out_len, output_dim)

    def forward(self, target_tokens: torch.Tensor) -> torch.Tensor:
        # target_tokens: [B, max_seq_len]
        embedded_xt = self.embedding_xt(target_tokens)  # [B, max_seq_len, embedding_dim]
        conv_xt = self.conv_xt_1(embedded_xt)  # [B, n_filters, embedding_dim - kernel_size + 1]
        xt = conv_xt.flatten(start_dim=1)  # [B, n_filters * conv_out_len]
        xt = self.fc1_xt(xt)  # [B, output_dim]
        return xt


class DeepDTAStyleCNN3TargetEncoder(nn.Module):
    """CNN de 3 camadas estilo DeepDTA para sequências de proteínas (Variante Secundária).

    Explicitamente rotulada como 'CNN de 3 camadas estilo DeepDTA' (conforme ADR e Task 3).
    Transpõe os embeddings para [B, embedding_dim, max_seq_len] e aplica 3 convoluções 1D
    com filtros progressivos (32, 64, 96), seguidas de AdaptiveMaxPool1d(1) e projeção linear.
    """

    def __init__(
        self,
        protein_vocab_size: int = 25,
        max_seq_len: int = 1000,
        embedding_dim: int = 128,
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        self.protein_vocab_size = protein_vocab_size
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim

        self.embedding = nn.Embedding(protein_vocab_size + 1, embedding_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=32, kernel_size=4)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=96, kernel_size=12)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(96, output_dim)

    def forward(self, target_tokens: torch.Tensor) -> torch.Tensor:
        # target_tokens: [B, L]
        x = self.embedding(target_tokens).transpose(1, 2)  # [B, embedding_dim, L]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)  # [B, 96]
        x = self.fc(x)  # [B, output_dim]
        return x


class GraphDTAGIN(nn.Module):
    """Modelo Baseline Oficial GraphDTA com GIN e encoder convolucional oficial de proteína.

    Reproduz fielmente GINConvNet de thinng/GraphDTA.
    Combina GINEncoder para o fármaco e OfficialGraphDTATargetEncoder para a proteína,
    processando a concatenação através de MLP:
    Linear(256, 1024) -> ReLU -> Dropout(0.2) -> Linear(1024, 256) -> ReLU -> Dropout(0.2) -> Linear(256, 1).
    Restrição D2: `drug_in_dim` não possui valor default mágico.
    """

    def __init__(
        self,
        drug_in_dim: int,
        protein_vocab_size: int = 25,
        max_seq_len: int = 1000,
        embedding_dim: int = 128,
        gin_hidden_dim: int = 32,
        gin_num_layers: int = 5,
        cnn_num_filters: int = 32,
        cnn_kernel_size: int = 8,
        output_dim: int = 128,
        mlp_hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [1024, 256]

        self.drug_encoder = GINEncoder(
            drug_in_dim=drug_in_dim,
            hidden_dim=gin_hidden_dim,
            num_layers=gin_num_layers,
            output_dim=output_dim,
            dropout=dropout,
        )
        self.target_encoder = OfficialGraphDTATargetEncoder(
            protein_vocab_size=protein_vocab_size,
            max_seq_len=max_seq_len,
            embedding_dim=embedding_dim,
            n_filters=cnn_num_filters,
            kernel_size=cnn_kernel_size,
            output_dim=output_dim,
        )

        layers: list[nn.Module] = []
        in_dim = output_dim * 2
        for hidden_dim in mlp_hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        drug_batch: Batch | Data | torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> torch.Tensor:
        xd = self.drug_encoder(drug_batch)
        xt = self.target_encoder(target_tokens)
        xc = torch.cat([xd, xt], dim=-1)
        out = self.mlp(xc)
        return out.view(-1)


class GraphDTACNN3(nn.Module):
    """Variante secundária GraphDTA combinando GINEncoder com CNN estilo DeepDTA (3 camadas).

    Restrição D2: `drug_in_dim` não possui valor default mágico.
    """

    def __init__(
        self,
        drug_in_dim: int,
        protein_vocab_size: int = 25,
        max_seq_len: int = 1000,
        embedding_dim: int = 128,
        gin_hidden_dim: int = 32,
        gin_num_layers: int = 5,
        cnn_num_filters: int = 32,
        cnn_kernel_size: int = 8,
        output_dim: int = 128,
        mlp_hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [1024, 256]

        self.drug_encoder = GINEncoder(
            drug_in_dim=drug_in_dim,
            hidden_dim=gin_hidden_dim,
            num_layers=gin_num_layers,
            output_dim=output_dim,
            dropout=dropout,
        )
        self.target_encoder = DeepDTAStyleCNN3TargetEncoder(
            protein_vocab_size=protein_vocab_size,
            max_seq_len=max_seq_len,
            embedding_dim=embedding_dim,
            output_dim=output_dim,
        )

        layers: list[nn.Module] = []
        in_dim = output_dim * 2
        for hidden_dim in mlp_hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        drug_batch: Batch | Data | torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> torch.Tensor:
        xd = self.drug_encoder(drug_batch)
        xt = self.target_encoder(target_tokens)
        xc = torch.cat([xd, xt], dim=-1)
        out = self.mlp(xc)
        return out.view(-1)


def build_baseline_model(
    model_name: Literal["graphdta_gin", "graphdta_cnn3"],
    config: dict[str, Any],
) -> nn.Module:
    """Constrói e retorna a instância do modelo baseline a partir da configuração do projeto.

    Args:
        model_name: "graphdta_gin" (baseline oficial) ou "graphdta_cnn3" (variante secundária).
        config: Dicionário contendo as seções 'graphdta' e/ou 'featurizer'.

    Returns:
        Instância inicializada de `GraphDTAGIN` ou `GraphDTACNN3`.
    """
    if model_name not in ("graphdta_gin", "graphdta_cnn3"):
        raise ValueError(
            f"Modelo desconhecido: '{model_name}'. Opções válidas: 'graphdta_gin', 'graphdta_cnn3'."
        )

    featurizer_cfg = config.get("featurizer", {})
    gdta_cfg = config.get("graphdta", {})

    drug_in_dim = featurizer_cfg.get("atom_feature_dim") or gdta_cfg.get("drug_in_dim")
    if drug_in_dim is None:
        raise ValueError(
            "A dimensão atômica 'drug_in_dim' deve ser fornecida em config['featurizer']['atom_feature_dim'] "
            "ou em config['graphdta']['drug_in_dim']."
        )

    kwargs = {
        "drug_in_dim": int(drug_in_dim),
        "protein_vocab_size": int(gdta_cfg.get("protein_vocab_size", 25)),
        "max_seq_len": int(gdta_cfg.get("max_seq_len", 1000)),
        "embedding_dim": int(gdta_cfg.get("embedding_dim", 128)),
        "gin_hidden_dim": int(gdta_cfg.get("gin_hidden_dim", 32)),
        "gin_num_layers": int(gdta_cfg.get("gin_num_layers", 5)),
        "cnn_num_filters": int(gdta_cfg.get("cnn_num_filters", 32)),
        "cnn_kernel_size": int(gdta_cfg.get("cnn_kernel_size", 8)),
        "output_dim": int(gdta_cfg.get("output_dim", 128)),
        "mlp_hidden_dims": list(gdta_cfg.get("mlp_hidden_dims", [1024, 256])),
        "dropout": float(gdta_cfg.get("dropout", 0.2)),
    }

    if model_name == "graphdta_gin":
        return GraphDTAGIN(**kwargs)
    return GraphDTACNN3(**kwargs)
