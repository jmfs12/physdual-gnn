"""Arquitetura da Fase 3 do PhysDual-GNN (`GIN + ESM-2 congelado + MLP`).

Implementa:
- `ESMTargetEncoder`: Projeção regularizada de embeddings pré-computados do ESM-2
  (`LayerNorm -> Dropout -> Linear -> GELU/ReLU -> LayerNorm -> Dropout`), com suporte
  tanto a vetores já agregados `[B, esm_dim]` quanto a tensores por resíduo `[B, L, esm_dim]`
  com máscara de padding opcional.
- `PhysDualPhase3`: Dual-encoder enxuto combinando `GINEncoder` (78d L1-norm) +
  `LayerNorm`, `ESMTargetEncoder` congelado e cabeça regressiva MLP.
- `build_physdual_phase3_model`: Factory a partir do dicionário `configs/base.yaml`.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from physdual.models.graphdta import GINEncoder


class ESMTargetEncoder(nn.Module):
    """Codificador proteico regularizado sobre embeddings congelados do ESM-2."""

    def __init__(
        self,
        esm_dim: int,
        output_dim: int = 128,
        dropout_in: float = 0.4,
        dropout_out: float = 0.3,
        activation: Literal["gelu", "relu"] = "gelu",
    ) -> None:
        super().__init__()
        self.esm_dim = int(esm_dim)
        self.output_dim = int(output_dim)
        self.dropout_in = float(dropout_in)
        self.dropout_out = float(dropout_out)

        self.in_norm = nn.LayerNorm(self.esm_dim)
        self.in_drop = nn.Dropout(self.dropout_in)
        self.proj = nn.Linear(self.esm_dim, self.output_dim)
        self.act: nn.Module = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.out_norm = nn.LayerNorm(self.output_dim)
        self.out_drop = nn.Dropout(self.dropout_out)

    def forward(
        self,
        esm_emb: torch.Tensor,
        esm_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Projeta embeddings ESM-2 `[B, esm_dim]` ou `[B, L, esm_dim]` para `[B, output_dim]`."""
        x = esm_emb.float()
        if x.ndim == 3:
            if esm_mask is not None:
                mask = esm_mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
                denom = mask.sum(dim=1).clamp(min=1.0)
                x = (x * mask).sum(dim=1) / denom
            else:
                x = x.mean(dim=1)
        elif x.ndim != 2:
            raise ValueError(f"esm_emb deve ter 2 ou 3 dimensões, recebido shape {tuple(x.shape)}")

        x = self.in_norm(x)
        x = self.in_drop(x)
        x = self.proj(x)
        x = self.act(x)
        x = self.out_norm(x)
        x = self.out_drop(x)
        return x


class PhysDualPhase3(nn.Module):
    """Dual-Encoder da Fase 3: `GINEncoder` + `ESMTargetEncoder` + `MLP` (late fusion)."""

    def __init__(
        self,
        drug_in_dim: int,
        esm_dim: int,
        gin_hidden_dim: int = 32,
        gin_num_layers: int = 5,
        drug_output_dim: int = 128,
        gin_dropout: float = 0.2,
        prot_output_dim: int = 128,
        esm_dropout_in: float = 0.4,
        prot_dropout: float = 0.3,
        prot_activation: Literal["gelu", "relu"] = "gelu",
        mlp_hidden_dims: Sequence[int] = (512, 256),
        mlp_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.drug_in_dim = int(drug_in_dim)
        self.esm_dim = int(esm_dim)
        self.drug_output_dim = int(drug_output_dim)
        self.prot_output_dim = int(prot_output_dim)

        self.drug_encoder = GINEncoder(
            drug_in_dim=self.drug_in_dim,
            hidden_dim=int(gin_hidden_dim),
            num_layers=int(gin_num_layers),
            output_dim=self.drug_output_dim,
            dropout=float(gin_dropout),
        )
        self.drug_norm = nn.LayerNorm(self.drug_output_dim)

        self.target_encoder = ESMTargetEncoder(
            esm_dim=self.esm_dim,
            output_dim=self.prot_output_dim,
            dropout_in=float(esm_dropout_in),
            dropout_out=float(prot_dropout),
            activation=prot_activation,
        )

        layers: list[nn.Module] = []
        in_dim = self.drug_output_dim + self.prot_output_dim
        for h_dim in mlp_hidden_dims:
            layers.append(nn.Linear(in_dim, int(h_dim)))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(float(mlp_dropout)))
            in_dim = int(h_dim)
        layers.append(nn.Linear(in_dim, 1))
        self.fusion_mlp = nn.Sequential(*layers)

    def forward(
        self,
        drug_batch: Batch | Data | torch.Tensor,
        target_esm: torch.Tensor,
        target_esm_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Executa forward pass retornando afinidade predita com shape `[B]`."""
        z_drug = self.drug_norm(self.drug_encoder(drug_batch))
        z_prot = self.target_encoder(target_esm, esm_mask=target_esm_mask)
        fused = torch.cat([z_drug, z_prot], dim=1)
        return self.fusion_mlp(fused).squeeze(-1)


def build_physdual_phase3_model(config: dict[str, Any]) -> PhysDualPhase3:
    """Instancia `PhysDualPhase3` a partir do dicionário de configuração global."""
    p3_cfg = config.get("physdual_phase3", {})
    feat_cfg = config.get("featurizer", {})
    esm_cfg = config.get("esm", {})

    drug_in_dim = int(p3_cfg.get("drug_in_dim", feat_cfg.get("atom_feature_dim", 78)))
    esm_dim = int(p3_cfg.get("esm_dim", esm_cfg.get("esm_dim", 1280)))

    return PhysDualPhase3(
        drug_in_dim=drug_in_dim,
        esm_dim=esm_dim,
        gin_hidden_dim=int(p3_cfg.get("gin_hidden_dim", 32)),
        gin_num_layers=int(p3_cfg.get("gin_num_layers", 5)),
        drug_output_dim=int(p3_cfg.get("drug_output_dim", 128)),
        gin_dropout=float(p3_cfg.get("gin_dropout", 0.2)),
        prot_output_dim=int(p3_cfg.get("prot_output_dim", 128)),
        esm_dropout_in=float(p3_cfg.get("esm_dropout_in", 0.4)),
        prot_dropout=float(p3_cfg.get("prot_dropout", 0.3)),
        prot_activation=str(p3_cfg.get("prot_activation", "gelu")),  # type: ignore[arg-type]
        mlp_hidden_dims=tuple(int(x) for x in p3_cfg.get("mlp_hidden_dims", [512, 256])),
        mlp_dropout=float(p3_cfg.get("mlp_dropout", 0.3)),
    )
