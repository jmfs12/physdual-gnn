"""Arquitetura da Fase 3 do PhysDual-GNN (`GIN + ESM-2 congelado + MLP` e ablação `ECFP-MLP`).

Implementa:
- `ESMTargetEncoder`: Projeção regularizada de embeddings pré-computados do ESM-2
  (`LayerNorm -> Dropout -> Linear -> GELU/ReLU -> LayerNorm -> Dropout`) sobre vetores
  mean-pooled `[B, esm_dim]`.
- `ECFPDrugEncoder`: Codificador MLP de fingerprints Morgan ECFP4 (`[B, ecfp_dim] -> [B, output_dim]`)
  para o controle diagnóstico `ecfp_mlp`.
- `PhysDualPhase3`: Dual-encoder enxuto combinando `GINEncoder` (78d L1-norm) +
  `LayerNorm`, `ESMTargetEncoder` congelado e cabeça regressiva MLP.
- `PhysDualECFPPhase3`: Variante controle substituindo apenas o ramo molecular (`GIN` -> `ECFP-MLP`),
  mantendo `ESMTargetEncoder` e `fusion_mlp` idênticos.
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

    def forward(self, esm_emb: torch.Tensor) -> torch.Tensor:
        """Projeta embeddings mean-pooled ESM-2 `[B, esm_dim]` para `[B, output_dim]`."""
        x = esm_emb.float()
        if x.ndim != 2:
            raise ValueError(f"esm_emb deve ter 2 dimensões [B, esm_dim], recebido shape {tuple(x.shape)}")

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
    ) -> torch.Tensor:
        """Executa forward pass retornando afinidade predita com shape `[B]`."""
        z_drug = self.drug_norm(self.drug_encoder(drug_batch))
        z_prot = self.target_encoder(target_esm)
        fused = torch.cat([z_drug, z_prot], dim=1)
        return self.fusion_mlp(fused).squeeze(-1)


class ECFPDrugEncoder(nn.Module):
    """Codificador MLP de fingerprints Morgan ECFP4 para controle diagnóstico (`ecfp_mlp`)."""

    def __init__(
        self,
        fp_dim: int = 2048,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.fp_dim = int(fp_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.net = nn.Sequential(
            nn.Linear(self.fp_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )

    def forward(self, drug_ecfp: torch.Tensor) -> torch.Tensor:
        """Projeta fingerprints `[B, fp_dim]` para `[B, output_dim]`."""
        x = drug_ecfp.float()
        if x.ndim != 2:
            raise ValueError(f"drug_ecfp deve ter 2 dimensões [B, fp_dim], recebido shape {tuple(x.shape)}")
        return self.net(x)


class PhysDualECFPPhase3(nn.Module):
    """Controle diagnóstico: `ECFP-MLP` + `ESMTargetEncoder` + `MLP` idênticos ao `PhysDualPhase3`."""

    expects_ecfp: bool = True

    def __init__(
        self,
        ecfp_dim: int,
        esm_dim: int,
        ecfp_hidden_dim: int = 256,
        drug_output_dim: int = 128,
        ecfp_dropout: float = 0.2,
        prot_output_dim: int = 128,
        esm_dropout_in: float = 0.4,
        prot_dropout: float = 0.3,
        prot_activation: Literal["gelu", "relu"] = "gelu",
        mlp_hidden_dims: Sequence[int] = (512, 256),
        mlp_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.drug_in_dim = int(ecfp_dim)
        self.ecfp_dim = int(ecfp_dim)
        self.esm_dim = int(esm_dim)
        self.drug_output_dim = int(drug_output_dim)
        self.prot_output_dim = int(prot_output_dim)

        self.drug_encoder = ECFPDrugEncoder(
            fp_dim=self.ecfp_dim,
            hidden_dim=int(ecfp_hidden_dim),
            output_dim=self.drug_output_dim,
            dropout=float(ecfp_dropout),
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
        drug_ecfp: torch.Tensor,
        target_esm: torch.Tensor,
    ) -> torch.Tensor:
        """Executa forward pass retornando afinidade predita com shape `[B]`."""
        z_drug = self.drug_norm(self.drug_encoder(drug_ecfp))
        z_prot = self.target_encoder(target_esm)
        fused = torch.cat([z_drug, z_prot], dim=1)
        return self.fusion_mlp(fused).squeeze(-1)


def build_physdual_phase3_model(
    config: dict[str, Any],
    model_name: str = "physdual_gin_esm_mlp",
) -> PhysDualPhase3 | PhysDualECFPPhase3:
    """Instancia `PhysDualPhase3` ou `PhysDualECFPPhase3` a partir do dicionário de configuração global."""
    p3_cfg = config.get("physdual_phase3", {})
    feat_cfg = config.get("featurizer", {})
    esm_cfg = config.get("esm", {})

    drug_in_dim = int(p3_cfg.get("drug_in_dim", feat_cfg.get("atom_feature_dim", 78)))
    esm_dim = int(p3_cfg.get("esm_dim", esm_cfg.get("esm_dim", 1280)))
    normalized_name = model_name.lower().strip()

    if normalized_name in ("physdual_ecfp_esm_mlp", "ecfp_mlp"):
        return PhysDualECFPPhase3(
            ecfp_dim=int(p3_cfg.get("ecfp_dim", 2048)),
            esm_dim=esm_dim,
            ecfp_hidden_dim=int(p3_cfg.get("ecfp_hidden_dim", 256)),
            drug_output_dim=int(p3_cfg.get("drug_output_dim", 128)),
            ecfp_dropout=float(p3_cfg.get("ecfp_dropout", p3_cfg.get("gin_dropout", 0.2))),
            prot_output_dim=int(p3_cfg.get("prot_output_dim", 128)),
            esm_dropout_in=float(p3_cfg.get("esm_dropout_in", 0.4)),
            prot_dropout=float(p3_cfg.get("prot_dropout", 0.3)),
            prot_activation=str(p3_cfg.get("prot_activation", "gelu")),  # type: ignore[arg-type]
            mlp_hidden_dims=tuple(int(x) for x in p3_cfg.get("mlp_hidden_dims", [512, 256])),
            mlp_dropout=float(p3_cfg.get("mlp_dropout", 0.3)),
        )

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
