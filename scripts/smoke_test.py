#!/usr/bin/env python3
"""Smoke test de prontidão da Fase 0 do PhysDual-GNN (Passo 5).

Verifica os 8 critérios de aceitação da Fase 0 com status OK / FALHA / PULADO:
1. Imports de todas as bibliotecas principais e versões.
2. PyTorch enxerga a GPU (ou avisa que está em CPU).
3. RDKit: 1 SMILES (aspirina) -> grafo PyG (x, edge_index) com featurização atômica mínima.
4. PyG: uma camada GINConv roda forward + backward nesse grafo.
5. ESM-2: carrega apenas `esm2_t6_8M_UR50D`, gera embedding de sequência curta, salva e recarrega do cache.
6. `set_seed` produz saídas idênticas em duas execuções.
7. (Opcional) PyTDC lista/consulta Davis sem travar o restante em caso de ausência/falha.
8. Persistência: grava arquivo de teste na raiz do projeto (Drive no Colab) e confirma leitura.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import random
import sys
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from physdual.utils.config import load_config  # noqa: E402
from physdual.utils.paths import ensure_project_dirs, get_project_root, is_colab  # noqa: E402
from physdual.utils.seed import set_seed  # noqa: E402

ASPIRIN_SMILES = "CC(=O)OC1=CC=CC=C1C(=O)O"
SHORT_PROTEIN_SEQ = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"


@dataclass
class CheckResult:
    """Resultado individual de um item do smoke test."""

    item: int
    name: str
    status: str  # "OK" | "FALHA" | "PULADO"
    detail: str


@contextlib.contextmanager
def _scoped_esm_weights_load() -> Iterator[None]:
    """Contorno estritamente escopado para `weights_only=True` no `fair-esm` em PyTorch >= 2.6.

    Os checkpoints oficiais do `fair-esm` serializam um objeto `argparse.Namespace` com os
    hiperparâmetros do modelo. Em versões recentes do PyTorch, `torch.load` ativa
    `weights_only=True` por padrão. Este gerenciador adiciona `argparse.Namespace` aos
    globais seguros da chamada e, caso o `load_regression_hub` interno do `fair-esm` ainda
    exija `weights_only=False`, aplica um wrapper temporário restrito exclusivamente ao
    bloco `with`, restaurando `torch.load` imediatamente no `finally`.
    """
    import torch

    orig_load = torch.load

    def _patched_load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return orig_load(*args, **kwargs)

    safe_ctx = (
        torch.serialization.safe_globals([argparse.Namespace])
        if hasattr(torch.serialization, "safe_globals")
        else contextlib.nullcontext()
    )
    with safe_ctx:
        torch.load = _patched_load  # type: ignore[assignment]
        try:
            yield
        finally:
            torch.load = orig_load  # type: ignore[assignment]


def smiles_to_minimal_pyg_graph(smiles: str) -> Any:
    """Converte um SMILES em um objeto `torch_geometric.data.Data` com featurização atômica mínima.

    Features por átomo (dim=5):
        [atomic_num, total_degree, formal_charge, total_num_hs, is_aromatic]
    Arestas:
        Grafo direcionado simétrico (duas arestas dirigidas por ligação covalente).
    """
    import torch
    from rdkit import Chem
    from torch_geometric.data import Data

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"SMILES inválido para o RDKit: {smiles}")

    atom_features: list[list[float]] = []
    for atom in mol.GetAtoms():
        atom_features.append(
            [
                float(atom.GetAtomicNum()),
                float(atom.GetTotalDegree()),
                float(atom.GetFormalCharge()),
                float(atom.GetTotalNumHs()),
                1.0 if atom.GetIsAromatic() else 0.0,
            ]
        )

    edges: list[tuple[int, int]] = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append((i, j))
        edges.append((j, i))

    x = torch.tensor(atom_features, dtype=torch.float32)
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, smiles=smiles)


def run_gin_forward_backward(data: Any, hidden_dim: int = 16) -> float:
    """Executa 1 passo forward + backward de uma camada `GINConv` sobre um grafo PyG."""
    import torch
    from torch import nn
    from torch_geometric.nn import GINConv, global_mean_pool

    set_seed(42)
    in_dim = int(data.x.size(-1))
    mlp = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )
    conv = GINConv(mlp)
    head = nn.Linear(hidden_dim, 1)

    batch = torch.zeros(data.x.size(0), dtype=torch.long)
    h = conv(data.x, data.edge_index)
    pooled = global_mean_pool(h, batch)
    pred = head(pooled).squeeze(-1)
    target = torch.tensor([7.5], dtype=torch.float32)
    loss = nn.functional.mse_loss(pred, target)
    loss.backward()

    grad_norm = sum(
        float(p.grad.norm().item())
        for p in list(conv.parameters()) + list(head.parameters())
        if p.grad is not None
    )
    if grad_norm <= 0.0:
        raise RuntimeError("Gradiente nulo após backward na GINConv.")
    return float(loss.item())


def check_1_imports() -> CheckResult:
    """1. Verifica imports de todas as bibliotecas essenciais e suas versões."""
    required = [
        ("torch", "torch"),
        ("torch_geometric", "torch_geometric"),
        ("rdkit", "rdkit"),
        ("fair-esm", "esm"),
        ("biopython", "Bio"),
        ("pyyaml", "yaml"),
        ("tqdm", "tqdm"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("scikit-learn", "sklearn"),
    ]
    optional = [
        ("lifelines", "lifelines"),
        ("tensorboard", "tensorboard"),
        ("PyTDC", "tdc"),
    ]

    found: list[str] = []
    missing_req: list[str] = []
    for label, mod_name in required:
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", "instalado")
            found.append(f"{label}={ver}")
        except ImportError:
            missing_req.append(label)

    opt_status: list[str] = []
    for label, mod_name in optional:
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", "instalado")
            opt_status.append(f"{label}={ver}")
        except ImportError:
            opt_status.append(f"{label}=ausente(opcional)")

    if missing_req:
        return CheckResult(
            1,
            "Imports e versões",
            "FALHA",
            f"Pacotes obrigatórios ausentes: {', '.join(missing_req)} | Presentes: {', '.join(found) or 'nenhum'}",
        )
    return CheckResult(
        1,
        "Imports e versões",
        "OK",
        f"Obrigatórios: {', '.join(found)} | Opcionais: {', '.join(opt_status)}",
    )


def check_2_gpu() -> CheckResult:
    """2. Verifica se o PyTorch enxerga a GPU (ou avisa que está em CPU)."""
    try:
        import torch
    except ImportError:
        return CheckResult(2, "Detecção de GPU (PyTorch)", "PULADO", "PyTorch não instalado.")

    if torch.cuda.is_available():
        dev_name = torch.cuda.get_device_name(0)
        vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        return CheckResult(
            2,
            "Detecção de GPU (PyTorch)",
            "OK",
            f"GPU ativa: {dev_name} ({vram_gb} GiB VRAM, CUDA {torch.version.cuda})",
        )
    return CheckResult(
        2,
        "Detecção de GPU (PyTorch)",
        "OK",
        "AVISO: Rodando em CPU (torch.cuda.is_available() == False). Ative GPU T4/L4 no Colab para treino/ESM-2.",
    )


def check_3_rdkit_to_pyg() -> tuple[CheckResult, Any | None]:
    """3. Converte SMILES da aspirina em grafo PyG (x, edge_index)."""
    try:
        data = smiles_to_minimal_pyg_graph(ASPIRIN_SMILES)
        num_nodes = int(data.x.shape[0])
        num_feat = int(data.x.shape[1])
        num_edges = int(data.edge_index.shape[1])
        if num_nodes != 13 or num_edges != 26:
            return (
                CheckResult(
                    3,
                    "RDKit SMILES -> Grafo PyG",
                    "FALHA",
                    f"Dimensões inesperadas para aspirina: x={tuple(data.x.shape)}, edge_index={tuple(data.edge_index.shape)}",
                ),
                None,
            )
        return (
            CheckResult(
                3,
                "RDKit SMILES -> Grafo PyG",
                "OK",
                f"Aspirina ({ASPIRIN_SMILES}) -> x=({num_nodes}, {num_feat}), edge_index=(2, {num_edges})",
            ),
            data,
        )
    except ImportError as exc:
        return (
            CheckResult(3, "RDKit SMILES -> Grafo PyG", "FALHA", f"Dependência ausente: {exc}"),
            None,
        )
    except Exception as exc:
        return CheckResult(3, "RDKit SMILES -> Grafo PyG", "FALHA", f"Erro: {exc}"), None


def check_4_gin_conv(graph_data: Any | None) -> CheckResult:
    """4. Roda forward + backward de uma GINConv no grafo da aspirina."""
    if graph_data is None:
        return CheckResult(
            4,
            "PyG GINConv forward+backward",
            "PULADO",
            "Grafo do item 3 indisponível.",
        )
    try:
        loss_val = run_gin_forward_backward(graph_data)
        return CheckResult(
            4,
            "PyG GINConv forward+backward",
            "OK",
            f"Forward + backward concluídos com sucesso (loss={loss_val:.4f}, gradientes verificados).",
        )
    except Exception as exc:
        return CheckResult(4, "PyG GINConv forward+backward", "FALHA", f"Erro: {exc}")


def check_5_esm2_cache(cache_dir: Path) -> CheckResult:
    """5. Carrega `esm2_t6_8M_UR50D`, extrai embedding de sequência curta, salva e recarrega do cache."""
    try:
        import esm  # type: ignore[import-untyped]
        import torch
    except ImportError as exc:
        return CheckResult(
            5,
            "ESM-2 (esm2_t6_8M_UR50D) + Cache",
            "FALHA",
            f"Dependência ausente ({exc}).",
        )

    try:
        with _scoped_esm_weights_load():
            model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as net_exc:
        return CheckResult(
            5,
            "ESM-2 (esm2_t6_8M_UR50D) + Cache",
            "PULADO",
            f"Sem acesso à internet para baixar os pesos do esm2_t6_8M_UR50D ({net_exc}).",
        )
    except Exception as exc:
        return CheckResult(
            5,
            "ESM-2 (esm2_t6_8M_UR50D) + Cache",
            "FALHA",
            f"Falha ao carregar esm2_t6_8M_UR50D: {exc}",
        )

    try:
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        batch_converter = alphabet.get_batch_converter()
        _, _, batch_tokens = batch_converter([("protein_smoke", SHORT_PROTEIN_SEQ)])
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[6], return_contacts=False)
        token_repr = results["representations"][6]
        # Remove tokens especiais <cls> (índice 0) e <eos> (índice L+1)
        residue_emb = token_repr[0, 1 : len(SHORT_PROTEIN_SEQ) + 1].detach().cpu()

        seq_hash = hashlib.sha256(SHORT_PROTEIN_SEQ.encode("utf-8")).hexdigest()[:16]
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{seq_hash}.pt"

        payload = {
            "seq_hash": seq_hash,
            "model_name": "esm2_t6_8M_UR50D",
            "seq_len": len(SHORT_PROTEIN_SEQ),
            "embedding": residue_emb,
        }
        torch.save(payload, cache_file)
        reloaded = torch.load(cache_file, map_location="cpu", weights_only=True)

        if not torch.allclose(residue_emb, reloaded["embedding"], atol=1e-6):
            return CheckResult(
                5,
                "ESM-2 (esm2_t6_8M_UR50D) + Cache",
                "FALHA",
                "Embedding recarregado do cache diverge do tensor original.",
            )

        return CheckResult(
            5,
            "ESM-2 (esm2_t6_8M_UR50D) + Cache",
            "OK",
            f"shape={tuple(residue_emb.shape)}, device={device.type}, salvo e recarregado de {cache_file}",
        )
    except Exception as exc:
        return CheckResult(
            5,
            "ESM-2 (esm2_t6_8M_UR50D) + Cache",
            "FALHA",
            f"Erro na inferência/cache do ESM-2: {exc}",
        )


def check_6_seed_determinism() -> CheckResult:
    """6. Verifica se `set_seed` produz saídas idênticas em duas execuções consecutivas."""
    try:
        set_seed(12345)
        py_a = [random.random() for _ in range(5)]

        set_seed(12345)
        py_b = [random.random() for _ in range(5)]

        if py_a != py_b:
            return CheckResult(6, "Determinismo de set_seed", "FALHA", "random.random() divergiu.")

        details = ["random=OK"]

        try:
            import numpy as np

            set_seed(12345)
            np_a = np.random.randn(8)
            set_seed(12345)
            np_b = np.random.randn(8)
            if not np.allclose(np_a, np_b):
                return CheckResult(6, "Determinismo de set_seed", "FALHA", "numpy.random divergiu.")
            details.append("numpy=OK")
        except ImportError:
            details.append("numpy=ausente")

        try:
            import torch

            set_seed(12345)
            t_a = torch.randn(8)
            set_seed(12345)
            t_b = torch.randn(8)
            if not torch.equal(t_a, t_b):
                return CheckResult(6, "Determinismo de set_seed", "FALHA", "torch.randn CPU divergiu.")
            details.append("torch_cpu=OK")

            if torch.cuda.is_available():
                set_seed(12345)
                c_a = torch.randn(8, device="cuda")
                set_seed(12345)
                c_b = torch.randn(8, device="cuda")
                if not torch.equal(c_a, c_b):
                    return CheckResult(6, "Determinismo de set_seed", "FALHA", "torch.randn CUDA divergiu.")
                details.append("torch_cuda=OK")
        except ImportError:
            details.append("torch=ausente")

        return CheckResult(
            6,
            "Determinismo de set_seed",
            "OK",
            f"Saídas idênticas em duas execuções ({', '.join(details)}).",
        )
    except Exception as exc:
        return CheckResult(6, "Determinismo de set_seed", "FALHA", f"Erro: {exc}")


def check_7_tdc_optional() -> CheckResult:
    """7. (Opcional) Verifica se o PyTDC está disponível e lista/consulta metadados do Davis."""
    try:
        import tdc  # type: ignore[import-untyped]
        from tdc.multi_pred import DTI  # type: ignore[import-untyped]

        _ = DTI
        ver = getattr(tdc, "__version__", "instalado")
        return CheckResult(
            7,
            "PyTDC (opcional)",
            "OK",
            f"PyTDC ({ver}) importado com sucesso (classe tdc.multi_pred.DTI disponível para Davis/KIBA).",
        )
    except ImportError:
        return CheckResult(
            7,
            "PyTDC (opcional)",
            "PULADO",
            "PyTDC não instalado (opcional). Na Fase 1 pode-se usar download direto de Davis/KIBA (DeepDTA).",
        )
    except Exception as exc:
        return CheckResult(
            7,
            "PyTDC (opcional)",
            "PULADO",
            f"PyTDC apresentou incompatibilidade não-bloqueante ({exc}). Usar download direto na Fase 1.",
        )


def check_8_persistence(root_dir: Path) -> CheckResult:
    """8. Grava um arquivo de teste na raiz do projeto (Drive no Colab) e confirma sua existência."""
    try:
        out_dir = root_dir / "outputs" / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        probe_file = out_dir / "_smoke_test_persistence_probe.txt"
        expected_text = f"physdual-gnn persistence ok | colab={is_colab()} | root={root_dir}\n"
        probe_file.write_text(expected_text, encoding="utf-8")

        if not probe_file.exists():
            return CheckResult(
                8,
                "Persistência em Disco/Drive",
                "FALHA",
                f"Arquivo não encontrado após escrita: {probe_file}",
            )

        read_back = probe_file.read_text(encoding="utf-8")
        if read_back != expected_text:
            return CheckResult(
                8,
                "Persistência em Disco/Drive",
                "FALHA",
                "Conteúdo lido diverge do conteúdo gravado.",
            )

        return CheckResult(
            8,
            "Persistência em Disco/Drive",
            "OK",
            f"Arquivo gravado e verificado em: {probe_file}",
        )
    except Exception as exc:
        return CheckResult(8, "Persistência em Disco/Drive", "FALHA", f"Erro de escrita/leitura: {exc}")


def run_all_checks() -> list[CheckResult]:
    """Executa todos os 8 itens do smoke test em sequência."""
    cfg = load_config()
    root_dir = get_project_root()
    ensure_project_dirs(root_dir)
    esm_cache_dir = Path(cfg["paths"]["esm_cache_dir"])

    results: list[CheckResult] = []
    results.append(check_1_imports())
    results.append(check_2_gpu())
    res3, graph_data = check_3_rdkit_to_pyg()
    results.append(res3)
    results.append(check_4_gin_conv(graph_data))
    results.append(check_5_esm2_cache(esm_cache_dir))
    results.append(check_6_seed_determinism())
    results.append(check_7_tdc_optional())
    results.append(check_8_persistence(root_dir))
    return results


def main() -> int:
    """Ponto de entrada CLI do smoke test."""
    print("============================================================")
    print("PhysDual-GNN — Smoke Test (Fase 0)")
    print(f"Raiz ativa do projeto: {get_project_root()} (Colab={is_colab()})")
    print("============================================================")

    results = run_all_checks()
    failures = 0
    for r in results:
        print(f"[{r.status:^6}] Item {r.item}: {r.name} — {r.detail}")
        if r.status == "FALHA":
            failures += 1

    print("------------------------------------------------------------")
    ok_count = sum(1 for r in results if r.status == "OK")
    skip_count = sum(1 for r in results if r.status == "PULADO")
    print(f"Resumo: {ok_count} OK | {skip_count} PULADO | {failures} FALHA")
    print("============================================================")
    return 1 if failures > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
