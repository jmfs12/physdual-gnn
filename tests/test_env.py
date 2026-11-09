"""Testes automatizados de ambiente e utilitários da Fase 0 (`pytest` / `unittest`).

Cobre obrigatoriamente os itens 1, 3, 4 e 6 do smoke test, além da resolução de caminhos
relativos em `configs/base.yaml`. Testes que exigem GPU/Colab ou bibliotecas pesadas não
instaladas localmente são pulados de forma explícita fora do runtime alvo.
"""

from __future__ import annotations

import importlib
import random
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
for p in (SRC_DIR, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from physdual.utils.config import load_config  # noqa: E402
from physdual.utils.paths import get_project_root, is_colab  # noqa: E402
from physdual.utils.seed import set_seed  # noqa: E402
from smoke_test import (  # noqa: E402
    ASPIRIN_SMILES,
    check_1_imports,
    run_gin_forward_backward,
    smiles_to_minimal_pyg_graph,
)


def _has_module(name: str) -> bool:
    """Retorna True se o módulo puder ser importado."""
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


def _has_cuda() -> bool:
    """Retorna True se o PyTorch estiver instalado e enxergar GPU CUDA."""
    if not _has_module("torch"):
        return False
    import torch

    return bool(torch.cuda.is_available())


class TestPhase0Environment(unittest.TestCase):
    """Suíte de verificação da Fase 0 compatível com `pytest` e `unittest`."""

    def test_item1_imports_and_versions(self) -> None:
        """Item 1: Verifica imports das bibliotecas obrigatórias quando no ambiente ML/Colab."""
        res = check_1_imports()
        if res.status != "OK" and not is_colab():
            raise unittest.SkipTest(
                f"Ambiente local sem stack ML completa instalada ({res.detail}). "
                "No Colab este teste deve passar obrigatoriamente."
            )
        self.assertEqual(res.status, "OK", msg=res.detail)

    def test_item3_rdkit_smiles_to_pyg_graph(self) -> None:
        """Item 3: Converte SMILES da aspirina em grafo PyG (x, edge_index)."""
        if not (_has_module("rdkit") and _has_module("torch") and _has_module("torch_geometric")):
            raise unittest.SkipTest("RDKit, PyTorch ou PyG não instalados neste ambiente local.")

        graph = smiles_to_minimal_pyg_graph(ASPIRIN_SMILES)
        self.assertEqual(tuple(graph.x.shape), (13, 5))
        self.assertEqual(tuple(graph.edge_index.shape), (2, 26))

    def test_item4_pyg_ginconv_forward_backward(self) -> None:
        """Item 4: Executa forward + backward de uma GINConv sobre o grafo da aspirina."""
        if not (_has_module("rdkit") and _has_module("torch") and _has_module("torch_geometric")):
            raise unittest.SkipTest("RDKit, PyTorch ou PyG não instalados neste ambiente local.")

        graph = smiles_to_minimal_pyg_graph(ASPIRIN_SMILES)
        loss_val = run_gin_forward_backward(graph)
        self.assertGreater(loss_val, 0.0)

    def test_item6_set_seed_determinism(self) -> None:
        """Item 6: Verifica determinismo de `set_seed` em duas execuções seguidas."""
        set_seed(2026)
        seq_a = [random.random() for _ in range(10)]
        set_seed(2026)
        seq_b = [random.random() for _ in range(10)]
        self.assertEqual(seq_a, seq_b)

        if _has_module("numpy"):
            import numpy as np

            set_seed(2026)
            arr_a = np.random.randn(6)
            set_seed(2026)
            arr_b = np.random.randn(6)
            self.assertTrue(np.allclose(arr_a, arr_b))

        if _has_module("torch"):
            import torch

            set_seed(2026)
            ten_a = torch.randn(6)
            set_seed(2026)
            ten_b = torch.randn(6)
            self.assertTrue(torch.equal(ten_a, ten_b))

    def test_config_loads_and_rejects_absolute_paths(self) -> None:
        """Verifica carregamento do `configs/base.yaml` e bloqueio de caminhos absolutos."""
        cfg = load_config()
        self.assertEqual(cfg["seeds"], [0, 1, 2, 3, 4])
        self.assertEqual(cfg["splits"]["available"], ["random", "cold_drug", "cold_target"])
        self.assertEqual(cfg["datasets"]["available"], ["davis", "kiba"])
        self.assertEqual(cfg["esm"]["model_name"], "esm2_t33_650M_UR50D")
        self.assertEqual(cfg["esm"]["esm_dim"], 1280)
        self.assertEqual(
            cfg["esm"]["lightweight_alternative"]["model_name"], "esm2_t30_150M_UR50D"
        )
        self.assertEqual(cfg["esm"]["lightweight_alternative"]["esm_dim"], 640)

        for resolved in cfg["paths"].values():
            self.assertIsInstance(resolved, Path)
            self.assertTrue(resolved.is_absolute())

        with tempfile.TemporaryDirectory() as tmpdir:
            bad_yaml = Path(tmpdir) / "bad.yaml"
            bad_yaml.write_text(
                "paths:\n  data_dir: '/content/drive/MyDrive/physdual-gnn/data'\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_config(config_path=bad_yaml)

    def test_gpu_and_colab_runtime(self) -> None:
        """Teste específico de GPU/Colab (pulado automaticamente fora do Colab ou sem GPU)."""
        if not is_colab() and not _has_cuda():
            raise unittest.SkipTest("Fora do Google Colab e sem GPU CUDA local.")

        if _has_cuda():
            import torch

            x = torch.ones((4, 4), device="cuda")
            self.assertEqual(float(x.sum().item()), 16.0)

        if is_colab():
            root = get_project_root()
            self.assertTrue(str(root).startswith("/content"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
