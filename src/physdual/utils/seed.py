"""Controle determinístico de sementes pseudoaleatórias (Python, NumPy, PyTorch e CUDA)."""

from __future__ import annotations

import os
import random


def set_seed(seed: int, deterministic_cudnn: bool = True, benchmark_cudnn: bool = False) -> None:
    """Fixa a semente aleatória em todas as bibliotecas relevantes para reprodutibilidade.

    Cobre:
    - `PYTHONHASHSEED`
    - `random` (biblioteca padrão do Python)
    - `numpy.random` (se instalado)
    - `torch` CPU e todas as GPUs `torch.cuda` (se instalado)
    - `torch.backends.cudnn.deterministic` e `torch.backends.cudnn.benchmark`

    Nota sobre desempenho vs. determinismo:
        `deterministic_cudnn=True` e `benchmark_cudnn=False` garantem que convoluções
        em GPU selecionem algoritmos determinísticos, ao custo de pequena redução de
        throughput. Para experimentos científicos com múltiplas seeds, mantenha o padrão.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = deterministic_cudnn
            torch.backends.cudnn.benchmark = benchmark_cudnn
    except ImportError:
        pass
