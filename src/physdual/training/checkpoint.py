"""Módulo de persistência e restauração atômica de checkpoints para PhysDual-GNN.

Garante:
- Sanitização de estruturas de dados e escalares NumPy para primitivos Python puros.
- Preservação intacta de tensores PyTorch.
- Gravação atômica via substituição no mesmo diretório (evitando escrita corrompida).
- Captura e restauração de estados RNG de CPU e CUDA.
- Sincronização atômica em arquivos de destino no Google Drive.
- Carga segura e restrita com `weights_only=True`.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer


def _sanitize_state_dict(obj: Any) -> Any:
    """Converte recursivamente escalares NumPy para tipos nativos do Python.

    Preserva integralmente instâncias de `torch.Tensor`.
    """
    if isinstance(obj, torch.Tensor):
        return obj

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return obj.item()
        return [_sanitize_state_dict(x) for x in obj.tolist()]

    if isinstance(obj, dict):
        sanitized_dict: dict[Any, Any] = {}
        for k, v in obj.items():
            sanitized_key = _sanitize_state_dict(k)
            sanitized_dict[sanitized_key] = _sanitize_state_dict(v)
        return sanitized_dict

    if isinstance(obj, (list, tuple)):
        return [_sanitize_state_dict(x) for x in obj]

    return obj


def save_checkpoint(state: dict[str, Any], path: Path | str) -> Path:
    """Salva um checkpoint atomicamente com captura de RNG states.

    Escreve primeiro em um arquivo temporário exclusivo no mesmo diretório do
    arquivo final e substitui atomicamente com `os.replace`.

    Args:
        state: Dicionário de estado a ser persistido.
        path: Caminho de destino para o arquivo de checkpoint (.pt).

    Returns:
        Path absoluto/resolvido para o checkpoint salvo.
    """
    dest_path = Path(path).resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    sanitized_state = _sanitize_state_dict(state)
    if not isinstance(sanitized_state, dict):
        raise TypeError("O estado do checkpoint deve ser um dicionário.")

    # Captura e registra estados RNG se não estiverem presentes
    if "torch_rng_state" not in sanitized_state:
        sanitized_state["torch_rng_state"] = torch.get_rng_state()

    if torch.cuda.is_available() and "cuda_rng_state_all" not in sanitized_state:
        sanitized_state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()

    # OBRIGATÓRIO: arquivo temporário no mesmo diretório pai do destino
    tmp_path = dest_path.parent / f".tmp_{dest_path.name}_{os.getpid()}"

    try:
        torch.save(sanitized_state, tmp_path)
        os.replace(tmp_path, dest_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

    return dest_path


def sync_checkpoint_to_drive(local_ckpt: Path | str, drive_ckpt: Path | str) -> Path:
    """Sincroniza atomicamente um checkpoint local para o diretório no Google Drive.

    Copia primeiramente para um arquivo temporário no mesmo diretório de destino
    no Drive e aplica `os.replace` para assegurar atomicidade de filesystem.

    Args:
        local_ckpt: Caminho do arquivo de checkpoint local fonte.
        drive_ckpt: Caminho de destino no Google Drive.

    Returns:
        Path para o arquivo sincronizado no Google Drive.
    """
    src_path = Path(local_ckpt).resolve()
    dest_drive = Path(drive_ckpt).resolve()

    if not src_path.exists():
        raise FileNotFoundError(f"Checkpoint local fonte não encontrado: {src_path}")

    dest_drive.parent.mkdir(parents=True, exist_ok=True)

    tmp_drive = dest_drive.parent / f".tmp_{dest_drive.name}_{os.getpid()}"

    try:
        shutil.copyfile(src_path, tmp_drive)
        os.replace(tmp_drive, dest_drive)
    finally:
        if tmp_drive.exists():
            try:
                tmp_drive.unlink()
            except OSError:
                pass

    return dest_drive


def load_checkpoint(
    path: Path | str,
    model: nn.Module | None = None,
    optimizer: Optimizer | None = None,
    restore_rng: bool = False,
) -> dict[str, Any]:
    """Carrega um checkpoint persistido com pesos e RNG states opcionais.

    Utiliza estritamente `weights_only=True` para segurança e compatibilidade.

    Args:
        path: Caminho do checkpoint (.pt).
        model: Módulo PyTorch opcional para carregar pesos.
        optimizer: Otimizador opcional para restaurar estado interno.
        restore_rng: Se True, restaura os geradores de números aleatórios CPU/CUDA.

    Returns:
        Dicionário com o payload carregado do checkpoint.
    """
    ckpt_path = Path(path).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado em: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("O payload carregado do checkpoint deve ser um dicionário.")

    if model is not None and "model_state_dict" in payload:
        model.load_state_dict(payload["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    if restore_rng:
        if "torch_rng_state" in payload:
            torch.set_rng_state(payload["torch_rng_state"])
        if torch.cuda.is_available() and "cuda_rng_state_all" in payload:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])

    return payload
