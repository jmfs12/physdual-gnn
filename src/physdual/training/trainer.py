"""Módulo de treinamento, parada antecipada e avaliação de modelos para PhysDual-GNN.

Implementa:
- Isolamento estrito do conjunto de teste (C1): o test_loader nunca é consultado durante as épocas.
- Ajuste dinâmico de paciência para regimes cold no Davis (mínimo 80 épocas).
- Early stopping com margem delta explícita.
- Monitoramento configurável (min/max) de métricas de regressão DTA.
- Suporte a agendador de taxa de aprendizado ReduceLROnPlateau opcional.
- Salvamento automático de checkpoint best e last com suporte a espelhamento no Drive.
- Detecção e alerta de proximidade do teto de épocas (> 0.8 * epochs).
- Avaliação final única de teste sobre o melhor checkpoint restaurado.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer

from physdual.eval.metrics import (
    compute_all_metrics,
    compute_metrics_by_seq_length,
    compute_uncensored_metrics,
)
from physdual.training.checkpoint import (
    load_checkpoint,
    save_checkpoint,
    sync_checkpoint_to_drive,
)


def get_monitor_mode(monitor: str) -> Literal["min", "max"]:
    """Resolve o critério de otimização (minimização ou maximização) da métrica monitorada.

    Args:
        monitor: Nome da métrica (ex: 'mse', 'rmse', 'loss', 'ci', 'pearson', 'spearman', 'rm2').

    Returns:
        'min' para métricas de erro ou 'max' para métricas de correlação/concordância.

    Raises:
        ValueError: Se a métrica de monitoramento for desconhecida.
    """
    key = monitor.strip().lower()
    if key in ("mse", "rmse", "loss"):
        return "min"
    if key in ("ci", "pearson", "spearman", "rm2"):
        return "max"

    raise ValueError(
        f"Métrica de monitoramento desconhecida: '{monitor}'. "
        "Métricas suportadas: mse, rmse, loss, ci, pearson, spearman, rm2."
    )


class EarlyStopping:
    """Controlador de parada antecipada com tolerância delta explícita."""

    def __init__(
        self,
        patience: int = 50,
        mode: Literal["min", "max"] = "min",
        delta: float = 1e-4,
    ) -> None:
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter: int = 0
        self.best: float = float("inf") if mode == "min" else float("-inf")
        self.should_stop: bool = False

    def step(self, metric: float) -> bool:
        """Atualiza o estado do early stopping com a métrica da época atual.

        Regra de delta explícita:
        - Para mode == 'min': melhorou se `metric < self.best - self.delta`.
        - Para mode == 'max': melhorou se `metric > self.best + self.delta`.

        Args:
            metric: Valor numérico da métrica observada.

        Returns:
            True se a paciência foi esgotada e o treinamento deve parar, False caso contrário.
        """
        if self.mode == "min":
            improved = metric < (self.best - self.delta)
        else:
            improved = metric > (self.best + self.delta)

        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True

        return self.should_stop


def _forward_batch(
    model: nn.Module,
    batch: Any,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Encaminha o lote através do modelo, suportando formatos de Batch PyG e dicionários."""
    if isinstance(batch, dict):
        if "drug_batch" in batch and "target_tokens" in batch:
            drug_batch = batch["drug_batch"].to(device)
            target_tokens = batch["target_tokens"].to(device)
            out = model(drug_batch, target_tokens)
        else:
            device_batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            try:
                out = model(device_batch)
            except TypeError:
                out = model(**device_batch)

        y = batch.get("affinity", batch.get("y"))
        return out, y

    if hasattr(batch, "affinity"):
        batch = batch.to(device)
        out = model(batch)
        return out, batch.affinity

    if hasattr(batch, "y"):
        batch = batch.to(device)
        out = model(batch)
        return out, batch.y

    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        x, y = batch[0], batch[1]
        if hasattr(x, "to"):
            x = x.to(device)
        out = model(x)
        return out, y

    if hasattr(batch, "to"):
        batch = batch.to(device)
    out = model(batch)
    return out, getattr(batch, "y", None)


def train_one_epoch(
    model: nn.Module,
    loader: Any,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: str | torch.device,
) -> float:
    """Executa uma época completa de treinamento otimizando os pesos do modelo.

    Args:
        model: Módulo PyTorch a ser treinado.
        loader: DataLoader de treinamento.
        optimizer: Otimizador configurado.
        criterion: Função de perda (ex: MSELoss).
        device: Dispositivo de execução ('cpu' ou 'cuda').

    Returns:
        Média da perda observada durante a época.
    """
    model.train()
    target_device = torch.device(device)
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        optimizer.zero_grad()
        out, y = _forward_batch(model, batch, target_device)

        if y is None:
            raise ValueError("O lote não forneceu rótulo de afinidade/alvo ('affinity' ou 'y').")

        target = y.to(target_device).float()

        # Ajusta shapes para evitar avisos ou inconsistências no broadcasting de perda
        if out.ndim == 1 and target.ndim == 2 and target.shape[1] == 1:
            target = target.view(-1)
        elif out.ndim == 2 and out.shape[1] == 1 and target.ndim == 1:
            out = out.view(-1)
        elif out.shape != target.shape and out.numel() == target.numel():
            target = target.view_as(out)

        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1

    return total_loss / max(num_batches, 1)


def evaluate(
    model: nn.Module,
    loader: Any,
    device: str | torch.device,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Avalia o modelo sem gradientes, coletando predições, valores reais e metadados.

    Args:
        model: Módulo PyTorch a ser avaliado.
        loader: DataLoader de validação ou teste.
        device: Dispositivo de execução ('cpu' ou 'cuda').

    Returns:
        Tupla com:
        - y_pred: Array NumPy 1D com predições.
        - y_true: Array NumPy 1D com afinidades reais.
        - metadata_list: Lista de dicionários por amostra preservando campos como
          target_id, sequence_length, uncensored_mask e affinity.
    """
    model.eval()
    target_device = torch.device(device)

    all_preds: list[np.ndarray] = []
    all_trues: list[np.ndarray] = []
    all_metadata: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            out, y = _forward_batch(model, batch, target_device)

            if y is None:
                raise ValueError("O lote não forneceu rótulo de afinidade/alvo ('affinity' ou 'y').")

            batch_preds = out.view(-1).detach().cpu().numpy().astype(float)
            batch_trues = y.view(-1).detach().cpu().numpy().astype(float)

            all_preds.append(batch_preds)
            all_trues.append(batch_trues)

            batch_size = len(batch_preds)

            if isinstance(batch, dict):
                t_ids = batch.get("target_id")
                d_ids = batch.get("drug_id")
                smiles_list = batch.get("smiles")
                seq_list = batch.get("sequence")
                seq_lens = batch.get("sequence_length", batch.get("seq_lengths"))
                uncensored = batch.get("uncensored_mask")

                for i in range(batch_size):
                    meta_item: dict[str, Any] = {"affinity": float(batch_trues[i])}
                    if t_ids is not None:
                        meta_item["target_id"] = t_ids[i] if isinstance(t_ids, (list, tuple)) or (isinstance(t_ids, np.ndarray) and t_ids.ndim > 0) else t_ids
                    if d_ids is not None:
                        meta_item["drug_id"] = d_ids[i] if isinstance(d_ids, (list, tuple)) or (isinstance(d_ids, np.ndarray) and d_ids.ndim > 0) else d_ids
                    if smiles_list is not None:
                        meta_item["smiles"] = smiles_list[i] if isinstance(smiles_list, (list, tuple)) else smiles_list
                    if seq_list is not None:
                        meta_item["sequence"] = seq_list[i] if isinstance(seq_list, (list, tuple)) else seq_list
                    if seq_lens is not None:
                        if hasattr(seq_lens, "__getitem__"):
                            val = seq_lens[i]
                            meta_item["sequence_length"] = int(val.item() if hasattr(val, "item") else val)
                        else:
                            meta_item["sequence_length"] = int(seq_lens.item() if hasattr(seq_lens, "item") else seq_lens)
                    if uncensored is not None:
                        val = uncensored[i]
                        meta_item["uncensored_mask"] = bool(val.item() if hasattr(val, "item") else val)
                    all_metadata.append(meta_item)
            else:
                for i in range(batch_size):
                    all_metadata.append({"affinity": float(batch_trues[i])})

    y_pred = np.concatenate(all_preds, axis=0) if all_preds else np.array([], dtype=float)
    y_true = np.concatenate(all_trues, axis=0) if all_trues else np.array([], dtype=float)

    return y_pred, y_true, all_metadata


def train_model(
    model: nn.Module,
    train_loader: Any,
    val_loader: Any,
    test_loader: Any | None = None,
    epochs: int = 500,
    lr: float = 5e-4,
    weight_decay: float = 0.0,
    patience: int = 50,
    davis_cold_patience: int = 80,
    monitor: str = "mse",
    checkpoint_dir: Path | str = "checkpoints",
    device: str | torch.device = "cpu",
    use_lr_scheduler: bool = False,
    dataset_name: str = "",
    split_type: str = "",
    save_last_every: int = 10,
    drive_checkpoint_dir: Path | str | None = None,
    delta: float = 1e-4,
    resume: bool = True,
) -> dict[str, Any]:
    """Orquestra o treinamento com parada antecipada, atomicidade e isolamento estrito de teste.

    Garantia C1 (Isolamento Estrito):
    Durante todo o loop de épocas, apenas o `val_loader` é consultado para monitoramento.
    O `test_loader` NUNCA é iterado ou avaliado dentro do loop de treinamento.

    Args:
        model: Rede neural PyTorch.
        train_loader: DataLoader do conjunto de treino.
        val_loader: DataLoader do conjunto de validação.
        test_loader: DataLoader opcional do conjunto de teste (avaliado apenas ao final).
        epochs: Limite máximo de épocas.
        lr: Taxa de aprendizado inicial para Adam.
        weight_decay: Fator de decaimento de peso L2.
        patience: Paciência base para early stopping.
        davis_cold_patience: Paciência mínima obrigatória para splits cold em Davis.
        monitor: Métrica monitorada para escolha do melhor modelo.
        checkpoint_dir: Diretório local onde os checkpoints são armazenados.
        device: Dispositivo de execução.
        use_lr_scheduler: Se True, aplica ReduceLROnPlateau sobre a métrica monitorada.
        dataset_name: Identificador do dataset ('davis', 'kiba', etc.).
        split_type: Tipo de partição ('random', 'cold_drug', 'cold_target', etc.).
        save_last_every: Frequência em épocas para salvar `checkpoint_last.pt`.
        drive_checkpoint_dir: Diretório de espelhamento do Drive opcional.
        delta: Margem mínima de melhora necessária para o early stopping.

    Returns:
        Dicionário com resumo completo da execução, checkpoints e métricas finais.
    """
    target_device = torch.device(device)
    model.to(target_device)

    # Regra de paciência: Davis cold requer no mínimo davis_cold_patience (80)
    effective_patience = patience
    if dataset_name.lower().strip() == "davis" and split_type.lower().strip() in ("cold_drug", "cold_target"):
        effective_patience = max(patience, davis_cold_patience)

    mode = get_monitor_mode(monitor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None
    if use_lr_scheduler:
        scheduler_patience = max(1, effective_patience // 2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=0.5,
            patience=scheduler_patience,
        )

    early_stopping = EarlyStopping(patience=effective_patience, mode=mode, delta=delta)

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "checkpoint_best.pt"
    last_path = ckpt_dir / "checkpoint_last.pt"

    drive_best_path: Path | None = None
    drive_last_path: Path | None = None
    if drive_checkpoint_dir is not None:
        drive_dir = Path(drive_checkpoint_dir)
        drive_dir.mkdir(parents=True, exist_ok=True)
        drive_best_path = drive_dir / "checkpoint_best.pt"
        drive_last_path = drive_dir / "checkpoint_last.pt"

    best_epoch = 0
    best_metric_val = float("inf") if mode == "min" else float("-inf")
    val_metrics_at_best: dict[str, float] = {}
    total_epochs_run = 0
    history: list[dict[str, float]] = []

    start_epoch = 1
    if resume and last_path.exists():
        try:
            last_ckpt = load_checkpoint(last_path, model=model, optimizer=optimizer, restore_rng=True)
            saved_epoch = int(last_ckpt.get("epoch", 0))
            if "history" in last_ckpt and isinstance(last_ckpt["history"], list):
                history = list(last_ckpt["history"])
            if saved_epoch >= 1:
                start_epoch = saved_epoch + 1
                total_epochs_run = saved_epoch
                print(f"[RESUME] Retomando treinamento a partir da época {start_epoch} (checkpoint: {last_path})")
                if best_path.exists():
                    best_ckpt = load_checkpoint(best_path)
                    best_epoch = int(best_ckpt.get("epoch", 0))
                    best_metric_val = float(best_ckpt.get("best_metric", best_metric_val))
                    val_metrics_at_best = dict(best_ckpt.get("val_metrics", {}))
                    early_stopping.best = best_metric_val
                    early_stopping.counter = max(0, saved_epoch - best_epoch)
        except Exception as exc:
            print(f"[WARN] Falha ao retomar checkpoint {last_path}: {exc}. Reiniciando da época 1.")
            start_epoch = 1
            history = []

    # -------------------------------------------------------------------------
    # Loop de Treinamento com Isolamento Estrito do Teste (C1)
    # test_loader NUNCA é iterado aqui dentro!
    # -------------------------------------------------------------------------
    for epoch in range(start_epoch, epochs + 1):
        total_epochs_run = epoch

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, target_device)
        val_pred, val_true, _ = evaluate(model, val_loader, target_device)
        val_metrics = compute_all_metrics(val_true, val_pred)
        current_metric = float(val_metrics[monitor])

        history.append({
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_mse": float(val_metrics.get("mse", 0.0)),
            "val_ci": float(val_metrics.get("ci", 0.0)),
        })

        if scheduler is not None:
            scheduler.step(current_metric)

        # Avalia se a métrica de validação melhorou
        if mode == "min":
            improved = current_metric < (best_metric_val - delta)
        else:
            improved = current_metric > (best_metric_val + delta)

        if improved:
            best_metric_val = current_metric
            best_epoch = epoch
            val_metrics_at_best = dict(val_metrics)

            best_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "best_metric": best_metric_val,
                "monitor": monitor,
                "history": history,
            }
            save_checkpoint(best_state, best_path)
            if drive_best_path is not None:
                sync_checkpoint_to_drive(best_path, drive_best_path)

        if epoch % save_last_every == 0 or epoch == epochs:
            last_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "monitor": monitor,
                "history": history,
            }
            save_checkpoint(last_state, last_path)
            if drive_last_path is not None:
                sync_checkpoint_to_drive(last_path, drive_last_path)

        if early_stopping.step(current_metric):
            break

    # Verificação de proximidade do teto
    near_epoch_ceiling_warn = bool(best_epoch > 0.8 * epochs)
    if near_epoch_ceiling_warn:
        print(
            f"[WARN] best_epoch ({best_epoch}) > 0.8 * epochs ({epochs}): "
            "possível sub-treinamento ou paciência excessiva; o modelo ainda estava melhorando próximo ao teto."
        )

    # -------------------------------------------------------------------------
    # Avaliação Final: restauração do melhor checkpoint e avaliação única de teste
    # -------------------------------------------------------------------------
    if best_path.exists():
        load_checkpoint(best_path, model=model, restore_rng=False)

    test_metrics_full: dict[str, float] | None = None
    test_metrics_uncensored: dict[str, float | int] | None = None
    test_metrics_by_seq_len: dict[str, dict[str, float | int]] | None = None
    oof_predictions: list[dict[str, Any]] | None = None

    if test_loader is not None:
        test_pred, test_true, test_meta = evaluate(model, test_loader, target_device)
        test_metrics_full = compute_all_metrics(test_true, test_pred)

        if dataset_name.lower().strip() == "davis":
            test_metrics_uncensored = compute_uncensored_metrics(test_true, test_pred)

        if test_meta and "sequence_length" in test_meta[0]:
            seq_lens = [m["sequence_length"] for m in test_meta]
            test_metrics_by_seq_len = compute_metrics_by_seq_length(test_true, test_pred, seq_lens)

        oof_predictions = []
        for i in range(len(test_true)):
            meta = test_meta[i] if (test_meta and i < len(test_meta)) else {}
            oof_predictions.append({
                "drug_id": meta.get("drug_id", ""),
                "target_id": meta.get("target_id", ""),
                "smiles": meta.get("smiles", ""),
                "sequence": meta.get("sequence", ""),
                "y_true": float(test_true[i]),
                "y_pred": float(test_pred[i]),
            })

    return {
        "best_epoch": best_epoch,
        "total_epochs_run": total_epochs_run,
        "near_epoch_ceiling_warn": near_epoch_ceiling_warn,
        "val_metrics_at_best": val_metrics_at_best,
        "test_metrics_full": test_metrics_full,
        "test_metrics_uncensored": test_metrics_uncensored,
        "test_metrics_by_seq_len": test_metrics_by_seq_len,
        "best_checkpoint_path": str(best_path),
        "last_checkpoint_path": str(last_path),
        "history": history,
        "oof_predictions": oof_predictions,
    }
