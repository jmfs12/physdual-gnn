"""Testes unitários para o pipeline de treinamento e checkpointing atômico (Task 4 da Fase 2).

Cobre:
- Sanitização de tipos NumPy para tipos primitivos do Python com preservação de Tensores PyTorch.
- Salvamento atômico de checkpoints com captura de RNG states (CPU e CUDA) e remoção de temporários.
- Carregamento de checkpoint com weights_only=True e restauração de pesos e RNG.
- Sincronização atômica de checkpoints para o Google Drive.
- Lógica de EarlyStopping com regras de delta explícitas para modos min e max.
- Resolução de modo do monitor de validação (min/max) e tratamento de erros.
- Funções train_one_epoch e evaluate.
- Isolamento estrito do conjunto de teste (C1): test_loader NUNCA é avaliado dentro do loop de épocas.
- Ajuste dinâmico de paciência para splits cold em Davis (mínimo davis_cold_patience).
- Aviso de proximidade do teto de épocas (near_epoch_ceiling_warn quando best_epoch > 0.8 * epochs).
- Restauração automática do melhor checkpoint antes da avaliação final no conjunto de teste.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from physdual.training.checkpoint import (
    _sanitize_state_dict,
    load_checkpoint,
    save_checkpoint,
    sync_checkpoint_to_drive,
)
from physdual.training.trainer import (
    EarlyStopping,
    evaluate,
    get_monitor_mode,
    train_model,
    train_one_epoch,
)


class SimpleDictDataset(Dataset):
    """Dataset auxiliar que retorna dicionários contendo afinidade e tensores."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, seq_len: int = 500, target_id: str = "T1"):
        self.x = x
        self.y = y
        self.seq_len = seq_len
        self.target_id = target_id

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "x": self.x[idx],
            "affinity": self.y[idx],
            "target_id": self.target_id,
            "sequence_length": self.seq_len,
        }


class SimpleDictModel(nn.Module):
    """Modelo simples compatível com dicionários de entrada."""

    def __init__(self, in_features: int = 4):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, batch: Any) -> torch.Tensor:
        if isinstance(batch, dict):
            features = batch["x"]
        else:
            features = batch
        return self.linear(features).squeeze(-1)


# -------------------------------------------------------------------------
# Testes do Módulo de Checkpoint
# -------------------------------------------------------------------------


def test_sanitize_state_dict_converts_numpy_and_preserves_tensors():
    """Valida conversão recursiva de escalares NumPy preservando tensores PyTorch."""
    t = torch.randn(3, 4)
    raw_payload = {
        "epoch": np.int64(10),
        "loss": np.float32(0.25),
        "is_best": np.bool_(True),
        "zero_dim": np.array(99),
        "tensor_data": t,
        "nested_dict": {
            np.int32(1): np.float64(0.123),
            "lr": np.float32(1e-4),
        },
        "nested_list": [np.int32(5), np.float64(1.5), np.bool_(False)],
        "nested_tuple": (np.int64(7), "string_val"),
    }

    sanitized = _sanitize_state_dict(raw_payload)

    assert isinstance(sanitized["epoch"], int) and type(sanitized["epoch"]) is int
    assert isinstance(sanitized["loss"], float) and type(sanitized["loss"]) is float
    assert isinstance(sanitized["is_best"], bool) and type(sanitized["is_best"]) is bool
    assert isinstance(sanitized["zero_dim"], int) and type(sanitized["zero_dim"]) is int
    assert isinstance(sanitized["tensor_data"], torch.Tensor)
    assert torch.equal(sanitized["tensor_data"], t)

    assert isinstance(sanitized["nested_dict"][1], float)
    assert type(sanitized["nested_dict"][1]) is float
    assert type(sanitized["nested_dict"]["lr"]) is float

    assert isinstance(sanitized["nested_list"], list)
    assert type(sanitized["nested_list"][0]) is int
    assert type(sanitized["nested_list"][1]) is float
    assert type(sanitized["nested_list"][2]) is bool

    assert isinstance(sanitized["nested_tuple"], list)
    assert type(sanitized["nested_tuple"][0]) is int
    assert sanitized["nested_tuple"][1] == "string_val"


def test_save_and_load_checkpoint_atomic(tmp_path: Path):
    """Valida gravação atômica, injeção de RNG states e restauração com weights_only=True."""
    ckpt_file = tmp_path / "models" / "checkpoint_best.pt"
    model = nn.Linear(4, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    initial_bias = model.bias.detach().clone()

    state = {
        "epoch": np.int64(5),
        "val_loss": np.float32(0.42),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    saved_path = save_checkpoint(state, ckpt_file)
    assert saved_path == ckpt_file
    assert ckpt_file.exists()

    # Garante que nenhum arquivo temporário ficou no diretório
    parent_files = list(ckpt_file.parent.iterdir())
    assert len(parent_files) == 1
    assert parent_files[0].name == "checkpoint_best.pt"

    # Altera parâmetros do modelo e otimizador
    with torch.no_grad():
        model.bias.add_(10.0)
    assert not torch.equal(model.bias, initial_bias)

    # Carrega com restauração
    loaded = load_checkpoint(ckpt_file, model=model, optimizer=optimizer, restore_rng=True)
    assert loaded["epoch"] == 5
    assert type(loaded["epoch"]) is int
    assert loaded["val_loss"] == pytest.approx(0.42)
    assert type(loaded["val_loss"]) is float
    assert "torch_rng_state" in loaded
    assert torch.equal(model.bias, initial_bias)


def test_sync_checkpoint_to_drive_atomic(tmp_path: Path):
    """Valida sincronização atômica para pasta do Drive sem deixar temporários."""
    local_dir = tmp_path / "local"
    drive_dir = tmp_path / "drive"
    local_dir.mkdir()
    drive_dir.mkdir()

    local_ckpt = local_dir / "checkpoint_best.pt"
    save_checkpoint({"epoch": 1, "score": 0.8}, local_ckpt)

    drive_ckpt = drive_dir / "checkpoint_best.pt"
    synced_path = sync_checkpoint_to_drive(local_ckpt, drive_ckpt)

    assert synced_path == drive_ckpt
    assert drive_ckpt.exists()
    assert drive_ckpt.stat().st_size == local_ckpt.stat().st_size

    # Confere se nenhum resíduo temporário permaneceu no Drive
    drive_files = list(drive_dir.iterdir())
    assert len(drive_files) == 1
    assert drive_files[0].name == "checkpoint_best.pt"


# -------------------------------------------------------------------------
# Testes do Módulo de Treinador e Early Stopping
# -------------------------------------------------------------------------


def test_get_monitor_mode():
    """Valida resolução de modo min/max e rejeição de métricas inválidas."""
    assert get_monitor_mode("mse") == "min"
    assert get_monitor_mode("rmse") == "min"
    assert get_monitor_mode("loss") == "min"
    assert get_monitor_mode("MSE") == "min"

    assert get_monitor_mode("ci") == "max"
    assert get_monitor_mode("pearson") == "max"
    assert get_monitor_mode("spearman") == "max"
    assert get_monitor_mode("rm2") == "max"
    assert get_monitor_mode("CI") == "max"

    with pytest.raises(ValueError, match="Métrica de monitoramento desconhecida"):
        get_monitor_mode("accuracy")


def test_early_stopping_min_and_max_mode():
    """Valida contador e critério de parada com delta explícito."""
    # Modo MIN com delta 0.01
    es_min = EarlyStopping(patience=3, mode="min", delta=0.01)
    assert not es_min.step(1.0)  # Melhoria inicial (best=1.0)
    assert es_min.counter == 0

    assert not es_min.step(0.995)  # Melhora de apenas 0.005 <= delta (0.01) -> não melhora!
    assert es_min.counter == 1

    assert not es_min.step(0.98)  # Melhora de 0.02 > delta (0.01) -> melhora!
    assert es_min.counter == 0
    assert es_min.best == pytest.approx(0.98)

    assert not es_min.step(0.98)  # Igual -> counter=1
    assert es_min.counter == 1

    assert not es_min.step(0.99)  # Piorou -> counter=2
    assert es_min.counter == 2

    should_stop = es_min.step(1.0)  # Piorou -> counter=3 >= patience (3)
    assert es_min.counter == 3
    assert should_stop is True

    # Modo MAX com delta 0.05
    es_max = EarlyStopping(patience=2, mode="max", delta=0.05)
    assert not es_max.step(0.5)  # Melhoria inicial
    assert es_max.counter == 0

    assert not es_max.step(0.53)  # Melhora de 0.03 <= delta -> não melhora
    assert es_max.counter == 1

    assert not es_max.step(0.60)  # Melhora de 0.10 > delta -> melhora
    assert es_max.counter == 0
    assert es_max.best == pytest.approx(0.60)

    assert not es_max.step(0.58)  # Piorou -> counter=1
    assert es_max.counter == 1

    assert es_max.step(0.57)  # Piorou -> counter=2 >= patience -> parada!
    assert es_max.counter == 2


def test_train_one_epoch_and_evaluate():
    """Valida iteração de treinamento de 1 época e avaliação estruturada."""
    torch.manual_seed(42)
    x = torch.randn(20, 4)
    y = torch.randn(20)
    dataset = SimpleDictDataset(x, y, seq_len=800, target_id="T_TEST")
    loader = DataLoader(dataset, batch_size=5, shuffle=False)

    model = SimpleDictModel(in_features=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.MSELoss()

    loss_1 = train_one_epoch(model, loader, optimizer, criterion, device="cpu")
    assert isinstance(loss_1, float)
    assert loss_1 > 0.0

    loss_2 = train_one_epoch(model, loader, optimizer, criterion, device="cpu")
    assert isinstance(loss_2, float)
    assert loss_2 < loss_1  # Modelo deve estar aprendendo

    preds, trues, metadata = evaluate(model, loader, device="cpu")
    assert isinstance(preds, np.ndarray)
    assert isinstance(trues, np.ndarray)
    assert len(preds) == 20
    assert len(trues) == 20
    assert len(metadata) == 20
    assert metadata[0]["target_id"] == "T_TEST"
    assert metadata[0]["sequence_length"] == 800


def test_strict_test_isolation_guarantee(tmp_path: Path):
    """Garante isolamento estrito (C1): test_loader NUNCA é iterado durante o loop de treino."""
    torch.manual_seed(42)
    x_train, y_train = torch.randn(30, 4), torch.randn(30)
    x_val, y_val = torch.randn(10, 4), torch.randn(10)
    x_test, y_test = torch.randn(10, 4), torch.randn(10)

    train_loader = DataLoader(SimpleDictDataset(x_train, y_train), batch_size=10)
    val_loader = DataLoader(SimpleDictDataset(x_val, y_val), batch_size=10)

    class SpyLoader:
        """DataLoader espião que registra quando foi iterado e se o treino ainda estava rodando."""

        def __init__(self, dataset):
            self.dataset = dataset
            self.iter_count = 0
            self.iterated_during_epochs = False
            self.current_epoch_state = 0

        def __iter__(self):
            self.iter_count += 1
            if self.current_epoch_state > 0:
                self.iterated_during_epochs = True
            loader = DataLoader(self.dataset, batch_size=10)
            return iter(loader)

        def __len__(self):
            return len(self.dataset) // 10

    test_spy_loader = SpyLoader(SimpleDictDataset(x_test, y_test))

    model = SimpleDictModel(in_features=4)

    # Executa por 3 épocas
    result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_spy_loader,
        epochs=3,
        checkpoint_dir=tmp_path / "checkpoints",
        monitor="mse",
    )

    # Verificações estritas
    assert test_spy_loader.iter_count == 1, (
        f"test_loader deve ser iterado EXATAMENTE 1 vez após o treino, mas foi iterado {test_spy_loader.iter_count} vezes!"
    )
    assert not test_spy_loader.iterated_during_epochs, (
        "Violação de isolamento C1: test_loader foi acessado dentro do loop de épocas!"
    )
    assert "test_metrics_full" in result
    assert result["test_metrics_full"] is not None
    assert "mse" in result["test_metrics_full"]


def test_davis_cold_split_patience_adjustment(tmp_path: Path):
    """Valida elevação de paciência para Davis cold split para no mínimo davis_cold_patience."""
    torch.manual_seed(42)
    x = torch.randn(10, 4)
    y = torch.randn(10)
    train_loader = DataLoader(SimpleDictDataset(x, y), batch_size=5)
    val_loader = DataLoader(SimpleDictDataset(x, y), batch_size=5)

    model = SimpleDictModel(in_features=4)

    # Treina com paciência inicial baixa (5), mas dataset=davis e split_type=cold_target
    result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=None,
        epochs=2,
        patience=5,
        davis_cold_patience=80,
        dataset_name="davis",
        split_type="cold_target",
        checkpoint_dir=tmp_path / "checkpoints",
    )

    assert result["total_epochs_run"] == 2


def test_near_epoch_ceiling_warning(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """Valida flag e mensagem de aviso quando best_epoch > 0.8 * epochs."""
    torch.manual_seed(42)
    x = torch.randn(10, 4)
    y = torch.randn(10)
    train_loader = DataLoader(SimpleDictDataset(x, y), batch_size=5)
    val_loader = DataLoader(SimpleDictDataset(x, y), batch_size=5)

    model = SimpleDictModel(in_features=4)

    # 5 épocas: se best_epoch for 5, 5 > 0.8 * 5 (4.0) -> flag True
    result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=None,
        epochs=5,
        checkpoint_dir=tmp_path / "checkpoints",
    )

    if result["best_epoch"] > 4:
        assert result["near_epoch_ceiling_warn"] is True
        captured = capsys.readouterr()
        assert "[WARN]" in captured.out or "[WARN]" in captured.err


def test_train_model_restores_best_checkpoint_on_test_eval(tmp_path: Path):
    """Valida restauração do melhor checkpoint antes da avaliação final de teste."""
    torch.manual_seed(42)
    x = torch.randn(15, 4)
    y = torch.randn(15)
    train_loader = DataLoader(SimpleDictDataset(x, y, seq_len=600), batch_size=5)
    val_loader = DataLoader(SimpleDictDataset(x, y, seq_len=600), batch_size=5)
    test_loader = DataLoader(SimpleDictDataset(x, y, seq_len=600), batch_size=5)

    model = SimpleDictModel(in_features=4)

    result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=4,
        checkpoint_dir=tmp_path / "checkpoints",
        dataset_name="davis",
        split_type="random",
    )

    assert Path(result["best_checkpoint_path"]).exists()
    assert Path(result["last_checkpoint_path"]).exists()
    assert result["test_metrics_full"] is not None
    assert "ci" in result["test_metrics_full"]
    assert "test_metrics_uncensored" in result
    assert "test_metrics_by_seq_len" in result
    assert "history" in result
    assert len(result["history"]) == 4
    assert "oof_predictions" in result
    assert len(result["oof_predictions"]) == 15


def test_train_model_resume_restores_early_stopping_counter(tmp_path):
    torch.manual_seed(42)
    x = torch.randn(10, 4)
    y = torch.randn(10)
    train_loader = DataLoader(SimpleDictDataset(x, y, seq_len=500), batch_size=5)
    val_loader = DataLoader(SimpleDictDataset(x, y, seq_len=500), batch_size=5)

    model = SimpleDictModel(in_features=4)
    ckpt_dir = tmp_path / "checkpoints_resume"

    # Fase 1: treina 2 épocas
    res1 = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2,
        checkpoint_dir=ckpt_dir,
    )
    assert res1["total_epochs_run"] == 2

    # Fase 2: retoma para 5 épocas com patience 2
    model2 = SimpleDictModel(in_features=4)
    res2 = train_model(
        model=model2,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,
        patience=2,
        checkpoint_dir=ckpt_dir,
        resume=True,
    )
    assert res2["total_epochs_run"] >= 2
    assert "history" in res2

