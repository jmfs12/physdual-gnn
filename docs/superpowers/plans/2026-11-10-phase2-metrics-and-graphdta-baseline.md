# Fase 2: Métricas, Loop de Treino e Baseline GraphDTA — Plano de Implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar a suíte rigorosa de métricas DTA (CI, MSE, RMSE, Pearson, Spearman, $r_m^2$), o pipeline de dados com DataLoader PyG e descompactação otimizada em SSD efêmero, o modelo baseline canônico GraphDTA (GIN atômico + 1D CNN para proteínas) e o loop de treinamento multi-seed com early stopping e checkpointing compatível com `weights_only=True` no Google Drive.

**Architecture:** 
- Módulo `src/physdual/eval/metrics.py` com implementação vetorizada em PyTorch/NumPy das 6 métricas de regressão e validação cruzada contra `lifelines.utils.concordance_index`.
- Módulo `src/physdual/data/datamodule.py` gerenciando tokenização de sequências proteicas, dataset PyG (`DTADataset`), collate customizado para grafos moleculares e utilitário de staging de I/O (`prepare_local_cache`) para extração rápida de tarball no SSD efêmero `/content/local_cache/`.
- Módulo `src/physdual/models/graphdta.py` implementando a arquitetura canônica do GraphDTA (GIN com 5 camadas + BatchNorm para ligantes, 1D CNN com 3 camadas convolucionais para alvos, e MLP de predição).
- Módulo `src/physdual/training/` (`trainer.py`, `checkpoint.py`) orquestrando treino, validação, early stopping, agendamento de taxa de aprendizado e persistência de checkpoints por época no Google Drive.
- Script CLI `scripts/run_phase2_baseline.py`, verificador de conformidade `scripts/verify_phase2.py` e notebook `notebooks/02_graphdta_baseline.ipynb` para execução no Colab e cálculo de **média ± desvio-padrão nas 5 seeds** para os 3 splits (`random`, `cold_drug`, `cold_target`).

**Tech Stack:** Python 3.10+, PyTorch e NumPy conforme snapshot da Fase 0 (`docs/colab_env_snapshot.txt`), PyTorch Geometric (PyG), lifelines, scipy, scikit-learn, pandas, pyyaml, pytest.

**Spec:** [`docs/PLANO_AJUSTADO.md`](docs/PLANO_AJUSTADO.md), [`docs/DECISOES.md`](docs/DECISOES.md) (ADRs #1, #2, #6), e [`docs/FASE1_DADOS.md`](docs/FASE1_DADOS.md).

> **Nota sobre execução de testes em máquina local (Windows + WSL):**
> Todos os comandos `Run:` neste plano são relativos à raiz do repositório (`pytest tests/... -v`). Caso execute no Windows invocando o WSL, use:
> `wsl bash -lc "cd /mnt/c/Users/joao.francisco/Documents/dev/physdual-gnn && <comando>"`
> Para execução rápida com interpretador local `uv run`:
> `uv run --no-project --python 3.12 --with pytest --with pandas --with numpy --with pyyaml --with rdkit --with scipy --with scikit-learn --with lifelines pytest <arquivo> -v`

---

## Global Constraints

- O Colab já traz `torch` e `numpy` pré-instalados; nenhum script deve reinstalar ou alterar versões de `torch` ou `numpy` (usar as versões registradas em `docs/colab_env_snapshot.txt`).
- Os caminhos de persistência devem respeitar `PHYSDUAL_ROOT` resolvido via `physdual.utils.paths.get_project_root()`.
- Todos os arquivos de checkpoints `.pt` salvam tensores e tipos primitivos Python (ex.: `epoch`, `model_state_dict`, `optimizer_state_dict`, `val_metrics`, `rng_state`) e devem ser estritamente recarregáveis com `torch.load(..., weights_only=True)`.
- Todo e qualquer commit no Git deve ser realizado com o usuário configurado: `João Miguel Francisco de Souza <joanomiguel@gmail.com>`.
- Avaliação multi-seed obrigatória nas 5 seeds `[0, 1, 2, 3, 4]`, reportando **média ± desvio-padrão** para cada métrica em cada regime de split.

---

## Review Focus

1. **Divisão por Zero e Variância Nula em Métricas:** Quando o modelo prediz valores constantes (ex.: no início do treino ou colapso de gradiente), o desvio-padrão de $\hat{y}$ é zero. As funções de Pearson, Spearman e $r_m^2$ devem tratar variância nula com segurança retornando `0.0` em vez de propagar `NaN` ou lançar `ZeroDivisionError`.
2. **Empates em Concordance Index (Ties):** No Davis, 69,64% dos pares estão censurados em $pK_d = 5.0$. O cálculo de CI não pode contabilizar pares onde $y_i == y_j$ no denominador de pares informativos, e deve conceder peso $0.5$ apenas para predições empatadas ($\hat{y}_i == \hat{y}_j$) quando os valores verdadeiros diferem ($y_i > y_j$). A implementação deve coincidir exatamente com `lifelines.utils.concordance_index`.
3. **Collate de Grafos Moleculares PyG de Tamanhos Variados:** Fármacos variam de 3 a dezenas de átomos. O `dta_collate_fn` deve empacotar os grafos individuais usando `torch_geometric.data.Batch.from_data_list`, preservando atributos atômicos `x`, topologia `edge_index` e atributos de ligação `edge_attr` sem vazamento de batch.
4. **Tokenização e Padding da Sequência Proteica (1D CNN):** Proteínas de Davis e KIBA variam de dezenas a milhares de aminoácidos (até >1022 resíduos). A 1D CNN do GraphDTA exige entradas de comprimento uniforme: aplicar padding/truncamento com token zero (`pad_idx=0`) até `max_seq_len` (padrão 1000 tokens) e inicializar `nn.Embedding` com `padding_idx=0`.
5. **Retomada Idempotente de Treinamento (`--resume`):** Em caso de desconexão da VM do Colab, o trainer deve conseguir carregar `checkpoints/<exp>_last.pt` com `weights_only=True`, restaurar pesos do modelo, otimizador, escalonador e sementes de RNG, continuando exatamente da época seguinte à interrupção.

---

### Task 1: Módulo de Métricas Bioinformáticas e Validação Cruzada com `lifelines`

**Files:**
- Create: `src/physdual/eval/metrics.py`
- Create: `src/physdual/eval/__init__.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: Tensores ou arrays 1D `y_true` e `y_pred` de floats
- Produces:
  - `compute_ci(y_true: np.ndarray | torch.Tensor, y_pred: np.ndarray | torch.Tensor) -> float`
  - `compute_mse(y_true: np.ndarray | torch.Tensor, y_pred: np.ndarray | torch.Tensor) -> float`
  - `compute_rmse(y_true: np.ndarray | torch.Tensor, y_pred: np.ndarray | torch.Tensor) -> float`
  - `compute_pearson(y_true: np.ndarray | torch.Tensor, y_pred: np.ndarray | torch.Tensor) -> float`
  - `compute_spearman(y_true: np.ndarray | torch.Tensor, y_pred: np.ndarray | torch.Tensor) -> float`
  - `compute_rm2(y_true: np.ndarray | torch.Tensor, y_pred: np.ndarray | torch.Tensor) -> float`
  - `compute_all_metrics(y_true: np.ndarray | torch.Tensor, y_pred: np.ndarray | torch.Tensor) -> dict[str, float]`

Fórmula canônica de Roy ($r_m^2$):
$r_m^2 = r^2 \cdot (1 - \sqrt{|r^2 - r_0^2|})$, onde $r^2$ é o quadrado do Pearson, e $r_0^2 = 1 - \frac{\sum (y_i - k \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ com $k = \frac{\sum y_i \hat{y}_i}{\sum \hat{y}_i^2}$.

- [ ] **Step 1: Escrever testes unitários em `tests/test_metrics.py`**

```python
import numpy as np
import pytest

from physdual.eval.metrics import (
    compute_all_metrics,
    compute_ci,
    compute_mse,
    compute_pearson,
    compute_rm2,
    compute_rmse,
    compute_spearman,
)


def test_standard_metrics_values():
    y_true = np.array([5.0, 6.0, 7.0, 8.0, 9.0], dtype=float)
    y_pred = np.array([5.2, 5.9, 7.1, 7.8, 9.2], dtype=float)

    mse = compute_mse(y_true, y_pred)
    rmse = compute_rmse(y_true, y_pred)
    pearson = compute_pearson(y_true, y_pred)
    spearman = compute_spearman(y_true, y_pred)
    ci = compute_ci(y_true, y_pred)
    rm2 = compute_rm2(y_true, y_pred)

    assert mse == pytest.approx(0.028, abs=1e-3)
    assert rmse == pytest.approx(np.sqrt(0.028), abs=1e-3)
    assert pearson > 0.99
    assert spearman == pytest.approx(1.0, abs=1e-4)
    assert ci == pytest.approx(1.0, abs=1e-4)
    assert rm2 > 0.95


def test_concordance_index_matches_lifelines():
    lifelines = pytest.importorskip("lifelines")
    rng = np.random.RandomState(42)
    # Inclui valores repetidos para testar tratamento estrito de empates (ties)
    y_true = np.array([5.0, 5.0, 6.0, 7.0, 8.0, 8.0, 9.0, 10.0])
    y_pred = y_true + rng.normal(0, 0.5, size=len(y_true))

    expected_ci = lifelines.utils.concordance_index(y_true, y_pred)
    our_ci = compute_ci(y_true, y_pred)
    assert our_ci == pytest.approx(expected_ci, abs=1e-6)


def test_constant_predictions_does_not_crash_or_nan():
    y_true = np.array([5.0, 6.0, 7.0, 8.0], dtype=float)
    y_pred_const = np.array([6.5, 6.5, 6.5, 6.5], dtype=float)

    metrics = compute_all_metrics(y_true, y_pred_const)
    assert not np.isnan(metrics["pearson"])
    assert not np.isnan(metrics["spearman"])
    assert not np.isnan(metrics["rm2"])
    assert metrics["pearson"] == 0.0
    assert metrics["spearman"] == 0.0
    assert metrics["rm2"] == 0.0
    assert metrics["ci"] == pytest.approx(0.5, abs=1e-4)


def test_rm2_zero_when_negative_correlation():
    y_true = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    y_pred_inv = np.array([4.0, 3.0, 2.0, 1.0], dtype=float)
    rm2 = compute_rm2(y_true, y_pred_inv)
    assert rm2 == pytest.approx(0.0, abs=1e-4)
```

- [ ] **Step 2: Executar teste e verificar que falha**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'physdual.eval.metrics'`

- [ ] **Step 3: Implementar `src/physdual/eval/metrics.py` e `src/physdual/eval/__init__.py`**

Implementar:
1. `_to_numpy(x)` convertendo Tensores PyTorch ou arrays para `np.ndarray` 1D `float64`.
2. `compute_ci(y_true, y_pred)` vetorizado sem loops $O(N^2)$ lentos em Python puro (usando ordenação ou broadcasting NumPy, ou delegando para `lifelines.utils.concordance_index` com fallback vetorizado próprio).
3. `compute_mse`, `compute_rmse`, `compute_pearson` e `compute_spearman` (via `scipy.stats.pearsonr` e `scipy.stats.spearmanr` com guarda para desvio-padrão nulo).
4. `compute_rm2` seguindo a formulação de Roy et al.
5. `compute_all_metrics` retornando dicionário com chaves: `["ci", "mse", "rmse", "pearson", "spearman", "rm2"]`.

- [ ] **Step 4: Executar teste e verificar que passa**

Run: `pytest tests/test_metrics.py -v`
Expected: PASS

- [ ] **Step 5: Commit da Task 1**

```bash
git add src/physdual/eval/metrics.py src/physdual/eval/__init__.py tests/test_metrics.py
git commit -m "feat(eval): implement robust DTA evaluation metrics with lifelines cross-validation"
```

---

### Task 2: Tokenização de Sequências, Dataset PyTorch/PyG e DataLoaders Otimizados

**Files:**
- Create: `src/physdual/data/datamodule.py`
- Test: `tests/test_datamodule.py`

**Interfaces:**
- Consumes: `data/processed/{dataset}_clean.csv`, `data/processed/{dataset}_drug_graphs.pt`, splits JSON da Task 2 (Fase 1).
- Produces:
  - `tokenize_protein_sequence(seq: str, max_len: int = 1000) -> np.ndarray` (array 1D `int64` de tamanho `max_len`, com aminoácidos mapeados de 1 a 25 e `pad_idx=0`).
  - `class DTADataset(torch.utils.data.Dataset)`:
    - Retorna dicionário: `{"drug_graph": Data, "target_tokens": Tensor, "smiles": str, "target_id": str, "drug_id": str, "affinity": Tensor}`.
  - `dta_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]`:
    - Combina grafos com `torch_geometric.data.Batch.from_data_list`.
    - Agrupa `target_tokens` em tensor 2D `[B, max_len]`.
    - Agrupa `affinity` em tensor 1D `[B]`.
  - `get_dta_dataloaders(dataset_name: str, split_type: str, seed: int, batch_size: int = 128, num_workers: int = 0, root: Path | None = None) -> tuple[DataLoader, DataLoader, DataLoader]`
  - `prepare_local_cache(tar_path: Path, local_cache_dir: Path) -> Path`:
    - Utilitário para descompactar o tarball `esm_embeddings_{model_name}.tar` para o SSD efêmero `/content/local_cache/` no Colab (preparando o terreno para a Fase 3).

- [ ] **Step 1: Escrever testes unitários em `tests/test_datamodule.py`**

```python
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from physdual.data.datamodule import (
    DTADataset,
    dta_collate_fn,
    tokenize_protein_sequence,
)
from physdual.data.featurizer_drug import smiles_to_graph_dict


def test_tokenize_protein_sequence_length_and_padding():
    seq = "MKTVRQ"
    tokens = tokenize_protein_sequence(seq, max_len=10)
    assert len(tokens) == 10
    assert (tokens[:6] > 0).all()
    assert (tokens[6:] == 0).all()  # Padding


def test_dta_dataset_and_collate_fn():
    from torch_geometric.data import Batch

    # Mock de cache de grafos
    cache = {
        "CCO": smiles_to_graph_dict("CCO", drug_id="D1"),
        "CCN": smiles_to_graph_dict("CCN", drug_id="D2"),
    }
    rows = [
        {"drug_id": "D1", "smiles": "CCO", "target_id": "T1", "sequence": "MKTV", "affinity": 7.5},
        {"drug_id": "D2", "smiles": "CCN", "target_id": "T2", "sequence": "ACDE", "affinity": 6.2},
    ]

    import pandas as pd
    df = pd.DataFrame(rows)
    ds = DTADataset(df, drug_graph_cache=cache, max_seq_len=20)
    assert len(ds) == 2

    sample = ds[0]
    assert sample["affinity"].item() == pytest.approx(7.5, abs=1e-4)
    assert sample["target_tokens"].shape == (20,)

    batch = dta_collate_fn([ds[0], ds[1]])
    assert isinstance(batch["drug_batch"], Batch)
    assert batch["drug_batch"].num_graphs == 2
    assert batch["target_tokens"].shape == (2, 20)
    assert batch["affinity"].shape == (2,)
```

- [ ] **Step 2: Executar teste e verificar que falha**

Run: `pytest tests/test_datamodule.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'physdual.data.datamodule'`

- [ ] **Step 3: Implementar `src/physdual/data/datamodule.py`**

Implementar dicionário de aminoácidos canônicos IUPAC (`A..Y`), função de tokenização com truncamento e preenchimento à direita (`pad_idx=0`), classe `DTADataset` compatível com o cache de grafos dicionário (`weights_only=True`), função `dta_collate_fn` e orquestrador de `get_dta_dataloaders`.

- [ ] **Step 4: Executar teste e verificar que passa**

Run: `pytest tests/test_datamodule.py -v`
Expected: PASS

- [ ] **Step 5: Commit da Task 2**

```bash
git add src/physdual/data/datamodule.py tests/test_datamodule.py
git commit -m "feat(data): implement protein tokenizer, DTADataset, and optimized PyG DataLoader"
```

---

### Task 3: Arquitetura do Modelo Baseline GraphDTA (GIN Atômico + 1D CNN)

**Files:**
- Create: `src/physdual/models/graphdta.py`
- Create: `src/physdual/models/__init__.py`
- Test: `tests/test_graphdta_model.py`

**Interfaces:**
- Consumes: Batches gerados por `dta_collate_fn` (Task 2): `drug_batch` (`Batch` PyG) e `target_tokens` (`Tensor [B, max_len]`)
- Produces:
  - `class GINEncoder(nn.Module)`:
    - 5 camadas `GINConv` sequenciais com MLPs de 2 camadas (`Linear -> BatchNorm1d -> ReLU -> Linear`), agregação global por soma (`global_add_pool`) e projeção Linear para `output_dim` (128).
  - `class Conv1DTargetEncoder(nn.Module)`:
    - `nn.Embedding(num_features=26, embedding_dim=128, padding_idx=0)`.
    - 3 camadas convolucionais 1D (`Conv1d(128, 32, k=4)`, `Conv1d(32, 64, k=8)`, `Conv1d(64, 96, k=12)`) seguidas por `AdaptiveMaxPool1d(1)` e projeção Linear para `output_dim` (128).
  - `class GraphDTAModel(nn.Module)`:
    - Concatena as representações de 128d do fármaco e do alvo (256d total) e alimenta MLP de predição:
      `Linear(256, 1024) -> ReLU -> Dropout(0.1) -> Linear(1024, 512) -> ReLU -> Dropout(0.1) -> Linear(512, 1)`.
    - `forward(drug_batch, target_tokens) -> torch.Tensor` (shape `[B]`).

- [ ] **Step 1: Escrever testes unitários em `tests/test_graphdta_model.py`**

```python
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from physdual.data.datamodule import DTADataset, dta_collate_fn
from physdual.data.featurizer_drug import smiles_to_graph_dict
from physdual.models.graphdta import Conv1DTargetEncoder, GINEncoder, GraphDTAModel


def test_encoders_and_model_forward_backward():
    cache = {
        "CCO": smiles_to_graph_dict("CCO", drug_id="D1"),
        "CCN": smiles_to_graph_dict("CCN", drug_id="D2"),
    }
    import pandas as pd
    df = pd.DataFrame([
        {"drug_id": "D1", "smiles": "CCO", "target_id": "T1", "sequence": "MKTVRQ", "affinity": 7.0},
        {"drug_id": "D2", "smiles": "CCN", "target_id": "T2", "sequence": "ACDEFG", "affinity": 8.0},
    ])
    ds = DTADataset(df, drug_graph_cache=cache, max_seq_len=50)
    batch = dta_collate_fn([ds[0], ds[1]])

    model = GraphDTAModel(drug_in_dim=9, protein_vocab_size=26, embedding_dim=128)
    preds = model(batch["drug_batch"], batch["target_tokens"])

    assert preds.shape == (2,)
    assert not torch.isnan(preds).any()

    loss = torch.nn.functional.mse_loss(preds, batch["affinity"])
    loss.backward()

    # Confirma fluxo de gradiente em ambos os ramos (droga e alvo)
    assert any(p.grad is not None and p.grad.norm().item() > 0 for p in model.drug_encoder.parameters())
    assert any(p.grad is not None and p.grad.norm().item() > 0 for p in model.target_encoder.parameters())
```

- [ ] **Step 2: Executar teste e verificar que falha**

Run: `pytest tests/test_graphdta_model.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'physdual.models.graphdta'`

- [ ] **Step 3: Implementar `src/physdual/models/graphdta.py` e `src/physdual/models/__init__.py`**

Implementar os encoders e a classe unificada `GraphDTAModel` respeitando as dimensões e camadas canônicas do GraphDTA.

- [ ] **Step 4: Executar teste e verificar que passa**

Run: `pytest tests/test_graphdta_model.py -v`
Expected: PASS

- [ ] **Step 5: Commit da Task 3**

```bash
git add src/physdual/models/graphdta.py src/physdual/models/__init__.py tests/test_graphdta_model.py
git commit -m "feat(models): implement canonical GraphDTA architecture (GIN + 1D CNN)"
```

---

### Task 4: Motor de Treinamento, Checkpointing Seguro (`weights_only=True`) e Early Stopping

**Files:**
- Create: `src/physdual/training/checkpoint.py`
- Create: `src/physdual/training/trainer.py`
- Create: `src/physdual/training/__init__.py`
- Test: `tests/test_trainer_checkpoint.py`

**Interfaces:**
- Consumes: `GraphDTAModel`, DataLoaders, `compute_all_metrics`
- Produces:
  - `save_checkpoint(state: dict[str, Any], path: Path) -> Path` (salva tensores e primitivos em `.pt`)
  - `load_checkpoint(path: Path, model: nn.Module, optimizer: Optimizer | None = None) -> dict[str, Any]` (carrega estritamente com `weights_only=True`)
  - `class EarlyStopping(patience=15, mode="max", delta=1e-4)`
  - `train_one_epoch(model, dataloader, optimizer, criterion, device) -> float`
  - `evaluate(model, dataloader, criterion, device) -> tuple[float, dict[str, float]]`
  - `train_model(model, train_loader, val_loader, test_loader, config, checkpoint_dir, ...)`:
    - Salva `checkpoint_last.pt` a cada época e `checkpoint_best.pt` no melhor valor de validação.
    - Suporta retomar com `resume=True`.

- [ ] **Step 1: Escrever testes unitários em `tests/test_trainer_checkpoint.py`**

```python
import tempfile
from pathlib import Path
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from physdual.models.graphdta import GraphDTAModel
from physdual.training.checkpoint import load_checkpoint, save_checkpoint
from physdual.training.trainer import EarlyStopping


def test_checkpoint_roundtrip_weights_only(tmp_path):
    model = GraphDTAModel(drug_in_dim=9, protein_vocab_size=26, embedding_dim=32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    ckpt_path = tmp_path / "ckpt_test.pt"
    state = {
        "epoch": 5,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "val_metrics": {"ci": 0.85, "mse": 0.3},
        "best_epoch": 5,
    }
    save_checkpoint(state, ckpt_path)
    assert ckpt_path.exists()

    # Recarrega em modelo novo usando estritamente weights_only=True
    new_model = GraphDTAModel(drug_in_dim=9, protein_vocab_size=26, embedding_dim=32)
    loaded_meta = load_checkpoint(ckpt_path, model=new_model, optimizer=None)
    assert loaded_meta["epoch"] == 5
    assert loaded_meta["val_metrics"]["ci"] == pytest.approx(0.85)

    # Verifica igualdade de parâmetros
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1, p2)


def test_early_stopping_logic():
    es = EarlyStopping(patience=3, mode="max", delta=0.01)
    assert not es.step(0.80)  # best
    assert not es.step(0.805) # melhorou menos que delta (patience=1)
    assert not es.step(0.79)  # piorou (patience=2)
    assert es.step(0.78)      # piorou (patience=3 -> stop!)
    assert es.should_stop
```

- [ ] **Step 2: Executar teste e verificar que falha**

Run: `pytest tests/test_trainer_checkpoint.py -v`
Expected: FAIL com `ModuleNotFoundError`

- [ ] **Step 3: Implementar `src/physdual/training/checkpoint.py`, `trainer.py` e `__init__.py`**

Implementar rotinas robustas com suporte a dispositivo (`cuda`/`cpu`), registro de loss por época, salvamento de checkpoints seguro (`weights_only=True`) e callback de early stopping.

- [ ] **Step 4: Executar teste e verificar que passa**

Run: `pytest tests/test_trainer_checkpoint.py -v`
Expected: PASS

- [ ] **Step 5: Commit da Task 4**

```bash
git add src/physdual/training/ tests/test_trainer_checkpoint.py
git commit -m "feat(training): implement training engine, weights_only checkpointing, and early stopping"
```

---

### Task 5: Orquestrador CLI de Treinamento Multi-Seed, Verificador de Gate da Fase 2 e Notebook Colab

**Files:**
- Create: `scripts/run_phase2_baseline.py`
- Create: `scripts/verify_phase2.py`
- Create: `notebooks/02_graphdta_baseline.ipynb`
- Test: `tests/test_verify_phase2.py`

**Interfaces:**
- CLI de Treinamento:
  `python scripts/run_phase2_baseline.py --dataset davis,kiba --split random,cold_drug,cold_target --seeds 0,1,2,3,4 --epochs 100 --device cuda`
  - Salva sumário de resultados em `outputs/results/baseline_graphdta_{dataset}_{split}.json`.
  - Reporta tabela consolidada com **média ± desvio-padrão nas 5 seeds** para cada uma das 6 métricas.
- CLI de Verificação do Gate da Fase 2:
  `python scripts/verify_phase2.py --config configs/base.yaml`

Critérios objetivos verificados por `scripts/verify_phase2.py` (Gate de Pronto da Fase 2):
1. Todas as 6 métricas passam nos testes unitários e coincidem com `lifelines` para o CI.
2. O modelo GraphDTA realiza forward e backward sem produzir `NaN` ou gradientes nulos.
3. Os DataLoaders funcionam para todos os splits e datasets sem vazamento de memória.
4. Todos os checkpoints salvos carregam com `torch.load(..., weights_only=True)`.
5. Os arquivos de resultados do baseline GraphDTA existem para os regimes avaliados e contêm métricas completas (CI, MSE, RMSE, Pearson, Spearman, $r_m^2$) agregadas por média ± desvio-padrão nas 5 seeds.
6. As métricas do baseline atingem os patamares de reprodução esperados da literatura no split `random` (ex.: CI $\ge 0{,}86$ no Davis).

- [ ] **Step 1: Escrever teste unitário para `scripts/verify_phase2.py` em `tests/test_verify_phase2.py`**

- [ ] **Step 2: Executar teste e verificar que falha**

Run: `pytest tests/test_verify_phase2.py -v`
Expected: FAIL com `ModuleNotFoundError`

- [ ] **Step 3: Implementar `scripts/run_phase2_baseline.py`, `scripts/verify_phase2.py` e `notebooks/02_graphdta_baseline.ipynb`**

- [ ] **Step 4: Executar validação de sintaxe e testes unitários**

Run: `pytest tests/test_verify_phase2.py -v && python scripts/run_phase2_baseline.py --help && python scripts/verify_phase2.py --help`
Expected: PASS e código de saída 0 nos comandos `--help`.

- [ ] **Step 5: Commit da Task 5**

```bash
git add scripts/run_phase2_baseline.py scripts/verify_phase2.py notebooks/02_graphdta_baseline.ipynb tests/test_verify_phase2.py
git commit -m "feat(pipeline): add Phase 2 baseline CLI orchestrator, Colab notebook, and verification gate"
```

---

### Task 6: Documentação Técnica da Fase 2 e Atualização do README

**Files:**
- Create: `docs/FASE2_BASELINE.md`
- Modify: `README.md`
- Modify: `docs/PROXIMOS_PASSOS.md`
- Modify: `configs/base.yaml` (atualizar `project.phase: 2`)

- [ ] **Step 1: Escrever `docs/FASE2_BASELINE.md`**
  - Fórmulas matemáticas completas das 6 métricas (CI, MSE, RMSE, Pearson, Spearman, Roy's $r_m^2$).
  - Especificação detalhada da arquitetura do GraphDTA (GIN 5 camadas + 1D CNN 3 camadas).
  - Tabela canônica de resultados esperados de benchmark (reprodução de literatura) para Davis e KIBA nos 3 regimes (`random`, `cold_drug`, `cold_target`).
  - Análise da variância de `cold_drug` no Davis com as 5 seeds.
  - Guia de I/O e execução rápida no Colab com checkpointing no Drive.
- [ ] **Step 2: Atualizar `README.md` e `docs/PROXIMOS_PASSOS.md` preparando a transição para a Fase 3 (Codificador de Proteínas com ESM-2 Congelado)**
- [ ] **Step 3: Commit da Task 6**

```bash
git add docs/FASE2_BASELINE.md README.md docs/PROXIMOS_PASSOS.md configs/base.yaml
git commit -m "docs(phase2): document GraphDTA baseline architecture, evaluation protocols, and benchmark tables"
```

---

## Gate de Pronto da Fase 2 (Critérios de Aceite Obrigatórios)

A Fase 2 **só é considerada concluída** quando `python scripts/verify_phase2.py` retornar `exit code 0` comprovando:

1. **Validação das Métricas:** Todas as 6 métricas passam nos testes com ausência de `NaN` para predições degeneradas e CI idêntico ao `lifelines`.
2. **Estabilidade de Treino:** Treinamento de 1 época roda sem falhas de GPU, vazamento de memória ou gradientes nulos.
3. **Compatibilidade `weights_only=True`:** Checkpoints salvos restauram perfeitamente sem permissões inseguras de unpickling.
4. **Resultados Multi-Seed Gerados:** Arquivos `outputs/results/baseline_graphdta_*.json` gravados com métricas das 5 seeds reportando média ± desvio-padrão.
5. **Rigor nos Splits Cold:** Confirmação da queda esperada de desempenho em `cold_drug` e `cold_target` em relação a `random`, comprovando a eficácia do isolamento por conteúdo implementado na Fase 1.

---

## Auto-Verificação do Plano (Self-Review)

1. **Cobertura da Especificação:** Cobre 100% dos requisitos da Fase 2 em `docs/PLANO_AJUSTADO.md` (CI, MSE, Pearson, Spearman, $r_m^2$, loop de treino, checkpointing no Drive, baseline GraphDTA com GIN).
2. **Granularidade e TDD:** Cada uma das 6 tarefas possui ciclo TDD explícito (teste que falha $\rightarrow$ execução $\rightarrow$ implementação $\rightarrow$ execução $\rightarrow$ commit).
3. **Consistência de Interfaces:** `DTADataset` e `dta_collate_fn` consomem exatamente o cache de grafos dicionário e os splits JSON criados na Fase 1; o modelo consome os batches do collate; o trainer consome o modelo e as métricas.
4. **Review Focus:** Casos limites (variância zero, empates em CI com dados censurados do Davis, grafos de tamanhos variáveis, sequências proteicas truncadas) possuem testes dedicados.
