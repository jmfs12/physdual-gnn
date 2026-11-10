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
- Todos os arquivos de checkpoints `.pt` salvam exclusivamente tensores PyTorch e tipos primitivos nativos do Python (`int`, `float`, `str`, `dict`, `list`). Qualquer escalar do NumPy (`np.float64`, `np.int64`, etc.) deve ser sanitizado recursivamente antes de `torch.save`, e estados de RNG devem usar apenas tensores nativos (`torch.get_rng_state()`), garantindo recarga estrita com `torch.load(..., weights_only=True)` no PyTorch 2.6+.
- Todo e qualquer commit no Git deve ser realizado com o usuário configurado: `João Miguel Francisco de Souza <joanomiguel@gmail.com>`.
- Avaliação multi-seed obrigatória nas 5 seeds `[0, 1, 2, 3, 4]`, reportando **média ± desvio-padrão** para cada métrica em cada regime de split.

---

## Review Focus

1. **Divisão por Zero, Variância Nula e Correlação Invertida em Métricas:** Quando o modelo prediz valores constantes (ex.: no início do treino ou colapso de gradiente), o desvio-padrão de $\hat{y}$ é zero. As funções de Pearson, Spearman e $r_m^2$ devem tratar variância nula com segurança retornando `0.0` (nativo `float`) em vez de propagar `NaN` ou lançar `ZeroDivisionError`. Para $r_m^2$, correlações negativas ($r \le 0$) também devem retornar estritamente `0.0`.
2. **Empates em Concordance Index (Ties):** No Davis, 69,64% dos pares estão censurados em $pK_d = 5.0$. O cálculo de CI não pode contabilizar pares onde $y_i == y_j$ no denominador de pares informativos, e deve conceder peso $0.5$ apenas para predições empatadas ($\hat{y}_i == \hat{y}_j$) quando os valores verdadeiros diferem ($y_i > y_j$). A implementação deve coincidir exatamente com `lifelines.utils.concordance_index`.
3. **Collate de Grafos Moleculares PyG e Dimensão Atômica (`drug_in_dim=9`):** Fármacos variam de 3 a dezenas de átomos, e os grafos cacheados na Fase 1 (`featurizer_drug.py`) possuem `x.shape == [N, 9]`. O `dta_collate_fn` deve empacotar os grafos usando `torch_geometric.data.Batch.from_data_list`, e `GINEncoder`/`GraphDTAModel` devem fixar `drug_in_dim: int = 9` como default explícito.
4. **Tokenização, Padding e Transposição de Eixos na 1D CNN:** Proteínas de Davis e KIBA variam de dezenas a milhares de aminoácidos (até >1022 resíduos). A 1D CNN do GraphDTA exige entradas de comprimento uniforme: aplicar padding/truncamento com token zero (`pad_idx=0`) até `max_seq_len` (padrão 1000 tokens), inicializar `nn.Embedding(num_embeddings=26, embedding_dim=128, padding_idx=0)` e transpor obrigatoriamente a saída de `[B, L, C]` para `[B, C, L]` via `.transpose(1, 2)` antes de `nn.Conv1d`.
5. **Sanitização `weights_only=True`, Direção do Early Stopping e Escrita Atômica no Drive FUSE:** Checkpoints devem converter recursivamente escalares NumPy para tipos nativos Python antes de `torch.save`, realizar escrita atômica via arquivo temporário para evitar corrupção em quedas de sessão do Colab, sincronizar `checkpoint_last.pt` periodicamente (`save_last_every=5`) para evitar HTTP 403 Rate Limit no Drive FUSE, e ajustar dinamicamente `mode="min"` (quando monitorando `mse`) ou `mode="max"` (quando monitorando `ci`).

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

Fórmula canônica de Roy ($r_m^2$) com guarda de estabilidade:
$$r_m^2 = \begin{cases} r^2 \cdot (1 - \sqrt{|r^2 - r_0^2|}), & \text{se } r > 0,\ \sum \hat{y}_i^2 > 0 \text{ e } \sum (y_i - \bar{y})^2 > 0 \\ 0.0, & \text{caso contrário} \end{cases}$$
onde $r$ é a correlação de Pearson, $r_0^2 = 1 - \frac{\sum (y_i - k \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ e $k = \frac{\sum y_i \hat{y}_i}{\sum \hat{y}_i^2}$. Todas as funções de métrica devem retornar estritamente o tipo nativo `float` do Python (convertendo qualquer `np.float64` via `float(...)`).

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

    all_m = compute_all_metrics(y_true, y_pred)
    assert all(type(v) is float for v in all_m.values())


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
2. `compute_ci(y_true, y_pred)` vetorizado sem loops $O(N^2)$ lentos em Python puro (delegando para `lifelines.utils.concordance_index` quando disponível, com fallback vetorizado idêntico).
3. `compute_mse`, `compute_rmse`, `compute_pearson` e `compute_spearman` (via `scipy.stats.pearsonr` e `scipy.stats.spearmanr` com guarda para desvio-padrão nulo).
4. `compute_rm2` seguindo a formulação definida por partes (retornando `0.0` se $r \le 0$, $\sum \hat{y}_i^2 == 0$ ou $\sum (y_i - \bar{y})^2 == 0$).
5. `compute_all_metrics` retornando dicionário de `float` nativos com chaves: `["ci", "mse", "rmse", "pearson", "spearman", "rm2"]`.

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
- Consumes: `data/processed/{dataset}_clean.csv`, `data/processed/{dataset}_drug_graphs.pt`, e splits JSON gerados na Fase 1 em `data/processed/splits/{dataset}_{split_type}_seed{seed}.json`.
- Produces:
  - Constantes exportadas: `PROTEIN_ALPHABET = "ABCDEFGHIKLMNOPQRSTUVWXYZ"` (25 caracteres, excluindo `'J'`, alinhado ao ADR #8 da Fase 1) e `PROTEIN_CHAR_TO_IDX: dict[str, int]` mapeando `1..25` (`0` reservado para `pad`/desconhecido).
  - `tokenize_protein_sequence(seq: str, max_len: int = 1000) -> np.ndarray` (array 1D `int64` de tamanho `max_len`, com aminoácidos mapeados de 1 a 25 e `pad_idx=0`).
  - `class DTADataset(torch.utils.data.Dataset)`:
    - Retorna dicionário: `{"drug_graph": Data, "target_tokens": Tensor, "smiles": str, "target_id": str, "drug_id": str, "affinity": Tensor}`.
  - `dta_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]`:
    - Combina grafos com `torch_geometric.data.Batch.from_data_list`.
    - Agrupa `target_tokens` em tensor 2D `[B, max_len]`.
    - Agrupa `affinity` em tensor 1D `[B]`.
  - `get_dta_dataloaders(dataset_name: str, split_type: str, seed: int, batch_size: int = 128, num_workers: int = 0, root: Path | None = None) -> tuple[DataLoader, DataLoader, DataLoader]`
  - `prepare_local_cache(tar_path: Path, local_cache_dir: Path) -> Path`:
    - Utilitário de infraestrutura para descompactar tarballs de cache (como `esm_embeddings_{model_name}.tar`) para o SSD efêmero `/content/local_cache/` no Colab. *Nota:* Na Fase 2, o GraphDTA opera sobre sequências tokenizadas via 1D CNN (não consumindo embeddings ESM-2); este utilitário antecipa e valida o staging de I/O para a Fase 3.

- [ ] **Step 1: Escrever testes unitários em `tests/test_datamodule.py`**

```python
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from physdual.data.datamodule import (
    DTADataset,
    PROTEIN_ALPHABET,
    dta_collate_fn,
    tokenize_protein_sequence,
)
from physdual.data.featurizer_drug import smiles_to_graph_dict


def test_tokenize_protein_sequence_length_and_padding():
    assert len(PROTEIN_ALPHABET) == 25
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

Implementar `PROTEIN_ALPHABET` (`"ABCDEFGHIKLMNOPQRSTUVWXYZ"`), `PROTEIN_CHAR_TO_IDX`, função de tokenização com truncamento e preenchimento à direita (`pad_idx=0`), classe `DTADataset` compatível com o cache de grafos dicionário (`weights_only=True`), função `dta_collate_fn`, `prepare_local_cache` e orquestrador `get_dta_dataloaders` lendo os splits em `data/processed/splits/{dataset}_{split_type}_seed{seed}.json`.

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
- Consumes: Batches gerados por `dta_collate_fn` (Task 2): `drug_batch` (`Batch` PyG com `x.shape == [N, 9]`) e `target_tokens` (`Tensor [B, max_len]`)
- Produces:
  - `class GINEncoder(nn.Module)`:
    - Recebe `in_dim: int = 9` como default explícito (correspondente aos 9 atributos atômicos de `src/physdual/data/featurizer_drug.py` da Fase 1).
    - 5 camadas `GINConv` sequenciais com MLPs de 2 camadas (`Linear -> BatchNorm1d -> ReLU -> Linear`), `BatchNorm1d` após cada convolução, agregação global por soma (`global_add_pool`) e projeção Linear para `output_dim` (128).
  - `class Conv1DTargetEncoder(nn.Module)`:
    - `nn.Embedding(num_embeddings=26, embedding_dim=128, padding_idx=0)`.
    - **Transposição obrigatória no `forward`:** `x = self.embedding(target_tokens).transpose(1, 2)` convertendo `[B, max_len, 128]` para `[B, 128, max_len]`.
    - 3 camadas convolucionais 1D com ativação `ReLU` (`Conv1d(128, 32, kernel_size=4)`, `Conv1d(32, 64, kernel_size=8)`, `Conv1d(64, 96, kernel_size=12)`), seguidas por `AdaptiveMaxPool1d(1).squeeze(-1)` e projeção `nn.Linear(96, output_dim)` (onde `output_dim=128`).
  - `class GraphDTAModel(nn.Module)`:
    - Assinatura com defaults alinhados à Fase 1: `__init__(self, drug_in_dim: int = 9, protein_vocab_size: int = 26, embedding_dim: int = 128, dropout: float = 0.1)`.
    - Concatena as representações de 128d do fármaco e do alvo (256d total) e alimenta MLP de predição:
      `Linear(256, 1024) -> ReLU -> Dropout(dropout) -> Linear(1024, 512) -> ReLU -> Dropout(dropout) -> Linear(512, 1)`.
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

    # Testa instanciação com defaults explícitos (drug_in_dim=9)
    model = GraphDTAModel()
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
  - `_sanitize_state_dict(obj: Any) -> Any`: converte recursivamente qualquer escalar NumPy (`np.generic`, `np.float64`, `np.int64`, arrays 0-d) para tipos primitivos nativos do Python (`float`, `int`) para impedir `UnpicklingError` em `torch.load(..., weights_only=True)` no PyTorch 2.6+.
  - `save_checkpoint(state: dict[str, Any], path: Path) -> Path`: aplica `_sanitize_state_dict` e realiza **escrita atômica** (grava em arquivo temporário local no SSD/mesmo diretório e move com `os.replace` / `shutil.move`) para evitar corrupção em quedas abruptas do Google Colab/Drive FUSE.
  - `load_checkpoint(path: Path, model: nn.Module, optimizer: Optimizer | None = None) -> dict[str, Any]`: carrega estritamente com `weights_only=True` e restaura estados de modelo, otimizador e `torch_rng_state` (apenas tensores nativos `torch.get_rng_state()`, nunca tuplas de NumPy).
  - `get_monitor_mode(monitor: str) -> Literal["min", "max"]`: retorna `"min"` para `"mse"` / `"rmse"` e `"max"` para `"ci"` / `"pearson"` / `"spearman"` / `"rm2"`.
  - `class EarlyStopping(patience: int = 15, mode: Literal["min", "max"] = "min", delta: float = 1e-4)`
  - `train_one_epoch(model, dataloader, optimizer, criterion, device) -> float`
  - `evaluate(model, dataloader, criterion, device) -> tuple[float, dict[str, float]]`
  - `train_model(model, train_loader, val_loader, test_loader, config, checkpoint_dir, monitor="mse", save_last_every=5, resume=False)`:
    - Infere automaticamente `mode = get_monitor_mode(monitor)` para o `EarlyStopping` e para `ReduceLROnPlateau`.
    - Salva `checkpoint_best.pt` imediatamente sempre que a métrica monitorada melhora.
    - Salva `checkpoint_last.pt` a cada `save_last_every` épocas (padrão 5) e na época final, mitigando gargalos de I/O e HTTP 403 Rate Limit no Google Drive FUSE.
    - Suporta retomada idempotente com `resume=True`.

- [ ] **Step 1: Escrever testes unitários em `tests/test_trainer_checkpoint.py`**

```python
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from physdual.models.graphdta import GraphDTAModel
from physdual.training.checkpoint import load_checkpoint, save_checkpoint
from physdual.training.trainer import EarlyStopping, get_monitor_mode


def test_checkpoint_roundtrip_weights_only_with_numpy_scalars(tmp_path):
    model = GraphDTAModel(drug_in_dim=9, protein_vocab_size=26, embedding_dim=32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    ckpt_path = tmp_path / "ckpt_test.pt"
    # Inclui escalares NumPy explícitos (np.float64, np.int64) que quebrariam weights_only=True sem sanitização
    state = {
        "epoch": np.int64(5),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "val_metrics": {"ci": np.float64(0.85), "mse": np.float32(0.3)},
        "best_epoch": np.int32(5),
        "torch_rng_state": torch.get_rng_state(),
    }
    save_checkpoint(state, ckpt_path)
    assert ckpt_path.exists()

    # Recarrega em modelo novo usando estritamente weights_only=True
    new_model = GraphDTAModel(drug_in_dim=9, protein_vocab_size=26, embedding_dim=32)
    loaded_meta = load_checkpoint(ckpt_path, model=new_model, optimizer=None)
    assert loaded_meta["epoch"] == 5
    assert type(loaded_meta["epoch"]) is int
    assert type(loaded_meta["val_metrics"]["ci"]) is float
    assert loaded_meta["val_metrics"]["ci"] == pytest.approx(0.85)

    # Verifica igualdade de parâmetros
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1, p2)


def test_early_stopping_both_modes():
    assert get_monitor_mode("mse") == "min"
    assert get_monitor_mode("ci") == "max"

    # Modo max (ex.: CI)
    es_max = EarlyStopping(patience=3, mode="max", delta=0.01)
    assert not es_max.step(0.80)   # best
    assert not es_max.step(0.805)  # melhorou menos que delta (counter=1)
    assert not es_max.step(0.79)   # piorou (counter=2)
    assert es_max.step(0.78)       # piorou (counter=3 -> stop!)
    assert es_max.should_stop

    # Modo min (ex.: MSE)
    es_min = EarlyStopping(patience=2, mode="min", delta=0.01)
    assert not es_min.step(0.50)   # best
    assert not es_min.step(0.495)  # melhorou menos que delta (counter=1)
    assert es_min.step(0.52)       # piorou (counter=2 -> stop!)
    assert es_min.should_stop
```

- [ ] **Step 2: Executar teste e verificar que falha**

Run: `pytest tests/test_trainer_checkpoint.py -v`
Expected: FAIL com `ModuleNotFoundError`

- [ ] **Step 3: Implementar `src/physdual/training/checkpoint.py`, `trainer.py` e `__init__.py`**

Implementar sanitização recursiva `_sanitize_state_dict`, escrita atômica de checkpoint e carregamento com `weights_only=True`, `get_monitor_mode`, `EarlyStopping` e loop `train_model` com suporte a `save_last_every` e `resume=True`.

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
  `python scripts/run_phase2_baseline.py --dataset davis,kiba --split random,cold_drug,cold_target --seeds 0,1,2,3,4 --epochs 100 --monitor mse --device cuda`
  - Salva sumário de resultados em `outputs/results/baseline_graphdta_{dataset}_{split}.json` seguindo o JSON Schema canônico:
    ```json
    {
      "dataset": "davis",
      "split": "random",
      "model": "graphdta",
      "monitor": "mse",
      "seeds": [0, 1, 2, 3, 4],
      "per_seed_results": {
        "0": {"ci": 0.881, "mse": 0.235, "rmse": 0.485, "pearson": 0.842, "spearman": 0.695, "rm2": 0.680}
      },
      "aggregated": {
        "ci": {"mean": 0.881, "std": 0.005},
        "mse": {"mean": 0.235, "std": 0.010}
      }
    }
    ```
  - Reporta tabela consolidada com **média ± desvio-padrão nas 5 seeds** para cada uma das 6 métricas.
- CLI de Verificação do Gate da Fase 2:
  - Modo rápido de CI / desenvolvimento local (< 5 segundos):
    `python scripts/verify_phase2.py --config configs/base.yaml --smoke`
  - Modo completo de validação pós-treino no Colab:
    `python scripts/verify_phase2.py --config configs/base.yaml --full`

Critérios objetivos verificados por `scripts/verify_phase2.py` (Gate de Pronto da Fase 2):
1. `[--smoke e --full]` Todas as 6 métricas passam nas asserções matemáticas, retornam `float` nativo e coincidem com `lifelines` para o CI.
2. `[--smoke e --full]` O modelo `GraphDTAModel` (`drug_in_dim=9`) realiza forward e backward sintético sem produzir `NaN` ou gradientes nulos.
3. `[--smoke e --full]` `DTADataset` e `dta_collate_fn` empacotam grafos de tamanhos variados sem vazamento de batch.
4. `[--smoke e --full]` Checkpoints contendo escalares NumPy são sanitizados, salvos atomicamente e recarregam estritamente com `torch.load(..., weights_only=True)`.
5. `[Apenas --full]` Os arquivos `outputs/results/baseline_graphdta_{dataset}_{split}.json` existem para os regimes avaliados e contêm as 6 métricas agregadas por média ± desvio-padrão nas 5 seeds.
6. `[Apenas --full]` As métricas do baseline atingem os patamares esperados da literatura no split `random` (ex.: CI $\ge 0{,}86$ no Davis) e demonstram a queda esperada em `cold_drug` e `cold_target`.

- [ ] **Step 1: Escrever teste unitário para `scripts/verify_phase2.py` em `tests/test_verify_phase2.py`**

```python
import subprocess
import sys
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")


def test_verify_phase2_smoke_mode():
    result = subprocess.run(
        [sys.executable, "scripts/verify_phase2.py", "--smoke"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"verify_phase2 --smoke falhou:\n{result.stdout}\n{result.stderr}"
    assert "RESULTADO FINAL: APROVADO" in result.stdout
```

- [ ] **Step 2: Executar teste e verificar que falha**

Run: `pytest tests/test_verify_phase2.py -v`
Expected: FAIL (arquivo `scripts/verify_phase2.py` ainda inexistente)

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
- Modify: `configs/base.yaml` (atualizar `project.phase: 2` e hiperparâmetros `graphdta`)

- [ ] **Step 1: Escrever `docs/FASE2_BASELINE.md`**
  - Fórmulas matemáticas completas das 6 métricas (CI, MSE, RMSE, Pearson, Spearman e a definição por partes de Roy's $r_m^2$).
  - Especificação detalhada da arquitetura do GraphDTA (GIN 5 camadas + 1D CNN 3 camadas com transposição `[B, 128, L]` e projeção `Linear(96, 128)`), documentando formalmente o uso do vetor atômico canônico de 9 dimensões (`drug_in_dim=9`) da Fase 1 em substituição ao one-hot de 78 dimensões do DeepChem original.
  - Tabela canônica de resultados esperados de benchmark (reprodução de literatura) para Davis e KIBA nos 3 regimes (`random`, `cold_drug`, `cold_target`).
  - Análise da variância de `cold_drug` no Davis com as 5 seeds.
  - Guia de I/O, escrita atômica de checkpoints e mitigação de quota do Google Drive FUSE no Colab.
- [ ] **Step 2: Atualizar `README.md`, `docs/PROXIMOS_PASSOS.md` e `configs/base.yaml` preparando a transição para a Fase 3 (Codificador de Proteínas com ESM-2 Congelado)**
- [ ] **Step 3: Commit da Task 6**

```bash
git add docs/FASE2_BASELINE.md README.md docs/PROXIMOS_PASSOS.md configs/base.yaml
git commit -m "docs(phase2): document GraphDTA baseline architecture, evaluation protocols, and benchmark tables"
```

---

## Gate de Pronto da Fase 2 (Critérios de Aceite Obrigatórios)

A Fase 2 **só é considerada concluída** quando `python scripts/verify_phase2.py --full` (após execução no Colab) retornar `exit code 0` comprovando:

1. **Validação das Métricas:** Todas as 6 métricas passam nos testes retornando `float` nativo, com ausência de `NaN` para predições degeneradas e CI idêntico ao `lifelines`.
2. **Estabilidade de Treino e Contratos (`--smoke`):** `GraphDTAModel(drug_in_dim=9)` executa forward/backward sem falhas de dimensão, vazamento de memória ou gradientes nulos.
3. **Compatibilidade `weights_only=True` com Sanitização NumPy:** Checkpoints salvos atomicamente restauram perfeitamente sem permissões inseguras de unpickling no PyTorch 2.6+.
4. **Resultados Multi-Seed Gerados (`--full`):** Arquivos `outputs/results/baseline_graphdta_*.json` gravados com métricas das 5 seeds reportando média ± desvio-padrão.
5. **Rigor nos Splits Cold (`--full`):** Confirmação da queda esperada de desempenho em `cold_drug` e `cold_target` em relação a `random`, comprovando a eficácia do isolamento por conteúdo implementado na Fase 1.

---

## Auto-Verificação do Plano (Self-Review)

1. **Cobertura da Especificação:** Cobre 100% dos requisitos da Fase 2 em `docs/PLANO_AJUSTADO.md` (CI, MSE, Pearson, Spearman, $r_m^2$, loop de treino, checkpointing no Drive, baseline GraphDTA com GIN).
2. **Granularidade e TDD:** Cada uma das 6 tarefas possui ciclo TDD explícito (teste que falha $\rightarrow$ execução $\rightarrow$ implementação $\rightarrow$ execução $\rightarrow$ commit).
3. **Consistência de Interfaces (Auditado pelo Planner):** `DTADataset` e `dta_collate_fn` consomem exatamente o cache de grafos dicionário (`drug_in_dim=9`) e os splits em `data/processed/splits/`; `Conv1DTargetEncoder` usa `num_embeddings=26` e transpõe eixos antes de `Conv1d`; `save_checkpoint` sanitiza escalares NumPy para compatibilidade estrita com `weights_only=True`.
4. **Review Focus:** Casos limites (variância zero, correlação invertida em $r_m^2$, empates em CI com dados censurados do Davis, escrita atômica e rate-limit no Drive FUSE) possuem guardas e testes dedicados.
