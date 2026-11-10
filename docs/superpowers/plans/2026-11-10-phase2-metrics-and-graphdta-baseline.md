# Fase 2: Emenda de Featurização (Fase 1), Métricas, Loop de Treino e Baseline GraphDTA — Plano de Implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Realizar a emenda versionada da featurização molecular para 78 dimensões one-hot (fiel ao GraphDTA original), auditar duplicatas de conteúdo/variantes nos datasets, implementar a suíte de métricas DTA (incluindo avaliação no subconjunto não-censurado, por faixa de comprimento proteico e baselines triviais), o pipeline eficiente de DataLoaders PyG, o baseline oficial `graphdta_gin` (reprodução fiel de `thinng/GraphDTA`) e a variante secundária `graphdta_cnn3` ("GraphDTA-style com CNN estilo DeepDTA"), e o motor de treinamento multi-seed retomável por run com isolamento estrito do conjunto de teste e escrita atômica de checkpoints compatíveis com `weights_only=True`.

**Architecture:**
- **Emenda à Fase 1 (`src/physdual/data/featurizer_drug.py`, `configs/base.yaml`, `scripts/verify_phase1.py`):** Implementação do featurizador atômico de 78 dimensões normalizado em $L_1$ (`featurizer_version: "v2_onehot78_l1norm"`, suportando também `"v2_onehot78_raw"` sem normalização e mantendo `"v1_int9"` para eventual ablação), cache de grafos versionado (`{dataset}_drug_graphs_v2_onehot78_l1norm.pt`) com validação estrita de versão, dimensão e flag `normalize_atom_features`, além de auditoria de colapso de `target_id`s/SMILES.
- **Métricas e Baselines Triviais (`src/physdual/eval/metrics.py`):** Cálculo vetorizado de CI (validado contra `lifelines`), MSE, RMSE, Pearson, Spearman e $r_m^2$, além de avaliação estratificada no subconjunto não-censurado do Davis ($pK_d > 5.0$), por faixa de comprimento de sequência ($\le 1000$ vs. $> 1000$ resíduos) e preditores triviais (média do treino e Ridge sobre composição de aminoácidos).
- **DataModule Eficiente (`src/physdual/data/datamodule.py`):** Tokenizador canônico de 25 resíduos, pré-construção em memória de objetos `Data` por SMILES único e de tensores tokenizados por sequência única (uma única vez no `__init__`), `dta_collate_fn` e DataLoaders com `num_workers`, `pin_memory` e `persistent_workers` configuráveis.
- **Modelos (`src/physdual/models/graphdta.py`):**
  1. **Baseline Oficial (`graphdta_gin` / `GraphDTAGIN`):** Reprodução fiel de `GINConvNet` do repositório oficial `thinng/GraphDTA` (`models/ginconv.py`), onde o ramo de proteína aplica uma única `Conv1d(in_channels=1000, out_channels=32, kernel_size=8)` tratando o comprimento da sequência como canais, seguida de `Linear(32 * 121, 128)`, e o MLP combinado segue `256 -> 1024 -> 256 -> 1` com `dropout=0.2`.
  2. **Variante Secundária (`graphdta_cnn3` / `GraphDTACNN3`):** Explicitamente rotulada como *"GraphDTA-style com CNN estilo DeepDTA"* (referência adicional, nunca citada como reprodução do paper), transpondo `[B, 1000, 128]` para `[B, 128, 1000]` com 3 camadas `Conv1d` (`32, 64, 96`), `AdaptiveMaxPool1d(1)` e `Linear(96, 128)`.
- **Motor de Treinamento e Checkpointing (`src/physdual/training/`):** Isolamento estrito do `test_loader` (avaliado **uma única vez** ao final com `checkpoint_best.pt`), escrita atômica real no mesmo filesystem (`os.replace` a partir de arquivo temporário no mesmo diretório de destino), salvamento completo de RNG (`torch.get_rng_state()` + `torch.cuda.get_rng_state_all()`), sanitização recursiva de escalares NumPy para `weights_only=True` e suporte a staging local em `/content` com sincronização periódica para o Google Drive.
- **Orquestração e Agregação Separadas (`scripts/run_phase2_baseline.py`, `scripts/verify_phase2.py`):** Gravação de 1 JSON por run em `outputs/results/runs/`, retomada idempotente pulando runs concluídos, passo prévio de medição de tempo/época e etapa independente de agregação nas 5 seeds.

**Tech Stack:** Python 3.10+, PyTorch e NumPy conforme `docs/colab_env_snapshot.txt` (usando a GPU que o Colab alocar, preferencialmente L4 no Colab Pro), PyTorch Geometric (PyG), RDKit, lifelines, scipy, scikit-learn, pandas, pyyaml, pytest.

**Spec:** [`docs/PLANO_AJUSTADO.md`](docs/PLANO_AJUSTADO.md), [`docs/DECISOES.md`](docs/DECISOES.md) e [`docs/FASE1_DADOS.md`](docs/FASE1_DADOS.md).

---

## Global Constraints

- O Colab já traz `torch` e `numpy` pré-instalados; nenhum script deve reinstalar ou alterar versões de `torch` ou `numpy` (usar exatamente o ambiente registrado em `docs/colab_env_snapshot.txt` e o dispositivo que o Colab alocar).
- Os caminhos de persistência devem respeitar `PHYSDUAL_ROOT` resolvido via `physdual.utils.paths.get_project_root()`.
- Todos os arquivos `.pt` (cache de grafos e checkpoints de treino) salvam exclusivamente tensores PyTorch e tipos primitivos nativos do Python (`int`, `float`, `str`, `dict`, `list`). Qualquer escalar do NumPy (`np.float64`, `np.int64`, etc.) deve ser sanitizado recursivamente antes de `torch.save`, garantindo recarga estrita com `torch.load(..., weights_only=True)`.
- Todo e qualquer commit no Git deve ser realizado com o usuário configurado: `João Miguel Francisco de Souza <joanomiguel@gmail.com>`.
- Avaliação multi-seed obrigatória nas 5 seeds `[0, 1, 2, 3, 4]`, reportando **média ± desvio-padrão** para cada métrica em cada regime de split.
- `drug_in_dim` nunca deve possuir valor default mágico no código dos modelos; deve vir obrigatoriamente do dicionário de configuração (`configs/base.yaml`).

> **Nota sobre ambiente de desenvolvimento local (fora dos comandos `Run:`):**
> Todos os comandos `Run:` nas tarefas abaixo assumem a raiz do repositório e usam `pytest` ou `python` diretamente. Quando executados localmente no Windows sem ambiente virtual ativo, o executor pode prefixar internamente com `uv run --no-project --python 3.12 --with pytest ...` ou invocar via WSL, mantendo os comandos do plano limpos e portáveis.

---

## Verificação no Repositório Oficial do GraphDTA (`thinng/GraphDTA`)

Os arquivos oficiais do repositório [`thinng/GraphDTA`](https://github.com/thinng/GraphDTA) foram inspecionados diretamente no commit `6162245c6776684b391d30610850fe7077478435` (data: `2021-04-13T05:13:35Z`):

1. **Featurização Atômica (`create_data.py:L230-L256`):**
   - `atom_features(atom)` concatena 5 blocos one-hot/binários totalizando **exatamente 78 dimensões**:
     - `atom.GetSymbol()` com `one_of_k_encoding_unk` em 44 categorias: `['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']` (**44 dims**).
     - `atom.GetDegree()` em `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]` (**11 dims**).
     - `atom.GetTotalNumHs()` com `one_of_k_encoding_unk` em `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]` (**11 dims**).
     - `atom.GetImplicitValence()` com `one_of_k_encoding_unk` em `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]` (**11 dims**).
     - `[atom.GetIsAromatic()]` (**1 dim**).
   - **Normalização L1 por átomo (`create_data.py:L256`):** No `smile_to_graph` oficial, cada vetor atômico binário de 78 dimensões é dividido pela sua soma (`feature = feature / sum(feature)`). Com essa normalização ativa (`v2_onehot78_l1norm`), **o vetor deixa de ser one-hot binário estrito (`{0, 1}`)** e passa a conter valores fracionários em $\{0, 0.25\}$ (para átomos não-aromáticos, cuja soma bruta é 4) ou $\{0, 0.20\}$ (para átomos aromáticos, cuja soma bruta é 5).
2. **Vocabulário e Tokenização de Proteínas (`create_data.py:L268-L273, L319-L322`):**
   - `seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"` (25 caracteres mapeados para `1..25`, `0` para padding à direita até `max_seq_len = 1000`).
3. **Arquitetura `GINConvNet` (`models/ginconv.py:L34-L111`):**
   - Parâmetros: `num_features_xd=78`, `num_features_xt=25`, `n_filters=32`, `embed_dim=128`, `output_dim=128`, `dropout=0.2`, `dim=32`.
   - **Ramo do fármaco:** 5 camadas `GINConv(Sequential(Linear(in_d, 32), ReLU(), Linear(32, 32)))`, cada uma seguida de `ReLU` e `BatchNorm1d(32)` (`x = F.relu(self.conv_k(x, edge_index)); x = self.bn_k(x)`). Agregação `global_add_pool(x, batch)`, seguida de `F.relu(self.fc1_xd(x))` (`Linear(32, 128)`) e `F.dropout(x, p=0.2, training=self.training)`.
   - **Ramo da proteína:** `embedding_xt = nn.Embedding(26, 128)` gerando `[B, 1000, 128]`. Aplica **uma única** `conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=32, kernel_size=8)` diretamente sobre `[B, 1000, 128]` (tratando o comprimento `1000` como canais de entrada e convoluindo sobre o eixo `128`, resultando em `[B, 32, 121]`), seguida de `conv_xt.view(-1, 32 * 121)` e `fc1_xt = nn.Linear(32 * 121, 128)`.
   - **MLP final:** Concatena `[x, xt]` (`256d`) $\rightarrow$ `fc1 = Linear(256, 1024)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.2)` $\rightarrow$ `fc2 = Linear(1024, 256)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.2)` $\rightarrow$ `out = Linear(256, 1)`.
4. **Treinamento Oficial (`training.py` e Tabela 1 de Nguyen et al., 2021):**
   - `TRAIN_BATCH_SIZE = 512`, `TEST_BATCH_SIZE = 512`, `LR = 0.0005` (constante, **sem** `ReduceLROnPlateau`), `optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)`, `loss_fn = nn.MSELoss()`, `NUM_EPOCHS = 1000`, camadas por variante na Tabela 1: `GCN=3`, `GIN=5`, `GAT=2`, `GAT_GCN=2`.

---

## Review Focus

1. **Prevenção de Vazamento do Conjunto de Teste (C1):** No `training.py` original do GraphDTA, o teste era avaliado a cada época (viés otimista de seleção no teste). No PhysDual-GNN, `train_model` **jamais** passa `test_loader` pelo modelo durante o loop de épocas. A seleção de `checkpoint_best.pt` e o `EarlyStopping` operam exclusivamente sobre `val_loader`. O `test_loader` é consumido **uma única vez** após restaurar `checkpoint_best.pt`.
2. **Versionamento Estrito do Cache de Grafos e Normalização L1 (D2):** O cache v2 (`{dataset}_drug_graphs_v2_onehot78_l1norm.pt`) embute `featurizer_version="v2_onehot78_l1norm"`, `atom_feature_dim=78` e `normalize_atom_features=True`. Tentativas de carregar um cache com versão, dimensão ou flag de normalização divergente do `configs/base.yaml` devem falhar imediatamente com `ValueError`.
3. **Escrita Atômica no Mesmo Filesystem e Checkpointing Seguro (C4):** `os.replace` só é atômico dentro do mesmo sistema de arquivos. O arquivo temporário deve ser gravado no **mesmo diretório do arquivo alvo** (`path.parent / f".tmp_{path.name}"`). Escalares NumPy devem ser convertidos para tipos nativos Python antes de `torch.save` para compatibilidade com `weights_only=True`.
4. **Censura (69,64% em $pK_d = 5.0$) e Truncamento em 1000 Resíduos:** Avaliar métricas tanto no conjunto de teste completo quanto no subconjunto não-censurado ($pK_d > 5.0$ no Davis) e estratificar o desempenho por faixa de comprimento ($\le 1000$ vs. $> 1000$ aa, dado que 83 proteínas excedem 1022 aa na Fase 1).
5. **Ruído de Validação nos Splits Cold do Davis e Alerta de Teto de Épocas:** Usar `patience: 80` tanto para `cold_drug` (~7 fármacos na validação) quanto para `cold_target` no Davis (`davis_cold_patience: 80`), desligar `ReduceLROnPlateau` por padrão (`use_lr_scheduler: false`), registrar `best_epoch` em cada run e emitir `[WARN]` se `best_epoch > 0.8 * epochs`.

---

### Task 0: Emenda à Fase 1 — Featurização Atômica One-Hot L1-Normalizada (78d), Cache Versionado, Auditoria de Duplicatas e Configuração Base (D2, C3)

**Files:**
- Modify: `configs/base.yaml`
- Modify: `src/physdual/data/featurizer_drug.py`
- Modify: `src/physdual/data/datasets.py`
- Modify: `scripts/run_phase1_data.py`
- Modify: `scripts/verify_phase1.py`
- Modify: `docs/DECISOES.md`
- Modify: `tests/test_featurizer_drug.py`
- Modify: `tests/test_datasets.py`

**Interfaces & Mudanças:**
- Em `configs/base.yaml` (C3 — fixado **antes** do modelo e do trainer):
  - Adicionar seção `featurizer`:
    ```yaml
    featurizer:
      version: "v2_onehot78_l1norm"
      atom_feature_dim: 78
      edge_feature_dim: 4
      normalize_atom_features: true
    ```
  - Adicionar seção `graphdta` com os hiperparâmetros verificados no repositório oficial (`thinng/GraphDTA` e Tabela 1 de Nguyen et al., 2021):
    ```yaml
    graphdta:
      official_model: "graphdta_gin"
      secondary_model: "graphdta_cnn3"
      drug_in_dim: 78
      protein_vocab_size: 25
      max_seq_len: 1000
      embedding_dim: 128
      gin_hidden_dim: 32
      gin_num_layers: 5
      cnn_num_filters: 32
      cnn_kernel_size: 8
      output_dim: 128
      mlp_hidden_dims: [1024, 256]
      dropout: 0.2
      training:
        optimizer: "adam"
        lr: 0.0005
        weight_decay: 0.0
        use_lr_scheduler: false
        batch_size: 512
        epochs: 500
        patience: 50
        davis_cold_patience: 80
        monitor: "mse"
        save_last_every: 10
        num_workers: 2
        pin_memory: true
        persistent_workers: true
    ```
- Em `src/physdual/data/featurizer_drug.py` (D2):
  - Suportar `featurizer_version: Literal["v2_onehot78_l1norm", "v2_onehot78_raw", "v1_int9"] = "v2_onehot78_l1norm"` e `normalize_atom_features: bool = True`.
  - Nas versões `v2_onehot78_*`: implementar a codificação exata de 78 dimensões de `thinng/GraphDTA` (`create_data.py:L230-L256`: 44 símbolos atômicos com `'Unknown'`, 11 graus `0..10`, 11 H totais `0..10` com unk, 11 valências implícitas `0..10` com unk, 1 aromaticidade):
    - Quando `featurizer_version == "v2_onehot78_l1norm"` (`normalize_atom_features=True`): aplicar **guarda contra soma zero** (`row_sum = float(feature.sum()); if row_sum > 0.0: feature = feature / row_sum`), reproduzindo `create_data.py:L256` onde cada linha soma `1.0` (deixando de ser one-hot binário estrito).
    - Quando `featurizer_version == "v2_onehot78_raw"` (`normalize_atom_features=False`): manter os valores binários `{0.0, 1.0}`, cuja soma por linha é `4.0` (átomo não-aromático) ou `5.0` (átomo aromático).
    - Manter `edge_index` e `edge_attr` (`[E, 4]`) intactos.
  - Manter `"v1_int9"` (9 dimensões escalares) acessível por parâmetro para eventual ablação futura.
  - `get_drug_graph_cache_filename(dataset_name: str, featurizer_version: str) -> str`: retorna `f"{dataset_name}_drug_graphs_{featurizer_version}.pt"`.
  - Em `build_drug_graph_cache(..., featurizer_version="v2_onehot78_l1norm", normalize_atom_features=True)`: incluir em cada `graph_dict` as chaves `"featurizer_version": featurizer_version`, `"atom_feature_dim": int(x.size(1))` e `"normalize_atom_features": bool(normalize_atom_features)`.
  - Em `load_drug_graph_cache(cache_path: Path, expected_version: str = "v2_onehot78_l1norm", expected_dim: int = 78, expected_normalize: bool = True)`: carregar com `weights_only=True` e validar que todos os grafos possuem `featurizer_version == expected_version`, `x.size(1) == expected_dim` e `normalize_atom_features == expected_normalize`, levantando `ValueError` claro se qualquer um divergir do config.
- Em `src/physdual/data/datasets.py` (Auditoria de duplicatas de conteúdo e mutantes):
  - Adicionar `audit_content_and_target_duplicates(df: pd.DataFrame) -> dict[str, Any]` retornando:
    - `num_unique_drugs_by_id`, `num_unique_drugs_by_smiles`
    - `num_unique_targets_by_id`, `num_unique_targets_by_sequence`
    - `shared_sequence_target_groups`: lista de grupos de `target_id`s que compartilham a mesma sequência (ex.: variantes/mutantes ou isoformas fosforiladas no Davis)
    - `duplicate_smiles_seq_pairs_count`: total de pares `(smiles, sequence)` repetidos
    - `duplicate_pairs_diff_affinity_count`: quantos pares `(smiles, sequence)` repetidos possuem desvio-padrão de afinidade $> 0$
    - `duplicate_policy`: `"keep_all"` (mantém todos os pares para preservar a volumetria canônica 30.056 / 118.254; como os splits cold agrupam por `smiles` e `sequence`, duplicatas nunca vazam entre partições).

- [x] **Step 1: Atualizar `tests/test_featurizer_drug.py` e `tests/test_datasets.py` para cobrir `v2_onehot78_l1norm` (soma 1.0), `v2_onehot78_raw` (soma 4.0 ou 5.0), `v1_int9` (9d), validação de versão/dimensão/flag e auditoria de duplicatas**

```python
# Trecho adicional em tests/test_featurizer_drug.py
def test_smiles_to_graph_dict_v2_onehot78_l1norm_and_raw_aspirin():
    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    # Com normalização L1 (padrão oficial v2_onehot78_l1norm): linhas somam 1.0
    g_l1 = smiles_to_graph_dict(
        aspirin,
        drug_id="ASP",
        featurizer_version="v2_onehot78_l1norm",
        normalize_atom_features=True,
    )
    assert g_l1["x"].shape == (13, 78)
    assert g_l1["edge_index"].shape == (2, 26)
    assert g_l1["edge_attr"].shape == (26, 4)
    assert g_l1["featurizer_version"] == "v2_onehot78_l1norm"
    assert g_l1["atom_feature_dim"] == 78
    assert g_l1["normalize_atom_features"] is True
    row_sums_l1 = g_l1["x"].sum(dim=1)
    assert torch.allclose(row_sums_l1, torch.ones_like(row_sums_l1), atol=1e-5)

    # Sem normalização L1 (v2_onehot78_raw): linhas somam 4.0 (não-aromático) ou 5.0 (aromático)
    g_raw = smiles_to_graph_dict(
        aspirin,
        drug_id="ASP",
        featurizer_version="v2_onehot78_raw",
        normalize_atom_features=False,
    )
    assert g_raw["x"].shape == (13, 78)
    assert g_raw["normalize_atom_features"] is False
    row_sums_raw = g_raw["x"].sum(dim=1)
    assert all(val in (4.0, 5.0) for val in row_sums_raw.tolist())

    # Versão legada v1_int9 preservada para ablação
    g_v1 = smiles_to_graph_dict(aspirin, drug_id="ASP", featurizer_version="v1_int9", normalize_atom_features=False)
    assert g_v1["x"].shape == (13, 9)


def test_versioned_cache_rejects_mismatched_version_dim_or_norm_flag(tmp_path):
    out_path = tmp_path / "davis_drug_graphs_v2_onehot78_raw.pt"
    build_drug_graph_cache(
        ["CCO"],
        out_path,
        drug_ids=["D1"],
        featurizer_version="v2_onehot78_raw",
        normalize_atom_features=False,
    )

    with pytest.raises(ValueError, match="featurizer_version|normalize_atom_features|dim"):
        load_drug_graph_cache(
            out_path,
            expected_version="v2_onehot78_l1norm",
            expected_dim=78,
            expected_normalize=True,
        )
```

- [x] **Step 2: Executar testes e verificar que falham**

Run: `pytest tests/test_featurizer_drug.py tests/test_datasets.py -v`
Expected: FAIL (assinatura `featurizer_version` e validação ainda não implementadas)

- [x] **Step 3: Implementar as atualizações em `configs/base.yaml`, `src/physdual/data/featurizer_drug.py`, `src/physdual/data/datasets.py`, `scripts/run_phase1_data.py` (adicionando flag `--only-drug-graphs` para regenerar apenas o cache de grafos v2 sem tocar no ESM-2 ou nos splits), `scripts/verify_phase1.py` e registrar ADRs #10 e #11 em `docs/DECISOES.md`**

- [x] **Step 4: Executar testes da Fase 1 atualizados e verificar que passam**

Run: `pytest tests/test_featurizer_drug.py tests/test_datasets.py tests/test_verify_phase1.py -v`
Expected: PASS

- [x] **Step 5: Commit da Task 0 (Emenda à Fase 1)**

```bash
git add configs/base.yaml src/physdual/data/featurizer_drug.py src/physdual/data/datasets.py scripts/run_phase1_data.py scripts/verify_phase1.py docs/DECISOES.md tests/test_featurizer_drug.py tests/test_datasets.py
git commit -m "feat(data): amend drug featurizer to 78d GraphDTA one-hot (v2) with versioned cache and duplicate audit"
```

---

### Task 1: Módulo de Métricas DTA, Subconjunto Não-Censurado, Faixas de Comprimento e Baselines Triviais

**Files:**
- Create: `src/physdual/eval/metrics.py`
- Create: `src/physdual/eval/__init__.py`
- Modify: `requirements-colab.txt` (confirmar presença de `lifelines`)
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: Tensores PyTorch (com `.detach().cpu()`) ou arrays NumPy 1D `y_true` e `y_pred`
- Produces:
  - `_to_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray`: usa obrigatoriamente `x.detach().cpu().numpy()` quando `x` for `torch.Tensor`.
  - `compute_ci(y_true, y_pred) -> float`: delega para `lifelines.utils.concordance_index` quando disponível; possui fallback vetorizado próprio documentando que a matriz de pares em broadcasting consome $O(N^2)$ de RAM para conjuntos de teste muito grandes (ex.: KIBA test com ~11.825 pares $\approx 1{,}1\text{ GB}$ se feito de uma vez, portanto o fallback opera em blocos/chunks de linhas para manter memória $O(N \cdot B)$).
  - `compute_mse(y_true, y_pred) -> float`
  - `compute_rmse(y_true, y_pred) -> float`
  - `compute_pearson(y_true, y_pred) -> float`
  - `compute_spearman(y_true, y_pred) -> float`
  - `compute_rm2(y_true, y_pred) -> float`:
    - Fórmula: $r_m^2 = r^2 \cdot (1 - \sqrt{|r^2 - r_0^2|})$ para $r > 0$ e denominadores $> 0$, retornando `0.0` quando $r \le 0$ ou variância nula.
    - *Nota documentada na docstring:* A guarda $r \le 0 \to 0.0$ afasta-se deliberadamente da implementação original de Roy et al. / DeepDTA (que eleva $r < 0$ ao quadrado e pode produzir $r_m^2 > 0$ para predições anticorrelacionadas), evitando reportar ajuste positivo espúrio quando a ordenação está invertida.
  - `compute_all_metrics(y_true, y_pred) -> dict[str, float]` (todos os valores retornados como `float` nativo Python).
  - `compute_uncensored_metrics(y_true, y_pred, censor_value: float = 5.0, tol: float = 1e-4) -> dict[str, float | int]`: filtra os pares com $|y_{\text{true}} - \text{censor\_value}| > \text{tol}$ (no Davis, os ~30,36% pares ativos com $pK_d > 5.0$) e calcula `ci`, `mse`, `rmse`, `pearson`, `spearman`, `rm2` e `num_uncensored_pairs`.
  - `compute_metrics_by_seq_length(y_true, y_pred, seq_lengths: np.ndarray | list[int], threshold: int = 1000) -> dict[str, dict[str, float | int]]`: calcula as métricas separadamente para `le_1000` ($\text{len} \le 1000$) e `gt_1000` ($\text{len} > 1000$), permitindo auditar o efeito do truncamento em 1000 resíduos sobre quinases longas.
  - `compute_trivial_baselines(y_train, y_test, X_train_aa: np.ndarray | None = None, X_test_aa: np.ndarray | None = None) -> dict[str, dict[str, float]]`:
    - Calcula as métricas no teste para:
      1. `mean_predictor`: prediz constantemente `float(np.mean(y_train))`.
      2. `ridge_aa_composition` (se `X_train_aa` e `X_test_aa` forem fornecidos): regressão `Ridge(alpha=1.0)` sobre frequências de aminoácidos (25d) concatenadas a descritores simples de comprimento/átomos.

- [x] **Step 1: Escrever testes unitários em `tests/test_metrics.py`**

```python
import numpy as np
import pytest

from physdual.eval.metrics import (
    compute_all_metrics,
    compute_ci,
    compute_metrics_by_seq_length,
    compute_mse,
    compute_pearson,
    compute_rm2,
    compute_rmse,
    compute_spearman,
    compute_trivial_baselines,
    compute_uncensored_metrics,
)


def test_standard_metrics_and_native_float_types():
    y_true = np.array([5.0, 6.0, 7.0, 8.0, 9.0], dtype=float)
    y_pred = np.array([5.2, 5.9, 7.1, 7.8, 9.2], dtype=float)

    all_m = compute_all_metrics(y_true, y_pred)
    assert all(type(v) is float for v in all_m.values())
    assert all_m["mse"] == pytest.approx(0.028, abs=1e-3)
    assert all_m["rmse"] == pytest.approx(np.sqrt(0.028), abs=1e-3)
    assert all_m["pearson"] > 0.99
    assert all_m["spearman"] == pytest.approx(1.0, abs=1e-4)
    assert all_m["ci"] == pytest.approx(1.0, abs=1e-4)
    assert all_m["rm2"] > 0.95


def test_concordance_index_matches_lifelines():
    lifelines = pytest.importorskip("lifelines")
    rng = np.random.RandomState(42)
    y_true = np.array([5.0, 5.0, 5.0, 6.0, 7.0, 8.0, 8.0, 9.0, 10.0])
    y_pred = y_true + rng.normal(0, 0.5, size=len(y_true))

    expected_ci = lifelines.utils.concordance_index(y_true, y_pred)
    our_ci = compute_ci(y_true, y_pred)
    assert our_ci == pytest.approx(expected_ci, abs=1e-6)


def test_constant_predictions_and_negative_correlation():
    y_true = np.array([5.0, 6.0, 7.0, 8.0], dtype=float)
    y_pred_const = np.array([6.5, 6.5, 6.5, 6.5], dtype=float)

    metrics = compute_all_metrics(y_true, y_pred_const)
    assert metrics["pearson"] == 0.0
    assert metrics["spearman"] == 0.0
    assert metrics["rm2"] == 0.0
    assert metrics["ci"] == pytest.approx(0.5, abs=1e-4)

    # Guarda deliberada r <= 0 -> rm2 == 0.0
    y_pred_inv = np.array([8.0, 7.0, 6.0, 5.0], dtype=float)
    assert compute_rm2(y_true, y_pred_inv) == 0.0


def test_uncensored_and_seq_length_stratified_metrics():
    y_true = np.array([5.0, 5.0, 6.0, 7.5, 8.5, 9.0])
    y_pred = np.array([5.1, 5.2, 6.1, 7.4, 8.6, 8.9])
    seq_lens = [400, 1200, 500, 800, 1150, 1300]

    unc = compute_uncensored_metrics(y_true, y_pred, censor_value=5.0)
    assert unc["num_uncensored_pairs"] == 4
    assert unc["ci"] == pytest.approx(1.0, abs=1e-4)

    by_len = compute_metrics_by_seq_length(y_true, y_pred, seq_lens, threshold=1000)
    assert by_len["le_1000"]["count"] == 3
    assert by_len["gt_1000"]["count"] == 3


def test_trivial_baselines():
    y_train = np.array([5.0, 5.0, 6.0, 8.0])
    y_test = np.array([5.0, 7.0, 9.0])
    res = compute_trivial_baselines(y_train, y_test)
    assert "mean_predictor" in res
    assert res["mean_predictor"]["mse"] == pytest.approx(np.mean((y_test - 6.0) ** 2))
```

- [x] **Step 2: Executar teste e verificar que falha**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'physdual.eval.metrics'`

- [x] **Step 3: Implementar `src/physdual/eval/metrics.py` e `src/physdual/eval/__init__.py`**

- [x] **Step 4: Executar teste e verificar que passa**

Run: `pytest tests/test_metrics.py -v`
Expected: PASS

- [x] **Step 5: Commit da Task 1**

```bash
git add src/physdual/eval/metrics.py src/physdual/eval/__init__.py requirements-colab.txt tests/test_metrics.py
git commit -m "feat(eval): implement DTA metrics, uncensored/length stratification, and trivial baselines"
```

---

### Task 2: Tokenizador de Sequências, `DTADataset` com Pré-Construção em Memória e DataLoaders Configuráveis

**Files:**
- Create: `src/physdual/data/datamodule.py`
- Test: `tests/test_datamodule.py`

**Interfaces:**
- Consumes:
  - `data/processed/{dataset}_clean.csv`
  - `data/processed/{dataset}_drug_graphs_{featurizer_version}.pt`
  - `data/processed/splits/{dataset}_{split_type}_seed{seed}.json`
- Produces:
  - `PROTEIN_ALPHABET = "ABCDEFGHIKLMNOPQRSTUVWXYZ"` (25 caracteres, mapeados de `1` a `25`; `0` reservado para padding).
  - `tokenize_protein_sequence(seq: str, max_len: int = 1000) -> np.ndarray` (`int64`, shape `(max_len,)`).
  - `compute_aa_composition(seq: str) -> np.ndarray` (`float32`, shape `(25,)`, frequências relativas de cada aminoácido para o baseline trivial Ridge).
  - `class DTADataset(torch.utils.data.Dataset)`:
    - **Eficiência de CPU/Memória:** No `__init__`, pré-constrói um dicionário `{smiles: Data}` uma única vez para os SMILES únicos do DataFrame e um dicionário `{sequence: Tensor}` uma única vez para as sequências únicas do DataFrame. O método `__getitem__(idx)` faz apenas lookup $O(1)$ de referências e retorna `{"drug_graph": Data, "target_tokens": Tensor, "seq_len": int, "smiles": str, "target_id": str, "drug_id": str, "affinity": Tensor}`.
    - Valida na inicialização que todos os SMILES da coluna `smiles` do CSV existem no `drug_graph_cache` (garantindo consistência de canonização).
  - `dta_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]`:
    - Empacota `drug_batch` via `Batch.from_data_list`, `target_tokens` `[B, max_len]`, `affinity` `[B]` e `seq_lengths` `[B]`.
  - `get_dta_dataloaders(dataset_name: str, split_type: str, seed: int, config: dict[str, Any], root: Path | None = None) -> tuple[DataLoader, DataLoader, DataLoader]`:
    - Configura `batch_size`, `num_workers`, `pin_memory` e `persistent_workers=(num_workers > 0)` a partir de `config["graphdta"]["training"]`.
  - `prepare_local_cache(tar_path: Path, local_cache_dir: Path) -> Path`:
    - Utilitário antecipado para a Fase 3 (descompactação de tarballs de cache no SSD `/content/local_cache/`).

- [ ] **Step 1: Escrever testes unitários em `tests/test_datamodule.py` (incluindo verificação de pré-construção única e consistência de canonização SMILES)**

```python
import numpy as np
import pandas as pd
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
    assert (tokens[6:] == 0).all()


def test_dta_dataset_prebuilds_unique_entities_and_collates():
    from torch_geometric.data import Batch

    cache = {
        "CCO": smiles_to_graph_dict("CCO", drug_id="D1", featurizer_version="v2_onehot78_l1norm"),
        "CCN": smiles_to_graph_dict("CCN", drug_id="D2", featurizer_version="v2_onehot78_l1norm"),
    }
    rows = [
        {"drug_id": "D1", "smiles": "CCO", "target_id": "T1", "sequence": "MKTV", "affinity": 7.5},
        {"drug_id": "D1", "smiles": "CCO", "target_id": "T1", "sequence": "MKTV", "affinity": 7.5},
        {"drug_id": "D2", "smiles": "CCN", "target_id": "T2", "sequence": "ACDE" * 300, "affinity": 6.2},
    ]
    df = pd.DataFrame(rows)
    ds = DTADataset(df, drug_graph_cache=cache, max_seq_len=1000)

    # Garante que Data e tokens foram pré-construídos apenas para entidades únicas
    assert len(ds.unique_graphs) == 2
    assert len(ds.unique_tokens) == 2
    assert ds[0]["drug_graph"] is ds[1]["drug_graph"]
    assert ds[2]["seq_len"] == 1200

    batch = dta_collate_fn([ds[0], ds[2]])
    assert isinstance(batch["drug_batch"], Batch)
    assert batch["drug_batch"].x.shape[1] == 78
    assert batch["target_tokens"].shape == (2, 1000)
    assert batch["seq_lengths"].tolist() == [4, 1200]
```

- [ ] **Step 2: Executar teste e verificar que falha**

Run: `pytest tests/test_datamodule.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'physdual.data.datamodule'`

- [ ] **Step 3: Implementar `src/physdual/data/datamodule.py`**

- [ ] **Step 4: Executar teste e verificar que passa**

Run: `pytest tests/test_datamodule.py -v`
Expected: PASS

- [ ] **Step 5: Commit da Task 2**

```bash
git add src/physdual/data/datamodule.py tests/test_datamodule.py
git commit -m "feat(data): implement pre-cached DTADataset, protein tokenizer, and configurable PyG DataLoaders"
```

---

### Task 3: Baseline Oficial `graphdta_gin` (Fiel ao Repositório Original) e Variante Secundária `graphdta_cnn3` (D1)

**Files:**
- Create: `src/physdual/models/graphdta.py`
- Create: `src/physdual/models/__init__.py`
- Test: `tests/test_graphdta_model.py`

**Interfaces:**
- Consumes: `drug_batch` (`Batch` PyG com `x` de dimensão `drug_in_dim` vinda do config) e `target_tokens` (`Tensor [B, max_seq_len]`)
- Produces:
  - **Restrição D2:** `drug_in_dim: int` é argumento posicional/obrigatório **sem valor default mágico** nas classes de modelo.
  - `class GINEncoder(nn.Module)`:
    - 5 camadas `GINConv(Sequential(Linear(in_d, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim)))` com `hidden_dim=32`, cada uma seguida de `ReLU` e `BatchNorm1d(32)`.
    - `global_add_pool(x, batch)` $\rightarrow$ `ReLU(Linear(32, output_dim))` (`output_dim=128`) $\rightarrow$ `Dropout(p=dropout)`.
  - `class OfficialGraphDTATargetEncoder(nn.Module)` (usado em `graphdta_gin`):
    - Reproduz fielmente `models/ginconv.py` (`GINConvNet`) do repositório oficial `thinng/GraphDTA`:
      - `embedding_xt = nn.Embedding(protein_vocab_size + 1, embedding_dim)` (`26, 128`).
      - `conv_xt_1 = nn.Conv1d(in_channels=max_seq_len, out_channels=n_filters, kernel_size=kernel_size)` (`in_channels=1000, out_channels=32, kernel_size=8`), aplicando a convolução 1D diretamente sobre o tensor `[B, 1000, 128]` sem transposição.
      - Comprimento de saída convolucional: $L_{\text{out}} = \text{embedding\_dim} - \text{kernel\_size} + 1 = 128 - 8 + 1 = 121$.
      - Flatten: `xt = conv_xt.view(-1, n_filters * L_out)` (`32 * 121 = 3872`), seguido de `fc1_xt = nn.Linear(n_filters * L_out, output_dim)` (`3872 -> 128`).
  - `class DeepDTAStyleCNN3TargetEncoder(nn.Module)` (usado na variante secundária `graphdta_cnn3`):
    - Rótulo explícito: *"CNN de 3 camadas estilo DeepDTA"*.
    - `nn.Embedding(protein_vocab_size + 1, embedding_dim, padding_idx=0)` $\rightarrow$ `.transpose(1, 2)` (`[B, 128, max_seq_len]`) $\rightarrow$ 3 blocos `Conv1d + ReLU` (`128->32 k=4`, `32->64 k=8`, `64->96 k=12`) $\rightarrow$ `AdaptiveMaxPool1d(1).squeeze(-1)` $\rightarrow$ `nn.Linear(96, output_dim)`.
  - `class GraphDTAGIN(nn.Module)` (**Baseline Oficial `graphdta_gin`**):
    - Combina `GINEncoder` + `OfficialGraphDTATargetEncoder` + MLP fiel ao `GINConvNet` original:
      `Linear(256, 1024) -> ReLU -> Dropout(0.2) -> Linear(1024, 256) -> ReLU -> Dropout(0.2) -> Linear(256, 1)`.
  - `class GraphDTACNN3(nn.Module)` (**Variante Secundária `graphdta_cnn3`**):
    - Combina `GINEncoder` + `DeepDTAStyleCNN3TargetEncoder` + MLP (`256 -> 1024 -> 256 -> 1`).
  - `build_baseline_model(model_name: Literal["graphdta_gin", "graphdta_cnn3"], config: dict[str, Any]) -> nn.Module`:
    - Lê `drug_in_dim = config["featurizer"]["atom_feature_dim"]` e demais hiperparâmetros de `config["graphdta"]`.

- [ ] **Step 1: Escrever testes unitários em `tests/test_graphdta_model.py` (incluindo verificação de ausência de default para `drug_in_dim`, forward/backward de ambos os modelos e overfit de um lote pequeno)**

```python
import inspect
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from physdual.data.datamodule import DTADataset, dta_collate_fn
from physdual.data.featurizer_drug import smiles_to_graph_dict
from physdual.models.graphdta import GraphDTACNN3, GraphDTAGIN, build_baseline_model


def test_drug_in_dim_has_no_magic_default():
    sig_gin = inspect.signature(GraphDTAGIN.__init__)
    sig_cnn3 = inspect.signature(GraphDTACNN3.__init__)
    assert sig_gin.parameters["drug_in_dim"].default is inspect.Parameter.empty
    assert sig_cnn3.parameters["drug_in_dim"].default is inspect.Parameter.empty


@pytest.mark.parametrize("model_cls", [GraphDTAGIN, GraphDTACNN3])
def test_both_models_forward_backward_and_small_batch_overfit(model_cls):
    torch.manual_seed(42)
    cache = {
        "CCO": smiles_to_graph_dict("CCO", drug_id="D1", featurizer_version="v2_onehot78_l1norm"),
        "CCN": smiles_to_graph_dict("CCN", drug_id="D2", featurizer_version="v2_onehot78_l1norm"),
    }
    df = pd.DataFrame([
        {"drug_id": "D1", "smiles": "CCO", "target_id": "T1", "sequence": "MKTVRQ", "affinity": 7.0},
        {"drug_id": "D2", "smiles": "CCN", "target_id": "T2", "sequence": "ACDEFG", "affinity": 9.0},
    ])
    ds = DTADataset(df, drug_graph_cache=cache, max_seq_len=1000)
    batch = dta_collate_fn([ds[0], ds[1]])

    model = model_cls(drug_in_dim=78, max_seq_len=1000)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    initial_loss = None
    final_loss = None
    for step in range(25):
        optimizer.zero_grad()
        preds = model(batch["drug_batch"], batch["target_tokens"])
        loss = torch.nn.functional.mse_loss(preds, batch["affinity"])
        if step == 0:
            initial_loss = loss.item()
            loss.backward()
            assert any(p.grad is not None and p.grad.norm().item() > 0 for p in model.drug_encoder.parameters())
            assert any(p.grad is not None and p.grad.norm().item() > 0 for p in model.target_encoder.parameters())
            optimizer.step()
        else:
            loss.backward()
            optimizer.step()
        final_loss = loss.item()

    assert final_loss < initial_loss * 0.2, f"Overfit falhou: {initial_loss} -> {final_loss}"
```

- [ ] **Step 2: Executar teste e verificar que falha**

Run: `pytest tests/test_graphdta_model.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'physdual.models.graphdta'`

- [ ] **Step 3: Implementar `src/physdual/models/graphdta.py` e `src/physdual/models/__init__.py`**

- [ ] **Step 4: Executar teste e verificar que passa**

Run: `pytest tests/test_graphdta_model.py -v`
Expected: PASS

- [ ] **Step 5: Commit da Task 3**

```bash
git add src/physdual/models/graphdta.py src/physdual/models/__init__.py tests/test_graphdta_model.py
git commit -m "feat(models): implement official GraphDTAGIN and secondary GraphDTACNN3 variant"
```

---

### Task 4: Motor de Treinamento, Isolamento Estrito de Teste (C1), Checkpointing Atômico no Mesmo Filesystem (C4) e Early Stopping

**Files:**
- Create: `src/physdual/training/checkpoint.py`
- Create: `src/physdual/training/trainer.py`
- Create: `src/physdual/training/__init__.py`
- Test: `tests/test_trainer_checkpoint.py`

**Interfaces:**
- Consumes: `GraphDTAGIN` / `GraphDTACNN3`, DataLoaders, `compute_all_metrics`
- Produces:
  - `_sanitize_state_dict(obj: Any) -> Any`: converte recursivamente qualquer escalar NumPy (`np.generic`, arrays 0-d) para `float` / `int` nativos do Python.
  - `save_checkpoint(state: dict[str, Any], path: Path) -> Path` (C4):
    - Sanitiza `state` com `_sanitize_state_dict`.
    - Cria o arquivo temporário **obrigatoriamente no mesmo diretório de destino** (`tmp_path = path.parent / f".tmp_{path.name}_{os.getpid()}"`), grava com `torch.save(sanitized, tmp_path)` e aplica `os.replace(tmp_path, path)`, garantindo operação atômica real no mesmo filesystem.
    - Inclui `torch_rng_state = torch.get_rng_state()` e, quando `torch.cuda.is_available()`, `cuda_rng_state_all = torch.cuda.get_rng_state_all()` (lista de tensores `uint8`).
  - `sync_checkpoint_to_drive(local_ckpt: Path, drive_ckpt: Path) -> Path` (C4):
    - Copia de `local_ckpt` (ex.: em `/content/local_checkpoints/`) para um arquivo temporário `drive_ckpt.parent / f".tmp_{drive_ckpt.name}"` no Drive e executa `os.replace` dentro do Drive, evitando escrita não-atômica entre filesystems.
  - `load_checkpoint(path: Path, model: nn.Module, optimizer: Optimizer | None = None, restore_rng: bool = False) -> dict[str, Any]`:
    - Carrega estritamente com `torch.load(path, map_location="cpu", weights_only=True)`.
  - `get_monitor_mode(monitor: str) -> Literal["min", "max"]`
  - `class EarlyStopping`:
    - Construtor: `__init__(self, patience: int = 50, mode: Literal["min", "max"] = "min", delta: float = 1e-4)`
    - Atributos públicos: `best: float`, `counter: int`, `should_stop: bool`
    - Método: `step(self, metric: float) -> bool`
    - **Regra explícita para `delta`:** `best` só é atualizado e `counter` só é zerado quando a melhora supera estritamente `delta` (`metric < self.best - self.delta` no modo `"min"`, ou `metric > self.best + self.delta` no modo `"max"`). Se a variação for menor ou igual a `delta`, `best` mantém seu valor anterior e `counter` é incrementado em 1. Retorna `self.should_stop`.
  - `train_model(model, train_loader, val_loader, test_loader, config, checkpoint_dir, ...) -> dict[str, Any]` (C1):
    - **Isolamento estrito do teste:** durante o loop de épocas (`1..epochs`), avalia **apenas** `val_loader`. `test_loader` **nunca** é iterado dentro do loop de épocas.
    - **LR constante por padrão:** `use_lr_scheduler: false` por padrão (mantendo `Adam` com `lr=0.0005` constante fiel ao GraphDTA original; `ReduceLROnPlateau` só é instanciado se `use_lr_scheduler=True` for solicitado explicitamente).
    - **Paciência para splits cold no Davis:** quando `dataset_name == "davis"` e `split_type in ("cold_drug", "cold_target")`, utiliza `davis_cold_patience` (`80` épocas) em vez da `patience` padrão (`50`).
    - **Alerta de proximidade do teto de épocas:** registra `best_epoch` e emite `[WARN]` (e grava `near_epoch_ceiling_warn: true` no resultado da run) quando `best_epoch > 0.8 * epochs`.
    - Ao encerrar o loop de épocas (por `EarlyStopping` ou `epochs` máximo), recarrega `checkpoint_best.pt` com `weights_only=True` e avalia `test_loader` **uma única vez**, calculando métricas completas, subconjunto não-censurado (para Davis) e estratificação por faixa de comprimento ($\le 1000$ vs. $> 1000$).

- [ ] **Step 1: Escrever testes unitários em `tests/test_trainer_checkpoint.py` (incluindo sanitização NumPy, atomicidade no mesmo diretório, `EarlyStopping` com `delta`, isolamento estrito do `test_loader` e 1–2 épocas em subconjunto real do Davis)**

```python
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from physdual.data.datamodule import DTADataset, dta_collate_fn
from physdual.data.featurizer_drug import smiles_to_graph_dict
from physdual.models.graphdta import GraphDTAGIN
from physdual.training.checkpoint import load_checkpoint, save_checkpoint
from physdual.training.trainer import EarlyStopping, get_monitor_mode, train_model


def test_checkpoint_roundtrip_weights_only_with_numpy_and_rng(tmp_path):
    model = GraphDTAGIN(drug_in_dim=78, max_seq_len=1000)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)

    ckpt_path = tmp_path / "ckpt_test.pt"
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
    # Garante que nenhum arquivo temporário residual ficou no diretório
    assert list(tmp_path.glob(".tmp_*")) == []

    new_model = GraphDTAGIN(drug_in_dim=78, max_seq_len=1000)
    loaded_meta = load_checkpoint(ckpt_path, model=new_model, optimizer=None, restore_rng=True)
    assert loaded_meta["epoch"] == 5
    assert type(loaded_meta["epoch"]) is int
    assert type(loaded_meta["val_metrics"]["ci"]) is float
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1, p2)


def test_early_stopping_both_modes_and_delta_rule():
    assert get_monitor_mode("mse") == "min"
    assert get_monitor_mode("ci") == "max"

    es_min = EarlyStopping(patience=2, mode="min", delta=0.01)
    assert not es_min.step(0.50)
    assert es_min.best == pytest.approx(0.50)
    # Melhora de 0.005 <= delta (0.01): NÃO atualiza best e incrementa counter
    assert not es_min.step(0.495)
    assert es_min.best == pytest.approx(0.50)
    assert es_min.counter == 1
    assert es_min.step(0.52)
    assert es_min.should_stop


def test_train_model_evaluates_test_loader_only_once_at_end(tmp_path):
    cache = {
        "CCO": smiles_to_graph_dict("CCO", drug_id="D1", featurizer_version="v2_onehot78_l1norm"),
        "CCN": smiles_to_graph_dict("CCN", drug_id="D2", featurizer_version="v2_onehot78_l1norm"),
    }
    df = pd.DataFrame([
        {"drug_id": "D1", "smiles": "CCO", "target_id": "T1", "sequence": "MKTVRQ", "affinity": 5.0},
        {"drug_id": "D2", "smiles": "CCN", "target_id": "T2", "sequence": "ACDEFG", "affinity": 7.5},
    ])
    ds = DTADataset(df, drug_graph_cache=cache, max_seq_len=1000)
    base_loader = torch.utils.data.DataLoader(ds, batch_size=2, collate_fn=dta_collate_fn)

    class CountingLoaderWrapper:
        def __init__(self, loader):
            self.loader = loader
            self.iter_count = 0

        def __iter__(self):
            self.iter_count += 1
            return iter(self.loader)

        def __len__(self):
            return len(self.loader)

    test_spy = CountingLoaderWrapper(base_loader)
    model = GraphDTAGIN(drug_in_dim=78, max_seq_len=1000)
    result = train_model(
        model=model,
        train_loader=base_loader,
        val_loader=base_loader,
        test_loader=test_spy,
        epochs=3,
        lr=5e-4,
        patience=5,
        monitor="mse",
        checkpoint_dir=tmp_path / "ckpts",
        device="cpu",
    )
    assert test_spy.iter_count == 1
    assert "test_metrics" in result
    assert "best_epoch" in result
```

- [ ] **Step 2: Executar teste e verificar que falha**

Run: `pytest tests/test_trainer_checkpoint.py -v`
Expected: FAIL com `ModuleNotFoundError`

- [ ] **Step 3: Implementar `src/physdual/training/checkpoint.py`, `src/physdual/training/trainer.py` e `src/physdual/training/__init__.py` (e adicionar teste de 1–2 épocas em subconjunto real do Davis quando `data/processed/davis_clean.csv` existir)**

- [ ] **Step 4: Executar teste e verificar que passa**

Run: `pytest tests/test_trainer_checkpoint.py -v`
Expected: PASS

- [ ] **Step 5: Commit da Task 4**

```bash
git add src/physdual/training/ tests/test_trainer_checkpoint.py
git commit -m "feat(training): implement training engine with strict test isolation and atomic checkpointing"
```

---

### Task 5: Orquestrador Retomável por Run, Agregador Separado, Medição de Tempo e Verificador `verify_phase2.py` (C1, C2)

**Files:**
- Create: `scripts/run_phase2_baseline.py`
- Create: `scripts/verify_phase2.py`
- Create: `notebooks/02_graphdta_baseline.ipynb`
- Test: `tests/test_verify_phase2.py`

**Interfaces & Ordem de Execução (C2):**
- **Persistência individual por run (Idempotente e Retomável):**
  - Cada combinação `(model, dataset, split, seed)` grava ao terminar um arquivo JSON atômico em:
    `outputs/results/runs/{model}_{dataset}_{split}_seed{seed}.json`
  - Conteúdo obrigatório do JSON por run:
    ```json
    {
      "model": "graphdta_gin",
      "dataset": "davis",
      "split": "random",
      "seed": 0,
      "featurizer_version": "v2_onehot78_l1norm",
      "normalize_atom_features": true,
      "drug_in_dim": 78,
      "effective_config": { "...": "..." },
      "best_epoch": 142,
      "total_epochs_run": 192,
      "near_epoch_ceiling_warn": false,
      "elapsed_sec": 215.4,
      "sec_per_epoch": 1.12,
      "val_metrics_at_best": {"ci": 0.889, "mse": 0.221, "rmse": 0.470, "pearson": 0.845, "spearman": 0.701, "rm2": 0.692},
      "test_metrics_full": {"ci": 0.885, "mse": 0.232, "rmse": 0.481, "pearson": 0.839, "spearman": 0.692, "rm2": 0.678},
      "test_metrics_uncensored": {"num_uncensored_pairs": 912, "ci": 0.795, "mse": 0.410, "rmse": 0.640, "pearson": 0.760, "spearman": 0.745, "rm2": 0.540},
      "test_metrics_by_seq_len": {
        "le_1000": {"count": 2650, "ci": 0.887, "mse": 0.228},
        "gt_1000": {"count": 356, "ci": 0.871, "mse": 0.262}
      },
      "trivial_baselines": {
        "mean_predictor": {"ci": 0.5, "mse": 0.815},
        "ridge_aa_composition": {"ci": 0.612, "mse": 0.710}
      }
    }
    ```
  - Se `outputs/results/runs/{model}_{dataset}_{split}_seed{seed}.json` já existir e `--force` não for passado, o orquestrador **pula** a run imediatamente. Se não existir, mas houver `checkpoint_last.pt`, retoma a partir da última época salva.
- **Agregação como Etapa Separada:**
  - `python scripts/run_phase2_baseline.py --aggregate-only` (também chamada ao final de um lote) lê os JSONs em `outputs/results/runs/` das 5 seeds `[0, 1, 2, 3, 4]` e gera:
    `outputs/results/baseline_{model}_{dataset}_{split}.json` contendo `per_seed_results`, `aggregated` (`mean` e `std` para `test_metrics_full`, `test_metrics_uncensored`, `test_metrics_by_seq_len`, `trivial_baselines`, `best_epoch` e `sec_per_epoch`).
- **Passo Obrigatório de Medição, Run de Controle Opcional (1000 épocas) e Ordem Condicional de Execução no Colab:**
  1. **Passo 1 (Medição):** `python scripts/run_phase2_baseline.py --benchmark-epochs 3 --dataset davis,kiba --model graphdta_gin` mede `sec_per_epoch` em Davis e KIBA e salva `outputs/results/phase2_timing_benchmark.json`.
  2. **Passo 2 (Oficial Davis):** `graphdta_gin` no Davis (`random`, `cold_drug`, `cold_target` $\times$ 5 seeds = 15 runs, com `epochs=500`, `patience=50` para `random` e `davis_cold_patience=80` para `cold_drug` e `cold_target`).
  3. **Passo 2b Opcional (Run de Controle de 1000 Épocas):** `python scripts/run_phase2_baseline.py --dataset davis --split random --seeds 0 --epochs 1000 --patience 200 --model graphdta_gin --tag control_1000ep` para verificar empiricamente se há ganho relevante entre 500 e 1000 épocas sob seleção por validação.
  4. **Passo 3 (Oficial KIBA):** `graphdta_gin` no KIBA (`random`, `cold_drug`, `cold_target` $\times$ 5 seeds = 15 runs).
  5. **Passo 4 (Variante Secundária Davis):** `graphdta_cnn3` no Davis (3 splits $\times$ 5 seeds = 15 runs).
  6. **Passo 5 Condicional (Variante Secundária KIBA):** Executar `graphdta_cnn3` no KIBA **somente se** o tempo medido por época no KIBA for $\le 15\text{ s/época}$ (ou se o tempo estimado para 15 runs for $\le 4\text{ horas}$ de sessão); caso contrário, pular e registrar a justificativa de orçamento computacional em `docs/FASE2_BASELINE.md`.
- **Verificador `scripts/verify_phase2.py` (`--smoke` e `--full`):**
  - No teste `tests/test_verify_phase2.py`, resolver o caminho do script via `Path(__file__).resolve().parents[1] / "scripts" / "verify_phase2.py"` (sem depender do diretório atual e sem meta de tempo artificial).
  - No modo `--full`:
    - O critério de CI $\ge 0{,}86$ no Davis `random` emite **`[WARN]` (alerta de sanidade)** caso fique abaixo, sem reprovar o código de saída (`exit code 0`).
    - Caso alguma run apresente `best_epoch > 0.8 * epochs`, emite **`[WARN]`** reportando quais runs atingiram a zona próxima ao teto de épocas.
    - **Não exige queda obrigatória** nos splits cold; apenas reporta os valores observados.

- [ ] **Step 1: Escrever testes unitários em `tests/test_verify_phase2.py` (testando `--smoke`, idempotência de runs concluídas e agregação multi-seed)**

```python
from pathlib import Path
import subprocess
import sys
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")


def test_verify_phase2_smoke_mode():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "verify_phase2.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "--smoke"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"verify_phase2 --smoke falhou:\n{result.stdout}\n{result.stderr}"
    assert "RESULTADO FINAL: APROVADO" in result.stdout
```

- [ ] **Step 2: Executar teste e verificar que falha**

Run: `pytest tests/test_verify_phase2.py -v`
Expected: FAIL (script `verify_phase2.py` ainda não existe)

- [ ] **Step 3: Implementar `scripts/run_phase2_baseline.py`, `scripts/verify_phase2.py` e `notebooks/02_graphdta_baseline.ipynb`**

- [ ] **Step 4: Executar testes e validar CLIs**

Run: `pytest tests/test_verify_phase2.py -v`
Expected: PASS

- [ ] **Step 5: Commit da Task 5**

```bash
git add scripts/run_phase2_baseline.py scripts/verify_phase2.py notebooks/02_graphdta_baseline.ipynb tests/test_verify_phase2.py
git commit -m "feat(pipeline): add resumable per-run Phase 2 orchestrator, aggregator, and gate verifier"
```

---

### Task 6: Documentação Científica da Fase 2, Tabelas Verificadas de Literatura e ADRs

**Files:**
- Create: `docs/FASE2_BASELINE.md`
- Modify: `docs/DECISOES.md` (ADRs #10, #11 e #12)
- Modify: `README.md`
- Modify: `docs/PROXIMOS_PASSOS.md`

- [ ] **Step 1: Escrever `docs/FASE2_BASELINE.md` contendo:**
  1. **Especificação exata do Baseline Oficial (`graphdta_gin`) vs. Variante Secundária (`graphdta_cnn3`):**
     - Documentar os valores extraídos diretamente de `thinng/GraphDTA` (`commit 6162245c6776684b391d30610850fe7077478435`) e deixar claro que todas as tabelas destacam `graphdta_gin` como baseline oficial e rotulam `graphdta_cnn3` apenas como referência adicional ("GraphDTA-style com CNN estilo DeepDTA").
  2. **Tabelas 1 e 2 Verificadas de Nguyen et al. (2021):**
     - Citação: Nguyen, T. et al. (2021). *"GraphDTA: predicting drug–target binding affinity with graph neural networks"*, Bioinformatics, 37(8), 1140–1147, Tabelas 1 e 2.
     - **Tabela 1 Verificada (Hiperparâmetros do GraphDTA):**
       | Hyper-parameters | Setting |
       | :--- | :--- |
       | Learning rate | `0.0005` |
       | Batch size | `512` |
       | Optimizer | `Adam` |
       | GCN layers | `3` |
       | GIN layers | `5` |
       | GAT layers | `2` |
       | GAT_GCN layers | `2` |
     - **Tabela 2 Verificada (Desempenho no Dataset Davis — Setting 1 / Random Split):**
       | Method | Protein rep. | Compound rep. | CI | MSE |
       | :--- | :--- | :--- | :---: | :---: |
       | DeepDTA (Öztürk et al., 2018) | Smith-Waterman | Pubchem-Sim | 0.790 | 0.608 |
       | DeepDTA (Öztürk et al., 2018) | Smith-Waterman | 1D | 0.886 | 0.420 |
       | DeepDTA (Öztürk et al., 2018) | 1D | Pubchem-Sim | 0.835 | 0.419 |
       | KronRLS (Cichonska et al., 2017, 2018) | Smith-Waterman | Pubchem-Sim | 0.871 | 0.379 |
       | SimBoost (He et al., 2017) | Smith-Waterman | Pubchem-Sim | 0.872 | 0.282 |
       | DeepDTA (Öztürk et al., 2018) | 1D | 1D | 0.878 | 0.261 |
       | WideDTA (Öztürk et al., 2019) | 1D + PDM | 1D + LMCS | 0.886 | 0.262 |
       | **GraphDTA — GCN** | 1D | Graph | 0.880 | 0.254 |
       | **GraphDTA — GAT_GCN** | 1D | Graph | 0.881 | 0.245 |
       | **GraphDTA — GAT** | 1D | Graph | 0.892 | 0.232 |
       | **GraphDTA — GIN** | 1D | Graph | **0.893** | **0.229** |
     - Marcar a Tabela 3 (KIBA) e métricas/regimes não incluídos na Tabela 2 (Pearson, Spearman, $r_m^2$, `cold_drug`, `cold_target`) como **"a verificar em Nguyen et al. (2021)"**.
     - **Análise crítica da meta do projeto e diferença metodológica (ADR #12):**
       1. Registrar que o CI do `GIN` no Davis reportado na Tabela 2 (`0.893`) já está no patamar da meta original do projeto ($\text{CI} > 0{,}89$), demonstrando que o valor científico real do PhysDual-GNN reside na superação do baseline nos regimes **`cold_drug` e `cold_target`**, no subconjunto não-censurado e nas proteínas longas ($> 1000$ resíduos).
       2. Registrar que o script `training.py` original do GraphDTA seleciona a melhor época avaliando o conjunto de teste a cada época ao longo de 1.000 épocas (introduzindo viés otimista de seleção no teste), enquanto o nosso protocolo seleciona `checkpoint_best.pt` exclusivamente pelo conjunto de validação (`val_loader`, com `epochs=500`, `patience=50` e `davis_cold_patience=80` para `cold_drug` e `cold_target`, sem `ReduceLROnPlateau`) e avalia o teste uma única vez ao final — portanto, os números não são diretamente comparáveis sem considerar essa distinção metodológica.
  3. **Auditoria de Duplicatas, Mutantes e Limitação para a Fase 7 (ADR #11):**
     - Documentar que os 442 `target_id`s do Davis colapsam em 379 sequências únicas (e 2.111 fármacos do KIBA em 2.065 SMILES únicos). Explicar que se variantes mutantes ou estados fosforilados no Davis compartilham a sequência de aminoácidos do tipo selvagem (wild-type) no arquivo `proteins.txt` do DeepDTA, **o objetivo de resiliência a mutações (Fase 7) não é testável diretamente com essas entradas sem construir explicitamente as sequências mutadas** (substituindo o resíduo na posição da mutação).
  4. **Determinismo, Censura e Truncamento:**
     - Explicar como cada `seed` $\in \{0, 1, 2, 3, 4\}$ controla simultaneamente a partição de dados (`{dataset}_{split}_seed{seed}.json`) e a inicialização de pesos/DataLoader (`seed_everything(seed)`), e registrar que `global_add_pool`/`scatter_add` em CUDA possui determinismo aproximado devido a operações atômicas em ponto flutuante.
     - Documentar o impacto da censura em $pK_d = 5.0$ (69,64% do Davis) e do truncamento em 1000 resíduos (83 proteínas $> 1022$ aa).

- [ ] **Step 2: Atualizar `docs/DECISOES.md` (ADRs #10, #11 e #12), `README.md` e `docs/PROXIMOS_PASSOS.md`**

- [ ] **Step 3: Commit da Task 6**

```bash
git add docs/FASE2_BASELINE.md docs/DECISOES.md README.md docs/PROXIMOS_PASSOS.md
git commit -m "docs(phase2): document verified GraphDTA tables, ADR #10-#12, and duplicate/mutation analysis"
```

---

## Gate de Pronto da Fase 2 (Critérios de Aceite Obrigatórios)

1. **Métricas:** As 6 métricas (`ci`, `mse`, `rmse`, `pearson`, `spearman`, `rm2`) retornam `float` nativo do Python, sem `NaN` em predições degeneradas, com CI idêntico ao `lifelines.utils.concordance_index`.
2. **`--smoke`:** Forward e backward do modelo oficial (`graphdta_gin`) e da variante secundária (`graphdta_cnn3`) executam sem `NaN`, com gradientes não nulos nos dois ramos (droga e proteína) e com `drug_in_dim` lido obrigatoriamente do config.
3. **Checkpoints Seguros e Atômicos:** Checkpoints sanitizados de escalares NumPy, gravados atomicamente no mesmo diretório via `os.replace`, incluindo estados de RNG (`torch.get_rng_state()` e `torch.cuda.get_rng_state_all()`), e recarregáveis estritamente com `torch.load(..., weights_only=True)`.
4. **Cache de Grafos v2 (`v2_onehot78_l1norm`):** O cache `{dataset}_drug_graphs_v2_onehot78_l1norm.pt` carrega com `weights_only=True` e valida estritamente `featurizer_version == "v2_onehot78_l1norm"`, `atom_feature_dim == 78` e `normalize_atom_features is True`.
5. **`--full`:** JSON individual por run gravado em `outputs/results/runs/` para todos os runs do escopo, e JSONs agregados (`outputs/results/baseline_{model}_{dataset}_{split}.json`) com média ± desvio-padrão nas 5 seeds, incluindo métricas no subconjunto não-censurado, desempenho por faixa de comprimento ($\le 1000$ vs. $> 1000$) e baselines triviais.
6. **Alertas de Sanidade (`WARN`):** Alertas não-bloqueantes para CI no Davis `random` ($\text{CI} < 0{,}86 \Rightarrow \text{WARN}$) e para proximidade do teto de épocas (`best_epoch > 0.8 * epochs` $\Rightarrow \text{WARN}$), **sem** critério de queda obrigatória nos splits cold.
7. **Relatório Operacional e Auditoria:** Relatório consolidado contendo tempo medido por época, `best_epoch` escolhido por run e contagens auditadas de duplicatas de conteúdo/sequência.

---

## Auto-Verificação do Plano (Self-Review)

1. **Alinhamento com Decisões D1 e D2:** Separa rigorosamente o baseline oficial `graphdta_gin` (verificado contra `thinng/GraphDTA` e Tabelas 1 e 2 de Nguyen et al., 2021) da variante secundária `graphdta_cnn3`, e inclui a Emenda à Fase 1 na Task 0 para o cache de 78 dimensões normalizado em $L_1$ (`v2_onehot78_l1norm`).
2. **Isolamento do Teste e Orquestração Retomável (C1–C4):** `test_loader` é avaliado uma única vez ao final com `checkpoint_best.pt`; hiperparâmetros são fixados em `configs/base.yaml` na Task 0 (`lr=0.0005` constante sem scheduler, `epochs=500`, `patience=50`, `davis_cold_patience=80`); escrita atômica usa temporário no mesmo diretório de destino; orquestrador grava 1 JSON por run, emite `WARN` se `best_epoch > 0.8 * epochs` e separa a etapa de agregação.
3. **Integridade Científica:** Inclui auditoria de colapso de `target_id`s (impacto de mutantes no Davis para a Fase 7), métricas no subconjunto não-censurado, avaliação por faixa de comprimento ($\le 1000$ vs. $> 1000$) e baselines triviais.
