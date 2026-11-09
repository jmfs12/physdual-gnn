# Fase 1: Dados e Cache ESM-2 — Plano de Implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar o pipeline determinístico de ingestão dos datasets Davis e KIBA, divisão de splits sem vazamento de dados por conteúdo (`random`, `cold_drug`, `cold_target`), featurização em grafos PyG para fármacos (serializados como dicionários de tensores) e extração offline em GPU dos embeddings proteicos do ESM-2 (com janela deslizante para sequências longas e armazenamento em `fp16`) com persistência em cache no Google Drive.

**Architecture:** Módulos desacoplados sob `src/physdual/data/`: `datasets.py` para download, validação de procedência e padronização (com unidade de afinidade explícita por fonte, preservação de `smiles_raw` e relatório de falhas/censura), `splits.py` para partição reprodutível em 5 seeds agrupada por conteúdo químico/biológico (SMILES canônico e sequência) com garantia de isolamento disjunto, `featurizer_drug.py` para conversão de SMILES em grafos moleculares PyG persistidos como dicionários de tensores (`weights_only=True`), e `esm_cache.py` para extração em GPU com janela deslizante sobreposta (>1022 resíduos), armazenamento em `fp16` isolado por `model_name` e carregamento seguro em `fp32`. Um script CLI `scripts/run_phase1_data.py`, um verificador de gate `scripts/verify_phase1.py` e um notebook `notebooks/01_data_and_esm_cache.ipynb` orquestram e auditam a execução.

**Tech Stack:** Python 3.10+, PyTorch e NumPy conforme snapshot do ambiente da Fase 0 (`docs/colab_env_snapshot.txt`), PyTorch Geometric (PyG), RDKit, fair-esm, biopython, kagglehub, pandas, pyyaml, pytest.

**Spec:** [`docs/PLANO_AJUSTADO.md`](docs/PLANO_AJUSTADO.md) e [`docs/DECISOES.md`](docs/DECISOES.md).

> **Nota sobre execução de testes em máquina local (Windows + WSL):**
> Todos os comandos `Run:` neste plano são relativos à raiz do repositório (`pytest ...`). Caso esteja executando a partir do Windows invocando o ambiente WSL, prefixe com:
> `wsl bash -lc "cd /mnt/c/Users/joao.francisco/Documents/dev/physdual-gnn && <comando>"`

## Global Constraints

- O Colab já traz `torch` e `numpy` pré-instalados; nenhum script deve reinstalar ou alterar versões de `torch` ou `numpy` (usar as versões registradas em `docs/colab_env_snapshot.txt`).
- Os caminhos de persistência devem respeitar `PHYSDUAL_ROOT` resolvido via `physdual.utils.paths.get_project_root()`.
- O backbone do ESM-2 permanece congelado (`freeze_backbone=True`); a extração de embeddings é executada na GPU alocada pelo Colab em `fp16`, salva em disco em `fp16` e convertida para `fp32` apenas no carregamento (`load_cached_protein_embedding`).
- Todos os arquivos de cache `.pt` salvam exclusivamente tensores e tipos primitivos Python (dicionários/strings/ints/floats), nunca objetos customizados como `torch_geometric.data.Data`, garantindo compatibilidade estrita com `torch.load(..., weights_only=True)`.
- Todo e qualquer commit no Git deve ser realizado com o usuário configurado: `João Miguel Francisco de Souza <joanomiguel@gmail.com>`.

## Review Focus

1. **B1 — Sequências Longas vs. Limite do ESM-2 (1022 resíduos):** O ESM-2 aceita no máximo 1022 resíduos (1024 tokens contando `<cls>` e `<eos>`), mas o Davis possui proteínas maiores. A extração deve aplicar **janela deslizante com sobreposição** (`max_len: 1022`, `overlap: 256` configuráveis em `configs/base.yaml`), descartando `<cls>` e `<eos>` em cada janela, promediando os embeddings nas regiões sobrepostas e garantindo shape final exato `[len(sequence), esm_dim]`.
2. **B2 — Escala de Afinidade ($pK_d$ vs. nM vs. KIBA score) e Valores Censurados:** É proibido usar heurística por linha para detectar $K_d$ vs. $pK_d$ ($K_d = 10\text{ nM}$ cai na faixa numérica de $pK_d$). A função deve receber `affinity_unit: Literal["nM", "pKd", "kiba_score"] | "auto"`. Quando `"auto"`, a detecção ocorre **no nível do dataset inteiro** (ex.: `df["affinity"].max() > 11.0` $\Rightarrow$ `"nM"` para Davis; se ambíguo, falhar com `ValueError` explícito). A conversão $pK_d = -\log_{10}(K_d / 10^9)$ aplica-se **apenas** ao Davis em `"nM"`, **nunca** ao KIBA. Contagens exatas devem ser validadas por `assert` (Davis: $68 \times 442 = 30.056$; KIBA: $2.111 \times 229 = 118.254$), e pares censurados do Davis ($K_d = 10.000\text{ nM} \Rightarrow pK_d = 5.0$) devem ser contabilizados no relatório.
3. **B3 — Cache de Grafos Compatível com `weights_only=True`:** Instâncias de `torch_geometric.data.Data` falham ao carregar com `torch.load(..., weights_only=True)`. O cache deve persistir cada grafo como um dicionário puro de tensores e metadados primitivos (`{"x": Tensor, "edge_index": Tensor, "edge_attr": Tensor, "smiles": str, "smiles_raw": str, "drug_id": str, "num_atoms": int}`) e reconstruir `Data(...)` em `load_drug_graph`.
4. **B4 — Isolamento de Cache entre Modelos ESM-2:** `model_name` é obrigatório em todas as assinaturas de `esm_cache.py`. Os arquivos `.pt` ficam em `cache/esm_embeddings/{model_name}/{hash}.pt`, com payload contendo `{"embedding": Tensor(fp16), "model_name": str, "dim": int, "seq_len": int, "windowed": bool}`, validando no carregamento se `dim == expected_dim` (`esm_dim` do config) e gerando tarball por modelo (`esm_embeddings_{model_name}.tar`).
5. **Vazamento por Conteúdo nos Splits Cold e Baixa Amostragem em Cold-Drug (Davis):** O particionamento e a função `verify_split_leakage` devem agrupar e verificar disjunção pelo **SMILES canônico** (`cold_drug`) e pela **sequência de aminoácidos** (`cold_target`), nunca apenas por `drug_id` ou `target_id` (evitando falsos negativos quando IDs distintos apontam para a mesma estrutura). Documentar que `cold_drug` no Davis (68 fármacos) resulta em ~7 fármacos em val/test por seed, exigindo reporte obrigatório de média ± desvio-padrão nas 5 seeds.
6. **Política de Aminoácidos e SMILES:** Conforme ADR #8 em `docs/DECISOES.md`, os aminoácidos ambíguos/estendidos `X`, `B`, `U`, `Z`, `O` pertencem ao vocabulário nativo do ESM-2 e devem ser **mantidos**, rejeitando-se apenas caracteres fora de `ACDEFGHIKLMNPQRSTVWYBXZUO` após remoção de quebras de linha/espaços. Para SMILES com sais/múltiplos fragmentos, manter o original em `smiles_raw`, gravar o SMILES canônico do maior fragmento em `smiles` e registrar alterações/falhas em relatório sem abortar o lote.

---

### Task 1: Ingestão, Validação de Procedência e Padronização dos Datasets Davis e KIBA

**Files:**
- Create: `src/physdual/data/datasets.py`
- Test: `tests/test_datasets.py`

**Interfaces:**
- Consumes: `physdual.utils.paths.get_project_root`
- Produces:
  - `validate_protein_sequence(seq: str) -> str` (remove espaços/quebras de linha, converte para uppercase, mantém `X, B, U, Z, O`, levanta `ValueError` para caracteres fora do vocabulário ESM-2)
  - `canonicalize_largest_fragment(smiles: str) -> tuple[str | None, bool, str | None]` (retorna `(canonical_smiles, was_modified_or_salt_stripped, error_msg)`)
  - `load_raw_dataset(dataset_name: str, raw_dir: Path | None = None, use_kaggle: bool = True) -> pd.DataFrame`
  - `standardize_dataset(df: pd.DataFrame, dataset_name: str, affinity_unit: Literal["nM", "pKd", "kiba_score", "auto"] = "auto", enforce_expected_counts: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]`
  - `prepare_dataset(dataset_name: str, affinity_unit: Literal["nM", "pKd", "kiba_score", "auto"] = "auto", save_clean: bool = True) -> tuple[pd.DataFrame, Path, dict[str, Any]]`

Colunas padronizadas do DataFrame de saída:
`["drug_id", "smiles", "smiles_raw", "target_id", "sequence", "affinity"]`

Chaves obrigatórias do dicionário de relatório (`report`):
`{"dataset_name", "num_pairs", "num_drugs", "num_targets", "affinity_unit_detected", "davis_censored_count", "davis_censored_pct", "smiles_modified_count", "smiles_failed_count", "smiles_failures"}`

- [ ] **Step 1: Escrever testes pytest que falham em `tests/test_datasets.py`**

```python
import pytest
import pandas as pd
from physdual.data.datasets import (
    standardize_dataset,
    validate_protein_sequence,
)


def test_standardize_davis_nm_dataset():
    # Dataset completo simulado em nM (max > 11.0 permite auto-detecção segura em nível de dataset)
    raw_df = pd.DataFrame({
        "compound_iso_smiles": [
            "CC(=O)OC1=CC=CC=C1C(=O)O",
            "CC(=O)NC1=CC=C(O)C=C1",
            "CC(=O)OC1=CC=CC=C1C(=O)O.[Na+]",  # sal a ser reduzido ao maior fragmento
        ],
        "target_sequence": [
            "MKTVRQERLKSIVRILERSKEPVS",
            "MKTVRQERLKSIVRILERSKEPVS",
            "MKTVRQERLKSIVRILERSKEPVSXBUZO",  # inclui aminoácidos estendidos válidos no ESM-2
        ],
        "affinity": [10.0, 1000.0, 10000.0],  # nM -> pKd: 8.0, 6.0, 5.0 (censurado)
    })
    std, report = standardize_dataset(raw_df, dataset_name="davis", affinity_unit="auto")
    assert list(std.columns) == [
        "drug_id", "smiles", "smiles_raw", "target_id", "sequence", "affinity"
    ]
    assert std.loc[0, "affinity"] == pytest.approx(8.0, abs=1e-2)
    assert std.loc[1, "affinity"] == pytest.approx(6.0, abs=1e-2)
    assert std.loc[2, "affinity"] == pytest.approx(5.0, abs=1e-2)
    assert report["affinity_unit_detected"] == "nM"
    assert report["davis_censored_count"] == 1
    assert report["smiles_modified_count"] >= 1
    assert std.loc[2, "smiles_raw"] == "CC(=O)OC1=CC=CC=C1C(=O)O.[Na+]"


def test_standardize_davis_pkd_dataset():
    # Dataset explicitamente em pKd (sem conversão adicional)
    raw_df = pd.DataFrame({
        "smiles": ["CC(=O)OC1=CC=CC=C1C(=O)O", "CC(=O)NC1=CC=C(O)C=C1"],
        "sequence": ["MKTVRQERLKSIVRILERSKEPVS", "ACDEFGHIKLMNPQRSTVWY"],
        "affinity": [7.5, 5.0],
    })
    std, report = standardize_dataset(raw_df, dataset_name="davis", affinity_unit="pKd")
    assert std.loc[0, "affinity"] == pytest.approx(7.5, abs=1e-4)
    assert std.loc[1, "affinity"] == pytest.approx(5.0, abs=1e-4)
    assert report["davis_censored_count"] == 1


def test_standardize_davis_auto_ambiguous_raises():
    # Se affinity_unit="auto" e todos os valores estão em faixa ambígua sem indicação clara (ex.: max <= 11 mas min < 4.5 ou amostra única indecidível)
    raw_df = pd.DataFrame({
        "smiles": ["CC(=O)OC1=CC=CC=C1C(=O)O"],
        "sequence": ["MKTVRQERLKSIVRILERSKEPVS"],
        "affinity": [7.5],
    })
    with pytest.raises(ValueError, match="ambígua|affinity_unit"):
        standardize_dataset(raw_df, dataset_name="davis", affinity_unit="auto")


def test_kiba_never_converted_to_pkd():
    # KIBA usa kiba_score; valores > 11 jamais devem sofrer transformação logarítmica de pKd
    raw_df = pd.DataFrame({
        "smiles": ["CC(=O)OC1=CC=CC=C1C(=O)O", "CC(=O)NC1=CC=C(O)C=C1"],
        "sequence": ["MKTVRQERLKSIVRILERSKEPVS", "ACDEFGHIKLMNPQRSTVWY"],
        "affinity": [11.4, 14.2],
    })
    std, report = standardize_dataset(raw_df, dataset_name="kiba", affinity_unit="auto")
    assert std.loc[0, "affinity"] == pytest.approx(11.4, abs=1e-4)
    assert std.loc[1, "affinity"] == pytest.approx(14.2, abs=1e-4)
    assert report["affinity_unit_detected"] == "kiba_score"


def test_enforce_expected_counts_raises_on_mismatch():
    raw_df = pd.DataFrame({
        "smiles": ["CC(=O)OC1=CC=CC=C1C(=O)O"],
        "sequence": ["MKTVRQERLKSIVRILERSKEPVS"],
        "affinity": [7.5],
    })
    with pytest.raises(AssertionError, match="30056|contagem"):
        standardize_dataset(
            raw_df, dataset_name="davis", affinity_unit="pKd", enforce_expected_counts=True
        )


def test_validate_protein_sequence_policy():
    # Mantém X, B, U, Z, O e remove quebras de linha
    clean = validate_protein_sequence("mktvrq\nXBUZO ")
    assert clean == "MKTVRQXBUZO"
    # Rejeita caracteres inválidos fora do vocabulário ESM-2 (ex.: dígitos, '*', 'J')
    with pytest.raises(ValueError, match="caractere inválido"):
        validate_protein_sequence("MKTVRQ*J123")
```

- [ ] **Step 2: Executar teste e verificar que falha**

Run: `pytest tests/test_datasets.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'physdual.data.datasets'`

- [ ] **Step 3: Implementar ingestão, validação de procedência e padronização em `src/physdual/data/datasets.py`**

Implementar:
1. Download via `kagglehub.dataset_download("rajaryan2315/davis-and-kiba-datasets")` (lendo credenciais `KAGGLE_USERNAME` / `KAGGLE_KEY` do ambiente ou dos Secrets do Colab) e suporte a URL de fallback configurável marcada como `# TODO: URL a verificar (repositório oficial DeepDTA/GraphDTA)`.
2. Validação estrita de procedência (`enforce_expected_counts=True` por padrão em `prepare_dataset`):
   - **Davis:** exatamente 68 fármacos únicos, 442 alvos únicos e 30.056 pares ($68 \times 442$).
   - **KIBA:** exatamente 2.111 fármacos únicos, 229 alvos únicos e 118.254 pares.
   - Falhar explicitamente com `AssertionError` detalhando os números encontrados vs. esperados caso diverja.
3. Tratamento de escala de afinidade (`affinity_unit`):
   - Para `dataset_name == "kiba"`, exigir `"kiba_score"` (ou `"auto"` resolvendo para `"kiba_score"`) e **jamais** converter para $pK_d$.
   - Para `dataset_name == "davis"`, se `affinity_unit == "auto"`, inspecionar a distribuição do dataset completo (`len(df) >= 100`, `max > 11.0` e `min > 0` $\Rightarrow$ `"nM"`; se `len(df) >= 100` e `4.9 <= min <= 5.1` e `max <= 11.0` $\Rightarrow$ `"pKd"`; caso contrário, levantar `ValueError` exigindo `affinity_unit` explícito).
   - Contabilizar pares censurados no Davis ($K_d \ge 10.000\text{ nM}$ ou $pK_d \le 5.0001$).
4. Sanitização de SMILES via RDKit preservando `smiles_raw`, extraindo o maior fragmento covalente para `smiles` e registrando falhas em `report["smiles_failures"]` sem derrubar o processamento em lote.
5. Sanitização de sequência proteica conforme ADR #8 (`docs/DECISOES.md`), mantendo `X, B, U, Z, O`.
6. Salvamento em `data/processed/{dataset}_clean.csv` e relatório de auditoria em `data/processed/{dataset}_ingestion_report.json`.

- [ ] **Step 4: Executar teste e verificar que passa**

Run: `pytest tests/test_datasets.py -v`
Expected: PASS

- [ ] **Step 5: Commit da Task 1**

```bash
git add docs/DECISOES.md src/physdual/data/datasets.py tests/test_datasets.py
git commit -m "feat(data): implement dataset ingestion, provenance checks, and explicit affinity unit handling"
```

---

### Task 2: Divisão de Splits Determinísticos sem Vazamento por Conteúdo

**Files:**
- Create: `src/physdual/data/splits.py`
- Test: `tests/test_splits.py`

**Interfaces:**
- Consumes: DataFrames padronizados da Task 1 (`smiles` canônico e `sequence` validada)
- Produces:
  - `create_splits(df: pd.DataFrame, split_type: str, seed: int, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> dict[str, list[int]]`
  - `save_splits(splits: dict[str, list[int]], output_path: Path) -> Path`
  - `load_splits(split_path: Path) -> dict[str, list[int]]`
  - `verify_split_leakage(df: pd.DataFrame, splits: dict[str, list[int]], split_type: str) -> bool`

Regras críticas de domínio:
- Em `cold_drug`, o agrupamento e a verificação de vazamento usam a coluna `smiles` (SMILES canônico), **não** apenas `drug_id`. Dois `drug_id` distintos com o mesmo `smiles` devem cair obrigatoriamente na mesma partição (`train`, `val` ou `test`).
- Em `cold_target`, o agrupamento e a verificação de vazamento usam a coluna `sequence`, **não** apenas `target_id`. Dois `target_id` distintos com a mesma `sequence` devem cair obrigatoriamente na mesma partição.
- `verify_split_leakage` verifica também que `train`, `val` e `test` são não vazios, mutuamente disjuntos em índices e cobrem 100% do DataFrame.

- [ ] **Step 1: Escrever bateria de testes pytest em `tests/test_splits.py`**

```python
import pytest
import pandas as pd
from physdual.data.splits import (
    create_splits,
    save_splits,
    load_splits,
    verify_split_leakage,
)


@pytest.fixture
def synthetic_dta_df():
    # 20 drogas únicas, 10 proteínas únicas, 200 interações
    records = []
    for d in range(20):
        for t in range(10):
            records.append({
                "drug_id": f"D{d}",
                "smiles": f"C{'C' * d}O",
                "target_id": f"T{t}",
                "sequence": f"MKTVRQ{'A' * (t + 1)}",
                "affinity": float(d + t),
            })
    return pd.DataFrame(records)


def test_cold_drug_split_has_zero_content_leakage(synthetic_dta_df):
    splits = create_splits(synthetic_dta_df, split_type="cold_drug", seed=42)
    assert verify_split_leakage(synthetic_dta_df, splits, split_type="cold_drug")
    train_smiles = set(synthetic_dta_df.iloc[splits["train"]]["smiles"])
    val_smiles = set(synthetic_dta_df.iloc[splits["val"]]["smiles"])
    test_smiles = set(synthetic_dta_df.iloc[splits["test"]]["smiles"])
    assert train_smiles.isdisjoint(val_smiles)
    assert train_smiles.isdisjoint(test_smiles)
    assert val_smiles.isdisjoint(test_smiles)


def test_cold_target_split_has_zero_content_leakage(synthetic_dta_df):
    splits = create_splits(synthetic_dta_df, split_type="cold_target", seed=42)
    assert verify_split_leakage(synthetic_dta_df, splits, split_type="cold_target")
    train_seqs = set(synthetic_dta_df.iloc[splits["train"]]["sequence"])
    val_seqs = set(synthetic_dta_df.iloc[splits["val"]]["sequence"])
    test_seqs = set(synthetic_dta_df.iloc[splits["test"]]["sequence"])
    assert train_seqs.isdisjoint(val_seqs)
    assert train_seqs.isdisjoint(test_seqs)
    assert val_seqs.isdisjoint(test_seqs)


def test_duplicate_content_with_different_ids_detected_and_grouped():
    # D0 e D_DUP têm IDs diferentes, mas o mesmo SMILES canônico ("CCO")
    records = []
    for d in range(10):
        smiles = "CCO" if d in (0, 1) else f"C{'C' * d}N"
        for t in range(5):
            records.append({
                "drug_id": f"D{d}",
                "smiles": smiles,
                "target_id": f"T{t}",
                "sequence": "MKTVRQ" if t in (0, 1) else f"ACDEFG{'H' * t}",
                "affinity": 6.0,
            })
    df = pd.DataFrame(records)

    # create_splits deve agrupar por SMILES/sequence, mantendo duplicatas químicas juntas
    drug_splits = create_splits(df, split_type="cold_drug", seed=0)
    assert verify_split_leakage(df, drug_splits, split_type="cold_drug")

    target_splits = create_splits(df, split_type="cold_target", seed=0)
    assert verify_split_leakage(df, target_splits, split_type="cold_target")

    # Se forçarmos manualmente D0 no treino e D1 (mesmo SMILES) no teste, verify_split_leakage deve retornar False
    corrupted_splits = {
        "train": [0],  # D0, smiles="CCO"
        "val": [10],
        "test": [5],   # D1, smiles="CCO"
    }
    assert not verify_split_leakage(df.iloc[[0, 5, 10]].reset_index(drop=True), {
        "train": [0], "val": [2], "test": [1]
    }, split_type="cold_drug")


def test_splits_determinism_proportions_and_non_empty(synthetic_dta_df, tmp_path):
    s1 = create_splits(synthetic_dta_df, split_type="random", seed=1)
    s2 = create_splits(synthetic_dta_df, split_type="random", seed=1)
    s3 = create_splits(synthetic_dta_df, split_type="random", seed=2)

    assert s1 == s2
    assert s1 != s3

    for part in ("train", "val", "test"):
        assert len(s1[part]) > 0

    n = len(synthetic_dta_df)
    assert len(s1["train"]) == pytest.approx(0.8 * n, abs=2)
    assert len(s1["val"]) == pytest.approx(0.1 * n, abs=2)
    assert len(s1["test"]) == pytest.approx(0.1 * n, abs=2)

    # Round-trip save_splits / load_splits
    out_file = save_splits(s1, tmp_path / "split_seed1.json")
    loaded = load_splits(out_file)
    assert loaded == s1
```

- [ ] **Step 2: Executar teste e verificar que falha**

Run: `pytest tests/test_splits.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'physdual.data.splits'`

- [ ] **Step 3: Implementar particionamento por conteúdo em `src/physdual/data/splits.py`**

Implementar geração determinística com `np.random.RandomState(seed)`, agrupando entidades únicas por `smiles` em `cold_drug` e por `sequence` em `cold_target` (ordenando chaves únicas antes do shuffle para reprodutibilidade total entre plataformas), garantindo que `train`, `val` e `test` nunca fiquem vazios, e persistindo JSON `{ "train": [...], "val": [...], "test": [...] }` sob `data/processed/splits/{dataset}_{split_type}_seed{seed}.json`.

- [ ] **Step 4: Executar teste e verificar que passa**

Run: `pytest tests/test_splits.py -v`
Expected: PASS

- [ ] **Step 5: Commit da Task 2**

```bash
git add src/physdual/data/splits.py tests/test_splits.py
git commit -m "feat(data): implement content-aware leakage-free random, cold-drug, and cold-target splits"
```

---

### Task 3: Featurização Molecular e Cache de Grafos PyG Compatível com `weights_only=True`

**Files:**
- Create: `src/physdual/data/featurizer_drug.py`
- Test: `tests/test_featurizer_drug.py`

**Interfaces:**
- Consumes: SMILES dos datasets padronizados
- Produces:
  - `smiles_to_graph_dict(smiles: str, drug_id: str | None = None, smiles_raw: str | None = None) -> dict[str, Any]` (dicionário de tensores `x`, `edge_index`, `edge_attr` e metadados primitivos `smiles`, `smiles_raw`, `drug_id`, `num_atoms`)
  - `graph_dict_to_pyg_data(graph_dict: dict[str, Any]) -> torch_geometric.data.Data`
  - `smiles_to_pyg_graph(smiles: str, drug_id: str | None = None) -> torch_geometric.data.Data`
  - `build_drug_graph_cache(smiles_list: list[str], output_path: Path, drug_ids: list[str] | None = None) -> tuple[Path, dict[str, Any]]` (salva dicionário `{canonical_smiles: graph_dict}` em `.pt` e retorna `(output_path, failure_report)`)
  - `load_drug_graph_cache(cache_path: Path) -> dict[str, dict[str, Any]]` (carrega com `torch.load(cache_path, weights_only=True)`)
  - `load_drug_graph(smiles: str, cache_dict: dict[str, dict[str, Any]]) -> torch_geometric.data.Data` (reconstrói e retorna `Data` a partir do dicionário de tensores)

Features atômicas (9 dimensões por nó: atomic number, formal charge, degree, explicit valence, total num H, hybridization, aromaticity, is in ring, chirality) e arestas bidirecionais com atributos de ligação (bond type, stereochemistry, is conjugated). Moléculas com 0 átomos ou falha no RDKit são registradas no `failure_report` e rejeitadas com erro claro em chamadas unitárias.

- [ ] **Step 1: Escrever testes pytest de featurização e round-trip `weights_only=True` em `tests/test_featurizer_drug.py`**

```python
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rdkit")
pytest.importorskip("torch_geometric")

from physdual.data.featurizer_drug import (
    smiles_to_pyg_graph,
    build_drug_graph_cache,
    load_drug_graph_cache,
    load_drug_graph,
)


def test_aspirin_graph_features():
    aspirin = "CC(=O)OC1=CC=CC=C1C(=O)O"
    g = smiles_to_pyg_graph(aspirin, drug_id="aspirin")
    assert g.x.size(0) == 13
    assert g.x.size(1) >= 9
    assert g.edge_index.size(0) == 2
    assert g.edge_index.size(1) == 26
    assert g.edge_attr is not None
    assert not torch.isnan(g.x).any()


def test_drug_graph_cache_roundtrip_weights_only(tmp_path):
    smiles_list = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",       # aspirina
        "CC(=O)NC1=CC=C(O)C=C1.[Cl-]",   # paracetamol com contra-íon (reduzido ao maior fragmento)
        "INVALID_SMILES_STRING",          # inválido: deve ir para o relatório de falhas sem quebrar o lote
    ]
    cache_file = tmp_path / "drug_graphs.pt"
    saved_path, report = build_drug_graph_cache(smiles_list, output_path=cache_file)

    assert saved_path.exists()
    assert report["num_succeeded"] == 2
    assert report["num_failed"] == 1
    assert report["num_modified_fragments"] == 1

    # Verifica carregamento estrito com weights_only=True diretamente via torch.load
    raw_cache = torch.load(saved_path, weights_only=True)
    assert isinstance(raw_cache, dict)

    loaded_cache = load_drug_graph_cache(saved_path)
    g_orig = smiles_to_pyg_graph("CC(=O)OC1=CC=CC=C1C(=O)O")
    g_loaded = load_drug_graph("CC(=O)OC1=CC=CC=C1C(=O)O", loaded_cache)

    assert torch.equal(g_orig.x, g_loaded.x)
    assert torch.equal(g_orig.edge_index, g_loaded.edge_index)
    assert torch.equal(g_orig.edge_attr, g_loaded.edge_attr)
```

- [ ] **Step 2: Executar teste e verificar que falha (ou skip caso sem RDKit/PyG local)**

Run: `pytest tests/test_featurizer_drug.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'physdual.data.featurizer_drug'` (ou SKIP se dependências de ML ausentes).

- [ ] **Step 3: Implementar featurizador molecular em `src/physdual/data/featurizer_drug.py`**

Implementar sanitização RDKit com seleção do maior fragmento covalente (`rdMolStandardize.LargestFragmentChooser` ou `Chem.GetMolFrags`), construção de tensores `x` (`float32`), `edge_index` (`int64`) e `edge_attr` (`float32`), serialização em dicionários primitivos compatíveis com `weights_only=True`, reconstrução em `torch_geometric.data.Data` sob demanda e relatório de falhas de parsing.

- [ ] **Step 4: Executar teste e verificar que passa**

Run: `pytest tests/test_featurizer_drug.py -v`
Expected: PASS (ou SKIP justificado se sem stack ML local).

- [ ] **Step 5: Commit da Task 3**

```bash
git add src/physdual/data/featurizer_drug.py tests/test_featurizer_drug.py
git commit -m "feat(data): implement weights_only-compatible PyG molecular graph featurizer and cache"
```

---

### Task 4: Motor de Extração Offline de Embeddings do ESM-2 (Janela Deslizante, Isolamento por Modelo e `fp16`)

**Files:**
- Modify: `configs/base.yaml` (adicionar `max_len: 1022` e `overlap: 256` na seção `esm`)
- Create: `src/physdual/data/esm_cache.py`
- Test: `tests/test_esm_cache.py`

**Interfaces:**
- Consumes: Sequências proteicas únicas de Davis (442) e KIBA (229); `configs/base.yaml`
- Produces:
  - `compute_seq_hash(sequence: str) -> str` (SHA-256 de 16 caracteres da sequência sanitizada)
  - `save_protein_embedding(sequence: str, embedding: torch.Tensor, model_name: str, cache_dir: Path, windowed: bool = False) -> Path` (salva em `cache_dir / model_name / f"{hash}.pt"` com `embedding` convertido para `torch.float16` e metadados primitivos `model_name`, `dim`, `seq_len`, `windowed`)
  - `load_cached_protein_embedding(sequence: str, model_name: str, cache_dir: Path, expected_dim: int | None = None) -> torch.Tensor` (carrega com `weights_only=True`, valida `seq_len == len(sequence)`, `model_name` e `dim == expected_dim`, e retorna tensor convertido para `torch.float32`)
  - `extract_esm_embeddings_batch(sequences: list[tuple[str, str]], model_name: str, cache_dir: Path, batch_tokens: int = 2048, max_len: int = 1022, overlap: int = 256, device: str = "cuda") -> tuple[list[Path], dict[str, Any]]` (aplica janela deslizante com sobreposição e média nas regiões sobrepostas para `len(seq) > max_len`, retorna caminhos gerados e relatório contendo `num_windowed_sequences`, `total_sequences`, `model_name`, `dim`)
  - `create_cache_tarball(cache_dir: Path, model_name: str, output_tar: Path | None = None) -> Path` (gera `esm_embeddings_{model_name}.tar`)

Política de Janela Deslizante (B1):
- Se `L <= max_len` (padrão `1022`): extrai em passagem única, removendo `<cls>` (índice 0) e `<eos>` (índice `L + 1`), resultando em `[L, dim]`.
- Se `L > max_len`: avança com passo `stride = max_len - overlap` (ex.: `1022 - 256 = 766`), garantindo que a última janela cubra até o final da sequência (`[start, end)`). Em cada janela, descarta `<cls>` e `<eos>` daquela sub-sequência, acumula as representações em um buffer `sum_emb` de shape `[L, dim]` (`float32`) e um contador `counts` de shape `[L, 1]`, dividindo `sum_emb / counts` ao final antes de salvar em `fp16`.

- [ ] **Step 1: Atualizar `configs/base.yaml` com `max_len` e `overlap` e escrever testes pytest em `tests/test_esm_cache.py`**

Adicionar em `configs/base.yaml` sob `esm:`:
```yaml
  max_len: 1022
  overlap: 256
```

Testes em `tests/test_esm_cache.py`:
```python
import pytest
from physdual.data.esm_cache import compute_seq_hash

torch = pytest.importorskip("torch")

from physdual.data.esm_cache import (
    save_protein_embedding,
    load_cached_protein_embedding,
    extract_esm_embeddings_batch,
    create_cache_tarball,
)


def test_seq_hash_deterministic():
    seq = "MKTVRQERLKSIVRILERSKEPVS"
    h1 = compute_seq_hash(seq)
    h2 = compute_seq_hash(seq)
    assert h1 == h2
    assert len(h1) == 16


def test_save_fp16_and_load_fp32_with_model_isolation(tmp_path):
    seq = "MKTVRQERLKSIV"
    emb_650m = torch.randn(len(seq), 1280, dtype=torch.float32)
    emb_150m = torch.randn(len(seq), 640, dtype=torch.float32)

    p1 = save_protein_embedding(seq, emb_650m, model_name="esm2_t33_650M_UR50D", cache_dir=tmp_path)
    p2 = save_protein_embedding(seq, emb_150m, model_name="esm2_t30_150M_UR50D", cache_dir=tmp_path)

    # Não podem colidir no disco
    assert p1 != p2
    assert p1.parent.name == "esm2_t33_650M_UR50D"
    assert p2.parent.name == "esm2_t30_150M_UR50D"

    # No disco o tensor deve estar em fp16 e carregável com weights_only=True
    raw_payload = torch.load(p1, weights_only=True)
    assert raw_payload["embedding"].dtype == torch.float16
    assert raw_payload["dim"] == 1280
    assert raw_payload["seq_len"] == len(seq)

    # Ao carregar pela API, retorna fp32 e valida expected_dim
    loaded_650m = load_cached_protein_embedding(
        seq, model_name="esm2_t33_650M_UR50D", cache_dir=tmp_path, expected_dim=1280
    )
    assert loaded_650m.dtype == torch.float32
    assert loaded_650m.shape == (len(seq), 1280)
    assert torch.allclose(emb_650m, loaded_650m, atol=1e-3)

    # Deve falhar com ValueError claro se expected_dim divergir do arquivo salvo
    with pytest.raises(ValueError, match="dim|1280"):
        load_cached_protein_embedding(
            seq, model_name="esm2_t33_650M_UR50D", cache_dir=tmp_path, expected_dim=640
        )


def test_tarball_per_model(tmp_path):
    seq = "MKTVRQ"
    emb = torch.randn(len(seq), 320, dtype=torch.float32)
    save_protein_embedding(seq, emb, model_name="esm2_t6_8M_UR50D", cache_dir=tmp_path)
    tar_path = create_cache_tarball(tmp_path, model_name="esm2_t6_8M_UR50D")
    assert tar_path.exists()
    assert tar_path.name == "esm_embeddings_esm2_t6_8M_UR50D.tar"


def test_long_sequence_sliding_window_1500aa(tmp_path):
    pytest.importorskip("esm")
    long_seq = ("MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG" * 24)[:1500]
    assert len(long_seq) == 1500

    try:
        paths, report = extract_esm_embeddings_batch(
            sequences=[("T_LONG", long_seq)],
            model_name="esm2_t6_8M_UR50D",
            cache_dir=tmp_path,
            max_len=1022,
            overlap=256,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    except Exception as exc:
        if "URLError" in str(type(exc)) or "Connection" in str(exc) or "HTTPError" in str(exc):
            pytest.skip(f"Sem conexão ou pesos locais para baixar esm2_t6_8M_UR50D: {exc}")
        raise

    assert len(paths) == 1
    assert report["num_windowed_sequences"] == 1
    emb = load_cached_protein_embedding(
        long_seq, model_name="esm2_t6_8M_UR50D", cache_dir=tmp_path, expected_dim=320
    )
    assert emb.shape == (1500, 320)
    assert emb.dtype == torch.float32
    assert not torch.isnan(emb).any()
    assert not torch.isinf(emb).any()
```

- [ ] **Step 2: Executar teste e verificar que falha**

Run: `pytest tests/test_esm_cache.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'physdual.data.esm_cache'`

- [ ] **Step 3: Implementar motor de extração com janela deslizante e cache isolado em `src/physdual/data/esm_cache.py`**

Implementar:
1. Carregamento do modelo ESM-2 utilizando o gerenciador de contexto escopado `_scoped_esm_weights_load` (ADR #5 em `docs/DECISOES.md`), congelando pesos (`model.eval()`, `requires_grad_(False)`).
2. Extração com janela deslizante (`max_len=1022`, `overlap=256`) para sequências com `len(seq) > max_len`, descartando `<cls>` e `<eos>` em cada janela e calculando a média exata por posição nos trechos sobrepostos.
3. Persistência em `fp16` no caminho `cache/esm_embeddings/{model_name}/{hash}.pt` com metadados primitivos (`model_name`, `dim`, `seq_len`, `windowed`) e conversão para `fp32` em `load_cached_protein_embedding` com validação de `expected_dim`.
4. Empacotamento em tarball por modelo (`esm_embeddings_{model_name}.tar`).

- [ ] **Step 4: Executar teste e verificar que passa**

Run: `pytest tests/test_esm_cache.py -v`
Expected: PASS

- [ ] **Step 5: Commit da Task 4**

```bash
git add configs/base.yaml src/physdual/data/esm_cache.py tests/test_esm_cache.py
git commit -m "feat(data): implement model-namespaced fp16 ESM-2 cache with sliding window for >1022aa"
```

---

### Task 5: Script Orquestrador CLI, Verificador de Gate (`verify_phase1.py`) e Notebook Colab da Fase 1

**Files:**
- Modify: `src/physdual/data/__init__.py`
- Create: `scripts/run_phase1_data.py`
- Create: `scripts/verify_phase1.py`
- Create: `notebooks/01_data_and_esm_cache.ipynb`
- Test: `tests/test_verify_phase1.py`

**Interfaces:**
- Consumes: Módulos `datasets`, `splits`, `featurizer_drug`, `esm_cache` e `load_config`
- Produces:
  - CLI de execução: `python scripts/run_phase1_data.py --dataset all --esm-model esm2_t33_650M_UR50D`
  - CLI de verificação do Gate de Pronto: `python scripts/verify_phase1.py --config configs/base.yaml`
  - Relatório consolidado: `outputs/logs/phase1_summary_report.json`
  - Notebook interativo com barras de progresso `tqdm`, instruções de Secrets para o Kaggle e validação na GPU alocada pelo Colab.

Critérios objetivos verificados por `scripts/verify_phase1.py` (Gate de Pronto da Fase 1):
1. Todas as 442 (Davis) + 229 (KIBA) sequências únicas têm arquivo `.pt` de embedding no cache do `model_name` configurado.
2. Para cada embedding de proteína, `shape[0] == len(sequence)` e `shape[1] == esm_dim` do config.
3. Nenhum `NaN` ou `Inf` nos embeddings proteicos ou tensores de fármacos; nenhum grafo de fármaco vazio (`num_atoms >= 1`).
4. Todos os 30 arquivos de splits (3 tipos $\times$ 5 seeds $\times$ 2 datasets) existem, possuem `train`, `val` e `test` não vazios e passam em `verify_split_leakage` por conteúdo (`smiles` e `sequence`).
5. Contagens de pares e entidades únicas batem exatamente com os valores esperados: Davis ($68 \times 442 = 30.056$) e KIBA ($2.111 \times 229 = 118.254$).
6. Todos os arquivos `.pt` (grafos de fármacos e embeddings ESM-2) carregam sem erro com `torch.load(..., weights_only=True)`.
7. O relatório final consolidado (`outputs/logs/phase1_summary_report.json`) lista explicitamente: quantidade e IDs de sequências processadas com janela deslizante, contagem e percentual de valores censurados do Davis ($K_d = 10.000\text{ nM} / pK_d = 5.0$), e contagem/lista de SMILES alterados (sais removidos) ou falhos.

- [ ] **Step 1: Escrever teste unitário para o verificador de gate em `tests/test_verify_phase1.py`**

Testar a função `run_phase1_verification(...)` sobre um diretório temporário contendo um mini-dataset mockado completo (ou validando que falha quando falta um split, quando `shape[0] != len(seq)` ou quando há `NaN`).

- [ ] **Step 2: Executar teste e verificar que falha**

Run: `pytest tests/test_verify_phase1.py -v`
Expected: FAIL com `ModuleNotFoundError`

- [ ] **Step 3: Implementar `src/physdual/data/__init__.py`, `scripts/run_phase1_data.py`, `scripts/verify_phase1.py` e `notebooks/01_data_and_esm_cache.ipynb`**

- Em `scripts/run_phase1_data.py`: aceitar `--dataset` (`davis`, `kiba`, `all`), `--affinity-unit` (`auto`, `nM`, `pKd`, `kiba_score`), `--esm-model`, `--skip-esm`, `--batch-tokens`, salvando `outputs/logs/phase1_summary_report.json`.
- Em `scripts/verify_phase1.py`: verificar os 7 itens do Gate de Pronto e retornar código de saída `0` apenas se todos passarem (`1` caso contrário).
- Em `notebooks/01_data_and_esm_cache.ipynb`:
  - Célula 1: Montagem do Drive, configuração de `KAGGLE_USERNAME`/`KAGGLE_KEY` via `google.colab.userdata` (painel Secrets do Colab) e setup de ambiente.
  - Célula 2: Download, validação de procedência contra contagens do DeepDTA/GraphDTA e inspeção de Davis (incluindo proporção censurada em $pK_d = 5.0$) e KIBA.
  - Célula 3: Geração e validação de não-vazamento por conteúdo dos splits (`random`, `cold_drug`, `cold_target`) nas 5 seeds.
  - Célula 4: Featurização de grafos moleculares como dicionários de tensores e salvamento em cache.
  - Célula 5: Extração em lote do ESM-2 em `fp16` na GPU alocada pelo Colab (com janela deslizante para proteínas >1022 resíduos).
  - Célula 6: Criação do tarball `esm_embeddings_{model_name}.tar` no Drive e execução de `scripts/verify_phase1.py`.

- [ ] **Step 4: Validar testes e sintaxe CLI**

Run: `pytest tests/test_verify_phase1.py -v && python scripts/run_phase1_data.py --help && python scripts/verify_phase1.py --help`
Expected: PASS nos testes e retorno 0 nos comandos `--help`.

- [ ] **Step 5: Commit da Task 5**

```bash
git add src/physdual/data/__init__.py scripts/run_phase1_data.py scripts/verify_phase1.py notebooks/01_data_and_esm_cache.ipynb tests/test_verify_phase1.py
git commit -m "feat(data): add Phase 1 pipeline CLI, verification gate script, and Colab notebook"
```

---

### Task 6: Documentação Técnica da Fase 1 e Atualização do README

**Files:**
- Create: `docs/FASE1_DADOS.md`
- Modify: `README.md`
- Modify: `docs/PROXIMOS_PASSOS.md`

- [ ] **Step 1: Escrever `docs/FASE1_DADOS.md` cobrindo protocolos, estatísticas e limitações conhecidas**
  - Contagens canônicas e verificação de procedência (DeepDTA/GraphDTA): Davis ($68 \times 442 = 30.056$) e KIBA ($2.111 \times 229 = 118.254$).
  - Configuração de credenciais do Kaggle via painel **Secrets** do Google Colab (`KAGGLE_USERNAME` e `KAGGLE_KEY`) e nota sobre URL de fallback a verificar.
  - Censura à direita no Davis: explicar que grande parte dos pares possui $K_d = 10.000\text{ nM}$ ($pK_d = 5.0$, limite superior do ensaio experimental), documentando o impacto disso nas métricas de regressão (MSE, CI, Pearson) e como o relatório contabiliza esses pares.
  - Alta variância no regime `cold_drug` do Davis: como o Davis possui apenas 68 fármacos únicos, uma divisão 80/10/10 aloca apenas ~7 fármacos para validação e ~7 para teste em cada seed. Documentar que a variância entre seeds será naturalmente alta e que todas as avaliações a partir da Fase 2 devem reportar **média ± desvio-padrão nas 5 seeds**.
  - Política de janela deslizante para proteínas >1022 resíduos (`max_len=1022`, `overlap=256`, média nas sobreposições) e política de manutenção de aminoácidos ambíguos (`X, B, U, Z, O`) no vocabulário do ESM-2.
  - Guia de I/O otimizado no Colab: descompactação de `esm_embeddings_{model_name}.tar` no SSD efêmero `/content/local_cache/` e execução do gate `python scripts/verify_phase1.py`.
- [ ] **Step 2: Atualizar `README.md` e `docs/PROXIMOS_PASSOS.md` refletindo o Gate de Pronto da Fase 1 e transição para a Fase 2**
- [ ] **Step 3: Commit da Task 6**

```bash
git add docs/FASE1_DADOS.md README.md docs/PROXIMOS_PASSOS.md
git commit -m "docs(phase1): document data provenance, Davis censoring, cold-drug variance, and Phase 1 gate"
```

---

## Gate de Pronto da Fase 1 (Critérios de Aceite Obrigatórios)

A Fase 1 **só é considerada concluída** quando `python scripts/verify_phase1.py` retornar `exit code 0` comprovando:

1. **Cobertura completa de embeddings:** Todas as 442 (Davis) + 229 (KIBA) sequências únicas têm embedding salvo em `cache/esm_embeddings/{model_name}/{hash}.pt`.
2. **Integridade dimensional:** Para cada embedding, `shape[0] == len(sequence)` e `shape[1] == esm_dim` definido em `configs/base.yaml`.
3. **Estabilidade numérica e topológica:** Nenhum `NaN` ou `Inf` nos embeddings ou features de grafos; nenhum grafo molecular vazio ($\ge 1$ átomo).
4. **Isolamento de splits por conteúdo:** Os 30 splits (3 tipos $\times$ 5 seeds $\times$ 2 datasets) existem, possuem partições `train`/`val`/`test` não vazias e passam em `verify_split_leakage` comparando `smiles` canônico e `sequence`.
5. **Procedência e volumetria exatas:** Davis contém exatamente 68 fármacos, 442 alvos e 30.056 pares; KIBA contém exatamente 2.111 fármacos, 229 alvos e 118.254 pares.
6. **Segurança de desserialização:** Todos os arquivos `.pt` de grafos e embeddings carregam com `torch.load(..., weights_only=True)`.
7. **Rastreabilidade no relatório final:** `outputs/logs/phase1_summary_report.json` registra: (a) sequências processadas com janela deslizante, (b) contagem e percentual de valores censurados do Davis ($pK_d = 5.0$), e (c) SMILES reduzidos ao maior fragmento ou que falharam na sanitização.

---

## Auto-Verificação do Plano (Self-Review)

1. **Bloqueadores Resolvidos (B1–B4):**
   - **B1:** Janela deslizante (`max_len=1022`, `overlap=256`) com média nas regiões sobrepostas e teste dedicado de 1500 aa na Task 4.
   - **B2:** Parâmetro explícito `affinity_unit` com auto-detecção apenas no nível do dataset inteiro, proibição de conversão $pK_d$ no KIBA, asserts de contagem ($30.056$ e $118.254$) e contagem de valores censurados ($pK_d = 5.0$) na Task 1.
   - **B3:** Grafos salvos como dicionários de tensores primitivos compatíveis com `weights_only=True` e reconstruídos em `load_drug_graph` na Task 3.
   - **B4:** `model_name` incluído em todas as funções, subdiretórios e tarballs da Task 4, com armazenamento em `fp16` e leitura em `fp32` validando `esm_dim`.
2. **Padronização de Testes:** Todos os exemplos e comandos `Run:` utilizam `pytest` com caminhos relativos à raiz do repositório, `pytest.importorskip` para pacotes da stack ML e tratamento de indisponibilidade de rede.
3. **Rigor Científico:** Vazamento verificado por estrutura química (`smiles`) e sequência biológica (`sequence`) na Task 2; variância do `cold_drug` no Davis (~7 fármacos em val/test) e censura em $pK_d = 5.0$ documentadas na Task 6.
