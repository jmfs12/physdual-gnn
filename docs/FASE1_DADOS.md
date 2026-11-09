# Fase 1 — Pipeline de Dados, Splits e Cache Offline do ESM-2

Este documento descreve a arquitetura, os protocolos de curadoria e particionamento, as limitações estatísticas dos benchmarks e os critérios do Gate de Pronto da **Fase 1** do projeto **PhysDual-GNN**.

---

## 1. Fontes de Dados, Credenciais e Validação de Procedência

### 1.1 Fonte Canônica Oficial (`hkmztrk/DeepDTA`) e Kaggle Opcional
A ingestão padrão baixa os dados brutos diretamente do repositório canônico oficial [DeepDTA (`hkmztrk/DeepDTA`)](https://github.com/hkmztrk/DeepDTA) fixado no commit `a546a8433a6822e958f36171c4356ad6f414d623` (verificado em 2026-11-09, ADR #9):
- **Download direto e autônomo:** Não requer conta, token ou configuração de credenciais no Colab ou na máquina local. O script baixa `ligands_can.txt`, `proteins.txt` e a matriz `Y` via HTTP direto do GitHub raw.
- **Alternativa Kaggle (`kagglehub`):** Caso prefira utilizar o espelho do Kaggle (`rajaryan2315/davis-and-kiba-datasets`), basta passar `--data-source kaggle` na CLI ou cadastrar `KAGGLE_USERNAME` e `KAGGLE_KEY` no painel lateral **Secrets (ícone de chave 🔑)** do Google Colab.

### 1.2 Volumetria Canônica na Tabela Bruta vs. Tabela Limpa
Os `assert`s de procedência rodam **sobre a tabela bruta antes de qualquer filtro ou canonização**:

| Dataset | Fármacos por ID (`num_drugs_by_id`) | Alvos por ID (`num_targets_by_id`) | Pares Brutos Esperados (`raw_num_pairs`) | Unidade de Afinidade |
| :--- | :---: | :---: | :---: | :---: |
| **Davis** | 68 | 442 | $68 \times 442 = 30.056$ | $pK_d = -\log_{10}(K_d / 10^9)$ |
| **KIBA** | 2.111 | 229 | $118.254$ | KIBA score (sem conversão log) |

Após a remoção de contra-íons/sais (seleção do maior fragmento covalente pelo RDKit, mantendo o SMILES original em `smiles_raw`) e a canonização isomérica, dois `drug_id`s distintos podem colapsar no mesmo `smiles` canônico (ou dois `target_id`s na mesma `sequence`). Por isso, o relatório `data/processed/{dataset}_ingestion_report.json` registra:
- `num_drugs_by_id` vs. `num_drugs_by_canonical_smiles`
- `num_targets_by_id` vs. `num_targets_by_sequence`
- `duplicate_content_pairs_count`: pares com mesmo `(smiles, sequence)` vindos de IDs diferentes.
- **Política padrão (`duplicate_policy="keep_both"`):** nenhum par é descartado silenciosamente (`raw_num_pairs - clean_num_pairs == num_pairs_dropped_recorded`). Como o particionamento `cold_drug`/`cold_target` agrupa pelo conteúdo (`smiles` e `sequence`), pares duplicados por conteúdo são sempre alocados na mesma partição (`train`, `val` ou `test`).

---

## 2. Escala de Afinidade e Censura à Direita no Davis ($pK_d = 5.0$)

- **Detecção no nível do dataset inteiro (B2):** A função `standardize_dataset` recebe `affinity_unit: "nM" | "pKd" | "kiba_score" | "auto"`. No modo `"auto"`, a inspeção é feita sobre a distribuição global do dataset (ex.: `max > 11.0` $\Rightarrow$ `"nM"` para Davis), levantando `ValueError` caso a distribuição seja ambígua. O **KIBA** utiliza exclusivamente `"kiba_score"` e jamais sofre conversão para $pK_d$.
- **Impacto dos valores censurados no Davis:** No ensaio experimental original de Davis et al. (2011), interações com dissociação fraca ou indetectável receberam o valor teto $K_d = 10.000\text{ nM}$, o que corresponde a **$pK_d = 5.0$** (cerca de ~70% dos 30.056 pares).
  - Essa concentração em $pK_d = 5.0$ cria uma distribuição altamente assimétrica (inflada no limite inferior de afinidade), afetando a sensibilidade do MSE em pares ativos vs. inativos e exigindo avaliação conjunta com métricas de ordenação (**Concordance Index — CI**, **Spearman**) e correlação (**Pearson**, $r_m^2$).
  - O relatório de ingestão contabiliza explicitamente `davis_censored_count` e `davis_censored_pct`.

---

## 3. Protocolo de Splits Sem Vazamento e Variância em `cold_drug` (Davis)

Os splits são gerados para **5 seeds** (`[0, 1, 2, 3, 4]`) nas proporções **80% treino / 10% validação / 10% teste** sobre:
- `random`: pares $(d, t)$ sorteados aleatoriamente.
- `cold_drug`: agrupamento estrito por **SMILES canônico** (`smiles`).
- `cold_target`: agrupamento estrito por **sequência de aminoácidos** (`sequence`).

### 3.1 Alerta de Alta Variância no `cold_drug` do Davis
Como o Davis possui apenas **68 fármacos por ID** (e $N_{\text{canon}} \le 68$ grupos de SMILES canônicos únicos), a divisão 80/10/10 aloca **apenas ~7 grupos de fármacos para validação e ~7 grupos para teste** em cada seed:

| Dataset | Regime de Split | Entidades Agrupadas | Grupos em `train` (80%) | Grupos em `val` (10%) | Grupos em `test` (10%) | Observação Estatística |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Davis** | `cold_drug` | $N_{\text{smiles}} \le 68$ | ~54 (ou $N_{\text{smiles}} - 14$) | ~7 | ~7 | **Variância elevada entre seeds**; obrigatório reportar **média ± desvio-padrão nas 5 seeds** (valores exatos por seed gravados em `phase1_summary_report.json`). |
| **Davis** | `cold_target` | $N_{\text{seq}} \le 442$ | ~354 | ~44 | ~44 | Amostragem de alvos estável entre seeds. |
| **KIBA** | `cold_drug` | $N_{\text{smiles}} \le 2.111$ | ~1.689 | ~211 | ~211 | Amostragem de fármacos ampla e estável. |
| **KIBA** | `cold_target` | $N_{\text{seq}} \le 229$ | ~183 | ~23 | ~23 | Variância moderada; reportar média ± desvio-padrão nas 5 seeds. |

---

## 4. Extração Offline do ESM-2, Janela Deslizante e Aminoácidos Estendidos

### 4.1 Política para Aminoácidos Ambíguos/Raros (ADR #8)
Os caracteres `X`, `B`, `U` (selenocisteína), `Z` e `O` (pirrolisina) fazem parte do vocabulário nativo do `fair-esm` e são **mantidos** intactos. Como `U` e `O` possuem embeddings pouco treinados no ESM-2 (e são raros ou inexistentes em Davis/KIBA), o relatório registra `ambiguous_aa_seq_count` e `ambiguous_aa_char_counts`.

### 4.2 Janela Deslizante para Sequências Longas (>1022 aa — B1)
O ESM-2 suporta no máximo 1022 resíduos por janela (1024 tokens com `<cls>` e `<eos>`). Para sequências com $L > 1022$:
- Aplica-se janela deslizante configurada em `configs/base.yaml` (`max_len: 1022`, `overlap: 256`, passo `stride = 766`).
- Em cada janela, os tokens `<cls>` e `<eos>` são removidos antes da agregação.
- Nas regiões sobrepostas, os embeddings por resíduo são combinados por **média aritmética exata**, produzindo um tensor final de shape `[len(sequence), esm_dim]`.

### 4.3 Isolamento por Modelo, Armazenamento `fp16` e Staging via Tarball (B4)
- Os embeddings são salvos em `torch.float16` em `cache/esm_embeddings/{model_name}/{hash}.pt` e convertidos para `torch.float32` em `load_cached_protein_embedding`, validando se `dim == esm_dim`.
- Para eliminar o gargalo de leitura de centenas de arquivos pequenos no Google Drive FUSE durante o treino na Fase 2, o pipeline empacota o diretório em `cache/esm_embeddings/esm_embeddings_{model_name}.tar`. No início da sessão de treino no Colab, descompacte para o SSD local da VM:

```bash
mkdir -p /content/local_cache/esm_embeddings
tar -xf /content/drive/MyDrive/physdual-gnn/cache/esm_embeddings/esm_embeddings_esm2_t33_650M_UR50D.tar -C /content/local_cache/esm_embeddings
```

---

## 5. Verificação do Gate de Pronto da Fase 1

Após executar a extração, rode o verificador automatizado:

```bash
python scripts/verify_phase1.py --config configs/base.yaml
```

O script retorna `exit code 0` somente se todos os 7 critérios do Gate de Pronto da Fase 1 passarem (cobertura total de embeddings, shape `[len(seq), esm_dim]`, ausência de `NaN`/`Inf` e grafos vazios, 30 splits sem vazamento por conteúdo, procedência bruta exata com perda zero não registrada, carregamento estrito com `weights_only=True` e completude do `phase1_summary_report.json`).
