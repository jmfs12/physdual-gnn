# Fase 2 — Baseline Oficial GraphDTA, Protocolo de Treinamento e Avaliação Empírica

Este documento formaliza as especificações arquiteturais, hiperparâmetros verificados, metodologia de avaliação e instruções operacionais da **Fase 2** do projeto **PhysDual-GNN**.

---

## 1. Visão Geral da Fase 2 e Justificativa Científica

O objetivo central da Fase 2 é estabelecer baselines de referência empírica rigorosos e auditáveis para predição de afinidade droga-alvo (*Drug-Target Affinity* — DTA). Para tanto, reproduzimos o modelo clássico **GraphDTA** ([Nguyen et al., 2021](https://doi.org/10.1093/bioinformatics/btaa921)), implementado no repositório de referência [`thinng/GraphDTA`](https://github.com/thinng/GraphDTA) (commit canônico `6162245c6776684b391d30610850fe7077478435`).

A arquitetura oficial adota a variante **GIN (Graph Isomorphism Network)** acoplada a um codificador convolucional 1D para a sequência do alvo proteico, combinando-os em um perceptron multicamadas (MLP).

### 1.1 Módulos Implementados no Repositório
- **Modelos e Encoders ([`src/physdual/models/graphdta.py`](file:///C:/Users/joao.francisco/Documents/dev/physdual-gnn/src/physdual/models/graphdta.py)):**
  - `GINEncoder`: Encoder de grafos moleculares com 5 camadas `GINConv`, `BatchNorm1d` e pooling por soma global (`global_add_pool`).
  - `OfficialGraphDTATargetEncoder`: Encoder oficial de sequências proteicas fiel ao código original do GraphDTA (`Conv1d(1000, 32, 8)` com flatten e projeção densa linear).
  - `DeepDTAStyleCNN3TargetEncoder`: Variante secundária com CNN de 3 camadas estilo DeepDTA (`Conv1d(128, 32, 4) -> Conv1d(32, 64, 8) -> Conv1d(64, 96, 12)` com `AdaptiveMaxPool1d(1)`).
  - `GraphDTAGIN` e `GraphDTACNN3`: Modelos compostos com verificação estrita de dimensionalidade (sem valores mágicos padrão na dimensão de entrada dos fármacos).
- **Engine de Treino e Checkpointing ([`src/physdual/training/`](file:///C:/Users/joao.francisco/Documents/dev/physdual-gnn/src/physdual/training/)):**
  - [`trainer.py`](file:///C:/Users/joao.francisco/Documents/dev/physdual-gnn/src/physdual/training/trainer.py): Loop de treino e validação com early stopping, monitoramento de MSE, gradientes limpos e isolamento estrito de teste.
  - [`checkpoint.py`](file:///C:/Users/joao.francisco/Documents/dev/physdual-gnn/src/physdual/training/checkpoint.py): Salvamento atômico (`save_checkpoint_atomic`) resistente a preempções e carregamento seguro com `torch.load(..., weights_only=True)`.
- **Métricas de Avaliação ([`src/physdual/eval/metrics.py`](file:///C:/Users/joao.francisco/Documents/dev/physdual-gnn/src/physdual/eval/metrics.py)):**
  - Cálculo validado de MSE, RMSE, Pearson ($r$), Spearman ($\rho$), $r_m^2$ e Concordance Index (CI vetorizado com fallback para `lifelines.utils.concordance_index`), além de baselines triviais (`mean_predictor` e `ridge_aa_composition`).
- **Orquestração e Verificação CLI:**
  - [`scripts/run_phase2_baseline.py`](file:///C:/Users/joao.francisco/Documents/dev/physdual-gnn/scripts/run_phase2_baseline.py): Orquestrador idempotente por run, benchmarks preliminares de GPU e agregação estatística com média $\pm$ desvio-padrão ($N=5$ seeds).
  - [`scripts/verify_phase2.py`](file:///C:/Users/joao.francisco/Documents/dev/physdual-gnn/scripts/verify_phase2.py): Verificador objetivo dos 6 Gates de Pronto da Fase 2.
  - [`notebooks/02_graphdta_baseline.ipynb`](file:///C:/Users/joao.francisco/Documents/dev/physdual-gnn/notebooks/02_graphdta_baseline.ipynb): Notebook interativo para o Google Colab.

---

## 2. Especificação Arquitetural: Baseline Oficial vs. Variante Secundária

A literatura de predição DTA pós-GraphDTA frequentemente mescla a representação molecular em grafos com codificadores proteicos de 3 camadas convolucionais inspirados no DeepDTA ([Öztürk et al., 2018](https://doi.org/10.1093/bioinformatics/bty593)). No PhysDual-GNN, estabelecemos uma distinção arquitetural formal:

```mermaid
flowchart TD
    subgraph Fármaco["Fármaco (SMILES -> Grafo Molecular PyG)"]
        A["Nós: Átomos com 78 features one-hot normalizadas em L1"] --> B["Arestas: Ligações covalentes"]
        B --> C["5x GINConv + BatchNorm1d (hidden_dim=32)"]
        C --> D["global_add_pool (Soma Global)"]
        D --> E["Linear(32, 128) + ReLU + Dropout(0.2)"]
    end

    subgraph Alvo["Alvo Proteico (Sequência de Aminoácidos)"]
        F["Sequência de Tokens (max_seq_len=1000, vocab_size=25)"] --> G["nn.Embedding(26, 128)"]
        G --> H["Conv1d(in=1000, out=32, kernel_size=8)"]
        H --> I["Flatten (32 * 121 = 3872)"]
        I --> J["Linear(3872, 128)"]
    end

    subgraph Fusão["Preditor Multicamadas (MLP)"]
        E --> K["Concatenação [xd, xt] -> Vetor 256d"]
        J --> K
        K --> L["Linear(256, 1024) -> ReLU -> Dropout(0.2)"]
        L --> M["Linear(1024, 256) -> ReLU -> Dropout(0.2)"]
        M --> N["Linear(256, 1) -> Afinidade Predita (pK_d / KIBA score)"]
    end
```

### 2.1 Comparativo Paramétrico Detalhado

| Componente | Baseline Oficial (`graphdta_gin`) | Variante Secundária (`graphdta_cnn3`) | Origem no Código / Justificativa |
| :--- | :--- | :--- | :--- |
| **Status no Projeto** | **Baseline Oficial de Comparação** | **Referência Secundária** | ADR #12 |
| **Representação Molecular** | Grafo molecular PyG; nós com 78 features $L_1$-normalizadas (`v2_onehot78_l1norm`) | Idêntica à oficial (`v2_onehot78_l1norm`) | `thinng/GraphDTA:create_data.py` (ADR #10) |
| **Encoder Molecular** | GIN com 5 camadas `GINConv` (dimensão oculta 32), `BatchNorm1d`, `global_add_pool`, projeção linear para 128 | Idêntico ao oficial | `thinng/GraphDTA:models/ginconv.py` |
| **Representação do Alvo** | Sequência inteira de tokens truncada/preenchida em 1000 aminoácidos | Sequência inteira de tokens truncada/preenchida em 1000 aminoácidos | `configs/base.yaml:max_seq_len: 1000` |
| **Encoder do Alvo** | **CNN 1D Original do GraphDTA:** `Embedding(26, 128)`, `Conv1d(1000, 32, kernel_size=8)`, flatten $32 \times 121 = 3872$, `Linear(3872, 128)` | **CNN 3 Camadas estilo DeepDTA:** `Embedding(26, 128, padding_idx=0)`, transposição, `Conv1d(128, 32, 4) -> Conv1d(32, 64, 8) -> Conv1d(64, 96, 12)`, `AdaptiveMaxPool1d(1)`, `Linear(96, 128)` | Oficial: `thinng/GraphDTA:models/ginconv.py`<br>Secundária: padrão adaptado de DeepDTA |
| **Camada de Fusão (MLP)** | `Linear(256, 1024) -> ReLU -> Dropout(0.2) -> Linear(1024, 256) -> ReLU -> Dropout(0.2) -> Linear(256, 1)` | Idêntica à oficial | `thinng/GraphDTA:models/ginconv.py` |
| **Otimizador e Taxa** | Adam com `lr = 0.0005`, `weight_decay = 0.0` (sem scheduler por padrão) | Adam com `lr = 0.0005`, `weight_decay = 0.0` | Nguyen et al. (2021), Tabela 1 |
| **Batch Size e Épocas** | `batch_size = 512`, `epochs = 500` (estendido para `1000` nas seeds `1` e `3` de `kiba_random`, ADR #12/#13) | `batch_size = 512`, `epochs = 500` | Nguyen et al. (2021), Tabela 1; ADR #12/#13 |
| **Paciência Early Stopping** | `patience = 50` (`random`); `davis_cold_patience = 80` (`cold_drug`, `cold_target`) | Idêntica à oficial | Otimização para splits cold (ADR #12) |
| **Escopo de Execução** | Davis (3 splits $\times$ 5 seeds = 15 runs); KIBA condicional ao tempo de benchmark ($\le 15\text{ s/época}$) | Davis (3 splits $\times$ 5 seeds = 15 runs); KIBA secundário | Gestão de cota no Colab (ADR #12) |

> [!IMPORTANT]
> A variante `graphdta_cnn3` é explicitamente rotulada como **"GraphDTA-style com CNN estilo DeepDTA"**. Ela serve apenas para avaliar a sensibilidade da representação convolucional da sequência, **nunca como substituto da reprodução oficial do GraphDTA**.

---

## 3. Tabelas Verificadas da Literatura: Nguyen et al. (2021)

Transcrevemos abaixo os valores publicados por **Nguyen et al. (2021)** no periódico *Bioinformatics* (Vol. 37, n. 8, pp. 1140–1147, DOI: [10.1093/bioinformatics/btaa921](https://doi.org/10.1093/bioinformatics/btaa921)).

### 3.1 Tabela 1 — Hiperparâmetros de Treinamento e Configurações de Redes

| Hiperparâmetro / Arquitetura | Valor Publicado em Nguyen et al. (2021) | Implementação no PhysDual-GNN |
| :--- | :--- | :--- |
| **Otimizador** | Adam | Adam (`torch.optim.Adam`) |
| **Taxa de Aprendizado ($lr$)** | 0.0005 | 0.0005 (`configs/base.yaml`) |
| **Tamanho do Lote (*Batch size*)** | 512 | 512 (`configs/base.yaml`) |
| **Número de Épocas** | 1000 épocas (com checkpoint a cada época) | 500 épocas (com early stopping em validação) |
| **Graph Convolutional Network (GCN)** | 3 camadas convolucionais | N/A (foco no modelo campeão GIN) |
| **Graph Isomorphism Network (GIN)** | **5 camadas GINConv**, dimensão oculta 32 | **5 camadas GINConv**, dimensão oculta 32 (`GINEncoder`) |
| **Graph Attention Network (GAT)** | 2 camadas de atenção | N/A |
| **GAT-GCN Híbrido** | 2 camadas GAT + 2 camadas GCN | N/A |

### 3.2 Tabela 2 — Resultados de Referência no Dataset Davis (Setting 1 / Random Split)

A Tabela 2 do artigo original avalia o desempenho dos modelos no particionamento aleatório (*random split*) do dataset Davis. Os valores são ordenados por desempenho relativo:

| Modelo / Método | Representação Alvo / Fármaco | Concordance Index (CI) $\uparrow$ | Erro Quadrático Médio (MSE) $\downarrow$ |
| :--- | :--- | :---: | :---: |
| DeepDTA | Smith-Waterman / Pubchem-Sim | 0.790 | 0.608 |
| DeepDTA | 1D (CNN) / Pubchem-Sim | 0.835 | 0.419 |
| DeepDTA | Smith-Waterman / 1D (CNN) | 0.886 | 0.420 |
| KronRLS | Perfil de similaridade | 0.871 | 0.379 |
| SimBoost | Boosting baseado em similaridade | 0.872 | 0.282 |
| DeepDTA | 1D (CNN) / 1D (CNN) | 0.878 | 0.261 |
| WideDTA | Palavras proteicas / SMILES | 0.886 | 0.262 |
| GraphDTA (GCN) | 1D (CNN) / GCN (3 camadas) | 0.880 | 0.254 |
| GraphDTA (GAT_GCN) | 1D (CNN) / GAT-GCN | 0.881 | 0.245 |
| GraphDTA (GAT) | 1D (CNN) / GAT (2 camadas) | 0.892 | 0.232 |
| **GraphDTA (GIN)** | **1D (CNN) / GIN (5 camadas)** | **0.893** | **0.229** |

> [!NOTE]
> - Segundo **Nguyen et al. (2021)**, *Bioinformatics* 37(8), p. 1144 (**Tabela 2**, Davis *Setting 1*): DeepDTA `CI = 0.878, MSE = 0.261`; WideDTA `CI = 0.886, MSE = 0.262`; GraphDTA GCN `CI = 0.880, MSE = 0.254`; GAT_GCN `CI = 0.881, MSE = 0.245`; GAT `CI = 0.892, MSE = 0.232`; **GraphDTA GIN `CI = 0.893, MSE = 0.229`**.
> - Na **Tabela 3** (p. 1144, KIBA *Setting 1*): DeepDTA `CI = 0.863, MSE = 0.194`; WideDTA `CI = 0.875, MSE = 0.179`; GraphDTA GAT `CI = 0.866, MSE = 0.179`; **GraphDTA GIN `CI = 0.882, MSE = 0.147`**; GCN `CI = 0.889, MSE = 0.139`; GAT_GCN `CI = 0.891, MSE = 0.139`.
> - Métricas ausentes nas Tabelas 2 e 3 (Pearson, Spearman, $r_m^2$, AUROC/AUPR e regimes *cold-start*) permanecem como *"a verificar"* na literatura externa.

---

## 4. Análise Crítica e Distinção Metodológica Crucial (ADR #12)

### 4.1 A Meta do Projeto (~0.89 CI) e a Saturação do Split Aleatório

O artigo original demonstra que o GraphDTA GIN já alcança **CI = 0.893** no particionamento aleatório do Davis. Esta observação conduz a uma conclusão científica de suma importância:

1. **Saturação do Particionamento Aleatório:** O split `random` permite que o mesmo fármaco ou a mesma proteína apareça simultaneamente no conjunto de treinamento e no conjunto de teste (em pares de interação distintos). Como o Davis possui apenas 68 fármacos únicos e 379 alvos de sequência única, o modelo memoriza com facilidade os perfis basais de afinidade dos ligantes e receptores.
2. **O Verdadeiro Desafio Científico do PhysDual-GNN:** O valor agregado e o poder preditivo do PhysDual-GNN não se medem no split aleatório, mas sim nos regimes de **generalização estrita (*cold-start*)**:
   - `cold_drug`: predição para moléculas químicas inteiramente novas (candidatos inéditos de reposicionamento);
   - `cold_target`: predição contra quinases ou receptores inéditos não vistos durante o treinamento;
   - Subpopulações ativas não censuradas ($pK_d > 5.0$);
   - Proteínas longas ($> 1000$ aminoácidos), nas quais a CNN 1D com truncamento sofre perda de resíduos e onde o embedding evolucionário global do **ESM-2** (Fase 3) oferece vantagem estrutural determinante.

### 4.2 Distinção Metodológica: Seleção por Teste (*Leakage*) vs. Isolamento Estrito de Teste

A análise aprofundada do script `training.py` do repositório oficial `thinng/GraphDTA` revelou a seguinte dinâmica no código original:
```python
# Trecho adaptado do training.py original do GraphDTA:
for epoch in range(NUM_EPOCHS):
    train(model, device, train_loader, optimizer, epoch + 1)
    G, P = predicting(model, device, test_loader)  # Avaliava o teste a cada época!
    ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P)]
    if ret[1] < best_mse:
        torch.save(model.state_dict(), model_file_name)
        best_mse = ret[1]
        best_ci = ret[-1]
```

**Diagnóstico de Risco Metodológico:**
O protocolo original do GraphDTA monitora o erro no conjunto de teste a cada época durante 1000 épocas e salva o checkpoint correspondente ao menor erro de teste. Isso constitui um **viés otimista de parada antecipada (*early stopping on test set*)**, incorrendo em vazamento indireto de dados (*data leakage* de hiperparâmetros de parada).

**Protocolo Adotado no PhysDual-GNN (Isolamento Estrito):**
1. O treinamento monitora **exclusivamente o conjunto de validação (`val_loader`)** a cada época.
2. O checkpoint ótimo (`checkpoint_best.pt`) é selecionado pelo menor MSE em `val`.
3. O conjunto de teste (`test_loader`) é avaliado **estritamente uma única vez ao término de todo o treinamento**, carregando os pesos de `checkpoint_best.pt`.
4. Os relatórios registram explicitamente o campo `test_isolation_verified: true`.

> [!WARNING]
> Devido à eliminação desse viés de seleção, é esperado que os resultados empíricos do nosso baseline oficial no Davis `random` apresentem valores ligeiramente inferiores ou mais conservadores que o CI de 0.893 publicado por Nguyen et al. (2021). Essa diferença reflete **rigor metodológico e aderência às melhores práticas de validação em aprendizado de máquina**, e não uma falha de implementação.

---

## 5. Auditoria de Duplicatas e Limitação de Mutantes para a Fase 7 (ADR #11)

Na curadoria de dados da Fase 1, auditamos as duplicatas por identificador e por conteúdo químico/biológico (`smiles` e `sequence`):

### 5.1 Colapso de Identificadores em Entidades Únicas
- **Davis:**
  - 68 `drug_id`s $\rightarrow$ 68 SMILES canônicos únicos (sem colapso químico).
  - 442 `target_id`s $\rightarrow$ **379 sequências únicas de aminoácidos** (colapso de 63 identificadores).
- **KIBA:**
  - 2.111 `drug_id`s $\rightarrow$ **2.065 SMILES canônicos únicos** (colapso de 46 moléculas por sais/isômeros).
  - 229 `target_id`s $\rightarrow$ 229 sequências únicas de aminoácidos.

### 5.2 Limitação Metodológica para Testes com Mutantes (Fase 7)
No dataset Davis bruto derivado do DeepDTA, diferentes isoformas, estados de fosforilação ou variantes mutantes de quinases (por exemplo, formas ativadas ou mutações de resistência clínica) foram indexadas sob `target_id`s distintos na matriz $Y$, mas receberam a **mesma sequência canônica *wild-type* de aminoácidos** no arquivo `proteins.txt`.

**Consequência para a Fase 7 (Resiliência a Mutações):**
> O objetivo da Fase 7 (avaliação de mutações patogênicas ou de resistência a inibidores) **não pode ser testado diretamente sobre essas duplicatas do Davis**, pois para o modelo a entrada proteica é estritamente a sequência de aminoácidos. Se a mutação pontual não estiver expressa na sequência primária, o modelo gerará predições idênticas para a quinase selvagem e para a variante mutada.
>
> Para a Fase 7, qualquer validação de sensibilidade a mutações exigirá **conjuntos dedicados contendo a sequência expressa com a substituição do resíduo (ex.: BRAF V600E, EGFR T790M) ou dados curados de ensaios com mutantes pontuais**.

---

## 6. Determinismo, Censura e Truncamento de Sequências

### 6.1 Determinismo Estrito vs. Prático em GPUs CUDA
A reprodução estrita de modelos baseados em grafos em GPUs CUDA enfrenta limitações de hardware:
- Operações de espalhamento e redução atômica (`torch_geometric.nn.global_add_pool` e `torch_scatter.scatter_add`) operam de forma assíncrona na GPU, gerando variações na ordem de soma de ponto flutuante em nível de bit.
- Para mitigar a dispersão estocástica e garantir auditabilidade:
  - Fixamos sementes globais via `set_seed(seed)` para Python `random`, `numpy` e PyTorch (`torch.manual_seed`, `torch.cuda.manual_seed_all`).
  - Executamos sistematicamente **5 seeds independentes** (`[0, 1, 2, 3, 4]`), reportando as métricas finais na forma de **média $\pm$ desvio-padrão amostral** ($ddof=1$).

### 6.2 Impacto dos 69.64% de Pares Censurados no Davis ($pK_d = 5.0$)
No ensaio experimental de Davis et al. (2011), o limite superior de medição de constante de dissociação foi de $K_d = 10.000\text{ nM}$, truncando interações nulas ou muito fracas em $pK_d = -\log_{10}(10^{-5}) = 5.0$.
- Isso gera uma concentração pontual em $5.0$ em **20.931 dos 30.056 pares (69.64%)**.
- Em razão disso, modelos ingênuos que prevêem valores próximos de $5.0$ podem alcançar MSE relativamente baixo sem aprender afinidades relativas entre inibidores potentes.
- Torna-se imperativo o uso combinado de **Concordance Index (CI)** e **Spearman ($\rho$)**, bem como a comparação com os baselines triviais (`mean_predictor` e `ridge_aa_composition`), exigida no Gate 6 da Fase 2.

### 6.3 Truncamento em 1000 Resíduos vs. Embeddings ESM-2
- No GraphDTA original, a sequência do alvo proteico é uniformemente truncada em **1000 aminoácidos** (`max_seq_len = 1000`).
- No dataset Davis, **70 sequências únicas (`109` `target_id`s)** possuem comprimento superior a `1022` resíduos (algumas ultrapassando `2000` aminoácidos), e **`76` sequências únicas (`7.820` pares)** ultrapassam `1000` aminoácidos. Com o truncamento em `1000` aminoácidos, o GraphDTA descarta completamente a porção C-terminal dessas enzimas.
- Na **Fase 3** (`GIN + ESM + MLP`; física opcional como ablação na Fase 6), os embeddings pré-computados do **ESM-2** processam a proteína inteira com janela deslizante de `1022` resíduos (`overlap = 256`) sobre `112` sequências únicas `> 1022` aa combinadas (`70` Davis + `42` KIBA).

---

## 7. Instruções Operacionais e Execução no Google Colab

### 7.1 Execução no Notebook Interativo
Abra o notebook [`notebooks/02_graphdta_baseline.ipynb`](file:///C:/Users/joao.francisco/Documents/dev/physdual-gnn/notebooks/02_graphdta_baseline.ipynb) no Google Colab com GPU (L4 ou T4):
1. Execute a célula de montagem do Google Drive e verificação de ambiente.
2. Execute a célula de benchmark de velocidade (assegura tempo $\le 15\text{ s/época}$ antes de disparar o treino completo).
3. Execute o treinamento dos baselines para o Davis (3 splits $\times$ 5 seeds).
4. Verifique a agregação estatística gerada em `outputs/results/baseline_graphdta_gin_davis.json`.

### 7.2 Execução via Linha de Comando (CLI)
A orquestração completa é executada pelo script [`scripts/run_phase2_baseline.py`](file:///C:/Users/joao.francisco/Documents/dev/physdual-gnn/scripts/run_phase2_baseline.py):

```bash
# 1. Executar benchmark preliminar de tempo (1 época) e teste rápido (dry-run):
python scripts/run_phase2_baseline.py --benchmark-only
python scripts/run_phase2_baseline.py --dry-run --dataset davis

# 2. Executar o Baseline Oficial (GraphDTA GIN) nas 5 seeds para Davis (splits random, cold_drug, cold_target):
python scripts/run_phase2_baseline.py --dataset davis --model graphdta_gin --split all --seeds 0,1,2,3,4

# 3. Executar a Variante Secundária (GraphDTA CNN3) para comparação:
python scripts/run_phase2_baseline.py --dataset davis --model graphdta_cnn3 --split all --seeds 0,1,2,3,4

# 4. Executar KIBA condicionalmente (se tempo <= 15 s/época):
python scripts/run_phase2_baseline.py --dataset kiba --model graphdta_gin --split random --seeds 0,1,2
```

### 7.3 Idempotência e Retomabilidade em Sessões Preemptivas
O orquestrador implementa persistência individual atômica:
- Cada execução salva seu resultado individual em:
  `outputs/results/runs/{model}_{dataset}_{split}_seed{seed}.json`
- Durante o treino, checkpoints de época e o melhor checkpoint de validação são salvos em:
  `outputs/checkpoints/{model}_{dataset}_{split}_seed{seed}/checkpoint_best.pt`
- Caso a sessão do Colab seja interrompida por inatividade ou preempção de GPU, o script reexecutado detecta runs já concluídas e avança automaticamente para as pendentes (`skip existing runs`).

### 7.4 Verificação Automatizada do Gate de Pronto da Fase 2
Ao término das execuções, execute o verificador formal de prontidão:

```bash
# Validação ágil de contratos, imports e roundtrip de tensores:
python scripts/verify_phase2.py --smoke

# Auditoria completa dos 6 critérios de prontidão da Fase 2:
python scripts/verify_phase2.py --full --config configs/base.yaml
```

**Critérios Auditados pelo Gate de Pronto da Fase 2:**
- **Gate 1:** Presença e integridade dos JSONs de runs individuais com todas as 6 métricas (MSE, RMSE, Pearson, Spearman, $r_m^2$, CI).
- **Gate 2:** Presença e consistência dos JSONs agregados com estatísticas amostrais (média e desvio-padrão com $ddof=1$).
- **Gate 3:** Ausência de arquivos temporários órfãos (`.tmp_*`).
- **Gate 4:** Verifica em `verify_phase2.py` que todo run esperado gravou `test_metrics_full` e `val_metrics_at_best` não vazios, em complemento à garantia estrutural C1 em `trainer.py` (onde `test_loader` nunca é iterado dentro do loop de épocas); a flag explícita `test_isolation_verified` foi incorporada ao schema JSON em `run_phase2_baseline.py`.
- **Gate 5:** Sanidade de convergência (Davis `random` com CI $\ge 0.86$ de referência).
- **Gate 6:** Superação empírica dos baselines triviais (`mean_predictor` e `ridge_aa_composition`).

---

## 8. Diagrama do Pipeline da Fase 2

```mermaid
sequenceDiagram
    autonumber
    actor Dev as Engenheiro / Colab
    participant CLI as run_phase2_baseline.py
    participant DM as get_dta_dataloaders
    participant Net as GraphDTAGIN
    participant Tr as Trainer
    participant Eval as metrics.py
    participant Disk as Google Drive / Disco

    Dev->>CLI: Disparar treino (model=graphdta_gin, dataset=davis, seed=0..4)
    CLI->>DM: Carregar splits e dataloaders (train, val, test)
    CLI->>Net: Instanciar GraphDTAGIN (drug_in_dim=78, seq_len=1000)
    loop Para cada Época (1..500)
        Tr->>Net: Treinar batch (Forward + Adam lr=0.0005)
        Tr->>Net: Validar batch em val_loader (Calcular MSE)
        alt MSE_val < menor MSE anterior
            Tr->>Disk: Salvar atomicamente checkpoint_best.pt
        end
        alt Early Stopping (patience esgotada)
            Tr-->>CLI: Interromper loop de épocas
        end
    end
    CLI->>Disk: Carregar checkpoint_best.pt (weights_only=True)
    CLI->>Net: Inferência no test_loader (UMA ÚNICA VEZ)
    CLI->>Eval: compute_metrics(y_true, y_pred)
    CLI->>Disk: Salvar outputs/results/runs/graphdta_gin_davis_split_seedX.json
    CLI->>Disk: Gerar outputs/results/baseline_graphdta_gin_davis.json (Média ± DP)
    Dev->>CLI: python scripts/verify_phase2.py --full
    CLI-->>Dev: Relatório de Prontidão (Todos os 6 Gates [OK])
```

---

## 9. Auditoria Forense dos Resultados da Fase 2 e Protocolo v2 (GroupKFold de 5 Dobras - ADR #13)

### 9.1 Síntese do Protocolo v1 (5 Seeds)

Após a conclusão das 9 combinações de baseline em protocolo de 5 seeds (`0..4`) — com adoção oficial de 1000 épocas nas seeds `1` e `3` do `kiba_random` conforme ADRs #12 e #13 — os valores consolidados em `outputs/results/baseline_*.json` são:

| Dataset / Split | Modelo | CI (Test) $\uparrow$ | MSE (Test) $\downarrow$ | Pearson ($r$) $\uparrow$ | $r_m^2$ $\uparrow$ | Vs. Média Global (`MSE`) | Veredito |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Davis random** | GIN (oficial) | 0.8845 ± 0.0057 | 0.2348 ± 0.0166 | 0.8455 ± 0.0151 | 0.7008 ± 0.0214 | -71.4% | **Aprovado** (reproduz Nguyen et al., 2021) |
| **Davis random** | CNN3 | 0.8895 ± 0.0035 | 0.2216 ± 0.0128 | 0.8552 ± 0.0088 | 0.7123 ± 0.0297 | -73.0% | **Aprovado** (variante secundária forte) |
| **Davis cold-drug** | GIN (oficial) | 0.6934 ± 0.0382 | 0.5221 ± 0.1495 | 0.4321 ± 0.0790 | 0.1512 ± 0.0580 | -15.2% | Alta variância em v1; reavaliado em GroupKFold v2 |
| **Davis cold-drug** | CNN3 | 0.6918 ± 0.0343 | 0.5169 ± 0.1671 | 0.4341 ± 0.0674 | 0.1630 ± 0.0551 | -16.0% | Comportamento próximo ao GIN; reavaliado em v2 |
| **Davis cold-target** | GIN (oficial) | 0.7284 ± 0.0312 | 0.7493 ± 0.3338 | 0.4652 ± 0.0520 | 0.2102 ± 0.0478 | -10.3% (`best_ep = 4.6 ± 2.1`) | **Falha estrutural da CNN 1D oficial** |
| **Davis cold-target** | CNN3 | 0.7698 ± 0.0374 | 0.5677 ± 0.2908 | 0.6106 ± 0.0833 | 0.3692 ± 0.0936 | -32.0% (`best_ep = 81.0 ± 19.3`) | **Supera baseline marginal de droga** (confirmado em v2) |
| **KIBA random** | GIN (oficial) | **0.8578 ± 0.0134** | **0.1753 ± 0.0190** | **0.8644 ± 0.0133** | **0.7337 ± 0.0301** | **-74.6%** | **Adotado 1000 ep. nas seeds 1 e 3** (seed 1: CI `0.8803`) |
| **KIBA cold-drug** | GIN (oficial) | 0.7560 ± 0.0107 | 0.3808 ± 0.0532 | 0.6672 ± 0.0260 | 0.4082 ± 0.0297 | -43.5% | Generalização consistente |
| **KIBA cold-target** | GIN (oficial) | 0.6579 ± 0.0437 | 0.5137 ± 0.0888 | 0.5137 ± 0.0708 | 0.2416 ± 0.0712 | -25.1% | Queda severa (`-0.1999` CI vs. `random`) e alta variância |

### 9.2 Causa-Raiz no Davis Cold-Target e Papel do CNN3

No Davis `cold_target`, o codificador proteico oficial do GraphDTA (`Conv1d(1000, 32, 8) + Linear(3872, 128)`) opera sobre canais posicionais absolutos sem pooling invariante, parando na época $4.6 \pm 2.1$. O modelo memoriza os alvos de treino e se degenera em um estimador ruidoso da afinidade média de cada fármaco.

Em contrapartida, o `GraphDTACNN3` (estilo DeepDTA com 3 camadas convolucionais 1D e `AdaptiveMaxPool1d(1)`) possui invariância translacional ao longo da sequência proteica. Ele treinou até a época $81 \pm 17$ e superou o baseline marginal por fármaco, estabelecendo-se como o **baseline primário e mais rigoroso a ser batido na Fase 3 pelo PhysDual-GNN com ESM-2**.

### 9.3 Especificação do Protocolo v2 (GroupKFold de 5 Dobras - Opção C)

Conforme estabelecido no **ADR #13**, o Davis é reavaliado sob validação cruzada agrupada de 5 dobras:
1. **Atribuição Determinística e Disjunta:**
   - 5 dobras geradas com `random_state=42`.
   - `cold_drug`: agrupado por `smiles` canônico.
   - `cold_target`: agrupado por `sequence` do alvo.
   - Cada grupo/entidade aparece no conjunto de teste de exatamente uma única dobra. O conjunto de teste consolidado (*Out-of-Fold* — OOF) cobre **100% dos pares do dataset**.
2. **Validação Interna:**
   - Em cada dobra, extrai-se um conjunto de validação interno de ~10% dos grupos de treino (mesmo critério de conteúdo) para controle de early stopping (`patience=80`).
3. **Métricas Consolidadas:**
   - Média $\pm$ desvio amostral ($ddof=1$) entre as 5 dobras.
   - Métricas globais sobre todas as previsões fora da dobra (OOF pooled).
   - Métricas de classificação binária (AUROC e AUPR em $pK_d > 5.0$ e $pK_d \ge 7.0$).
   - Bootstrap agrupado no nível de cluster (*cluster-level bootstrap*, 1.000 replicações, IC 95%) para $\Delta\text{MSE}$ e $\Delta\text{CI}$ pareados contra os baselines marginais de promiscuidade (`drug_mean_predictor` e `target_mean_predictor`).
4. **Controle de Épocas no KIBA Random (Seeds 1 e 3):**
   - As seeds 1 e 3 de KIBA `random` foram estendidas de 500 para 1000 épocas (`--extend-epochs`), preservando o histórico de loss e o estado do otimizador e early stopping a partir do `checkpoint_last.pt`.

### 9.4 Resultados Consolidados do Protocolo v2 (Davis 5-Fold GroupKFold)

Avaliação completa fora da dobra (OOF sobre todos os 30.056 pares do Davis) consolidada em `outputs/results/baseline_*_kfold.json`:

| Split / Modelo | Média entre Dobras (CI / MSE) | OOF Pooled (CI / MSE / $r_m^2$) | AUROC ($pK_d > 5.0$ / $\ge 7.0$) | Bootstrap Agrupado Exploratório vs. Marginal (`IC 95%`, fração favorável) | Referência kNN (CI / MSE) |
| :--- | :---: | :---: | :---: | :--- | :---: |
| **`cold_drug` / GIN** | `0.6623 ± 0.0572` / `0.7739 ± 0.2059` | `0.6424` / `0.7716` / `0.0923` | `0.6569` / `0.7199` | vs. `target_mean`: $\Delta\text{MSE} = +0.0410$ `[-0.0487, 0.1369]`, `220/1000` | `knn_tanimoto_drug`: `0.7100` / `0.7580` |
| **`cold_drug` / CNN3** | `0.6572 ± 0.0776` / `0.7610 ± 0.1403` | `0.6519` / `0.7605` / `0.0911` | `0.6668` / `0.7351` | vs. `target_mean`: $\Delta\text{MSE} = +0.0277$ `[-0.0431, 0.1021]`, `220/1000` | `knn_tanimoto_drug`: `0.7100` / `0.7580` |
| **`cold_target` / GIN** | `0.7117 ± 0.0112` / `0.7524 ± 0.2713` | `0.6861` / `0.7648` / `0.1327` | `0.7093` / `0.7296` | vs. `drug_mean`: $\Delta\text{MSE} = +0.1315$ `[0.0694, 0.2187]`, `0/1000` | `knn_kmer_target`: `0.8128` / `0.4591` |
| **`cold_target` / CNN3** | **`0.7740 ± 0.0180` / `0.5350 ± 0.1152`** | **`0.7715` / `0.5407` / `0.3384`** | **`0.8020` / `0.8368`** | vs. `drug_mean`: $\Delta\text{MSE} = -0.0910$ `[-0.1415, -0.0464]`, `1000/1000` | `knn_kmer_target`: `0.8128` / `0.4591` |

* **Diagnóstico crítico para a Fase 3 (comparações exploratórias):** No Davis `cold_target`, o `GraphDTAGIN` oficial perde em **1000/1000 reamostras bootstrap** para `drug_mean_predictor` (`CI = 0.7488 ± 0.0150`). O `GraphDTACNN3` vence `drug_mean_predictor` em **1000/1000 reamostras** em MSE e **992/1000** em CI ($\Delta\text{CI} = +0.0264$, `IC 95% [0.0038, 0.0463]`), porém ainda permanece abaixo do baseline de homologia `knn_kmer_target` (`CI = 0.8128 ± 0.0231`, `MSE = 0.4591 ± 0.1386`).

### 9.5 Resolução de 1000 Épocas no KIBA Random e Análise Diagnóstica do KIBA

1. **Aplicação da Regra de Decisão e Paridade Metodológica (Teto Efetivo de 1000 Épocas):**
   - Antes da extensão, o agregado das 5 seeds com teto de 500 épocas registrava `CI = 0.8540 ± 0.0062` e `MSE = 0.1812 ± 0.0075`. As seeds `0`, `2` e `4` já haviam encerrado por *early stopping* (`patience = 50`) antes da época 500 (`best_epoch = 294`, `286`, `343`; `total_epochs_run = 344`, `336`, `393`; `CI = 0.8485`, `0.8467`, `0.8556`), enquanto as seeds `1` e `3` ainda estavam evoluindo na época 500.
   - Após extensão até `1000` épocas (preservando otimizador e estado de *early stopping*):
     - **Seed 1:** `best_epoch = 774` (`tot = 824`), passando de `val_ci@501 = 0.8572` (`val_mse = 0.1846`) para **`CI = 0.8803`**, **`MSE = 0.1451`**, `Pearson = 0.8860`, `RM2 = 0.7758`.
     - **Seed 3:** `best_epoch = 509` (`tot = 559`), atingindo **`CI = 0.8579`**, **`MSE = 0.1697`**, `Pearson = 0.8688`, `RM2 = 0.7477`.
   - Como as seeds `0`, `2` e `4` encerraram por *early stopping* muito antes de `500`, seu resultado é idêntico ao que obteriam sob um teto de `1000` épocas; portanto, **o teto efetivo do protocolo de `kiba_random` é `1000` épocas para todas as 5 seeds** (sem necessidade de re-execução das seeds 0, 2 e 4).
   - Para cálculo de velocidade real por época em `kiba_random`, consideram-se apenas os 3 runs não retomados (`seeds 0, 2, 4`): **`22.338 ± 0.014 s/época`** (`23.968,34 s` somados em `357.7 ± 30.9` épocas), desconsiderando os artefatos distorcidos de `sec_per_epoch` das seeds 1 e 3 (`0.646 s` e `2.347 s`), que dividiram apenas o tempo pós-500 pelo total acumulado de épocas.
2. **Comparação com Baselines Marginais e kNN em `kiba_random` (Todas as 5 Seeds, `cpu_audits.json`):**
   - O `graphdta_gin` (`CI = 0.8578 ± 0.0134`, `MSE = 0.1753 ± 0.0190`) vence em **5/5 seeds** tanto em MSE (`ΔMSE = -0.2525 ± 0.0103`) quanto em CI (`ΔCI = +0.1568 ± 0.0134`) frente a `drug_mean_predictor` (`CI = 0.7010`, `MSE = 0.4278`) e supera `target_mean_predictor` (`CI = 0.6852`, `MSE = 0.5944`), `knn_tanimoto_drug` (`CI = 0.7899`, `MSE = 0.3448`) e `knn_kmer_target` (`CI = 0.7451`, `MSE = 0.3649`). Já em `kiba_cold_target` (5/5 seeds), repete-se o colapso do Davis: o `graphdta_gin` (`CI = 0.6579`, `MSE = 0.5137`) perde **0/5 seeds** em MSE (`ΔMSE = +0.0672 ± 0.0281`) e **1/5 seeds** em CI (`ΔCI = -0.0384 ± 0.0400`) frente a `drug_mean_predictor` (`CI = 0.6962`, `MSE = 0.4465`), ficando também atrás de `knn_kmer_target` (`CI = 0.7320`, `MSE = 0.3730`).
3. **Impacto do Truncamento (`> 1000` aa) no KIBA:**
   - Embora de `15.5%` a `20.4%` dos pares de teste possuam alvos com mais de `1000` resíduos, o erro quadrático médio degrada sistematicamente nessa subpopulação em todos os splits:
     - **`kiba_random`** (`20.4%` > 1000 aa): `MSE 0.1642 → 0.2189` (**`+33.3%`**), `CI 0.8624 → 0.8362`.
     - **`kiba_cold_drug`** (`20.2%` > 1000 aa): `MSE 0.3528 → 0.4909` (**`+39.1%`**), `CI 0.7641 → 0.7165`.
     - **`kiba_cold_target`** (`15.5%` > 1000 aa): `MSE 0.4948 → 0.7113` (**`+43.8%`**), com alta dispersão entre seeds (`± 0.3516`).


