# Rascunho 01 — Materiais, Dados e Baselines (Fases 1 e 2)

> **Escopo:** Documentação metodológica estritamente restrita ao que foi implementado e auditado nas **Fases 1 (Dados e Cache ESM-2)** e **2 (Métricas e Baselines GraphDTA)**. Nenhum componente de modelagem da Fase 3 (`physics.enabled: false` em `configs/base.yaml`) é descrito como implementado.
>
> **Referências de apoio:** [00_INVENTARIO.md](file:///C:/Users/joao.francisco/Documents/dev/physdual-gnn/docs/paper/00_INVENTARIO.md), [T1](file:///C:/Users/joao.francisco/Documents/dev/physdual-gnn/docs/paper/tables/T1_dataset_stats.md), [T2](file:///C:/Users/joao.francisco/Documents/dev/physdual-gnn/docs/paper/tables/T2_splits.md), [T8](file:///C:/Users/joao.francisco/Documents/dev/physdual-gnn/docs/paper/tables/T8_configuration.md), [F1](file:///C:/Users/joao.francisco/Documents/dev/physdual-gnn/docs/paper/figures/F1_affinity_distributions.png), [configs/base.yaml](file:///C:/Users/joao.francisco/Documents/dev/physdual-gnn/configs/base.yaml), [docs/DECISOES.md](file:///C:/Users/joao.francisco/Documents/dev/physdual-gnn/docs/DECISOES.md).

---

## 1. Conjuntos de Dados e Curadoria (Fase 1)

Utilizamos dois benchmarks públicos clássicos de predição de afinidade droga-alvo (*Drug-Target Affinity*, DTA), extraídos das fontes canônicas do repositório DeepDTA (`commit a546a8433a6822e958f36171c4356ad6f414d623`, conforme ADR #1 em `docs/DECISOES.md`):

1. **Davis** (`data/processed/davis_clean.csv`, `data/processed/davis_ingestion_report.json`, `T1`, `F1`):
   - **Dimensionalidade e escala:** `30.056` pares droga-alvo observados entre `68` `drug_id`s (`68` SMILES canônicos únicos) e `442` `target_id`s (`379` sequências únicas de aminoácidos). Os valores originais em constante de dissociação ($K_d$, em nM) são transformados para $pK_d = 9 - \log_{10}(K_d)$, resultando em intervalo `[5.000000, 10.795880]`, média `5.451535`, mediana `5.000000` e desvio-padrão `0.894717`.
   - **Censura à esquerda:** Exatamente `20.931` pares (`69.64%` do total) encontram-se censurados no piso experimental de detecção $pK_d = 5{,}0$ ($K_d \ge 10\,\mu\text{M}$). Apenas `9.125` pares (`30.36%`) possuem $pK_d > 5{,}0$ e `2.502` (`8.32%`) possuem $pK_d \ge 7{,}0$.
   - **Comprimento proteico e aminoácidos estendidos:** Entre as `379` sequências únicas, `76` (`7.820` pares) ultrapassam `1.000` aminoácidos (aa) e `70` (`7.412` pares; equivalentes a `109` `target_id`s) ultrapassam `1.022` aa (`máx. = 2.549` aa, `mín. = 244` aa). Uma única sequência (`TESK1`, `244` aa) contém um caractere ambíguo `X` (`ambiguous_aa_seq_count = 1`), preservado integralmente pela política do ADR #3.
   - **Colisão entre identificador (`target_id`) e sequência (`sequence`) e piso de ruído:** Em `proteins.txt`, `18` sequências únicas mapeiam para `81` `target_id`s distintos (incluindo proteínas selvagens e variantes mutantes/isoformas, como `BRAF` vs. `BRAF(V600E)`, `RET` vs. `RET(M918T)/RET(V804L)/RET(V804M)`, `LRRK2` vs. `LRRK2(G2019S)`, `MET` vs. `MET(M1250T)/MET(Y1235D)`, `CDK4-cyclinD1/D3` e domínios quinase de `RSK3`/`RPS6KA4`/`RPS6KA5`). Isso produz `1.224` grupos duplicados por par de conteúdo `(smiles, sequence)` (`5.508` linhas), dos quais `546` grupos (`2.894` linhas) exibem afinidades experimentais distintas (política `keep_both`, ADR #2). Como qualquer modelo baseado apenas na sequência 1D de `proteins.txt` recebe exatamente a mesma entrada nesses grupos, o piso irreduzível de ruído (MSE da média dentro do grupo, `cpu_audits.json`) é `MSE = 0.0322` (`RMSE = 0.1795`) no Davis completo e `MSE = 0.3347` nas linhas divergentes — implicando também que **o benchmark Davis atual não testa resiliência a mutações pontuais**.

2. **KIBA** (`data/processed/kiba_clean.csv`, `data/processed/kiba_ingestion_report.json`, `T1`, `F1`):
   - **Dimensionalidade e escala:** `118.254` pares observados entre `2.111` `drug_id`s (`2.065` SMILES canônicos únicos; `35` SMILES compartilhados por `81` IDs) e `229` `target_id`s (`229` sequências únicas, sem colisão de alvos). O *KIBA score* varia em `[0.000000, 17.200179]`, com média `11.719935`, mediana `11.500000`, desvio-padrão `0.836947` e sem piso de censura (`0.00%`).
   - **Comprimento proteico:** `44` sequências únicas (`24.103` pares) superam `1.000` aa e `42` (`22.848` pares) superam `1.022` aa (`máx. = 4.128` aa, `mín. = 215` aa). Zero aminoácidos estendidos (`X:0, B:0, U:0, Z:0, O:0`).
   - **Duplicatas de conteúdo e piso de ruído:** `524` grupos com mesmo par `(smiles, sequence)` (`1.172` linhas) decorrentes de `drug_id`s sinônimos, sendo `310` grupos (`744` linhas) com afinidades distintas (`keep_both`, ADR #2), gerando piso irreduzível global `MSE = 0.0011` (`RMSE = 0.0335`; `0.1780` nas linhas divergentes, `cpu_audits.json`).

3. **Normalização química:** Todos os SMILES passaram por canonização RDKit sem falha (`smiles_failed_count = 0` em Davis e KIBA). Embora `smiles_modified_count` registre `68` (Davis) e `2.066` (KIBA) strings reescritas pela canonização sintática do RDKit, `outputs/logs/phase1_summary_report.json` confirma `num_modified_fragments = 0` no Davis (nenhum sal/contra-íon multifragmento removido).

---

## 2. Representações Moleculares, Tokenização e Cache ESM-2

1. **Grafos moleculares (`v2_onehot78_l1norm`):**
   - Implementado em `src/physdual/data/featurizer_drug.py` (ADR #5, `configs/base.yaml`). Cada átomo é codificado por um vetor one-hot de `78` dimensões: símbolo atômico (`44` categorias, 43 elementos + `X`), grau (`11` categorias, `0..10`), hidrogênios implícitos (`11`, `0..10`), valência implícita (`11`, `0..10`) e flag de aromaticidade (`1`).
   - Cada vetor nodal recebe normalização $L_1$ por átomo (`normalize_atom_features: true`, fiel a `thinng/GraphDTA`). Arestas bidirecionais usam `edge_feature_dim = 4`. O featurizador converteu `68/68` moléculas no Davis (`data/processed/davis_drug_graphs_v2_onehot78_l1norm.pt`, KIBA equivalente em `kiba_drug_graphs_v2_onehot78_l1norm.pt`).

2. **Tokenização proteica para baselines da Fase 2 (`GraphDTAGIN` e `GraphDTACNN3`):**
   - Sequências são tokenizadas caractere a caractere (`protein_vocab_size = 25`) e **truncadas ou preenchidas com zeros em `max_seq_len = 1.000` resíduos** (`src/physdual/models/graphdta.py`, `configs/base.yaml`). Consequentemente, `76` sequências únicas do Davis (`7.820` pares) e `44` do KIBA (`24.103` pares) têm sua porção C-terminal além do resíduo `1.000` descartada pelos modelos baseline.

3. **Pipeline de cache ESM-2 (Fase 1, não consumido pelos baselines da Fase 2):**
   - Configurado em `configs/base.yaml` e implementado em `src/physdual/data/esm_cache.py` com `esm2_t33_650M_UR50D` (`repr_layer = 33`, `esm_dim = 1280`, backbone congelado, inferência FP16, `batch_tokens = 2048`).
   - Sequências com comprimento superior a `1.022` aa (`70` únicas / `109` `target_id`s no Davis; `42` no KIBA; total `112` por dataset / `83` globalmente únicas, registrado em `outputs/logs/phase1_summary_report.json`) utilizam janela deslizante (`max_len = 1022`, `overlap = 256`, passo efetivo `766`) com **média aritmética** dos embeddings nas regiões sobrepostas. Os arquivos de cache `.pt` residem no Google Drive (`cache/esm_embeddings/` encontra-se vazio no workspace local); a verificação física de `windowed` (esperado `112`) e shape `[len(seq), 1280]` deve ser executada no Colab via `scripts/paper/audit_esm_cache_colab.py`.

---

## 3. Protocolos de Particionamento (Estritamente Separados)

Todas as `40` partições armazenadas em `data/processed/splits/*.json` (`T2`) foram auditadas quanto à interseção de conteúdo (`smiles` canônico e `sequence` única) e apresentam **zero vazamento (`leak = 0`)** entre treino, validação e teste. Mantemos estritamente separados dois protocolos de avaliação:

1. **Protocolo v1 (`v1_seeds` — splits 80/10/10 em 5 seeds `0..4`):**
   - Aplicado a `davis` e `kiba` nos três cenários: `random` (estratificação por par, Davis `24.044 / 3.006 / 3.006`; KIBA `94.604 / 11.825 / 11.825`), `cold_drug` (Davis `54 / 7 / 7` SMILES canônicos; KIBA `1.653 / 206 / 206` SMILES canônicos) e `cold_target` (Davis `303 / 38 / 38` sequências únicas; KIBA `183 / 23 / 23` sequências).
   - No Davis `cold_drug` (`v1`), o conjunto de teste contém apenas `7` fármacos distintos por seed (`3.094` pares), gerando elevada variância entre seeds.

2. **Protocolo v2 (`v2_kfold` — 5-Fold `GroupKFold` para Davis Cold, ADRs #10/#11):**
   - Criado para avaliar o regime frio sobre **todos os `30.056` pares em predições fora da dobra (OOF)** com `assignment_seed = 42` (`split_seed = 0`) e particionamento em 5 dobras (3 dobras para treino, 1 para validação interna, 1 para teste):
     - **`cold_drug_kfold`**: agrupado por `smiles` canônico (`49 / 5 / 14` ou `49 / 6 / 13` fármacos em treino/val/teste; `5.746` a `6.188` pares em teste por dobra).
     - **`cold_target_kfold`**: agrupado por `sequence` única (`273..274 / 30 / 75..76` sequências únicas em treino/val/teste; `5.372` a `6.596` pares em teste por dobra).
   - **Protocolo v1 e Protocolo v2 jamais são agregados na mesma média ou painel comparativo.**

---

## 4. Arquiteturas Baseline, Treinamento e Baselines de Sanidade (Fase 2)

1. **Modelos Neurais (`src/physdual/models/graphdta.py`):**
   - **`GraphDTAGIN` (Baseline Oficial):** Reproduz `GINConvNet` de `thinng/GraphDTA` (`commit 6162245c6776684b391d30610850fe7077478435`). Combina `GINEncoder` (`5` camadas `GINConv` + `BatchNorm1d`, `hidden_dim = 32`, `global_add_pool`, projeção `output_dim = 128`, `dropout = 0.2`) com `OfficialGraphDTATargetEncoder` (`Embedding(26, 128)` -> `Conv1d(in_channels=max_seq_len=1000, out_channels=32, kernel_size=8)` -> `flatten` -> `Linear(32 * 121, 128)`), seguido por MLP `256 -> 1024 -> ReLU -> Dropout(0.2) -> 256 -> ReLU -> Dropout(0.2) -> 1`. Notavelmente, o encoder proteico oficial trata o índice posicional `0..999` da sequência como canal de entrada da `Conv1d`, tornando cada peso específico da posição absoluta.
   - **`GraphDTACNN3` (Variante Secundária):** Mantém o mesmo `GINEncoder` e o mesmo MLP, substituindo apenas o codificador proteico por `DeepDTAStyleCNN3TargetEncoder`, que transpõe os embeddings para `[B, 128, L]` e aplica `3` convoluções 1D invariantes à translação (`32@k4`, `64@k8`, `96@k12` com ReLU), `AdaptiveMaxPool1d(1)` e `Linear(96, 128)`.
   - No dataset **KIBA**, apenas `GraphDTAGIN` foi treinado; `GraphDTACNN3` foi adiado para após os resultados iniciais da Fase 3 devido ao custo medido no benchmark de tempo (`outputs/results/phase2_timing_benchmark.json`: `20.544 s/época` em NVIDIA L4, acima do teto `eligible_for_cnn3_secondary = false`, ADR #12).

2. **Otimização e Ambiente (`T8`, `configs/base.yaml`, `docs/colab_env_snapshot.txt`):**
   - Otimizador `Adam` (`lr = 0.0005`, `weight_decay = 0.0`, sem scheduler de taxa de aprendizado), `batch_size = 512`, `dropout = 0.2`, loss `MSELoss` sobre variável alvo contínua.
   - Orçamento padrão `epochs = 500`, `patience = 50` (`davis_cold_patience = 80` no Davis frio), seleção do checkpoint pelo menor `val_mse`. **Paridade em `kiba_random` (ADRs #12/#13):** as seeds `1` e `3` foram retomadas e estendidas até `1000` épocas (`best_epoch = 774` e `509`); como as seeds `0, 2, 4` já haviam encerrado por *early stopping* (`patience = 50`) em `344, 336, 393` épocas (muito antes de 500), **o teto efetivo é `1000` épocas para todas as 5 seeds** (sem necessidade de re-execução). A velocidade limpa por época em `kiba_random` (`22.338 ± 0.014 s/época`, `cpu_audits.json`) é calculada apenas nos runs não retomados (`seeds 0, 2, 4`).
   - Isolamento estrito do conjunto de teste: garantido estruturalmente pelo loop C1 em `src/physdual/training/trainer.py` (`test_loader` nunca iterado durante treino/validação), auditado via Gate 4 (`scripts/verify_phase2.py`) e gravado em `test_isolation_verified: true` nos JSONs.
   - Ambiente de treino: Google Colab com `NVIDIA L4` (`22.03 GiB VRAM`), Python `3.13.16`, `torch==2.14.1+cu130`, `torch-geometric==2.8.1`, `rdkit==2026.9.1`, `lifelines==0.30.3`, `scikit-learn==1.6.1`.

3. **Baselines Não Neurais de Sanidade e Diagnóstico (`src/physdual/eval/metrics.py` e `scripts/paper/cpu_audits.py`):**
   - **`mean_predictor`:** prediz a média global do treino no teste (também recomputada em CPU isoladamente sobre o subconjunto `pKd > 5.0` em `docs/paper/cpu_audits.json`).
   - **`ridge_aa_composition`:** `RidgeCV(alphas=logspace(-2, 4, 13))` ajustado exclusivamente sobre a composição percentual de `20` aminoácidos (sem informação do fármaco).
   - **Baselines marginais de promiscuidade** (`compute_marginal_baselines`): `drug_mean_predictor` (média por SMILES no treino; relevante em `cold_target` e `random`) e `target_mean_predictor` (média por sequência no treino; relevante em `cold_drug` e `random`), recomputados também em CPU para todos os `45` runs `v1_seeds` a partir dos CSVs e splits (`cpu_audits.json`).
   - **Baselines de similaridade `kNN` (`compute_knn_baselines`, $k=5$):** `knn_tanimoto_drug` (média ponderada por Tanimoto ECFP4 dos $k=5$ fármacos mais similares testados contra o mesmo alvo) e `knn_kmer_target` (média ponderada por cosseno de 3-mers de aminoácidos dos $k=5$ alvos mais similares testados com o mesmo fármaco), salvos diretamente no `v2_kfold` e recomputados de forma determinística (`argsort kind="stable"`) para todos os `45` runs `v1_seeds` em `cpu_audits.json`.

---

## 5. Métricas e Procedimentos Estatísticos (`src/physdual/eval/metrics.py`)

- **Regressão:** Concordance Index (`CI`, via `lifelines` ou fallback vetorizado exato, omitindo empates em `y_true` e atribuindo `0.5` a empates em `y_pred`), `MSE`, `RMSE`, correlações de `Pearson` ($r$) e `Spearman` ($\rho$), e $r_m^2$ (com guarda `r <= 0 -> 0.0`).
- **Estratificações diagnósticas:** (i) subconjunto não censurado do Davis (`|y_true - 5.0| > 1e-4`), **observando-se explicitamente que o filtro condicionado no rótulo real enviesa o MSE para cima**; (ii) estratificação por comprimento proteico (`<= 1000` vs. `> 1000` aa); (iii) classificação binária (`AUROC` e `AUPR` nos limiares `pKd > 5.0` e `pKd >= 7.0`).
- **Comparação pareada e inferência por cluster (Protocolo v2):** calcula-se a diferença pareada por seed (`v1_seeds`) ou por dobra (`v2_kfold`) $\Delta\text{MSE} = \text{MSE}_{\text{modelo}} - \text{MSE}_{\text{marginal}}$ ($\Delta < 0$ favorável ao modelo) e $\Delta\text{CI} = \text{CI}_{\text{modelo}} - \text{CI}_{\text{marginal}}$ ($\Delta > 0$ favorável). Exclusivamente no `v2_kfold`, realiza-se **bootstrap agrupado no nível de cluster** (`smiles` em `cold_drug`, `sequence` em `cold_target`, `1.000` reamostras) sobre os `30.056` pares OOF, reportando **IC 95% e fração de reamostras favoráveis (`wins/1000`) como análises exploratórias** (evitando valores-p pontuais extremos nas caudas do bootstrap).

---

## 6. Separação Estrita: Observação vs. Hipótese

- **Observações metodológicas verificadas nos artefatos:** contagens brutas e limpas (`30.056` e `118.254`), `69.64%` de censura em $pK_d = 5{,}0$ no Davis, `18` sequências compartilhadas entre `81 target_id`s no Davis e respectivo piso de ruído (`0.0322`), truncamento em `1.000` aa no `GraphDTA`, vazamento zero (`leak = 0`) por conteúdo nos `40` splits, arquiteturas implementadas em `src/physdual/models/graphdta.py` e protocolo efetivo dos `65` JSONs de execução.
- **Hipóteses a serem testadas nas análises subsequentes (não assumidas como fatos causais):** (H1) que o truncamento absoluto em `1.000` aa explique causalmente a perda de desempenho em proteínas longas (visto que famílias de proteínas longas também podem diferir na distribuição de rótulos e censura); (H2) que a sensibilidade posicional absoluta de `OfficialGraphDTATargetEncoder` seja a causa direta do colapso precoce do `GraphDTAGIN` em `cold_target`; (H3) que o modelo da Fase 3 (`GIN + ESM-2 + MLP`, com módulo de física avaliado opcionalmente por ablação na Fase 6) supere os baselines marginais, `kNN` e baselines lineares sobre ESM-2 em cenários frios.
