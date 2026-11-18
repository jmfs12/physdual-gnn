# Passo 6 — Síntese Crítica da Literatura Correlacionada e Posicionamento Metodológico (2023–2026)

> **Data de Atualização:** 18 de Novembro de 2026  
> **Escopo:** Correlação metodológica, arquitetural e experimental entre trabalhos recentes de predição de afinidade droga-alvo (DTA) e o enquadramento unificado de avaliação e diagnóstico do **PhysDual-GNN** ([TRANSFERENCIA_CONHECIMENTO.md](../../TRANSFERENCIA_CONHECIMENTO.md), [docs/DECISOES.md](../DECISOES.md) — ADRs #14 e #15, [docs/paper/02_RESULTADOS_PRELIMINARES.md](02_RESULTADOS_PRELIMINARES.md) e [docs/paper/05_LACUNAS.md](05_LACUNAS.md)).

---

## 1. Trabalhos Analisados

| Referência | Arquivo no Repositório | Modelo / Arquitetura | Regime de Entrada | Periódico / Evento |
| :--- | :--- | :--- | :--- | :--- |
| **Jiang et al. (2026)** | [`docs/paper/mafi dta/12915_2026_Article_2589.pdf`](mafi%20dta/12915_2026_Article_2589.pdf) *(pasta herdada como `mafi dta/`, não é Nguyen et al., 2021)* | **MAFI-DTA** | Sequência + Grafo molecular 2D + Subgrafos de contato ESM-2 | *BMC Biology* 24:113 (2026) |
| **Cui et al. (2024)** | [`docs/paper/cip/Enhanced_Protein-Ligand_Affinity_Prediction_with_Conditional_Updating_and_Proxim.pdf`](cip/Enhanced_Protein-Ligand_Affinity_Prediction_with_Conditional_Updating_and_Proxim.pdf) | **CIP** (*Conditional Updating and Proximity Embedding*) | Complexo 3D proteína–ligante (cristal ou *docking* SMINA) | *IEEE BIBM* (2024) |
| **Yoon, Kim & Lee (2026)** | [`docs/paper/tuna/TUNA_A_Target-Aware_Unified_Network_for_Protein-Ligand_Binding_Affinity_Prediction_via_Multi-Modal_Feature_Integration.pdf`](tuna/TUNA_A_Target-Aware_Unified_Network_for_Protein-Ligand_Binding_Affinity_Prediction_via_Multi-Modal_Feature_Integration.pdf) | **TUNA** (*Target-Aware Unified Network for Affinity*) | Sequência + Bolso (experimental no PDBbind ou AlphaFold DB + `fpocket`/`DeepSurf` no BindingDB) + SMILES 1D + Grafo 2D | *IEEE J. Biomed. Health Inform.*, vol. 30, no. 8 (ago. 2026; online dez. 2025) |

---

## 2. Resumo Técnico Individual

### 2.1 MAFI-DTA (Jiang et al., 2026)
- **Problema:** Superar a dependência de MSAs (`PSI-BLAST`/`HHblits`) para informação evolutiva e capturar múltiplos níveis estruturais proteicos sem estruturas 3D experimentais.
- **Arquitetura (3 módulos, p. 12–14):**
  1. **EFE (*Evolutionary Feature Extraction*):** Divide a sequência em $N=10$ segmentos, mascara um segmento por vez e usa **ESM-3** para gerar $N=10$ sequências pseudo-evolutivas ($H = \{S\} \cup \{S_{\text{evo}}^{(i)}\}_{i=1}^{10}$). Codifica as $N+1$ sequências com **ESM-2** (mean pooling em $s_t \in \mathbb{R}^{1280}$), aplica *Group Enhance Module* (GEM) + BiLSTM, codifica SMILES em paralelo e aplica atenção.
  2. **MGFE (*Multi-scale Graph Feature Extraction*):** Além do grafo molecular do fármaco (GCN), constrói 3 grafos proteicos ($k \in \{1, 3, 5\}$ resíduos/nó) usando o mapa de contato $P \in \mathbb{R}^{L\times L}$ predito pelo próprio ESM-2 (binarizado em $\tau = 0.5$, com aresta se $> k^2/2$ contatos ocorrerem na submatriz $k \times k$).
  3. **FFM (*Feature Fusion Module*):** Projeta, concatena e regride via FCNs.
- **Resultados e ablação central:**
  - Avaliado em Davis, KIBA, ToxCast, PDBbind e BindingDB (split padrão) e em cenários *cold-start* (`Unseen drug`, `Unseen target`, `All unseen`).
  - No Davis *random*: `MSE = 0.208`, `CI = 0.898` (cita `GraphDTA` com `MSE = 0.229`, `CI = 0.893`). No Davis `Unseen target`: `CI = 0.796 ± 0.017` (atrás de `DMFF` `0.840`, `FusionDTA` `0.826`, `MSGNDTA` `0.816` e `MGDTA` `0.813`).
  - **Achado central da ablação (Fig. 1–4, p. 8–9):** Remover o extrator de sequência (`MEFseq`, que contém o ramo **ESM-2 + SMILES**) causa a maior degradação disparada — no Davis *random*, MSE salta de `0.208` para `0.387`; no Davis `All unseen`, MSE vai de `0.704` para `0.976` e CI cai de `0.703` para `0.605` — muito acima da perda ao remover os subgrafos de contato (`Subgraph`: MSE `0.229` / `0.845`) ou as sequências do ESM-3 (`Evolutionary`: MSE `0.225` / `0.857`).
  - Na discussão (p. 12), os autores reconhecem sensibilidade a medições ruidosas/conflitantes; a escolha de $k$ nos grafos proteicos exigiu `11–14 GB` de VRAM (p. 7).

### 2.2 CIP (Cui et al., 2024)
- **Problema:** Modelos de duas torres desconectadas extraem representações de proteína e ligante separadamente e só concatenam ao final (*late fusion*), perdendo complementaridade e falhando em destacar resíduos críticos do sítio.
- **Mecanismos (Seção II, p. 2–3):**
  1. **Conditional Updating bidirecional (Eqs. 1–6):** Usa atenção multi-head ($G=8$) para extrair um descritor global de um grafo (ex.: $G_p \in \mathbb{R}^{1\times c}$ da proteína) e condiciona localmente os nós do outro grafo via *gating* residual:
     $$X'_l = X_l + (X_l \odot a_l^T), \quad a_l = \text{softmax}(W_l X_l (T^T W_g G_p))$$
  2. **Proximity Embedding (Eq. 7):** Codifica a distância 3D $d_i$ de cada resíduo ao átomo mais próximo do ligante via RBF gaussiana $\phi(d_i, c_j) = \exp(-(d_i - c_j)^2 / 2\sigma^2)$.
  3. **Fusão covalente + não-covalente:** Une interações intramoleculares e intermoleculares antes da MLP.
- **Avaliação:** Pré-treinado no ChEMBL-Dock (313.224 pares com poses SMINA) e avaliado no core set do PDBbind v2016 (`Test2016_290`), reduzindo o RMSE da base MBP de `1.263` para `1.223` (`R`: `0.825` $\to$ `0.843`; só o *Conditional Updating* já reduz para `1.247`, `R = 0.834`). Usa `EdgeSHAPer` (PDB `3ARU`) para explicabilidade.

### 2.3 TUNA (Yoon, Kim & Lee, 2026)
- **Problema:** Combinar escalabilidade *complex-free* com consciência do sítio de ligação (*target-aware*), usando bolsos experimentais quando disponíveis (PDBbind) ou preditos quando ausentes (BindingDB).
- **Arquitetura (Fig. 1–2, p. 6500–6503):**
  1. **Proteína global:** `ESM2-650M` (token-level) $\to$ `Bi-LSTM & Transformer`.
  2. **Bolso (*Pocket*):** `ESM2-8M` pré-treinado por MLM (15% de máscara) em 18.065 bolsos do PDBbind ($\le 10\text{ \AA}$) $\to$ `Bi-LSTM`. No BindingDB (`TUNA-Pred`), obtém estruturas do **AlphaFold DB** e prediz bolsos com **`fpocket`** ou **`DeepSurf`**.
  3. **Ligante multimodal (1D + 2D):** `Chemformer` sobre SMILES (323 tokens do ChEMBL 27) + grafo RDKit (41d/átomo) via blocos de **Graph Diffusion** (2 GCN + kernel de calor $S = e^{-\alpha L}$ + potências da matriz de transição com *self-loop*). As duas modalidades são alinhadas átomo-token ($x_{\text{aligned 2D}} = A^T x_{2\text{D}}$) com ***one-padding*** ($+P$ nos símbolos sintáticos do SMILES) e fundidas via produto de Hadamard ($x_{\text{fusion}} = x_{1\text{D}} \odot x_{\text{final 2D}}$).
  4. **Cross-Attention em 2 estágios:** (a) *Multi-head* entre bolso e ligante 1D; (b) *Single-head* entre proteína global e ligante fundido (cujos pesos médios ao longo do ligante inferem os resíduos do sítio de ligação).
- **Resultados e achado crítico nas ablações (Tabelas I–VI e Fig. 3–5):**
  - No PDBbind (`Test_2016` 5-fold CV *protein-unseen*, Tabela II): TUNA com bolso experimental atinge `PCC = 0.756 ± 0.019`, `CI = 0.776 ± 0.010` (vs. `GraphDTA` `PCC = 0.646 ± 0.041`, `CI = 0.725 ± 0.015`); `TUNA-Pred (DeepSurf)` cai para `PCC = 0.722 ± 0.014`, `CI = 0.762 ± 0.007`.
  - No BindingDB 5-fold CV *protein-unseen* (Tabela IV): `TUNA-Pred (DeepSurf)` atinge `PCC = 0.477 ± 0.041`, `RMSE = 1.346 ± 0.031` no `Test` geral (vs. `GraphDTA` `PCC = 0.302 ± 0.044`, `RMSE = 1.539 ± 0.059`), mas colapsa na família `Channel` (`PCC = 0.028 ± 0.139`; no split *random* chega a `-0.148`). Em 10% do BindingDB ($40.695$ pares), `TUNA-Pred (fpocket)` supera o modelo 3D `GIGN` alimentado com `AlphaFold2 + fpocket + AutoDock` (Fig. 3, $p = 2.23\times 10^{-6}$ em RMSE), evidenciando degradação do modelo 3D por acúmulo de erros nas etapas intermediárias.
  - **Achado central nas ablações (Tabelas V–VI e Fig. 4):** No PDBbind (bolso experimental), cada componente melhora significativamente o modelo ($p < 0.05$). Porém, **no BindingDB (bolso predito, Tabela VI e Fig. 4c)**, `Model 4` (`ESM2 + Chemformer`, sem bolso e sem grafo 2D) obtém `PCC = 0.455 ± 0.039`, `RMSE = 1.384 ± 0.048`; adicionar bolso **predito** (`Model 5`) **degrada** para `PCC = 0.420 ± 0.034`, `RMSE = 1.406 ± 0.038` ($p = 0.085$); e o `Model 6` (TUNA completo, `PCC = 0.468 ± 0.039`, `RMSE = 1.350 ± 0.042`) **não apresenta diferença estatisticamente significativa em PCC frente ao `Model 4` ($p = 0.249$)**.
  - Na fusão 1D+2D (Fig. 5), concatenação e *zero-padding* ficam piores que usar apenas 1D; apenas *one-padding* no produto de Hadamard traz ganho, especialmente em macrociclos (Fig. 6).

---

## 3. Matriz Comparativa com o PhysDual-GNN

| Eixo | **MAFI-DTA (2026)** | **CIP (2024)** | **TUNA (2026)** | **PhysDual-GNN (Fases 1–3 Executadas)** |
| :--- | :--- | :--- | :--- | :--- |
| **Codificador de Proteína** | `ESM-3` + `ESM-2` (`1280d`) + GEM + BiLSTM + 3 subgrafos de contato `ESM-2` | GNN 3D + RBF de distância ao ligante | `ESM2-650M` (`Bi-LSTM+Transformer`) + `ESM2-8M` de bolso | `ESM2-650M` (`1280d` `fp16` congelado em cache `429/429`, janela `1022/256`); ablação `mean`/`max`/`mean_std` (`T16`) |
| **Codificador de Ligante** | SMILES + Grafo GCN | Grafo 2D/3D (MBP + GNN) | `Chemformer` (1D) $\odot$ `Graph Diffusion` (2D, *one-padding*) | `GINEncoder` (`78d L1`) vs. controle `ECFP4` (`ecfp_esm_mlp`, `T14`) |
| **Interação / Fusão** | Atenção + concatenação + FCN | *Conditional Updating* por descritor global ($O(N+L)$) | *Cross-attention* 2 estágios + concatenação | *Late fusion* (`MLP`); arquitetura `L×1280` com atenção cruzada encerrada por **`NO-GO`** (`T16`) |
| **Dependência 3D / Bolso** | Sem 3D (contatos do próprio ESM-2) | Exige complexo 3D ligado | Experimental (PDBbind) ou AlphaFold DB + `fpocket`/`DeepSurf` | **Sem 3D** (foco estritamente *sequence/graph-only* e diagnóstico de representação) |
| **Avaliação Cold-Start** | `Unseen drug`, `Unseen target`, `All unseen` | **Nenhuma** (`Test2016_290` apenas) | `Unseen protein` + 4 famílias (sem `cold_drug`) | `v2_kfold` (`cold_drug`, `cold_target`) + `v1_seeds` (estritamente separados) |
| **Controle de Homologia** | Nenhum (`442` IDs brutos no Davis) | Nenhum | Nenhum (`seqid` não controlado) | Deduplicação (`442 -> 379` no Davis) + **`seqid40`/`seqid30`** (`BLOSUM62`) + curva contínua SW (`T17`, `F12`) |
| **Baselines Triviais / Lineares** | Nenhum | Nenhum | Nenhum | **Obrigatórios e pareados:** `drug/target_mean`, `knn_tanimoto`, `knn_kmer`, `knn_esm`, `ridge_esm` (`T9`–`T17`) |
| **Auditoria de Ruído/Censura** | Apenas citada na Discussão | Não | Não | Quantificada (`69,64%` censura $pK_d=5.0$; piso de ruído `MSE=0.0322` / `0.3347`; sobreposição `78,17%` KIBA–Davis) |

---

## 4. Releitura Crítica após o Resultado Negativo da Fase 3 e o `NO-GO` da Fase 4

### 4.1 O que os três artigos contextualizaram e o que os diagnósticos `T9`–`T17` demonstraram
1. **ESM2-650M congelado como núcleo proteico (`T9`, `T16`):**
   - **MAFI-DTA** (via ablação de `MEFseq`) e **TUNA** (`Models 1–4` nas Tabelas V–VI) motivaram a avaliação disciplinada de `GINEncoder (78d L1) + ESM2-650M congelado + MLP` antes de introduzir módulos pesados.
   - Na execução pré-registrada do **PhysDual-GNN (ADR #15)**, porém, `gin_esm_mlp` recebeu veredito **`nao_atendido_negativo` em `2/2` regimes primários (`kiba/cold_drug`, `kiba/cold_target`) e `2/2` secundários (`kiba/seqid40`, `kiba/seqid30`)**, não superando o melhor baseline sem ESM-2 (`knn_tanimoto_drug` ou `knn_kmer_target`, `T9`, `T13`) e ficando abaixo do próprio `drug_mean_predictor` em `kiba/seqid40` (`0.5072` vs `0.4553`) e `kiba/seqid30` (`0.5069` vs `0.4929`).
   - Adicionalmente, o diagnóstico `T16_ridge_pooling_ablation` (`outputs/results/phase3_colab_ridge_pooling.json`) mostrou que, embora `max` pooling sobre `[L, 1280]` eleve `intra_CI` em `5/5` dobras em `davis/cold_target` (`0.6211 -> 0.6450`) e `kiba/seqid40` (`0.5352 -> 0.5714`), **nem `max` nem `mean_std` recuperam ordenação intrafármaco em `kiba/cold_target_seqid30`** (`residual_drug_mean_esm`: `mean = 0.5075 ± 0.0231`, `max = 0.5155 ± 0.0415` com `4/5` dobras entre `0.4865` e `0.5056` e ganho isolado na dobra `3`, `mean_std = 0.5060 ± 0.0179`), decretando **`NO-GO` para a Fase 4 (`L × 1280` + atenção cruzada)**.
2. **Consistência da proibição de bolsos preditos prematuros e do comportamento do `GraphDTA`:**
   - A Tabela VI e a Fig. 4c do **TUNA** mostram que, no BindingDB (bolsos preditos sem cristalografia), inserir bolsos preditos degrada ou não supera significativamente `ESM2 + encoder de ligante` (`p = 0.249` entre `Model 4` e `Model 6`). De modo convergente, nosso resultado em `T14` e `T17` reforça que, sem controle rigoroso contra baselines lineares (`Ridge-ESM`) e marginais (`drug_mean_predictor`), melhorias aparentes de arquiteturas profundas confundem-se com captura do viés marginal do fármaco ou suporte homólogo (`rho = 0.3982` no Davis e `0.2958` no KIBA).
   - Em Davis *random*, nossos resultados (`T3`: `graphdta_gin` `MSE = 0.2348 ± 0.0166`, `CI = 0.8845 ± 0.0057`; `graphdta_cnn3` `MSE = 0.2216 ± 0.0128`, `CI = 0.8895 ± 0.0035`) coincidem com a literatura citada no **MAFI-DTA** (`MSE = 0.229`, `CI = 0.893`). Em proteínas inéditas, a queda de `GraphDTA` reportada pelo **TUNA** no BindingDB (`PCC = 0.820 -> 0.302 ± 0.044`) converge com o colapso de `graphdta_gin` que medimos em `davis/cold_target` (`v2_kfold`: `CI = 0.7117 ± 0.0112`, margem negativa em todas as `4` faixas de identidade em `T17`, derrotado `0/5` em CI e `1/5` em MSE por `drug_mean_predictor` `0.7488 ± 0.0150` em `T4`).

### 4.2 Direções arquiteturais condicionais (não constituem roadmap ativo pós-`NO-GO`)
Em vista do resultado negativo consolidado na Fase 3 (`T9`) e do **`NO-GO` da Fase 4 (`T16`)**, o presente estudo **encerra o ciclo de desenvolvimento arquitetural** e não executará módulos de *Conditional Updating*, *Cross-Attention*, contatos/AlphaFold ou interpretabilidade sobre `L × 1280`. Qualquer linha arquitetural futura inspirada nesses trabalhos exigiria **superar primeiro o gargalo diagnosticado em `T16` e `T17`** (ausência de sinal linear ou de estatísticas por resíduo `[L, 1280]` do `ESM-2` congelado abaixo de `30%` de identidade no KIBA):
1. **Substituição de *late fusion* (CIP / TUNA):** o *Conditional Updating* por descritor global ($O(N_{\text{átomos}} + L)$) do **CIP** e a *Cross-Attention* ligante$\times$resíduo do **TUNA** permanecem referências arquiteturais apenas caso se supere o limite da representação proteica congelada (por exemplo, via ajuste fino direcionado ou informação experimental de sítio, e adotando critérios alternativos de *early stopping* quando houver validação pequena como `5–6` fármacos em `davis/cold_drug`).
2. **Informação estrutural sem cadeia de *docking* (MAFI-DTA / TUNA):** caso trabalhos futuros investiguem topologia proteica ou fusão multimodal 1D+2D do ligante, o contraste **MAFI-DTA** (contatos sem MSA) vs. **TUNA** (Fig. 3 e Fig. 5: falha do pipeline `AlphaFold2 + fpocket + AutoDock` em `GIGN` e superioridade de *one-padding* no produto de Hadamard) orienta o desenho, mas sempre condicionado à comparação pré-registrada contra `drug_mean`, `kNN` e `Ridge-ESM`.

### 4.3 Diferencial metodológico exclusivo do protocolo de avaliação do PhysDual-GNN frente aos três artigos
Nenhum dos três trabalhos avalia sistematicamente:
1. **Baselines marginais, `kNN` e modelos lineares sobre PLM (`Ridge-ESM` em `mean`/`max`/`mean_std` e *residual stacking*):** sem esses controles pareados, não se separa o viés marginal do fármaco (`drug_mean_predictor`) ou a leitura linear do `ESM-2` do ganho real da arquitetura neural.
2. **Partições por identidade de sequência (`seqid40`/`seqid30`, `BLOSUM62`) combinadas com curva contínua SW (`T17`, `F12`):** permitem quantificar exatamente como a margem de erro sobre `drug_mean` e `Ridge-ESM` decai monotonicamente com a distância ao treino até colapsar abaixo de `30%–40%`.
3. **Auditoria de conteúdo e qualidade de rótulos:** deduplicação `442 -> 379` no Davis (`81 target_id`s em `18` sequências), quantificação da sobreposição Davis–KIBA (`179/229 = 78,17%`), censura de `69,64%` em $pK_d = 5.0$ e piso irreduzível de ruído (`MSE = 0.0322` global / `0.3347` nas `546` duplicatas divergentes).

---

## 5. Como Posicionar os Três Artigos no Manuscrito (Trabalhos Relacionados)

- **MAFI-DTA (Jiang et al., 2026):** Citar como referência representativa de modelos *sequence-only* baseados em `ESM-2 1280d`, destacando sua ablação de `MEFseq` e contrastando com a ausência de controle por `seqid30/seqid40`, marginais e `Ridge-ESM`.
- **TUNA (Yoon et al., 2026):** Citar como referência estado-da-arte *complex-free* baseada em `ESM2-650M` + *Cross-Attention*, destacando seu próprio achado de ablação no BindingDB (Tabela VI / Fig. 4c) de que bolsos preditos sem cristalografia degradam ou não superam significativamente `ESM2 + encoder de ligante` (`p = 0.249`).
- **CIP (Cui et al., 2024):** Citar como abordagem 3D com *Conditional Updating* ($O(N+L)$) que motiva interações ligante–alvo além da concatenação tardia, contrastando com a exigência de complexos 3D e a ausência de testes *cold-start*/*seqid*.

---

## 6. Posicionamento Frente à Literatura Recente de Avaliação Rigorosa e Limites de Generalização em DTA (2023–2026)

Sob o enquadramento unificado de avaliação e diagnóstico deste trabalho, nossos achados dialogam diretamente com estudos recentes (2023–2026) voltados a auditar vieses, vazamento de homologia e poder preditivo real em predição de afinidade droga-alvo:

1. **MIRAGE (Yazdani-Jahromi, Padhi & Garibay, 2026, `arXiv:2609.14491`):**
   - Propõe um protocolo rigoroso para modelos 3D e de *co-folding* no PDBbind, estruturado por suporte de família (clusterização MMseqs2 a `30%`), baselines simples (*family-mean* e *ligand-kNN*) e *bootstrap* agrupado em 2 níveis.
   - **Contraste de domínio e contribuição complementar:** enquanto o MIRAGE avalia modelos 3D/*co-folding* em PDBbind, nosso estudo transfere o rigor equivalente para o domínio predominante em triagem virtual larga baseada em **sequência e grafo (`GraphDTA` e `GIN + ESM-2`) nos benchmarks Davis e KIBA**, adicionando três peças diagnósticas complementares: (i) leitura linear direta do mesmo PLM congelado (**`Ridge-ESM`** com ablação `mean`/`max`/`mean_std` em `T16` e *residual stacking*), (ii) ranqueamento relativo condicional por fármaco em `cold-target` (**`intra_CI`** / `intra_Spearman` estritamente sem concatenação entre dobras) e (iii) **curva contínua de margem de erro** sobre `drug_mean_predictor` e `Ridge-ESM` contra identidade global (`global_id_max`) e local de domínio (`sw_id_aln_len_ge150aa`) com *cluster bootstrap* (`B=2000`, `T17`, `F12`).

2. **PLABench (Wang & Cheng, 2026, `bioRxiv 10.64898/2026.09.17.752433`):**
   - Investiga benchmarks e limites de generalização na predição de afinidade proteína–ligante, discutindo inclusive o fenômeno de modelos aprenderem a média do alvo.
   - **Precisão de escopo:** no PLABench, as análises *within-target* e a discussão detalhada sobre predição da média concentram-se em **CASP16 e ChEMBL35**; já **Davis e KIBA figuram apenas com métricas globais** em partições *warm*/*cold*, **sem** métricas intragrupo (`intra_CI` por fármaco) e **sem** comparação pareada contra baselines marginais (`drug_mean_predictor`/`target_mean_predictor`), de similaridade (`kNN`) ou lineares sobre PLM (`Ridge-ESM`). Nosso estudo preenche exatamente essa lacuna em Davis e KIBA, demonstrando quantitativamente que em `kiba/seqid30` a ordenação intrafármaco desaparece (`intra_CI <= 0.5155` mesmo com `max`/`mean_std` em `T16`) e que modelos profundos podem cair abaixo de `drug_mean_predictor` (`T9`, `T17`).

3. **PLM-GraphDTA (Marijan & Tanasijević, 2026, `bioRxiv 10.64898/2026.08.31.748056`):**
   - Integra modelos de linguagem proteica (PLMs) ao arcabouço do `GraphDTA` (incluindo `GIN`) e relata com exatidão a censura massiva de **`20.931/30.056` (`69,64%`)** pares em `10.000 nM` ($pK_d = 5{,}0$) no Davis, examinando também a contribuição *por proteína* ao CI.
   - **Diferencial frente ao nosso protocolo:** apesar de compartilhar a arquitetura base `GIN + PLM` e a caracterização da censura no Davis, o PLM-GraphDTA **não** deduplica os `442 -> 379` alvos do Davis (`81` `target_id`s colapsados em `18` sequências idênticas), **não** inclui `drug_mean_predictor`, `kNN` (`knn_tanimoto_drug`, `knn_kmer_target`) ou `Ridge-ESM` como baselines pareados, e **não** avalia ranqueamento relativo por fármaco (`intra_CI`) em `cold-target` nem partições por identidade de sequência (`seqid40`/`seqid30`) com curva contínua de margem (`T17`).

4. **Kanakala et al. (2023, *ACS Omega* `8(2):2389–2397`, `doi:10.1021/acsomega.2c06781`):**
   - Utilizam ablações unimodais e validação cruzada agrupada para demonstrar o quanto métricas globais em DTA são sustentadas por informação unilateral e vieses marginais de entidade. Nosso protocolo converge com essa linha ao combinar validação cruzada agrupada (`v2_kfold`), baselines marginais/unimodais (`ridge_aa_composition`, `drug_mean_predictor`, `target_mean_predictor`) e ablação de ligante (`ecfp_esm_mlp` vs `gin_esm_mlp` em `T14`), estendendo-a para leitura linear/não linear de tensores `[L, 1280]` do `ESM-2` (`T16`) e estratificação contínua por identidade (`T17`).

5. **Pré-print `ChemRxiv 10.26434/chemrxiv.15011207/v1` (out. 2026):**
   - Avalia `ESM-2 650M` congelado acoplado a GBDT, inclui uma seção explícita de **resultados negativos** e documenta que o benchmark Davis atribui sequências *wild-type* completas tanto a variantes mutantes quanto a **`18` nomes de domínios `JAK1–3` e `RSK1–4`**.
   - **Convergência:** corrobora independentemente nossa auditoria de colisão de sequências em `davis_clean.csv` (`442` IDs em `379` sequências, piso `MSE = 0.0322`, Seção 1.2 de [03_LIMITACOES_E_AMEACAS_A_VALIDADE.md](03_LIMITACOES_E_AMEACAS_A_VALIDADE.md)) e reforça a conclusão negativa convergente sobre os limites de generalização obtidos com embeddings congelados do `ESM-2 650M` sob controle rigoroso de homologia (`T16`, `T17`).

6. **Controles de permutação/perturbação proteica — *All That Glitters Is Not Gold* (*J. Chem. Inf. Model.* `65(19):10239`) e Gorantla et al. (*J. Chem. Inf. Model.* `64(7):2496`, PMID `37983381`):**
   - Estabelecem controles de permutação/randomização de representações proteicas (também empregados em abordagens como `InceptionDTA` e `ReGeoDTA`) para verificar se modelos de afinidade proteína–ligante exploram de fato o descritor da proteína ou dependem apenas de atalhos do ligante.
   - **Posicionamento e divergência empírica:** posicionamos nosso `--shuffle-esm` (`C4`, `T14`) estritamente como **controle diagnóstico de protocolo** derivado dessa literatura, sem reivindicação de novidade metodológica. Enquanto estudos em proteoquimiometria frequentemente observam pouca degradação sob perturbação do ramo proteico (sugerindo contribuição proteica mínima), em `davis/cold_target` embaralhar os embeddings `ESM-2` derruba `gin_esm_mlp` significativamente **abaixo de `drug_mean_predictor`** (`CI` cai de `0.8195 ± 0.0272` para `0.7146..0.7251` vs `0.7488 ± 0.0150`, `boot dMSE` excluindo zero em `shuf0/1/2`). Isso demonstra que o modelo efetivamente usa a representação `ESM-2` quando há suporte homólogo no treino — mas a decomposição contínua (`T17`, `F12`) mostra que esse ganho sobre `Ridge-ESM` e `drug_mean_predictor` se anula na cauda `<30%` de identidade no Davis e torna-se negativo abaixo de `40%` no KIBA.
