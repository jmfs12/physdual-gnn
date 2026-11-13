# Passo 6 — Síntese Crítica da Literatura Correlacionada (MAFI-DTA, CIP e TUNA)

> **Data de Consolidação:** 13 de Novembro de 2026  
> **Escopo:** Correlação metodológica, arquitetural e experimental entre três trabalhos recentes de predição de afinidade droga-alvo (DTA) e o protocolo do **PhysDual-GNN** ([TRANSFERENCIA_CONHECIMENTO.md](../../TRANSFERENCIA_CONHECIMENTO.md), [docs/PLANO_AJUSTADO.md](../PLANO_AJUSTADO.md), [docs/DECISOES.md](../DECISOES.md) — ADR #14 e [docs/paper/05_LACUNAS.md](05_LACUNAS.md)).

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

| Eixo | **MAFI-DTA (2026)** | **CIP (2024)** | **TUNA (2026)** | **PhysDual-GNN (Fase 2 $\to$ Fases 3–6)** |
| :--- | :--- | :--- | :--- | :--- |
| **Codificador de Proteína** | `ESM-3` + `ESM-2` (`1280d`) + GEM + BiLSTM + 3 subgrafos de contato `ESM-2` | GNN 3D + RBF de distância ao ligante | `ESM2-650M` (`Bi-LSTM+Transformer`) + `ESM2-8M` de bolso | **Fase 3:** `ESM2-650M` (`1280d` `fp16` congelado em cache, janela `1024/256`) |
| **Codificador de Ligante** | SMILES + Grafo GCN | Grafo 2D/3D (MBP + GNN) | `Chemformer` (1D) $\odot$ `Graph Diffusion` (2D, *one-padding*) | **Fase 3:** `GINEncoder` (`78d L1`); **Fase 4:** fragmentos BRICS/RECAP ou 1D |
| **Interação / Fusão** | Atenção + concatenação + FCN | *Conditional Updating* por descritor global ($O(N+L)$) | *Cross-attention* 2 estágios + concatenação | **Fase 3:** *late fusion* (`MLP`); **Fases 4–5:** *Conditional Updating* ou *Cross-Attention* |
| **Dependência 3D / Bolso** | Sem 3D (contatos do próprio ESM-2) | Exige complexo 3D ligado | Experimental (PDBbind) ou AlphaFold DB + `fpocket`/`DeepSurf` | **Sem 3D na Fase 3/4**; contatos ESM-2 vs. AlphaFold DB apenas na Fase 5 |
| **Avaliação Cold-Start** | `Unseen drug`, `Unseen target`, `All unseen` | **Nenhuma** (`Test2016_290` apenas) | `Unseen protein` + 4 famílias (sem `cold_drug`) | `v2_kfold` (`cold_drug`, `cold_target`) + `v1_seeds` |
| **Controle de Homologia** | Nenhum (`442` IDs brutos no Davis) | Nenhum | Nenhum (`seqid` não controlado) | Deduplicação por conteúdo (`442 -> 379` no Davis) + **`seqid30` e `seqid40`** (`BLOSUM62`) |
| **Baselines Triviais / Lineares** | Nenhum | Nenhum | Nenhum | **Obrigatórios e pareados:** `drug/target_mean`, `knn_tanimoto`, `knn_kmer`, `knn_esm`, `ridge_esm` |
| **Auditoria de Ruído/Censura** | Apenas citada na Discussão | Não | Não | Quantificada (`69,64%` censura $pK_d=5.0$; piso de ruído `MSE=0.0322` / `0.3347`) |

---

## 4. Implicações Diretas para o Roadmap do PhysDual-GNN

### 4.1 O que os três artigos validam na estratégia da Fase 3 (`GIN + ESM2-650M congelado + MLP`)
1. **ESM2-650M congelado como núcleo proteico principal:**
   - **MAFI-DTA** (via ablação de `MEFseq`) e **TUNA** (`Models 1–4` nas Tabelas V–VI) demonstram de forma convergente que a maior parte do ganho em regimes *complex-free* vem da substituição de codificações rasas por representações de PLM (`ESM-2 1280d`).
   - Isso valida diretamente a escolha do PhysDual-GNN ([docs/DECISOES.md](../DECISOES.md#L185-L206)) de consolidar primeiro `GINEncoder (78d L1) + ESM2-650M congelado + MLP` com regularização forte e cache `fp16` (cabendo em T4/L4 `16–22,5 GB`), antes de introduzir módulos pesados.
2. **Proibição empírica de bolsos preditos ou *docking* prematuros (Fase 3 e 4):**
   - A Tabela VI e a Fig. 4c do **TUNA** provam que, quando não há cristalografia (BindingDB), inserir bolsos preditos (`AlphaFold2 + fpocket`) degrada `ESM2 + encoder de ligante` (`Model 4 -> Model 5`), e tentar gerar complexos 3D completos (`AlphaFold2 + fpocket + AutoDock`) derruba modelos 3D (`GIGN`, Fig. 3).
   - Isso confirma o acerto da diretriz em [docs/PLANO_AJUSTADO.md](../PLANO_AJUSTADO.md#L11-L14) de não usar *docking* e manter qualquer informação estrutural/de contato isolada para a **Fase 5**.
3. **Consistência externa do colapso do `GraphDTA`:**
   - Em Davis *random*, nossos números locais ([`T3_main_results.csv`](tables/T3_main_results.csv): `graphdta_gin` `MSE = 0.2348 ± 0.0166`, `CI = 0.8845 ± 0.0057`; `graphdta_cnn3` `MSE = 0.2216 ± 0.0128`, `CI = 0.8895 ± 0.0035`) coincidem com a literatura citada no MAFI-DTA (`MSE = 0.229`, `CI = 0.893`).
   - Em proteínas não vistas, o **TUNA** (Tabela IV, BindingDB 5-fold CV: `GraphDTA` cai de `PCC = 0.820` para `0.302 ± 0.044`) confirma o diagnóstico forense que encontramos no Davis `cold_target` (`graphdta_gin` `v2_kfold`: `CI = 0.7117 ± 0.0112`, derrotado `0/5` em CI e `1/5` em MSE por `drug_mean_predictor` `CI = 0.7488 ± 0.0150` em [`T4_marginal_and_paired.csv`](tables/T4_marginal_and_paired.csv)).

### 4.2 O que aproveitar seletivamente nas Fases 4, 5 e 9 (sem importar complexidade gratuita)
1. **Ordem disciplinada entre *late fusion* (Fase 3) e interação entre ramos (Fases 4–5):**
   - **CIP** fornece o argumento teórico contra a concatenação tardia e oferece um mecanismo de baixo custo computacional (**Conditional Updating** por descritor global, complexidade $O(N_{\text{átomos}} + L)$ em vez de $O(N_{\text{átomos}} \times L)$), muito atrativo para sequências longas ($>1.000$ aa, presentes em `70` sequências únicas do Davis e `42` do KIBA em [`T1_dataset_stats.csv`](tables/T1_dataset_stats.csv)).
   - **TUNA** mostra que *Cross-Attention* ligante$\times$resíduo sobre `ESM2-650M` permite inferir resíduos do sítio de ligação sem 3D de entrada (Fig. 8 e Fig. 10), criando uma ponte direta com a **Fase 9 (Interpretabilidade)** — desde que dispensado o ramo ruidoso `ESM2-8M + fpocket`.
2. **Alternativa sem MSA/bolso para grafos proteicos (Fase 5):**
   - Se a Fase 5 investigar topologia proteica, o **MAFI-DTA** (mapa de contato predito pelo próprio `ESM-2`) é preferível à cadeia `AlphaFold2 + fpocket/DeepSurf` do TUNA em cenários *sequence-only*, desde que validado em ablação estrita contra `GIN + ESM-2 + MLP`.
3. **Cuidados na fusão multimodal do ligante (Fase 4):**
   - Caso experimentos futuros combinem SMILES 1D pré-treinado (`ChemBERTa`/`Chemformer`) com grafos `GIN` 2D, a Fig. 5 do **TUNA** alerta que concatenação direta e *zero-padding* degradam o desempenho frente ao 1D puro, exigindo alinhamento átomo-token com *one-padding* no produto de Hadamard.

### 4.3 Diferencial metodológico exclusivo do PhysDual-GNN frente aos três artigos
Nenhum dos três trabalhos avalia:
1. **Baselines marginais, `kNN` e modelos lineares sobre PLM (`ridge_esm`, `knn_esm_target`):** Sem esses controles pareados, não é possível separar o viés marginal do fármaco (`drug_mean_predictor`: `CI = 0.7488 ± 0.0150` em `v2_kfold`) ou a separabilidade linear do próprio `ESM-2` do ganho real da arquitetura neural — risco explicitamente exposto pelo resultado `p = 0.249` entre `Model 4` e `Model 6` no BindingDB do TUNA.
2. **Partições por identidade global de sequência (`seqid30` e `seqid40`, `BLOSUM62`):** Como já medido em [`outputs/results/phase3_seqid_classical_baselines.json`](../../outputs/results/phase3_seqid_classical_baselines.json), remover homólogos próximos derruba `knn_kmer_target` de `CI = 0.8128` (`v2_kfold`) para `0.7051 ± 0.0397` (`seqid30`) e `0.7217 ± 0.0562` (`seqid40`), abaixo inclusive de `drug_mean_predictor` (`0.7360 ± 0.0291` em `seqid30`).
3. **Auditoria de conteúdo e qualidade de rótulos:** Deduplicação de `442 -> 379` sequências únicas no Davis (`81 target_id`s colapsando em `18` sequências), separação estrita Davis vs. KIBA (`179` sequências compartilhadas entre ambos), censura de `69,64%` em $pK_d = 5.0$ e piso de ruído de duplicatas (`MSE = 0.0322` global / `0.3347` nas `546` duplicatas divergentes).

---

## 5. Como Posicionar os Três Artigos no Manuscrito (Trabalhos Relacionados e Justificativa)

- **MAFI-DTA (Jiang et al., 2026):** Citar como evidência recente por ablação de que o ramo baseado em `ESM-2 1280d` concentra a maior parte do poder preditivo *sequence-only*, e como contraste frente à ausência de controle por `seqid30/seqid40`, marginais e `Ridge-ESM`.
- **TUNA (Yoon et al., 2026):** Citar como referência estado-da-arte *complex-free* baseada em `ESM2-650M` + *Cross-Attention*, destacando seu próprio achado de ablação no BindingDB (Tabela VI / Fig. 4c) de que bolsos preditos sem cristalografia degradam ou não superam significativamente `ESM2 + encoder de ligante`, justificando a não utilização de `fpocket`/docking na Fase 3 do PhysDual-GNN.
- **CIP (Cui et al., 2024):** Citar como motivação arquitetural para superar o gargalo de *late fusion* via condicionamento global leve ($O(N+L)$) nas fases posteriores à validação da linha de base limpa (`GIN + ESM-2 + MLP`), contrastando com a dependência de coordenadas 3D do módulo de proximidade e com a ausência de testes *cold-start*/*seqid*.
