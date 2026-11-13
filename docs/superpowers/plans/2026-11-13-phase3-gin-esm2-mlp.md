# Fase 3: Dual-Encoder Enxuto (`GIN + ESM-2 congelado + MLP`), Controles ESM em CPU e Splits por Identidade — Plano de Implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar e validar, com rigor comparativo pareado, a arquitetura da Fase 3 (**`GINEncoder` para o ligante + embeddings `ESM-2` congelados para a proteína + `MLP` de interação**), antecedida pelos splits de identidade de sequência (`seqid30` e `seqid40`) no Davis e pelos controles baratos em CPU sobre ESM-2 (`knn_esm_target`, `ridge_esm_target_only` e `ridge_esm`).

**Escopo Estrito (Decisão de 2026-11-13 / ADR #14):**
- **Dentro do escopo:**
  1. `GIN` molecular (`78d L1-norm` validado na Fase 2) + embeddings `esm2_t33_650M_UR50D` congelados (`1280d`, `fp16` pré-computado em `cache/esm_embeddings/esm2_t33_650M_UR50D/`) + `MLP`.
  2. Regularização forte no ramo proteico e diagnóstico obrigatório por `best_epoch` + curvas `train/val`.
  3. Avaliação no **Protocolo v2** (`v2_kfold` 5 dobras para Davis `cold_drug` e `cold_target`), **Davis `cold_target` por identidade de sequência (`seqid30` exigente + `seqid40` sensibilidade)**, controle de não regressão em `Davis random (v1_seeds)` e extensão para **KIBA (`v1_seeds`: `random`, `cold_drug`, `cold_target`)**.
- **Fora de escopo na Fase 3:**
  - Módulo físico (Lennard-Jones / Coulomb) → reservado para **Fase 6 (ablação opcional)**.
  - Cross-attention droga-resíduo e grafo de contatos proteicos → reservados para **Fases 4–5**.
  - Testes em mutantes pontuais → **retirados do artigo atual** (o colapso `81 target_id`s → `18` sequências no Davis permanece registrado como diagnóstico de auditoria; construção de dataset curado de mutantes fica como trabalho futuro).
  - Validação cross-dataset Davis ↔ KIBA → **proibida** devido às `179` proteínas idênticas em comum e à incompatibilidade de escalas ($pK_d$ censurado em `5,0` vs. KIBA score).

**Tech Stack:** Python 3.10+, NumPy, Pandas, SciPy, RDKit, Biopython (`1.88`), PyTorch, PyTorch Geometric (PyG), YAML, Pytest.

---

## Global Constraints

1. Não alterar/reinstalar `torch` ou `numpy` no Google Colab (`docs/colab_env_snapshot.txt`).
2. Todo caminho de dados/saída deve respeitar `PHYSDUAL_ROOT` via `physdual.utils.paths.get_project_root()`.
3. Seleção de hiperparâmetros ($\alpha$ do Ridge, regularização do ramo ESM-2, `best_epoch` via `early_stopping`) opera **exclusivamente sobre treino/validação**. O conjunto de teste de cada dobra/seed é avaliado **uma única vez** após o encerramento (`test_isolation_verified: true`).
4. Todos os arquivos `.pt` salvos devem ser 100% compatíveis com `torch.load(..., weights_only=True)` (sem escalares NumPy).
5. Commits em Git sempre com autor `João Miguel Francisco de Souza <joanomiguel@gmail.com>`.

---

## Critério Objetivo de Sucesso (Gate da Fase 3)

Para ser aprovado, o modelo `physdual_gin_esm_mlp` deve **vencer pareado por dobra/seed** (`ΔMSE < 0` e `ΔCI > 0`) no **Protocolo v2 (`cold_target_kfold`, `cold_drug_kfold`)** e nos **splits por identidade (`seqid30`, `seqid40`)** do Davis:
1. **Baselines marginais:** `drug_mean_predictor` e `target_mean_predictor`.
2. **Baselines clássicos de similaridade (`kNN`, $k=5$):** `knn_kmer_target` (`CI = 0.8128` no v2) e `knn_tanimoto_drug` (`CI = 0.7100` no v2).
3. **Controles baratos sobre ESM-2:** `knn_esm_target`, `ridge_esm_target_only` e `ridge_esm`.
4. **Baselines neurais da Fase 2:** `graphdta_gin` e `graphdta_cnn3`.

E manter superioridade consistente em **KIBA (`v1_seeds`)** frente a `graphdta_gin`, marginais, `knn_kmer`/`knn_tanimoto` e controles ESM em CPU, sem regressão em `Davis random`.

---

## Etapa 3.0 — Splits por Identidade de Sequência e Controles Baratos em CPU (Executar Primeiro)

### Task 1: Splits `seqid30` e `seqid40` para Davis `cold_target`
- **Files:**
  - Modify: `src/physdual/data/splits.py`
  - Create: `scripts/generate_davis_seqid_splits.py`
  - Create: `tests/test_splits_seqid.py`
- **Spec técnica:**
  - Alinhamento global Needleman-Wunsch (`Bio.Align.PairwiseAligner`, `mode="global"`, `BLOSUM62`, `open_gap_score=-10.0`, `extend_gap_score=-0.5`).
  - Identidade: `exact_matches / max(len(seq_a), len(seq_b))` (penaliza diferenças de comprimento em proteínas multidomínio).
  - Agrupamento: **ligação simples (*single-linkage*) via componentes conexos** (garante corte estrito `< τ` entre todo par `(treino/val, teste)`).
  - Balanceamento em 5 dobras via algoritmo guloso *LPT* por carga de pares (`test = fold f`, `val = fold (f+1) mod 5`, `train = 3 dobras restantes`), garantindo que `train`, `val` e `test` sejam mutuamente disjuntos com identidade `< τ`.
  - Documentar no JSON de sumário: em `id >= 30%`, *chaining* gera `120` clusters (`60` singletons, Dobra 0 monolítica com `1` cluster = `92` sequências / `6.664` pares); em `id >= 40%`, `199` clusters (`109` singletons, maior cluster `11` sequências / `1.088` pares, `39–40` clusters por dobra).
  - Materializar `data/processed/splits/davis_cold_target_seqid30_fold{0..4}.json` e `davis_cold_target_seqid40_fold{0..4}.json` + relatório `outputs/logs/davis_seqid_splits_summary.json`.
- **Verification:** `pytest tests/test_splits_seqid.py -v` + execução local de `python scripts/generate_davis_seqid_splits.py`.

### Task 2: Controles Baratos em CPU sobre ESM-2 (`knn_esm_target`, `ridge_esm_target_only`, `ridge_esm`)
- **Files:**
  - Create: `src/physdual/eval/esm_baselines.py`
  - Create: `scripts/run_phase3_esm_cpu_baselines.py`
  - Create: `tests/test_esm_baselines.py`
- **Spec técnica:**
  - `mean_pool_embeddings` sobre tensores `fp16` `[L, 1280]` convertidos para `float64` (sem truncamento).
  - `knn_esm_target`: predição no teste pelos `k=5` alvos de treino mais similares em **similaridade do cosseno no espaço `1280d` do ESM-2** que foram testados contra o mesmo fármaco (fallback para `target_mean_predictor`, desempate determinístico com `argsort(kind="stable")`).
  - `ridge_esm_target_only`: `Ridge` (ajuste de `StandardScaler` apenas no treino; seleção de $\alpha \in \{10^{-2}, 10^{-1}, 1, 10, 100, 10^3, 10^4\}$ **somente por `val_mse`**) sobre o vetor `1280d` do alvo.
  - `ridge_esm`: `Ridge` sobre `[ECFP4 (1024 bits, raio 2) ; mean_pool(ESM-2) (1280d)]` (`2304` dimensões), avaliando quanto a combinação linear entrega sem rede neural.
  - Executar sobre: Davis `v2_kfold` (`cold_drug`, `cold_target`), Davis `seqid30`, Davis `seqid40`, Davis `v1_seeds` e KIBA `v1_seeds` → salvar `outputs/results/phase3_esm_cpu_baselines.json`.
- **Verification:** `pytest tests/test_esm_baselines.py -v` localmente (com embeddings sintéticos) e execução de `scripts/run_phase3_esm_cpu_baselines.py` no Colab com Drive montado.

### Task 3: Avaliação dos Baselines da Fase 2 (`graphdta_gin`, `graphdta_cnn3`, `knn_kmer`) nos Splits `seqid30` e `seqid40`
- **Files:**
  - Modify: `scripts/run_phase2_baseline.py` (suportar `--split cold_target_seqid30` e `--split cold_target_seqid40` sob protocolo `kfold`)
- **Objective:** Medir diretamente se `knn_kmer` (`CI = 0.8128` no `v2_kfold`) sofre queda acentuada sob `30%`/`40%` — confirmando se o `cold_target` por identidade exata era favorecido por homólogos — e fixar o patamar exato de `graphdta_cnn3` e `graphdta_gin` nesses splits.

---

## Etapa 3.1 — Arquitetura `PhysDualPhase3` com Regularização Forte e Diagnóstico Anti-Memorização

### Task 4: Modelo `PhysDualPhase3` e DataModule ESM-2 (`src/physdual/models/physdual.py`, `src/physdual/data/datamodule.py`)
- **Ramo Molecular:** `GINEncoder` da Fase 2 (`drug_in_dim=78`, `5` camadas GIN, `hidden_dim=32`, `global_add_pool`, `output_dim=128`, `dropout=0.2`) seguido de `LayerNorm(128)` antes da fusão.
- **Ramo Proteico (`ESMTargetEncoder` — enxuto e fortemente regularizado):**
  1. Leitura em RAM dos `429` embeddings pré-computados `fp16` `[L, 1280]` (`<= 5 MB` após `mean_pool` ou tensores completos sob demanda em SSD local `/content/local_cache/`), convertidos para `fp32`.
  2. **Agregação base:** `Mean Pooling` mascarado sobre `[L, 1280]` (sem parâmetros extras na agregação; pooling por atenção intra-modal leve fica reservado apenas caso `Mean Pooling` não supere `ridge_esm`).
  3. **Normalização e gargalo regularizado:**
     - `LayerNorm(1280)` → `Dropout(p_esm_in)` (`0.4` padrão, configurável `[0.3, 0.5]`)
     - `Linear(1280, d_prot)` (`d_prot = 128` ou `64`) → `GELU`/`ReLU` → `LayerNorm(d_prot)` → `Dropout(p_prot)` (`0.3` padrão)
- **Fusão e Cabeça MLP:** concatenação `[z_drug_norm, z_prot_norm]` → `Linear(128 + d_prot, 512) -> ReLU -> Dropout(p_mlp) -> Linear(512, 256) -> ReLU -> Dropout(p_mlp) -> Linear(256, 1)`.
- **Otimizador e Regularização:** `AdamW` com `weight_decay` explícito (default `1e-3`, busca em `[1e-4, 1e-3, 1e-2]` exclusivamente em `Davis cold_target_kfold` por `val_mse`, antes de rodar teste ou outros splits).
- **Diagnóstico de Memorização (`best_epoch`):**
  - Salvar `history` completo (`train_mse`, `val_mse`, `val_ci`) por época, `best_epoch` e `train_val_mse_gap_at_best`.
  - **Regra de alerta/rejeição de configuração:** `best_epoch < 15` no Davis `cold_target` sinaliza memorização precoce das ~220 proteínas de treino e dispara aumento de regularização antes da avaliação final.

---

## Etapa 3.2 — Orquestração, Treino em GPU e Validação da Fase 3

### Task 5: Orquestrador e Gate da Fase 3 (`scripts/run_phase3_physdual.py`, `scripts/verify_phase3.py`, `docs/FASE3_PHYSDUAL.md`)
1. **Passo A (Prova de conceito no Davis Cold + Identidade em L4):**
   - `Davis v2_kfold`: `cold_target_kfold`, `cold_drug_kfold` (`5` dobras + OOF CSV).
   - `Davis seqid30` e `seqid40` (`5` dobras + OOF CSV).
2. **Passo B (Extensão após passar no Passo A):**
   - `KIBA v1_seeds`: `random`, `cold_drug`, `cold_target` (`5` seeds `[0..4]`).
   - `Davis v1_seeds random` (`5` seeds `[0..4]`, controle de não regressão).
3. **Verificação Automática (`verify_phase3.py`):**
   - Exige `test_isolation_verified: true`, zero `NaN`/`Inf`, presença de `history`, `best_epoch`, `marginal_baselines`, `knn_baselines` e comparação pareada explícita por dobra/seed contra marginais, `knn_kmer`, `knn_tanimoto`, `knn_esm_target`, `ridge_esm`, `graphdta_gin` e `graphdta_cnn3`.
