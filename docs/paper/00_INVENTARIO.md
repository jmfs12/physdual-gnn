# Passo 0 — Inventário e Auditoria de Proveniência (Fases 1 e 2)

- **Gerado automaticamente por:** `scripts/paper/build_inventory_and_repro.py`
- **Commit HEAD do repositório:** `06d1249`
- **Versões de pipeline identificadas:**
  - **Featurizador molecular:** `['não disponível', 'v2_onehot78_l1norm']` (`configs/base.yaml`: `v2_onehot78_l1norm`)
  - **Cache ESM-2 configurado:** `esm2_t33_650M_UR50D` (`esm_dim=1280`, `max_len=1022`, `overlap=256`, `fp16_inference=True`)
  - **Protocolo v1 (`v1_seeds`):** 5 seeds (`0..4`) em splits 80/10/10 (`random`, `cold_drug`, `cold_target`) — 45 runs presentes (`30` Davis + `15` KIBA `graphdta_gin`).
  - **Protocolo v2 (`v2_kfold`):** **EXECUTADO** para Davis Cold (`cold_drug_kfold`, `cold_target_kfold`) com 5-Fold `GroupKFold` (`assignment_seed=42`, agrupado por `smiles` e `sequence`) — 20 runs presentes (`10` `graphdta_gin` + `10` `graphdta_cnn3`), 4 agregados `*_kfold.json` e 4 CSVs OOF (`30.056` pares cada). **Nunca misturado com v1 nas mesmas médias ou figuras.**

## 1. Inventário Geral de Artefatos

| Artefato | Fase | Protocolo / Versão | Commit | Status |
| :--- | :--- | :--- | :--- | :--- |
| `configs/base.yaml` | Fase 0/1/2 | featurizer=v2_onehot78_l1norm, esm=esm2_t33_650M_UR50D (1280d) | `06d1249` | completo |
| `docs/DECISOES.md` (ADRs #1–#13) | Fase 0/1/2 | ADRs #1–#13 | `23927b2` | completo |
| `docs/FASE1_DADOS.md` | Fase 1 | v1_seeds | `6efe404` | completo |
| `docs/FASE2_BASELINE.md` | Fase 2 | v1_seeds + v2_kfold | `1eb1be4` | completo |
| `docs/colab_env_snapshot.txt` | Fase 0/1/2 | Colab NVIDIA L4 (PyTorch 2.14.1+cu130) | `11d9fca` | completo |
| `notebooks/00..02_*.ipynb` (4 arquivos) | Fase 0/1/2 | Colab / Drive | `09a9ac5` | completo |
| `data/raw/davis/{ligands_can.txt,proteins.txt,Y}` | Fase 1 | DeepDTA `a546a84` | `gitignored (local)` | completo |
| `data/raw/kiba/{ligands_can.txt,proteins.txt,Y}` | Fase 1 | DeepDTA `a546a84` | `gitignored (local)` | completo |
| `data/processed/davis_clean.csv` + `davis_ingestion_report.json` | Fase 1 | 30.056 pares | `gitignored (local)` | completo |
| `data/processed/kiba_clean.csv` + `kiba_ingestion_report.json` | Fase 1 | 118.254 pares | `gitignored (local)` | completo |
| `data/processed/{davis,kiba}_drug_graphs_v2_onehot78_l1norm.pt` | Fase 1/2 | `v2_onehot78_l1norm` | `gitignored (local)` | completo (arquivos `.pt` presentes; sem `torch` no `.venv` local) |
| `data/processed/{davis,kiba}_drug_graphs.pt` | Fase 1 | legado / cópia padrão | `gitignored (local)` | completo |
| `data/processed/splits/*.json` (40 arquivos) | Fase 1/2 | 30 `v1_seeds` + 10 `v2_kfold` | `gitignored (local)` | completo (zero vazamento por conteúdo) |
| `cache/esm_embeddings/` | Fase 1 | `esm2_t33_650M_UR50D`, `esm_dim=1280` | `gitignored` | completo |
| `scripts/paper/audit_esm_cache_colab.py` | Fase 1/2 (auditoria) | Colab / Drive (`esm2_t33_650M_UR50D/*.pt`, `fp16`, `NaN/Inf=0`, `dim=1280`, `seq_len`, `windowed=83` global) | `bd5feaf` | executado no Drive (`all_checks_passed: true`, 429/429 válidos, 0 resíduos) |
| `scripts/paper/cpu_audits.py` + `docs/paper/cpu_audits.json` | Fase 1/2 (auditoria CPU) | Noise floor, `pKd>5`, marginais/kNN `v1`, timing limpo, seq-id Davis | `64cae10` | completo (recomputado em CPU a partir de CSVs + splits) |
| `outputs/logs/phase1_summary_report.json` | Fase 1 | datasets=['davis', 'kiba'], `windowed_sequences=112` | `gitignored (local)` | completo (davis, kiba; janelas=112) |
| `outputs/results/phase2_timing_benchmark.json` | Fase 2 | NVIDIA L4, 3 épocas | `gitignored (local)` | completo |
| `outputs/results/baseline_graphdta_cnn3_davis_cold_drug.json` | Fase 2 | v1_seeds | `gitignored (local)` | completo (5 unidades; marginais/kNN em `cpu_audits.json`) |
| `outputs/results/baseline_graphdta_cnn3_davis_cold_drug_kfold.json` | Fase 2 | v2_kfold | `gitignored (local)` | completo (5 unidades) |
| `outputs/results/baseline_graphdta_cnn3_davis_cold_target.json` | Fase 2 | v1_seeds | `gitignored (local)` | completo (5 unidades; marginais/kNN em `cpu_audits.json`) |
| `outputs/results/baseline_graphdta_cnn3_davis_cold_target_kfold.json` | Fase 2 | v2_kfold | `gitignored (local)` | completo (5 unidades) |
| `outputs/results/baseline_graphdta_cnn3_davis_cold_target_seqid30_kfold.json` | Fase 2 | v2_kfold | `gitignored (local)` | completo (5 unidades) |
| `outputs/results/baseline_graphdta_cnn3_davis_cold_target_seqid40_kfold.json` | Fase 2 | v2_kfold | `gitignored (local)` | completo (5 unidades) |
| `outputs/results/baseline_graphdta_cnn3_davis_random.json` | Fase 2 | v1_seeds | `gitignored (local)` | completo (5 unidades; marginais/kNN em `cpu_audits.json`) |
| `outputs/results/baseline_graphdta_gin_davis_cold_drug.json` | Fase 2 | v1_seeds | `gitignored (local)` | completo (5 unidades; marginais/kNN em `cpu_audits.json`) |
| `outputs/results/baseline_graphdta_gin_davis_cold_drug_kfold.json` | Fase 2 | v2_kfold | `gitignored (local)` | completo (5 unidades) |
| `outputs/results/baseline_graphdta_gin_davis_cold_target.json` | Fase 2 | v1_seeds | `gitignored (local)` | completo (5 unidades; marginais/kNN em `cpu_audits.json`) |
| `outputs/results/baseline_graphdta_gin_davis_cold_target_kfold.json` | Fase 2 | v2_kfold | `gitignored (local)` | completo (5 unidades) |
| `outputs/results/baseline_graphdta_gin_davis_cold_target_seqid30_kfold.json` | Fase 2 | v2_kfold | `gitignored (local)` | completo (5 unidades) |
| `outputs/results/baseline_graphdta_gin_davis_cold_target_seqid40_kfold.json` | Fase 2 | v2_kfold | `gitignored (local)` | completo (5 unidades) |
| `outputs/results/baseline_graphdta_gin_davis_random.json` | Fase 2 | v1_seeds | `gitignored (local)` | completo (5 unidades; marginais/kNN em `cpu_audits.json`) |
| `outputs/results/baseline_graphdta_gin_kiba_cold_drug.json` | Fase 2 | v1_seeds | `gitignored (local)` | completo (5 unidades; marginais/kNN em `cpu_audits.json`) |
| `outputs/results/baseline_graphdta_gin_kiba_cold_target.json` | Fase 2 | v1_seeds | `gitignored (local)` | completo (5 unidades; marginais/kNN em `cpu_audits.json`) |
| `outputs/results/baseline_graphdta_gin_kiba_random.json` | Fase 2 | v1_seeds | `gitignored (local)` | completo (5 unidades) |
| `outputs/results/oof/cpu_baselines_davis_cold_drug_oof.csv` | Fase 2 | v2_kfold | `gitignored (local)` | completo (30056 pares OOF) |
| `outputs/results/oof/cpu_baselines_davis_cold_target_oof.csv` | Fase 2 | v2_kfold | `gitignored (local)` | completo (30056 pares OOF) |
| `outputs/results/oof/cpu_baselines_davis_cold_target_seqid30_oof.csv` | Fase 2 | v2_kfold | `gitignored (local)` | completo (30056 pares OOF) |
| `outputs/results/oof/cpu_baselines_davis_cold_target_seqid40_oof.csv` | Fase 2 | v2_kfold | `gitignored (local)` | completo (30056 pares OOF) |
| `outputs/results/oof/graphdta_cnn3_davis_cold_drug_oof.csv` | Fase 2 | v2_kfold | `gitignored (local)` | completo (30056 pares OOF) |
| `outputs/results/oof/graphdta_cnn3_davis_cold_target_oof.csv` | Fase 2 | v2_kfold | `gitignored (local)` | completo (30056 pares OOF) |
| `outputs/results/oof/graphdta_cnn3_davis_cold_target_seqid30_oof.csv` | Fase 2 | v2_kfold | `gitignored (local)` | completo (30056 pares OOF) |
| `outputs/results/oof/graphdta_cnn3_davis_cold_target_seqid40_oof.csv` | Fase 2 | v2_kfold | `gitignored (local)` | completo (30056 pares OOF) |
| `outputs/results/oof/graphdta_gin_davis_cold_drug_oof.csv` | Fase 2 | v2_kfold | `gitignored (local)` | completo (30056 pares OOF) |
| `outputs/results/oof/graphdta_gin_davis_cold_target_oof.csv` | Fase 2 | v2_kfold | `gitignored (local)` | completo (30056 pares OOF) |
| `outputs/results/oof/graphdta_gin_davis_cold_target_seqid30_oof.csv` | Fase 2 | v2_kfold | `gitignored (local)` | completo (30056 pares OOF) |
| `outputs/results/oof/graphdta_gin_davis_cold_target_seqid40_oof.csv` | Fase 2 | v2_kfold | `gitignored (local)` | completo (30056 pares OOF) |
| `outputs/results/oof/physdual_gin_esm_mlp_davis_cold_drug_oof.csv` | Fase 2 | v2_kfold | `gitignored (local)` | completo (30056 pares OOF) |
| `outputs/results/oof/physdual_gin_esm_mlp_davis_cold_target_oof.csv` | Fase 2 | v2_kfold | `gitignored (local)` | completo (30056 pares OOF) |
| `outputs/results/oof/physdual_gin_esm_mlp_davis_cold_target_seqid30_oof.csv` | Fase 2 | v2_kfold | `gitignored (local)` | completo (30056 pares OOF) |
| `outputs/results/oof/physdual_gin_esm_mlp_davis_cold_target_seqid40_oof.csv` | Fase 2 | v2_kfold | `gitignored (local)` | completo (30056 pares OOF) |
| `outputs/results/runs/*_seed*.json` (45 runs) | Fase 2 | v1_seeds (teto efetivo 1000 em `kiba_random`; 500 nos demais) | `gitignored (local)` | completo (`history` ausente em 43; marginais/kNN recomputados em `cpu_audits.json`) |
| `outputs/results/runs/*_fold*.json` (20 runs) | Fase 2 | v2_kfold (`GroupKFold` 5 dobras, `seed=42`) | `gitignored (local)` | completo (com `history`, `marginal`, `knn`, `oof`) |
| `outputs/results/runs/graphdta_cnn3_kiba_*.json` | Fase 2 | v1_seeds | `—` | ausente (omitido pelo critério de orçamento `>15 s/época` no ADR #12) |

## 2. Cobertura por Experimento (`outputs/results/runs/`)

| Protocolo | Modelo | Dataset | Split | Runs | Agregado Presente | Observações |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| `v1_seeds` | `graphdta_cnn3` | `davis` | `cold_drug` | 5 | sim | Sem `history`; marginais/kNN recomputados em `cpu_audits.json` |
| `v1_seeds` | `graphdta_cnn3` | `davis` | `cold_target` | 5 | sim | Sem `history`; marginais/kNN recomputados em `cpu_audits.json` |
| `v1_seeds` | `graphdta_cnn3` | `davis` | `random` | 5 | sim | Sem `history`; marginais/kNN recomputados em `cpu_audits.json` |
| `v1_seeds` | `graphdta_gin` | `davis` | `cold_drug` | 5 | sim | Sem `history`; marginais/kNN recomputados em `cpu_audits.json` |
| `v1_seeds` | `graphdta_gin` | `davis` | `cold_target` | 5 | sim | Sem `history`; marginais/kNN recomputados em `cpu_audits.json` |
| `v1_seeds` | `graphdta_gin` | `davis` | `random` | 5 | sim | Sem `history`; marginais/kNN recomputados em `cpu_audits.json` |
| `v1_seeds` | `graphdta_gin` | `kiba` | `cold_drug` | 5 | sim | Sem `history`; marginais/kNN recomputados em `cpu_audits.json` |
| `v1_seeds` | `graphdta_gin` | `kiba` | `cold_target` | 5 | sim | Sem `history`; marginais/kNN recomputados em `cpu_audits.json` |
| `v1_seeds` | `graphdta_gin` | `kiba` | `random` | 5 | sim | Teto efetivo `1000` épocas (seeds 0,2,4 pararam <500; 1,3 estendidas) |
| `v2_kfold` | `graphdta_cnn3` | `davis` | `cold_drug` | 5 | sim | Com `history`, `marginal`, `knn` e OOF |
| `v2_kfold` | `graphdta_cnn3` | `davis` | `cold_target` | 5 | sim | Com `history`, `marginal`, `knn` e OOF |
| `v2_kfold` | `graphdta_cnn3` | `davis` | `cold_target_seqid30` | 5 | sim | Com `history`, `marginal`, `knn` e OOF |
| `v2_kfold` | `graphdta_cnn3` | `davis` | `cold_target_seqid40` | 5 | sim | Com `history`, `marginal`, `knn` e OOF |
| `v2_kfold` | `graphdta_gin` | `davis` | `cold_drug` | 5 | sim | Com `history`, `marginal`, `knn` e OOF |
| `v2_kfold` | `graphdta_gin` | `davis` | `cold_target` | 5 | sim | Com `history`, `marginal`, `knn` e OOF |
| `v2_kfold` | `graphdta_gin` | `davis` | `cold_target_seqid30` | 5 | sim | Com `history`, `marginal`, `knn` e OOF |
| `v2_kfold` | `graphdta_gin` | `davis` | `cold_target_seqid40` | 5 | sim | Com `history`, `marginal`, `knn` e OOF |
| `v2_kfold` | `physdual_gin_esm_mlp` | `davis` | `cold_drug` | 5 | não | Com `history`, `marginal`, `knn` e OOF |
| `v2_kfold` | `physdual_gin_esm_mlp` | `davis` | `cold_target` | 5 | não | Com `history`, `marginal`, `knn` e OOF |
| `v2_kfold` | `physdual_gin_esm_mlp` | `davis` | `cold_target_seqid30` | 5 | não | Com `history`, `marginal`, `knn` e OOF |
| `v2_kfold` | `physdual_gin_esm_mlp` | `davis` | `cold_target_seqid40` | 5 | não | Com `history`, `marginal`, `knn` e OOF |
| `v1_seeds` | `graphdta_cnn3` | `kiba` | `random / cold_drug / cold_target` | 0 | não | **Adiado** para após os resultados iniciais da Fase 3 (custo `20.544 s/época` > teto `15 s/época`) |

## 3. Auditoria de Consistência, Correções P2 e Limitações de Rastreabilidade

1. **`outputs/logs/phase1_summary_report.json` (Corrigido nesta auditoria):**
   - Originalmente registrava apenas `datasets: ["davis"]` e `windowed_sequences.count = 0`; após atualização em CPU com os relatórios e CSVs limpos, passa a conter **`davis` e `kiba`** e **`windowed_sequences.count = 112`** (`70` sequências únicas/`109` `target_id`s no Davis + `42` no KIBA; `83` sequências únicas deduplicadas globalmente).
   - No `docs/FASE2_BASELINE.md` (§6.3), a menção prévia a "83 quinases > 1022 resíduos" foi corrigida para **`70` sequências únicas** (`109` `target_id`s) no Davis e **`42`** no KIBA (que coincidem em `83` sequências únicas combinadas, totalizando `112` por soma de dataset).
2. **Contagem de censura no Davis (`pKd = 5.0`):**
   - `davis_clean.csv`, `davis_ingestion_report.json`, `phase1_summary_report.json` e `docs/FASE2_BASELINE.md` (§6.2, corrigido de `20.932`) agora concordam exatamente em **`20931` pares** (`69.64%`).
3. **`smiles_modified_count` vs. `num_modified_fragments`:**
   - `davis_ingestion_report.json` registra `smiles_modified_count = 68` (e KIBA `2066`), refletindo recanonização sintática RDKit, enquanto `num_modified_fragments = 0` indica que nenhuma molécula sofreu remoção de sal/contra-íon multifragmento.
4. **Gate 4 de isolamento do teste e chave `test_isolation_verified`:**
   - *O que o Gate 4 verificava:* em `scripts/verify_phase2.py` (`gate4_strict_test_isolation`), o gate checava apenas que os dicionários `val_metrics_at_best` e `test_metrics_full` estavam presentes e não vazios em todos os JSONs de run; o isolamento estrutural do conjunto de teste (nunca iterar `test_loader` durante o loop de treino/early stopping) é garantido em tempo de execução pelo critério **C1** em `src/physdual/training/trainer.py`.
   - *Estado atual nos JSONs:* após atualização em `scripts/run_phase2_baseline.py` e nos artefatos locais, a chave `test_isolation_verified: true` consta em todos os runs e agregados (`missing_runs=65/105`).
5. **Divergências de `epochs` vs. `configs/base.yaml` (`epochs=500`) e paridade em `kiba_random`:**
   - `graphdta_gin_kiba_random_seed1.json`: `epochs=1000` != `configs/base.yaml` (`500`)
   - `graphdta_gin_kiba_random_seed3.json`: `epochs=1000` != `configs/base.yaml` (`500`)
   - `physdual_gin_esm_mlp_davis_cold_drug_fold0.json`: `epochs=1000` != `configs/base.yaml` (`500`)
   - `physdual_gin_esm_mlp_davis_cold_drug_fold1.json`: `epochs=1000` != `configs/base.yaml` (`500`)
   - `physdual_gin_esm_mlp_davis_cold_drug_fold2.json`: `epochs=1000` != `configs/base.yaml` (`500`)
   - `physdual_gin_esm_mlp_davis_cold_drug_fold3.json`: `epochs=1000` != `configs/base.yaml` (`500`)
   - `physdual_gin_esm_mlp_davis_cold_drug_fold4.json`: `epochs=1000` != `configs/base.yaml` (`500`)
   - `physdual_gin_esm_mlp_davis_cold_target_seqid30_fold0.json`: `epochs=1000` != `configs/base.yaml` (`500`)
   - `physdual_gin_esm_mlp_davis_cold_target_seqid30_fold1.json`: `epochs=1000` != `configs/base.yaml` (`500`)
   - `physdual_gin_esm_mlp_davis_cold_target_seqid30_fold2.json`: `epochs=1000` != `configs/base.yaml` (`500`)
   - `physdual_gin_esm_mlp_davis_cold_target_seqid30_fold3.json`: `epochs=1000` != `configs/base.yaml` (`500`)
   - `physdual_gin_esm_mlp_davis_cold_target_seqid30_fold4.json`: `epochs=1000` != `configs/base.yaml` (`500`)
   - `physdual_gin_esm_mlp_davis_cold_target_seqid40_fold0.json`: `epochs=1000` != `configs/base.yaml` (`500`)
   - `physdual_gin_esm_mlp_davis_cold_target_seqid40_fold1.json`: `epochs=1000` != `configs/base.yaml` (`500`)
   - `physdual_gin_esm_mlp_davis_cold_target_seqid40_fold2.json`: `epochs=1000` != `configs/base.yaml` (`500`)
   - `physdual_gin_esm_mlp_davis_cold_target_seqid40_fold3.json`: `epochs=1000` != `configs/base.yaml` (`500`)
   - `physdual_gin_esm_mlp_davis_cold_target_seqid40_fold4.json`: `epochs=1000` != `configs/base.yaml` (`500`)
   - Como as seeds `0, 2, 4` encerraram por *early stopping* (`patience=50`) em `total_epochs_run = 344, 336, 393` (< `500`), **não** precisam ser re-executadas: o teto efetivo é **`1000` épocas para todas as 5 seeds de `kiba_random`**.
   - As seeds `1` e `3` retomadas possuem `history` restrito às épocas pós-500 (`501..824` e `501..559`) e `sec_per_epoch` distorcido (`0.646 s` e `2.347 s`) por dividirem o tempo da retomada pelo total acumulado. Em `T8`, a velocidade limpa de `kiba_random` é recalculada exclusivamente nos 3 runs não retomados (`seeds 0, 2, 4`: **`22.338 ± 0.014 s/época`**).
6. **Recomputação determinística em CPU (`docs/paper/cpu_audits.json`):**
   - Os 43 runs `v1_seeds` originais não gravaram `marginal_baselines`, `knn_baselines` nem `history` em disco, e os JSONs originais não salvavam `mean_predictor` restrito ao subconjunto `pKd > 5.0`. Por via puramente determinística a partir dos CSVs limpos e arquivos de split (sem necessidade de checkpoints GPU), `scripts/paper/cpu_audits.py` recalculou todos os marginais, `kNN`, deltas pareados por seed para os 45 runs `v1_seeds`, baselines em `pKd > 5.0` (`T5`) e piso de ruído de duplicatas (`T1`).
   - **Única lacuna não recomputável localmente:** curvas por época (`history`) dos 43 runs `v1_seeds` originais. Já a auditoria física completa dos `.pt` no Google Drive (`/content/drive/MyDrive/physdual-gnn/cache/esm_embeddings/esm2_t33_650M_UR50D`) foi executada e aprovada (`all_checks_passed: true`): **`429/429`** arquivos presentes (`0` ausentes), **`83/83`** sequências únicas `> 1022 aa` com `windowed=True` (`0` sem campo), **`0` erros de `weights_only=True`**, **`0` erros de `dtype=fp16`**, **`0` ocorrências de `NaN`/`Inf`**, **`0` erros de shape/metadado `dim=1280`**, **`0` divergências de `seq_len` contra os CSVs** e `0` resíduos na raiz.
7. **Limpeza de inchaço em `baseline_graphdta_gin_kiba_random.json`:**
   - Removido `oof_predictions` de `per_seed_results` em `scripts/run_phase2_baseline.py` (`aggregate_seed_runs`) e no arquivo local (reduzido de `~10,1 MB` para `87,448 bytes`).

## 4. Histórico Recente de Commits (`git log`)

| Hash | Data e Mensagem |
| :--- | :--- |
| `06d1249` | 2026-11-13 — feat(phase3): raise default ceiling to 1000 epochs, add --extend-epochs and flush logs |
| `0d64b81` | 2026-11-13 — fix(phase3): preserve YAML config in CLI, harden full gates, reuse ESM cache and test seeds |
| `1ae2853` | 2026-11-13 — feat(phase3): implement PhysDualPhase3 (GIN + ESM-2 + MLP), training orchestrator and gate verifier |
| `8f6f9ad` | 2026-11-13 — docs(paper): documentar sintese critica dos artigos MAFI-DTA, CIP e TUNA |
| `28f32a5` | 2026-11-13 — feat(phase3): add Davis seqid30/40 kfold splits and CPU ESM-2 baseline harness |
| `23927b2` | 2026-11-13 — docs(phase3): formalize Phase 3 plan, ADR #14 and isolate Phase 1/2 paper splits |
| `3f05833` | 2026-11-13 — docs(paper): consolidar fechamento com auditoria ESM-2 Drive all_checks_passed=true |
| `1b82cfb` | 2026-11-13 — docs(paper): registrar aprovacao completa da auditoria ESM-2 no inventário |
| `4a12e48` | 2026-11-13 — docs(paper): atualizar inventario e reproducao apos endurecimento do auditor ESM-2 |
| `bd5feaf` | 2026-11-13 — feat(paper): validar fp16, finitude NaN/Inf, dim 1280 e seq_len vs CSV no auditor ESM-2 |
| `0db79f8` | 2026-11-13 — docs(paper): registrar auditoria do cache ESM-2 no Drive (429/429 validos, 83/83 windowed) |
| `c8ddd87` | 2026-11-13 — fix(paper): corrigir auditor ESM-2 para subpasta do modelo e contagem global 83 |
| `64cae10` | 2026-11-13 — docs(paper): consolidar pacote cientifico rastreavel das Fases 1 e 2 (T1-T8, F1-F6 e auditorias CPU) |
| `f3c0b0e` | 2026-11-13 — feat(paper): adicionar auditorias em CPU, scripts do pacote cientifico e validacoes pytest |
| `1eb1be4` | 2026-11-13 — docs(phase2): sincronizar FASE2_BASELINE e ADRs #12/#13 com auditoria CPU e teto efetivo 1000 |
