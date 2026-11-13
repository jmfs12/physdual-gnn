# Passo 0 — Inventário e Auditoria de Proveniência (Fases 1 e 2)

- **Gerado automaticamente por:** `scripts/paper/build_inventory_and_repro.py`
- **Commit HEAD do repositório:** `c8ddd87`
- **Versões de pipeline identificadas:**
  - **Featurizador molecular:** `['v2_onehot78_l1norm']` (`configs/base.yaml`: `v2_onehot78_l1norm`)
  - **Cache ESM-2 configurado:** `esm2_t33_650M_UR50D` (`esm_dim=1280`, `max_len=1022`, `overlap=256`, `fp16_inference=True`)
  - **Protocolo v1 (`v1_seeds`):** 5 seeds (`0..4`) em splits 80/10/10 (`random`, `cold_drug`, `cold_target`) — 45 runs presentes (`30` Davis + `15` KIBA `graphdta_gin`).
  - **Protocolo v2 (`v2_kfold`):** **EXECUTADO** para Davis Cold (`cold_drug_kfold`, `cold_target_kfold`) com 5-Fold `GroupKFold` (`assignment_seed=42`, agrupado por `smiles` e `sequence`) — 20 runs presentes (`10` `graphdta_gin` + `10` `graphdta_cnn3`), 4 agregados `*_kfold.json` e 4 CSVs OOF (`30.056` pares cada). **Nunca misturado com v1 nas mesmas médias ou figuras.**

## 1. Inventário Geral de Artefatos

| Artefato | Fase | Protocolo / Versão | Commit | Status |
| :--- | :--- | :--- | :--- | :--- |
| `configs/base.yaml` | Fase 0/1/2 | featurizer=v2_onehot78_l1norm, esm=esm2_t33_650M_UR50D (1280d) | `5ae3e12` | completo |
| `docs/DECISOES.md` (ADRs #1–#13) | Fase 0/1/2 | ADRs #1–#13 | `1eb1be4` | completo |
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
| `cache/esm_embeddings/` | Fase 1 | `esm2_t33_650M_UR50D`, `esm_dim=1280` | `gitignored` | ausente localmente (auditado no Google Drive: 429/429 .pt [len(seq), 1280] e 83/83 windowed íntegros; 1 resíduo smoke_test na raiz) |
| `scripts/paper/audit_esm_cache_colab.py` | Fase 1/2 (auditoria) | Colab / Drive (`esm2_t33_650M_UR50D/*.pt`, `windowed=83` global / `112` por dataset, `[len(seq), 1280]`) | `c8ddd87` | executado no Drive (429/429 válidos; script corrigido para isolar resíduo smoke_test) |
| `scripts/paper/cpu_audits.py` + `docs/paper/cpu_audits.json` | Fase 1/2 (auditoria CPU) | Noise floor, `pKd>5`, marginais/kNN `v1`, timing limpo, seq-id Davis | `64cae10` | completo (recomputado em CPU a partir de CSVs + splits) |
| `outputs/logs/phase1_summary_report.json` | Fase 1 | datasets=['davis', 'kiba'], `windowed_sequences=112` | `gitignored (local)` | completo (davis, kiba; janelas=112) |
| `outputs/results/phase2_timing_benchmark.json` | Fase 2 | NVIDIA L4, 3 épocas | `gitignored (local)` | completo |
| `outputs/results/baseline_graphdta_cnn3_davis_cold_drug.json` | Fase 2 | v1_seeds | `gitignored (local)` | completo (5 unidades; marginais/kNN em `cpu_audits.json`) |
| `outputs/results/baseline_graphdta_cnn3_davis_cold_drug_kfold.json` | Fase 2 | v2_kfold | `gitignored (local)` | completo (5 unidades) |
| `outputs/results/baseline_graphdta_cnn3_davis_cold_target.json` | Fase 2 | v1_seeds | `gitignored (local)` | completo (5 unidades; marginais/kNN em `cpu_audits.json`) |
| `outputs/results/baseline_graphdta_cnn3_davis_cold_target_kfold.json` | Fase 2 | v2_kfold | `gitignored (local)` | completo (5 unidades) |
| `outputs/results/baseline_graphdta_cnn3_davis_random.json` | Fase 2 | v1_seeds | `gitignored (local)` | completo (5 unidades; marginais/kNN em `cpu_audits.json`) |
| `outputs/results/baseline_graphdta_gin_davis_cold_drug.json` | Fase 2 | v1_seeds | `gitignored (local)` | completo (5 unidades; marginais/kNN em `cpu_audits.json`) |
| `outputs/results/baseline_graphdta_gin_davis_cold_drug_kfold.json` | Fase 2 | v2_kfold | `gitignored (local)` | completo (5 unidades) |
| `outputs/results/baseline_graphdta_gin_davis_cold_target.json` | Fase 2 | v1_seeds | `gitignored (local)` | completo (5 unidades; marginais/kNN em `cpu_audits.json`) |
| `outputs/results/baseline_graphdta_gin_davis_cold_target_kfold.json` | Fase 2 | v2_kfold | `gitignored (local)` | completo (5 unidades) |
| `outputs/results/baseline_graphdta_gin_davis_random.json` | Fase 2 | v1_seeds | `gitignored (local)` | completo (5 unidades; marginais/kNN em `cpu_audits.json`) |
| `outputs/results/baseline_graphdta_gin_kiba_cold_drug.json` | Fase 2 | v1_seeds | `gitignored (local)` | completo (5 unidades; marginais/kNN em `cpu_audits.json`) |
| `outputs/results/baseline_graphdta_gin_kiba_cold_target.json` | Fase 2 | v1_seeds | `gitignored (local)` | completo (5 unidades; marginais/kNN em `cpu_audits.json`) |
| `outputs/results/baseline_graphdta_gin_kiba_random.json` | Fase 2 | v1_seeds | `gitignored (local)` | completo (5 unidades) |
| `outputs/results/oof/graphdta_cnn3_davis_cold_drug_oof.csv` | Fase 2 | v2_kfold | `gitignored (local)` | completo (30056 pares OOF) |
| `outputs/results/oof/graphdta_cnn3_davis_cold_target_oof.csv` | Fase 2 | v2_kfold | `gitignored (local)` | completo (30056 pares OOF) |
| `outputs/results/oof/graphdta_gin_davis_cold_drug_oof.csv` | Fase 2 | v2_kfold | `gitignored (local)` | completo (30056 pares OOF) |
| `outputs/results/oof/graphdta_gin_davis_cold_target_oof.csv` | Fase 2 | v2_kfold | `gitignored (local)` | completo (30056 pares OOF) |
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
| `v2_kfold` | `graphdta_gin` | `davis` | `cold_drug` | 5 | sim | Com `history`, `marginal`, `knn` e OOF |
| `v2_kfold` | `graphdta_gin` | `davis` | `cold_target` | 5 | sim | Com `history`, `marginal`, `knn` e OOF |
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
   - *Estado atual nos JSONs:* após atualização em `scripts/run_phase2_baseline.py` e nos artefatos locais, a chave `test_isolation_verified: true` consta em todos os runs e agregados (`missing_runs=0/65`).
5. **Divergências de `epochs` vs. `configs/base.yaml` (`epochs=500`) e paridade em `kiba_random`:**
   - `graphdta_gin_kiba_random_seed1.json`: `epochs=1000` != `configs/base.yaml` (`500`)
   - `graphdta_gin_kiba_random_seed3.json`: `epochs=1000` != `configs/base.yaml` (`500`)
   - Como as seeds `0, 2, 4` encerraram por *early stopping* (`patience=50`) em `total_epochs_run = 344, 336, 393` (< `500`), **não** precisam ser re-executadas: o teto efetivo é **`1000` épocas para todas as 5 seeds de `kiba_random`**.
   - As seeds `1` e `3` retomadas possuem `history` restrito às épocas pós-500 (`501..824` e `501..559`) e `sec_per_epoch` distorcido (`0.646 s` e `2.347 s`) por dividirem o tempo da retomada pelo total acumulado. Em `T8`, a velocidade limpa de `kiba_random` é recalculada exclusivamente nos 3 runs não retomados (`seeds 0, 2, 4`: **`22.338 ± 0.014 s/época`**).
6. **Recomputação determinística em CPU (`docs/paper/cpu_audits.json`):**
   - Os 43 runs `v1_seeds` originais não gravaram `marginal_baselines`, `knn_baselines` nem `history` em disco, e os JSONs originais não salvavam `mean_predictor` restrito ao subconjunto `pKd > 5.0`. Por via puramente determinística a partir dos CSVs limpos e arquivos de split (sem necessidade de checkpoints GPU), `scripts/paper/cpu_audits.py` recalculou todos os marginais, `kNN`, deltas pareados por seed para os 45 runs `v1_seeds`, baselines em `pKd > 5.0` (`T5`) e piso de ruído de duplicatas (`T1`).
   - **Única lacuna não recomputável localmente:** curvas por época (`history`) dos 43 runs `v1_seeds` originais. Já a auditoria física dos `.pt` no Google Drive (`/content/drive/MyDrive/physdual-gnn/cache/esm_embeddings`) foi executada: encontrou `430` arquivos no total (`429` sequências únicas esperadas + `1` resíduo `4637bffbde58d549.pt` com shape `[65, 320]` e sem campo `windowed` gerado por `scripts/smoke_test.py` via `esm2_t6_8M_UR50D` na raiz do cache), `83/83` sequências únicas `> 1022 aa` com `windowed=True` (`70 + 42 - 29` compartilhadas = `83`, contra `112` por soma de dataset) e `0` erros entre os `429` tensores `[len(seq), 1280]` da Fase 1.
7. **Limpeza de inchaço em `baseline_graphdta_gin_kiba_random.json`:**
   - Removido `oof_predictions` de `per_seed_results` em `scripts/run_phase2_baseline.py` (`aggregate_seed_runs`) e no arquivo local (reduzido de `~10,1 MB` para `87,448 bytes`).

## 4. Histórico Recente de Commits (`git log`)

| Hash | Data e Mensagem |
| :--- | :--- |
| `c8ddd87` | 2026-11-13 — fix(paper): corrigir auditor ESM-2 para subpasta do modelo e contagem global 83 |
| `64cae10` | 2026-11-13 — docs(paper): consolidar pacote cientifico rastreavel das Fases 1 e 2 (T1-T8, F1-F6 e auditorias CPU) |
| `f3c0b0e` | 2026-11-13 — feat(paper): adicionar auditorias em CPU, scripts do pacote cientifico e validacoes pytest |
| `1eb1be4` | 2026-11-13 — docs(phase2): sincronizar FASE2_BASELINE e ADRs #12/#13 com auditoria CPU e teto efetivo 1000 |
| `c2b7afe` | 2026-11-13 — fix(phase1,phase2): corrigir sumario fase 1, remover oof de agregados e registrar test_isolation_verified |
| `09a9ac5` | 2026-11-13 — perf(phase2): acelerar bootstrap por piso yt_min/T compartilhado, enxugar JSON agregado e corrigir parser --aggregate-only |
| `2b9bacd` | 2026-11-13 — perf(metrics): otimizar bootstrap_grouped_comparison com precomputacao matricial exata (100x mais rapido) |
| `ceda909` | 2026-11-13 — fix(phase2): corrigir valor padrao de runs_dir, recálculo por dobra e caminho PHYSDUAL_ROOT no Passo 11.4 |
| `ffea4c5` | 2026-11-13 — fix(phase2): auto-recuperar sequencia/smiles em OOF e reparar bootstrap cluster-level |
| `b8fffcf` | 2026-11-12 — refactor(metrics): migrar AllChem.GetMorganFingerprintAsBitVect para rdFingerprintGenerator.GetMorganGenerator |
| `6e8f26f` | 2026-11-12 — fix(splits,notebook): aceitar group_col e random_state em create_group_kfold_splits e tornar setup de paths robusto |
| `172b7a1` | 2026-11-12 — feat(phase2): implementar GroupKFold 5-fold Davis, baselines marginais/kNN, OOF e extensão KIBA (Opção C, ADR #13) |
| `4743e01` | 2026-11-10 — fix(metrics): guard pearson and spearman against constant inputs within numerical tolerance |
| `29f0977` | 2026-11-10 — docs(plan): mark Task 6 as completed in Phase 2 implementation plan |
| `28cfedc` | 2026-11-10 — docs(phase2): document verified GraphDTA tables, ADR #10-#12, and duplicate/mutation analysis |
