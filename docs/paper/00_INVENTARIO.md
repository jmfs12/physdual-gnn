# Passo 0 — Inventário e Auditoria de Proveniência (Fases 1, 2 e 3)

- **Gerado automaticamente por:** `scripts/paper/build_inventory_and_repro.py`
- **Commit HEAD do repositório:** `461421b`
- **Auditoria detalhada da Fase 3:** ver [`docs/paper/06_AUDITORIA_FASE3.md`](06_AUDITORIA_FASE3.md) e [`docs/paper/phase3_audits.json`](phase3_audits.json).
- **Versões de pipeline identificadas:**
  - **Featurizador molecular:** `['v2_onehot78_l1norm']` (`configs/base.yaml`: `v2_onehot78_l1norm`)
  - **Cache ESM-2:** `esm2_t33_650M_UR50D` (`esm_dim=1280`, `max_len=1022`, `overlap=256`, `fp16_inference=True` — `429` arquivos `.pt` locais)
  - **Protocolo v1 (`v1_seeds`):** 5 seeds (`0..4`) em splits 80/10/10 (`random`, `cold_drug`, `cold_target`) — `45` runs GraphDTA + `20` runs Fase 3 (`kiba` `cold_drug`/`cold_target`/`random` e `davis` `random`).
  - **Protocolo v2 (`v2_kfold`):** `GroupKFold` 5 dobras (`assignment_seed=42`) em `cold_drug`, `cold_target`, `cold_target_seqid40` e `cold_target_seqid30` — `40` runs GraphDTA + `61` runs Fase 3 (incl. controles diagnósticos). **Nunca misturado com `v1_seeds` nas mesmas médias ou figuras.**

## 1. Inventário Geral de Artefatos

| Artefato | Fase | Protocolo / Versão | Commit | Status |
| :--- | :--- | :--- | :--- | :--- |
| `configs/base.yaml` | Fase 0–3 | featurizer=v2_onehot78_l1norm, esm=esm2_t33_650M_UR50D (1280d) | `06d1249` | completo (congelado ADR #15 em `46d13f8`) |
| `docs/DECISOES.md` (ADRs #1–#15) | Fase 0–3 | ADRs #1–#15 (`phase3-adr15-pre-kiba`) | `46d13f8` | completo |
| `docs/FASE1_DADOS.md` | Fase 1 | v1_seeds | `6efe404` | completo |
| `docs/FASE2_BASELINE.md` | Fase 2 | v1_seeds + v2_kfold | `1eb1be4` | completo |
| `docs/FASE3_PHYSDUAL.md` | Fase 3 | ADR #15 & protocolos | `46d13f8` | completo |
| `docs/colab_env_snapshot.txt` | Fase 0–3 | Colab NVIDIA L4 (PyTorch 2.14.1+cu130) | `11d9fca` | completo |
| `notebooks/00..02_*.ipynb` (4 arquivos) | Fase 0/1/2 | Colab / Drive | `09a9ac5` | completo |
| `data/raw/davis/{ligands_can.txt,proteins.txt,Y}` | Fase 1 | DeepDTA `a546a84` | `gitignored (local)` | completo |
| `data/raw/kiba/{ligands_can.txt,proteins.txt,Y}` | Fase 1 | DeepDTA `a546a84` | `gitignored (local)` | completo |
| `data/processed/davis_clean.csv` + `davis_ingestion_report.json` | Fase 1 | 30.056 pares | `gitignored (local)` | completo |
| `data/processed/kiba_clean.csv` + `kiba_ingestion_report.json` | Fase 1 | 118.254 pares | `gitignored (local)` | completo |
| `data/processed/{davis,kiba}_drug_graphs_v2_onehot78_l1norm.pt` | Fase 1–3 | `v2_onehot78_l1norm` | `gitignored (local)` | completo (`78d` em disco) |
| `data/processed/{davis,kiba}_drug_graphs.pt` | Fase 1 | legado / cópia padrão | `gitignored (local)` | completo |
| `data/processed/splits/*.json` (60 arquivos) | Fase 1–3 | 30 `v1_seeds` + 10 Davis `v2_kfold` + 10 Davis `seqid` + 10 KIBA `seqid` | `gitignored (local)` | completo (zero vazamento por conteúdo) |
| `cache/esm_embeddings/` | Fase 1–3 | `esm2_t33_650M_UR50D`, `esm_dim=1280` | `gitignored` | completo (429 `.pt` locais + Drive validado: 429/429 .pt fp16, dim=1280) |
| `scripts/paper/audit_esm_cache_colab.py` | Fase 1/2 (auditoria) | Colab / Drive (`esm2_t33_650M_UR50D/*.pt`, `fp16`, `NaN/Inf=0`, `dim=1280`) | `bd5feaf` | executado no Drive (`all_checks_passed: true`, 429/429 válidos, 0 resíduos) |
| `scripts/paper/cpu_audits.py` + `docs/paper/cpu_audits.json` | Fase 1/2 (auditoria CPU) | Noise floor, `pKd>5`, marginais/kNN `v1`, timing limpo, seq-id Davis | `64cae10` | completo (recomputado em CPU) |
| `scripts/paper/phase3_audits.py` + `docs/paper/phase3_audits.json` | Fase 3 (auditoria CPU) | Pré-registro Git, integridade 166 runs, OOFs, KIBA v1/v2 & bootstrap `n=1000` | `d6d225f` | completo (recomputado em CPU a partir de cache `.pt` + CSVs) |
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
| `outputs/results/phase3_physdual_ecfp_esm_mlp_davis_cold_drug_kfold.json` | Fase 3 | v2_kfold | `gitignored (local)` | completo (5 unidades) |
| `outputs/results/phase3_physdual_ecfp_esm_mlp_davis_cold_target_kfold.json` | Fase 3 | v2_kfold | `gitignored (local)` | completo (5 unidades) |
| `outputs/results/phase3_physdual_gin_esm_mlp_davis_cold_drug_kfold.json` | Fase 3 | v2_kfold | `gitignored (local)` | completo (5 unidades) |
| `outputs/results/phase3_physdual_gin_esm_mlp_davis_cold_target_kfold.json` | Fase 3 | v2_kfold | `gitignored (local)` | completo (5 unidades) |
| `outputs/results/phase3_physdual_gin_esm_mlp_davis_cold_target_seqid30_kfold.json` | Fase 3 | v2_kfold | `gitignored (local)` | completo (5 unidades) |
| `outputs/results/phase3_physdual_gin_esm_mlp_davis_cold_target_seqid40_kfold.json` | Fase 3 | v2_kfold | `gitignored (local)` | completo (5 unidades) |
| `outputs/results/phase3_physdual_gin_esm_mlp_davis_random.json` | Fase 3 | v1_seeds | `gitignored (local)` | completo (5 unidades) |
| `outputs/results/phase3_physdual_gin_esm_mlp_kiba_cold_drug.json` | Fase 3 | v1_seeds | `gitignored (local)` | completo (5 unidades) |
| `outputs/results/phase3_physdual_gin_esm_mlp_kiba_cold_target.json` | Fase 3 | v1_seeds | `gitignored (local)` | completo (5 unidades) |
| `outputs/results/phase3_physdual_gin_esm_mlp_kiba_cold_target_seqid30_kfold.json` | Fase 3 | v2_kfold | `gitignored (local)` | completo (5 unidades) |
| `outputs/results/phase3_physdual_gin_esm_mlp_kiba_cold_target_seqid40_kfold.json` | Fase 3 | v2_kfold | `gitignored (local)` | completo (5 unidades) |
| `outputs/results/phase3_physdual_gin_esm_mlp_kiba_random.json` | Fase 3 | v1_seeds | `gitignored (local)` | completo (5 unidades) |
| `outputs/results/phase3_physdual_gin_esm_mlp_shuf0_davis_cold_target_kfold.json` | Fase 3 | v2_kfold | `gitignored (local)` | completo (5 unidades) |
| `outputs/results/phase3_physdual_gin_esm_mlp_shuf1_davis_cold_target_kfold.json` | Fase 3 | v2_kfold | `gitignored (local)` | completo (5 unidades) |
| `outputs/results/phase3_physdual_gin_esm_mlp_shuf2_davis_cold_target_kfold.json` | Fase 3 | v2_kfold | `gitignored (local)` | completo (5 unidades) |
| `outputs/results/phase3_physdual_gin_esm_mlp_tsoff1000_davis_cold_drug_kfold.json` | Fase 3 | v2_kfold | `gitignored (local)` | completo (1 unidades) |
| `outputs/results/phase3_physdual_gin_esm_mlp_tsoff1000_davis_cold_target_seqid30_kfold.json` | Fase 3 | v2_kfold | `gitignored (local)` | completo (5 unidades) |
| `outputs/results/phase3_esm_cpu_baselines.json` | Fase 3 | auditoria / CPU | `gitignored (local)` | completo |
| `outputs/results/kiba_cpu_classical_baselines.json` | Fase 3 | auditoria / CPU | `gitignored (local)` | completo |
| `outputs/results/phase3_seqid_classical_baselines.json` | Fase 3 | auditoria / CPU | `gitignored (local)` | completo |
| `outputs/results/phase3_kiba_paired_bootstrap.json` | Fase 3 | auditoria / CPU | `gitignored (local)` | completo |
| `outputs/results/phase3_davis_paired_bootstrap.json` | Fase 3 | auditoria / CPU | `gitignored (local)` | DEFASADO (pré-extensão fold 1 `davis/cold_target`; ver `06_AUDITORIA_FASE3.md`) |
| `outputs/results/oof/cpu_baselines_davis_cold_drug_oof.csv` | Fase 3 | v2_kfold | `gitignored (local)` | completo (30056 pares OOF) |
| `outputs/results/oof/cpu_baselines_davis_cold_target_oof.csv` | Fase 3 | v2_kfold | `gitignored (local)` | completo (30056 pares OOF) |
| `outputs/results/oof/cpu_baselines_davis_cold_target_seqid30_oof.csv` | Fase 3 | v2_kfold | `gitignored (local)` | completo (30056 pares OOF) |
| `outputs/results/oof/cpu_baselines_davis_cold_target_seqid40_oof.csv` | Fase 3 | v2_kfold | `gitignored (local)` | completo (30056 pares OOF) |
| `outputs/results/oof/cpu_baselines_kiba_cold_target_seqid30_oof.csv` | Fase 3 | v2_kfold | `gitignored (local)` | completo (118254 pares OOF) |
| `outputs/results/oof/cpu_baselines_kiba_cold_target_seqid40_oof.csv` | Fase 3 | v2_kfold | `gitignored (local)` | completo (118254 pares OOF) |
| `outputs/results/oof/graphdta_cnn3_davis_cold_drug_oof.csv` | Fase 2 | v2_kfold | `gitignored (local)` | completo (30056 pares OOF) |
| `outputs/results/oof/graphdta_cnn3_davis_cold_target_oof.csv` | Fase 2 | v2_kfold | `gitignored (local)` | completo (30056 pares OOF) |
| `outputs/results/oof/graphdta_cnn3_davis_cold_target_seqid30_oof.csv` | Fase 2 | v2_kfold | `gitignored (local)` | completo (30056 pares OOF) |
| `outputs/results/oof/graphdta_cnn3_davis_cold_target_seqid40_oof.csv` | Fase 2 | v2_kfold | `gitignored (local)` | completo (30056 pares OOF) |
| `outputs/results/oof/graphdta_gin_davis_cold_drug_oof.csv` | Fase 2 | v2_kfold | `gitignored (local)` | completo (30056 pares OOF) |
| `outputs/results/oof/graphdta_gin_davis_cold_target_oof.csv` | Fase 2 | v2_kfold | `gitignored (local)` | completo (30056 pares OOF) |
| `outputs/results/oof/graphdta_gin_davis_cold_target_seqid30_oof.csv` | Fase 2 | v2_kfold | `gitignored (local)` | completo (30056 pares OOF) |
| `outputs/results/oof/graphdta_gin_davis_cold_target_seqid40_oof.csv` | Fase 2 | v2_kfold | `gitignored (local)` | completo (30056 pares OOF) |
| `outputs/results/oof/physdual_ecfp_esm_mlp_davis_cold_drug_oof.csv` | Fase 3 | v2_kfold | `gitignored (local)` | completo (30056 pares OOF) |
| `outputs/results/oof/physdual_ecfp_esm_mlp_davis_cold_target_oof.csv` | Fase 3 | v2_kfold | `gitignored (local)` | completo (30056 pares OOF) |
| `outputs/results/oof/physdual_gin_esm_mlp_davis_cold_drug_oof.csv` | Fase 3 | v2_kfold | `gitignored (local)` | completo (30056 pares OOF) |
| `outputs/results/oof/physdual_gin_esm_mlp_davis_cold_target_oof.csv` | Fase 3 | v2_kfold | `gitignored (local)` | completo (30056 pares OOF) |
| `outputs/results/oof/physdual_gin_esm_mlp_davis_cold_target_seqid30_oof.csv` | Fase 3 | v2_kfold | `gitignored (local)` | completo (30056 pares OOF) |
| `outputs/results/oof/physdual_gin_esm_mlp_davis_cold_target_seqid40_oof.csv` | Fase 3 | v2_kfold | `gitignored (local)` | completo (30056 pares OOF) |
| `outputs/results/oof/physdual_gin_esm_mlp_kiba_cold_target_seqid30_oof.csv` | Fase 3 | v2_kfold | `gitignored (local)` | completo (118254 pares OOF) |
| `outputs/results/oof/physdual_gin_esm_mlp_kiba_cold_target_seqid40_oof.csv` | Fase 3 | v2_kfold | `gitignored (local)` | completo (118254 pares OOF) |
| `outputs/results/oof/physdual_gin_esm_mlp_shuf0_davis_cold_target_oof.csv` | Fase 3 | v2_kfold | `gitignored (local)` | completo (30056 pares OOF) |
| `outputs/results/oof/physdual_gin_esm_mlp_shuf1_davis_cold_target_oof.csv` | Fase 3 | v2_kfold | `gitignored (local)` | completo (30056 pares OOF) |
| `outputs/results/oof/physdual_gin_esm_mlp_shuf2_davis_cold_target_oof.csv` | Fase 3 | v2_kfold | `gitignored (local)` | completo (30056 pares OOF) |
| `outputs/results/oof/physdual_gin_esm_mlp_tsoff1000_davis_cold_drug_oof.csv` | Fase 3 | v2_kfold | `gitignored (local)` | completo (6188 pares OOF) |
| `outputs/results/oof/physdual_gin_esm_mlp_tsoff1000_davis_cold_target_seqid30_oof.csv` | Fase 3 | v2_kfold | `gitignored (local)` | completo (30056 pares OOF) |
| `outputs/results/runs/graphdta_*` (85 runs) | Fase 2 | v1_seeds + v2_kfold (`GroupKFold` + `seqid`) | `gitignored (local)` | completo (`5` folds/seeds por grupo) |
| `outputs/results/runs/physdual_*` (81 runs) | Fase 3 | v1_seeds + v2_kfold + seqid + controles (`19` pré-ADR #15 / `62` pós-ADR #15) | `gitignored (local)` | completo (`81/81` `test_isolation_verified=True`, `0` truncados) |
| `outputs/results/runs/graphdta_cnn3_kiba_*.json` | Fase 2/3 | v1_seeds / seqid | `—` | ausente (omitido pelo critério de orçamento `>15 s/época` no ADR #12) |

## 2. Cobertura por Experimento (`outputs/results/runs/`)

| Protocolo | Modelo / Controle | Dataset | Split | Runs | Schema ADR #15 | Observações |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| `v1_seeds` | `graphdta_cnn3` | `davis` | `cold_drug` | 5 | `legado F2` | Normal |
| `v1_seeds` | `graphdta_cnn3` | `davis` | `cold_target` | 5 | `legado F2` | Normal |
| `v1_seeds` | `graphdta_cnn3` | `davis` | `random` | 5 | `legado F2` | Normal |
| `v1_seeds` | `graphdta_gin` | `davis` | `cold_drug` | 5 | `legado F2` | Normal |
| `v1_seeds` | `graphdta_gin` | `davis` | `cold_target` | 5 | `legado F2` | Normal |
| `v1_seeds` | `graphdta_gin` | `davis` | `random` | 5 | `legado F2` | Normal |
| `v1_seeds` | `graphdta_gin` | `kiba` | `cold_drug` | 5 | `legado F2` | Normal |
| `v1_seeds` | `graphdta_gin` | `kiba` | `cold_target` | 5 | `legado F2` | Normal |
| `v1_seeds` | `graphdta_gin` | `kiba` | `random` | 5 | `legado F2` | Teto efetivo `1000` épocas (seeds 0,2,4 pararam <500; 1,3 estendidas) |
| `v1_seeds` | `physdual_gin_esm_mlp` | `davis` | `random` | 5 | `5/5` | Normal |
| `v1_seeds` | `physdual_gin_esm_mlp` | `kiba` | `cold_drug` | 5 | `5/5` | Normal |
| `v1_seeds` | `physdual_gin_esm_mlp` | `kiba` | `cold_target` | 5 | `5/5` | Normal |
| `v1_seeds` | `physdual_gin_esm_mlp` | `kiba` | `random` | 5 | `5/5` | Normal |
| `v2_kfold` | `graphdta_cnn3` | `davis` | `cold_drug` | 5 | `legado F2` | Normal |
| `v2_kfold` | `graphdta_cnn3` | `davis` | `cold_target` | 5 | `legado F2` | Normal |
| `v2_kfold` | `graphdta_cnn3` | `davis` | `cold_target_seqid30` | 5 | `legado F2` | Normal |
| `v2_kfold` | `graphdta_cnn3` | `davis` | `cold_target_seqid40` | 5 | `legado F2` | Normal |
| `v2_kfold` | `graphdta_gin` | `davis` | `cold_drug` | 5 | `legado F2` | Normal |
| `v2_kfold` | `graphdta_gin` | `davis` | `cold_target` | 5 | `legado F2` | Normal |
| `v2_kfold` | `graphdta_gin` | `davis` | `cold_target_seqid30` | 5 | `legado F2` | Normal |
| `v2_kfold` | `graphdta_gin` | `davis` | `cold_target_seqid40` | 5 | `legado F2` | Normal |
| `v2_kfold` | `physdual_ecfp_esm_mlp` | `davis` | `cold_drug` | 5 | `5/5` | Controle diagnóstico (ver `T14`) |
| `v2_kfold` | `physdual_ecfp_esm_mlp` | `davis` | `cold_target` | 5 | `5/5` | Controle diagnóstico (ver `T14`) |
| `v2_kfold` | `physdual_gin_esm_mlp` | `davis` | `cold_drug` | 5 | `0/5` | Normal |
| `v2_kfold` | `physdual_gin_esm_mlp` | `davis` | `cold_target` | 4 | `0/4` | 4 dobras `epochs=500` pré-`06d1249`; dobra 1 estendida `525/605/1000` |
| `v2_kfold` | `physdual_gin_esm_mlp` | `davis` | `cold_target` | 1 | `1/1` | 4 dobras `epochs=500` pré-`06d1249`; dobra 1 estendida `525/605/1000` |
| `v2_kfold` | `physdual_gin_esm_mlp` | `davis` | `cold_target_seqid30` | 5 | `0/5` | Normal |
| `v2_kfold` | `physdual_gin_esm_mlp` | `davis` | `cold_target_seqid40` | 5 | `0/5` | Normal |
| `v2_kfold` | `physdual_gin_esm_mlp` | `kiba` | `cold_target_seqid30` | 5 | `5/5` | Normal |
| `v2_kfold` | `physdual_gin_esm_mlp` | `kiba` | `cold_target_seqid40` | 5 | `5/5` | Normal |
| `v2_kfold` | `physdual_gin_esm_mlp_shuf0` | `davis` | `cold_target` | 5 | `5/5` | Controle diagnóstico (ver `T14`) |
| `v2_kfold` | `physdual_gin_esm_mlp_shuf1` | `davis` | `cold_target` | 5 | `5/5` | Controle diagnóstico (ver `T14`) |
| `v2_kfold` | `physdual_gin_esm_mlp_shuf2` | `davis` | `cold_target` | 5 | `5/5` | Controle diagnóstico (ver `T14`) |
| `v2_kfold` | `physdual_gin_esm_mlp_tsoff1000` | `davis` | `cold_drug` | 1 | `1/1` | Controle diagnóstico (ver `T14`) |
| `v2_kfold` | `physdual_gin_esm_mlp_tsoff1000` | `davis` | `cold_target_seqid30` | 5 | `5/5` | Controle diagnóstico (ver `T14`) |
| `v1_seeds` | `graphdta_cnn3` | `kiba` | `random / cold_drug / cold_target` | 0 | — | **Adiado / ausente** (custo `20.544 s/época` > teto `15 s/época`) |

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
   - *Estado atual nos JSONs:* após atualização em `scripts/run_phase2_baseline.py` e nos artefatos locais, a chave `test_isolation_verified: true` consta em todos os runs e agregados (`missing_runs=65/166`).
5. **Divergências de `epochs` vs. `configs/base.yaml` (`500` GraphDTA / `1000` Fase 3) e paridade em `kiba_random`:**
   - `graphdta_gin_kiba_random_seed1.json`: `epochs=1000` != `configs/base.yaml` (`500`)
   - `graphdta_gin_kiba_random_seed3.json`: `epochs=1000` != `configs/base.yaml` (`500`)
   - `physdual_gin_esm_mlp_davis_cold_target_fold0.json`: `epochs=500` != `configs/base.yaml` (`1000`)
   - `physdual_gin_esm_mlp_davis_cold_target_fold2.json`: `epochs=500` != `configs/base.yaml` (`1000`)
   - `physdual_gin_esm_mlp_davis_cold_target_fold3.json`: `epochs=500` != `configs/base.yaml` (`1000`)
   - `physdual_gin_esm_mlp_davis_cold_target_fold4.json`: `epochs=500` != `configs/base.yaml` (`1000`)
   - No `graphdta_gin` (`kiba_random`), as seeds `0, 2, 4` encerraram por *early stopping* (`patience=50`) em `total_epochs_run = 344, 336, 393` (< `500`), **não** requerendo re-execução: o teto efetivo é **`1000` épocas para todas as 5 seeds de `kiba_random`**.
   - Na Fase 3, as `4` dobras não estendidas de `davis/cold_target` (`0, 2, 3, 4`) mantiveram `epochs=500` pré-`06d1249` (parando entre `225` e `368` épocas), enquanto a dobra `1` foi estendida para `1000` (`best_epoch=525`, `total_epochs_run=605`). Todos os `62/62` runs pós-ADR #15 possuem `0` desvios de configuração.
   - As seeds `1` e `3` retomadas do `graphdta_gin` (`kiba_random`) e a dobra `1` estendida do `davis/cold_target` (`gin_esm_mlp`) possuem `sec_per_epoch` subestimado (`0.646 s`, `2.347 s` e `0.260 s`) por dividirem o tempo da retomada pelo total acumulado de épocas. Em `T8`, a velocidade limpa de `kiba_random` é recalculada exclusivamente nos 3 runs não retomados (`seeds 0, 2, 4`: **`22.338 ± 0.014 s/época`**), e em `T15` a dobra `1` é excluída da média limpa de velocidade do `davis/cold_target`.
6. **Recomputação determinística em CPU (`docs/paper/cpu_audits.json` e `docs/paper/phase3_audits.json`):**
   - Os 43 runs `v1_seeds` originais da Fase 2 não gravaram `marginal_baselines`, `knn_baselines` nem `history` em disco, e os JSONs originais não salvavam `mean_predictor` restrito ao subconjunto `pKd > 5.0`. `scripts/paper/cpu_audits.py` recalculou todos os marginais, `kNN`, deltas pareados e piso de ruído para as Fases 1–2.
   - Na Fase 3, `scripts/paper/phase3_audits.py` auditou todos os `166` runs, reproduziu baselines/OOFs dos `27` CSVs e das `20` seeds `v1` em CPU usando os `429` `.pt` locais do ESM-2, e executou bootstrap pareado por cluster (`n=1000`, `seed=42`) em `docs/paper/phase3_audits.json`.
7. **Limpeza de inchaço e artefato secundário defasado (`outputs/results/`):**
   - Removido `oof_predictions` de `baseline_graphdta_gin_kiba_random.json` (reduzido para `87,448 bytes`).
   - **Artefato defasado reportado (sem alteração silenciosa de `outputs/`):** `outputs/results/phase3_davis_paired_bootstrap.json` (`mtime 2026-11-14 00:13:05`) antecede a extensão da dobra `1` de `davis/cold_target`; todas as análises e tabelas deste pacote usam os valores recomputados diretamente do OOF vigente em `docs/paper/phase3_audits.json` (comando proposto: `python scripts/run_phase3_paired_bootstrap.py --dataset davis`).

## 4. Histórico Recente de Commits (`git log`)

| Hash | Data e Mensagem |
| :--- | :--- |
| `461421b` | 2026-11-15 — docs(paper): update drafts 02-05 and pipeline tests with Phase 3 ADR #15 evaluation |
| `815e016` | 2026-11-15 — docs(paper): regenerate Phase 3 tables, figures, captions, inventory and reproduction guide |
| `06d2132` | 2026-11-15 — feat(paper): extend tables T7/T9-T15, figures F7-F11 and inventory/repro builders for Phase 3 |
| `d6d225f` | 2026-11-15 — feat(paper): add Phase 3 CPU integrity audit, bootstrap recomputation and 06_AUDITORIA_FASE3 |
| `936c32f` | 2026-11-14 — feat(phase3): add KIBA seqid OOF export and --dataset option in paired bootstrap |
| `9a60916` | 2026-11-14 — fix(phase3): support KIBA seqid30/40 baselines, Gate 6 dataset scoping and diagnostic --aggregate-only |
| `46d13f8` | 2026-11-14 — docs(phase3): ADR #15 exploratory reclassification, KIBA pre-registered criteria & diagnostic controls |
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
