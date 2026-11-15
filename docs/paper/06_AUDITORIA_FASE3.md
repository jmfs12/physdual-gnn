# Auditoria de Integridade da Fase 3 (`Passo 0`)

> **Escopo e regra de integridade:** este documento registra a auditoria forense **pré-análise** dos artefatos produzidos na Fase 3 (KIBA `v1_seeds` e `v2_kfold`, Davis `random`, extensão de `davis/cold_target` dobra 1 e controles diagnósticos no Davis), além da conferência retroativa dos runs herdados das Fases 1–2 (`graphdta_*`). Todos os números abaixo provêm exclusivamente de [`docs/paper/phase3_audits.json`](phase3_audits.json) (gerado em modo somente-leitura por `scripts/paper/phase3_audits.py`) e dos JSONs originais em `outputs/results/`. **Nenhum artefato de `outputs/`, `data/`, `cache/` ou `configs/` foi alterado.**

---

## 1. Inventário completo dos artefatos auditados

| Categoria | Localização | Contagem observada | Detalhes e provenance |
| :--- | :--- | ---: | :--- |
| Runs individuais Fase 3 (`physdual_*`) | `outputs/results/runs/physdual_*.json` | **81** | **19** pré-ADR #15 (`davis` cold sem campo `diagnostic_control`) + **62** pós-ADR #15 (`base_model`, `diagnostic_control`, `run_seed`): `1` dobra estendida (`davis/cold_target` fold 1), `5` `davis/random`, `25` KIBA (`cold_drug`, `cold_target`, `random`, `seqid40`, `seqid30`), `31` controles (`10` `ecfp_esm_mlp`, `15` `shuf0/1/2`, `6` `tsoff1000`). |
| Runs individuais GraphDTA (Fases 1–2) | `outputs/results/runs/graphdta_*.json` | **85** | `davis` (`cnn3`: `v1_seeds` `random`/`cold_drug`/`cold_target` + `v2_kfold` `cold_drug`/`cold_target`/`seqid40`/`seqid30` = 35; `gin`: idem = 35) + `kiba` (`gin` `v1_seeds` `random`/`cold_drug`/`cold_target` = 15; sem `graphdta_cnn3` no KIBA e sem `graphdta_*` em `kiba/seqid*`). |
| **Total de runs individuais** | `outputs/results/runs/*.json` | **166** | Chaveamento completo sem ausências nem duplicatas; `drug_in_dim = 78` (featurizer v2 `onehot78_l1norm`), `esm_model = facebook/esm2_t33_650M_UR50D` (`esm_dim = 1280`). |
| Agregados de modelos (`baseline_*` + `phase3_physdual_*`) | `outputs/results/*.json` | **34** | `17` `baseline_graphdta_*` + `10` `phase3_physdual_gin_esm_mlp_*` (Davis e KIBA) + `7` agregados de controle (`ecfp_esm_mlp` `cold_drug`/`cold_target`, `shuf0/1/2` `cold_target`, `tsoff1000` `cold_target_seqid30` e `cold_drug`). |
| Auditorias auxiliares em `outputs/results/` | `outputs/results/*.json` | **5** | `phase3_esm_cpu_baselines.json`, `kiba_cpu_classical_baselines.json`, `phase3_seqid_classical_baselines.json`, `phase3_kiba_paired_bootstrap.json`, `phase3_davis_paired_bootstrap.json` (**regenerado e sincronizado**, ver Seção 5). |
| Predições OOF em CSV | `outputs/results/oof/*.csv` | **27** | `6` `cpu_baselines_*_oof.csv`, `8` `graphdta_*_oof.csv` (Davis `v2_kfold`), `13` `physdual_*_oof.csv` (Davis `v2_kfold`, KIBA `seqid30/40` e controles). |
| Logs / mapas de clusterização `seqid` | `outputs/logs/*.json` | **5** | `davis_seqid_cluster_map.json`, `davis_seqid_splits_summary.json`, `kiba_seqid_cluster_map.json`, `kiba_seqid_splits_summary.json`, `phase1_summary_report.json`. |
| Cache ESM-2 e grafos processados | `cache/esm_embeddings/esm2_t33_650M_UR50D/`, `data/processed/` | **429 `.pt`**, **60 splits** | `429` tensores `.pt` cobrindo 100% das sequências únicas de Davis (`379`) e KIBA (`229`), considerando as `179` compartilhadas; `60` JSONs em `data/processed/splits/`; `130` checkpoints herdados (`graphdta_*`) em `outputs/checkpoints/`. |

---

## 2. Verificação de pré-registro (`ADR #15`)

### 2.1 Linha do tempo verificada no Git

| Ref / Tag | Hash completo | Data (`git show --format=%ci`) | Papel |
| :--- | :--- | :--- | :--- |
| `phase2-closed` | `3f05833bf60b6020e2d46792193cba938168bdea` | `2026-11-13 12:26:12 -0300` | Fechamento das Fases 1–2. |
| `1ae2853` | `1ae2853a0df362ee5c51d51b8689831675de571d` | `2026-11-13 19:03:09 -0300` | Implementação de `PhysDualPhase3` (`gin_esm_mlp`), orquestrador e gates (`epochs=500`). |
| `0d64b81` | `0d64b818bc7e3916f1ef52ee69951c8dc6d8d95d` | `2026-11-13 19:18:10 -0300` | Preservação de config YAML na CLI e reuso de cache ESM. |
| `06d1249` | `06d1249be07e288ee604b9e7b3af9aae8f881d1b` | `2026-11-13 21:52:44 -0300` | Emenda pré-congelamento: elevação do teto `epochs: 500 -> 1000` em `configs/base.yaml` e opção `--extend-epochs`. |
| **`46d13f8`** / tag **`phase3-adr15-pre-kiba`** | **`46d13f8e7aa054939d4f5867a7c460732d585d61`** | **`2026-11-14 00:32:14 -0300`** | **Congelamento do `ADR #15`** (`docs/DECISOES.md`), reclassificação dos splits `cold` do Davis como exploratórios, especificação dos critérios primário e negativo no KIBA e criação dos controles diagnósticos e splits `kiba_seqid30/40`. |
| `9a60916` | `9a60916806b6ff0b6c80e519f24144f7d97f67c5` | `2026-11-14 00:48:39 -0300` | Suporte a baselines `kiba_seqid30/40` e escopo por dataset no Gate 6. |
| `936c32f` (`HEAD`) | `936c32fbc09e83627474251197b4cf87d1982587` | `2026-11-14 00:48:39 -0300` | Exportação de OOF no `kiba/seqid` e opção `--dataset` no script de bootstrap pareado. |

### 2.2 Imutabilidade pós-congelamento e proveniência de schema (com ressalva forense)

- **Imutabilidade técnica (`git diff 46d13f8..HEAD`):** **`0` bytes** de diferença em `configs/base.yaml`, `docs/DECISOES.md` e `src/physdual/models/`.
- **Orçamento de tuning observado:** exatamente **`1`** configuração congelada em `configs/base.yaml` (`physdual_phase3`: `lr=0.0005`, `weight_decay=0.001`, `batch_size=512`, `epochs=1000`, `patience=50`, `davis_cold_patience=80`, `esm_dropout_in=0.4`, `prot_dropout=0.3`, `prot_output_dim=128`, `mlp_dropout=0.3`, `device=cuda`), com uma única alteração pré-congelamento (`epochs 500 -> 1000` em `06d1249`) e **`0`** edições de hiperparâmetros após `46d13f8`.
- **Proveniência de schema nos runs (`81/81`):**
  - `19` runs prévios ao ADR #15 (`davis` `cold_drug`, `cold_target` folds `0,2,3,4`, `cold_target_seqid40`, `cold_target_seqid30`) **não** possuem as chaves `base_model`, `diagnostic_control` e `run_seed`.
  - **`62/62`** runs pós-ADR #15 — incluindo todos os `25` runs de KIBA (`cold_drug`, `cold_target`, `random`, `seqid40`, `seqid30`), `5` de `davis/random`, `31` de controles (`ecfp_esm_mlp`, `shuf0/1/2`, `tsoff1000`) e a dobra `1` estendida de `davis/cold_target` — possuem as chaves `base_model`, `diagnostic_control` e `run_seed` introduzidas em `46d13f8`.
- **Auditoria forense dos timestamps de INÍCIO (`run_integrity.start_timestamp_audit`):** Embora os arquivos JSON não gravem `timestamp_utc` internamente e `st_mtime` tenha sido sobrescrito pelo sync do Drive (`~2026-11-14 23:25 -0300`), o `st_ctime` NTFS preservou a criação com offset constante de `-5h` (`UTC-8 PST` Drive/Colab vs `-03:00 BRT` Git), calibrado pelos `19` runs pré-ADR #15 após `1ae2853` e pelo log do Colab (`04:07 UTC = 01:07 BRT`). Estimando `t_início = st_ctime_calibrado - elapsed_sec`, obtém-se:
  - `19` runs pré-ADR #15 iniciaram entre `2026-11-13 19:33:27 -0300` e encerraram até `21:46:44 -0300`.
  - Primeiro run pós-ADR #15: `physdual_gin_esm_mlp_davis_cold_target_fold1.json` com início calibrado em **`2026-11-14 00:59:10 -0300`** (`26 min` após o commit `46d13f8` em `00:32:14 -0300`).
  - Primeiro run KIBA (`kiba_cold_drug_seed0`): **`2026-11-14 03:48:06 -0300`**; primeiro `davis_random`: **`2026-11-14 14:29:36 -0300`**; último término pós-ADR #15: `2026-11-14 19:58:28 -0300`.
  - **`post_runs_started_before_adr15_count = 0` (`0/62` inícios antes de `46d13f8`)**. Esse encadeamento convergente (timestamps de início calibrados + schema `46d13f8` + splits `kiba_seqid30/40` + `diff = 0`) valida a precedência temporal do pré-registro para os **`2/2` regimes primários** (`kiba/cold_drug`, `kiba/cold_target`) e **`2/2` secundários** (`kiba/cold_target_seqid40`, `kiba/cold_target_seqid30`) no KIBA, mantendo todos os regimes `cold` do Davis como **exploratórios**.

---

## 3. Completude, tetos de épocas, truncamento e memorização precoce

- **Completude dos agregados (`34/34`):** todos os `33` agregados de `5` unidades possuem exatamente `[0, 1, 2, 3, 4]` seeds/dobras completas; o agregado diagnóstico `phase3_physdual_gin_esm_mlp_tsoff1000_davis_cold_drug.json` possui exatamente `[2]`, como desenhado. Nenhum agregado incompleto (como o antigo incidente de `kiba_random` com 2 seeds) existe atualmente.
- **Teto de épocas efetivo (`effective_config.epochs`):**
  - `77/81` runs Fase 3 rodaram sob `epochs = 1000`;
  - Exceção única esperada (mistura de tetos `500/1000`): os `4` runs de `davis/cold_target` pré-`06d1249` (`folds 0, 2, 3, 4`) mantiveram `epochs = 500` e convergiram com folga (`best_epoch` entre `145` e `288`, `total_epochs_run` entre `225` e `368`), enquanto a dobra `1` (que antes atingira `481/500`) foi estendida para `epochs = 1000` e convergiu em `best_epoch = 525`, encerrando por *early stopping* na época `605` (`ci=0.8496, mse=0.2971`).
  - Nos `85` runs de GraphDTA (Fases 1–2): `83/85` com `epochs = 500`; `graphdta_gin` no `kiba/random` possui mistura `500` (`seeds 0, 2, 4`, `best_epoch = 286..343`) e `1000` (`seeds 1, 3`, `best_epoch = 509, 774`).
- **Truncamento / regra simétrica de `80%` (`best_epoch > 0,8 × epochs`):** verificada **simetricamente sobre todos os modelos neurais**, com **`0/81`** ocorrências em `physdual_*` (`gin_esm_mlp`, `ecfp_esm_mlp` e controles) e **`0/85`** em `graphdta_*` (`graphdta_gin`, `graphdta_cnn3`).
- **Tamanho da validação em `davis/cold_drug`:** cada dobra retém apenas **`5–6` fármacos únicos na validação** (`dobras 0..2 = 5`, `dobras 3..4 = 6`; `2.210–2.652` pares), explicando ruído de *early stopping* na dobra `2`. Note-se também que `ecfp_esm_mlp` apresentou **`2` colapsos precoces (`best_epoch = 3` nas dobras `3, 4`)** contra **`1` de `gin_esm_mlp` (`best_epoch = 5` na dobra `2`)**.
- **Isolamento, integridade numérica e tamanhos de split:** **`81/81`** com `test_isolation_verified = True`; **`0/81`** divergências de tamanho entre `oof_predictions` e `splits/*.json`; **`0/81`** ocorrências de `NaN` ou `Inf` em métricas escalares.
- **Memorização precoce (`best_epoch < 15` em regime `cold`):**
  - **`physdual_*` (`10` ocorrências, todas no Davis):**
    - `physdual_ecfp_esm_mlp_davis_cold_drug` dobras `3` e `4` (`best_epoch = 3, 3`; flag JSON `True`);
    - `physdual_gin_esm_mlp_davis_cold_drug` dobra `2` (`best_epoch = 5`, **`early_memorization_warn = False` no JSON histórico** anterior ao ajuste do alerta);
    - Controle `--shuffle-esm`: `shuf0` dobras `0, 2, 3, 4` (`6, 3, 4, 10`), `shuf1` dobra `0` (`13`), `shuf2` dobras `0, 4` (`11, 10`).
    - **KIBA:** **`0` ocorrências** (`best_epoch` mínimo = `54` em `kiba/cold_target_seqid40`).
  - **`graphdta_*` (`19` ocorrências retroativas, todas `graphdta_gin` no Davis):** `davis/cold_drug` dobra `4` (`2`); `davis/cold_target` `v1_seeds` 5/5 (`2..7`) e `v2_kfold` 4/5 (`3..13`); `davis/cold_target_seqid40` 4/5 (`3..8`); `davis/cold_target_seqid30` 5/5 (`3..7`). `graphdta_cnn3` teve `0`.

---

## 4. Reprodução numérica em CPU e integridade dos OOFs

1. **Regimes `v2_kfold` (via `outputs/results/oof/*.csv`, `6` regimes):**
   - `y_pred_drug_mean` / `y_pred_target_mean` (entre CSV do `physdual` e CSV de `cpu_baselines`): erro absoluto máximo `<= 8.88e-16`.
   - `gin_esm_mlp` métricas OOF globais recomputadas vs JSON agregado: erro máximo `<= 1.418e-04` (CI) e `<= 8.24e-09` (MSE).
   - Baselines clássicos e ESM vs JSONs de referência (`phase3_esm_cpu_baselines.json`, `phase3_seqid_classical_baselines.json`, `kiba_cpu_classical_baselines.json`): erro máximo em `<= 2.345e-05` no KIBA e no Davis `cold_target`/`cold_drug`/`seqid40`. Exceção pontual: `knn_esm_target` em `davis/cold_target_seqid30` apresentou `mse_diff = 2.22e-16` mas `ci_diff = 2.586e-03` (`spearman 0.4146 -> 0.4092`), causado por quebra de empates numéricos no arredondamento IEEE-754 ULP ao serializar o CSV em texto.
   - Reprodução exata do bootstrap agrupado salvo em `outputs/results/phase3_kiba_paired_bootstrap.json` (`n=1000`, `seed=42`): erro máximo de **`2.776e-17`** (`kiba/seqid40`) e **`1.110e-16`** (`kiba/seqid30`).
2. **Regimes `v1_seeds` (via recomputação direta em CPU a partir do cache ESM local de `429` `.pt` + `splits/*.json`, `4` regimes):**
   - `gin_esm_mlp` (`oof_predictions` vs `test_metrics_full` nos `20` run JSONs): diferença máxima `0.0` no KIBA (`2.80e-09` em `davis/random`).
   - `kiba/cold_drug`, `kiba/cold_target` e `kiba/random` reproduzem `phase3_esm_cpu_baselines.json` e `kiba_cpu_classical_baselines.json` com erro máximo `<= 9.22e-05` (e `paired_vs_baselines <= 9.19e-05`).
   - Em `davis/random`: `ridge_esm` tem erro `<= 8.25e-07`, `knn_esm_target` MSE exato (`<= 1.43e-09`) e CI diff `1.79e-03`, e `knn_kmer_target` / `knn_tanimoto_drug` diferem até `4.99e-03` em CI (`1.37e-03` em MSE) devido a desempates arbitrários entre vizinhos com similaridade `1.0` quando a mesma sequência/ligante aparece no treino e no teste do split `random` (agravado pelos mutantes colapsados do Davis).
3. **Escala dos grupos e sobreposição Davis–KIBA (calculado sobre `data/processed/*_clean.csv` e `splits/*.json`):**
   - Davis possui **`379`** sequências únicas (`81` `target_id`s colapsam em **`18`** sequências idênticas); KIBA possui **`229`** sequências únicas.
   - Sobreposição exata de sequências primárias entre Davis e KIBA: **`179` sequências (`78,17%` dos `229` alvos do KIBA já pertencem ao conjunto de sequências do Davis)**. Isso impede tratar o KIBA como domínio proteico independente para hiperparâmetros ou escolhas de arquitetura informadas pelo Davis.
   - Unidades efetivas no KIBA `v1_seeds` (5 seeds empilhadas): `kiba/cold_drug` testa **`206` SMILES por seed** (**`836`** únicos no agregado; `62.931` linhas empilhadas, `50.302` pares únicos); `kiba/cold_target` testa exatamente **`23` proteínas por seed** (**`93`** únicas no agregado; `58.004` linhas empilhadas, `47.219` pares únicos). No KIBA `v2_kfold` (`N = 118.254` pares OOF sem repetição): **`117`** clusters em `seqid40` e **`75`** clusters em `seqid30`.

---

## 5. Artefatos secundários defasados ou anômalos (reportados sem alteração de `outputs/`)

1. **`outputs/results/phase3_davis_paired_bootstrap.json` — RECONCILIADO (`is_stale = false`):**
   - Anteriormente defasado (`mtime = 2026-11-14 00:13:05`, `oof ci=0.8146, mse=0.4385`); **regenerado** via `python scripts/run_phase3_paired_bootstrap.py --dataset davis` (`mtime = 2026-11-15 11:42:00`), agora exatamente alinhado ao run estendido da dobra 1 (`oof ci=0.814896, mse=0.437192`).
   - Status auditado em `phase3_audits.json` (`run_integrity.stale_artifacts.phase3_davis_paired_bootstrap_json.status`): **`"regenerado_sincronizado"`**.
2. **`outputs/results/runs/physdual_gin_esm_mlp_davis_cold_drug_fold2.json` — metadado de alerta histórico:**
   - Teve `best_epoch = 5 < 15`, mas gravou `early_memorization_warn = False` (run anterior ao ADR #15). Auditado corretamente em `T7` com coluna separada retroativa.
3. **`outputs/results/runs/physdual_gin_esm_mlp_davis_cold_target_fold1.json` — artefato de medição de `sec_per_epoch`:**
   - `sec_per_epoch = 0.26` resulta da divisão do tempo apenas da retomada `--extend-epochs` (`~157s` entre épocas 500 e 605) pelo total acumulado `605`, e não representa aceleração de hardware (`~1.50s/época` reais na retomada, compatível com `~1.25s/época` nas dobras 0, 2, 3, 4).
4. **`outputs/results/phase3_physdual_gin_esm_mlp_tsoff1000_davis_cold_target_seqid30_kfold.json` — metadado de clusterização:**
   - `bootstrap_analysis.cluster_group_col` registra `"sequence"` em vez de `"seqid_cluster"`. Em `docs/paper/phase3_audits.json`, o bootstrap do controle `tsoff1000` foi corretamente agrupado por `seqid_cluster` (`120` clusters).
5. **`outputs/results/baseline_graphdta_gin_kiba_random.json` — teto heterogêneo herdado da Fase 2:**
   - Seeds `0, 2, 4` com `epochs = 500` e seeds `1, 3` com `epochs = 1000`.
6. **Lacunas estruturais de comparação no KIBA (não erro, limitação de protocolo):**
   - `graphdta_cnn3` não foi treinado no KIBA (nenhum regime); `graphdta_gin` tem desempenho fortemente degradado em `kiba/cold_target` (`CI=0.6579 / MSE=0.5137`) e está ausente em `kiba/seqid30/40`. Assim, a comparação primária no KIBA recai prioritariamente sobre os baselines tabulares/ESM/similares (`ridge_esm`, `knn_kmer_target`, `knn_tanimoto_drug`, `drug_mean_predictor`).
   - Limiares binários de AUROC/AUPR no Davis (`pKd > 5.0` e `pKd >= 7.0`) permanecem sem verificação em citação externa da literatura (marcados como `"a verificar"` nos documentos e tabelas).

---

## 6. Veredito operacional do `Passo 0`, gate estrutural e extensões CPU/Colab

**APROVADO PARA ANÁLISE COMPLETA (Passos 1–8), sem bloqueios nos runs primários ou de controle.** Os dados primários no KIBA (`2/2` primários + `2/2` secundários), a sanidade em `davis/random` e todos os controles diagnósticos no Davis estão íntegros, completos e reproduzidos em CPU dentro de precisão numérica esperada, usando exclusivamente `docs/paper/phase3_audits.json` e `docs/paper/phase3_intragroup_and_stacking.json` como bases auditadas.

- **Significado do gate (`verify_phase3.py --full --no-require-paired-win`):** sua aprovação valida **exclusivamente integridade estrutural, isolamento de splits e completude técnica** do pipeline; **não** representa sucesso científico sob o ADR #15 (cujo veredito permanece `nao_atendido_negativo` nos `2/2` primários e `2/2` secundários do KIBA).
- **Novas análises CPU somente leitura (`docs/paper/phase3_intragroup_and_stacking.json`) e diagnósticos Colab:** gerados via `scripts/paper/phase3_intragroup_and_stacking.py` (ranking intragrupo não enviesado por dobra/seed e *residual stacking* sobre OOFs existentes) e scripts diagnósticos prontos para Colab (`scripts/paper/colab_ridge_esm_pooling.py` e `scripts/paper/colab_kiba_cold_drug_ecfp.py`).
