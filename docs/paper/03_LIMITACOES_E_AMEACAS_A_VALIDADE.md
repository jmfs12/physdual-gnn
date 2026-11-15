# Rascunho 03 — Limitações, Divergências Auditadas e Ameaças à Validade (Fases 1, 2 e 3)

> **Objetivo:** Registrar com rigor científico todas as ameaças à validade experimental identificadas nas Fases 1, 2 e 3, bem como todas as divergências entre artefatos encontradas nas auditorias de proveniência (`00_INVENTARIO.md` e `06_AUDITORIA_FASE3.md`), sem correções silenciosas.

---

## 1. Ameaças à Validade de Construto

1. **Censura à esquerda massiva no Davis (`69.64%` em $pK_d = 5{,}0$):**
   - `20.931` de `30.056` pares (`T1`, `F1`) compartilham exatamente o valor piso `5.0`, tornando `MSE` e `CI` no conjunto completo fortemente influenciados pela capacidade de prever o piso de inatividade e pelas médias marginais de promiscuidade por fármaco/alvo.
   - A avaliação isolada em `pKd > 5.0` (`T5`) expõe a perda severa de discriminação (`CI` cai de `0.7740` para `0.6315` no melhor caso `cold_target` v2), porém introduz **viés de seleção pelo rótulo verdadeiro**, elevando artificialmente o MSE. Os baselines no mesmo recorte `pKd > 5.0` foram recomputados em CPU (`docs/paper/cpu_audits.json`, `T5`), mostrando `mean_predictor` (média do treino completo) com `MSE = 2.12–2.23` (`1.58` no `v1 cold_drug`) e `mean_predictor` ativo (`pKd > 5.0` no treino) com `MSE = 0.95–1.14`.

2. **Colisão entre identificador de alvo e sequência em `Davis` (`proteins.txt`) e ausência de teste de mutações:**
   - Em `davis_clean.csv`, `18` sequências únicas colapsam `81 target_id`s distintos (ex.: `BRAF` vs. `BRAF(V600E)`, `RET` vs. `RET(M918T)/RET(V804L)/RET(V804M)`, `LRRK2` vs. `LRRK2(G2019S)`, `MET` vs. `MET(M1250T)/MET(Y1235D)`, `T1`), produzindo `546` grupos duplicados (`2.894` linhas) com **afinidades experimentais diferentes para exatamente a mesma entrada `(smiles, sequence)`**.
   - Embora o agrupamento estrito dos splits por `sequence` única garanta vazamento zero entre treino e teste (`T2`), o piso irreduzível de ruído medido em CPU (`docs/paper/cpu_audits.json`) é `MSE = 0.0322` (`RMSE = 0.1795`) no Davis completo e `MSE = 0.3347` sobre as linhas conflitantes — demonstrando ainda que **o Davis atual não testa resiliência a mutações pontuais** porque a variante e o tipo selvagem possuem sequência idêntica em `proteins.txt`.

3. **Particionamento frio por identidade exata vs. controle contínuo de similaridade (auditoria `seqid` e Smith-Waterman):**
   - Os splits `seqid40` (`199` clusters Davis; `117` KIBA) e `seqid30` (`120` Davis; `75` KIBA) foram construídos com `PairwiseAligner(mode='global', matrix='BLOSUM62')` sob identidade `exact_matches / max(len_a, len_b)` e *single-linkage* + LPT (`01_METODOS_DADOS_E_BASELINES.md`, `src/physdual/data/splits.py`), o que gera grandes componentes conexos em `seqid30` (`92` seqs no Davis; `63` no KIBA em `fold0`) e penaliza pares assimétricos em comprimento. Na auditoria por dobra (`docs/paper/seqid_identity_audit.json`), embora `global_id_max` tenha `0%` de violações, **homólogos locais de domínio permanecem frequentes**: sob Smith-Waterman `>= 150 aa` em `seqid30`, **`33,8%`** das proteínas teste do Davis atingem `>= 40%` (máximo `70,3%`) contra **`21,0%`** no KIBA (máximo `46,2%`, `0% >= 50%`), e `100%` em ambos atingem `global exact_matches / min(len) >= 30%`. Observa-se que `Davis seqid30` ainda retém sinal intragrupo (`0.580..0.586`) enquanto `KIBA seqid30` perde o sinal linear (`~0.50` em Ridge / *stacking*); qualquer nexo causal permanece **hipótese**.
   - Para fármacos (`cold_drug`), **não há splits por Bemis–Murcko scaffold nem por similaridade de Tanimoto** implementados. Por regra metodológica, novos splits químicos **não são propostos para reverter o veredito negativo da Fase 3** — devendo figurar apenas, se pré-registrados em fase futura, como regimes adicionais reportados **lado a lado** com os splits originais.

4. **Truncamento de sequências em `max_seq_len = 1.000` aa nos baselines:**
   - Afeta `76` sequências únicas (`7.820` pares, `26.02%`) no Davis e `44` (`24.103` pares, `20.38%`) no KIBA (`T1`). Embora `T6` e `F6` mostrem MSE consistentemente maior para proteínas `> 1.000` aa em todos os `13` grupos, este dado isolado é observacional e confunde possível perda de domínios C-terminais com diferenças de variância/dificuldade intrínseca desse estrato.

---

## 2. Ameaças à Validade Interna e Consistência Experimental

1. **Paridade metodológica e distorção de `sec_per_epoch` nas retomadas de `kiba_random`:**
   - As seeds `1` e `3` de `graphdta_gin_kiba_random` foram retomadas até `epochs = 1000` (`best_epoch = 774` e `509`), enquanto `seed0, seed2, seed4` haviam encerrado por *early stopping* (`patience = 50`) em `344, 336, 393` épocas (muito abaixo de 500). Logo, **o teto efetivo é `1000` épocas para todas as 5 seeds** sem necessidade de re-execução das seeds 0, 2 e 4.
   - Contudo, nas duas runs retomadas, `history` contém apenas as épocas pós-500 (`501..824` e `501..559`), `fold` foi gravado como `null` e o `sec_per_epoch` bruto nos JSONs (`0.646 s` e `2.347 s`) subestima o custo real porque divide apenas o tempo pós-500 pelo total acumulado; por isso, `T8` e `cpu_audits.json` recalculam a velocidade real usando apenas os 3 runs não retomados (`22.338 ± 0.014 s/época`).

2. **Ausência de histórico época a época (`history`) em `43/45` runs de `v1_seeds`:**
   - Embora os baselines marginais, `kNN` e deltas pareados de todas as `45/45` runs `v1_seeds` tenham sido recomputados deterministicamente em CPU em `docs/paper/cpu_audits.json` (`T4`, `F2`, `F3`), os `43` runs originais de `v1_seeds` não gravaram curvas `history` por época em disco — inviabilizando traçar `F4` para esses runs sem novo treino.

3. **Colapso precoce de treinamento no `GraphDTAGIN` em Davis `cold_target`:**
   - Em `v1_seeds`, todas as 5 seeds selecionaram checkpoints entre as épocas `2` e `7`; em `v2_kfold`, 4 de 5 dobras estacionaram entre `3` e `13` (`fold0` em `236`, `T7`, `F5`). O critério de parada baseado em `val_mse` em um conjunto de validação de apenas `30..38` sequências únicas exibe alta volatilidade época a época (`F4`), o que limita a capacidade de identificar se a falha é puramente arquitetural ou parcialmente amplificada por instabilidade de otimização/validação sem `lr_scheduler`.

4. **Auditoria de Gate 4, correções P2 aplicadas e auditoria do cache ESM-2 no Drive:**
   - **O que o Gate 4 verificava vs. isolamento C1:** em `scripts/verify_phase2.py`, `gate4_strict_test_isolation` originalmente verificava apenas que `val_metrics_at_best` e `test_metrics_full` estavam não vazios nos JSONs de run (e `test_isolation_verified` não existia nos artefatos); o isolamento estrutural real durante o treino (critério **C1**: `test_loader` nunca iterado dentro do loop de épocas) é garantido em `src/physdual/training/trainer.py`. O código de `scripts/run_phase2_baseline.py` e todos os `65` runs + `13` agregados locais foram atualizados com `test_isolation_verified: true`.
   - **Divergências P2 já saneadas localmente:** `outputs/logs/phase1_summary_report.json` agora inclui KIBA e `windowed_sequences.count = 112` (`70` Davis + `42` KIBA; `83` sequências globalmente únicas); `docs/FASE2_BASELINE.md` corrigiu `20.932` -> `20.931` e `83 quinases` -> `70 seqs / 109 IDs`; e `oof_predictions` foi removido do agregado `baseline_graphdta_gin_kiba_random.json` (reduzido de `~10.1 MB` para `~87 KB`).
   - **Auditoria do cache ESM-2 no Google Drive (Concluída — `all_checks_passed: true`):** embora `cache/esm_embeddings/` esteja vazio localmente e `.venv` não possua `torch`, a execução final de `scripts/paper/audit_esm_cache_colab.py` sobre `/content/drive/MyDrive/physdual-gnn/cache/esm_embeddings/esm2_t33_650M_UR50D` aprovou `429/429` arquivos com **`0` erros** em todas as checagens: carregamento `weights_only=True`, `dtype == torch.float16` em disco, `0` valores `NaN`/`Inf`, shape `[len(seq), 1280]` + metadado `dim == 1280`, `seq_len == len(seq)` validado contra os CSVs limpos, **`83/83`** sequências únicas `> 1.022` aa com `windowed = True` (`70 + 42 - 29` compartilhadas = `83`, contra `112` por soma de dataset) e `root_residual_pt_files: []`.

---

## 3. Ameaças à Validade Externa e Conclusão Estatística

1. **Baixo número efetivo de fármacos no Davis (`68` moléculas únicas) e natureza exploratória do bootstrap:**
   - Em `v1_seeds cold_drug`, o teste tem apenas `7` fármacos por seed; mesmo no `v2_kfold` são apenas `13..14` fármacos por dobra (`T2`). Os ICs 95% e frações favoráveis (`wins/1000`) do bootstrap agrupado por cluster em `v2_kfold` são reportados estritamente como **comparações exploratórias** (substituindo valores-p `0.000/1.000`) e exibem intervalos largos em `cold_drug` (`ΔMSE ∈ [-0.0487, 0.1369]` para GIN; `[-0.0431, 0.1021]` para CNN3, `T4`).

2. **Adiamento de `GraphDTACNN3` e do Protocolo `v2_kfold` no KIBA (Fases 1–2):**
   - Apenas `GraphDTAGIN` sob `v1_seeds` foi treinado no KIBA (`T3`, `00_INVENTARIO.md`). Na Fase 3, esse cenário persistiu: não há `GraphDTACNN3` no KIBA nem execuções GraphDTA em `kiba/cold_target_seqid30` e `seqid40` (ver Seção 4).

3. **Ausência de controle de variância por inicialização dentro da dobra (`v2_kfold`):**
   - Cada dobra do `v2_kfold` executou uma única seed de pesos (`split_seed = 0`), exceto a sensibilização diagnóstica `--train-seed-offset 1000` em `davis/cold_target_seqid30` e na dobra 2 de `davis/cold_drug` (`T14`). Assim, a dispersão nas tabelas `v2_kfold` reflete variabilidade entre dobras de validação cruzada agrupada.

---

## 4. Limitações Específicas e Ameaças à Validade da Fase 3 (`gin_esm_mlp`)

1. **Sobreposição massiva de sequências proteicas entre Davis e KIBA (`78,17%` do KIBA):**
   - De acordo com `phase3_audits.json` (`dataset_overlap_and_units`), o KIBA possui `229` sequências únicas de proteínas e o Davis `379`, compartilhando **`179` sequências exatas (`78,17%` de `229`)**. Portanto, embora o KIBA expanda drasticamente o espaço químico (`2.068` / `2.065` fármacos únicos vs. `68` no Davis), os dois datasets **não representam validações independentes no domínio proteico**; ganhos ou falhas de `gin_esm_mlp` em `kiba/cold_target` refletem parcialmente o mesmo universo de quinases já presente no Davis.

2. **Ausência de `GraphDTACNN3` no KIBA, ausência de GraphDTA em `kiba/seqid*` e fraqueza de `GraphDTAGIN` em `cold_target`:**
   - No KIBA (`v1_seeds` e `v2_kfold`), `GraphDTACNN3` não foi executado (`não disponível`), e no KIBA `seqid30/40` não há nenhum baseline neural GraphDTA (`T10`). Embora `GraphDTAGIN` exista em `kiba/cold_target`, ele reproduz o colapso de codificação posicional (`CI = 0.6579 ± 0.0437 / MSE = 0.5137 ± 0.0888`), inferior ao baseline não neural `knn_kmer_target` (`0.7320 / 0.3730`). Consequentemente, no KIBA não há comparação neural não-ESM forte em alvos novos — lacuna parcialmente compensada pelo critério do **ADR #15**, que usou `knn_kmer_target` e `knn_tanimoto_drug` como melhores comparadores sem ESM.

3. **Amostragem reduzida de clusters únicos e ruído de *early stopping* em `davis/cold_drug` e `kiba/cold_target`:**
   - Em `kiba/cold_target` (`v1_seeds`, `phase3_audits.json`), cada seed testa apenas **`23` sequências únicas de alvo** (`93` clusters únicos no pool das 5 seeds; `8.264` a `14.469` pares/seed), e em `kiba/cold_drug` apenas `206` SMILES por seed (`836` no pool). Em `davis/cold_drug` (`v2_kfold`), os `68` compostos totais deixam apenas **`13–14` fármacos únicos no teste** e apenas **`5–6` fármacos únicos na validação de cada dobra** (`dobras 0..2 = 5` fármacos / `2.210` pares; `dobras 3..4 = 6` fármacos / `2.652` pares), tornando o critério `val_mse` de *early stopping* altamente ruidoso (ex.: dobra 2 de `gin_esm_mlp` salta de `best_epoch = 5, MSE = 1.3670` para `best_epoch = 31, MSE = 1.0765` apenas ao alterar a semente via `tsoff1000`). Um protocolo alternativo de seleção de época (p. ex. parada baseada em métrica intragrupo ou validação com mais grupos) é proposto exclusivamente como item a pré-registrar antes de eventual Fase 4.

4. **Dependência entre treinos nas dobras `v2_kfold` e baixo poder de testes com `n = 5`:**
   - As `5` dobras do `v2_kfold` compartilham `~80%` dos dados de treino entre si, e `v1_seeds` reamostra o mesmo conjunto KIBA/Davis (`n = 5` unidades), conferindo baixo poder para comparações pareadas em nível de unidade. O bootstrap pareado em bloco (`n=1000, seed=42`, `T13`) mitiga a pseudorreplicação por par ao reamostrar **clusters únicos** (`68` a `836`), mas não elimina a sobreposição entre conjuntos de treinamento das dobras/seeds. Além disso, as métricas intragrupo são calculadas exclusivamente **dentro de cada dobra/seed** (`per_fold_min2`, com `drug_mean_predictor = 0.5000 / 0.0000`, `phase3_intragroup_and_stacking.json`), evitando artefatos de centralização decorrentes de concatenação entre dobras.

5. **Simetria do teto de épocas (`80%`), mistura `500/1000` em `davis/cold_target` e orçamento de ajuste:**
   - A regra de não truncamento (`best_epoch <= 0,8 × epochs`) foi aplicada e verificada **simetricamente em todos os modelos neurais**: **`0/81` runs `physdual_*`** e **`0/85` runs `graphdta_*`** ultrapassaram `80%` do teto. Em `davis/cold_target`, os `4` folds `0,2,3,4` pré-`06d1249` mantiveram `epochs = 500` (convergindo com folga em `best_epoch = 145..288`), enquanto a dobra `1` foi estendida via `--extend-epochs` para `epochs = 1000` (`best_epoch = 525`, parada na `605`, gerando mistura de tetos `500/1000` e `sec_per_epoch = 0.260` artificial pela retomada do checkpoint). Todos os `62` runs pós-ADR #15 rodaram sob a mesma configuração congelada (`epochs = 1000`, `0` edições após `46d13f8`), restando `1` ponto de configuração avaliado na Fase 3.

6. **Múltiplas comparações exploratórias sem ajuste de multiplicidade:**
   - O **ADR #15** confinou o teste confirmatório estritamente aos **`2/2` regimes primários** (`kiba/cold_drug`, `kiba/cold_target`) e **`2/2` regimes secundários** (`kiba/cold_target_seqid40`, `kiba/cold_target_seqid30`) do KIBA (`2` comparadores por regime). Todas as demais comparações em `T10`–`T14` (Davis cold, médias marginais, subconjuntos não censurados e classificação binária com limiares a verificar) são exploratórias e não passaram por correção de múltiplos testes. Ademais, a aprovação de `scripts/verify_phase3.py --full --no-require-paired-win` atesta **exclusivamente integridade estrutural**, e **não** sucesso científico.

7. **Natureza estritamente diagnóstica dos controles e colapsos precoces (`best_epoch < 15`):**
   - Os controles `--shuffle-esm` (`shuf0/1/2` em `davis/cold_target`), `ecfp_esm_mlp` (`davis/cold_target` e `davis/cold_drug`), `--train-seed-offset 1000` e a análise *post-hoc* sem a dobra 2 (`T14`) não conferem evidência confirmatória. Ocorrências de `best_epoch < 15` em `physdual_*` somaram `10/81` no Davis (`0/30` no KIBA): `7` em `--shuffle-esm`, **`2` em `ecfp_esm_mlp` (`davis/cold_drug` dobras `3` e `4`, `best_epoch = 3`)** e **`1` em `gin_esm_mlp` (`davis/cold_drug` dobra `2`, `best_epoch = 5`)** — mostrando que `ecfp_esm_mlp` **não** é mais imune a colapso precoce do que `gin_esm_mlp` (vs. `19/85` de `GraphDTAGIN`).

8. **Mutantes colapsados em `davis/random` e sincronização de `phase3_davis_paired_bootstrap.json`:**
   - O dataset Davis contém **`81` IDs de alvo colapsados em apenas `18` sequências exatas idênticas** (`dataset_overlap_and_units.davis_collapsed_mutants`). No split `davis/random`, as mesmas proteínas e sequências idênticas aparecem simultaneamente em treino e teste, explicando parte do desempenho elevado de `gin_esm_mlp` (`CI = 0.9057`) e `knn_kmer_target` (`0.8481`) sem implicar capacidade de generalização.
   - **Artefato previamente defasado já reconciliado:** `outputs/results/phase3_davis_paired_bootstrap.json` foi regenerado via `python scripts/run_phase3_paired_bootstrap.py --dataset davis` e encontra-se **sincronizado** (`is_stale = false`, `OOF CI = 0.814896`, `MSE = 0.437192`, `phase3_audits.json`).
