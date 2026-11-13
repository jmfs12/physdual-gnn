# Rascunho 03 — Limitações, Divergências Auditadas e Ameaças à Validade

> **Objetivo:** Registrar com rigor científico todas as ameaças à validade experimental identificadas nas Fases 1 e 2, bem como todas as divergências entre artefatos encontradas na auditoria de proveniência (`00_INVENTARIO.md`), sem correções silenciosas.

---

## 1. Ameaças à Validade de Construto

1. **Censura à esquerda massiva no Davis (`69.64%` em $pK_d = 5{,}0$):**
   - `20.931` de `30.056` pares (`T1`, `F1`) compartilham exatamente o valor piso `5.0`, tornando `MSE` e `CI` no conjunto completo fortemente influenciados pela capacidade de prever o piso de inatividade e pelas médias marginais de promiscuidade por fármaco/alvo.
   - A avaliação isolada em `pKd > 5.0` (`T5`) expõe a perda severa de discriminação (`CI` cai de `0.7740` para `0.6315` no melhor caso `cold_target` v2), porém introduz **viés de seleção pelo rótulo verdadeiro**, elevando artificialmente o MSE. Os baselines no mesmo recorte `pKd > 5.0` foram recomputados em CPU (`docs/paper/cpu_audits.json`, `T5`), mostrando `mean_predictor` (média do treino completo) com `MSE = 2.12–2.23` (`1.58` no `v1 cold_drug`) e `mean_predictor` ativo (`pKd > 5.0` no treino) com `MSE = 0.95–1.14`.

2. **Colisão entre identificador de alvo e sequência em `Davis` (`proteins.txt`) e ausência de teste de mutações:**
   - Em `davis_clean.csv`, `18` sequências únicas colapsam `81 target_id`s distintos (ex.: `BRAF` vs. `BRAF(V600E)`, `RET` vs. `RET(M918T)/RET(V804L)/RET(V804M)`, `LRRK2` vs. `LRRK2(G2019S)`, `MET` vs. `MET(M1250T)/MET(Y1235D)`, `T1`), produzindo `546` grupos duplicados (`2.894` linhas) com **afinidades experimentais diferentes para exatamente a mesma entrada `(smiles, sequence)`**.
   - Embora o agrupamento estrito dos splits por `sequence` única garanta vazamento zero entre treino e teste (`T2`), o piso irreduzível de ruído medido em CPU (`docs/paper/cpu_audits.json`) é `MSE = 0.0322` (`RMSE = 0.1795`) no Davis completo e `MSE = 0.3347` sobre as linhas conflitantes — demonstrando ainda que **o Davis atual não testa resiliência a mutações pontuais** porque a variante e o tipo selvagem possuem sequência idêntica em `proteins.txt`.

3. **Particionamento frio por identidade exata, sem controle de similaridade contínua:**
   - Tanto `v1_seeds` quanto `v2_kfold` garantem ausência de repetição exata de `smiles` canônico (`cold_drug`) ou `sequence` única (`cold_target`), mas não impõem limiares de dissimilaridade estrutural (scaffold/Tanimoto) nem de identidade de sequência. O alto desempenho de `knn_tanimoto` (`CI = 0.7100`) e `knn_kmer` (`CI = 0.8128`) em `T4` sugere coocorrência de análogos próximos entre treino e teste. Para viabilizar esse controle como item **P0** antes da Fase 3, realizamos em CPU a clusterização por identidade global `BLOSUM62` (`biopython==1.88`, `PairwiseAligner`, `cpu_audits.json`) nas `379` sequências do Davis, obtendo `120` clusters em `id >= 30%` (5 folds balanceados com `5.848–6.664` pares) e `199` clusters em `id >= 40%` (`5.984–6.052` pares), prontos para execução de treino na próxima fase.

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
   - **Auditoria do cache ESM-2 no Google Drive (Concluída):** embora `cache/esm_embeddings/` esteja vazio localmente e `.venv` não possua `torch`, a execução de `scripts/paper/audit_esm_cache_colab.py` no Drive montado (`/content/drive/MyDrive/physdual-gnn/cache/esm_embeddings`) confirmou integridade dos **`429/429`** embeddings únicos de `esm2_t33_650M_UR50D` (`0` erros de shape `[len(seq), 1280]` nos `429` arquivos e **`83/83`** sequências únicas `> 1.022` aa com `windowed = True`, correspondendo a `70 + 42 - 29` compartilhadas). O único alerta (`430` arquivos, `missing_windowed_field = 1`, `4637bffbde58d549.pt` com shape `[65, 320]`) refere-se exclusivamente ao arquivo residual do *smoke test* da Fase 0 (`scripts/smoke_test.py` com `esm2_t6_8M_UR50D` na raiz do cache); `audit_esm_cache_colab.py` foi atualizado para inspecionar a subpasta do modelo e separar resíduos da raiz.

---

## 3. Ameaças à Validade Externa e Conclusão Estatística

1. **Baixo número efetivo de fármacos no Davis (`68` moléculas únicas) e natureza exploratória do bootstrap:**
   - Em `v1_seeds cold_drug`, o teste tem apenas `7` fármacos por seed; mesmo no `v2_kfold` são apenas `13..14` fármacos por dobra (`T2`). Os ICs 95% e frações favoráveis (`wins/1000`) do bootstrap agrupado por cluster em `v2_kfold` são reportados estritamente como **comparações exploratórias** (substituindo valores-p `0.000/1.000`) e exibem intervalos largos em `cold_drug` (`ΔMSE ∈ [-0.0487, 0.1369]` para GIN; `[-0.0431, 0.1021]` para CNN3, `T4`).

2. **Adiamento de `GraphDTACNN3` e do Protocolo `v2_kfold` no KIBA:**
   - Apenas `GraphDTAGIN` sob `v1_seeds` foi treinado no KIBA (`T3`, `00_INVENTARIO.md`). Ambos os complementos no KIBA (`CNN3` e `v2_kfold`) foram adiados até os resultados iniciais da Fase 3, de modo que a superioridade de `CNN3` observada em Davis `cold_target` ainda não foi testada no espaço químico maior do KIBA (`2.065` fármacos).

3. **Ausência de controle de variância por inicialização dentro da dobra (`v2_kfold`):**
   - Cada dobra do `v2_kfold` executou uma única seed de pesos (`split_seed = 0`), de modo que a dispersão reportada nas tabelas `v2_kfold` reflete variabilidade entre dobras de validação cruzada agrupada.
