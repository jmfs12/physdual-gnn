# Rascunho 02 — Resultados Preliminares (Fases 1 e 2)

> **Regra de integridade:** Todos os números abaixo provêm diretamente das tabelas `T1`–`T8`, figuras `F1`–`F6` e JSONs em `outputs/results/`. Protocolos **v1 (`v1_seeds`)** e **v2 (`v2_kfold`)** são reportados em seções estritamente separadas e nunca combinados em uma mesma média. Nenhum estudo de caso de fármaco/alvo específico é reportado nesta etapa.

---

## 1. Reprodutibilidade em Particionamento Aleatório (`random`, Protocolo v1)

Na avaliação sob particionamento `random` (`80/10/10`, `5` seeds `0..4`, `T3`, `F2`), os baselines neurais reproduzem a ordem de grandeza reportada na literatura de referência para Davis e superam amplamente os baselines triviais (`mean_predictor` e `ridge_aa_composition`):

- **Davis `random` (`v1_seeds`, n=5):**
  - `graphdta_gin` (oficial): `CI = 0.8845 ± 0.0057`, `MSE = 0.2348 ± 0.0166`, `RMSE = 0.4843 ± 0.0172`, `Pearson = 0.8455 ± 0.0151`, `Spearman = 0.6864 ± 0.0149`, `rm2 = 0.7008 ± 0.0214`. Na Tabela 2 de Nguyen et al. (2021, *Bioinformatics* 37(8), p. 1144), o GraphDTA-GIN reportado em Davis *Setting 1* atinge `CI = 0.893` e `MSE = 0.229` (split aleatório único, diferença compatível com avaliação multi-seed 80/10/10 com *early stopping* na validação).
  - `graphdta_cnn3` (variante): `CI = 0.8895 ± 0.0035`, `MSE = 0.2216 ± 0.0128`, `RMSE = 0.4706 ± 0.0136`, `Pearson = 0.8552 ± 0.0088`, `Spearman = 0.6931 ± 0.0079`, `rm2 = 0.7123 ± 0.0297`.
  - Ambos superam o `mean_predictor` (`MSE = 0.8214 ± 0.0312`), `ridge_aa_composition` (`CI = 0.5931 ± 0.0080`, `MSE = 0.7965 ± 0.0302`), os marginais (`drug_mean CI = 0.7487 / MSE = 0.6469`; `target_mean 0.6617 / 0.7310`) e os baselines `kNN` (`knn_tanimoto 0.7619 / 0.4988`; `knn_kmer 0.8491 / 0.3373`), com `5/5` vitórias pareadas (`ΔMSE = -0.4121` e `-0.4253`, `T4`).

- **KIBA `random` (`v1_seeds`, n=5, teto efetivo 1000 épocas):**
  - `graphdta_gin`: `CI = 0.8578 ± 0.0134`, `MSE = 0.1753 ± 0.0190`, `RMSE = 0.4182 ± 0.0232`, `Pearson = 0.8644 ± 0.0133`, `Spearman = 0.8428 ± 0.0173`, `rm2 = 0.7337 ± 0.0301` (contra `mean_predictor MSE = 0.6910 ± 0.0152` e `ridge_aa CI = 0.5878 ± 0.0020`, `MSE = 0.6686 ± 0.0151`). Na Tabela 3 de Nguyen et al. (2021, p. 1144), o GraphDTA-GIN reporta `CI = 0.882` e `MSE = 0.147` no KIBA *Setting 1* (e `GAT_GCN` / `GCN` reportam `0.891` / `0.889` e `MSE = 0.139`); métricas de `Pearson`, `Spearman`, `rm2` e *cold-start* não constam na referida tabela (permanecem `a verificar` em fontes externas).
  - **Paridade de 1000 épocas e baselines marginais/kNN (`5/5` seeds, `T4`, `T7`, `cpu_audits.json`):** As `3` seeds `0, 2, 4` encerraram por *early stopping* (`patience = 50`) antes de `500` épocas (`best_epoch = 294, 286, 343`; `CI = 0.8485, 0.8467, 0.8556`), enquanto as `2` seeds estendidas até `1000` (`seed1, seed3`) alcançaram `best_epoch = 774` (`CI = 0.8803, MSE = 0.1451`) e `509` (`CI = 0.8579, MSE = 0.1697`) — tornando `1000` o teto efetivo para todas as `5` seeds. Na comparação pareada sobre todas as `5/5` seeds (`T4`), o `graphdta_gin` vence em `5/5` seeds o `drug_mean_predictor` (`CI = 0.7010 ± 0.0008`, `MSE = 0.4278 ± 0.0096`) com `ΔMSE = -0.2525 ± 0.0103` e `ΔCI = +0.1568 ± 0.0134`, superando também `knn_tanimoto` (`0.7899 / 0.3448`) e `knn_kmer` (`0.7451 / 0.3649`).
  - `graphdta_cnn3` no KIBA: **`não disponível`** (adiado para após os resultados iniciais da Fase 3 pelo critério computacional no ADR #12).

---

## 2. Desempenho sob Generalização Fria: Protocolo v1 (`v1_seeds`, 5 seeds 80/10/10)

Nos regimes frios sob `v1_seeds` (`T3`, `T4`, `F2`, `F3`), ambos os modelos degradam fortemente face ao `random`, e a comparação pareada recomputada em CPU sobre as `5/5` seeds revela padrão distinto entre `cold_drug` e `cold_target`:

- **Davis `cold_drug` (`v1_seeds`, 7 fármacos em teste/seed):**
  - `graphdta_gin`: `CI = 0.6934 ± 0.0382`, `MSE = 0.5221 ± 0.1495`, `rm2 = 0.1512 ± 0.0580`; vs. `target_mean_predictor` (`CI = 0.6721 ± 0.0178`, `MSE = 0.5625 ± 0.1163`): `ΔMSE = -0.0404 ± 0.0474`, `ΔCI = +0.0214 ± 0.0339` (`4/5 MSE | 3/5 CI`).
  - `graphdta_cnn3`: `CI = 0.6918 ± 0.0343`, `MSE = 0.5169 ± 0.1671`, `rm2 = 0.1630 ± 0.0551`; vs. `target_mean_predictor`: `ΔMSE = -0.0456 ± 0.0746`, `ΔCI = +0.0198 ± 0.0356` (`4/5 MSE | 3/5 CI`).
  - `knn_tanimoto_drug` obtém `CI = 0.7041 / MSE = 0.6139` (`T4`, CI numericamente superior ao de ambos os modelos neurais, ainda que com MSE maior).

- **Davis `cold_target` (`v1_seeds`, 38 sequências únicas em teste/seed):**
  - `graphdta_gin`: `CI = 0.7284 ± 0.0312`, `MSE = 0.7493 ± 0.3338`, `Pearson = 0.4652 ± 0.0520`, `rm2 = 0.2102 ± 0.0478` — **perde `0/5` seeds em MSE e `0/5` em CI** para `drug_mean_predictor` (`CI = 0.7579 ± 0.0287`, `MSE = 0.6417 ± 0.2626`): `ΔMSE = +0.1076 ± 0.0940`, `ΔCI = -0.0295 ± 0.0178`, com **colapso precoce unânime (`best_epoch` entre `2` e `7`, `T7`, `F5`)**.
  - `graphdta_cnn3`: `CI = 0.7698 ± 0.0374`, `MSE = 0.5677 ± 0.2908`, `Pearson = 0.6106 ± 0.0833`, `rm2 = 0.3692 ± 0.0936`, com `best_epoch` entre `47` e `94` e **vitória em `5/5` seeds** sobre `drug_mean_predictor` (`ΔMSE = -0.0740 ± 0.0420`, `ΔCI = +0.0119 ± 0.0101`), porém ainda abaixo de `knn_kmer_target` (`CI = 0.8137 / MSE = 0.4864`).

- **KIBA frio (`v1_seeds`, apenas `graphdta_gin`):**
  - `cold_drug`: `CI = 0.7560 ± 0.0107`, `MSE = 0.3808 ± 0.0532`, `rm2 = 0.4082 ± 0.0297`; supera `target_mean_predictor` (`CI = 0.6869 ± 0.0048`, `MSE = 0.5732 ± 0.0585`) em **`5/5` seeds** (`ΔMSE = -0.1925 ± 0.0196`, `ΔCI = +0.0691 ± 0.0075`), mas fica atrás de `knn_tanimoto_drug` (`CI = 0.7832 / MSE = 0.3521`).
  - `cold_target`: `CI = 0.6579 ± 0.0437`, `MSE = 0.5137 ± 0.0888`, `rm2 = 0.2416 ± 0.0712` — **repete o fracasso do Davis frente a `drug_mean_predictor` (`CI = 0.6964 ± 0.0122`, `MSE = 0.4465 ± 0.0843`)**, perdendo **`0/5` seeds em MSE** (`ΔMSE = +0.0672 ± 0.0281`) e vencendo apenas `1/5` em CI (`ΔCI = -0.0384 ± 0.0400`), enquanto `knn_kmer_target` atinge `CI = 0.7320 / MSE = 0.3730` (`T4`, `F3`).

---

## 3. Avaliação Pareada em Davis Frio: Protocolo v2 (`v2_kfold`, 5-Fold `GroupKFold`)

O Protocolo v2 (`T3`, `T4`, `F2`, `F3`) cobre integralmente os `30.056` pares de Davis em avaliação OOF e permite comparação pareada dobra a dobra e **bootstrap exploratório agrupado no nível de cluster** (`1.000` reamostras, reportado com IC 95% e fração de reamostras favoráveis):

1. **Davis `cold_drug_kfold` (relevante: `target_mean_predictor`, `CI = 0.6690 ± 0.0326`, `MSE = 0.7310 ± 0.2189`):**
   - **`graphdta_gin`:** `CI = 0.6623 ± 0.0572` (`OOF = 0.6424`), `MSE = 0.7739 ± 0.2059` (`OOF = 0.7716`), `AUROC(>5.0) = 0.6569`, `AUROC(>=7.0) = 0.7199`. Na comparação pareada, **não supera o `target_mean_predictor`**: `ΔMSE = +0.0430 ± 0.0884` (`1/5` vitórias em MSE), `ΔCI = -0.0067 ± 0.0445` (`2/5` vitórias em CI). No bootstrap exploratório agrupado por `smiles`: `ΔMSE = +0.0410 [-0.0487, 0.1369]` (`220/1000` reamostras favoráveis) e `ΔCI = -0.0099 [-0.0563, 0.0373]` (`345/1000` favoráveis).
   - **`graphdta_cnn3`:** `CI = 0.6572 ± 0.0776` (`OOF = 0.6519`), `MSE = 0.7610 ± 0.1403` (`OOF = 0.7605`), `AUROC(>5.0) = 0.6668`, `AUROC(>=7.0) = 0.7351`. Também **não supera o `target_mean_predictor`**: `ΔMSE = +0.0300 ± 0.1237` (`2/5` vitórias em MSE), `ΔCI = -0.0118 ± 0.0785` (`3/5` vitórias em CI); bootstrap exploratório: `ΔMSE = +0.0277 [-0.0431, 0.1021]` (`220/1000` favoráveis) e `ΔCI = +0.0004 [-0.0588, 0.0557]` (`508/1000` favoráveis).
   - **Baseline de similaridade química (`knn_tanimoto`):** atinge `CI = 0.7100 / MSE = 0.7580` (`T4`), superando ambos os modelos neurais em CI (`+0.0477` sobre GIN; `+0.0528` sobre CNN3) com MSE equivalente.

2. **Davis `cold_target_kfold` (relevante: `drug_mean_predictor`, `CI = 0.7488 ± 0.0150`, `MSE = 0.6253 ± 0.1263`):**
   - **`graphdta_gin`:** `CI = 0.7117 ± 0.0112` (`OOF = 0.6861`), `MSE = 0.7524 ± 0.2713` (`OOF = 0.7648`), `AUROC(>5.0) = 0.7093`, `AUROC(>=7.0) = 0.7296`. É **superado pelo `drug_mean_predictor`**: `ΔMSE = +0.1271 ± 0.1599` (`1/5` vitórias em MSE), `ΔCI = -0.0370 ± 0.0151` (`0/5` vitórias em CI); bootstrap exploratório agrupado por `sequence`: `ΔMSE = +0.1315 [0.0694, 0.2187]` (`0/1000` reamostras favoráveis) e `ΔCI = -0.0569 [-0.0861, -0.0366]` (`0/1000` favoráveis). O diagnóstico de treinamento (`T7`, `F4`, `F5`) confirma parada extremamente precoce em 4 de 5 dobras (`best_epoch = [236, 13, 3, 6, 9]`), com forte oscilação de `val_mse` na curva observada (`F4`, dobra 3).
   - **`graphdta_cnn3`:** `CI = 0.7740 ± 0.0180` (`OOF = 0.7715`), `MSE = 0.5350 ± 0.1152` (`OOF = 0.5407`), `AUROC(>5.0) = 0.8020`, `AUROC(>=7.0) = 0.8368`. **Supera consistentemente o `drug_mean_predictor`**: `ΔMSE = -0.0903 ± 0.0224` (`5/5` vitórias em MSE), `ΔCI = +0.0252 ± 0.0231` (`4/5` vitórias em CI); bootstrap exploratório: `ΔMSE = -0.0910 [-0.1415, -0.0464]` (`1000/1000` favoráveis) e `ΔCI = +0.0264 [0.0038, 0.0463]` (`992/1000` favoráveis).
   - **Baseline de homologia (`knn_kmer`):** atinge `CI = 0.8128 / MSE = 0.4591` (`T4`), superando inclusive o `graphdta_cnn3` (`+0.0388` em CI e `-0.0759` em MSE).

---

## 4. Subconjunto Não Censurado (Davis, $pK_d > 5{,}0$), Piso de Ruído e Comprimento Proteico

1. **Subconjunto não censurado (`T5`) e piso de ruído (`T1`):**
   - Ao filtrar o conjunto de teste para os `9.125` pares ativos (`pKd > 5.0`), o desempenho discriminativo cai substancialmente em todos os protocolos e o MSE sobe (refletindo tanto maior erro nas afinidades ativas quanto o viés de seleção condicionado no rótulo):
     - No `random` (`v1_seeds`): `graphdta_gin` cai para `CI = 0.7531 ± 0.0109 / MSE = 0.6183 ± 0.0593`; `graphdta_cnn3` cai para `CI = 0.7613 ± 0.0093 / MSE = 0.5743 ± 0.0310` (vs. `mean_predictor` treino completo `MSE = 2.2299 ± 0.0774`; `mean_predictor` ativo `MSE = 1.1412 ± 0.0520`; `drug_mean = 1.6700`, `cpu_audits.json`).
     - Em `v2_kfold` `cold_drug`: `gin` cai para `CI = 0.5890 ± 0.0264 / MSE = 1.9941 ± 0.5528` e `cnn3` para `CI = 0.5844 ± 0.0364 / MSE = 2.0005 ± 0.3765` — praticamente empatados com `mean_predictor` treino completo (`MSE = 2.1286 ± 0.4938`) e piores que `target_mean` (`1.8053 ± 0.4877`) e `mean_predictor` ativo (`1.1113 ± 0.0930`).
     - Em `v2_kfold` `cold_target`: `gin` (`CI = 0.5910 ± 0.0130 / MSE = 2.1854 ± 0.7256`) perde até para `mean_predictor` treino completo (`2.1214 ± 0.4043`) e `drug_mean` (`1.5835 ± 0.3764`); já `cnn3` alcança `CI = 0.6315 ± 0.0171 / MSE = 1.4846 ± 0.2814` (acima de `drug_mean` no MSE, embora acima do `mean_predictor` ativo `1.0834 ± 0.1579`).
   - Adicionalmente, nas `546` entradas duplicadas divergentes do Davis (`2.894` linhas resultantes de `81 target_id`s colapsando em `18` sequências), o piso irreduzível de ruído é `MSE = 0.3347` (`MSE = 0.0322` global no Davis, `T1`).

2. **Estratificação por comprimento da proteína (`T6`, `F6`):**
   - Em **todos os `13` grupos experimentais (`9` em `v1_seeds` e `4` em `v2_kfold`)**, o MSE médio nas proteínas com `> 1.000` aa (truncadas pelo GraphDTA) é numericamente superior ao das proteínas com `<= 1.000` aa:
     - Exemplos em `v2_kfold` (Davis): `cnn3 cold_drug` sobe de `0.6740 ± 0.1902` para `1.0083 ± 0.0811`; `cnn3 cold_target` de `0.4472 ± 0.0847` para `0.7710 ± 0.2431`; `gin cold_drug` de `0.6957 ± 0.2436` para `0.9963 ± 0.1588`; `gin cold_target` de `0.6208 ± 0.1859` para `1.0974 ± 0.6145`.
     - Exemplos em `v1_seeds` (KIBA GIN): `random` sobe de `0.1642 ± 0.0184` para `0.2189 ± 0.0264`; `cold_drug` de `0.3528 ± 0.0515` para `0.4909 ± 0.0630`; `cold_target` de `0.4948 ± 0.0482` para `0.7113 ± 0.3516`.
   - Em CI, o padrão difere por regime: em `v2_kfold cold_target` (`cnn3`: `0.7798 -> 0.7496`; `gin`: `0.7317 -> 0.6946`) e no KIBA (`random` e `cold_drug`), proteínas `> 1.000` aa apresentam CI menor; já em Davis `cold_drug` e `random`, o CI médio em `> 1.000` aa aparece numericamente ligeiramente maior com elevada dispersão entre unidades.

---

## 5. Separação Estrita: Observação vs. Hipótese

- **Observações diretas:** os valores de `CI`, `MSE`, $\Delta\text{MSE}$, $\Delta\text{CI}$, bootstrap, `best_epoch` (`2..7` em v1; `3..13` em 4/5 dobras v2 para GIN em `cold_target`), queda de desempenho em `pKd > 5.0` e MSE maior no estrato `> 1.000` aa em todos os `13` grupos.
- **Hipóteses interpretativas (não demonstradas causalmente pelos dados atuais):**
  1. Que o colapso do `GraphDTAGIN` em `cold_target` seja causado pela indexação posicional absoluta (`in_channels = max_seq_len = 1000`) de `OfficialGraphDTATargetEncoder`, a qual teria menos capacidade de abstrair motivos proteicos deslocados ao longo da sequência do que a `CNN3` com `AdaptiveMaxPool1d(1)` (hipótese mecanística consistente com o contraste `GIN vs CNN3`, mas ainda não submetida a ablação controlada de parâmetros).
  2. Que a degradação de MSE em proteínas `> 1.000` aa decorra do truncamento em `1.000` aa e não de diferenças intrínsecas na variância/distribuição de afinidade dessas quinases longas (não foi verificada a variância marginal do alvo nem treinado modelo sem truncamento).
  3. Que o desempenho próximo ao baseline marginal em `cold_drug` reflita memorização de médias por alvo aliada a baixa diversidade de fármacos em Davis (`68` compostos), e não apenas limitação da arquitetura `GIN`.
