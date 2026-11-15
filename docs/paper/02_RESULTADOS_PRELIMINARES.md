# Rascunho 02 — Resultados Preliminares e Avaliação da Fase 3 (`gin_esm_mlp`, Fases 1–3)

> **Regra de integridade:** Todos os números abaixo provêm diretamente das tabelas `T1`–`T15`, figuras `F1`–`F11`, `docs/paper/phase3_audits.json`, `docs/paper/06_AUDITORIA_FASE3.md` e JSONs em `outputs/results/`. Protocolos **v1 (`v1_seeds`)** e **v2 (`v2_kfold`, incluídos `seqid40` e `seqid30`)** são reportados em seções estritamente separadas e nunca combinados em uma mesma média. Em conformidade com o **ADR #15**, o modelo avaliado na Fase 3 é denominado estritamente **`gin_esm_mlp`** (e seu controle de ligante **`ecfp_esm_mlp`**). Nenhum estudo de caso de fármaco/alvo específico é reportado nesta etapa.

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

## 5. Separação Estrita: Observação vs. Hipótese (Fases 1 e 2)

- **Observações diretas (Fases 1 e 2):** os valores de `CI`, `MSE`, $\Delta\text{MSE}$, $\Delta\text{CI}$, bootstrap, `best_epoch` (`2..7` em v1; `3..13` em 4/5 dobras v2 para GIN em `cold_target`), queda de desempenho em `pKd > 5.0` e MSE maior no estrato `> 1.000` aa em todos os `13` grupos.
- **Hipóteses interpretativas (Fases 1 e 2, não demonstradas causalmente pelos dados):**
  1. Que o colapso do `GraphDTAGIN` em `cold_target` seja causado pela indexação posicional absoluta (`in_channels = max_seq_len = 1000`) de `OfficialGraphDTATargetEncoder`, a qual teria menos capacidade de abstrair motivos proteicos deslocados ao longo da sequência do que a `CNN3` com `AdaptiveMaxPool1d(1)` (hipótese mecanística consistente com o contraste `GIN vs CNN3`, mas ainda não submetida a ablação controlada de parâmetros).
  2. Que a degradação de MSE em proteínas `> 1.000` aa decorra do truncamento em `1.000` aa e não de diferenças intrínsecas na variância/distribuição de afinidade dessas quinases longas (não foi verificada a variância marginal do alvo nem treinado modelo sem truncamento).
  3. Que o desempenho próximo ao baseline marginal em `cold_drug` reflita memorização de médias por alvo aliada a baixa diversidade de fármacos em Davis (`68` compostos), e não apenas limitação da arquitetura `GIN`.

---

## 6. Pré-Registro da Fase 3 (`ADR #15`), Auditoria e Critério de Decisão (`Passo 1` / `T9`)

A auditoria de integridade (`06_AUDITORIA_FASE3.md` e `phase3_audits.json`, chave `git_preregistration.timeline`) confirmou por histórico Git e timestamps (`mtime`) que o **ADR #15** foi congelado e tagueado (`46d13f8`, tag `phase3-adr15-pre-kiba`, `2026-11-14 00:32:14 -0300`) **antes** dos `62` runs confirmatórios e de sanidade no KIBA, `davis/random` e controles diagnósticos (`mtime` dos JSONs após `2026-11-14 00:50`). Os `19` runs exploratórios anteriores de `gin_esm_mlp` no Davis cold foram formalmente reclassificados como **exploratórios** pelo próprio ADR #15. Os `34/34` agregados em `outputs/results/` estão completos (`n_units` exato, `0` falhas de isolamento ou de tamanho de split, `0` valores `NaN/Inf`, reprodução CPU OOF com diferença `MSE <= 5.2e-8` e `CI <= 0.0026`, exceto `knn_*` em `davis/random` por empates discretos `<= 0.0050`), com ressalva explícita para `phase3_davis_paired_bootstrap.json` (`mtime 2026-11-14 00:13:05`, desatualizado antes da extensão da dobra 1 de `davis/cold_target` e excluído dos cálculos atuais).

Por força do **ADR #15**, cada regime confirmatório do KIBA exige superar **ambos** os comparadores primários (`ridge_esm` e o melhor baseline **sem ESM-2**) com **>= 4/5 unidades favoráveis** e **IC 95% de bootstrap em bloco (`n=1000, seed=42`) excluindo zero** (`T9_phase3_decision_adr15.csv`):

| Protocolo | Papel (`ADR #15`) | Dataset / Regime | `gin_esm_mlp` (`CI` / `MSE`) | Comparador 1 (`ridge_esm`): `vitorias`; `IC 95% dMSE [wins/1000]` | Comparador 2 (melhor não-ESM): `vitorias`; `IC 95% dMSE [wins/1000]` | Veredito (`ADR #15`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `v1_seeds` | Primário confirmativo | `kiba / cold_drug` | `0.7535 ± 0.0101` / `0.4112 ± 0.0481` | `MSE 5/5; CI 5/5`; `[-0.1952, -0.1039] [1000/1000]` (**atendido**) | `knn_tanimoto_drug`: `MSE 0/5; CI 0/5`; `[0.0310, 0.0886] [0/1000]` (**não atendido**) | **`nao_atendido_negativo`** |
| `v1_seeds` | Primário confirmativo | `kiba / cold_target` | `0.7252 ± 0.0244` / `0.3735 ± 0.0644` | `MSE 5/5; CI 5/5`; `[-0.1804, -0.0930] [1000/1000]` (**atendido**) | `knn_kmer_target`: `MSE 3/5; CI 1/5`; `[-0.0355, 0.0368] [562/1000]` (**não atendido**) | **`nao_atendido_negativo`** |
| `v2_kfold` | Secundário confirmativo | `kiba / cold_target_seqid40` | `0.6774 ± 0.0313` / `0.5072 ± 0.1479` | `MSE 4/5; CI 5/5`; `[-0.0912, 0.0474] [852/1000]` (**não atendido**) | `knn_tanimoto_drug` (`=drug_mean`): `MSE 1/5; CI 0/5`; `[-0.0059, 0.1367] [44/1000]` (**não atendido**) | **`nao_atendido_negativo`** |
| `v2_kfold` | Secundário confirmativo | `kiba / cold_target_seqid30` | `0.6736 ± 0.0241` / `0.5069 ± 0.1479` | `MSE 5/5; CI 5/5`; `[-0.1705, -0.0425] [1000/1000]` (**atendido**) | `knn_tanimoto_drug` (`=drug_mean`): `MSE 2/5; CI 1/5`; `[-0.0319, 0.0543] [233/1000]` (**não atendido**) | **`nao_atendido_negativo`** |

**Conclusão confirmatória da Fase 3 (`T9`, `phase3_audits.json`):** em **todos os `4/4` regimes confirmatórios de generalização no KIBA** (`cold_drug`, `cold_target`, `cold_target_seqid40` e `cold_target_seqid30`), `gin_esm_mlp` recebeu veredito **`nao_atendido_negativo`**.

---

## 7. Resultados no KIBA — Protocolos Separados (`T10`, `T11`, `T12`, `T13`, `F7`, `F8`)

### 7.1. Protocolo `v1_seeds` no KIBA (`5` seeds de split; sem `graphdta_cnn3` disponível)

- **`kiba / cold_drug` (`T10`, `T11`, `T13`; `836` clusters únicos de fármaco, `206` por seed):**
  - `gin_esm_mlp`: `CI = 0.7535 ± 0.0101`, `MSE = 0.4112 ± 0.0481`, `R = 0.6536 ± 0.0381`, `rm2 = 0.3584 ± 0.0564`.
  - Comparação com baselines clássicos e `ridge_esm`: supera `ridge_esm` (`0.6817 ± 0.0087 / 0.5586 ± 0.0705`; $\Delta\text{MSE} = -0.1474 \pm 0.0509$, `5/5`), `knn_kmer_target` (`= target_mean_predictor`: `0.6869 ± 0.0048 / 0.5732 ± 0.0585`; `5/5`) e `drug_mean_predictor` (`0.5000 / 0.6739 ± 0.0635`). Contudo, é **superado pelo kNN puramente químico `knn_tanimoto_drug`** (`CI = 0.7831 ± 0.0083`, `MSE = 0.3531 ± 0.0432`, `rm2 = 0.4374 ± 0.0327`), com $\Delta\text{MSE} = +0.0581 \pm 0.0202$ (`0/5` em MSE e `0/5` em CI; bootstrap $\Delta\text{CI} \in [-0.0422, -0.0190]$, `0/1000`).
  - Comparação exploratória com `graphdta_gin` (`0.7560 ± 0.0107 / 0.3808 ± 0.0532`, `rm2 = 0.4082 ± 0.0297`): `gin_esm_mlp` perde em MSE em `0/5` seeds ($\Delta\text{MSE} = +0.0305 \pm 0.0363$) e empata em CI (`3/5`, $\Delta\text{CI} = -0.0025 \pm 0.0130`). `graphdta_cnn3` não foi executado no KIBA (`não disponível`).
- **`kiba / cold_target` (`T10`, `T13`; `93` clusters únicos de sequência no agregado, apenas `23` por seed):**
  - `gin_esm_mlp`: `CI = 0.7252 ± 0.0244`, `MSE = 0.3735 ± 0.0644`, `R = 0.6755 ± 0.0684`, `rm2 = 0.4249 ± 0.0999`.
  - Supera `ridge_esm` (`0.6818 ± 0.0247 / 0.5056 ± 0.0766`, `5/5`), `drug_mean_predictor` (`= knn_tanimoto_drug`: `0.6964 ± 0.0122 / 0.4465 ± 0.0843`, `4/5` em MSE, `5/5` em CI) e `graphdta_gin` (`0.6579 ± 0.0437 / 0.5137 ± 0.0888`, `5/5`, embora `graphdta_gin` sofra colapso conhecido de codificação absoluta; `graphdta_cnn3` não existe para servir de baseline não-ESM neural forte no KIBA).
  - Frente ao melhor não-ESM medido no regime, **`knn_kmer_target`** (`CI = 0.7320 ± 0.0115`, `MSE = 0.3730 ± 0.0351`, `rm2 = 0.4102 ± 0.0925`), não há ganho: $\Delta\text{MSE} = +0.0005 \pm 0.0390$ (`3/5` MSE, `1/5` CI, `562/1000` fav. em MSE; $\Delta\text{CI} = -0.0068 \pm 0.0164$, `243/1000` fav. em CI).
- **`kiba / random` (sanidade `v1_seeds`, `T10`, `T13`):**
  - `gin_esm_mlp` atinge `CI = 0.8749 ± 0.0021`, `MSE = 0.1536 ± 0.0095`, `rm2 = 0.7579 ± 0.0187`, superando `graphdta_gin` (`0.8578 ± 0.0134 / 0.1753 ± 0.0190`; `4/5` seeds, $\Delta\text{MSE} = -0.0218 \pm 0.0150$, $\Delta\text{CI} = +0.0171 \pm 0.0124$), `knn_tanimoto_drug` (`0.7900 / 0.3442`, `5/5`) e `ridge_esm` (`0.7547 / 0.4079`, `5/5`). Nota de equivalência: no regime interpolativo `random` ambos os inputs já estão vistos no treino, não constituindo evidência de generalização.

### 7.2. Protocolo `v2_kfold` no KIBA com Identidade de Sequência (`seqid40` e `seqid30`, `T10`–`T13`, `F7`, `F8`)

- **Observação de identidade algébrica dos baselines:** em splits `cold_target` (incluindo `seqid40` e `seqid30`), `knn_tanimoto_drug` não dispõe de pares do alvo teste no treino e reduz-se à média por fármaco, apresentando valores idênticos a `drug_mean_predictor` (`T10` linhas `55–56`, `61–62`).
- **`kiba / cold_target_seqid40` (`N_OOF = 118.254`, `117` clusters):** `gin_esm_mlp` obtém `CI = 0.6774 ± 0.0313`, `MSE = 0.5072 ± 0.1479`, `rm2 = 0.2863 ± 0.1375` (OOF `0.6732 / 0.5072`, `rm2 = 0.2549`). Perde numericamente em MSE e CI para **`knn_tanimoto_drug` / `drug_mean_predictor`** (`0.6986 ± 0.0239 / 0.4553 ± 0.1013`; $\Delta\text{MSE} = +0.0519 \pm 0.0725$, `1/5` MSE, `0/5` CI; `44/1000` e `10/1000` fav.) e também em CI para `knn_kmer_target` (`0.6995 ± 0.0311 / 0.4973 ± 0.0937`; `0/5` CI, `6/1000` fav.). Frente a `ridge_esm` (`0.6542 ± 0.0327 / 0.5419 ± 0.0981`), o IC bootstrap em MSE (`[-0.0912, 0.0474]`, `852/1000`) e em CI (`[-0.0081, 0.0295]`, `891/1000`) **cruza zero** porque na dobra 0 `gin_esm_mlp` tem MSE pior que `ridge_esm` (`0.7145` vs `0.5987`, `T12`).
- **`kiba / cold_target_seqid30` (`N_OOF = 118.254`, `75` clusters):** `gin_esm_mlp` obtém `CI = 0.6736 ± 0.0241`, `MSE = 0.5069 ± 0.1479`, `rm2 = 0.2659 ± 0.0602` (OOF `0.6649 / 0.5078`, `rm2 = 0.2549`). Embora supere `ridge_esm` (`0.6201 ± 0.0365 / 0.6097 ± 0.1555`, `5/5`, IC MSE `[-0.1705, -0.0425]`, `1000/1000`), isso decorre em parte da degradação do modelo linear regularizado abaixo da média por fármaco (`0.6097` vs `0.4929`). Frente a **`knn_tanimoto_drug` / `drug_mean_predictor`** (`0.6880 ± 0.0193 / 0.4929 ± 0.1595`; OOF `0.6845 / 0.4892`), `gin_esm_mlp` continua numericamente pior ($\Delta\text{MSE} = +0.0141 \pm 0.0283$, `2/5` MSE, `1/5` CI; bootstrap MSE `[-0.0319, 0.0543]`, `233/1000`). Comparações com GraphDTA no KIBA `seqid30/40` não estão disponíveis.

---

## 8. Resultados no Davis — Sanidade `random` (`v1_seeds`) e Regime Exploratório Cold (`v2_kfold`, `T9`–`T13`, `F7`, `F8`)

### 8.1. `davis / random` (`v1_seeds`, sanidade não confirmativa)

- `gin_esm_mlp` alcança `CI = 0.9057 ± 0.0063`, `MSE = 0.1921 ± 0.0143`, `R = 0.8754 ± 0.0105`, `rm2 = 0.7535 ± 0.0211`, superando em `5/5` seeds `graphdta_cnn3` (`0.8895 ± 0.0035 / 0.2216 ± 0.0128`; $\Delta\text{MSE} = -0.0295 \pm 0.0078$), `graphdta_gin` (`0.8845 / 0.2348`, $\Delta\text{MSE} = -0.0427 \pm 0.0096`), `knn_kmer_target` (`0.8481 / 0.3408`) e `ridge_esm` (`0.7958 / 0.5519`). Nos subconjuntos derivados (`T11`): não censurados (`pKd > 5.0`) com `CI = 0.7732 ± 0.0049 / MSE = 0.5076 ± 0.0325`; classificação binária `> 5` com `AUROC = 0.9346 ± 0.0076 / AUPR = 0.8910 ± 0.0091` e `>= 7` com `AUROC = 0.9738 ± 0.0028 / AUPR = 0.8183 ± 0.0290` (`limiar binario a verificar`). Ressalva crítica (`phase3_audits.json`): o Davis possui `81` IDs de alvo colapsados em apenas `18` sequências idênticas de mutantes/variantes, inflando o desempenho em `random` para arquiteturas profundas e `knn_kmer_target` sem refletir novidade de proteína.

### 8.2. Curva Exploratória de Generalização em Davis (`v2_kfold`, pós-ADR #15, `N_OOF = 30.056`)

- **`davis / cold_target` (`379` clusters; dobra 1 estendida para `1000` épocas com `best_epoch=525`):** `gin_esm_mlp` atinge `CI = 0.8195 ± 0.0272`, `MSE = 0.4319 ± 0.1378`, `rm2 = 0.4396 ± 0.0675` (OOF `0.8149 / 0.4372`; não censurados `N=9125`: `0.6727 / 1.1577`; `AUROC >5 / >=7 = 0.8457 / 0.8948`). Supera `ridge_esm` (`0.7649 / 0.5808`; `5/5`, `1000/1000`), `graphdta_cnn3` (`0.7740 / 0.5350`; `5/5` MSE, `4/5` CI; `1000/1000`) e `graphdta_gin` (`0.7117 / 0.7524`; `5/5`), mas frente a **`knn_kmer_target`** (`0.8143 ± 0.0218 / 0.4572 ± 0.1379`, OOF `0.8127 / 0.4621`) o IC 95% do bootstrap **cruza zero** tanto em MSE (`[-0.0618, 0.0071]`, `922/1000`) quanto em CI (`[-0.0059, 0.0111]`, `707/1000`).
- **Queda progressiva com filtro de identidade de sequência (`F7`):**
  - `davis / cold_target_seqid40` (`199` clusters): `gin_esm_mlp` cai para `CI = 0.7754 ± 0.0238`, `MSE = 0.5627 ± 0.0844`, `rm2 = 0.2825 ± 0.0923` (OOF `0.7744 / 0.5629`). Supera `graphdta_cnn3` (`0.7475 / 0.6386`, IC MSE `[-0.1517, -0.0083]`, `987/1000`, porém só `3/5` em MSE) e `ridge_esm` (`0.7471 / 0.6138`, `3/5` MSE, `986/1000`), mas fica **inconclusivo/negativo** frente a `knn_tanimoto_drug` (`= drug_mean`: `0.7443 / 0.6303`, `3/5` MSE, IC MSE `[-0.1484, 0.0047]`, `968/1000`).
  - `davis / cold_target_seqid30` (`120` clusters): `gin_esm_mlp` cai para `CI = 0.7436 ± 0.0335`, `MSE = 0.6414 ± 0.1544`, `rm2 = 0.2087 ± 0.1104` (OOF `0.7407 / 0.6355`), perdendo sua vantagem estatística sobre **todos** os principais comparadores exceto `knn_kmer_target` (`120` clusters com homologia `<=30%` degradam k-mers: `0.7051 / 0.7464`, `995/1000`) e `graphdta_gin`: vs `ridge_esm` (`0.7399 / 0.6448`, `2/5` MSE, `582/1000`), vs `knn_tanimoto_drug`/`drug_mean` (`0.7360 / 0.6565`, `2/5` MSE, `642/1000`) e vs `graphdta_cnn3` (`0.7283 / 0.7253`, `3/5` MSE, `926/1000` com IC `[-0.2140, 0.0168]` cruzando zero).
- **`davis / cold_drug` (`68` clusters; `13–14` fármacos teste/dobra):** `gin_esm_mlp` obtém `CI = 0.6607 ± 0.0752`, `MSE = 0.8467 ± 0.3258`, `rm2 = 0.1206 ± 0.0613` (OOF `0.6451 / 0.8462`). Fica **numericamente atrás em MSE e CI** de `knn_tanimoto_drug` (`0.7065 / 0.7580`), `knn_kmer_target` (`= target_mean`: `0.6690 / 0.7310`, `1/5`), `ridge_esm` (`0.6698 / 0.7566`, `2/5`), `graphdta_cnn3` (`0.6572 / 0.7610`, `1/5` MSE) e `graphdta_gin` (`0.6623 / 0.7739`, `2/5` MSE), e acima de `drug_mean` apenas em CI (`+0.1607`, mas $\Delta\text{MSE} = +0.0364$, `261/1000`).

---

## 9. Bootstrap Pareado em Bloco (`Passo 4` — `seed=42, n=1000`, `T13`, `F8`)

Todas as inferências pareadas foram recalculadas em CPU via `scripts/paper/phase3_audits.py` usando amostragem em bloco com reposição (`seed=42`, `n=1000` réplicas): por **sequência única** nos splits `cold_target*` (`379` em Davis padrão, `199` em `seqid40`, `120` em `seqid30`; `93` agrupados no KIBA `cold_target`, `117` em `kiba/seqid40`, `75` em `kiba/seqid30`) e por **SMILES canônico único** nos splits `cold_drug` (`68` em Davis, `836` agrupados no KIBA). Em comparação com o arquivo antigo em disco, no KIBA `cold_target` e `cold_drug` há apenas `23` e `206` grupos por seed (concentrando incerteza no sorteio de `23` alvos/seed), e os intervalos de confiança da Tabela `T13` são reportados acompanhados exatamente da fração de réplicas favoráveis (`w/1000`), sem `p-valores` extremos truncados.

---

## 10. Controles Diagnósticos em Davis (`Passo 5` — `T14`, `F9`, `F10`, Não Confirmatórios)

Por regra do **ADR #15**, os controles diagnósticos não são usados para ajuste de hiperparâmetros nem para decisão confirmatória, mas testam hipóteses mecanísticas diretamente (`T14_phase3_diagnostic_controls.csv`, `T12`, `phase3_audits.json`):

1. **Permutação dos embeddings ESM-2 (`--shuffle-esm`, `davis / cold_target`, seeds `0, 1, 2`):**
   - `gin_esm_mlp` sem permutação obtém `CI = 0.8195 ± 0.0272` e `MSE = 0.4319 ± 0.1378`. Ao permutar o mapa proteína $\rightarrow$ ESM-2 (`shuf0`, `shuf1`, `shuf2`), o desempenho cai para `0.7251 ± 0.0186 / 0.7030 ± 0.1291`, `0.7146 ± 0.0310 / 0.6939 ± 0.1576` e `0.7158 ± 0.0162 / 0.7146 ± 0.1453`, respectivamente (`boot dMSE vs gin` entre `+0.1907` e `+0.3721`, `0/1000` fav. para os shuffled; `boot dCI vs gin` entre `-0.1257` e `-0.0684`, `0/1000`).
   - Crucialmente, os três controles shuffled também ficam **piores do que `drug_mean_predictor`** (`0.7488 ± 0.0150 / 0.6253 ± 0.1263`), com ICs bootstrap `dMSE vs drug_mean` excluindo zero (`[0.0468, 0.1196]`, `[0.0361, 0.1049]` e `[0.0616, 0.1250]`). Isso demonstra que o alinhamento real proteína–embedding sustenta o ganho de `gin_esm_mlp` acima da média por fármaco no `davis/cold_target` permissivo, ao passo que a memorização de embeddings desalinhados degrada a generalização até abaixo do preditor marginal.
2. **Substituição da GNN do ligante por fingerprint molecular (`ecfp_esm_mlp`):**
   - Em **`davis / cold_target`**, `ecfp_esm_mlp` (`CI = 0.8175 ± 0.0262`, `MSE = 0.4221 ± 0.1237`) é **estatisticamente indistinguível** de `gin_esm_mlp` (`0.8195 ± 0.0272 / 0.4319 ± 0.1378`): `boot dMSE [ecfp - gin] = [-0.0418, 0.0088]`, `dCI = [-0.0124, 0.0086]`.
   - Em **`davis / cold_drug`**, `ecfp_esm_mlp` (`CI = 0.6633 ± 0.0682`, `MSE = 0.8188 ± 0.1588`) também empata no agregado com `gin_esm_mlp` (`0.6607 ± 0.0752 / 0.8467 ± 0.3258`; `boot dMSE = [-0.1440, 0.1171]`, `dCI = [-0.0247, 0.0309]`), porém exibe estabilidade muito maior na **dobra 2**: `ecfp_esm_mlp` atinge `CI = 0.6696 / MSE = 1.0419` contra `0.5720 / 1.3670` de `gin_esm_mlp` (`T12`).
   - Implicação direta para a hipótese arquitetural: dentro desta arquitetura com MLP e `ESM-2` em *mean pooling*, o codificador `GIN` de ligante não agrega ganho mensurável sobre `ECFP4`.
3. **Sensibilidade à inicialização / semente de treinamento (`--train-seed-offset 1000`, `tsoff1000`):**
   - Na **dobra 2 de `davis / cold_drug`** (`T12`, `phase3_audits.json` chave `davis_cold_drug_fold2_tsoff1000`), alterar apenas a semente de inicialização altera drasticamente o resultado: de `CI = 0.5720 / MSE = 1.3670` (`best_epoch = 5`) para `CI = 0.7090 / MSE = 1.0765` (`best_epoch = 31`, `total_epochs_run = 111`, `train_val_mse_gap_at_best = 0.2738`), com análise descritiva *post-hoc* sem a dobra 2 (`4` dobras restantes, não confirmatória) mostrando média `0.6829 / 0.7166` e mediana `0.6518 / 0.7109`.
   - Já no regime completo `davis / cold_target_seqid30` (`5` dobras, `n=120` clusters), `gin_esm_mlp_tsoff1000` (`CI = 0.7475 ± 0.0194`, `MSE = 0.6482 ± 0.1450`) reproduz o original (`0.7436 ± 0.0335 / 0.6414 ± 0.1544`; `boot dMSE = [-0.0355, 0.0364]`), indicando estabilidade na média de `seqid30` em contraste com a alta variância por dobra em `davis/cold_drug` (`13–14` fármacos).

---

## 11. Diagnósticos de Treinamento e Custo Computacional (`Passo 6` — `T7`, `T15`, `F11`)

- **Aderência ao teto de épocas:** dos `81` runs da Fase 3, **`0/81` ultrapassaram `0.8 * epochs_cfg`** (`over_80_pct_runs_physdual = []`), indicando convergência por *early stopping* antes do teto após a ampliação `500 -> 1000` (incluindo a dobra 1 de `davis/cold_target` estendida para `epochs=1000`, que convergiu em `best_epoch=525`, parada na `605`).
- **Memorização precoce (`best_epoch < 15` em regimes cold, `T7`):** ocorreu em **`10/81` runs da Fase 3** (`0/30` runs confirmatórios do KIBA!). Dos `10` casos em Davis, `7` são controles com `ESM` embaralhado (`shuf0` dobras `0,2,3,4` em `ep 3..10`; `shuf1` dobra `0` em `ep 13`; `shuf2` dobras `0,4` em `ep 10,11`), `2` são `ecfp_esm_mlp` em `davis/cold_drug` (dobras `3,4` em `ep=3`) e `1` é `gin_esm_mlp` na dobra 2 de `davis/cold_drug` (`ep=5`). Para comparação retrospectiva, `19/85` runs `cold` do `GraphDTAGIN` em Davis apresentam `best_epoch < 15`.
- **Dinâmica por regime (`T7`):** nos regimes confirmatórios de KIBA, `best_epoch` ocorre tardiamente (`159..234` em `cold_drug`, `162..436` em `cold_target`, `54..220` em `seqid40`, `75..168` em `seqid30`, e `401..792` em `random`), com `train_val_mse_gap_at_best` médio moderado (`+0.153` em `kiba/cold_drug`, `+0.080` em `kiba/cold_target`, `-0.021` em `seqid40`, `-0.110` em `seqid30`). Em Davis `cold_drug`, o gap médio chega a `+0.194`, e nos controles `shuf0/1/2` cai para `-0.251..-0.146` (validação sistematicamente muito pior que o treino sem sinal proteico).
- **Custo computacional (`T15`):** em L4 Colab, `gin_esm_mlp` executou a `~5.65–6.14 s/época` no KIBA (`cold_drug 2.20 h`, `cold_target 2.83 h`, `seqid40 1.63 h`, `seqid30 1.32 h`, `random 5.28 h` totais por grupo de `5` runs) e `~1.25–1.61 s/época` no Davis (`cold_target` limpo `1.251 ± 0.251 s/época` em `4` dobras, total `0.44 h`, excluindo `0.260 s/época` da dobra 1 retomada do checkpoint).

---

## 12. Síntese Científica da Fase 3: Observação vs. Hipótese

- **Observações diretas:**
  1. Sob o pré-registro **ADR #15**, `gin_esm_mlp` obteve veredito **`nao_atendido_negativo` em `4/4` regimes cold do KIBA** (`cold_drug`, `cold_target`, `seqid40`, `seqid30`): embora supere `ridge_esm` na maioria dos cenários (`3/4`, falhando apenas em `kiba/seqid40`), **não superou o melhor baseline sem ESM-2 (`knn_tanimoto_drug` em `3/4` e `knn_kmer_target` em `cold_target`)** em nenhum regime confirmatório do KIBA (`T9`, `T13`).
  2. Em Davis exploratório, `gin_esm_mlp` supera `GraphDTAGIN`, `GraphDTACNN3` e `ridge_esm` em `davis/cold_target` e `davis/random`, porém **não supera `knn_kmer_target` em `davis/cold_target`** (`IC 95%` cruza zero), perde vantagem em `davis/cold_target_seqid40` frente a `knn_tanimoto_drug`/`drug_mean`, empata com `ridge_esm` e `drug_mean` em `davis/cold_target_seqid30`, e tem desempenho inferior a `knn_tanimoto_drug`, `knn_kmer_target`, `ridge_esm` e `GraphDTACNN3` em MSE médio no `davis/cold_drug`.
  3. Permutar `ESM-2` (`--shuffle-esm`) reduz `davis/cold_target` para abaixo de `drug_mean_predictor` (`boot IC` exclui zero), confirmando dependência real do alinhamento proteína–embedding nesse regime; substituir `GIN` por `ECFP4` (`ecfp_esm_mlp`) produz resultados estatisticamente indistinguíveis de `gin_esm_mlp` em `davis/cold_target` e `davis/cold_drug` (com menos sensibilidade na dobra 2).
- **Hipóteses interpretativas para a Fase 4 (não demonstradas causalmente pelos dados atuais):**
  1. Que a incapacidade de `gin_esm_mlp` superar baselines baseados apenas na média por fármaco (`drug_mean` / `knn_tanimoto_drug`) ou em k-mers sob baixa homologia (`seqid30/40`) decorra da **compressão de toda a sequência proteica em um único vetor por *mean pooling* (`1280d`)**, que diluiria sinais locais do sítio de ligação que uma arquitetura com **embeddings de resíduos (`L x 1280`) + atenção cruzada ligante–resíduo** poderia capturar.
  2. Que o colapso linear de `ridge_esm` em `kiba/seqid30` (`MSE = 0.6097`, pior que `drug_mean = 0.4929`) reflita sensibilidade à regularização $L_2$ fixa sob mudança severa de covariável proteica (`<= 30%` de identidade).
  3. Que a equivalência `gin_esm_mlp ≈ ecfp_esm_mlp` indique que, sem atenção espacial/por subestrutura ou pré-treino estrutural, um MLP sobre `GIN` + `ESM mean-pool` explora principalmente identidade global de fármaco/alvo e estatísticas marginais já capturadas por fingerprints `ECFP4` e similaridade de Tanimoto (`knn_tanimoto_drug` em KIBA).
