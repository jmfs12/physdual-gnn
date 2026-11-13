# T2 — Tamanhos e Grupos Efetivos por Conteúdo (SMILES Canônico e Sequência) nos Splits

**Fonte:** data/processed/splits/*.json, data/processed/{davis,kiba}_clean.csv e docs/paper/cpu_audits.json

**Formato numérico:** Política numérica padronizada: métricas contínuas (CI, MSE, RMSE, Pearson, Spearman, rm2, AUROC, AUPR, deltas) com 4 casas decimais (0.0000); percentuais com 2 casas decimais (0.00%); médias de épocas com 1 casa (0.0) e épocas individuais como inteiros; tempo por época com 3 casas (0.000 s) e horas totais com 2 casas (0.00 h). Dispersão expressa como média ± desvio-padrão amostral (ddof=1).

| dataset | protocolo | split | unidade | treino_pares | val_pares | teste_pares | treino_smiles_canon | val_smiles_canon | teste_smiles_canon | treino_drug_ids | val_drug_ids | teste_drug_ids | treino_seq_unicas | val_seq_unicas | teste_seq_unicas | treino_target_ids | val_target_ids | teste_target_ids |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| davis | v2_kfold | cold_drug | fold_0 | 21658 | 2210 | 6188 | 49 | 5 | 14 | 49 | 5 | 14 | 379 | 379 | 379 | 442 | 442 | 442 |
| davis | v2_kfold | cold_drug | fold_1 | 21658 | 2210 | 6188 | 49 | 5 | 14 | 49 | 5 | 14 | 379 | 379 | 379 | 442 | 442 | 442 |
| davis | v2_kfold | cold_drug | fold_2 | 21658 | 2210 | 6188 | 49 | 5 | 14 | 49 | 5 | 14 | 379 | 379 | 379 | 442 | 442 | 442 |
| davis | v2_kfold | cold_drug | fold_3 | 21658 | 2652 | 5746 | 49 | 6 | 13 | 49 | 6 | 13 | 379 | 379 | 379 | 442 | 442 | 442 |
| davis | v2_kfold | cold_drug | fold_4 | 21658 | 2652 | 5746 | 49 | 6 | 13 | 49 | 6 | 13 | 379 | 379 | 379 | 442 | 442 | 442 |
| davis | v1_seeds | cold_drug | seed0 | 23868 | 3094 | 3094 | 54 | 7 | 7 | 54 | 7 | 7 | 379 | 379 | 379 | 442 | 442 | 442 |
| davis | v1_seeds | cold_drug | seed1 | 23868 | 3094 | 3094 | 54 | 7 | 7 | 54 | 7 | 7 | 379 | 379 | 379 | 442 | 442 | 442 |
| davis | v1_seeds | cold_drug | seed2 | 23868 | 3094 | 3094 | 54 | 7 | 7 | 54 | 7 | 7 | 379 | 379 | 379 | 442 | 442 | 442 |
| davis | v1_seeds | cold_drug | seed3 | 23868 | 3094 | 3094 | 54 | 7 | 7 | 54 | 7 | 7 | 379 | 379 | 379 | 442 | 442 | 442 |
| davis | v1_seeds | cold_drug | seed4 | 23868 | 3094 | 3094 | 54 | 7 | 7 | 54 | 7 | 7 | 379 | 379 | 379 | 442 | 442 | 442 |
| davis | v2_kfold | cold_target | fold_0 | 22032 | 2108 | 5916 | 68 | 68 | 68 | 68 | 68 | 68 | 273 | 30 | 76 | 324 | 31 | 87 |
| davis | v2_kfold | cold_target | fold_1 | 22576 | 2108 | 5372 | 68 | 68 | 68 | 68 | 68 | 68 | 273 | 30 | 76 | 332 | 31 | 79 |
| davis | v2_kfold | cold_target | fold_2 | 21420 | 2040 | 6596 | 68 | 68 | 68 | 68 | 68 | 68 | 273 | 30 | 76 | 315 | 30 | 97 |
| davis | v2_kfold | cold_target | fold_3 | 22100 | 2108 | 5848 | 68 | 68 | 68 | 68 | 68 | 68 | 273 | 30 | 76 | 325 | 31 | 86 |
| davis | v2_kfold | cold_target | fold_4 | 21624 | 2108 | 6324 | 68 | 68 | 68 | 68 | 68 | 68 | 274 | 30 | 75 | 318 | 31 | 93 |
| davis | v1_seeds | cold_target | seed0 | 24548 | 2856 | 2652 | 68 | 68 | 68 | 68 | 68 | 68 | 303 | 38 | 38 | 361 | 42 | 39 |
| davis | v1_seeds | cold_target | seed1 | 23664 | 2584 | 3808 | 68 | 68 | 68 | 68 | 68 | 68 | 303 | 38 | 38 | 348 | 38 | 56 |
| davis | v1_seeds | cold_target | seed2 | 23528 | 2720 | 3808 | 68 | 68 | 68 | 68 | 68 | 68 | 303 | 38 | 38 | 346 | 40 | 56 |
| davis | v1_seeds | cold_target | seed3 | 23188 | 4080 | 2788 | 68 | 68 | 68 | 68 | 68 | 68 | 303 | 38 | 38 | 341 | 60 | 41 |
| davis | v1_seeds | cold_target | seed4 | 24684 | 2652 | 2720 | 68 | 68 | 68 | 68 | 68 | 68 | 303 | 38 | 38 | 363 | 39 | 40 |
| davis | v1_seeds | random | seed0 | 24044 | 3006 | 3006 | 68 | 68 | 68 | 68 | 68 | 68 | 379 | 379 | 379 | 442 | 442 | 442 |
| davis | v1_seeds | random | seed1 | 24044 | 3006 | 3006 | 68 | 68 | 68 | 68 | 68 | 68 | 379 | 379 | 379 | 442 | 442 | 442 |
| davis | v1_seeds | random | seed2 | 24044 | 3006 | 3006 | 68 | 68 | 68 | 68 | 68 | 68 | 379 | 379 | 379 | 442 | 442 | 442 |
| davis | v1_seeds | random | seed3 | 24044 | 3006 | 3006 | 68 | 68 | 68 | 68 | 68 | 68 | 379 | 379 | 378 | 442 | 442 | 441 |
| davis | v1_seeds | random | seed4 | 24044 | 3006 | 3006 | 68 | 68 | 68 | 68 | 68 | 68 | 379 | 379 | 379 | 442 | 442 | 441 |
| kiba | v1_seeds | cold_drug | seed0 | 92760 | 11831 | 13663 | 1653 | 206 | 206 | 1693 | 207 | 211 | 229 | 222 | 227 | 229 | 222 | 227 |
| kiba | v1_seeds | cold_drug | seed1 | 95298 | 11276 | 11680 | 1653 | 206 | 206 | 1680 | 216 | 215 | 229 | 226 | 223 | 229 | 226 | 223 |
| kiba | v1_seeds | cold_drug | seed2 | 94963 | 11013 | 12278 | 1653 | 206 | 206 | 1687 | 209 | 215 | 229 | 226 | 227 | 229 | 226 | 227 |
| kiba | v1_seeds | cold_drug | seed3 | 95635 | 10730 | 11889 | 1653 | 206 | 206 | 1696 | 207 | 208 | 229 | 226 | 228 | 229 | 226 | 228 |
| kiba | v1_seeds | cold_drug | seed4 | 93946 | 10887 | 13421 | 1653 | 206 | 206 | 1679 | 214 | 218 | 229 | 223 | 224 | 229 | 223 | 224 |
| kiba | v1_seeds | cold_target | seed0 | 99622 | 10368 | 8264 | 2065 | 1938 | 1607 | 2111 | 1978 | 1626 | 183 | 23 | 23 | 183 | 23 | 23 |
| kiba | v1_seeds | cold_target | seed1 | 94665 | 11349 | 12240 | 2065 | 1993 | 1642 | 2111 | 2038 | 1663 | 183 | 23 | 23 | 183 | 23 | 23 |
| kiba | v1_seeds | cold_target | seed2 | 95100 | 11551 | 11603 | 2065 | 1844 | 2014 | 2111 | 1874 | 2059 | 183 | 23 | 23 | 183 | 23 | 23 |
| kiba | v1_seeds | cold_target | seed3 | 93342 | 10443 | 14469 | 2065 | 1966 | 2005 | 2111 | 2009 | 2050 | 183 | 23 | 23 | 183 | 23 | 23 |
| kiba | v1_seeds | cold_target | seed4 | 94818 | 12008 | 11428 | 2065 | 2008 | 1971 | 2111 | 2054 | 2010 | 183 | 23 | 23 | 183 | 23 | 23 |
| kiba | v1_seeds | random | seed0 | 94604 | 11825 | 11825 | 2065 | 1849 | 1841 | 2111 | 1886 | 1874 | 229 | 220 | 220 | 229 | 220 | 220 |
| kiba | v1_seeds | random | seed1 | 94604 | 11825 | 11825 | 2065 | 1850 | 1838 | 2111 | 1884 | 1872 | 229 | 227 | 223 | 229 | 227 | 223 |
| kiba | v1_seeds | random | seed2 | 94604 | 11825 | 11825 | 2065 | 1865 | 1860 | 2111 | 1899 | 1888 | 229 | 227 | 227 | 229 | 227 | 227 |
| kiba | v1_seeds | random | seed3 | 94604 | 11825 | 11825 | 2065 | 1869 | 1837 | 2111 | 1900 | 1869 | 229 | 227 | 222 | 229 | 227 | 222 |
| kiba | v1_seeds | random | seed4 | 94604 | 11825 | 11825 | 2065 | 1835 | 1848 | 2111 | 1876 | 1882 | 229 | 226 | 226 | 229 | 226 | 226 |

- Em `davis` (`cold_drug` e `cold_target`), o agrupamento estrito é por conteúdo (`smiles` canônico para drogas; `sequence` para alvos). Note que `379` sequências únicas correspondem a `442` `target_id`s no Davis (`81` `target_id`s colapsam em `18` sequências), enquanto no KIBA `2065` SMILES canônicos correspondem a `2111` `drug_id`s.
- Verificação de vazamento por conteúdo em todos os 30 splits cold (`v1_seeds` + `v2_kfold`): `0` sobreposições entre treino, validação e teste.
- Proposta P0 de split por identidade de sequência em `davis_cold_target` (`biopython==1.88`, `PairwiseAligner(mode='global', matrix='BLOSUM62', open_gap=-10.0, extend_gap=-0.5) exact_matches / max(len_a, len_b) + connected components + greedy LPT 5-fold balancing` em `379` sequências únicas, `67,590` alinhamentos em `117.62 s`): (i) `id >= 30%`: `120` clusters (`60` singletons, maior cluster `92` seqs / `6664` pares), 5 folds `[pares/seqs_unicas] = [6664/92, 5848/71, 5848/71, 5848/64, 5848/81]`; (ii) `id >= 40%`: `199` clusters (`109` singletons, maior cluster `11` seqs / `1088` pares), 5 folds `[5984/72, 6052/72, 6052/72, 5984/83, 5984/80]`; (iii) `id >= 50%`: `251` clusters (`173` singletons, maior cluster `10` seqs / `1088` pares), 5 folds `[6052/74, 6052/75, 5984/81, 5984/74, 5984/75]`.
