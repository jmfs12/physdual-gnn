# T1 — Estatísticas, Duplicatas e Piso de Ruído dos Datasets (Davis e KIBA)

**Fonte:** data/processed/{davis,kiba}_ingestion_report.json, data/processed/{davis,kiba}_clean.csv e docs/paper/cpu_audits.json

**Formato numérico:** Política numérica padronizada: métricas contínuas (CI, MSE, RMSE, Pearson, Spearman, rm2, AUROC, AUPR, deltas) com 4 casas decimais (0.0000); percentuais com 2 casas decimais (0.00%); médias de épocas com 1 casa (0.0) e épocas individuais como inteiros; tempo por época com 3 casas (0.000 s) e horas totais com 2 casas (0.00 h). Dispersão expressa como média ± desvio-padrão amostral (ddof=1).

| dataset | pares_brutos | pares_limpos | farmacos_id | farmacos_smiles_canonico | alvos_id | alvos_sequencia_unica | alvos_seq_compartilhadas | seqs_gt_1000aa_unicas_e_pares | seqs_janela_gt_1022aa_unicas_e_pares | smiles_alterados_rdkit | smiles_falhos | aa_estendidos_seqs | aa_estendidos_contagem | grupos_duplicados_conteudo | duplicatas_afinidade_distinta | piso_ruido_mse_global | piso_ruido_mse_linhas_distintas | pares_censurados_pkd_5_0 | pct_censurados |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DAVIS | 30056 | 30056 | 68 | 68 | 442 | 379 | 18 seqs -> 81 target_ids | 76 (7820 pares) | 70 (7412 pares) | 68 | 0 | 1 | X:1, B:0, U:0, Z:0, O:0 | 1224 | 546 | 0.0322 | 0.3347 | 20931 | 69.64% |
| KIBA | 118254 | 118254 | 2111 | 2065 | 229 | 229 | 0 | 44 (24103 pares) | 42 (22848 pares) | 2066 | 0 | 0 | X:0, B:0, U:0, Z:0, O:0 | 524 | 310 | 0.0011 | 0.1780 | 0 | 0.00% (N/A) |

- Sequências com janela deslizante ESM-2 (>1022 aa): `70` únicas (`109` `target_id`s) no Davis + `42` únicas no KIBA = `112` por soma de dataset (`83` sequências únicas globalmente deduplicadas), sincronizadas em `outputs/logs/phase1_summary_report.json`.
- No Davis, `81` `target_id`s colapsam em apenas `18` sequências únicas idênticas em `proteins.txt` (`shared_sequences_count=18`), gerando `1.224` grupos duplicados (`5.508` linhas) e `546` grupos com afinidades distintas (`2.894` linhas); o piso irreduzível de ruído (MSE da média do grupo) é `0.0322` global (`RMSE=0.1795`) e `0.3347` nas linhas divergentes — e indica que o Davis atual não testa resiliência a mutações.
- No KIBA, `524` grupos duplicados (`1.172` linhas; `310` grupos/744 linhas divergentes) resultam de `2.111` `drug_id`s que colapsam em `2.065` SMILES canônicos; piso global `MSE = 0.0011` (`0.1780` nas linhas divergentes).
- No Davis, `smiles_modified_count = 68` decorre de recanonização sintática RDKit; `num_modified_fragments = 0` em `phase1_summary_report.json` (0 sais multifragmento removidos).
