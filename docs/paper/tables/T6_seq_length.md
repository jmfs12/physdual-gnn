# T6 — Desempenho Estratificado por Comprimento da Sequência Proteica (<=1000 vs. >1000 aa)

**Fonte:** outputs/results/baseline_*.json

**Formato numérico:** Política numérica padronizada: métricas contínuas (CI, MSE, RMSE, Pearson, Spearman, rm2, AUROC, AUPR, deltas) com 4 casas decimais (0.0000); percentuais com 2 casas decimais (0.00%); médias de épocas com 1 casa (0.0) e épocas individuais como inteiros; tempo por época com 3 casas (0.000 s) e horas totais com 2 casas (0.00 h). Dispersão expressa como média ± desvio-padrão amostral (ddof=1).

| protocolo | modelo | dataset | split | faixa_comprimento | n_pares_medio_teste | ci | mse | fonte |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v1_seeds | graphdta_cnn3 | davis | cold_drug | <= 1000 aa | 2289.0 ± 0.0 | 0.6831 ± 0.0346 | 0.4734 ± 0.1723 | outputs/results/baseline_graphdta_cnn3_davis_cold_drug.json |
| v1_seeds | graphdta_cnn3 | davis | cold_drug | > 1000 aa | 805.0 ± 0.0 | 0.7018 ± 0.0540 | 0.6404 ± 0.2083 | outputs/results/baseline_graphdta_cnn3_davis_cold_drug.json |
| v1_seeds | graphdta_cnn3 | davis | cold_target | <= 1000 aa | 2216.8 ± 156.5 | 0.7671 ± 0.0585 | 0.5030 ± 0.2496 | outputs/results/baseline_graphdta_cnn3_davis_cold_target.json |
| v1_seeds | graphdta_cnn3 | davis | cold_target | > 1000 aa | 938.4 ± 587.7 | 0.7660 ± 0.0600 | 0.7262 ± 0.3169 | outputs/results/baseline_graphdta_cnn3_davis_cold_target.json |
| v1_seeds | graphdta_cnn3 | davis | random | <= 1000 aa | 2243.6 ± 28.2 | 0.8825 ± 0.0019 | 0.2189 ± 0.0174 | outputs/results/baseline_graphdta_cnn3_davis_random.json |
| v1_seeds | graphdta_cnn3 | davis | random | > 1000 aa | 762.4 ± 28.2 | 0.9064 ± 0.0118 | 0.2299 ± 0.0286 | outputs/results/baseline_graphdta_cnn3_davis_random.json |
| v1_seeds | graphdta_gin | davis | cold_drug | <= 1000 aa | 2289.0 ± 0.0 | 0.6795 ± 0.0387 | 0.4820 ± 0.1835 | outputs/results/baseline_graphdta_gin_davis_cold_drug.json |
| v1_seeds | graphdta_gin | davis | cold_drug | > 1000 aa | 805.0 ± 0.0 | 0.7168 ± 0.0477 | 0.6361 ± 0.1513 | outputs/results/baseline_graphdta_gin_davis_cold_drug.json |
| v1_seeds | graphdta_gin | davis | cold_target | <= 1000 aa | 2216.8 ± 156.5 | 0.7413 ± 0.0355 | 0.6466 ± 0.3087 | outputs/results/baseline_graphdta_gin_davis_cold_target.json |
| v1_seeds | graphdta_gin | davis | cold_target | > 1000 aa | 938.4 ± 587.7 | 0.7233 ± 0.0521 | 1.0049 ± 0.5239 | outputs/results/baseline_graphdta_gin_davis_cold_target.json |
| v1_seeds | graphdta_gin | davis | random | <= 1000 aa | 2243.6 ± 28.2 | 0.8793 ± 0.0058 | 0.2317 ± 0.0201 | outputs/results/baseline_graphdta_gin_davis_random.json |
| v1_seeds | graphdta_gin | davis | random | > 1000 aa | 762.4 ± 28.2 | 0.8983 ± 0.0084 | 0.2442 ± 0.0235 | outputs/results/baseline_graphdta_gin_davis_random.json |
| v1_seeds | graphdta_gin | kiba | cold_drug | <= 1000 aa | 10040.8 ± 731.7 | 0.7641 ± 0.0108 | 0.3528 ± 0.0515 | outputs/results/baseline_graphdta_gin_kiba_cold_drug.json |
| v1_seeds | graphdta_gin | kiba | cold_drug | > 1000 aa | 2545.4 ± 179.4 | 0.7165 ± 0.0128 | 0.4909 ± 0.0630 | outputs/results/baseline_graphdta_gin_kiba_cold_drug.json |
| v1_seeds | graphdta_gin | kiba | cold_target | <= 1000 aa | 9806.0 ± 2279.0 | 0.6589 ± 0.0468 | 0.4948 ± 0.0482 | outputs/results/baseline_graphdta_gin_kiba_cold_target.json |
| v1_seeds | graphdta_gin | kiba | cold_target | > 1000 aa | 1794.8 ± 1146.8 | 0.6887 ± 0.0554 | 0.7113 ± 0.3516 | outputs/results/baseline_graphdta_gin_kiba_cold_target.json |
| v1_seeds | graphdta_gin | kiba | random | <= 1000 aa | 9411.0 ± 53.6 | 0.8624 ± 0.0134 | 0.1642 ± 0.0184 | outputs/results/baseline_graphdta_gin_kiba_random.json |
| v1_seeds | graphdta_gin | kiba | random | > 1000 aa | 2414.0 ± 53.6 | 0.8362 ± 0.0155 | 0.2189 ± 0.0264 | outputs/results/baseline_graphdta_gin_kiba_random.json |
| v2_kfold | graphdta_cnn3 | davis | cold_drug | <= 1000 aa | 4447.2 ± 179.1 (total=22236) | 0.6510 ± 0.0877 | 0.6740 ± 0.1902 | outputs/results/baseline_graphdta_cnn3_davis_cold_drug_kfold.json (per_fold_results[*].test_metrics_by_seq_len) |
| v2_kfold | graphdta_cnn3 | davis | cold_drug | > 1000 aa | 1564.0 ± 63.0 (total=7820) | 0.6605 ± 0.0617 | 1.0083 ± 0.0811 | outputs/results/baseline_graphdta_cnn3_davis_cold_drug_kfold.json (per_fold_results[*].test_metrics_by_seq_len) |
| v2_kfold | graphdta_cnn3 | davis | cold_target | <= 1000 aa | 4447.2 ± 294.8 (total=22236) | 0.7798 ± 0.0162 | 0.4472 ± 0.0847 | outputs/results/baseline_graphdta_cnn3_davis_cold_target_kfold.json (per_fold_results[*].test_metrics_by_seq_len) |
| v2_kfold | graphdta_cnn3 | davis | cold_target | > 1000 aa | 1564.0 ± 393.6 (total=7820) | 0.7496 ± 0.0646 | 0.7710 ± 0.2431 | outputs/results/baseline_graphdta_cnn3_davis_cold_target_kfold.json (per_fold_results[*].test_metrics_by_seq_len) |
| v2_kfold | graphdta_gin | davis | cold_drug | <= 1000 aa | 4447.2 ± 179.1 (total=22236) | 0.6541 ± 0.0672 | 0.6957 ± 0.2436 | outputs/results/baseline_graphdta_gin_davis_cold_drug_kfold.json (per_fold_results[*].test_metrics_by_seq_len) |
| v2_kfold | graphdta_gin | davis | cold_drug | > 1000 aa | 1564.0 ± 63.0 (total=7820) | 0.6694 ± 0.0462 | 0.9963 ± 0.1588 | outputs/results/baseline_graphdta_gin_davis_cold_drug_kfold.json (per_fold_results[*].test_metrics_by_seq_len) |
| v2_kfold | graphdta_gin | davis | cold_target | <= 1000 aa | 4447.2 ± 294.8 (total=22236) | 0.7317 ± 0.0108 | 0.6208 ± 0.1859 | outputs/results/baseline_graphdta_gin_davis_cold_target_kfold.json (per_fold_results[*].test_metrics_by_seq_len) |
| v2_kfold | graphdta_gin | davis | cold_target | > 1000 aa | 1564.0 ± 393.6 (total=7820) | 0.6946 ± 0.0417 | 1.0974 ± 0.6145 | outputs/results/baseline_graphdta_gin_davis_cold_target_kfold.json (per_fold_results[*].test_metrics_by_seq_len) |

- Total de sequências únicas no dataset completo (T1): Davis = `303` (`<=1000` aa) vs. `76` (`>1000` aa); KIBA = `185` (`<=1000` aa) vs. `44` (`>1000` aa).
