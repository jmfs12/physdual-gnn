# Fase 3 — Dual-Encoder Enxuto (`gin_esm_mlp`: `GIN + ESM-2 Congelado + MLP`)

> **Referências Arquiteturais:** [docs/DECISOES.md](DECISOES.md) (**ADR #14** e **ADR #15**) e [docs/superpowers/plans/2026-11-13-phase3-gin-esm2-mlp.md](superpowers/plans/2026-11-13-phase3-gin-esm2-mlp.md).
> **Regra de Nomenclatura (ADR #15):** Nas tabelas, textos e análises da Fase 3, o modelo atual (sem módulo físico e sem atenção cruzada) é referido exclusivamente como **`gin_esm_mlp`** (identificador técnico em JSONs: `physdual_gin_esm_mlp`). O nome **"PhysDual"** fica reservado para a arquitetura completa das Fases 4–6.

---

## 1. Arquitetura Congelada (`gin_esm_mlp`) e Orçamento de Ajuste

- **Ramo Molecular (`GINEncoder` + `LayerNorm(128)`):** 5 camadas `GINConv` (`hidden_dim=32`, `drug_in_dim=78` L1-normalizadas, `global_add_pool`, `dropout=0.2`).
- **Ramo Proteico (`ESMTargetEncoder`):** Embeddings offline `fp16` do `esm2_t33_650M_UR50D` (`1280d`, janela deslizante `1022/256`, convertidos para `fp32` com `weights_only=True`), `mean_pool` sem truncamento, `LayerNorm(1280) -> Dropout(0.4) -> Linear(1280, 128) -> GELU -> LayerNorm(128) -> Dropout(0.3)`.
- **Cabeça de Fusão (`MLP`):** Concatenação `[z_drug_norm ; z_prot_norm]` (`256d`) $\to$ `Linear(256, 512) -> ReLU -> Dropout(0.3) -> Linear(512, 256) -> ReLU -> Dropout(0.3) -> Linear(256, 1)`.
- **Orçamento de Ajuste e Auditoria (`git log` + `outputs/results/runs/`):**
  - **Configurações tentadas:** **1 única configuração fixa** introduzida no commit `1ae2853` em [`configs/base.yaml`](../configs/base.yaml) (`physdual_phase3`: `AdamW`, `lr=5e-4`, `weight_decay=1e-3`, `batch_size=512`, `esm_dropout_in=0.4`, `prot_dropout=0.3`, `prot_output_dim=128`, `mlp_hidden_dims=[512, 256]`, `mlp_dropout=0.3`), com alteração única do teto padrão `epochs: 500 -> 1000` no commit `06d1249` (preservada em todos os `20` JSONs de `outputs/results/runs/physdual_gin_esm_mlp_davis_*.json`). **Zero busca em grade** ou ajuste no conjunto de teste.
  - **Critério de seleção:** Seleção de checkpoint e *early stopping* (`patience=80` no Davis cold; `patience=50` no KIBA e `random`) feitos exclusivamente por **`val_mse` na validação interna** (`test_isolation_verified=true` em 20/20 runs).
  - **Esforço de ajuste dos baselines:** `graphdta_gin` e `graphdta_cnn3` seguiram a configuração fixa do protocolo oficial (`configs/base.yaml`, `Adam lr=5e-4, wd=0, dropout=0.2`, 85 runs em `outputs/results/runs/`); `ridge_esm` e `ridge_esm_target_only` selecionaram $\alpha \in \{10^{-2}, \dots, 10^4\}$ estritamente na validação interna; `knn_kmer_target`, `knn_tanimoto_drug` ($k=5$) e os marginais são determinísticos sem ajuste no teste.

---

## 2. Tetos de Épocas e Diagnóstico de Treinamento (`gin_esm_mlp`, Davis Exploratório)

Fonte direta: campos `effective_config.epochs`, `total_epochs_run` e `best_epoch` em `outputs/results/runs/physdual_gin_esm_mlp_davis_*_fold{0..4}.json`.

| Regime (Davis `v2_kfold`) | Teto (`epochs`) | Paciência | `best_epoch` por dobra (`0..4`) | Mediana `best_epoch` | Status / Alerta |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`cold_target`** | `500` | `80` | `[288, 481, 235, 161, 145]` | `235` | Dobra `1` truncada (`481/500`, `near_epoch_ceiling_warn=true`); re-executar dobra 1 com `--epochs 1000 --extend-epochs`. |
| **`cold_drug`** | `1000` | `80` | `[180, 80, 5, 137, 196]` | `137` | Dobra `2` colapsou precocemente (`best_epoch=5 < 15`, `early_memorization_warn`). |
| **`cold_target_seqid40`** | `1000` | `80` | `[41, 178, 99, 103, 141]` | `103` | Todas convergiram antes do teto (`121..258` épocas). |
| **`cold_target_seqid30`** | `1000` | `80` | `[71, 85, 180, 90, 161]` | `90` | Todas convergiram antes do teto (`151..260` épocas). |

> O controle automático `early_memorization_warn` (`best_epoch < 15`) e o Gate 5 em `scripts/verify_phase3.py` agora cobrem **todos os regimes `cold`** (`cold_drug`, `cold_target`, `seqid30`, `seqid40`), registrando apenas o alerta nos JSONs sem alterar pesos, perda ou critério de parada (`trainer.py`).

---

## 3. Resultados Exploratórios no Davis e Bootstrap Pareado Agrupado (OOF)

> **Classificação (ADR #15):** Todos os resultados abaixo nos 4 regimes `cold` do Davis têm caráter **exploratório** (`N=1000` reamostras bootstrap pareadas por cluster usando exatamente os mesmos clusters para modelo e baseline). Todos os baselines em CPU (`drug_mean`, `knn_tanimoto`, `knn_kmer`, `ridge_esm`, `knn_esm`) reproduziram as métricas salvas nos JSONs originais dentro da tolerância numérica (`<= 1.78e-15` para `drug_mean`, `<= 9.04e-8` para `ridge_esm` / modelos neurais, `<= 0.0147` para `knn_kmer` em `seqid40` por desempates). Fonte: `outputs/results/phase3_davis_paired_bootstrap.json`.

### 3.1 Resumo por Regime (Média $\pm$ DP e Mediana por Dobra + OOF Global)

| Regime (Cluster Bootstrap) | `gin_esm_mlp` (Fold CI $\pm$ DP / Med; MSE $\pm$ DP / Med) | `gin_esm_mlp` (OOF CI / MSE) | Melhor Baseline (CI / MSE Fold $\mid$ OOF) | `ridge_esm` (CI / MSE Fold $\mid$ OOF) |
| :--- | :---: | :---: | :---: | :---: |
| **`cold_target`** (`379` seqs) | `0.8189±0.0265` (`0.8326`) / `0.4333±0.1361` (`0.4631`) | `0.8146 / 0.4385` | `knn_kmer_target`: `0.8143 / 0.4572` $\mid$ `0.8127 / 0.4621` | `0.7649 / 0.5808` $\mid$ `0.7647 / 0.5868` |
| **`cold_target_seqid40`** (`199` clusters) | `0.7754±0.0238` (`0.7815`) / `0.5627±0.0844` (`0.5161`) | `0.7744 / 0.5629` | `ridge_esm`: `0.7471 / 0.6138` $\mid$ `0.7446 / 0.6140` | `0.7471 / 0.6138` $\mid$ `0.7446 / 0.6140` |
| **`cold_target_seqid30`** (`120` clusters) | `0.7436±0.0335` (`0.7545`) / `0.6414±0.1544` (`0.7071`) | `0.7407 / 0.6355` | `ridge_esm`: `0.7399 / 0.6448` $\mid$ `0.7444 / 0.6405` | `0.7399 / 0.6448` $\mid$ `0.7444 / 0.6405` |
| **`cold_drug`** (`68` SMILES) | `0.6607±0.0752` (`0.6496`) / `0.8467±0.3258` (`0.8419`) | `0.6451 / 0.8462` | `knn_esm`/`target_mean` (MSE `0.7311`), `knn_tanimoto` (CI `0.6981`) | `0.6698 / 0.7566` $\mid$ `0.6591 / 0.7566` |

### 3.2 Bootstrap Pareado por Cluster contra `ridge_esm` e Melhor Baseline de Cada Regime *(Exploratório)*

| Regime Davis | Comparador | $\Delta\text{MSE}$ [IC 95%] (fav./1000) | $\Delta\text{CI}$ [IC 95%] (fav./1000) | Leitura Científica |
| :--- | :--- | :---: | :---: | :--- |
| **`cold_target`** | vs. `ridge_esm` | **`-0.1489 [-0.1935, -0.1095]` (`1000/1000`)** | **`+0.0508 [+0.0298, +0.0711]` (`1000/1000`)** | Vitória consistente sobre `ridge_esm`, `CNN3` (`1000/1000`), `GIN` (`1000/1000`) e `knn_esm` (`1000/1000`). |
| **`cold_target`** | vs. `knn_kmer_target` *(melhor)* | `-0.0227 [-0.0603, +0.0085]` (`911/1000`) | `+0.0021 [-0.0061, +0.0106]` (`691/1000`) | **Dentro do ruído** estatístico contra `knn_kmer` (IC cruza zero em MSE e CI). |
| **`cold_target_seqid40`** | vs. `ridge_esm` *(melhor)* | **`-0.0507 [-0.0985, -0.0076]` (`986/1000`)** | **`+0.0295 [+0.0132, +0.0461]` (`999/1000`)** | Ganho positivo sobre `ridge_esm`, mas reduzido a `~1/3` do `cold_target` padrão. |
| **`cold_target_seqid40`** | vs. `knn_kmer_target` | **`-0.0919 [-0.1735, -0.0303]` (`999/1000`)** | **`+0.0515 [+0.0209, +0.0929]` (`1000/1000`)** | Supera `knn_kmer_target` quando homólogos `>= 40%` são removidos. |
| **`cold_target_seqid30`** | vs. `ridge_esm` *(melhor)* | **`-0.0020 [-0.0449, +0.0518]` (`582/1000`)** | **`-0.0033 [-0.0293, +0.0232]` (`405/1000`)** | **Sem ganho demonstrável a 30%:** decaimento monotônico `-0.1489 -> -0.0507 -> -0.0020`. |
| **`cold_drug`** | vs. `ridge_esm` | `+0.0868 [+0.0006, +0.1876]` (`25/1000`) | `-0.0130 [-0.0538, +0.0254]` (`265/1000`) | Desfavorável (`gin_esm_mlp` pior em MSE com IC acima de zero). |
| **`cold_drug`** | vs. `knn_tanimoto_drug` | `+0.0878 [-0.0348, +0.2241]` (`85/1000`) | **`-0.0528 [-0.0982, -0.0091]` (`11/1000`)** | Perde significativamente em CI (`11/1000` fav.). |

---

## 4. Análise Descritiva por Dobra no Davis `cold_drug` e Sensibilidade Sem a Dobra 2 *(Descritivo / Pós-Hoc)*

> **Aviso:** A exclusão da dobra `2` (`best_epoch = 5`) abaixo é estritamente **descritiva**, calculada igualmente sobre as **mesmas 4 dobras (`0, 1, 3, 4`) para todos os modelos e baselines** (`outputs/results/phase3_davis_paired_bootstrap.json`, chave `cold_drug_descriptive_all_models`). Qualquer ajuste motivado por esse achado será pós-hoc e validado apenas em `val_mse`.

| Modelo / Baseline (`cold_drug`) | Dobra 0 (CI/MSE) | Dobra 1 (CI/MSE) | Dobra 2 *(outlier)* | Dobra 3 (CI/MSE) | Dobra 4 (CI/MSE) | Média 5 Dobras (CI / MSE) | **4 Dobras s/ Dobra 2** (CI $\pm$ DP [Med] / MSE $\pm$ DP [Med]) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`gin_esm_mlp`** | `0.7807 / 0.5617` | `0.6496 / 0.5799` | `0.5720 / 1.3670` | `0.6539 / 0.8829` | `0.6473 / 0.8419` | `0.6607 / 0.8467` | `0.6829±0.0652 [0.6518]` / `0.7166±0.1694 [0.7109]` |
| **`knn_tanimoto_drug`** | `0.7891 / 0.5061` | `0.6462 / 0.7090` | `0.6867 / 0.9578` | `0.7073 / 0.8144` | `0.7034 / 0.8026` | `0.7065 / 0.7580` | `0.7115±0.0588 [0.7053]` / `0.7080±0.1426 [0.7558]` |
| **`target_mean` / `knn_kmer` / `knn_esm`** | `0.7070 / 0.5291` | `0.6565 / 0.5811` | `0.6254 / 1.0893` | `0.6610 / 0.7246` | `0.6951 / 0.7307` | `0.6690 / 0.7310` | `0.6799±0.0250 [0.6781]` / `0.6414±0.1019 [0.6528]` |
| **`ridge_esm`** | `0.7392 / 0.6327` | `0.6636 / 0.5884` | `0.6359 / 1.0457` | `0.6362 / 0.7463` | `0.6740 / 0.7699` | `0.6698 / 0.7566` | `0.6783±0.0437 [0.6688]` / `0.6843±0.0876 [0.6895]` |
| **`graphdta_gin`** | `0.7617 / 0.5390` | `0.6142 / 0.5972` | `0.6471 / 1.0285` | `0.6444 / 0.7978` | `0.6440 / 0.9071` | `0.6623 / 0.7739` | `0.6661±0.0653 [0.6442]` / `0.7103±0.1718 [0.6975]` |
| **`graphdta_cnn3`** | `0.7412 / 0.5594` | `0.5484 / 0.7558` | `0.6823 / 0.9365` | `0.7048 / 0.7182` | `0.6095 / 0.8352` | `0.6572 / 0.7610` | `0.6510±0.0881 [0.6571]` / `0.7171±0.1159 [0.7370]` |
| **`drug_mean` / `mean_predictor`** | `0.5000 / 0.5883` | `0.5000 / 0.6151` | `0.5000 / 1.1707` | `0.5000 / 0.8145` | `0.5000 / 0.8628` | `0.5000 / 0.8103` | `0.5000±0.0000 [0.5000]` / `0.7202±0.1386 [0.7148]` |

---

## 5. Registro de Afirmações da Fase 3 (Observação vs. Hipótese)

| ID | Afirmação | Natureza | Evidência / Condição |
| :--- | :--- | :---: | :--- |
| **F3-O1** | Em Davis `cold_target` padrão (`exploratório`), `gin_esm_mlp` supera `graphdta_cnn3`, `graphdta_gin`, `ridge_esm` e `knn_esm_target` (`1000/1000` bootstrap fav.), porém empata dentro do ruído com `knn_kmer_target` (`ΔMSE = -0.0227 [-0.0603, +0.0085]`, `911/1000`; `ΔCI = +0.0021 [-0.0061, +0.0106]`). | **Observação** *(exploratória)* | `phase3_davis_paired_bootstrap.json` |
| **F3-O2** | A vantagem de `gin_esm_mlp` sobre `ridge_esm` decai monotonicamente com a redução de homologia: `ΔMSE = -0.1489` (`cold_target`) $\to$ `-0.0507` (`seqid40`) $\to$ `-0.0020` (`seqid30`), sem ganho demonstrável a `30%` (`582/1000` MSE, `405/1000` CI). | **Observação** *(exploratória)* | `phase3_davis_paired_bootstrap.json` |
| **F3-O3** | Em Davis `cold_drug` (`exploratório`), `gin_esm_mlp` não supera `knn_tanimoto_drug`, `ridge_esm` nem `target_mean_predictor` (`ΔMSE = +0.0868 [+0.0006, +0.1876]` vs. `ridge_esm`; dobra `2` memorizou na época `5`). Mesmo removendo a dobra `2` descritivamente nas mesmas 4 dobras, seu MSE médio (`0.7166`) permanece acima de `target_mean_predictor` (`0.6414`), `ridge_esm` (`0.6843`) e `knn_tanimoto_drug` (`0.7080`). | **Observação** *(exploratória + descritiva)* | `phase3_davis_paired_bootstrap.json` |
| **F3-H1** | O platô do `gin_esm_mlp` em `30%` de identidade e frente a `knn_kmer` decorre da ausência de atenção cruzada / contatos explícitos ligante–resíduo sobre o *mean-pooling* do ESM-2 (e possível subajuste do ramo molecular GIN em `cold_drug`). | **HIPÓTESE** | Não testada na Fase 3; será testada nos controles (`ecfp_mlp`, `--shuffle-esm`) e nas Fases 4–6 (`PhysDual` completo). |

---

## 6. Preparação do KIBA (Splits `seqid30`/`seqid40` + Baselines em CPU)

- **Restrições Estruturais Obrigatórias:** KIBA (`229` sequências únicas, `2.065` SMILES canônicos) compartilha **`179` sequências exatas com o Davis (`78,2%`)**, e o split `v1_seeds cold_target` testa apenas **`~23` proteínas únicas por seed**. Além disso, `graphdta_gin` é um **adversário fraco** em `cold_target` no KIBA (`0/5` vitórias em MSE contra `drug_mean_predictor`) e **`graphdta_cnn3` não existe no KIBA**. Portanto, o **KIBA `cold_drug`** é priorizado para avaliar o ramo molecular e **`kiba_cold_target_seqid40` / `seqid30` (`v2_kfold`)** entram como **regimes primários** (`1000` épocas, `patience=50`).
- **Baselines Calculados em CPU (`outputs/results/kiba_cpu_classical_baselines.json` + `phase3_esm_cpu_baselines.json`):**

| Regime KIBA | `drug_mean_predictor` | `knn_tanimoto_drug` | `knn_kmer_target` | `knn_esm_target` | `ridge_esm` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`v1_seeds random`** *(sanidade)* | `0.7010 / 0.4278` | `0.7900 / 0.3442` | `0.7452 / 0.3648` | `0.7320 / 0.3855` | `0.7547 / 0.4079` |
| **`v1_seeds cold_drug`** *(primário)* | `0.5000 / 0.6739` | **`0.7831 / 0.3531`** | `0.6869 / 0.5732` | `0.6869 / 0.5732` | `0.6817 / 0.5586` |
| **`v1_seeds cold_target`** *(primário)* | `0.6964 / 0.4465` | `0.6964 / 0.4465` | **`0.7320 / 0.3730`** | `0.7245 / 0.4098` | `0.6818 / 0.5056` |
| **`v2_kfold cold_target_seqid40`** *(primário, 117 clusters)* | `0.6986 / 0.4553` | `0.6986 / 0.4553` | **`0.6995 / 0.4973`** | `0.6839 / 0.5100` | `0.6542 / 0.5419` |
| **`v2_kfold cold_target_seqid30`** *(primário, 75 clusters)* | **`0.6880 / 0.4929`** | `0.6880 / 0.4929` | `0.6843 / 0.5435` | `0.6690 / 0.5544` | `0.6201 / 0.6097` |

---

## 7. Comandos de Execução GPU no Google Colab (Após Tag `phase3-adr15-pre-kiba`)

> **Ordem obrigatória:** Todas as execuções abaixo utilizam exatamente a configuração congelada em `configs/base.yaml` (`ADR #15`). Caso o cache ESM-2 esteja no SSD local do Colab (ex.: `/content/local_esm_cache`), passe `--esm-cache-dir /content/local_esm_cache`.

### Passo A — Extensão da Dobra 1 do Davis `cold_target` (`500 -> 1000` épocas)
```bash
python -u scripts/run_phase3_physdual.py --protocol kfold --dataset davis --split cold_target --folds 1 --epochs 1000 --extend-epochs
```

### Passo B — Controles Diagnósticos Baratos (Não usar para reajuste no teste)
```bash
# 1. Controle --shuffle-esm (>=3 permutações fixas entre splits; interpretar principalmente no cold_target):
python -u scripts/run_phase3_physdual.py --protocol kfold --dataset davis --split cold_target --folds 0,1,2,3,4 --shuffle-esm --shuffle-esm-seeds 0,1,2

# 2. Controle --train-seed-offset (2a seed no seqid30 e na dobra 2 do cold_drug):
python -u scripts/run_phase3_physdual.py --protocol kfold --dataset davis --split cold_target_seqid30 --folds 0,1,2,3,4 --train-seed-offset 1000
python -u scripts/run_phase3_physdual.py --protocol kfold --dataset davis --split cold_drug --folds 2 --train-seed-offset 1000

# 3. Controle ecfp_mlp (ECFP4 2048d no lugar do GIN, ESM encoder e MLP idênticos):
python -u scripts/run_phase3_physdual.py --model physdual_ecfp_esm_mlp --protocol kfold --dataset davis --split cold_drug --folds 0,1,2,3,4
python -u scripts/run_phase3_physdual.py --model physdual_ecfp_esm_mlp --protocol kfold --dataset davis --split cold_target --folds 0,1,2,3,4
```

### Passo C — KIBA Primário (`v1_seeds` + `seqid30`/`seqid40`) e Sanidade Davis `random`
```bash
# KIBA v1_seeds (cold_drug priorizado + cold_target + sanidade random):
python -u scripts/run_phase3_physdual.py --protocol seeds --dataset kiba --split cold_drug --seeds 0,1,2,3,4
python -u scripts/run_phase3_physdual.py --protocol seeds --dataset kiba --split cold_target --seeds 0,1,2,3,4
python -u scripts/run_phase3_physdual.py --protocol seeds --dataset kiba --split random --seeds 0,1,2,3,4

# KIBA v2_kfold (regimes primários por identidade de sequência):
python -u scripts/run_phase3_physdual.py --protocol kfold --dataset kiba --split cold_target_seqid40 --folds 0,1,2,3,4
python -u scripts/run_phase3_physdual.py --protocol kfold --dataset kiba --split cold_target_seqid30 --folds 0,1,2,3,4

# Controle de sanidade Davis random (sem reajuste):
python -u scripts/run_phase3_physdual.py --protocol seeds --dataset davis --split random --seeds 0,1,2,3,4
```

### Passo D — Auditoria e Gates
```bash
python scripts/verify_phase3.py --smoke
python scripts/verify_phase3.py --full --no-require-paired-win
```
