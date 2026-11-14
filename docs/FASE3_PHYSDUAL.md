# Fase 3 — Dual-Encoder Enxuto (`GIN + ESM-2 Congelado + MLP`)

> **Referência Arquitetural:** [docs/DECISOES.md](DECISOES.md) (ADR #14) e [docs/superpowers/plans/2026-11-13-phase3-gin-esm2-mlp.md](superpowers/plans/2026-11-13-phase3-gin-esm2-mlp.md).

## 1. Arquitetura Implementada (`PhysDualPhase3`)

- **Ramo Molecular (`GINEncoder` + `LayerNorm(128)`):** 5 camadas `GINConv` (`hidden_dim=32`, `drug_in_dim=78` L1-normalizadas, `global_add_pool`, `dropout=0.2`).
- **Ramo Proteico (`ESMTargetEncoder`):** consome embeddings offline `fp16` do `esm2_t33_650M_UR50D` (`1280d`, janela deslizante `1022/256`, convertidos para `fp32` via `weights_only=True`), aplica `mean_pool` sem truncamento, `LayerNorm(1280) -> Dropout(0.4) -> Linear(1280, 128) -> GELU -> LayerNorm(128) -> Dropout(0.3)`.
- **Cabeça de Fusão (`MLP`):** concatenação `[z_drug_norm ; z_prot_norm]` (`256d`) $\to$ `Linear(256, 512) -> ReLU -> Dropout(0.3) -> Linear(512, 256) -> ReLU -> Dropout(0.3) -> Linear(256, 1)`.
- **Otimizador e Diagnóstico:** `AdamW` (`lr=5e-4`, `weight_decay=1e-3`), registro de `train_loss_at_best`, `train_val_mse_gap_at_best` e alerta automático `early_memorization_warn` se `best_epoch < 15` em `cold_target`.

## 2. Alvos Numéricos a Superar (Etapa 3.0 + Task 3)

| Regime (Davis) | `drug_mean_predictor` | `knn_kmer_target` | `graphdta_cnn3` | `knn_esm_target` | `ridge_esm` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`v2_kfold cold_target`** | `0.7488 / 0.6253` | `0.8128 / 0.4591` | `0.7740 / 0.5350` | `0.8036 / 0.5076` | `0.7649 / 0.5808` |
| **`cold_target_seqid40`** | `0.7443 / 0.6303` | `0.7226 / 0.6573` | `0.7475 / 0.6386` | `0.7430 / 0.6734` | `0.7471 / 0.6138` |
| **`cold_target_seqid30`** | `0.7360 / 0.6565` | `0.7044 / 0.7498` | `0.7283 / 0.7253` | `0.7165 / 0.7456` | `0.7399 / 0.6448` |

## 3. Comandos de Execução no Google Colab (GPU L4/T4)

### Passo A — Prova de Conceito no Davis (`v2_kfold` + `seqid30` + `seqid40`, teto padrão `1000` épocas)
```bash
# Se o cache ESM-2 estiver em SSD local rápido (ex.: /content/local_esm_cache), passe --esm-cache-dir:
python -u scripts/run_phase3_physdual.py --protocol kfold --dataset davis --split cold_target --folds 0,1,2,3,4
python -u scripts/run_phase3_physdual.py --protocol kfold --dataset davis --split cold_drug --folds 0,1,2,3,4
python -u scripts/run_phase3_physdual.py --protocol kfold --dataset davis --split cold_target_seqid30 --folds 0,1,2,3,4
python -u scripts/run_phase3_physdual.py --protocol kfold --dataset davis --split cold_target_seqid40 --folds 0,1,2,3,4

# Caso alguma dobra tenha sido iniciada com teto de 500 e atingido próximo ao teto:
python -u scripts/run_phase3_physdual.py --protocol kfold --dataset davis --split cold_target --folds 0,1,2,3,4 --epochs 1000 --extend-epochs
```

### Passo B — Extensão para KIBA (`v1_seeds`) e Controle Davis `random` (após aprovar Passo A)
```bash
python -u scripts/run_phase3_physdual.py --protocol seeds --dataset davis --split random --seeds 0,1,2,3,4
python -u scripts/run_phase3_physdual.py --protocol seeds --dataset kiba --split all --seeds 0,1,2,3,4
```

### Verificação do Gate da Fase 3
```bash
python scripts/verify_phase3.py --smoke
python scripts/verify_phase3.py --full
```
