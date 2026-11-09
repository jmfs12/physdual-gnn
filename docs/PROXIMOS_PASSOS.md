# Próximos Passos — Execução do Gate da Fase 1 no Colab e Transição para a Fase 2

A implementação de código e testes da **Fase 1 (Dados, Splits, Featurização Molecular e Cache ESM-2)** está concluída.

---

## 1. Checklist de Execução da Fase 1 no Google Colab (Gate de Pronto)

Para materializar os artefatos no Google Drive e validar o Gate de Pronto da Fase 1:

1. No Google Colab (com GPU ativada e credenciais `KAGGLE_USERNAME` / `KAGGLE_KEY` configuradas no painel **Secrets**), abra e execute `notebooks/01_data_and_esm_cache.ipynb` (ou rode via terminal: `python scripts/run_phase1_data.py --dataset all`).
2. Execute o verificador automatizado do Gate de Pronto:
   ```bash
   python scripts/verify_phase1.py --config configs/base.yaml
   ```
3. Confirme que todos os 7 critérios retornam `[OK]` (`exit code 0`) e que o relatório `outputs/logs/phase1_summary_report.json` foi gerado.

---

## 2. Escopo Planejado para a Fase 2 (Métricas, Loop de Treino e Baseline GraphDTA)

Após a aprovação do `verify_phase1.py`:

1. **Métricas de Avaliação (`src/physdual/eval/metrics.py`):**
   - Implementação validada e testada de **MSE**, **RMSE**, **Pearson ($r$)**, **Spearman ($\rho$)**, **$r_m^2$** e **Concordance Index (CI)** (com validação cruzada contra `lifelines.utils.concordance_index`).
2. **Dataset PyTorch / PyG e DataLoader Otimizado (`src/physdual/data/datamodule.py`):**
   - Descompactação automática do tarball `esm_embeddings_{model_name}.tar` para `/content/local_cache/` no início da sessão do Colab.
   - Collate customizado combinando `Batch` de grafos moleculares PyG (`load_drug_graph`) com tensores de proteínas.
3. **Baseline GraphDTA (`src/physdual/models/graphdta.py`) e Loop de Treino:**
   - Reprodução do codificador GIN atômico + CNN 1D sobre sequência proteica.
   - Treinamento com checkpointing por época no Google Drive, early stopping em `val` e agregação de **média ± desvio-padrão nas 5 seeds** (`[0, 1, 2, 3, 4]`) para `random`, `cold_drug` e `cold_target`.
