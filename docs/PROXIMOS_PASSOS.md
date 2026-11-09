# Próximos Passos — Roteiro para a Fase 1 (Dados e Cache ESM-2)

Este documento registra as atividades planejadas para a **Fase 1**, mantendo o escopo da **Fase 0** estritamente focado em infraestrutura e configuração.

---

## Checklist de Entrada para a Fase 1 (Ação do Usuário no Colab)

Antes de iniciarmos a implementação da Fase 1:
1. Abra e execute `notebooks/00_colab_bootstrap.ipynb` no Google Colab com GPU T4 ou L4 ativada.
2. Confirme que todos os itens do smoke test retornaram `[  OK  ]` (ou `[PULADO]` apenas para o item 7 opcional `PyTDC`).
3. Faça commit/envie de volta o arquivo `docs/colab_env_snapshot.txt` gerado pela Célula 6 no runtime do Colab e o output do smoke test.

---

## Escopo Técnico da Fase 1 (A Implementar Após Aprovação da Fase 0)

1. **Ingestão e Padronização dos Datasets (`src/physdual/data/datasets.py`):**
   - Download automatizado e reprodutível de **Davis** ($68 \times 442$, $pK_d = -\log_{10}(K_d / 10^9)$) e **KIBA** ($2111 \times 229$, KIBA score) diretamente das fontes canônicas (DeepDTA/GraphDTA), com suporte opcional ao `PyTDC`.
   - Validação de SMILES via RDKit (canonização, remoção de sais se aplicável) e verificação de integridade das sequências FASTA.

2. **Divisões Experimentais Determinísticas (`src/physdual/data/splits.py`):**
   - Geração e persistência em `data/processed/splits/` dos índices de treino/validação/teste (80/10/10) para as 5 seeds `[0, 1, 2, 3, 4]` nos três cenários:
     - `random`: divisão aleatória de pares $(d, t)$.
     - `cold_drug`: fármacos do conjunto de teste/validação nunca vistos no treino.
     - `cold_target`: proteínas do conjunto de teste/validação nunca vistas no treino.

3. **Featurização Atômica de Ligantes (`src/physdual/data/featurizer_drug.py`):**
   - Conversão de SMILES únicos em grafos PyG com features atômicas e de ligação química completas, salvos em cache para evitar recomputação a cada época.

4. **Extração Offline de Embeddings do ESM-2 (`scripts/precompute_esm.py`):**
   - Processamento das proteínas únicas de Davis (442) e KIBA (229) com `esm2_t33_650M_UR50D` congelado em `fp16` (ou `esm2_t30_150M_UR50D`).
   - Persistência de um arquivo `<sha256>.pt` por sequência em `cache/esm_embeddings/` no Google Drive e utilitário de cópia rápida para `/content/local_cache/` no início da sessão de treino.
