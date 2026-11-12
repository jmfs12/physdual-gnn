# PhysDual-GNN

Predição de afinidade droga-alvo (**DTA** — $pK_d$, $pK_i$, KIBA) com **Graph Neural Networks (GIN)** e embeddings evolutivos congelados do **ESM-2**, voltada à geração e priorização de hipóteses de reposicionamento de fármacos.

> **Nota Científica:** Candidatos e alvos mencionados na contextualização do projeto (como *Donepezil*, *Brexpiprazole* e a quinase *DYRK2*) são tratados exclusivamente como **hipóteses científicas a validar**, nunca como resultados pré-estabelecidos.

---

## Status do Projeto e Fases

| Fase | Título | Status | Principais Entregas |
| :---: | :--- | :---: | :--- |
| **0** | Setup de Ambiente e Governança | **Concluída** | Bootstrap Colab + Drive, governança de dependências sem pin agressivo (ADRs #1-#7). |
| **1** | Pipeline de Dados, Splits e Cache ESM-2 | **Concluída** | Ingestão autônoma DeepDTA, splits `random`/`cold_drug`/`cold_target` (5 seeds), grafos PyG e cache offline ESM-2 ([`docs/FASE1_DADOS.md`](docs/FASE1_DADOS.md), ADRs #8-#9). |
| **2** | Baseline Oficial GraphDTA e Métricas | **Auditada / v2 Pronto** | Baseline oficial `GraphDTAGIN` e variante `GraphDTACNN3`, protocolo v1 auditado (Davis random reproduz o paper), protocolo v2 implementado (GroupKFold 5-fold Davis, baselines marginais/kNN, OOF, bootstrap agrupado e extensão KIBA, ADRs #10-#13). |
| **3** | Dual-Encoder PhysDual (GIN + ESM-2) | *Planejada* | Fusão multimodal do encoder molecular GIN com embeddings evolutivos contextuais do ESM-2. |
| **4** | Cross-Attention e Interpretabilidade | *Planejada* | Mecanismo de atenção cruzada droga-resíduo para mapeamento de sítios catalíticos. |
| **5** | Otimização de Hiperparâmetros e Eficiência | *Planejada* | Ajuste fino guiado e inferência acelerada em GPU. |
| **6** | Restrições Físicas Covalentes (Opcional) | *Planejada* | Módulo indutivo analítico baseado em potenciais LJ e Coulomb. |
| **7** | Resiliência a Mutações e Generalização | *Planejada* | Testes em mutantes pontuais de resistência clínica com sequências explícitas. |
| **8** | Triagem Virtual em Larga Escala | *Planejada* | Triagem virtual para reposicionamento contra alvos selecionados. |
| **9** | Relatório Científico e Apresentação | *Planejada* | Consolidação de resultados, benchmarks e documentação final. |

---

## Setup no Google Colab (≤ 5 Passos)

O ambiente principal de execução é o **Google Colab (GPU L4 22.5 GB ou T4 16 GB)** integrado ao **Google Drive**:

1. **Abra o notebook de bootstrap:** Carregue [`notebooks/00_colab_bootstrap.ipynb`](file:///C:/Users/joao.francisco/Documents/dev/physdual-gnn/notebooks/00_colab_bootstrap.ipynb) no Google Colab.
2. **Ative a GPU:** Vá em *Ambiente de execução → Alterar o tipo de ambiente de execução* e selecione **GPU (T4 ou L4)**.
3. **(Opcional, se o repo for privado) Configure o Secret:** No painel lateral **Secrets (ícone de chave)** do Colab, adicione `GITHUB_TOKEN` e ative o acesso ao notebook.
4. **Execute as células de 1 a 6:** O notebook montará o Google Drive em `/content/drive/MyDrive/physdual-gnn`, clonará/atualizará o repositório, instalará apenas os pacotes que faltam no Colab (sem reinstalar o `torch`) e gravará o snapshot em [`docs/colab_env_snapshot.txt`](file:///C:/Users/joao.francisco/Documents/dev/physdual-gnn/docs/colab_env_snapshot.txt).
5. **Valide com o Smoke Test (Célula 7):** Confirme que todos os itens do [`scripts/smoke_test.py`](file:///C:/Users/joao.francisco/Documents/dev/physdual-gnn/scripts/smoke_test.py) reportam `OK` antes de iniciar a Fase 1.

Consulte o guia completo em [`docs/COLAB.md`](file:///C:/Users/joao.francisco/Documents/dev/physdual-gnn/docs/COLAB.md) para detalhes sobre cache em `/content`, retomada após queda de sessão e limites do Colab.

---

## Setup em Máquina Local

Para desenvolvimento local, execução de testes unitários e lint:

```bash
# 1. Crie e ative um ambiente virtual (Python >= 3.10)
python -m venv .venv
source .venv/bin/activate  # Linux/WSL/macOS (ou .venv\Scripts\activate no Windows)

# 2. Instale as dependências completas (inclui PyTorch e stack científica)
pip install -r requirements-local.txt
pip install -e .

# 3. Execute o diagnóstico de ambiente, testes e smoke test
python scripts/diagnose_env.py
pytest
python scripts/smoke_test.py
```

---

## Estrutura do Repositório

```text
physdual-gnn/
├── configs/
│   └── base.yaml                  # Seeds, splits, datasets, modelos ESM-2, GraphDTA e treinamento
├── src/physdual/
│   ├── utils/                     # paths.py, config.py, seed.py, env_info.py
│   ├── data/                      # (Fase 1) datasets.py, splits.py, featurizer_drug.py, esm_cache.py, datamodule.py
│   ├── models/                    # (Fases 2-5) GraphDTA (graphdta.py) e PhysDual-GNN
│   ├── training/                  # (Fases 2-5) trainer.py, checkpoint.py (isolamento estrito de teste)
│   ├── physics/                   # (Fase 6, opcional) Bias analítico LJ/Coulomb via ablação
│   ├── eval/                      # (Fase 2) metrics.py (CI, MSE, RMSE, Pearson, Spearman, rm²)
│   └── screening/                 # (Fases 8-9) Triagem virtual e Grad-AAM
├── scripts/
│   ├── diagnose_env.py            # Diagnóstico de SO, Python, Torch/CUDA, GPU, RAM e Disco
│   ├── smoke_test.py              # Verificação fim-a-fim dos 8 critérios de pronto da Fase 0
│   ├── run_phase1_data.py         # Orquestrador CLI da Fase 1 (Dados, Splits, Grafos e ESM-2)
│   ├── verify_phase1.py           # Auditoria dos 7 itens do Gate de Pronto da Fase 1
│   ├── run_phase2_baseline.py     # Orquestrador CLI idempotente da Fase 2 (GraphDTA, 5 seeds, benchmarks)
│   └── verify_phase2.py           # Auditoria dos 6 critérios do Gate de Pronto da Fase 2 (--smoke / --full)
├── notebooks/
│   ├── 00_colab_bootstrap.ipynb   # Fluxo idempotente de setup no Google Colab + Drive
│   ├── 00_smoke_test.ipynb        # Execução interativa do smoke test no Colab
│   ├── 01_data_and_esm_cache.ipynb# Execução interativa da Fase 1 no Colab
│   └── 02_graphdta_baseline.ipynb # Execução interativa do baseline GraphDTA da Fase 2 no Colab
├── tests/
│   ├── test_env.py                # Testes de ambiente e configuração (Fase 0)
│   ├── test_datasets.py           # Testes de ingestão, procedência e padronização (Fase 1)
│   ├── test_splits.py             # Testes de splits sem vazamento por conteúdo (Fase 1)
│   ├── test_featurizer_drug.py    # Testes de grafos PyG com weights_only=True (Fase 1)
│   ├── test_esm_cache.py          # Testes de cache ESM-2 fp16 e janela deslizante (Fase 1)
│   ├── test_verify_phase1.py      # Testes do verificador de Gate da Fase 1
│   ├── test_metrics.py            # Testes de métricas DTA e concordância lifelines (Fase 2)
│   ├── test_datamodule.py         # Testes de collate customizado e carregamento de batches (Fase 2)
│   ├── test_graphdta.py           # Testes de integridade do GraphDTAGIN e CNN3 (Fase 2)
│   ├── test_trainer.py            # Testes de treino, early stopping e checkpoints atômicos (Fase 2)
│   └── test_verify_phase2.py      # Testes do orquestrador e do verificador de Gate da Fase 2
└── docs/
    ├── PLANO_AJUSTADO.md          # Escopo, fases (0 a 9) e arquitetura ajustada
    ├── DECISOES.md                # Decisões técnicas (ADRs #1 a #12)
    ├── COLAB.md                   # Guia operacional do Google Colab + Google Drive
    ├── FASE1_DADOS.md             # Protocolos de dados, splits, censura do Davis e cache ESM-2
    ├── FASE2_BASELINE.md          # Especificação do baseline GraphDTA, tabelas verificadas e protocolo de treino
    ├── PROXIMOS_PASSOS.md         # Roteiro de transição para a Fase 3
    └── colab_env_snapshot.txt     # Snapshot do ambiente gerado no bootstrap
```

---

## Documentação Essencial

- [`docs/PLANO_AJUSTADO.md`](docs/PLANO_AJUSTADO.md) — Fonte de verdade para escopo, arquitetura e fases do projeto.
- [`docs/DECISOES.md`](docs/DECISOES.md) — Racional das decisões de arquitetura e engenharia (ADRs #1 a #12).
- [`docs/COLAB.md`](docs/COLAB.md) — Guia de operação no Google Colab e otimização de I/O com Google Drive.
- [`docs/FASE1_DADOS.md`](docs/FASE1_DADOS.md) — Documentação técnica da Fase 1 (procedência, censura em $pK_d=5.0$, variância em `cold_drug`, janela deslizante ESM-2 e Gate de Pronto).
- [`docs/FASE2_BASELINE.md`](docs/FASE2_BASELINE.md) — Documentação técnica da Fase 2 (baseline GraphDTA oficial vs. secundário, tabelas verificadas de Nguyen et al. (2021), isolamento de teste e auditoria de mutantes).
- [`docs/PROXIMOS_PASSOS.md`](docs/PROXIMOS_PASSOS.md) — Roteiro de execução no Colab e transição para a Fase 3 (Dual-Encoder PhysDual: GIN + ESM-2).
