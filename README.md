# PhysDual-GNN

Predição de afinidade droga-alvo (**DTA** — $pK_d$, $pK_i$, KIBA) com **Graph Neural Networks (GIN)** e embeddings evolutivos congelados do **ESM-2**, voltada à geração e priorização de hipóteses de reposicionamento de fármacos.

> **Nota Científica:** Candidatos e alvos mencionados na contextualização do projeto (como *Donepezil*, *Brexpiprazole* e a quinase *DYRK2*) são tratados exclusivamente como **hipóteses científicas a validar**, nunca como resultados pré-estabelecidos.

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
│   └── base.yaml                  # Seeds, splits, datasets, modelos ESM-2 e caminhos relativos
├── src/physdual/
│   ├── utils/                     # paths.py, config.py, seed.py, env_info.py
│   ├── data/                      # (Fase 1) Datasets Davis/KIBA, splits e featurização
│   ├── models/                    # (Fases 2-5) GraphDTA baseline e PhysDual-GNN
│   ├── physics/                   # (Fase 6, opcional) Bias analítico LJ/Coulomb via ablação
│   ├── eval/                      # Métricas: CI, MSE, Pearson, Spearman, rm²
│   └── screening/                 # (Fases 8-9) Triagem virtual e Grad-AAM
├── scripts/
│   ├── diagnose_env.py            # Diagnóstico de SO, Python, Torch/CUDA, GPU, RAM e Disco
│   └── smoke_test.py              # Verificação fim-a-fim dos 8 critérios de pronto da Fase 0
├── notebooks/
│   ├── 00_colab_bootstrap.ipynb   # Fluxo idempotente de setup no Google Colab + Drive
│   └── 00_smoke_test.ipynb        # Execução interativa do smoke test no Colab
├── tests/
│   └── test_env.py                # Testes automatizados (pytest / unittest)
└── docs/
    ├── PLANO_AJUSTADO.md          # Escopo, fases (0 a 9) e arquitetura ajustada
    ├── DECISOES.md                # Decisões técnicas (sem TorchDrug, sem pin de torch, etc.)
    ├── COLAB.md                   # Guia operacional do Google Colab + Google Drive
    ├── PROXIMOS_PASSOS.md         # Roteiro detalhado para a Fase 1
    └── colab_env_snapshot.txt     # Snapshot do ambiente gerado no bootstrap
```

---

## Documentação Essencial

- [`docs/PLANO_AJUSTADO.md`](file:///C:/Users/joao.francisco/Documents/dev/physdual-gnn/docs/PLANO_AJUSTADO.md) — Fonte de verdade para escopo, arquitetura e fases do projeto.
- [`docs/DECISOES.md`](file:///C:/Users/joao.francisco/Documents/dev/physdual-gnn/docs/DECISOES.md) — Racional das decisões de engenharia da Fase 0.
- [`docs/COLAB.md`](file:///C:/Users/joao.francisco/Documents/dev/physdual-gnn/docs/COLAB.md) — Guia de operação no Google Colab e otimização de I/O com Google Drive.
- [`docs/PROXIMOS_PASSOS.md`](file:///C:/Users/joao.francisco/Documents/dev/physdual-gnn/docs/PROXIMOS_PASSOS.md) — Próximas etapas planejadas para a Fase 1.
