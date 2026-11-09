# Registro de Decisões de Arquitetura e Engenharia (ADR — Fase 0)

Este documento registra as decisões técnicas tomadas na configuração do projeto **PhysDual-GNN**, assegurando rastreabilidade e evitando retrabalho nas fases seguintes.

---

## 1. Exclusão do TorchDrug
- **Decisão:** Não instalar nem utilizar a biblioteca `torchdrug`.
- **Motivo:**
  1. O `torchdrug` possui dependências rígidas e desatualizadas que entram em conflito direto com versões recentes do `torch` e `torch_geometric` pré-instaladas no Google Colab.
  2. Exige compilação JIT/C++ de extensões CUDA que frequentemente falham ou demoram vários minutos a cada reinício de sessão no Colab.
  3. Toda a funcionalidade necessária ao projeto já é coberta de forma nativa, leve e modular pela combinação **RDKit** (grafos moleculares e fragmentos) + **PyTorch Geometric (`torch_geometric`)** (`Data`, `Batch`, `GINConv`) + **fair-esm** (embeddings proteicos).

---

## 2. Não Fixar (`pin`) Versões de `torch`, `numpy` e `pandas` no Colab
- **Decisão:** Manter `torch`, `numpy`, `pandas`, `scipy` e `scikit-learn` fora do `requirements-colab.txt` e registrar o estado real do ambiente em `docs/colab_env_snapshot.txt`.
- **Motivo:**
  1. O Google Colab atualiza periodicamente sua imagem base (Python, driver CUDA, PyTorch compilado para a GPU do runtime). Forçar downgrade/reinstalação de `torch` ou `numpy` quebra compatibilidade binária (ABI) com pacotes pré-instalados e exige longos downloads.
  2. Para garantir reprodutibilidade científica sem quebrar o runtime, o script `scripts/diagnose_env.py --snapshot` grava a versão exata do Python, PyTorch, CUDA, modelo de GPU e o `pip freeze` completo em `docs/colab_env_snapshot.txt` a cada bootstrap.

---

## 3. ESM-2 Congelado e Pré-Computado em Disco (Redução de 80 GB para 16–22.5 GB VRAM)
- **Decisão:** Utilizar o ESM-2 com pesos congelados (`freeze_backbone: true`), extraindo os embeddings por resíduo uma única vez em `fp16` e salvando um arquivo `.pt` por proteína (chaveado pelo hash SHA-256 da sequência) em `cache/esm_embeddings/`.
- **Motivo:**
  1. O plano original previa treinar/executar o ESM-2 em conjunto com as GNNs em tempo real, exigindo GPUs A100/H100 de 80 GB.
  2. Como Davis possui apenas 442 proteínas únicas e KIBA possui 229 proteínas únicas, pré-computar os embeddings de `esm2_t33_650M_UR50D` (dim=1280) ou da alternativa leve `esm2_t30_150M_UR50D` (dim=640) leva poucos minutos em uma única GPU T4 (16 GB) ou L4 (22.5 GB) e reduz o custo de VRAM durante o treino da GNN a menos de 4 GB.

---

## 4. Exclusão de Pconsc4 e CCMpred para Mapas de Contato
- **Decisão:** Não instalar `Pconsc4` nem `CCMpred`.
- **Motivo:**
  1. Ambas as ferramentas dependem de pipelines de alinhamento múltiplo de sequências (MSA via HHblits/PSIBLAST contra bancos de dezenas de gigabytes) e versões legadas de Python/TensorFlow 1.x/compiladores C específicos.
  2. Na Fase 5, os grafos de resíduos proteicos serão construídos a partir de duas alternativas modernas e compatíveis com o Colab:
     - **Opção A:** Mapas de contato extraídos diretamente das cabeças de atenção do próprio ESM-2 (`return_contacts=True`).
     - **Opção B:** Mapas de distância $C_\alpha - C_\alpha$ obtidos de estruturas 3D pré-computadas no **AlphaFold Protein Structure Database (AFDB)** ou geradas sob demanda via API do **ESMFold**.

---

## 5. Contorno Escopado para `torch.load(..., weights_only=True)` no `fair-esm`
- **Decisão:** Encapsular o carregamento de modelos `esm.pretrained.*` em um gerenciador de contexto estritamente escopado (`_scoped_esm_weights_load`).
- **Motivo:**
  1. A partir do PyTorch 2.6+, `torch.load` passou a adotar `weights_only=True` por padrão para mitigar execução arbitrária via pickle.
  2. Os checkpoints oficiais do `fair-esm` armazenam hiperparâmetros como instâncias de `argparse.Namespace` e invocam `torch.load` internamente sem passar `weights_only=False`, causando `pickle.UnpicklingError`.
  3. Desativar a verificação globalmente é uma má prática de segurança. Nosso gerenciador de contexto registra `argparse.Namespace` via `torch.serialization.safe_globals` e aplica o fallback de `weights_only=False` **exclusivamente** durante a chamada de carregamento do checkpoint oficial do ESM-2, restaurando `torch.load` imediatamente no bloco `finally`. Todos os nossos próprios arquivos de cache `.pt` salvam apenas tensores e tipos primitivos e são carregados com `weights_only=True`.

---

## 6. Concordance Index (CI) via `lifelines` e `PyTDC` Opcional
- **Decisão:** Adotar `lifelines` em `requirements-colab.txt` para cálculo validado do Concordance Index (com implementação vetorizada própria em PyTorch/NumPy na Fase 2 para validação cruzada) e manter `PyTDC` como dependência **opcional**.
- **Motivo:**
  1. `lifelines` é leve, estável e fornece `lifelines.utils.concordance_index` sem conflitar com a stack do Colab.
  2. `PyTDC` frequentemente força a reinstalação de `numpy`, `pandas` e `rdkit-pypi`, exigindo reinício do runtime do Colab. Na Fase 1, o pipeline suportará tanto `PyTDC` (se instalado) quanto download direto dos arquivos canônicos de Davis e KIBA (repositório DeepDTA/GraphDTA).

---

## 7. Módulo de Física (Fase 6) e Hipóteses de Reposicionamento
- **Decisão:** Isolar os potenciais de Lennard-Jones e Coulomb em `src/physdual/physics/` como módulo opcional (`physics.enabled: false` por padrão em `configs/base.yaml`), sem instalar AutoDock Vina ou DiffDock. Tratar Donepezil, Brexpiprazole e DYRK2 exclusivamente como hipóteses de teste.
- **Motivo:**
  1. Como não há estruturas de co-cristal para todos os pares droga-alvo em Davis/KIBA, termos de energia aproximados no espaço latente podem introduzir ruído se não forem devidamente calibrados. Por isso, serão avaliados por ablação estrita na Fase 6.
  2. Afirmar eficácia de fármacos reposicionados sem validação experimental/estatística viola a integridade científica do projeto.
