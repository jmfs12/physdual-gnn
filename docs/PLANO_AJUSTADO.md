# Plano de Projeto Ajustado — PhysDual-GNN

> **Fonte de verdade de escopo, fases e arquitetura.** Este documento adapta o plano original ([docs/PLANO.MD](file:///C:/Users/joao.francisco/Documents/dev/physdual-gnn/docs/PLANO.MD)) às restrições reais de execução no **Google Colab (GPU L4 22.5 GB / T4 16 GB) + Google Drive** e ao rigor experimental de aprendizado de máquina em bioinformática.

---

## 1. Objetivo e Premissas Fundamentais

Desenvolver o **PhysDual-GNN**, uma arquitetura baseada em Graph Neural Networks (GNNs) e embeddings evolutivos de proteínas (ESM-2) para predição contínua de afinidade droga-alvo (DTA — $pK_d$, $pK_i$, KIBA score), voltada à geração e priorização de hipóteses de reposicionamento de fármacos.

### Diretrizes Inegociáveis do Plano Ajustado
1. **Viabilidade em GPU de 16–22.5 GB (T4/L4):** O requisito original de 80 GB de VRAM (A100/H100) foi eliminado congelando o backbone do **ESM-2** e pré-computando seus embeddings em `fp16` para cache em disco (`cache/esm_embeddings/`).
2. **Sem dependências legadas quebradiças:** **TorchDrug**, **Pconsc4** e **CCMpred** foram removidos da stack. Grafos de resíduos proteicos serão construídos a partir de mapas de contato do próprio ESM-2 ou de estruturas AlphaFold DB / ESMFold (decisão empírica na Fase 5).
3. **Física como hipótese testável (Ablação):** O módulo de potenciais analíticos (Lennard-Jones e Coulomb) é opcional (Fase 6) e só permanece no modelo final se demonstrar ganho estatisticamente significativo sobre a versão sem física. Não serão instalados AutoDock Vina nem DiffDock.
4. **Avaliação rigorosa:** Todo modelo será avaliado em três regimes de divisão (**random**, **cold-drug** e **cold-target**) com **5 sementes aleatórias** (`[0, 1, 2, 3, 4]`), reportando média e desvio-padrão de CI, MSE, Pearson, Spearman e $r_m^2$.
5. **Rigor científico sobre reposicionamento:** Fármacos e alvos citados na motivação do projeto (**Donepezil**, **Brexpiprazole** e a quinase **DYRK2**) são **hipóteses científicas a validar**, jamais resultados pré-determinados.

---

## 2. Arquitetura Ajustada do Sistema

```
[ SMILES ] -----> [ RDKit: Grafo Atômico + Fragmentos ] -----> [ GIN Encoders ] \
                                                                                  +--> [ Fusão / Cross-Attention ] --> [ MLP ] --> Afinidade (pKd / KIBA)
[ Sequência ] --> [ Cache ESM-2 Congelado (1280d/640d) ] ----> [ Protein GNN ]  /                ^
     |                                                                ^                          |
     +----------> [ Contatos ESM-2 ou AlphaFold DB ] -----------------+            [ Bias Físico Opcional: Fase 6 ]
                                                                                    (Lennard-Jones + Coulomb)
```

---

## 3. Cronograma Incremental por Fases

| Fase | Escopo | Entregável Principal / Critério de Pronto |
| :--- | :--- | :--- |
| **Fase 0** | **Setup e Infraestrutura Colab/Local** | Estrutura do repositório, utilitários (`paths`, `config`, `seed`, `env_info`), notebook de bootstrap idempotente e smoke test de 8 itens. |
| **Fase 1** | **Pipeline de Dados e Cache ESM-2** | Download e curadoria de **Davis** e **KIBA**, geração determinística dos splits (`random`, `cold_drug`, `cold_target`), grafos PyG de ligantes e pré-computação offline de embeddings do ESM-2 no Drive. |
| **Fase 2** | **Métricas, Loop de Treino e Baseline GraphDTA** | Implementação de CI, MSE, Pearson, Spearman e $r_m^2$; loop de treino com checkpointing por época no Drive; reprodução do baseline GraphDTA (GIN). |
| **Fase 3** | **Codificador de Proteínas com ESM-2 Congelado** | Substituição da CNN 1D de sequência do baseline por projeção e agregação dos embeddings pré-computados do ESM-2 (`esm2_t33_650M_UR50D` ou `esm2_t30_150M_UR50D`). |
| **Fase 4** | **Representação Multiescala do Fármaco** | Codificadores GIN paralelos nos níveis atômico e de fragmentos funcionais (BRICS/RECAP via RDKit). |
| **Fase 5** | **Grafo de Resíduos e Fusão Droga-Alvo** | Construção do grafo de resíduos proteicos (contatos ESM-2 vs. estruturas AlphaFold DB) e mecanismo de interação mútua (Cross-Attention / Bitransport). |
| **Fase 6** | **Módulo de Física Opcional (Ablação)** | Implementação de termos analíticos regularizadores de Lennard-Jones e Coulomb no espaço latente de interação; estudo de ablação formal. |
| **Fase 7** | **Consolidação Experimental e Mutações** | Avaliação completa em 5 seeds nos 3 splits (Davis e KIBA) e análise de sensibilidade a mutações pontuais de alvos. |
| **Fase 8** | **Triagem Virtual Exploratória** | Pipeline de inferência em lote com filtros físico-químicos/ADMET básicos e avaliação exploratória de hipóteses (incluindo avaliação crítica de pares como Donepezil/Brexpiprazole contra DYRK2). |
| **Fase 9** | **Interpretabilidade (Grad-AAM) e Demonstração** | Mapeamento de ativação por gradiente sobre átomos/resíduos e interface opcional em Gradio. |
