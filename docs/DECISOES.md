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

---

## 8. Política para Aminoácidos Ambíguos ou Não-Padrão nas Sequências Proteicas (Fase 1)
- **Decisão:** **Manter** os códigos IUPAC estendidos/ambíguos (`X`, `B`, `U`, `Z`, `O`) intactos na sequência sanitizada, removendo apenas espaços/quebras de linha, convertendo para maiúsculas, rejeitando com erro explícito qualquer caractere fora do vocabulário oficial de aminoácidos do ESM-2 (`ACDEFGHIKLMNPQRSTVWYBXZUO`) e registrando no relatório de ingestão a contagem exata de sequências que contêm qualquer um desses 5 caracteres.
- **Motivo:**
  1. O alfabeto nativo do `fair-esm` (treinado sobre UniRef50) inclui explicitamente os tokens `X` (desconhecido), `B` (Asx), `U` (selenocisteína), `Z` (Glx) e `O` (pirrolisina). Normalizá-los para `X` descartaria informação sem ganho e alteraria o hash SHA-256 da sequência original.
  2. **Ressalvas registradas:** (a) Em Davis e KIBA esses caracteres são raros ou inexistentes (impacto prático mínimo; se a contagem for zero, a decisão permanece documentada mas inócua); (b) os tokens raros `U` e `O` possuem embeddings pouco treinados no ESM-2, razão adicional pela qual o pipeline contabiliza e reporta explicitamente sua ocorrência por dataset.

---

## 9. Fonte Canônica dos Dados (DeepDTA Oficial), Fallback Kaggle e Política de Colapso Químico/Biológico (Fase 1)
- **Decisão:**
  1. **Fonte Canônica Oficial e Verificação (DeepDTA):**
     - O repositório canônico oficial dos dados de benchmark é **`hkmztrk/DeepDTA`**. Em verificação técnica em 2026-11-09, os arquivos e caminhos reais foram inspecionados diretamente via API do GitHub e fixados no commit estável **`a546a8433a6822e958f36171c4356ad6f414d623`** (data: 2023-08-16T12:18:16Z).
     - **Caminhos canônicos validados:**
       - Davis: `data/davis/ligands_can.txt`, `data/davis/proteins.txt`, `data/davis/Y`
       - KIBA: `data/kiba/ligands_can.txt`, `data/kiba/proteins.txt`, `data/kiba/Y`
     - **Checagem cruzada e validação de procedência:**
       - Davis bruto: exatamente 68 fármacos, 442 alvos, matriz $Y$ de shape $(68, 442)$ com 30.056 pares medidos (zero NaN), $K_d$ em nM ($\min=0.016$, $\max=10.000$, média=7.415,87, com 69,64% dos pares censurados no teto de 10.000 nM / $pK_d=5.0$).
       - KIBA bruto: exatamente 2.111 fármacos (2.065 SMILES canônicos únicos), 229 alvos, matriz $Y$ de shape $(2.111, 229)$ com 118.254 pares medidos em KIBA score ($\min=0.0$, $\max=17.200$, média=11.72, e valores NaN representando pares não medidos).
     - **Prioridade de Download:** A função `download_canonical_dataset` em `src/physdual/data/datasets.py` baixa esses arquivos diretamente via HTTP do GitHub raw, tornando o download **completamente autônomo e sem necessidade de credenciais, tokens ou login** em plataformas de terceiros.
     - **Kaggle como Alternativa Secundária:** A ingestão via `kagglehub` (`rajaryan2315/davis-and-kiba-datasets`) é mantida como opção configurável (`source="kaggle"` ou `use_kaggle=True`), para flexibilidade, lendo credenciais do Colab se disponíveis.
  2. **Separação entre Asserts de Procedência (Tabela Bruta) e Agrupamento por Conteúdo (Tabela Limpa):**
     - Os `assert`s estritos de procedência (Davis: $68 \times 442 = 30.056$; KIBA: $2.111 \times 229 = 118.254$ pares válidos) rodam **sobre os dados brutos de entrada, antes de qualquer filtro ou colapso de canonização**.
     - O relatório de ingestão registra separadamente `num_drugs_by_id` vs. `num_drugs_by_canonical_smiles` e `num_targets_by_id` vs. `num_targets_by_sequence`.
     - **Política para pares duplicados por conteúdo (`(smiles, sequence)` idênticos vindos de IDs distintos):** **Manter ambos os registros** na tabela limpa (não descartar nem agregar silenciosamente para preservar os 30.056 / 118.254 pares de benchmark e permitir auditoria de `num_pairs_lost == 0`), registrando `duplicate_content_pairs_count` e `duplicate_content_affinity_std` no relatório. Como o split `cold_drug`/`cold_target` agrupa estritamente por `smiles` canônico e `sequence`, ambos os registros caem obrigatoriamente na mesma partição (`train`, `val` ou `test`), impedindo vazamento.

---

## 10. Featurização Atômica One-Hot com Normalização L1 (78d) vs. Representação Escalar Inteira (9d) (Fase 2)
- **Decisão:** Adotar como featurização atômica molecular padrão (`v2_onehot78_l1norm`) a codificação exata de 78 dimensões especificada no repositório oficial do GraphDTA (`thinng/GraphDTA`, `create_data.py:L230-L256`), composta por:
  1. `atom.GetSymbol()` com `one_of_k_encoding_unk` em 44 categorias atômicas;
  2. `atom.GetDegree()` em 11 categorias (`0..10`);
  3. `atom.GetTotalNumHs()` com `one_of_k_encoding_unk` em 11 categorias (`0..10`);
  4. `atom.GetImplicitValence()` com `one_of_k_encoding_unk` em 11 categorias (`0..10`);
  5. `atom.GetIsAromatic()` (1 dimensão binária).
  Além disso, aplicar **normalização L1 por átomo com guarda contra soma zero** (`row_sum = float(feature.sum()); if row_sum > 0.0: feature = feature / row_sum`), reproduzindo fielmente a implementação oficial em que cada vetor atômico tem soma igual a 1.0. A versão binária sem normalização (`v2_onehot78_raw`, com soma 4.0 ou 5.0) e a versão legada (`v1_int9`, 9 dimensões escalares inteiras da Fase 1) são preservadas no código e selecionáveis por parâmetro para eventuais estudos de ablação.
- **Motivo:**
  1. A representação de 9 dimensões escalares inteiras utilizada na Fase 1 tratava propriedades qualitativas/nominais (ex.: símbolo atômico `atomic_num` e tipo de hibridização) como escalares contínuos, impondo uma métrica euclidiana artificial entre elementos químicos (ex.: flúor vs. carbono).
  2. A arquitetura convolucional GIN (`GINConvNet`) do GraphDTA original foi projetada e validada na literatura para projetar vetores atômicos esparsos de 78 dimensões (`x.shape[1] == 78`). Usar a featurização exata do GraphDTA elimina divergências de representação no baseline oficial, garantindo reproducibilidade estrita.
  3. O cache de grafos passa a ser explicitamente versionado em disco (`{dataset}_drug_graphs_{version}.pt`), incluindo metadados (`featurizer_version`, `atom_feature_dim`, `normalize_atom_features`) validados na deserialização com `weights_only=True`.

---

## 11. Auditoria de Colapso de Alvos/Fármacos e Limitação Metodológica para Mutações (Fase 2 / Fase 7)
- **Decisão:**
  1. Manter a política de duplicatas `keep_all` (preservando todos os pares para manter a volumetria canônica de 30.056 pares no Davis e 118.254 no KIBA), garantindo que os particionamentos `cold_drug` e `cold_target` agrupem estritamente por conteúdo químico (`smiles`) e biológico (`sequence`), o que impede vazamento entre partições mesmo com IDs distintos.
  2. Implementar a auditoria formal `audit_content_and_target_duplicates` reportando a contagem de alvos que compartilham sequência e pares com afinidades divergentes.
  3. **Registro de Limitação Metodológica (Fase 7):** Documentar formalmente que os 442 `target_id`s do Davis colapsam em 379 sequências únicas (e 2.111 fármacos do KIBA em 2.065 SMILES únicos). Se variantes mutantes ou estados fosforilados no Davis compartilham a sequência de aminoácidos do tipo selvagem (*wild-type*) no arquivo `proteins.txt` fornecido pelo DeepDTA, **o objetivo de predição de impacto de mutações (Fase 7) não é mensurável diretamente a partir dessas entradas sem a reconstrução explícita das sequências com a mutação pontual**. Para a Fase 7, qualquer teste com mutantes exigirá sequências contendo a substituição do resíduo ou datasets dedicados de mutações.

---

## 12. Adoção do Baseline Oficial GraphDTA GIN, Seleção por Validação vs. Teste e Orçamento Computacional (Fase 2)
- **Contexto:**
  1. O repositório original `thinng/GraphDTA` implementa quatro variantes para grafos de fármacos (GCN, GIN, GAT, GAT-GCN) e uma arquitetura convolucional 1D para alvos proteicos baseada em `Conv1d(1000, 32, 8)` com flatten e projeção linear. Dentre todas as variantes, o GraphDTA com **GIN (5 camadas)** obteve o melhor desempenho no benchmark Davis no artigo original de Nguyen et al. (2021) (*Bioinformatics*, CI = 0.893, MSE = 0.229).
  2. Diversas reimplementações secundárias na literatura adaptaram o codificador proteico do DeepDTA (uma CNN de 3 camadas com `Conv1d(128, 32, 4) -> Conv1d(32, 64, 8) -> Conv1d(64, 96, 12)` com max pooling) para o GraphDTA, gerando divergências conceituais sobre o que constitui a arquitetura "GraphDTA".
  3. No repositório oficial do GraphDTA (`training.py:L70-L86`), o conjunto de teste era avaliado a cada época ao longo de 1000 épocas, e o modelo persistido como final era aquele que atingisse o menor MSE no conjunto de teste. Essa prática introduz **viés de seleção otimista (*data leakage* de hiperparâmetros de parada antecipada no conjunto de teste)**, inflacionando artificialmente as métricas reportadas na literatura.
  4. O treinamento completo de 5 seeds nos 3 regimes de split (`random`, `cold_drug`, `cold_target`) para Davis e KIBA exige centenas de épocas por modelo, sob limites estritos de cota de GPU no Google Colab (sessões preemptíveis com teto de 12 horas).
- **Decisões:**
  1. **Adoção do Baseline Oficial (`graphdta_gin`) vs. Variante Secundária (`graphdta_cnn3`):**
     - O modelo oficial de referência para a Fase 2 é o `GraphDTAGIN` (`src/physdual/models/graphdta.py`), reproduzindo com exatidão a implementação original do commit `6162245c6776684b391d30610850fe7077478435`: `GINEncoder` com 5 camadas `GINConv` (dimensão oculta 32) + `OfficialGraphDTATargetEncoder` com `Conv1d(1000, 32, 8)` linearizada para 128 dimensões + MLP de fusão `256 -> 1024 -> 256 -> 1`.
     - A variante de 3 camadas (`GraphDTACNN3`) é mantida e implementada com o rótulo explícito *"GraphDTA-style com CNN estilo DeepDTA"*, servindo estritamente como controle e referência secundária.
  2. **Isolamento Estrito de Teste (Eliminação do Data Leakage):**
     - O conjunto de teste (`test_loader`) é avaliado **estritamente uma única vez ao término do treinamento**.
     - Durante as épocas de treino, o critério de salvamento do melhor checkpoint (`checkpoint_best.pt`) e de parada antecipada (*early stopping*) baseia-se **exclusivamente na perda de validação (`val_loader`)**.
     - Todos os relatórios de execução e verificações de integridade auditam e exigem a flag `test_isolation_verified: true`.
  3. **Paciência Diferenciada para Splits Cold no Davis:**
     - Definir `patience = 50` épocas para o split `random` e estender para `davis_cold_patience = 80` épocas para os regimes `cold_drug` e `cold_target`. Em virtude da alta variância induzida pela pequena quantidade de entidades agrupadas no Davis (apenas 68 fármacos), as curvas de perda em splits cold apresentam estagnação transitória antes de convergir.
  4. **Orçamento Computacional e Priorização Davis vs. KIBA:**
     - **Prioridade 1 (Obrigatória):** Treinar integralmente o Davis nos 3 splits (`random`, `cold_drug`, `cold_target`) em 5 seeds (`0, 1, 2, 3, 4`) para `graphdta_gin` (15 runs) e `graphdta_cnn3` (15 runs).
     - **Prioridade 2 (Condicional):** O dataset KIBA (~118 mil pares) só é executado no Colab se a medição preliminar de benchmark confirmar tempo $\le 15\text{ s/época}$. Caso contrário, prioriza-se o split `random` com 3 seeds para economizar unidades de computação de GPU para a Fase 3 (Fusão com ESM-2).
- **Consequências Científicas e Técnicas:**
  - *Positivas:*
    - **Rigor Metodológico Superior:** Eliminação total do viés de parada antecipada no conjunto de teste, garantindo comparabilidade honesta com os futuros modelos PhysDual-GNN.
    - **Transparência na Saturação do Split Aleatório:** Como o GraphDTA GIN atinge CI de 0.893 no Davis `random` (próximo à meta global de ~0.89), fica explicitado que a contribuição científica do PhysDual-GNN reside na generalização para novas drogas e alvos (`cold_drug`, `cold_target`), em afinidades não censuradas ($pK_d > 5.0$) e na captura de contexto em sequências longas via ESM-2.
    - **Resiliência Operacional:** A persistência atômica por run individual (`save_json_atomic`) e o checkpointing seguro por época tornam a execução no Google Colab completamente tolerante a preempções de hardware.
  - *Negativas / Mitigações:*
    - As métricas obtidas pelo nosso baseline oficial no Davis `random` podem ser ligeiramente menores ou mais conservadoras do que o CI = 0.893 e MSE = 0.229 reportados na Tabela 2 de Nguyen et al. (2021). Esse desvio é esperado e documentado formalmente como reflexo do protocolo de isolamento estrito de teste.

---

## 13. Protocolo de Avaliação v2 para Davis Cold: GroupKFold de 5 Dobras, Baselines Marginais/kNN, Avaliação Out-of-Fold (OOF) e Bootstrap Agrupado
- **Contexto:**
  1. A auditoria forense dos resultados consolidados da Fase 2 revelou que, no Davis `cold_target`, o modelo oficial `graphdta_gin` convergiu em apenas ~5 épocas (`best_epoch = 4.6 ± 2.1`) e obteve desempenho inferior ao baseline trivial de promiscuidade marginal por fármaco (`drug_mean_predictor`: $\Delta\text{MSE} = +0.1076 \pm 0.0940$, $\Delta\text{CI} = -0.0295 \pm 0.0178$).
  2. A causa-raiz identificada é estrutural: o codificador proteico oficial do GraphDTA (`Conv1d(1000, 32, 8) + Linear(3872, 128)`) trata posições absolutas da sequência como canais fixos sem invariância translacional, memorizando a composição das proteínas de treino e degenerando para um estimador de média por fármaco quando exposto a novos alvos.
  3. Em contrapartida, o modelo secundário `graphdta_cnn3` (estilo DeepDTA com convoluções 1D e Global Max Pooling) possui invariância translacional intrínseca, convergiu na época ~81 e superou a média marginal por droga ($\Delta\text{MSE} = -0.0740 \pm 0.0420$, $\Delta\text{CI} = +0.0119 \pm 0.0101$).
  4. O particionamento em 5 seeds pseudoaleatórias com 20% no teste gera alta variância e não garante que todas as entidades do Davis (apenas 68 fármacos e 379 proteínas) sejam avaliadas como entidades não vistas. Além disso, a comparação ingênua contra a média global mascara o efeito de promiscuidade/homologia.
- **Decisões:**
  1. **Adoção da "Opção C" (5-Fold GroupKFold no Davis Cold):**
     - O protocolo oficial para avaliação nos regimes `cold_drug` e `cold_target` do Davis passa a ser o **GroupKFold de 5 dobras**, agrupado estritamente por conteúdo químico (`smiles` canônico) e biológico (`sequence`).
     - Atribuição determinística fixa (`seed=42`) onde cada cluster/entidade pertence ao conjunto de teste de exatamente uma única dobra (100% de cobertura do dataset ao longo das 5 dobras, com zero vazamento entre treino/val e teste).
     - Validação interna de cada dobra extraída dos grupos de treino (~10% dos grupos de treino) sob o mesmo critério de conteúdo; paciência de early stopping mantida em 80 épocas.
  2. **Isolamento de Versões de Protocolo:**
     - Os resultados anteriores baseados em seeds aleatórias são arquivados formalmente como *"Protocolo v1 (seeds)"*, não sendo misturados com os resultados das dobras K-Fold (*"Protocolo v2 (kfold)"*).
  3. **Avaliação Out-Of-Fold (OOF) e Análise por Bootstrap Agrupado:**
     - Todas as previsões de teste fora da dobra são salvas em CSV (`outputs/results/oof/{model}_{dataset}_{split}_oof.csv`) com os pares `(drug_id, target_id, smiles, sequence, y_true, y_pred, y_pred_drug_mean, y_pred_target_mean)`.
     - Relatar tanto a média ± desvio amostral entre as 5 dobras quanto a métrica agrupada em todas as previsões OOF.
     - Implementar bootstrap agrupado no nível de cluster (*cluster-level bootstrap* com 1.000 replicações e IC de 95%) para o $\Delta\text{MSE}$ e $\Delta\text{CI}$ pareados contra os baselines marginais de promiscuidade (`drug_mean_predictor` e `target_mean_predictor`).
  4. **Baselines de Promiscuidade e Homologia (kNN):**
     - Integração obrigatória de preditores marginais e kNN por CPU no pipeline de avaliação:
       - `knn_tanimoto_drug`: similaridade de Tanimoto sobre Morgan Fingerprints (ECFP4, 1024 bits, raio 2).
       - `knn_kmer_target`: similaridade de cosseno sobre frequências de 3-mers de aminoácidos.
  5. **Métricas de Classificação Binária para Reposicionamento:**
     - Computação de AUROC e AUPR em limiares clinicamente relevantes: $pK_d > 5.0$ (afinidade biologicamente detectável) e $pK_d \ge 7.0$ (potência farmacológica submicromolar, $K_d \le 100\text{ nM}$).
  6. **Controle de Épocas no KIBA Random (Seeds 1 e 3):**
     - Apenas as seeds 1 e 3 de KIBA `random` atingiram o teto de 500 épocas (479 e 494 épocas). O orquestrador permite retomar essas duas seeds a partir do `checkpoint_last.pt` (época 500) até 1000 épocas (`--extend-epochs`), preservando o estado do otimizador e restaurando `EarlyStopping.counter = max(0, saved_epoch - best_epoch)`.
     - Regra de decisão: se o CI de teste variar menos que o desvio inter-seeds (~0.006), documenta-se e mantém-se 500; caso contrário, adota-se o resultado de 1000 épocas.
- **Consequências Científicas e Técnicas:**
  - *Positivas:*
    - **Validade Estatística Inatacável:** Todo o dataset Davis é avaliado fora da dobra (OOF), com inferência em 100% dos pares em condição estritamente cold.
    - **Definição de Baseline Realista para a Fase 3:** O alvo real a ser superado pelo PhysDual-GNN com ESM-2 no Davis `cold_target` deixa de ser a média global e passa a ser o `graphdta_cnn3` e os baselines marginais/kNN.
    - **Interpretabilidade Farmacológica:** As métricas AUROC/AUPR em $pK_d \ge 7.0$ fornecem valor direto para triagem virtual e reposicionamento.
  - *Negativas / Mitigações:*
    - Custo computacional adicional de 5 dobras para GIN e CNN3 no Davis (~1 a 2 horas no Colab), plenamente comportado dentro da cota gratuita de GPU.



