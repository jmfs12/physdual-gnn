# Guia de Execução no Google Colab — PhysDual-GNN

Este projeto foi desenhado para rodar no **Google Colab** (GPU NVIDIA L4 de 22.5 GB ou T4 de 16 GB) usando o **Google Drive** como armazenamento persistente.

---

## 1. Como Ativar a GPU no Colab

1. Abra o notebook `notebooks/00_colab_bootstrap.ipynb` no Google Colab.
2. No menu superior, acesse **Ambiente de execução → Alterar o tipo de ambiente de execução** (*Runtime → Change runtime type*).
3. Em **Acelerador de hardware**, selecione **GPU** (**T4** no plano gratuito ou **L4** no Colab Pro/Pay-as-you-go).
4. Clique em **Salvar** e execute a Célula 1 (`!nvidia-smi`) para confirmar que a GPU está ativa.

---

## 2. Montagem do Google Drive e Estrutura em `PHYSDUAL_ROOT`

O disco local da máquina virtual do Colab (`/content`) é **efêmero** e todos os dados nele são perdidos quando a sessão expira ou reinicia. Por isso, todos os artefatos persistentes vivem no Google Drive sob a pasta definida por `PHYSDUAL_ROOT` (padrão: `/content/drive/MyDrive/physdual-gnn`):

```text
/content/drive/MyDrive/physdual-gnn/
├── repo/                  # Clone do repositório Git (código-fonte e configs)
├── data/                  # Datasets brutos (raw/) e processados (processed/)
├── cache/
│   └── esm_embeddings/    # Um arquivo <sha256>.pt por sequência proteica
└── outputs/
    ├── checkpoints/       # Pesos do modelo salvos a cada época (last.pt / best.pt)
    └── results/           # Métricas, tabelas de avaliação e predições
```

A função [`get_project_root()`](file:///C:/Users/joao.francisco/Documents/dev/physdual-gnn/src/physdual/utils/paths.py) detecta automaticamente se o código está rodando no Colab (`'google.colab' in sys.modules`) e resolve todos os caminhos relativos de [`configs/base.yaml`](file:///C:/Users/joao.francisco/Documents/dev/physdual-gnn/configs/base.yaml) para `PHYSDUAL_ROOT`.

---

## 3. Estratégia de Cache e Desempenho de I/O (Drive $\rightarrow$ `/content`)

O Google Drive é montado no Colab via sistema de arquivos em espaço de usuário (**FUSE**). Embora seja excelente para persistência, **ler milhares de arquivos pequenos (`.pt`) diretamente do Drive a cada época de treinamento cria um severo gargalo de I/O**, deixando a GPU ociosa.

### Regra de Ouro para as Fases 1 a 7:
1. **Geração (uma única vez):** Gere os grafos processados e os embeddings do ESM-2 e salve-os permanentemente no Drive (`/content/drive/MyDrive/physdual-gnn/cache/esm_embeddings/`).
2. **Início da Sessão de Treino (cópia para SSD local):** Antes de iniciar o treinamento, copie o diretório de cache do Drive para o SSD local da VM (`/content/local_cache/`) e aponte a leitura do Dataset para `/content/local_cache/`:
   ```bash
   mkdir -p /content/local_cache
   rsync -a --info=progress2 /content/drive/MyDrive/physdual-gnn/cache/ /content/local_cache/
   ```
   *(Dica: empacotar o cache em um único arquivo `esm_embeddings.tar` no Drive e extrair em `/content/local_cache` com `tar -xf` é até 20x mais rápido que copiar arquivos individuais pelo FUSE).*
3. **Escrita de Checkpoints:** Continue salvando os checkpoints (`best.pt`, `last.pt`) e logs de resultados **diretamente no Drive**, pois são poucas escritas sequenciais ao final de cada época.

---

## 4. Como Retomar após Queda de Sessão

Sessões do Google Colab podem ser interrompidas a qualquer momento. Para retomar sem perder progresso:

1. **Bootstrap Idempotente:** Reexecute o `notebooks/00_colab_bootstrap.ipynb`. Ele monta o Drive, faz `git pull` sem sobrescrever seus dados, reinstala apenas os pacotes ausentes na VM recém-criada e revalida o ambiente.
2. **Checkpoints por Época:** A partir da Fase 2, o loop de treino salvará ao final de cada época `outputs/checkpoints/<experimento>_seed<S>_last.pt` contendo:
   - `epoch`, `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `best_val_metric` e estados de RNG.
3. Ao reiniciar o script de treino com `--resume`, ele detectará automaticamente o checkpoint no Drive e continuará da época seguinte.

---

## 5. Uso Seguro de Credenciais com Colab Secrets (`google.colab.userdata`)

**Nunca cole tokens do GitHub, chaves de API ou senhas em células de código ou arquivos de configuração.**

Caso o repositório seja privado ou você precise fazer `git push` a partir do Colab:
1. Clique no ícone de **Chave (Secrets)** na barra lateral esquerda do Google Colab.
2. Adicione um novo segredo chamado `GITHUB_TOKEN` com seu Personal Access Token (escopo `repo`) e habilite o botão **"Acesso ao notebook"**.
3. O `00_colab_bootstrap.ipynb` lê o token de forma segura em memória via:
   ```python
   from google.colab import userdata
   token = userdata.get("GITHUB_TOKEN")
   ```
   Sem expor a credencial no output das células nem no histórico do Git.

---

## 6. Limites do Plano Gratuito e Boas Práticas

- **Timeout por Ociosidade (~30–90 min):** Se a aba do navegador ficar sem interação, a VM pode ser desconectada. Mantenha a aba ativa durante extrações ou treinos longos.
- **Duração Máxima de Sessão (~12h no plano gratuito):** Planeje execuções por `seed` de forma modular para que cada corrida individual termine com folga dentro de 1–2 horas.
- **Reinício após Alteração de `numpy`:** Nunca atualize `torch`, `numpy` ou `pandas` manualmente. Caso algum pacote opcional altere a versão do `numpy`, o Colab exibirá um aviso de incompatibilidade C-API — clique em **Ambiente de execução → Reiniciar sessão** e reexecute o notebook a partir da Célula 2.
