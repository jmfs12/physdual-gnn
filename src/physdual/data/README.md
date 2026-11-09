# Módulo `physdual.data` — Fase 1

Responsável pelo download, limpeza e padronização dos datasets de afinidade droga-alvo (Davis e KIBA).
Inclui geração dos splits reprodutíveis (`random`, `cold_drug`, `cold_target`), featurização atômica/fragmentos de SMILES via RDKit e cache de embeddings proteicos do ESM-2 congelado.
