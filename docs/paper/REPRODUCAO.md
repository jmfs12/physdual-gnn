# Reprodutibilidade do Pacote do Artigo (Fases 1 e 2)

- **Gerado em (UTC):** `2026-11-13T15:09:15Z`
- **Commit HEAD:** `c8ddd87`
- **Variável de raiz de artefatos:** `PHYSDUAL_ROOT` (padrão local: raiz do repositório Git; no Colab: `/content/drive/MyDrive/physdual-gnn`).

## 1. Ordem Exata de Execução

```bash
# 0. (Opcional, se os dados estiverem fora da raiz do repo)
export PHYSDUAL_ROOT=/caminho/para/physdual-gnn

# 1. (CPU, deterministico) Recomputar auditorias em CPU (noise floor, pKd>5, marginais/kNN v1, seq-id Davis)
python scripts/paper/cpu_audits.py

# 2. (Opcional, se necessário reagregar os JSONs de baseline a partir de outputs/results/runs/)
python scripts/run_phase2_baseline.py --aggregate-only

# 3. Pipeline completo do pacote do artigo (inventário, tabelas T1..T8, figuras F1..F6 e REPRODUCAO.md)
python scripts/paper/build_all.py

# 4. Validação automatizada de esquema, ausência de NaNs e bloqueio contra valores-p de cauda extremos
pytest tests/test_paper_pipeline.py

# 5. (Executar no Google Colab com Drive montado) Auditoria física dos .pt do cache ESM-2
PHYSDUAL_ROOT=/content/drive/MyDrive/physdual-gnn python scripts/paper/audit_esm_cache_colab.py
```

Ou invocando os módulos individuais do passo 3 na ordem exata:
1. `python scripts/paper/build_inventory_and_repro.py`
2. `python scripts/paper/build_tables.py`
3. `python scripts/paper/make_figures.py`

## 2. Ambiente de Execução dos Treinos (`docs/colab_env_snapshot.txt`)

```text
============================================================
PhysDual-GNN — Diagnóstico do Ambiente
============================================================
Data/Hora (UTC)    : 2026-11-09T17:01:32.119091+00:00
Rodando no Colab   : False ('google.colab' in sys.modules)
Sistema Operacional: Linux-6.6.159+-x86_64-with-glibc2.39 (x86_64)
Python             : 3.13.16 (/usr/bin/python3)
Raiz do Projeto    : /content/drive/MyDrive/physdual-gnn
Raiz do Repositório: /content/drive/MyDrive/physdual-gnn/repo
------------------------------------------------------------
PyTorch            : 2.14.1+cu130 (CUDA compilado: 13.0)
CUDA Disponível    : True (GPUs: 1)
  - GPU [0]        : NVIDIA L4 | VRAM: 22.03 GiB (CC 8.9)
nvidia-smi         : 0, NVIDIA L4, 580.82.07, 23034 MiB, 22561 MiB
RAM do Sistema     : total=52.96 GiB | disponível=50.65 GiB
Disco Local        : /content (livre=235.3 GiB / total=235.68 GiB)
Google Drive       : MONTADO em /content/drive/MyDrive (livre=223.54 GiB / total=235.68 GiB)
============================================================
# --- pip freeze ---
```

**Versões-chave do ambiente de treino (Colab NVIDIA L4):**
- `fair-esm==2.0.0`
- `lifelines==0.30.3`
- `matplotlib==3.10.0`
- `numpy==2.4.6`
- `pandas==2.2.3`
- `PyYAML==6.0.3`
- `rdkit==2026.9.1`
- `scikit-learn==1.6.1`
- `scipy==1.16.3`
- `torch==2.14.1+cu130`
- `torch-geometric==2.8.1`

## 3. Checksums SHA-256 dos Artefatos JSON de Entrada

Total de arquivos JSON auditados: **123**.

| Arquivo (relativo a `PHYSDUAL_ROOT`) | Tamanho (bytes) | SHA-256 |
| :--- | :---: | :--- |
| `data/processed/davis_ingestion_report.json` | 629 | `3a8b54148da4cdb3c7bca6eac9c9bab0b629cf89c13a2267e6c04d0283e42c52` |
| `data/processed/kiba_ingestion_report.json` | 637 | `ac396ea068124919509155b2f4a83f0b8146a6e7d0d46a7f96f749db542aad74` |
| `data/processed/splits/davis_cold_drug_fold0.json` | 319742 | `5c48263e008c2bf7bf576f049a6aa5380d66c556871d8b3d70b0159e52de9a83` |
| `data/processed/splits/davis_cold_drug_fold1.json` | 319742 | `251d3c8832dc74e656fd3b5a9d02c457e854c71a9e00d60a70b9fbb9a7e323b5` |
| `data/processed/splits/davis_cold_drug_fold2.json` | 319742 | `baa1b10cda20b23479fbffc823fe3f433b91935c009a24fd0dec20a4a5526ba4` |
| `data/processed/splits/davis_cold_drug_fold3.json` | 319742 | `5b3411a37fb95b8787afd9c1f833703dc4134ebe4f57436c5f70585e6386ca8d` |
| `data/processed/splits/davis_cold_drug_fold4.json` | 319742 | `4fb300ac8decb27f0b30de8dd0c514a165f1835b85442a99cda7e9ba45c9445a` |
| `data/processed/splits/davis_cold_drug_seed0.json` | 319556 | `66dd7cd7a2236b68fc5ab33f052a5fd03697cbd93fcaf08beaff30317487d036` |
| `data/processed/splits/davis_cold_drug_seed1.json` | 319556 | `08e7c0869fc607b39fb6e5b6bb9f1f37f75a63af8e1d8c969c2f22c7e8c52e9e` |
| `data/processed/splits/davis_cold_drug_seed2.json` | 319556 | `a07a43669933956a3c1081d6ce548e4aeea2fee2896335a9a38636d55411e455` |
| `data/processed/splits/davis_cold_drug_seed3.json` | 319556 | `5181aa77bc5d31d2565b29aa15e16a6ba8162a6a392893a12aecdf86a52da0fb` |
| `data/processed/splits/davis_cold_drug_seed4.json` | 319556 | `fb3958d9c8d55ba696696c25cbdb51e98f172c7179a4564964725e14d7a683a1` |
| `data/processed/splits/davis_cold_target_fold0.json` | 319748 | `44f886c90e2d0a45ca6b6f2689030e520c1d6bc721fe4ae4adbcc985424c4052` |
| `data/processed/splits/davis_cold_target_fold1.json` | 319748 | `d20103eaf351df99a30ffde9eaf33a68943fae8575eb2c8721ba28fd312c1595` |
| `data/processed/splits/davis_cold_target_fold2.json` | 319748 | `ad615b42068d43fa1b657dd7cbbb89299292c93b2aa290ad41c6f7b129ebcbe2` |
| `data/processed/splits/davis_cold_target_fold3.json` | 319748 | `987a494ea31c881936c4aa647b4551e4d727aafbe01ba806896af1666f56bdac` |
| `data/processed/splits/davis_cold_target_fold4.json` | 319748 | `9b6b13de989d09b054e7608b77b2d76b8a805d3165a55465d2a4a2fef8796ef7` |
| `data/processed/splits/davis_cold_target_seed0.json` | 319556 | `cf5666f9bc2934f47b1c17f0dba6cf9052c7863f8d3a8d451002149f68584e70` |
| `data/processed/splits/davis_cold_target_seed1.json` | 319556 | `016992e76015ef4b8c532460fdf0116d1d63ac288bdd77f2153c90ebb6040493` |
| `data/processed/splits/davis_cold_target_seed2.json` | 319556 | `9cd8c424ac6cfe99654cacb2bc7ee3acdfa6c064b9815146f4b8c93d7c10b048` |
| `data/processed/splits/davis_cold_target_seed3.json` | 319556 | `69f6064940078262c9a091ff257eb3ff5430632fcd4d240cc3b9a967ed6ad6cb` |
| `data/processed/splits/davis_cold_target_seed4.json` | 319556 | `e121dcf3c1a376c49e806fc50dc2f5de0d3bd748c0cc86cbeaad2f4af6cd3d0c` |
| `data/processed/splits/davis_random_seed0.json` | 319556 | `a22f23571e254733f43589ee35e1e56ad1fa2e321a7f7d460f8658cca3b4e143` |
| `data/processed/splits/davis_random_seed1.json` | 319556 | `a8b1f050caa26db2afa508f869005dc2d9281291f8b11ada3c8c03ddd56a6f7b` |
| `data/processed/splits/davis_random_seed2.json` | 319556 | `be861eb354c31a0a32704af1738161af4f2cf655f1b2a92a6bd11c265da15aeb` |
| `data/processed/splits/davis_random_seed3.json` | 319556 | `69c3d2b9a8b20a0dbbd809a8232408a38270641b6c2165708a75a4087eb040ee` |
| `data/processed/splits/davis_random_seed4.json` | 319556 | `933c46b594e8fb15d353a0d5867c2059d8f87c5c4ed2c880f4f7155c6caad6c3` |
| `data/processed/splits/kiba_cold_drug_seed0.json` | 1307988 | `1ba374179cadf0a1dbddd17ada484351269fb6abbbcae0764f3a2228f9ef29dc` |
| `data/processed/splits/kiba_cold_drug_seed1.json` | 1307988 | `54db17b316154038fc8df9efbe7c522bf130ec1864fbc0e8b1d2539abb82d930` |
| `data/processed/splits/kiba_cold_drug_seed2.json` | 1307988 | `cee96aecb1613f992cc8a6755a5a10b71f049e487a553354b8916a2e49ae0b4e` |
| `data/processed/splits/kiba_cold_drug_seed3.json` | 1307988 | `e6a1c160b46747ddf66b6d4106c6b0aff9bff6e03297294b9c205d361820dbc3` |
| `data/processed/splits/kiba_cold_drug_seed4.json` | 1307988 | `9ce881bc6161125cbbd4bfab0d369cf546608ae77741e3a8e12b101e02699890` |
| `data/processed/splits/kiba_cold_target_seed0.json` | 1307988 | `45f78f06362392ee0580f549efc55a5e9cdf599e257ccad4542d14466633f3b8` |
| `data/processed/splits/kiba_cold_target_seed1.json` | 1307988 | `369e5f7ea85b10832b4550749597d521737866040669501af24a1f8182cba212` |
| `data/processed/splits/kiba_cold_target_seed2.json` | 1307988 | `7ef0ee187d32b9c2f2793ea781a2ba284f44e590d80e527906004dcfa46540ca` |
| `data/processed/splits/kiba_cold_target_seed3.json` | 1307988 | `ab503014c378f6f0d90bb63878589188826eb5336ec39a2f75d1ddebc0c9dc82` |
| `data/processed/splits/kiba_cold_target_seed4.json` | 1307988 | `b5a249cbe9e01b4de0dede7f2c58d7755d37bad7da11ab7a2fd4237f36efbf82` |
| `data/processed/splits/kiba_random_seed0.json` | 1307988 | `cd57f53872266c1430d69ba7b4c07d0b03834afde9cb862183926201cc0126af` |
| `data/processed/splits/kiba_random_seed1.json` | 1307988 | `d2a1a911afa38a14558e6e2ad5faeba5490693c9d8785bd1a847a4545c43afa8` |
| `data/processed/splits/kiba_random_seed2.json` | 1307988 | `69c78b7e34640ea1b32f897c7fa26c319cbb18233691b5f19099f15e94e310e8` |
| `data/processed/splits/kiba_random_seed3.json` | 1307988 | `427f5e0b0bb1bda6f7395029597be78f495df5aa79fe4f678384dc32cdf18996` |
| `data/processed/splits/kiba_random_seed4.json` | 1307988 | `24ce643209963321aefa1de970a47b5b2be863c72457b0d548559eb5e0dc24af` |
| `docs/paper/cpu_audits.json` | 147907 | `b9feb38f606fddb7a4bda1ddcb6105a9f0c7399fa80add2f225b53050d45d211` |
| `outputs/logs/phase1_summary_report.json` | 21616 | `09ae6f898e996ce1799a65d3ac08bac9fe2ce57317d292eb820df630c31952c4` |
| `outputs/results/baseline_graphdta_cnn3_davis_cold_drug.json` | 18578 | `b9b3adf47b382ff46b21de100c13706b8b2031628438cc837799ba0f22e98974` |
| `outputs/results/baseline_graphdta_cnn3_davis_cold_drug_kfold.json` | 147454 | `ce11c694de060c139089314773e4988bcc779c3849fd28c95ea4480b1a7b6e6f` |
| `outputs/results/baseline_graphdta_cnn3_davis_cold_target.json` | 18567 | `3d232463c6d06b3339b342411bd72d64b32ada00a82703f420e4997512bd13ad` |
| `outputs/results/baseline_graphdta_cnn3_davis_cold_target_kfold.json` | 146833 | `7e14e0268c703b4109b1ca8671eb6d1be58f2e28cee7a376a19bf5635b17270d` |
| `outputs/results/baseline_graphdta_cnn3_davis_random.json` | 18591 | `79c5ccc52059ba64fd831190aff55da33d7bc8c6fb2a20f92ef95fc3f2637bed` |
| `outputs/results/baseline_graphdta_gin_davis_cold_drug.json` | 18556 | `e98c0183812d7ed980a0487c9854b84330a943307f0be0220e32444f70770ced` |
| `outputs/results/baseline_graphdta_gin_davis_cold_drug_kfold.json` | 222500 | `97ee74cc452e3671bdd8fcd699b51e04177a0d42b18b678460729425197b819b` |
| `outputs/results/baseline_graphdta_gin_davis_cold_target.json` | 18573 | `52fe6acedaa99cc8d3d5a40cffe45bef926c289297b5b11a0f6c5693e2f4f812` |
| `outputs/results/baseline_graphdta_gin_davis_cold_target_kfold.json` | 139155 | `6c652e8f5c9eccbfcfdaadfbeb918749ff592ebc817a301230498e1688c74968` |
| `outputs/results/baseline_graphdta_gin_davis_random.json` | 18572 | `edd784dff257d689c419b3031f1d4ce9f5453531385b15d83e94b951af7fc940` |
| `outputs/results/baseline_graphdta_gin_kiba_cold_drug.json` | 16534 | `68545695afb76ca4a7ee62436f097b39b49e111439e03ac555f5c27a4a6ddc6c` |
| `outputs/results/baseline_graphdta_gin_kiba_cold_target.json` | 16547 | `8cb2fa713c14246ddf73b9ff7d23a5c03d21fab501f4fdca30b2b9db1b35b2cd` |
| `outputs/results/baseline_graphdta_gin_kiba_random.json` | 87448 | `54e810340f4e5a6bf9c32ce5645a02421456936973f38350271d2879eaccdec8` |
| `outputs/results/phase2_timing_benchmark.json` | 444 | `401f6b684e352779ad24e6553486106b2579c2043071c9baf07dc0d5e1c99213` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_drug_fold0.json` | 2299155 | `cc1c2314b8e25ce682362c68431bf9d6527ce81ba942d9d7127ef4b10c903120` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_drug_fold1.json` | 2277726 | `6df509d31cb59feedce49dc5f50136ecd9d3cf92f63a0e9e302d28a2aca32a14` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_drug_fold2.json` | 2328288 | `a1f77413475609f43335fb3631e179ea839ddcdc1b6e5a45c4c232f0ac430aff` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_drug_fold3.json` | 2136741 | `b154278a4c7f06d8ef48d93c7347351867f0f1618194819ff1ea156be03a2947` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_drug_fold4.json` | 2089729 | `98e2d186c7410e3c8328d066effdfc3fa96310e350a0c928d14967329a54e338` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_drug_seed0.json` | 2332 | `4cb15117985ab015f7c70ece46f82d808646fee8ec2295c6958c2b8c60ba6f8c` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_drug_seed1.json` | 2343 | `f6947d12d7e9b224f302a7beeb80724dafb3208a78a902e9abb1d81c568634f4` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_drug_seed2.json` | 2343 | `53d8893680fcc68943dc14c8cc145d6253c9943a7c62f8e9fd82cb4028339f87` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_drug_seed3.json` | 2341 | `c75cfc5a7e022f1c706b09183bcd4902c36cb078a873c125d003fe18a47bf7c4` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_drug_seed4.json` | 2344 | `e1b7c28b7cb2a8f1d9e6812d6b1a8c487195c697582362022fb1a165a998c83b` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_target_fold0.json` | 2192689 | `1dcab1a158e630f58bcec35a27606d9da3a9989ff694c29803baed2fdde37ed5` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_target_fold1.json` | 1991780 | `17435e42b7f7edbefb1885d01f4c983b7e42772b7b3189677ae4cd4ff77fad62` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_target_fold2.json` | 2445180 | `430de6136f3241a13a396f2b67c243129483e0027f110cd71120c378f6a63dd9` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_target_fold3.json` | 2155392 | `28520aeb3bdf9e4f217e546f1abd08da05f449c9c2a7edd5acf00e5d2e0bad49` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_target_fold4.json` | 2344005 | `1de7c9e8ab2addb2ae6f876446cc53d01977afd2dafa704fb5fbb2c925739f16` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_target_seed0.json` | 2327 | `7c548bc4a577a9a93dc10cfa4836a390c7f478b0dc74bc23938273eac0ec067f` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_target_seed1.json` | 2338 | `d09d056c231cefadf0ae446e6dbc7c533b37729acb32da65855012d789a26613` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_target_seed2.json` | 2338 | `151413e786bd2bf9e6c45df0f3db15c69d556a1694a645bff38ac446844c439b` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_target_seed3.json` | 2336 | `61fc0b30b422e3b9519ed6fcf1f29d36113b33908975f33d8ed7bb19d6d0fe3a` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_target_seed4.json` | 2337 | `9284002d42f65b363db8b7a9bc25f01a2190865188667f2dbe7dfb41a8a54ba1` |
| `outputs/results/runs/graphdta_cnn3_davis_random_seed0.json` | 2330 | `073fb46ccb0c50dc6a1cfc4092e5b5fb34a745755e3676a87717aaddfc126c31` |
| `outputs/results/runs/graphdta_cnn3_davis_random_seed1.json` | 2337 | `3a41254aa58af4a826477c684720dd37e16d950332653ff914232103ec9ff3ef` |
| `outputs/results/runs/graphdta_cnn3_davis_random_seed2.json` | 2334 | `7bdbf3cc69bc9312d4d648905d8f351341c276ed7b56a4ae4c47f291965ab4c1` |
| `outputs/results/runs/graphdta_cnn3_davis_random_seed3.json` | 2334 | `f7b48883fc2e1e2008f052e528d73d3a54fece3ed9d3f759192da408aa204b73` |
| `outputs/results/runs/graphdta_cnn3_davis_random_seed4.json` | 2335 | `ed88e2fdb7582dae907d83c1534856e4cf450c050610ca4346e981c2a39af96b` |
| `outputs/results/runs/graphdta_gin_davis_cold_drug_fold0.json` | 2323220 | `79209647736115163915b21cf0626aa56198c368aaadb499efb6224f9192d8d7` |
| `outputs/results/runs/graphdta_gin_davis_cold_drug_fold1.json` | 2303459 | `b8ecad859e6f0507f6870752c0aa651a4bf7bfc1baf73aea5541cb4c04bca704` |
| `outputs/results/runs/graphdta_gin_davis_cold_drug_fold2.json` | 2354999 | `e56cbec573a967b6b46a58e1eb987df50442f1a2d31909ffa79f7ef361a80c0e` |
| `outputs/results/runs/graphdta_gin_davis_cold_drug_fold3.json` | 2128819 | `6a53869ab489fe5eccc6334b5d43fa286b985210bd58046eba4cc2e72e10ea75` |
| `outputs/results/runs/graphdta_gin_davis_cold_drug_fold4.json` | 2085564 | `707a6e9ffb8b0c0b7f3b9b04cd0ecbcd5ecbac2a8d5421faac22d0edb7a92e9d` |
| `outputs/results/runs/graphdta_gin_davis_cold_drug_seed0.json` | 2332 | `be0e8b18af579db7676c812011a4b766d708fe3c4ee87a201828a1d5a84ee6a8` |
| `outputs/results/runs/graphdta_gin_davis_cold_drug_seed1.json` | 2340 | `d6f1741efe2e8b93e136de22e35f0c904e0aa3d0bd3404930b6637bf351e274e` |
| `outputs/results/runs/graphdta_gin_davis_cold_drug_seed2.json` | 2337 | `ffb8da320f9cf40345d8681378f92dbb96c7050f47f97fe757638c7d4a8d200c` |
| `outputs/results/runs/graphdta_gin_davis_cold_drug_seed3.json` | 2337 | `e85961c17bec27a3a47dbe24131cc967c8531068e5fa1464750214da080b426b` |
| `outputs/results/runs/graphdta_gin_davis_cold_drug_seed4.json` | 2340 | `39581c78037dbc13e4db559bff0b6d54fb5ba72637524a2219805fb4990977db` |
| `outputs/results/runs/graphdta_gin_davis_cold_target_fold0.json` | 2216786 | `850bf2971185cfcca2724274b6b78d7226a436f76c95b38f765eb2718a97df31` |
| `outputs/results/runs/graphdta_gin_davis_cold_target_fold1.json` | 1981791 | `562a8c6460a2942a9a27c90d135efb074a51b2b0b3f55c48da5eb9141710ebd1` |
| `outputs/results/runs/graphdta_gin_davis_cold_target_fold2.json` | 2436736 | `e94b0b47d194aa9d1f6cdaa13f0c2bce9e3ba36d2655f73fad1492fcd23db4dc` |
| `outputs/results/runs/graphdta_gin_davis_cold_target_fold3.json` | 2149091 | `3931e90a2b9eb810dcd4ec2fe7682b10966e66a06614eb165c844fc63114133b` |
| `outputs/results/runs/graphdta_gin_davis_cold_target_fold4.json` | 2337905 | `54f1185425f3bdbd83c4c15eaabe746376bf72bae80fe278656109d011f24440` |
| `outputs/results/runs/graphdta_gin_davis_cold_target_seed0.json` | 2325 | `755f415b0f39edfbf3bd099ed798275612e94858b49452e39f743005b3e1bdec` |
| `outputs/results/runs/graphdta_gin_davis_cold_target_seed1.json` | 2341 | `1a082ae34f3d3c439f50d19e2d04172b858a8d8eb324daa6c4706f1351ad98e4` |
| `outputs/results/runs/graphdta_gin_davis_cold_target_seed2.json` | 2337 | `f2871f433e9d0e00c905f96cfc053243bf97a03f4747a23f6181e4f769e38f53` |
| `outputs/results/runs/graphdta_gin_davis_cold_target_seed3.json` | 2330 | `2e1f6f40959c295436e32db067f88130712d28b4813614d8b16ec52ae3db3e57` |
| `outputs/results/runs/graphdta_gin_davis_cold_target_seed4.json` | 2335 | `a7eb18c1b524a7f64b21dba3da4e703c7f9d2c4df02f08000c07fdadfabedb65` |
| `outputs/results/runs/graphdta_gin_davis_random_seed0.json` | 2333 | `6ad227af5a24aefa9bb9cc2e973d4a701d3f5eb2ac4930cf11ed742b27fa4ed5` |
| `outputs/results/runs/graphdta_gin_davis_random_seed1.json` | 2331 | `3cee2ca9f33ba9c03a856a72e3f2d78c98fadb39a82eb016950ca43bb2c382f5` |
| `outputs/results/runs/graphdta_gin_davis_random_seed2.json` | 2328 | `3ebdbe629266fcb0387aa6a8ae49d0c7f2b1e71f5d1031ea008cf8f852642829` |
| `outputs/results/runs/graphdta_gin_davis_random_seed3.json` | 2330 | `30011b404f401feb554eb3a84adafa5743659220852f0022c6a6a2fd2ed7d03d` |
| `outputs/results/runs/graphdta_gin_davis_random_seed4.json` | 2332 | `6226d90ba748659c07e1003a315b545b6593a91235fa9143fbab4b97922e9452` |
| `outputs/results/runs/graphdta_gin_kiba_cold_drug_seed0.json` | 2101 | `ea31d47a910f85e9276017122b6d35cf2f63d6068ec434097ce04630a4121fdd` |
| `outputs/results/runs/graphdta_gin_kiba_cold_drug_seed1.json` | 2091 | `ad284c60c294bbb11f1479da3827ca67078d30f5ad944fe1a0acc30d55a952d8` |
| `outputs/results/runs/graphdta_gin_kiba_cold_drug_seed2.json` | 2100 | `6ecacb45f35d04851eecbdd3dab2306c0fecf902429593e68d7e77a1ea0f72de` |
| `outputs/results/runs/graphdta_gin_kiba_cold_drug_seed3.json` | 2102 | `72a320d55315fcd82011392f568d18c6c1978c5a0ad5ab5bbba68a1a532be78f` |
| `outputs/results/runs/graphdta_gin_kiba_cold_drug_seed4.json` | 2094 | `65cf6fa40cd8a64d5caa8b0a070e396c7102c3b55edaf01d857c13ba1a556853` |
| `outputs/results/runs/graphdta_gin_kiba_cold_target_seed0.json` | 2105 | `36f1a735c878087743d3821646b01bfebe02132d433ff135885dab388eeb8310` |
| `outputs/results/runs/graphdta_gin_kiba_cold_target_seed1.json` | 2101 | `137471ee91c02b11f938201cdbcbe5ba23f78d2f4e76fbf3b1e02e3bab4d14c5` |
| `outputs/results/runs/graphdta_gin_kiba_cold_target_seed2.json` | 2096 | `ad2e790b9bab8d112fb70a37006f134f9cd67db7cb678ee7d3c7e7d3d490442b` |
| `outputs/results/runs/graphdta_gin_kiba_cold_target_seed3.json` | 2105 | `30c20eed24c89e61ceaa3d0823a8e2d7dbfdbc9156946d1ebd98bcc47e731bc3` |
| `outputs/results/runs/graphdta_gin_kiba_cold_target_seed4.json` | 2101 | `1ee1e89baf095e398db37af4759bf58a67e6c86d5e86d6bff99e406c3b6ef1a1` |
| `outputs/results/runs/graphdta_gin_kiba_random_seed0.json` | 2092 | `2d76d8e66f995ccc5e79242756533f1acd1e6bf96056f87632d7a1cb9fbbb872` |
| `outputs/results/runs/graphdta_gin_kiba_random_seed1.json` | 4504466 | `f0a098c199cc9eac53c0669bbf5ea1e9836c4a9c1e5a7d14b7ba8e55beb581e8` |
| `outputs/results/runs/graphdta_gin_kiba_random_seed2.json` | 2097 | `ef44c7ba190cfe107fd3d56891f4f3a4e24c77ddd130494fc82509f8a24f0b96` |
| `outputs/results/runs/graphdta_gin_kiba_random_seed3.json` | 4461030 | `e838ed059556a7449a4b91af0b77e8db4a8cbd970cda118658717d2c064d216b` |
| `outputs/results/runs/graphdta_gin_kiba_random_seed4.json` | 2097 | `ed44e428a423c6de242e856fb3dcf1a0c38177f58f9fcba7647c0ade7aa0b9ef` |
