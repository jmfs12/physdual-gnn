# Reprodutibilidade do Pacote do Artigo (Fases 1, 2 e 3)

- **Gerado em (UTC):** `2026-11-15T03:43:27Z`
- **Commit HEAD:** `461421b`
- **Variável de raiz de artefatos:** `PHYSDUAL_ROOT` (padrão local: raiz do repositório Git; no Colab: `/content/drive/MyDrive/physdual-gnn`).

## 1. Ordem Exata de Execução

```bash
# 0. (Opcional, se os dados estiverem fora da raiz do repo)
export PHYSDUAL_ROOT=/caminho/para/physdual-gnn

# 1. (CPU, determinístico) Recomputar auditorias base Fases 1-2
python scripts/paper/cpu_audits.py

# 2. (CPU, determinístico) Recomputar auditoria forense e bootstrap pareado da Fase 3
#    (ou fracionado em --step meta, --step v2, --step v1_cold, --step v1_random)
python scripts/paper/phase3_audits.py --step all

# 3. (Opcional — recomendação Fase 3 para atualizar outputs/results/phase3_davis_paired_bootstrap.json)
# python scripts/run_phase3_paired_bootstrap.py --dataset davis

# 4. Pipeline completo do pacote do artigo (inventário, T1..T15, figuras F1..F11 e REPRODUCAO.md)
python scripts/paper/build_all.py

# 5. Validação automatizada de esquema, completude, ausência de NaNs e proteção contra valores-p extremos
pytest tests/test_paper_pipeline.py

# 6. (Opcional Colab Drive) Auditoria física dos .pt do cache ESM-2
PHYSDUAL_ROOT=/content/drive/MyDrive/physdual-gnn python scripts/paper/audit_esm_cache_colab.py
```

Ou invocando os módulos individuais do passo 4 na ordem exata:
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

Total de arquivos JSON auditados: **275**.

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
| `data/processed/splits/davis_cold_target_seqid30_fold0.json` | 350451 | `e87688be77e95eef6c3386276888c0265850dcf569e6cec3ea070d4622d19bae` |
| `data/processed/splits/davis_cold_target_seqid30_fold1.json` | 350452 | `44fa6c886f1194006e2d55c25f144714cb8b4ac47e0de2e5e3ecd71e0fd8c40b` |
| `data/processed/splits/davis_cold_target_seqid30_fold2.json` | 350452 | `aa0820ce9d5664a9ca6a04020344dc679925cce8d3b74348cbe7e4c19a745c20` |
| `data/processed/splits/davis_cold_target_seqid30_fold3.json` | 350452 | `20ceb2e3f4a5906dba818bc2aa59f1f81f34de69639f751135335113abae5e9e` |
| `data/processed/splits/davis_cold_target_seqid30_fold4.json` | 350451 | `be8e18c08d9fa7526f9bde2e9cd9229ddfb93d92d39007251ba087d91f15a127` |
| `data/processed/splits/davis_cold_target_seqid40_fold0.json` | 350454 | `9a9e12ed4c47502ef25c0be24aad67b6a378b6158a7a758574d3c9a6c63d0327` |
| `data/processed/splits/davis_cold_target_seqid40_fold1.json` | 350454 | `922049d2be648226244bb52943a023a67a1fcb20e21f6ff52da438972cd95a34` |
| `data/processed/splits/davis_cold_target_seqid40_fold2.json` | 350454 | `513da1a7ee5380200f79064fbe3f72abb0977188137b1a7b0b9983db053b6997` |
| `data/processed/splits/davis_cold_target_seqid40_fold3.json` | 350454 | `f961e4bc3618deb3b72f9b738e7bb30cc1e6accf585a068728540c2f3d46c720` |
| `data/processed/splits/davis_cold_target_seqid40_fold4.json` | 350454 | `4d79655d4ad8e32ca1c04825012ea00f4481686e6846962433c67f21122c2b00` |
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
| `data/processed/splits/kiba_cold_target_seqid30_fold0.json` | 1427081 | `891a43640e550ea6ba576751591ad63087aa62bf5c7f63237c220db31cbc2890` |
| `data/processed/splits/kiba_cold_target_seqid30_fold1.json` | 1427082 | `fe108140bf30ac8b2575a81adc3deaa5408570ba7a92a1b3c4b2909f68063b70` |
| `data/processed/splits/kiba_cold_target_seqid30_fold2.json` | 1427082 | `691a7a6ced4f907b26c5610579876c8b838d5e565ce861d1000cb6e22effdb7f` |
| `data/processed/splits/kiba_cold_target_seqid30_fold3.json` | 1427082 | `59cbb34bdbed025d9c76deae4e284af7603e13ef2fa5806ffa9bbd4af0c01835` |
| `data/processed/splits/kiba_cold_target_seqid30_fold4.json` | 1427081 | `ff874745bcec8133d4364a51d4635e597d3143696031d15adc8ef6aa0cf6616d` |
| `data/processed/splits/kiba_cold_target_seqid40_fold0.json` | 1427082 | `511835e7b294442e3b1e68dde48f421b2c234c5f21d4303c147fb89892abf9c0` |
| `data/processed/splits/kiba_cold_target_seqid40_fold1.json` | 1427082 | `3fed439a5633fb98aa7db4b9c57e2eeb1bc8c9e663daab13f4cafaaad50c0b97` |
| `data/processed/splits/kiba_cold_target_seqid40_fold2.json` | 1427082 | `2b7e2301063534b8d8b88839e521087c0eb9dad5b93b8bf713d6935b631e6490` |
| `data/processed/splits/kiba_cold_target_seqid40_fold3.json` | 1427082 | `a74f77641295ecd80c96758303bccfd0566964f61d9b8e914a6d615407dbfd30` |
| `data/processed/splits/kiba_cold_target_seqid40_fold4.json` | 1427082 | `7006f4cd632e3f768d9f1c1c35af5341906dd4d5020b8eba3cf83d1c6f036c3e` |
| `data/processed/splits/kiba_random_seed0.json` | 1307988 | `cd57f53872266c1430d69ba7b4c07d0b03834afde9cb862183926201cc0126af` |
| `data/processed/splits/kiba_random_seed1.json` | 1307988 | `d2a1a911afa38a14558e6e2ad5faeba5490693c9d8785bd1a847a4545c43afa8` |
| `data/processed/splits/kiba_random_seed2.json` | 1307988 | `69c78b7e34640ea1b32f897c7fa26c319cbb18233691b5f19099f15e94e310e8` |
| `data/processed/splits/kiba_random_seed3.json` | 1307988 | `427f5e0b0bb1bda6f7395029597be78f495df5aa79fe4f678384dc32cdf18996` |
| `data/processed/splits/kiba_random_seed4.json` | 1307988 | `24ce643209963321aefa1de970a47b5b2be863c72457b0d548559eb5e0dc24af` |
| `docs/paper/cpu_audits.json` | 147907 | `b9feb38f606fddb7a4bda1ddcb6105a9f0c7399fa80add2f225b53050d45d211` |
| `docs/paper/phase3_audits.json` | 648787 | `2f8888d31334583ec5ecbb89336fb70d2018203bbb7815360fb6be75c3c0a09c` |
| `outputs/logs/davis_seqid_cluster_map.json` | 577846 | `93df09eb21f9d546760f640f3aa25a0fa1ec6c454f0da24582507dd04ee81da6` |
| `outputs/logs/davis_seqid_splits_summary.json` | 5532 | `60f5f828e4466911ab412827b6cbcc2ffa16597cfcc787ef1c9ed184d68e1867` |
| `outputs/logs/kiba_seqid_cluster_map.json` | 341913 | `de2c01c36a03e29b97b135fc6dac428d181e838a1be27ac66f8bd167f481031b` |
| `outputs/logs/kiba_seqid_splits_summary.json` | 3937 | `3bede4265a0b8a71fb468953e266025ca9b18eb3b3a4b7d4284b79d76016cfac` |
| `outputs/logs/phase1_summary_report.json` | 21616 | `09ae6f898e996ce1799a65d3ac08bac9fe2ce57317d292eb820df630c31952c4` |
| `outputs/results/baseline_graphdta_cnn3_davis_cold_drug.json` | 18578 | `c7c3f2228b17f476ff44241af42710da51619433881802212f2c23197a07061f` |
| `outputs/results/baseline_graphdta_cnn3_davis_cold_drug_kfold.json` | 147454 | `6c8321cf323dea1e0c305aa25c55c36f6c9bdd503fbe90a545a55679fe2289a5` |
| `outputs/results/baseline_graphdta_cnn3_davis_cold_target.json` | 18567 | `df3431be77fe34fd418c4567123f01352c7b0414af69a376212ea257bebcb693` |
| `outputs/results/baseline_graphdta_cnn3_davis_cold_target_kfold.json` | 146833 | `836378028324210a14ed24101f2d80d9813c80c71c0b76498131e241662dc3df` |
| `outputs/results/baseline_graphdta_cnn3_davis_cold_target_seqid30_kfold.json` | 144647 | `c527c17807859e00c5827f58874ca63c3e8fb36a49ce29598d393fca557051db` |
| `outputs/results/baseline_graphdta_cnn3_davis_cold_target_seqid40_kfold.json` | 138296 | `342d505bb3c852c5ca5280e96d02094c448bb8b8fd6077042670c733e06da27c` |
| `outputs/results/baseline_graphdta_cnn3_davis_random.json` | 18591 | `a0dafe08ed19c2440a59bab580422ad9354872b1d9a88bedebd2582ef67333db` |
| `outputs/results/baseline_graphdta_gin_davis_cold_drug.json` | 18556 | `8cbe6009cef1b01ce3efab599fe5cccb82a39d9bd1549bc4294a46834372319c` |
| `outputs/results/baseline_graphdta_gin_davis_cold_drug_kfold.json` | 222500 | `632cc7dc80ae59e4bc0b45199031cfaf868da2dd8e6ab8a2fa5f8019a72651d7` |
| `outputs/results/baseline_graphdta_gin_davis_cold_target.json` | 18573 | `2c72051ad9b54fa7376bb207794a947875bf9c0566ecb1cb2d31d5008a0a048c` |
| `outputs/results/baseline_graphdta_gin_davis_cold_target_kfold.json` | 139155 | `4971a90346f7d8f781867ac50ff44c0fe02538815aafdfe25b8077585952ab28` |
| `outputs/results/baseline_graphdta_gin_davis_cold_target_seqid30_kfold.json` | 93248 | `723637bcdc621f15ca129a5ea1504313f43b13834421c59af951db955cd025fe` |
| `outputs/results/baseline_graphdta_gin_davis_cold_target_seqid40_kfold.json` | 96703 | `28d9caf50fe91cde25be0cac069928081314232bd205124b160ada0be0c4fe42` |
| `outputs/results/baseline_graphdta_gin_davis_random.json` | 18572 | `244365c4d6053e596c59a35836e0ee50a28d3f0f22e2ac2218ea34e9ebd170ab` |
| `outputs/results/baseline_graphdta_gin_kiba_cold_drug.json` | 16534 | `dcec4e14431349790a5148ceecc6e37def46715570479e19e9aac9e007886899` |
| `outputs/results/baseline_graphdta_gin_kiba_cold_target.json` | 16547 | `55baffb44713ff3eef9cf0ba9d51206dcf41b70613f769cb4a2f21d30cd7939e` |
| `outputs/results/baseline_graphdta_gin_kiba_random.json` | 87448 | `23ee01dea5e235fc76b51939fe2d1a7089f288385022864939df66992f82a5db` |
| `outputs/results/kiba_cpu_classical_baselines.json` | 63074 | `3414df48cb2ba5140a044458941a6b738cd7cd3acf3ff5d50bd1a6c35586fc30` |
| `outputs/results/phase2_timing_benchmark.json` | 444 | `401f6b684e352779ad24e6553486106b2579c2043071c9baf07dc0d5e1c99213` |
| `outputs/results/phase3_davis_paired_bootstrap.json` | 101792 | `e975fe2ef24b49c0447a81b004a7f6fab95b297d681df7224c48ed75e3ff0418` |
| `outputs/results/phase3_esm_cpu_baselines.json` | 88087 | `c52850af1f80f4ab20a98dc5e1e584a231e7dbaa62f8a0d5eadce71ef7dd0092` |
| `outputs/results/phase3_kiba_paired_bootstrap.json` | 34397 | `fedcfb78546e6adc9b231f51c1eba538fbe72cdae296e9c13fa27cb3aae681a7` |
| `outputs/results/phase3_physdual_ecfp_esm_mlp_davis_cold_drug_kfold.json` | 108520 | `c6cfaee4d96451721d185c81aabc1c8852404a5ac28f0cead2708b3db5ed7e18` |
| `outputs/results/phase3_physdual_ecfp_esm_mlp_davis_cold_target_kfold.json` | 287044 | `abf26948456555ef18779653915edd312bcaaa54c00b19c4548a43cca0bb566d` |
| `outputs/results/phase3_physdual_gin_esm_mlp_davis_cold_drug_kfold.json` | 193941 | `c491395ce3f34a9cbb47db3b8bbab978be44f74e96cebae30cd6e8df8d39b0e8` |
| `outputs/results/phase3_physdual_gin_esm_mlp_davis_cold_target_kfold.json` | 322923 | `b501c6bc6346e87321e23f0cc18bdc717c16d97295ab82ba4517b334a2748ffb` |
| `outputs/results/phase3_physdual_gin_esm_mlp_davis_cold_target_seqid30_kfold.json` | 192191 | `9bc251f3b09ac84514c0ca90985c157059a69075990d97e18f5269a69a617c10` |
| `outputs/results/phase3_physdual_gin_esm_mlp_davis_cold_target_seqid40_kfold.json` | 187883 | `e51db0a1f5c8f550452b1cd7550de0b3979da7696b5b9d9f48e4a9aff1ddcb90` |
| `outputs/results/phase3_physdual_gin_esm_mlp_davis_random.json` | 458734 | `8faa3b0c6d4b1e3dc3a9cdb66c7a0b8d927a2b5bb68f17955d14938a092c1bd6` |
| `outputs/results/phase3_physdual_gin_esm_mlp_kiba_cold_drug.json` | 244118 | `5d95500ceef5256d90d4557f980539545ffa740311c89278459e06ed5c036816` |
| `outputs/results/phase3_physdual_gin_esm_mlp_kiba_cold_target.json` | 307609 | `d162d1dc3bfdfe3a79cad614be2ad8ac50c76fd615fa48290d42dfed3b00e10d` |
| `outputs/results/phase3_physdual_gin_esm_mlp_kiba_cold_target_seqid30_kfold.json` | 165733 | `652268ca0f9d6fe55fb587e0fdbcbbe596f4c105417be49f0213cd68cf9bca09` |
| `outputs/results/phase3_physdual_gin_esm_mlp_kiba_cold_target_seqid40_kfold.json` | 199562 | `26531238643b19c459afc53fee09d785d0047501d132d90d35eca323534f75ad` |
| `outputs/results/phase3_physdual_gin_esm_mlp_kiba_random.json` | 549661 | `622b4a8f3fec744e8aa9480c618480981d1429963f3f3a4a0def43b1c243c996` |
| `outputs/results/phase3_physdual_gin_esm_mlp_shuf0_davis_cold_target_kfold.json` | 113378 | `19ca90783c82b95a6c2167abc6bbd73f3c9b46a944487db6c207f5dbe8882d08` |
| `outputs/results/phase3_physdual_gin_esm_mlp_shuf1_davis_cold_target_kfold.json` | 124175 | `acf302d0ad5dcb2a4cde57faab6a51dc9956cabb6f28b5f9090439d9e8f74eed` |
| `outputs/results/phase3_physdual_gin_esm_mlp_shuf2_davis_cold_target_kfold.json` | 109780 | `8de995330014111019dc5a9fff6ed9292788240829d7305a006c904687b8719f` |
| `outputs/results/phase3_physdual_gin_esm_mlp_tsoff1000_davis_cold_drug_kfold.json` | 27795 | `9fdea6cd963077d0c3357e47db34a3ea705bc31bfc1b88dc41cac9c8eeaa58f3` |
| `outputs/results/phase3_physdual_gin_esm_mlp_tsoff1000_davis_cold_target_seqid30_kfold.json` | 246626 | `f0b6783af7559e8e0ffddd95d049de4b30c56f0fd83bb0206a56c6c8641d541c` |
| `outputs/results/phase3_seqid_classical_baselines.json` | 26779 | `7469cced8678671286af2855c65bd5324bd4ca13939f4d0624f2b9ef12a0ea1f` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_drug_fold0.json` | 2224075 | `48f662d146e245ecb0f5e14d32e25037792debb1d03573fd48c6e4053f85d695` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_drug_fold1.json` | 2202652 | `6789b9366472026d79025756c651f1dc684418f389b8f44c5b82c6d72ee4b70f` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_drug_fold2.json` | 2252680 | `ee1747cdf103f53133ea2d008f8991289adf1c40a95086fdb2dfae4471ef9e1c` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_drug_fold3.json` | 2066491 | `f9d9b4fab371dfea43c35bf2c7651dadcf03826726f190181e6bbd6060d15099` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_drug_fold4.json` | 2019959 | `91a6b64fb7507396ebd90b354233ceba06477a595308a4c5a72e947f9c945758` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_drug_seed0.json` | 2211 | `fca5526e7769540896f27402e16a83c9e712d2725fe22250029694969000acef` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_drug_seed1.json` | 2222 | `bb463b8537babeda2b021239b5b860ff79dc53df49486e4fa5acc32310b34a9a` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_drug_seed2.json` | 2222 | `730cf3e8f9ef8c9e0fd621f09a17019a77927d47f35b673f3553f40713ac67bd` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_drug_seed3.json` | 2220 | `9425b5f56ab33c5cbc7b18296ad65f77db06ee11f5904ddc6184a46f76a14b81` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_drug_seed4.json` | 2223 | `87ef09dfa4bad5e1b772c3e92ed2c8308e8664c390084abf3e5143c0f3880ecd` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_target_fold0.json` | 2120591 | `ae6c195421751be6db56a747a01cc9ff3fc4a3df18ca1849f6112dce6a3a1893` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_target_fold1.json` | 1926198 | `dd63487b23ffe3f0bbdf1526c60c4e66116db7d5b71b381dab88330a15bfb649` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_target_fold2.json` | 2365030 | `4ddb01623600ba76ffedc920ba579ac49ee80cac8ed9d62dd87377be03b89afb` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_target_fold3.json` | 2084290 | `fd802fc496d92be2d8811f417f79c3ae556b5897fae23e1f1735e3809a94ae0b` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_target_fold4.json` | 2267179 | `6f30f645b2013cecf5a170c56ed042bd7d0ce8d022a23ad2343d99d8c53c401c` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_target_seed0.json` | 2206 | `c23650f12197e4306aee5c7ba7e134f1f500f72005f1ba4fa7f43af644838a58` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_target_seed1.json` | 2217 | `4141b87703ab2305c221ac04744af07ed1daeac16fd9d3c26c84c6061ddbeef3` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_target_seed2.json` | 2217 | `2e0ae79f9a1ffba4adb69b71a125e3db4abc69407a44434e0ccec12edbecebce` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_target_seed3.json` | 2215 | `49670fae6d6d860e54f4323fbf66366ff0b6a801fbb78b302ac845799a2c0998` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_target_seed4.json` | 2216 | `8e9a48929e3ef37d7396912c0b29bfd71dc788e1c606fc37f3793a3429e58eec` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_target_seqid30_fold0.json` | 5580403 | `037c469fd7d749ec62ee802129d0aee36780479e4002a2cfb0345856d19aa8fc` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_target_seqid30_fold1.json` | 7106114 | `062adda843690b9a3c28185652e2492af7dbaab248b5704fd6dd8ccaf0bff786` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_target_seqid30_fold2.json` | 6772144 | `3a893b407bd0c6e1dcfc79946a11f3641c304cea235535ebeb24ca3e6a096688` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_target_seqid30_fold3.json` | 7723066 | `f59d875b7dec714caa2f8d6ecfbd64f8c55e771fcc757ddc15e85b2280319551` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_target_seqid30_fold4.json` | 7303903 | `1be2efbfac0bf8a156a288e26dd6dfb1816e8dc4ecd00c598d71aaa09cefe4f7` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_target_seqid40_fold0.json` | 6392223 | `c3e15af44fb72163299e6f2c7bf0b55e2d21f054d2593cb0c14aa069edd84f2a` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_target_seqid40_fold1.json` | 7565191 | `55592904e1741a27d107646197df7030d8e58b49a508ccda67616b2d2ce886a3` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_target_seqid40_fold2.json` | 7582650 | `df8981481f4578d4f2899fa0b7e0d15996ca0f755819afca168f970f67fe2b7d` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_target_seqid40_fold3.json` | 6856361 | `1d93569961c86d1a5946ac6f2bb87e67abd48a3be02ae7de90ccb406ddd667d4` |
| `outputs/results/runs/graphdta_cnn3_davis_cold_target_seqid40_fold4.json` | 6093387 | `0c21a104c4d61cf025203528f10bf0bef60970f697deda12dedbda1813ee77c7` |
| `outputs/results/runs/graphdta_cnn3_davis_random_seed0.json` | 2209 | `517fd9048de3ea62435f1b2c69bbcdca942d6abcd1485df3acc212d3994e38da` |
| `outputs/results/runs/graphdta_cnn3_davis_random_seed1.json` | 2216 | `634e3f9187905830575d6f6f0a44ef865bf2ff699f4458a3b0ce34c76175f611` |
| `outputs/results/runs/graphdta_cnn3_davis_random_seed2.json` | 2213 | `324d9e2025526679db8e77cea39a1e010384981872b7f7ddaedfb3d9a6377f45` |
| `outputs/results/runs/graphdta_cnn3_davis_random_seed3.json` | 2213 | `583bce5b80ca08724b25fd3d379f8d22061ba94f910db27fd59629a199fc88f2` |
| `outputs/results/runs/graphdta_cnn3_davis_random_seed4.json` | 2214 | `6be48db4b9c2570d5846e7ddc3113c104d807f97cc978cf6efcd34b02f4d7216` |
| `outputs/results/runs/graphdta_gin_davis_cold_drug_fold0.json` | 2247180 | `a8dea8487ef6fddde893e85dd771cfe3ef86f1d8f81287fb8f91d1111cf02b15` |
| `outputs/results/runs/graphdta_gin_davis_cold_drug_fold1.json` | 2227365 | `6e9f2f2052a5db528914208eda672b0d62e81147c80d5dfce70ca6c7735576c5` |
| `outputs/results/runs/graphdta_gin_davis_cold_drug_fold2.json` | 2278335 | `2ed5dcb3e43728a6ed93a23b6619e2f8fc36255bce7dd8238318ac2737051daf` |
| `outputs/results/runs/graphdta_gin_davis_cold_drug_fold3.json` | 2058881 | `f6f237b4b7345e23c0133defcfb2d9f8b33514fd1709bc82111283a2580c63c6` |
| `outputs/results/runs/graphdta_gin_davis_cold_drug_fold4.json` | 2015956 | `1baed7f4f3f51f98e07f1bf28d8cffaa58741d02a1e9554ac523024d85a38824` |
| `outputs/results/runs/graphdta_gin_davis_cold_drug_seed0.json` | 2211 | `bd3fcfb983120df0e067cfbc595cdefb4daac23c97aa155e62470fc861f4c644` |
| `outputs/results/runs/graphdta_gin_davis_cold_drug_seed1.json` | 2219 | `b55a8bedbabdda1e2ff21a0c9236fe437276191093bb489114c89fdd146367b5` |
| `outputs/results/runs/graphdta_gin_davis_cold_drug_seed2.json` | 2216 | `c1b87a133b55c4bd44d874b3b5d5198439f342487c24ebd8deb1cf509a359faf` |
| `outputs/results/runs/graphdta_gin_davis_cold_drug_seed3.json` | 2216 | `e7981db4825172e616c07b742d3e64396e1f4d69d2fe0647c5f1d54b84dd0f2b` |
| `outputs/results/runs/graphdta_gin_davis_cold_drug_seed4.json` | 2219 | `627632f50f0f7fcbd4eb415460322f3a09258145502bd089bcb6f1c35696dfff` |
| `outputs/results/runs/graphdta_gin_davis_cold_target_fold0.json` | 2143734 | `769a5a83726c5dff401a36cb1c6a18f2e9d6de5b90721a18584cafa112d3fc05` |
| `outputs/results/runs/graphdta_gin_davis_cold_target_fold1.json` | 1916605 | `095775f67825cd424afa02f34c2b2e4d1b182fdfb93b07c10797f9c9de835c49` |
| `outputs/results/runs/graphdta_gin_davis_cold_target_fold2.json` | 2356922 | `87d62f9ce99fc9150085cb5cc8446a8ec051ce2fd876283ba5e2db19cc6c2494` |
| `outputs/results/runs/graphdta_gin_davis_cold_target_fold3.json` | 2078235 | `5134fc18b432ddc058d55b9841574fddfe9e96e417e18e5f7fc11b9533c761cf` |
| `outputs/results/runs/graphdta_gin_davis_cold_target_fold4.json` | 2261319 | `39da2ac5f72e75fddde9a4899eeb008cb8cb2cc43303b255773b8031d047a3e3` |
| `outputs/results/runs/graphdta_gin_davis_cold_target_seed0.json` | 2204 | `bcf455c8fdef47cf74e74740bbe46bbf6d4810e8b68d31122b699976fae59af0` |
| `outputs/results/runs/graphdta_gin_davis_cold_target_seed1.json` | 2220 | `5e9939d15d8ada04f2280183cc9249006e0758359c1074aae8841ea3d461379d` |
| `outputs/results/runs/graphdta_gin_davis_cold_target_seed2.json` | 2216 | `81a78ac7aa3f75234c1428168d5173fb3d795479df87896f8697b6fbf932a2ff` |
| `outputs/results/runs/graphdta_gin_davis_cold_target_seed3.json` | 2209 | `a5f84138db096b2dde73a8dfcacfd9931c51fdf768e51d94b6008bac787cf307` |
| `outputs/results/runs/graphdta_gin_davis_cold_target_seed4.json` | 2214 | `7ed449e886480500ab441fdedda92c95028c7a7f7a28052040346a1791613054` |
| `outputs/results/runs/graphdta_gin_davis_cold_target_seqid30_fold0.json` | 5564755 | `84c9020c1896259eaf56499e8de63c002fa7f35e3eb70e00daba507422019332` |
| `outputs/results/runs/graphdta_gin_davis_cold_target_seqid30_fold1.json` | 7096802 | `2c6d66871aa10d3fc9a75dfec288ff4f1e915b5446313160e84e2913f7d1b6a1` |
| `outputs/results/runs/graphdta_gin_davis_cold_target_seqid30_fold2.json` | 6766009 | `0d1e71080f9f5719cd35b2dc1825b1cc52a780fd33cdd3ea2c199ad901a3eb32` |
| `outputs/results/runs/graphdta_gin_davis_cold_target_seqid30_fold3.json` | 7717499 | `e36321a67f6e4a29bab680dab52718d53d17cc0cdf7c4364eb4d16631637e9aa` |
| `outputs/results/runs/graphdta_gin_davis_cold_target_seqid30_fold4.json` | 7296421 | `b10eddd2a96f2814e102a554b30b8e70a79ec0e58c355af8c72475cafe6b9cca` |
| `outputs/results/runs/graphdta_gin_davis_cold_target_seqid40_fold0.json` | 6383922 | `de7c0f01fedf78671f74dbc4d21f7b4adfb18d98b30a440fa4f52869e6bd0d0e` |
| `outputs/results/runs/graphdta_gin_davis_cold_target_seqid40_fold1.json` | 7560042 | `42e2e4ca5c6c464658658c1d1ba8f4a1feef9289a98dff064f1e2ccec845eada` |
| `outputs/results/runs/graphdta_gin_davis_cold_target_seqid40_fold2.json` | 7575437 | `d096f8db3a3fb57b4d0c8dfedba42e1b82dd999fa9f4663001620ac669ba30b0` |
| `outputs/results/runs/graphdta_gin_davis_cold_target_seqid40_fold3.json` | 6848335 | `80bcfaf89af0776857e17bf38ed8c17dfea1d12cd90cee10fa4a987788dbb617` |
| `outputs/results/runs/graphdta_gin_davis_cold_target_seqid40_fold4.json` | 6086452 | `ff1e71161bbabda185ba7d468ebd7a639c6ae7b8a41ebba40078e7f22db76fec` |
| `outputs/results/runs/graphdta_gin_davis_random_seed0.json` | 2212 | `33ee95b322e2d10405cc711b4d2c69bbdbc98f3fa4f289c26cab9a4c68537e75` |
| `outputs/results/runs/graphdta_gin_davis_random_seed1.json` | 2210 | `a6ff9f67f485116782abc0911a988c2f8d6a765502309bc691cb1ddf253a1477` |
| `outputs/results/runs/graphdta_gin_davis_random_seed2.json` | 2207 | `8be2fadd511a5bb35f6fb53b3cba91849e08c9a6ac84c649e590b46b207ac2cc` |
| `outputs/results/runs/graphdta_gin_davis_random_seed3.json` | 2209 | `3c60208178409d9254c38b15b705f6174b0272ad184693aa614ac228ba093131` |
| `outputs/results/runs/graphdta_gin_davis_random_seed4.json` | 2211 | `730676f0877b1d7184cf0f4282d0381cdd7eca6cf52360c32486fbf57992595a` |
| `outputs/results/runs/graphdta_gin_kiba_cold_drug_seed0.json` | 1988 | `a5735df0eb30327735e8be15b3fd9e7fccf13597fc36c3a9704510603ce82afb` |
| `outputs/results/runs/graphdta_gin_kiba_cold_drug_seed1.json` | 1978 | `7692489278da1a85662d01cc57cbf9ef912f1e5b10b4f465c7561baf6138f803` |
| `outputs/results/runs/graphdta_gin_kiba_cold_drug_seed2.json` | 1987 | `4c44d88c8f519f8c68e3cff762f14d0500974f7d0f9f8cbf9300bf205fd91019` |
| `outputs/results/runs/graphdta_gin_kiba_cold_drug_seed3.json` | 1989 | `a6c93ce8e8fd654fba41a3fd3886743e607b65b35fd486a92e9173424bab7cf9` |
| `outputs/results/runs/graphdta_gin_kiba_cold_drug_seed4.json` | 1981 | `bd542530d4f430c04c8ccec367b108dbab0c91311ab02b29e0b00e8eba02f1a5` |
| `outputs/results/runs/graphdta_gin_kiba_cold_target_seed0.json` | 1992 | `9910c5588c6edaa0cf558ad4808c7a8864877e81905af2d5fe9eaf7ac5de5bfa` |
| `outputs/results/runs/graphdta_gin_kiba_cold_target_seed1.json` | 1988 | `b470ba9d9d630cd06f07f14c0ef387ea8d76b4e8aab2b3656e60e04e45f2356a` |
| `outputs/results/runs/graphdta_gin_kiba_cold_target_seed2.json` | 1983 | `43f1c74035508fd6f6645bec2e1d4bf908eeeba2c790151ad038957030bdc238` |
| `outputs/results/runs/graphdta_gin_kiba_cold_target_seed3.json` | 1992 | `3939eb4501239535b71fdc7fd0ad2a7e0b9b39f8efcd17bd51223e5605866eb7` |
| `outputs/results/runs/graphdta_gin_kiba_cold_target_seed4.json` | 1988 | `6a18416d87a9edca070978836800210cfabcf81d94619a88c3fc606155d9eef6` |
| `outputs/results/runs/graphdta_gin_kiba_random_seed0.json` | 1979 | `5618cc18989cceaf2415b926b845f6c160845842f095a54b33e51e67006f6cdf` |
| `outputs/results/runs/graphdta_gin_kiba_random_seed1.json` | 4360466 | `ef5d9a7dda769335a206077affe0613eefbd091613f84811d90346e3fe9ac2f1` |
| `outputs/results/runs/graphdta_gin_kiba_random_seed2.json` | 1984 | `eadd886f9b8cd13534e7d55da0b7b20475f99f029de74eba1b71ef92eba4a354` |
| `outputs/results/runs/graphdta_gin_kiba_random_seed3.json` | 4318620 | `2d9c9a6a15afa786191156fddabe6b9e4420879e7e77994039d9068ab00e6385` |
| `outputs/results/runs/graphdta_gin_kiba_random_seed4.json` | 1984 | `fb81ac87be433c07d8151ed97ddbd9d6f3071e519100654e1eb8c395450f89bc` |
| `outputs/results/runs/physdual_ecfp_esm_mlp_davis_cold_drug_fold0.json` | 6983691 | `262ec0b226352ea07dcb2cac6bbb96b625c8c25b2da8c4bbbed1ecdf4828e33b` |
| `outputs/results/runs/physdual_ecfp_esm_mlp_davis_cold_drug_fold1.json` | 6959988 | `116debb2f701f159b3a73e5e56d9049ef52ea06a6358ed949f071a5a98072fc1` |
| `outputs/results/runs/physdual_ecfp_esm_mlp_davis_cold_drug_fold2.json` | 6997702 | `8ea2aca37aa98593cca3382bd7e9d430b2fdb0fcf33bbd1cb5bed5fa498bdd10` |
| `outputs/results/runs/physdual_ecfp_esm_mlp_davis_cold_drug_fold3.json` | 6470211 | `8cadc0f60fb758f6c38d9f6c5cff72443c85badceed3bbe8124f56bd0686f3da` |
| `outputs/results/runs/physdual_ecfp_esm_mlp_davis_cold_drug_fold4.json` | 6440446 | `0c9535c436a5bf9899206f256012a740eeca7cb39c34e8b26f9e5c02689d8ac9` |
| `outputs/results/runs/physdual_ecfp_esm_mlp_davis_cold_target_fold0.json` | 6663964 | `8e0bdc01bd237a7c6aa91971598bbf9b4f85bd6627f14e05cef0e645aa477c81` |
| `outputs/results/runs/physdual_ecfp_esm_mlp_davis_cold_target_fold1.json` | 5892420 | `be90c29168a848116bfba54cf35fd794defdb2f4483c03266deb2f8922560895` |
| `outputs/results/runs/physdual_ecfp_esm_mlp_davis_cold_target_fold2.json` | 7855742 | `01497ff5037d741828652f69e84b6f4f2c477080bbfb7f8f245bc692909f0118` |
| `outputs/results/runs/physdual_ecfp_esm_mlp_davis_cold_target_fold3.json` | 6691280 | `5e8f2891e08ae55d641e4b42bd34c7a1826d68c9c04a472ad56c0f373f2000ca` |
| `outputs/results/runs/physdual_ecfp_esm_mlp_davis_cold_target_fold4.json` | 6894368 | `6c23c50158141d535eea7988ec1ec1647052e91d7ac389e7c676a3a7c7a76abf` |
| `outputs/results/runs/physdual_gin_esm_mlp_davis_cold_drug_fold0.json` | 7004154 | `0c172291a36fda81a27a16d0f68b920758110ff71078a4b55df1daaf8b93f73b` |
| `outputs/results/runs/physdual_gin_esm_mlp_davis_cold_drug_fold1.json` | 6968653 | `dd18bef8ec25cd580fa86e94cf7fb099e1c6deb20574c02aa94fa83025e410c5` |
| `outputs/results/runs/physdual_gin_esm_mlp_davis_cold_drug_fold2.json` | 6994767 | `9cc73acb3dc49e880598f64dde638f0bff6042e9d506045ec2b98166f058fcc6` |
| `outputs/results/runs/physdual_gin_esm_mlp_davis_cold_drug_fold3.json` | 6489278 | `318e08032598f944eaffc93efce08194139d40ee02af36e3c79c4c88e578a96f` |
| `outputs/results/runs/physdual_gin_esm_mlp_davis_cold_drug_fold4.json` | 6468158 | `b21d7ff9c242547e5c5e9c7052c734c472419dad487ba0c039d5e9b2c3849b9b` |
| `outputs/results/runs/physdual_gin_esm_mlp_davis_cold_target_fold0.json` | 6677763 | `42df3cd00f53fe0b9b9e7ea33b52aa5203a971710c5feedcaeb5b0ee59540985` |
| `outputs/results/runs/physdual_gin_esm_mlp_davis_cold_target_fold1.json` | 5949024 | `01b05e667dacad7491c9a03c108322f30aabda8d0cad3bb594a5044ffdafab78` |
| `outputs/results/runs/physdual_gin_esm_mlp_davis_cold_target_fold2.json` | 7813994 | `24fda41e2418c6ec48a5feb0a6958a9ed6b2e9e2ace1606f143181f5bb1784a4` |
| `outputs/results/runs/physdual_gin_esm_mlp_davis_cold_target_fold3.json` | 6697169 | `7a2952725d417c364e75004c594a345d5dba6c003d9993e4983b4b33057bead3` |
| `outputs/results/runs/physdual_gin_esm_mlp_davis_cold_target_fold4.json` | 6890755 | `c89a192c1173c70bece90b89039869dca80878fd6011ee3132004a86d57a2b9b` |
| `outputs/results/runs/physdual_gin_esm_mlp_davis_cold_target_seqid30_fold0.json` | 5441662 | `8e83990fd3c78af5a23f5e2dbde726a00a96cbc4444acb322b9ae0eca1956268` |
| `outputs/results/runs/physdual_gin_esm_mlp_davis_cold_target_seqid30_fold1.json` | 6991736 | `38c878ad068b3026d2a2748582e4aaff82e79a45e821ff082f00696d61d5f35b` |
| `outputs/results/runs/physdual_gin_esm_mlp_davis_cold_target_seqid30_fold2.json` | 6675001 | `e66b7801b440ea2bb545792771507630819fee96cd694294f06dcb6689e8244c` |
| `outputs/results/runs/physdual_gin_esm_mlp_davis_cold_target_seqid30_fold3.json` | 7612942 | `b414ad143e18b2dfb5c67bc3a41b0c00edb8d832ddd36b19e633c952057cbdb5` |
| `outputs/results/runs/physdual_gin_esm_mlp_davis_cold_target_seqid30_fold4.json` | 7202464 | `0fb451bf874f622e71d4e262d86ce4c67b6c549f79b909db1778ca7587f69f3b` |
| `outputs/results/runs/physdual_gin_esm_mlp_davis_cold_target_seqid40_fold0.json` | 6269350 | `eb13401f53baaab81f48c6fe0ff08d7039fb63dab26a904a53a9abc814a5c271` |
| `outputs/results/runs/physdual_gin_esm_mlp_davis_cold_target_seqid40_fold1.json` | 7464385 | `5d84807ff52edd054d5c4a0f46ceb613da6158343c86714f1685bf1d84a99449` |
| `outputs/results/runs/physdual_gin_esm_mlp_davis_cold_target_seqid40_fold2.json` | 7466012 | `523c256299283cd8ddc587b7376397fa5862ecb9eb7bcf9c4aa0ed9374127f5e` |
| `outputs/results/runs/physdual_gin_esm_mlp_davis_cold_target_seqid40_fold3.json` | 6742716 | `fcf9c276f095d34e4781029a1d3efb153d33c92c9cfe4fee5eb813601f8c8ccc` |
| `outputs/results/runs/physdual_gin_esm_mlp_davis_cold_target_seqid40_fold4.json` | 5986914 | `12c94c785beb0430d5314767c7a692c180c5df845430fe7359cb8c687a162b2c` |
| `outputs/results/runs/physdual_gin_esm_mlp_davis_random_seed0.json` | 3407831 | `fabfc30786c51f971b185f759e746e2f2ab614ffa0f223af83fde4f21bd55fd4` |
| `outputs/results/runs/physdual_gin_esm_mlp_davis_random_seed1.json` | 3456435 | `0205c7beef8753c031d18e7bbef5a0efca69251463aa26a54ef957a58643577a` |
| `outputs/results/runs/physdual_gin_esm_mlp_davis_random_seed2.json` | 3488426 | `bd8dca3becd5b71d79579a9be8288371ef9d2d9f4cd95d3f859642f3fbd306cb` |
| `outputs/results/runs/physdual_gin_esm_mlp_davis_random_seed3.json` | 3456650 | `1154f4dcff5d46e4cc568f420c60f6a00b40c051895ddc0f78b86aa4b0e6ea2a` |
| `outputs/results/runs/physdual_gin_esm_mlp_davis_random_seed4.json` | 3466932 | `940266f4d08f7c40966fb6475cbb2fb1e3f27129ee196030bc1d019802cfd038` |
| `outputs/results/runs/physdual_gin_esm_mlp_kiba_cold_drug_seed0.json` | 14729592 | `f68549ae5fa7ffeea30a0228210d7873befe4588edf2e6ddf286dc2fea7d65ff` |
| `outputs/results/runs/physdual_gin_esm_mlp_kiba_cold_drug_seed1.json` | 12611184 | `dbe28dda1983e2ea40c44a4637c0495564da77caa6a454bac8cb74e59ce71653` |
| `outputs/results/runs/physdual_gin_esm_mlp_kiba_cold_drug_seed2.json` | 13290253 | `a576c534a35698a9f21db53e731c02e957232ede50e55d207015b84b81494631` |
| `outputs/results/runs/physdual_gin_esm_mlp_kiba_cold_drug_seed3.json` | 12917946 | `9037a19e7b9688361abfa9006d537d2cde72de93863f92440b190af38f0c247e` |
| `outputs/results/runs/physdual_gin_esm_mlp_kiba_cold_drug_seed4.json` | 14529287 | `bf0481af778aca44a05f931511d9f1def4a79a9d42e2c4e9b46756dd7076b3d7` |
| `outputs/results/runs/physdual_gin_esm_mlp_kiba_cold_target_seed0.json` | 8730905 | `cf3e6a68dafd8b4f70b75e32369d74eb7c7e1e8d121447f9cc3dc63926a0b6ca` |
| `outputs/results/runs/physdual_gin_esm_mlp_kiba_cold_target_seed1.json` | 12703832 | `c468d7c3a396f4ef9b9d642de1d5fbdd33e99f7a5781aae5b830e3910a65710b` |
| `outputs/results/runs/physdual_gin_esm_mlp_kiba_cold_target_seed2.json` | 13515762 | `bdb9eb5921f6d5840d54b7a21cb299777d7bf5731edd24c0242085f21c8f6f7c` |
| `outputs/results/runs/physdual_gin_esm_mlp_kiba_cold_target_seed3.json` | 13694734 | `1eeb73d5843e8b13ca012b1aac0761303f6e448f53a2c075410f9dd72eda9f98` |
| `outputs/results/runs/physdual_gin_esm_mlp_kiba_cold_target_seed4.json` | 11865118 | `dfca57542eb66efcec52e3451df60384f37863346e52be34641fd79675493409` |
| `outputs/results/runs/physdual_gin_esm_mlp_kiba_cold_target_seqid30_fold0.json` | 29182570 | `3e8eff64155b18d08ce18e67b528155257365c9889e36bae114ed24485f2427d` |
| `outputs/results/runs/physdual_gin_esm_mlp_kiba_cold_target_seqid30_fold1.json` | 26678473 | `81356468bd8f8cb1292657329f7a6a8f5ddecd9598184e0a4ddecb8e36facdd4` |
| `outputs/results/runs/physdual_gin_esm_mlp_kiba_cold_target_seqid30_fold2.json` | 21838304 | `3fd957336dbe131ec2b67025a82e98cc9e9f6461d701f09e58e0b100718c4b25` |
| `outputs/results/runs/physdual_gin_esm_mlp_kiba_cold_target_seqid30_fold3.json` | 23257156 | `2761ef2f9cf90d14793241ffe68d74c640a9376d191dbd9218e3c831b3143fd5` |
| `outputs/results/runs/physdual_gin_esm_mlp_kiba_cold_target_seqid30_fold4.json` | 26286690 | `f5091326a46c2df17a30c63d7028517b4b613bdb502d9e9a421401afa8fcb25f` |
| `outputs/results/runs/physdual_gin_esm_mlp_kiba_cold_target_seqid40_fold0.json` | 24027858 | `95d281196cb28747c91dd8b3e49f164b3c54c32542816130d3d2fe43d04a69bf` |
| `outputs/results/runs/physdual_gin_esm_mlp_kiba_cold_target_seqid40_fold1.json` | 24837452 | `4717f9d658517d40e924bd42729299cc908541f24185e50b56465ff71f84dd31` |
| `outputs/results/runs/physdual_gin_esm_mlp_kiba_cold_target_seqid40_fold2.json` | 25185690 | `fd8fd2ca78f30950b0b2fc806b587ca8c290d5c8686fe257a989865d3c009077` |
| `outputs/results/runs/physdual_gin_esm_mlp_kiba_cold_target_seqid40_fold3.json` | 28356324 | `5e5de9b0c03f4db51aba0135e8e25c74e9940ec65da07d9105e77b4defc25921` |
| `outputs/results/runs/physdual_gin_esm_mlp_kiba_cold_target_seqid40_fold4.json` | 24868502 | `dc8e8bf2fc400eba0a49528dfdc01cfd9ff083c069d3e24e7e75c2795049470c` |
| `outputs/results/runs/physdual_gin_esm_mlp_kiba_random_seed0.json` | 12826760 | `2c00a5f06a70f662d6604ad3d6ffbd0b310d0bba825363b70221fad13a370fdb` |
| `outputs/results/runs/physdual_gin_esm_mlp_kiba_random_seed1.json` | 12868701 | `5cac9458f4600fe9e8612aaec314e9d747cbe3cb8d501bf40940a52709129608` |
| `outputs/results/runs/physdual_gin_esm_mlp_kiba_random_seed2.json` | 12834333 | `0e727d4b9bd963386622df783c2b826c30775b3d1435db4892fd788ebd89cef6` |
| `outputs/results/runs/physdual_gin_esm_mlp_kiba_random_seed3.json` | 12897466 | `5ae5173678dbda23563f33c9ee76a2a194e1db233109efd1851691e874123f23` |
| `outputs/results/runs/physdual_gin_esm_mlp_kiba_random_seed4.json` | 12769806 | `3bea70a62f834de535b9b3a29d5217d43ebbd20682addd05fdcded304bcf4e3d` |
| `outputs/results/runs/physdual_gin_esm_mlp_shuf0_davis_cold_target_fold0.json` | 6636904 | `604742d5aeb855b8657a4f740faac125e102e95e860947fb2952d44182c135da` |
| `outputs/results/runs/physdual_gin_esm_mlp_shuf0_davis_cold_target_fold1.json` | 5885812 | `6397b98bf090d85466d0f162767719fc74a566ff358ab480875f27f1c8305c16` |
| `outputs/results/runs/physdual_gin_esm_mlp_shuf0_davis_cold_target_fold2.json` | 7780431 | `cb5597c2f7831c52a80b85b5da8bcd4e32a93e24e35c752a59cd251c04b7b90c` |
| `outputs/results/runs/physdual_gin_esm_mlp_shuf0_davis_cold_target_fold3.json` | 6674491 | `a0d857ed187e9bcf88d37f4b11df8f463550c21b5ec36363dd1687d63a679922` |
| `outputs/results/runs/physdual_gin_esm_mlp_shuf0_davis_cold_target_fold4.json` | 6871236 | `71a2980cd9cd4bb990bdb9bef2bceb4c5189e98f3b0e5bce1e3329126fa72396` |
| `outputs/results/runs/physdual_gin_esm_mlp_shuf1_davis_cold_target_fold0.json` | 6637860 | `ad2de261af339c3f81c2475e546fb509345251f4863416fc4859e287b14ae338` |
| `outputs/results/runs/physdual_gin_esm_mlp_shuf1_davis_cold_target_fold1.json` | 5875597 | `69721b7b20a77bd7f20ec96dbaec2ce41ba6aebdcf091daf171bdabc5c2dd671` |
| `outputs/results/runs/physdual_gin_esm_mlp_shuf1_davis_cold_target_fold2.json` | 7786557 | `d6e01ac0d91a7f0c5a1619ab71d0b6a78c9a73e7e260e79c4dd6622380fbeff4` |
| `outputs/results/runs/physdual_gin_esm_mlp_shuf1_davis_cold_target_fold3.json` | 6684478 | `de144c3cb7359729c49f3805c8366fd32b423b0567cdbf7382793f28cb5abc9f` |
| `outputs/results/runs/physdual_gin_esm_mlp_shuf1_davis_cold_target_fold4.json` | 6873254 | `e2cba1532900092052a5e436d7e3b8035b8aa780daf3197d47ba621a74a9554c` |
| `outputs/results/runs/physdual_gin_esm_mlp_shuf2_davis_cold_target_fold0.json` | 6637568 | `cbf7ac7b706cdb423a32791018c2b10e55aa282195a1947c81233a7e108f159c` |
| `outputs/results/runs/physdual_gin_esm_mlp_shuf2_davis_cold_target_fold1.json` | 5878378 | `484f209b74f246d760cae957bf6efa56e40485cdf3b4378c81fca141341a50f9` |
| `outputs/results/runs/physdual_gin_esm_mlp_shuf2_davis_cold_target_fold2.json` | 7782262 | `6bf7a18049d2365812e733e440206784789074f97d6ef6f1ed9a7f42477bd0d1` |
| `outputs/results/runs/physdual_gin_esm_mlp_shuf2_davis_cold_target_fold3.json` | 6676331 | `b72aa43669be9b52a3dd599b9b8562133eb3fcf7c09321b745ae8e3c8488a448` |
| `outputs/results/runs/physdual_gin_esm_mlp_shuf2_davis_cold_target_fold4.json` | 6871176 | `eb7c97c633b35bac8e5c163f774e78d450fbe8ea0ae7e70ea1512be7333398eb` |
| `outputs/results/runs/physdual_gin_esm_mlp_tsoff1000_davis_cold_drug_fold2.json` | 6998675 | `7bc72a7da17255bec5b816747df43cea0522a499e5c264bf6123153d59808edd` |
| `outputs/results/runs/physdual_gin_esm_mlp_tsoff1000_davis_cold_target_seqid30_fold0.json` | 5444825 | `4fff5c0f5d3a93cd61ca0d5dc695975615c60310828249349c15c780d66d1d69` |
| `outputs/results/runs/physdual_gin_esm_mlp_tsoff1000_davis_cold_target_seqid30_fold1.json` | 7015405 | `63579d8bc8a70dbd620a8235eff7ced038a83a671471235ace49092f1ccebdb9` |
| `outputs/results/runs/physdual_gin_esm_mlp_tsoff1000_davis_cold_target_seqid30_fold2.json` | 6673017 | `253a375d5c2bcda3ed550ed98dae3426842758aea671984408e2a44df6f3564c` |
| `outputs/results/runs/physdual_gin_esm_mlp_tsoff1000_davis_cold_target_seqid30_fold3.json` | 7617150 | `8efd6564d365730401092db88bf18ae38b969839dd92d2599e8630810e6d027a` |
| `outputs/results/runs/physdual_gin_esm_mlp_tsoff1000_davis_cold_target_seqid30_fold4.json` | 7219694 | `2ee22d895e6a19bb6d2f12893db351084dfd74f58b46daba785d30ef8d7c593b` |
