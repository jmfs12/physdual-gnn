# Reprodutibilidade do Pacote do Artigo (Fases 1 e 2)

- **Gerado em (UTC):** `2026-11-14T03:31:43Z`
- **Commit HEAD:** `06d1249`
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

Total de arquivos JSON auditados: **167**.

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
| `outputs/results/phase2_timing_benchmark.json` | 444 | `401f6b684e352779ad24e6553486106b2579c2043071c9baf07dc0d5e1c99213` |
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
| `outputs/results/runs/physdual_gin_esm_mlp_davis_cold_drug_fold0.json` | 7004154 | `0c172291a36fda81a27a16d0f68b920758110ff71078a4b55df1daaf8b93f73b` |
| `outputs/results/runs/physdual_gin_esm_mlp_davis_cold_drug_fold1.json` | 6968653 | `dd18bef8ec25cd580fa86e94cf7fb099e1c6deb20574c02aa94fa83025e410c5` |
| `outputs/results/runs/physdual_gin_esm_mlp_davis_cold_drug_fold2.json` | 6994767 | `9cc73acb3dc49e880598f64dde638f0bff6042e9d506045ec2b98166f058fcc6` |
| `outputs/results/runs/physdual_gin_esm_mlp_davis_cold_drug_fold3.json` | 6489278 | `318e08032598f944eaffc93efce08194139d40ee02af36e3c79c4c88e578a96f` |
| `outputs/results/runs/physdual_gin_esm_mlp_davis_cold_drug_fold4.json` | 6468158 | `b21d7ff9c242547e5c5e9c7052c734c472419dad487ba0c039d5e9b2c3849b9b` |
| `outputs/results/runs/physdual_gin_esm_mlp_davis_cold_target_fold0.json` | 6677763 | `42df3cd00f53fe0b9b9e7ea33b52aa5203a971710c5feedcaeb5b0ee59540985` |
| `outputs/results/runs/physdual_gin_esm_mlp_davis_cold_target_fold1.json` | 5933417 | `17e9fd098087a4183e60e99f79363b38579c1b1b3e3d57db4b0d1d1441ece989` |
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
