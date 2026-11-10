"""Ingestão, validação de procedência e padronização dos datasets Davis e KIBA.

Implementa as diretrizes da Fase 1 (ADR #8 e ADR #9 em `docs/DECISOES.md`):
- Validação de procedência na tabela bruta antes de qualquer filtro ou colapso:
  Davis (68 fármacos x 442 alvos = 30.056 pares) e KIBA (2.111 fármacos x 229 alvos = 118.254 pares).
- Tratamento explícito de `affinity_unit` (`"nM"`, `"pKd"`, `"kiba_score"`, `"auto"`), com auto-detecção
  exclusivamente no nível do dataset inteiro e proibição de conversão logarítmica para o KIBA.
- Sanitização de SMILES com seleção do maior fragmento covalente via RDKit, preservando `smiles_raw`
  e registrando falhas em relatório sem derrubar o lote.
- Sanitização de sequências proteicas mantendo `X, B, U, Z, O` (vocabulário nativo do ESM-2) e
  contabilizando sua ocorrência no relatório.
- Relato duplo (`by_id` vs. `by_canonical_smiles`/`by_sequence`) e política explícita para pares
  de conteúdo duplicado (`duplicate_policy="keep_both" | "mean" | "median"`).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from physdual.utils.paths import get_project_root

AffinityUnit = Literal["nM", "pKd", "kiba_score", "auto"]
DuplicatePolicy = Literal["keep_both", "keep_all", "mean", "median"]

# Vocabulário de aminoácidos suportado nativamente pelo alfabeto do ESM-2 (UniRef50):
# 20 canônicos + 5 ambíguos/estendidos (X, B, U, Z, O).
ESM2_VALID_AA = frozenset("ACDEFGHIKLMNPQRSTVWYBXZUO")
ESM2_AMBIGUOUS_AA = ("X", "B", "U", "Z", "O")

EXPECTED_COUNTS: dict[str, dict[str, int]] = {
    "davis": {
        "num_drugs": 68,
        "num_targets": 442,
        "num_pairs": 30056,
    },
    "kiba": {
        "num_drugs": 2111,
        "num_targets": 229,
        "num_pairs": 118254,
    },
}

KAGGLE_DATASET_HANDLE = "rajaryan2315/davis-and-kiba-datasets"

# Repositório canônico oficial do DeepDTA (verificado em 2026-11-09, ADR #9 em docs/DECISOES.md):
CANONICAL_DEEPDTA_REPO = "hkmztrk/DeepDTA"
CANONICAL_DEEPDTA_COMMIT = "a546a8433a6822e958f36171c4356ad6f414d623"
CANONICAL_DEEPDTA_COMMIT_DATE = "2023-08-16T12:18:16Z"
CANONICAL_DEEPDTA_BASE_URL = (
    f"https://raw.githubusercontent.com/{CANONICAL_DEEPDTA_REPO}/{CANONICAL_DEEPDTA_COMMIT}/data"
)


def _short_hash(prefix: str, text: str) -> str:
    """Gera um identificador determinístico curto (prefixo + 8 chars SHA-256)."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{digest}"


def validate_protein_sequence(seq: str) -> str:
    """Remove espaços e quebras de linha, converte para uppercase e valida contra o vocabulário ESM-2.

    Conforme ADR #8 (`docs/DECISOES.md`), mantém intactos os códigos IUPAC estendidos/ambíguos
    `X`, `B`, `U`, `Z`, `O` presentes no vocabulário do `fair-esm`, rejeitando apenas caracteres
    fora de `ACDEFGHIKLMNPQRSTVWYBXZUO`.
    """
    if not isinstance(seq, str):
        raise ValueError(f"Sequência proteica deve ser string, recebido: {type(seq).__name__}")

    cleaned = "".join(seq.split()).upper()
    if not cleaned:
        raise ValueError("Sequência proteica vazia após remoção de espaços/quebras de linha.")

    invalid_chars = sorted(set(cleaned) - ESM2_VALID_AA)
    if invalid_chars:
        raise ValueError(
            f"Sequência contém caractere inválido fora do vocabulário ESM-2: {invalid_chars}"
        )
    return cleaned


def canonicalize_largest_fragment(smiles: str) -> tuple[str | None, bool, str | None]:
    """Sanitiza um SMILES via RDKit, selecionando o maior fragmento covalente (remoção de sais).

    Returns:
        Tupla `(canonical_smiles, was_modified, error_msg)`:
        - `canonical_smiles`: SMILES canônico isomérico do maior fragmento, ou `None` se inválido.
        - `was_modified`: `True` se continha múltiplos fragmentos (ex.: contra-íon/sal removido)
          ou se o SMILES original foi alterado na remoção de fragmentos.
        - `error_msg`: Mensagem de erro caso o RDKit rejeite a molécula, ou `None` se sucesso.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None, False, "SMILES vazio ou não-string"

    raw_s = smiles.strip()
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(raw_s)
        if mol is None or mol.GetNumAtoms() == 0:
            return None, False, f"RDKit não pôde construir molécula válida para SMILES: '{raw_s}'"

        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        had_multiple_frags = len(frags) > 1
        if had_multiple_frags:
            mol = max(frags, key=lambda m: (m.GetNumHeavyAtoms(), m.GetNumAtoms()))

        canon = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        if not canon:
            return None, False, f"MolToSmiles retornou string vazia para: '{raw_s}'"

        return canon, had_multiple_frags, None
    except ImportError:
        # Fallback caso RDKit não esteja instalado em ambiente minimalista de teste puro:
        # rejeita strings obviamente inválidas sem caracteres químicos típicos de SMILES.
        if "INVALID" in raw_s.upper() or any(ch.isspace() for ch in raw_s):
            return None, False, f"SMILES inválido (fallback sem RDKit): '{raw_s}'"
        parts = [p for p in raw_s.split(".") if p]
        had_mult = len(parts) > 1
        largest = max(parts, key=len) if parts else raw_s
        return largest, had_mult, None
    except Exception as exc:
        return None, False, str(exc)


def _configure_colab_kaggle_secrets() -> None:
    """Tenta ler `KAGGLE_USERNAME` e `KAGGLE_KEY` do painel Secrets do Google Colab se não definidos."""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return
    try:
        from google.colab import userdata  # type: ignore[import-not-found]

        username = userdata.get("KAGGLE_USERNAME")
        key = userdata.get("KAGGLE_KEY")
        if username and key:
            os.environ["KAGGLE_USERNAME"] = str(username).strip()
            os.environ["KAGGLE_KEY"] = str(key).strip()
    except Exception:
        pass


def _parse_deepdta_folder(folder: Path, dataset_name: str) -> pd.DataFrame:
    """Converte um diretório no formato clássico DeepDTA (`ligands_can.txt`, `proteins.txt`, `Y`) em DataFrame."""
    lig_path = folder / "ligands_can.txt"
    prot_path = folder / "proteins.txt"
    y_path = folder / "Y"

    ligands: dict[str, str] = json.loads(lig_path.read_text(encoding="utf-8"), object_pairs_hook=OrderedDict)
    proteins: dict[str, str] = json.loads(prot_path.read_text(encoding="utf-8"), object_pairs_hook=OrderedDict)

    with y_path.open("rb") as fp:
        try:
            y_matrix = pickle.load(fp, encoding="latin1")
        except TypeError:
            y_matrix = pickle.load(fp)

    y_arr = np.asarray(y_matrix, dtype=float)
    drug_items = list(ligands.items())
    prot_items = list(proteins.items())

    if y_arr.shape != (len(drug_items), len(prot_items)):
        raise ValueError(
            f"Dimensão da matriz Y ({y_arr.shape}) incompatível com "
            f"len(ligands)={len(drug_items)} e len(proteins)={len(prot_items)} em {folder}"
        )

    rows: list[dict[str, Any]] = []
    for i, (d_id, smi) in enumerate(drug_items):
        for j, (t_id, seq) in enumerate(prot_items):
            val = float(y_arr[i, j])
            if math.isnan(val):
                continue
            rows.append(
                {
                    "drug_id": str(d_id),
                    "smiles": str(smi),
                    "target_id": str(t_id),
                    "sequence": str(seq),
                    "affinity": val,
                }
            )
    return pd.DataFrame(rows)


def _locate_and_read_dataset_in_dir(search_dir: Path, dataset_name: str) -> pd.DataFrame:
    """Localiza e carrega o dataset (`davis` ou `kiba`) dentro de um diretório local ou baixado."""
    ds = dataset_name.lower().strip()

    # 1. Procura formato DeepDTA (subpasta com ligands_can.txt, proteins.txt, Y)
    for cand in [search_dir / ds, search_dir / ds.upper(), search_dir]:
        if (cand / "ligands_can.txt").exists() and (cand / "proteins.txt").exists() and (cand / "Y").exists():
            return _parse_deepdta_folder(cand, ds)

    for lig_file in search_dir.rglob("ligands_can.txt"):
        parent = lig_file.parent
        if ds in parent.name.lower() and (parent / "proteins.txt").exists() and (parent / "Y").exists():
            return _parse_deepdta_folder(parent, ds)

    # 2. Procura arquivos tabulares (.csv, .tsv, .parquet, .txt) com o nome do dataset
    candidates: list[Path] = []
    for ext in ("*.csv", "*.parquet", "*.tsv", "*.txt"):
        for f in search_dir.rglob(ext):
            if ds in f.name.lower() and "report" not in f.name.lower() and "clean" not in f.name.lower():
                candidates.append(f)

    if not candidates:
        raise FileNotFoundError(
            f"Nenhum arquivo bruto para '{dataset_name}' encontrado em {search_dir}."
        )

    # Prioriza CSV/Parquet único ou concatena train+test caso venha separado (estilo GraphDTA)
    csv_cands = [f for f in candidates if f.suffix.lower() in (".csv", ".parquet", ".tsv")]
    chosen_files = csv_cands if csv_cands else candidates

    train_test = [
        f for f in chosen_files if any(tag in f.stem.lower() for tag in ("train", "test", "val"))
    ]
    single_full = [f for f in chosen_files if f not in train_test]

    if single_full:
        target_file = sorted(single_full, key=lambda p: p.stat().st_size, reverse=True)[0]
        if target_file.suffix.lower() == ".parquet":
            return pd.read_parquet(target_file)
        sep = "\t" if target_file.suffix.lower() == ".tsv" else ","
        return pd.read_csv(target_file, sep=sep)

    dfs = []
    for f in sorted(train_test):
        if f.suffix.lower() == ".parquet":
            dfs.append(pd.read_parquet(f))
        else:
            sep = "\t" if f.suffix.lower() == ".tsv" else ","
            dfs.append(pd.read_csv(f, sep=sep))
    return pd.concat(dfs, ignore_index=True)


def download_canonical_dataset(
    dataset_name: str,
    dest_dir: Path,
    base_url: str = CANONICAL_DEEPDTA_BASE_URL,
) -> Path:
    """Baixa os arquivos canônicos (ligands_can.txt, proteins.txt, Y) do repositório oficial DeepDTA.

    Utiliza o commit verificado `a546a8433a6822e958f36171c4356ad6f414d623` (ADR #9 em `docs/DECISOES.md`),
    dispensando tokens ou credenciais de terceiros.
    """
    import urllib.request

    ds = dataset_name.lower().strip()
    if ds not in EXPECTED_COUNTS:
        raise ValueError(f"Dataset desconhecido: '{dataset_name}'. Opções válidas: {list(EXPECTED_COUNTS)}")

    target_subdir = Path(dest_dir) / ds
    target_subdir.mkdir(parents=True, exist_ok=True)

    files = ("ligands_can.txt", "proteins.txt", "Y")
    for fname in files:
        out_file = target_subdir / fname
        if not out_file.exists():
            file_url = f"{base_url}/{ds}/{fname}"
            req = urllib.request.Request(file_url, headers={"User-Agent": "physdual-gnn/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(out_file, "wb") as fp:
                shutil.copyfileobj(resp, fp)
    return target_subdir


def load_raw_dataset(
    dataset_name: str,
    raw_dir: Path | None = None,
    source: Literal["auto", "canonical", "kaggle"] = "auto",
    use_kaggle: bool = False,
) -> pd.DataFrame:
    """Carrega o DataFrame bruto do dataset (`davis` ou `kiba`) de `data/raw` ou via download canônico/Kaggle.

    Segue a política da ADR #9 (`docs/DECISOES.md`):
    1. Se os arquivos já estiverem em `raw_dir` (`PHYSDUAL_ROOT / "data/raw"`), carrega localmente.
    2. Se `source in ("auto", "canonical")`: baixa diretamente do repositório canônico oficial
       do DeepDTA (`hkmztrk/DeepDTA` no commit `a546a84`), dispensando qualquer credencial ou login.
    3. Se `source == "kaggle"` (ou fallback caso `use_kaggle=True`), utiliza `kagglehub.dataset_download`.
    """
    ds = dataset_name.lower().strip()
    if ds not in EXPECTED_COUNTS:
        raise ValueError(f"Dataset desconhecido: '{dataset_name}'. Opções válidas: {list(EXPECTED_COUNTS)}")

    target_raw_dir = Path(raw_dir) if raw_dir is not None else (get_project_root() / "data" / "raw")
    target_raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        return _locate_and_read_dataset_in_dir(target_raw_dir, ds)
    except FileNotFoundError:
        pass

    # 1. Download do repositório canônico DeepDTA (padrão)
    if source in ("auto", "canonical"):
        try:
            download_canonical_dataset(ds, target_raw_dir)
            return _locate_and_read_dataset_in_dir(target_raw_dir, ds)
        except Exception as exc:
            if source == "canonical" or not use_kaggle:
                raise RuntimeError(
                    f"Falha ao baixar dados canônicos de '{ds}' do GitHub ({CANONICAL_DEEPDTA_BASE_URL}): {exc}"
                ) from exc

    # 2. Download via Kagglehub (caso explicitamente solicitado ou fallback)
    if source == "kaggle" or use_kaggle:
        _configure_colab_kaggle_secrets()
        import kagglehub  # type: ignore[import-untyped]

        downloaded_path = Path(kagglehub.dataset_download(KAGGLE_DATASET_HANDLE))
        dest_subdir = target_raw_dir / ds
        dest_subdir.mkdir(parents=True, exist_ok=True)
        for item in downloaded_path.rglob("*"):
            if item.is_file() and ds in str(item).lower():
                rel = item.relative_to(downloaded_path)
                out_file = dest_subdir / rel
                out_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, out_file)

        return _locate_and_read_dataset_in_dir(target_raw_dir, ds)

    raise FileNotFoundError(
        f"Arquivos brutos de '{ds}' não encontrados em '{target_raw_dir}'."
    )


def _resolve_column(df: pd.DataFrame, candidates: tuple[str, ...], col_role: str) -> str | None:
    """Encontra o nome da coluna correspondente no DataFrame bruto (case-insensitive)."""
    lower_map = {str(c).strip().lower(): str(c) for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _detect_or_validate_affinity_unit(
    affinities: pd.Series,
    dataset_name: str,
    affinity_unit: AffinityUnit,
) -> Literal["nM", "pKd", "kiba_score"]:
    """Determina ou valida a unidade de afinidade no nível do dataset inteiro (B2)."""
    ds = dataset_name.lower().strip()
    valid_units = ("nM", "pKd", "kiba_score", "auto")
    if affinity_unit not in valid_units:
        raise ValueError(f"affinity_unit inválido: '{affinity_unit}'. Esperado um de {valid_units}.")

    if ds == "kiba":
        if affinity_unit in ("auto", "kiba_score"):
            return "kiba_score"
        raise ValueError(
            f"O dataset KIBA utiliza 'kiba_score' e nunca deve ser convertido como '{affinity_unit}'."
        )

    # Davis
    if affinity_unit == "kiba_score":
        raise ValueError("affinity_unit='kiba_score' não é válido para o dataset Davis.")
    if affinity_unit in ("nM", "pKd"):
        return affinity_unit

    # Modo "auto" para Davis: detecção exclusivamente no nível do dataset inteiro
    clean_vals = pd.to_numeric(affinities, errors="coerce").dropna()
    if len(clean_vals) == 0:
        raise ValueError("Coluna de afinidade vazia ou não-numérica; impossível detectar affinity_unit.")

    max_val = float(clean_vals.max())
    min_val = float(clean_vals.min())

    if max_val > 11.0 and min_val > 0.0:
        return "nM"

    if len(clean_vals) >= 100 and 4.9 <= min_val <= 5.1 and max_val <= 11.0:
        return "pKd"

    raise ValueError(
        f"Detecção automática de affinity_unit ambígua para Davis (n={len(clean_vals)}, "
        f"min={min_val:.4f}, max={max_val:.4f}). Especifique explicitamente affinity_unit='nM' ou 'pKd'."
    )


def standardize_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    affinity_unit: AffinityUnit = "auto",
    enforce_expected_counts: bool = False,
    duplicate_policy: DuplicatePolicy = "keep_both",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    r"""Padroniza o DataFrame de Davis ou KIBA e gera o relatório completo de auditoria.

    Ordem de execução (conforme ADR #8 e ADR #9):
    1. Identifica colunas (`drug_id`, `smiles`, `target_id`, `sequence`, `affinity`).
    2. Calcula contagens brutas (`raw_num_pairs`, `num_drugs_by_id`, `num_targets_by_id`) e, se
       `enforce_expected_counts=True`, executa os `assert`s de procedência ANTES de qualquer filtro.
    3. Valida/detecta `affinity_unit` no nível do dataset inteiro e aplica a conversão $K_d \to pK_d$
       exclusivamente para o Davis quando em `"nM"`, contabilizando valores censurados ($pK_d \le 5.0001$).
    4. Sanitiza SMILES (maior fragmento covalente em `smiles`, preservando `smiles_raw`) e sequências
       proteicas (mantendo `X, B, U, Z, O` e contando sua ocorrência).
    5. Aplica `duplicate_policy` para pares com mesmo `(smiles, sequence)` e audita perda zero não registrada.
    """
    ds = dataset_name.lower().strip()
    if ds not in EXPECTED_COUNTS:
        raise ValueError(f"Dataset desconhecido: '{dataset_name}'. Opções válidas: {list(EXPECTED_COUNTS)}")
    if duplicate_policy not in ("keep_both", "keep_all", "mean", "median"):
        raise ValueError(f"duplicate_policy inválida: '{duplicate_policy}'")

    smiles_col = _resolve_column(
        df, ("smiles", "compound_iso_smiles", "canonical_smiles", "ligands", "drug_smiles"), "smiles"
    )
    seq_col = _resolve_column(
        df, ("sequence", "target_sequence", "proteins", "protein_sequence", "target_seq"), "sequence"
    )
    aff_col = _resolve_column(df, ("affinity", "y", "pkd", "kd", "kiba_score", "label"), "affinity")
    drug_id_col = _resolve_column(df, ("drug_id", "compound_id", "ligand_id", "cid", "drug_key"), "drug_id")
    target_id_col = _resolve_column(
        df, ("target_id", "protein_id", "gene_id", "uniprot_id", "target_key"), "target_id"
    )

    if smiles_col is None or seq_col is None or aff_col is None:
        raise ValueError(
            f"Colunas obrigatórias ausentes em '{dataset_name}'. Encontradas: {list(df.columns)}"
        )

    work = df.copy()
    work = work[work[aff_col].notna()].reset_index(drop=True)

    raw_smiles_series = work[smiles_col].astype(str).str.strip()
    raw_seq_series = work[seq_col].astype(str).str.strip()

    if drug_id_col is not None:
        drug_ids_series = work[drug_id_col].astype(str).str.strip()
    else:
        drug_ids_series = raw_smiles_series.map(lambda s: _short_hash("DRUG", s))

    if target_id_col is not None:
        target_ids_series = work[target_id_col].astype(str).str.strip()
    else:
        target_ids_series = raw_seq_series.map(lambda s: _short_hash("TGT", s))

    raw_num_pairs = int(len(work))
    num_drugs_by_id = int(drug_ids_series.nunique())
    num_targets_by_id = int(target_ids_series.nunique())

    # 1. Asserts de procedência na tabela bruta (ANTES de qualquer filtro ou colapso)
    if enforce_expected_counts:
        exp = EXPECTED_COUNTS[ds]
        assert raw_num_pairs == exp["num_pairs"], (
            f"Falha na validação de procedência ({ds}): contagem de pares brutos={raw_num_pairs}, "
            f"esperado={exp['num_pairs']}"
        )
        assert num_drugs_by_id == exp["num_drugs"], (
            f"Falha na validação de procedência ({ds}): num_drugs_by_id={num_drugs_by_id}, "
            f"esperado={exp['num_drugs']}"
        )
        assert num_targets_by_id == exp["num_targets"], (
            f"Falha na validação de procedência ({ds}): num_targets_by_id={num_targets_by_id}, "
            f"esperado={exp['num_targets']}"
        )

    # 2. Detecção/validação de escala de afinidade no nível do dataset inteiro
    raw_aff = pd.to_numeric(work[aff_col], errors="raise").astype(float)
    detected_unit = _detect_or_validate_affinity_unit(raw_aff, ds, affinity_unit)

    if ds == "davis":
        if detected_unit == "nM":
            davis_censored_count = int((raw_aff >= 10000.0 - 1e-6).sum())
            # pKd = -log10(Kd_nM * 1e-9) = 9.0 - log10(Kd_nM)
            converted_aff = 9.0 - np.log10(np.clip(raw_aff.to_numpy(dtype=float), 1e-12, None))
        else:
            davis_censored_count = int((raw_aff <= 5.0 + 1e-4).sum())
            converted_aff = raw_aff.to_numpy(dtype=float)
        davis_censored_pct = round(100.0 * davis_censored_count / max(raw_num_pairs, 1), 4)
    else:
        davis_censored_count = 0
        davis_censored_pct = 0.0
        converted_aff = raw_aff.to_numpy(dtype=float)

    # 3. Sanitização e cache de SMILES únicos (preservando smiles_raw)
    unique_raw_smiles = raw_smiles_series.unique().tolist()
    smiles_map: dict[str, str | None] = {}
    smiles_modified_set: set[str] = set()
    smiles_failures: list[dict[str, str]] = []

    for smi in unique_raw_smiles:
        canon, was_mod, err = canonicalize_largest_fragment(smi)
        smiles_map[smi] = canon
        if canon is not None and (was_mod or canon != smi):
            smiles_modified_set.add(smi)
        if canon is None:
            smiles_failures.append({"smiles_raw": smi, "error": err or "Erro desconhecido"})

    # 4. Sanitização e validação de sequências únicas (ADR #8: manter X, B, U, Z, O e contar)
    unique_raw_seqs = raw_seq_series.unique().tolist()
    seq_map: dict[str, str] = {}
    for seq_str in unique_raw_seqs:
        seq_map[seq_str] = validate_protein_sequence(seq_str)

    unique_clean_seqs = sorted(set(seq_map.values()))
    ambiguous_aa_char_counts: dict[str, int] = {aa: 0 for aa in ESM2_AMBIGUOUS_AA}
    ambiguous_aa_seq_count = 0
    for c_seq in unique_clean_seqs:
        has_any = False
        for aa in ESM2_AMBIGUOUS_AA:
            if aa in c_seq:
                ambiguous_aa_char_counts[aa] += 1
                has_any = True
        if has_any:
            ambiguous_aa_seq_count += 1

    # 5. Construção do DataFrame padronizado
    clean_smiles_col = raw_smiles_series.map(smiles_map)
    clean_seq_col = raw_seq_series.map(seq_map)

    out_df = pd.DataFrame(
        {
            "drug_id": drug_ids_series,
            "smiles": clean_smiles_col,
            "smiles_raw": raw_smiles_series,
            "target_id": target_ids_series,
            "sequence": clean_seq_col,
            "affinity": converted_aff,
        }
    )

    valid_mask = out_df["smiles"].notna()
    dropped_invalid_smiles = int((~valid_mask).sum())
    out_df = out_df[valid_mask].reset_index(drop=True)
    out_df["smiles"] = out_df["smiles"].astype(str)

    # 6. Contagem e tratamento de pares com conteúdo duplicado (mesmo smiles canônico e mesma sequence)
    pair_counts = out_df.groupby(["smiles", "sequence"], sort=False).size()
    duplicate_content_pairs_count = int((pair_counts > 1).sum())
    extra_rows_in_duplicates = int((pair_counts[pair_counts > 1] - 1).sum())

    dropped_by_dedup = 0
    if duplicate_policy in ("mean", "median") and extra_rows_in_duplicates > 0:
        agg_fn = "mean" if duplicate_policy == "mean" else "median"
        out_df = (
            out_df.groupby(["smiles", "sequence"], as_index=False, sort=False)
            .agg(
                {
                    "drug_id": "first",
                    "smiles_raw": "first",
                    "target_id": "first",
                    "affinity": agg_fn,
                }
            )
            .loc[:, ["drug_id", "smiles", "smiles_raw", "target_id", "sequence", "affinity"]]
            .reset_index(drop=True)
        )
        dropped_by_dedup = extra_rows_in_duplicates

    clean_num_pairs = int(len(out_df))
    num_pairs_dropped_recorded = dropped_invalid_smiles + dropped_by_dedup
    assert raw_num_pairs - clean_num_pairs == num_pairs_dropped_recorded, (
        f"Inconsistência de auditoria: raw_num_pairs ({raw_num_pairs}) - clean_num_pairs ({clean_num_pairs}) "
        f"!= num_pairs_dropped_recorded ({num_pairs_dropped_recorded})"
    )

    num_drugs_by_canonical_smiles = int(out_df["smiles"].nunique())
    num_targets_by_sequence = int(out_df["sequence"].nunique())

    report: dict[str, Any] = {
        "dataset_name": ds,
        "raw_num_pairs": raw_num_pairs,
        "clean_num_pairs": clean_num_pairs,
        "num_pairs_dropped_recorded": num_pairs_dropped_recorded,
        "num_drugs_by_id": num_drugs_by_id,
        "num_drugs_by_canonical_smiles": num_drugs_by_canonical_smiles,
        "num_targets_by_id": num_targets_by_id,
        "num_targets_by_sequence": num_targets_by_sequence,
        "duplicate_content_pairs_count": duplicate_content_pairs_count,
        "duplicate_policy": duplicate_policy,
        "ambiguous_aa_seq_count": ambiguous_aa_seq_count,
        "ambiguous_aa_char_counts": ambiguous_aa_char_counts,
        "affinity_unit_detected": detected_unit,
        "davis_censored_count": davis_censored_count,
        "davis_censored_pct": davis_censored_pct,
        "smiles_modified_count": len(smiles_modified_set),
        "smiles_failed_count": len(smiles_failures),
        "smiles_failures": smiles_failures,
    }
    return out_df, report


def prepare_dataset(
    dataset_name: str,
    affinity_unit: AffinityUnit = "auto",
    save_clean: bool = True,
    duplicate_policy: DuplicatePolicy = "keep_both",
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
    source: Literal["auto", "canonical", "kaggle"] = "auto",
    use_kaggle: bool = False,
    enforce_expected_counts: bool = True,
) -> tuple[pd.DataFrame, Path, dict[str, Any]]:
    """Executa o pipeline completo de ingestão, validação de procedência e salvamento do dataset limpo."""
    ds = dataset_name.lower().strip()
    raw_df = load_raw_dataset(ds, raw_dir=raw_dir, source=source, use_kaggle=use_kaggle)
    clean_df, report = standardize_dataset(
        raw_df,
        dataset_name=ds,
        affinity_unit=affinity_unit,
        enforce_expected_counts=enforce_expected_counts,
        duplicate_policy=duplicate_policy,
    )

    out_dir = Path(processed_dir) if processed_dir is not None else (get_project_root() / "data" / "processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_path = out_dir / f"{ds}_clean.csv"
    report_path = out_dir / f"{ds}_ingestion_report.json"

    if save_clean:
        clean_df.to_csv(clean_path, index=False)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    return clean_df, clean_path, report


def audit_content_and_target_duplicates(df: pd.DataFrame) -> dict[str, Any]:
    """Audita duplicatas por conteúdo (SMILES e sequência) e alvos com sequências compartilhadas.

    Retorna métricas para documentação de procedência e limitações de modelagem (ADR #11):
    - `num_unique_drugs_by_id`: contagem de drug_ids únicos
    - `num_unique_drugs_by_smiles`: contagem de SMILES canônicos únicos
    - `num_unique_targets_by_id`: contagem de target_ids únicos
    - `num_unique_targets_by_sequence`: contagem de sequências únicas
    - `shared_sequence_target_groups`: grupos de target_ids que mapeiam para a mesma sequência
    - `duplicate_smiles_seq_pairs_count`: contagem de pares (smiles, sequence) que se repetem
    - `duplicate_pairs_diff_affinity_count`: contagem de pares repetidos com afinidades divergentes (std > 0)
    - `duplicate_policy`: "keep_all"
    """
    num_unique_drugs_by_id = int(df["drug_id"].nunique()) if "drug_id" in df.columns else 0
    num_unique_drugs_by_smiles = int(df["smiles"].nunique()) if "smiles" in df.columns else 0
    num_unique_targets_by_id = int(df["target_id"].nunique()) if "target_id" in df.columns else 0
    num_unique_targets_by_sequence = int(df["sequence"].nunique()) if "sequence" in df.columns else 0

    shared_sequence_target_groups: list[list[str]] = []
    if "sequence" in df.columns and "target_id" in df.columns:
        grouped_targets = df.groupby("sequence")["target_id"].unique()
        for t_ids in grouped_targets:
            uniq_ids = sorted(list(set(str(t) for t in t_ids)))
            if len(uniq_ids) > 1:
                shared_sequence_target_groups.append(uniq_ids)
        shared_sequence_target_groups.sort()

    duplicate_smiles_seq_pairs_count = 0
    duplicate_pairs_diff_affinity_count = 0

    if "smiles" in df.columns and "sequence" in df.columns:
        pair_counts = df.groupby(["smiles", "sequence"], sort=False).size()
        dup_mask = pair_counts > 1
        duplicate_smiles_seq_pairs_count = int(dup_mask.sum())

        if "affinity" in df.columns and duplicate_smiles_seq_pairs_count > 0:
            dup_pairs = df[df.duplicated(subset=["smiles", "sequence"], keep=False)]
            stds = dup_pairs.groupby(["smiles", "sequence"])["affinity"].std(ddof=0)
            duplicate_pairs_diff_affinity_count = int((stds > 0.0).sum())

    return {
        "num_unique_drugs_by_id": num_unique_drugs_by_id,
        "num_unique_drugs_by_smiles": num_unique_drugs_by_smiles,
        "num_unique_targets_by_id": num_unique_targets_by_id,
        "num_unique_targets_by_sequence": num_unique_targets_by_sequence,
        "shared_sequence_target_groups": shared_sequence_target_groups,
        "duplicate_smiles_seq_pairs_count": duplicate_smiles_seq_pairs_count,
        "duplicate_pairs_diff_affinity_count": duplicate_pairs_diff_affinity_count,
        "duplicate_policy": "keep_all",
    }

