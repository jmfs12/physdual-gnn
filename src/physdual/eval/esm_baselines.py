"""Baselines baratos em CPU sobre embeddings ESM-2 congelados (`knn_esm_target`, `ridge_esm`).

Controles obrigatórios da Fase 3 (Etapa 3.0) antes de treinar redes neurais:
1. `knn_esm_target`: kNN (k=5) por similaridade do cosseno sobre `mean_pool(ESM-2)`.
2. `ridge_esm_target_only`: Regressão Ridge sobre `mean_pool(ESM-2)` (1280d).
3. `ridge_esm`: Regressão Ridge sobre `[ECFP4 (1024d) ; mean_pool(ESM-2) (1280d)]`.

Todas as regressões ajustam padronização exclusivamente no conjunto de treino e selecionam
`alpha` exclusivamente minimizando o MSE no conjunto de validação (zero vazamento de teste).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import linalg

from physdual.data.esm_cache import compute_seq_hash, get_embedding_cache_path
from physdual.eval.metrics import compute_all_metrics

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    try:
        from rdkit.Chem import rdFingerprintGenerator

        _ECFP4_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    except Exception:
        _ECFP4_GEN = None
    _HAS_RDKIT = True
except ImportError:
    _ECFP4_GEN = None
    _HAS_RDKIT = False


DEFAULT_RIDGE_ALPHAS: tuple[float, ...] = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


def compute_morgan_ecfp4_map(
    smiles_list: Iterable[str],
    radius: int = 2,
    n_bits: int = 1024,
) -> dict[str, np.ndarray]:
    """Gera dicionário `smiles -> vetor float64 (n_bits,)` de fingerprints ECFP4."""
    if not _HAS_RDKIT:
        raise ImportError("RDKit é obrigatório para calcular ECFP4.")
    out: dict[str, np.ndarray] = {}
    for smi in set(str(s) for s in smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            out[smi] = np.zeros(n_bits, dtype=np.float64)
            continue
        if _ECFP4_GEN is not None and radius == 2 and n_bits == 1024:
            arr = _ECFP4_GEN.GetFingerprintAsNumPy(mol).astype(np.float64)
        else:
            bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            arr = np.asarray(bv, dtype=np.float64)
        out[smi] = arr
    return out


def load_mean_pooled_esm_embeddings(
    cache_dir: Path,
    sequences: Iterable[str],
    model_name: str = "esm2_t33_650M_UR50D",
    expected_dim: int = 1280,
) -> dict[str, np.ndarray]:
    """Carrega embeddings `fp16` do disco (`weights_only=True`) e calcula mean pooling em `float64`."""
    import torch

    uniq = sorted(set(str(s) for s in sequences))
    out: dict[str, np.ndarray] = {}
    for seq in uniq:
        pt_path = get_embedding_cache_path(seq, model_name=model_name, cache_dir=cache_dir)
        if not pt_path.exists():
            raise FileNotFoundError(
                f"Embedding ESM-2 ausente para hash={compute_seq_hash(seq)}: {pt_path}"
            )
        payload = torch.load(pt_path, map_location="cpu", weights_only=True)
        emb = payload["embedding"]
        if emb.ndim != 2 or int(emb.size(1)) != int(expected_dim):
            raise ValueError(
                f"Shape inválido em {pt_path}: {tuple(emb.shape)} (esperado [L, {expected_dim}])."
            )
        if not bool(torch.isfinite(emb).all()):
            raise ValueError(f"NaN/Inf detectado no embedding {pt_path}.")
        pooled = emb.to(dtype=torch.float32).mean(dim=0).numpy().astype(np.float64)
        out[seq] = pooled
    return out


def predict_knn_esm_target(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seq_to_vec: dict[str, np.ndarray],
    k: int = 5,
) -> np.ndarray:
    """Prediz afinidade no teste via kNN ponderado por cosseno sobre vetores ESM-2 de alvos."""
    y_train = train_df["affinity"].to_numpy(dtype=np.float64)
    global_mean = float(np.mean(y_train)) if len(y_train) > 0 else 0.0
    target_means = train_df.groupby("sequence")["affinity"].mean().to_dict()

    all_seqs = sorted(set(train_df["sequence"].astype(str)).union(set(test_df["sequence"].astype(str))))
    if not all_seqs:
        return np.zeros(len(test_df), dtype=np.float64)

    dim = len(next(iter(seq_to_vec.values())))
    mat = np.zeros((len(all_seqs), dim), dtype=np.float64)
    seq_to_idx: dict[str, int] = {}
    for idx, seq in enumerate(all_seqs):
        vec = np.asarray(seq_to_vec[seq], dtype=np.float64).reshape(-1)
        norm = float(np.linalg.norm(vec))
        mat[idx] = vec / norm if norm > 1e-12 else vec
        seq_to_idx[seq] = idx
    sim_mat = mat @ mat.T

    train_by_drug: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    grouped_pairs: dict[str, list[tuple[int, float]]] = {}
    for smi, seq, aff in zip(
        train_df["smiles"].astype(str),
        train_df["sequence"].astype(str),
        y_train,
    ):
        grouped_pairs.setdefault(smi, []).append((seq_to_idx[seq], float(aff)))
    for smi, lst in grouped_pairs.items():
        idxs = np.array([x[0] for x in lst], dtype=np.int64)
        affs = np.array([x[1] for x in lst], dtype=np.float64)
        train_by_drug[smi] = (idxs, affs)

    preds = np.full(len(test_df), fill_value=global_mean, dtype=np.float64)
    for row_i, (smi, seq) in enumerate(zip(test_df["smiles"].astype(str), test_df["sequence"].astype(str))):
        cand = train_by_drug.get(smi)
        if cand is None:
            preds[row_i] = float(target_means.get(seq, global_mean))
            continue
        c_indices, c_affs = cand
        t_idx = seq_to_idx[seq]
        sims = sim_mat[c_indices, t_idx]
        order = np.argsort(-sims, kind="stable")[:k]
        top_w = np.maximum(sims[order], 0.0)
        top_a = c_affs[order]
        w_sum = float(np.sum(top_w))
        if w_sum > 1e-6:
            preds[row_i] = float(np.sum(top_w * top_a) / w_sum)
        else:
            preds[row_i] = float(np.mean(top_a)) if len(top_a) > 0 else float(target_means.get(seq, global_mean))
    return preds


def fit_predict_ridge_val_alpha(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    alphas: tuple[float, ...] = DEFAULT_RIDGE_ALPHAS,
) -> dict[str, Any]:
    """Ajusta Ridge com padronização no treino e seleciona `alpha` exclusivamente em `val`."""
    X_tr = np.asarray(X_train, dtype=np.float64)
    X_va = np.asarray(X_val, dtype=np.float64)
    X_te = np.asarray(X_test, dtype=np.float64)
    y_tr = np.asarray(y_train, dtype=np.float64).reshape(-1)
    y_va = np.asarray(y_val, dtype=np.float64).reshape(-1)

    mu = np.mean(X_tr, axis=0)
    sigma = np.std(X_tr, axis=0)
    sigma = np.where(sigma > 1e-12, sigma, 1.0)

    X_tr_s = (X_tr - mu) / sigma
    X_va_s = (X_va - mu) / sigma
    X_te_s = (X_te - mu) / sigma

    y_mean = float(np.mean(y_tr))
    y_tr_c = y_tr - y_mean

    n_samples, n_features = X_tr_s.shape
    use_primal = n_samples >= n_features

    if use_primal:
        xtx = X_tr_s.T @ X_tr_s
        xty = X_tr_s.T @ y_tr_c
        eye_p = np.eye(n_features, dtype=np.float64)
    else:
        xxt = X_tr_s @ X_tr_s.T
        eye_n = np.eye(n_samples, dtype=np.float64)

    best_alpha = float(alphas[0])
    best_val_mse = float("inf")
    best_w: np.ndarray | None = None

    for alpha in alphas:
        a = float(alpha)
        if use_primal:
            w = linalg.solve(xtx + a * eye_p, xty, assume_a="pos")
        else:
            dual_u = linalg.solve(xxt + a * eye_n, y_tr_c, assume_a="pos")
            w = X_tr_s.T @ dual_u

        val_pred = y_mean + X_va_s @ w
        val_mse = float(np.mean((y_va - val_pred) ** 2))
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_alpha = a
            best_w = w

    assert best_w is not None
    y_pred_val = y_mean + X_va_s @ best_w
    y_pred_test = y_mean + X_te_s @ best_w
    return {
        "best_alpha": best_alpha,
        "val_mse": best_val_mse,
        "y_pred_val": y_pred_val,
        "y_pred_test": y_pred_test,
    }


def _build_pair_matrices(
    df: pd.DataFrame,
    seq_to_vec: dict[str, np.ndarray],
    drug_to_fp: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Constrói matriz proteica ESM-2 `(N, D_esm)` e opcional concatenada `(N, D_fp + D_esm)`."""
    seqs = df["sequence"].astype(str).tolist()
    X_prot = np.vstack([seq_to_vec[s] for s in seqs]).astype(np.float64)
    if drug_to_fp is None:
        return X_prot, None
    smiles = df["smiles"].astype(str).tolist()
    X_drug = np.vstack([drug_to_fp[s] for s in smiles]).astype(np.float64)
    return X_prot, np.hstack([X_drug, X_prot])


def evaluate_esm_cpu_baselines_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seq_to_vec: dict[str, np.ndarray],
    drug_to_fp: dict[str, np.ndarray] | None = None,
    k: int = 5,
    alphas: tuple[float, ...] = DEFAULT_RIDGE_ALPHAS,
) -> dict[str, Any]:
    """Avalia `knn_esm_target`, `ridge_esm_target_only` e `ridge_esm` numa dobra/seed."""
    y_tr = train_df["affinity"].to_numpy(dtype=np.float64)
    y_va = val_df["affinity"].to_numpy(dtype=np.float64)
    y_te = test_df["affinity"].to_numpy(dtype=np.float64)

    out: dict[str, Any] = {}

    # 1. kNN cosseno sobre ESM-2
    pred_knn_esm = predict_knn_esm_target(train_df, test_df, seq_to_vec=seq_to_vec, k=k)
    m_knn = compute_all_metrics(y_te, pred_knn_esm)
    out["knn_esm_target"] = {"metrics": m_knn, "y_pred": pred_knn_esm}

    # 2. Matrizes para Ridge
    X_tr_p, X_tr_full = _build_pair_matrices(train_df, seq_to_vec, drug_to_fp)
    X_va_p, X_va_full = _build_pair_matrices(val_df, seq_to_vec, drug_to_fp)
    X_te_p, X_te_full = _build_pair_matrices(test_df, seq_to_vec, drug_to_fp)

    res_prot = fit_predict_ridge_val_alpha(X_tr_p, y_tr, X_va_p, y_va, X_te_p, alphas=alphas)
    m_prot = compute_all_metrics(y_te, res_prot["y_pred_test"])
    out["ridge_esm_target_only"] = {
        "metrics": m_prot,
        "best_alpha": res_prot["best_alpha"],
        "val_mse": res_prot["val_mse"],
        "y_pred": res_prot["y_pred_test"],
    }

    if X_tr_full is not None and X_va_full is not None and X_te_full is not None:
        res_full = fit_predict_ridge_val_alpha(
            X_tr_full, y_tr, X_va_full, y_va, X_te_full, alphas=alphas
        )
        m_full = compute_all_metrics(y_te, res_full["y_pred_test"])
        out["ridge_esm"] = {
            "metrics": m_full,
            "best_alpha": res_full["best_alpha"],
            "val_mse": res_full["val_mse"],
            "y_pred": res_full["y_pred_test"],
        }

    return out
