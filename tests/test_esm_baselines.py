"""Testes unitários para controles em CPU sobre embeddings ESM-2 (`knn_esm_target`, `ridge_esm`)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from physdual.eval.esm_baselines import (
    compute_morgan_ecfp4_map,
    evaluate_esm_cpu_baselines_split,
    fit_predict_ridge_val_alpha,
    predict_knn_esm_target,
)


def _make_synthetic_esm_dfs():
    """Constrói problema linear separável em espaço ESM-2/ECFP4 sem vazamento entre splits."""
    rng = np.random.RandomState(7)
    smiles_list = ["CCO", "CCN", "CCC", "CCCl", "c1ccccc1", "CC(=O)O"]
    seqs = [f"SEQ_{i}_" + ("ACDEFGHIKL" * 2) for i in range(12)]

    seq_to_vec = {s: rng.randn(16) for s in seqs}
    w_true_s = rng.randn(16)
    drug_bias = {smi: 0.3 * i for i, smi in enumerate(smiles_list)}

    rows = []
    for smi in smiles_list:
        for seq in seqs:
            y = 6.0 + drug_bias[smi] + float(seq_to_vec[seq] @ w_true_s)
            rows.append({
                "drug_id": smi,
                "smiles": smi,
                "target_id": seq[:6],
                "sequence": seq,
                "affinity": y,
            })
    df = pd.DataFrame(rows)
    train_seqs = set(seqs[:8])
    val_seqs = set(seqs[8:10])
    test_seqs = set(seqs[10:])

    train_df = df[df["sequence"].isin(train_seqs)].reset_index(drop=True)
    val_df = df[df["sequence"].isin(val_seqs)].reset_index(drop=True)
    test_df = df[df["sequence"].isin(test_seqs)].reset_index(drop=True)
    return train_df, val_df, test_df, seq_to_vec


def test_fit_predict_ridge_handles_both_primal_and_dual_and_zero_std_columns() -> None:
    rng = np.random.RandomState(123)
    # Caso dual: n_samples < n_features + coluna constante -> deve bater solução primal exata e vencer média nula
    X_tr = rng.randn(12, 20)
    X_tr[:, 0] = 5.0
    w_true = rng.randn(20)
    w_true[0] = 0.0
    y_tr = 3.0 + X_tr @ w_true
    X_va = rng.randn(8, 20)
    X_va[:, 0] = 5.0
    y_va = 3.0 + X_va @ w_true
    X_te = rng.randn(10, 20)
    X_te[:, 0] = 5.0
    y_te = 3.0 + X_te @ w_true

    res = fit_predict_ridge_val_alpha(X_tr, y_tr, X_va, y_va, X_te)
    assert np.isfinite(res["y_pred_test"]).all()
    null_mse = float(np.mean((y_te - np.mean(y_tr)) ** 2))
    assert float(np.mean((y_te - res["y_pred_test"]) ** 2)) < null_mse * 0.6

    # Conferir equivalência matemática entre fórmula dual e primal para o mesmo alpha
    mu = np.mean(X_tr, axis=0)
    sigma = np.where(np.std(X_tr, axis=0) > 1e-12, np.std(X_tr, axis=0), 1.0)
    X_tr_s = (X_tr - mu) / sigma
    X_te_s = (X_te - mu) / sigma
    y_c = y_tr - np.mean(y_tr)
    alpha = float(res["best_alpha"])
    w_primal = np.linalg.solve(X_tr_s.T @ X_tr_s + alpha * np.eye(20), X_tr_s.T @ y_c)
    pred_primal = np.mean(y_tr) + X_te_s @ w_primal
    assert np.allclose(res["y_pred_test"], pred_primal, atol=1e-10)

    # Caso primal: n_samples >= n_features
    X_tr_p = rng.randn(40, 8)
    w_p = rng.randn(8)
    res_p = fit_predict_ridge_val_alpha(
        X_tr_p, 2.0 + X_tr_p @ w_p,
        rng.randn(10, 8), 2.0 + rng.randn(10, 8) @ w_p,
        rng.randn(10, 8),
    )
    assert res_p["best_alpha"] in (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


def test_ridge_selects_alpha_exclusively_on_validation_not_test() -> None:
    rng = np.random.RandomState(99)
    X_tr = rng.randn(30, 10)
    w = rng.randn(10)
    y_tr = X_tr @ w + 0.1 * rng.randn(30)
    X_va = rng.randn(20, 10)
    y_va = X_va @ w + 0.1 * rng.randn(20)
    X_te = rng.randn(20, 10)

    res_a = fit_predict_ridge_val_alpha(X_tr, y_tr, X_va, y_va, X_te)
    # Mesmo modificando totalmente a escala de X_te (não usado em y_te, apenas teste dummy), best_alpha vem só de val
    res_b = fit_predict_ridge_val_alpha(X_tr, y_tr, X_va, y_va, X_te * 100.0)
    assert res_a["best_alpha"] == res_b["best_alpha"]
    assert res_a["val_mse"] == pytest.approx(res_b["val_mse"])


def test_knn_esm_and_ridge_esm_achieve_strong_ranking_on_linear_synthetic() -> None:
    train_df, val_df, test_df, seq_to_vec = _make_synthetic_esm_dfs()
    drug_to_fp = compute_morgan_ecfp4_map(
        set(train_df["smiles"]).union(set(val_df["smiles"])).union(set(test_df["smiles"]))
    )

    knn_pred = predict_knn_esm_target(train_df, test_df, seq_to_vec=seq_to_vec, k=3)
    assert len(knn_pred) == len(test_df)
    assert np.isfinite(knn_pred).all()

    out = evaluate_esm_cpu_baselines_split(
        train_df, val_df, test_df, seq_to_vec=seq_to_vec, drug_to_fp=drug_to_fp, k=3
    )
    assert set(out.keys()) == {"knn_esm_target", "ridge_esm_target_only", "ridge_esm"}
    assert out["ridge_esm"]["metrics"]["ci"] > 0.85
    assert out["ridge_esm"]["metrics"]["mse"] < out["ridge_esm_target_only"]["metrics"]["mse"] + 0.5
