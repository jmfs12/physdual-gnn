"""Testes unitários para o módulo de métricas DTA, estratificações e baselines triviais."""

import numpy as np
import pytest

from physdual.eval.metrics import (
    _compute_ci_vectorized,
    _to_numpy,
    bootstrap_grouped_comparison,
    compute_all_metrics,
    compute_binary_classification_metrics,
    compute_ci,
    compute_knn_baselines,
    compute_marginal_baselines,
    compute_metrics_by_seq_length,
    compute_mse,
    compute_pearson,
    compute_rm2,
    compute_rmse,
    compute_spearman,
    compute_trivial_baselines,
    compute_uncensored_metrics,
)


def test_standard_metrics_and_native_float_types():
    """Valida cálculo de métricas padrão e garantia de retorno em float nativo."""
    y_true = np.array([5.0, 6.0, 7.0, 8.0, 9.0], dtype=float)
    y_pred = np.array([5.2, 5.9, 7.1, 7.8, 9.2], dtype=float)

    all_m = compute_all_metrics(y_true, y_pred)
    assert set(all_m.keys()) == {"ci", "mse", "rmse", "pearson", "spearman", "rm2"}
    assert all(type(v) is float for v in all_m.values())
    assert all_m["mse"] == pytest.approx(0.028, abs=1e-3)
    assert all_m["rmse"] == pytest.approx(np.sqrt(0.028), abs=1e-3)
    assert all_m["pearson"] > 0.99
    assert all_m["spearman"] == pytest.approx(1.0, abs=1e-4)
    assert all_m["ci"] == pytest.approx(1.0, abs=1e-4)
    assert all_m["rm2"] > 0.95

    # Valida invocações individuais diretas
    assert compute_mse(y_true, y_pred) == pytest.approx(all_m["mse"], abs=1e-9)
    assert compute_rmse(y_true, y_pred) == pytest.approx(all_m["rmse"], abs=1e-9)
    assert compute_pearson(y_true, y_pred) == pytest.approx(all_m["pearson"], abs=1e-9)
    assert compute_spearman(y_true, y_pred) == pytest.approx(all_m["spearman"], abs=1e-9)


def test_concordance_index_matches_lifelines():
    """Valida concordância estrita de compute_ci com lifelines em dados com empates."""
    lifelines = pytest.importorskip("lifelines")
    rng = np.random.RandomState(42)
    y_true = np.array([5.0, 5.0, 5.0, 6.0, 7.0, 8.0, 8.0, 9.0, 10.0])
    y_pred = y_true + rng.normal(0, 0.5, size=len(y_true))

    expected_ci = lifelines.utils.concordance_index(y_true, y_pred)
    our_ci = compute_ci(y_true, y_pred)
    assert our_ci == pytest.approx(expected_ci, abs=1e-6)
    assert isinstance(our_ci, float)


def test_compute_ci_vectorized_fallback_matches_lifelines():
    """Valida o algoritmo vetorizado em blocos (fallback) contra lifelines."""
    lifelines = pytest.importorskip("lifelines")
    rng = np.random.RandomState(123)
    y_true = rng.randint(5, 11, size=500).astype(float)
    y_pred = y_true + rng.normal(0, 1.0, size=500)

    expected_ci = lifelines.utils.concordance_index(y_true, y_pred)
    # Testa com chunk_size pequeno para forçar múltiplos blocos
    chunked_ci = _compute_ci_vectorized(y_true, y_pred, chunk_size=120)
    assert chunked_ci == pytest.approx(expected_ci, abs=1e-6)
    assert isinstance(chunked_ci, float)


def test_torch_tensor_inputs_with_gradients():
    """Valida suporte a tensores PyTorch com gradiente ativo e em dispositivos."""
    torch = pytest.importorskip("torch")
    y_true_t = torch.tensor([5.0, 6.0, 7.0, 8.0], requires_grad=False)
    y_pred_t = torch.tensor([5.2, 5.8, 7.2, 7.9], requires_grad=True)

    arr_true = _to_numpy(y_true_t)
    arr_pred = _to_numpy(y_pred_t)
    assert isinstance(arr_true, np.ndarray)
    assert isinstance(arr_pred, np.ndarray)
    assert arr_true.dtype == np.float64
    assert arr_pred.dtype == np.float64

    metrics_torch = compute_all_metrics(y_true_t, y_pred_t)
    metrics_np = compute_all_metrics(arr_true, arr_pred)
    for k in metrics_torch:
        assert metrics_torch[k] == pytest.approx(metrics_np[k], abs=1e-9)


def test_constant_predictions_and_negative_correlation():
    """Valida predições degeneradas (constantes) e anticorrelacionadas."""
    y_true = np.array([5.0, 6.0, 7.0, 8.0], dtype=float)
    y_pred_const = np.array([6.5, 6.5, 6.5, 6.5], dtype=float)

    metrics = compute_all_metrics(y_true, y_pred_const)
    assert metrics["pearson"] == 0.0
    assert metrics["spearman"] == 0.0
    assert metrics["rm2"] == 0.0
    assert metrics["ci"] == pytest.approx(0.5, abs=1e-4)

    # Guarda deliberada r <= 0 -> rm2 == 0.0
    y_pred_inv = np.array([8.0, 7.0, 6.0, 5.0], dtype=float)
    assert compute_rm2(y_true, y_pred_inv) == 0.0


def test_zero_informative_pairs_ci():
    """Quando y_true é constante, não há pares informativos e CI deve retornar 0.5."""
    y_true_const = np.array([5.0, 5.0, 5.0, 5.0], dtype=float)
    y_pred = np.array([5.1, 5.2, 5.3, 5.4], dtype=float)

    ci_val = compute_ci(y_true_const, y_pred)
    assert ci_val == 0.5
    assert isinstance(ci_val, float)

    ci_fallback = _compute_ci_vectorized(y_true_const, y_pred)
    assert ci_fallback == 0.5


def test_uncensored_and_seq_length_stratified_metrics():
    """Valida filtro de não-censurados e estratificação por comprimento de sequência."""
    y_true = np.array([5.0, 5.0, 6.0, 7.5, 8.5, 9.0])
    y_pred = np.array([5.1, 5.2, 6.1, 7.4, 8.6, 8.9])
    seq_lens = [400, 1200, 500, 800, 1150, 1300]

    unc = compute_uncensored_metrics(y_true, y_pred, censor_value=5.0)
    assert unc["num_uncensored_pairs"] == 4
    assert unc["ci"] == pytest.approx(1.0, abs=1e-4)
    expected_keys = ["ci", "mse", "rmse", "pearson", "spearman", "rm2", "num_uncensored_pairs"]
    assert all(k in unc for k in expected_keys)

    by_len = compute_metrics_by_seq_length(y_true, y_pred, seq_lens, threshold=1000)
    assert by_len["le_1000"]["count"] == 3
    assert by_len["gt_1000"]["count"] == 3
    assert by_len["le_1000"]["ci"] == pytest.approx(1.0, abs=1e-4)
    assert by_len["gt_1000"]["ci"] == pytest.approx(1.0, abs=1e-4)


def test_uncensored_empty_subset():
    """Valida caso extremo em que todos os rótulos são censurados."""
    y_true = np.array([5.0, 5.0, 5.0])
    y_pred = np.array([5.1, 5.2, 5.3])
    unc = compute_uncensored_metrics(y_true, y_pred, censor_value=5.0)
    assert unc["num_uncensored_pairs"] == 0
    assert unc["ci"] == 0.5
    assert unc["mse"] == 0.0


def test_trivial_baselines():
    """Valida predição trivial pela média do treino e regressão Ridge."""
    y_train = np.array([5.0, 5.0, 6.0, 8.0])
    y_test = np.array([5.0, 7.0, 9.0])
    res = compute_trivial_baselines(y_train, y_test)
    assert "mean_predictor" in res
    assert res["mean_predictor"]["mse"] == pytest.approx(np.mean((y_test - 6.0) ** 2))
    assert res["mean_predictor"]["rmse"] == pytest.approx(np.sqrt(np.mean((y_test - 6.0) ** 2)))

    # Teste com características de aminoácidos
    rng = np.random.RandomState(42)
    X_train_aa = rng.normal(0, 1, size=(len(y_train), 25))
    X_test_aa = rng.normal(0, 1, size=(len(y_test), 25))
    res_ridge = compute_trivial_baselines(
        y_train, y_test, X_train_aa=X_train_aa, X_test_aa=X_test_aa
    )
    assert "ridge_aa_composition" in res_ridge
    assert "ci" in res_ridge["ridge_aa_composition"]
    assert "mse" in res_ridge["ridge_aa_composition"]


def test_binary_classification_metrics_auroc_aupr():
    y_true = np.array([5.0, 5.0, 6.0, 7.5, 8.2, 5.0, 9.0])
    y_pred = np.array([5.1, 5.2, 6.2, 7.0, 8.0, 5.3, 8.8])
    res = compute_binary_classification_metrics(y_true, y_pred, thresholds=[5.0, 7.0])
    assert "threshold_5.0" in res
    assert "threshold_7.0" in res
    assert 0.5 <= res["threshold_5.0"]["auroc"] <= 1.0
    assert 0.0 <= res["threshold_5.0"]["aupr"] <= 1.0
    assert res["threshold_5.0"]["num_positives"] == 4


def test_marginal_baselines_drug_and_target_mean():
    import pandas as pd
    train_df = pd.DataFrame([
        {"smiles": "CCO", "sequence": "MKTVRQ", "affinity": 5.0},
        {"smiles": "CCO", "sequence": "ACDEFG", "affinity": 7.0},
        {"smiles": "CCN", "sequence": "MKTVRQ", "affinity": 9.0},
        {"smiles": "CCN", "sequence": "ACDEFG", "affinity": 11.0},
    ])
    test_df = pd.DataFrame([
        {"smiles": "CCO", "sequence": "WWWWWW", "affinity": 6.0},  # target novo -> usa drug_mean CCO = 6.0
        {"smiles": "CCN", "sequence": "WWWWWW", "affinity": 10.0}, # target novo -> usa drug_mean CCN = 10.0
    ])
    res = compute_marginal_baselines(train_df, test_df)
    assert "drug_mean_predictor" in res
    assert "target_mean_predictor" in res
    assert res["drug_mean_predictor"]["mse"] == pytest.approx(0.0)


def test_knn_baselines_tanimoto_and_kmer():
    import pandas as pd
    train_df = pd.DataFrame([
        {"smiles": "CCO", "sequence": "MKTVRQMKTVRQ", "affinity": 5.0},
        {"smiles": "CCCO", "sequence": "MKTVRQMKTVRQ", "affinity": 6.0},
        {"smiles": "CCCC", "sequence": "ACDEFGACDEFG", "affinity": 8.0},
    ])
    test_df = pd.DataFrame([
        {"smiles": "CCO", "sequence": "MKTVRQMKTVRQ", "affinity": 5.5},
        {"smiles": "CCCO", "sequence": "ACDEFGACDEFG", "affinity": 7.0},
    ])
    res = compute_knn_baselines(train_df, test_df, k=2)
    assert "knn_tanimoto_drug" in res
    assert "knn_kmer_target" in res
    assert "mse" in res["knn_tanimoto_drug"]
    assert "ci" in res["knn_kmer_target"]


def test_bootstrap_grouped_comparison():
    import pandas as pd
    oof_df = pd.DataFrame([
        {"group": "G1", "y_true": 5.0, "y_pred_model": 5.1, "y_pred_baseline": 6.0},
        {"group": "G1", "y_true": 5.2, "y_pred_model": 5.3, "y_pred_baseline": 6.0},
        {"group": "G2", "y_true": 7.0, "y_pred_model": 7.1, "y_pred_baseline": 6.0},
        {"group": "G2", "y_true": 7.5, "y_pred_model": 7.4, "y_pred_baseline": 6.0},
        {"group": "G3", "y_true": 9.0, "y_pred_model": 8.9, "y_pred_baseline": 6.0},
        {"group": "G3", "y_true": 9.2, "y_pred_model": 9.1, "y_pred_baseline": 6.0},
    ])
    res = bootstrap_grouped_comparison(
        oof_df,
        group_col="group",
        model_col="y_pred_model",
        baseline_col="y_pred_baseline",
        target_col="y_true",
        n_bootstraps=50,
        seed=42,
    )
    assert "delta_mse" in res
    assert "delta_ci" in res
    assert res["delta_mse"]["mean"] < 0  # model tem MSE menor que baseline
    assert "ci_95" in res["delta_mse"]
    assert len(res["delta_mse"]["ci_95"]) == 2


def test_bootstrap_grouped_comparison_exact_match_with_naive():
    """Garante equivalência numérica exata da versão matricial contra reamostragem ingênua."""
    import numpy as np
    import pandas as pd
    from physdual.eval.metrics import compute_ci

    rng = np.random.RandomState(7)
    rows = []
    for g in range(6):
        for _ in range(8):
            yt = float(rng.choice([5.0, 5.0, 6.5, 8.0]))  # com empates
            ym = float(yt + rng.randn() * 0.4)
            yb = float(5.5 + 0.1 * g)  # constante dentro de alguns grupos
            rows.append({"group": f"G{g}", "y_true": yt, "y_pred": ym, "y_base": yb})
    oof_df = pd.DataFrame(rows)

    fast_res = bootstrap_grouped_comparison(
        oof_df,
        group_col="group",
        model_col="y_pred",
        baseline_col="y_base",
        target_col="y_true",
        n_bootstraps=30,
        seed=123,
    )

    # Executa referência ingênua com mesma semente
    unique_c = np.array(sorted(oof_df["group"].unique()))
    cluster_indices = oof_df.groupby("group").indices
    yt_all = oof_df["y_true"].values
    ym_all = oof_df["y_pred"].values
    yb_all = oof_df["y_base"].values

    rng_ref = np.random.RandomState(123)
    dmse_ref, dci_ref = [], []
    for _ in range(30):
        sampled = rng_ref.choice(unique_c, size=len(unique_c), replace=True)
        idx = np.concatenate([cluster_indices[c] for c in sampled])
        dmse_ref.append(
            float(np.mean((yt_all[idx] - ym_all[idx]) ** 2) - np.mean((yt_all[idx] - yb_all[idx]) ** 2))
        )
        dci_ref.append(compute_ci(yt_all[idx], ym_all[idx]) - compute_ci(yt_all[idx], yb_all[idx]))

    assert fast_res["delta_mse"]["mean"] == pytest.approx(float(np.mean(dmse_ref)), abs=1e-10)
    assert fast_res["delta_mse"]["std"] == pytest.approx(float(np.std(dmse_ref, ddof=1)), abs=1e-10)
    assert fast_res["delta_ci"]["mean"] == pytest.approx(float(np.mean(dci_ref)), abs=1e-10)
    assert fast_res["delta_ci"]["std"] == pytest.approx(float(np.std(dci_ref, ddof=1)), abs=1e-10)


