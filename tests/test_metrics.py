"""Testes unitários para o módulo de métricas DTA, estratificações e baselines triviais."""

import numpy as np
import pytest

from physdual.eval.metrics import (
    _compute_ci_vectorized,
    _to_numpy,
    compute_all_metrics,
    compute_ci,
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
