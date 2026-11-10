"""Módulo de métricas de avaliação DTA, estratificação e baselines triviais.

Implementa Concordance Index (CI), MSE, RMSE, correlações de Pearson e Spearman,
e a métrica r_m^2 canônica com guarda contra predições anticorrelacionadas.
Oferece avaliação de subconjunto não-censurado (Davis pKd > 5.0), estratificação
por comprimento de sequência de proteína e baselines de referência triviais.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge

try:
    import lifelines.utils

    _HAS_LIFELINES = True
except ImportError:
    _HAS_LIFELINES = False


def _to_numpy(x: Any) -> np.ndarray:
    """Converte tensores PyTorch, sequências ou arrays NumPy em array 1D float64.

    Garante isolamento de gradientes e dispositivo através de .detach().cpu().numpy()
    quando o objeto de entrada pertencer ao ecossistema PyTorch.
    """
    if hasattr(x, "detach"):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    return np.asarray(arr, dtype=np.float64).reshape(-1)


def _compute_ci_vectorized(
    y_true: Any,
    y_pred: Any,
    chunk_size: int = 1000,
) -> float:
    """Calcula Concordance Index (CI) em blocos vetorizados com memória O(N * B).

    A matriz completa de pares (N x N) consumiria O(N^2) de memória. Em conjuntos
    de teste como KIBA (~11.825 amostras), uma matriz NxN em float64 excederia 1.1 GB.
    O processamento em blocos de linhas de tamanho `chunk_size` mantém o consumo
    em O(N * B), garantindo execução contida na RAM mesmo em datasets extensos.

    Regras de concordância e empates:
      - Pares com y_i == y_j não são informativos e são omitidos do denominador.
      - Se y_i > y_j e y_hat_i > y_hat_j: concordante (+1.0).
      - Se y_i > y_j e y_hat_i == y_hat_j: empate (+0.5).
      - Se não houver pares informativos (y_true constante), retorna 0.5.
    """
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    n = len(yt)
    if n <= 1:
        return 0.5

    total_pairs = 0.0
    concordant_pairs = 0.0

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        yt_chunk = yt[start:end, None]
        yp_chunk = yp[start:end, None]

        diff_t = yt_chunk - yt[None, :]
        diff_p = yp_chunk - yp[None, :]

        # Seleciona pares ordenados informativos onde y_i > y_j
        valid = diff_t > 0
        n_valid = int(np.count_nonzero(valid))
        if n_valid > 0:
            total_pairs += n_valid
            concordant_pairs += float(np.sum((diff_p > 0)[valid]))
            concordant_pairs += 0.5 * float(np.sum((diff_p == 0)[valid]))

    if total_pairs == 0.0:
        return 0.5
    return float(concordant_pairs / total_pairs)


def compute_ci(y_true: Any, y_pred: Any) -> float:
    """Calcula o Concordance Index (CI) entre rótulos reais e predições.

    Delega para lifelines.utils.concordance_index quando disponível. Caso lifelines
    não esteja instalado ou ocorra ZeroDivisionError (ex.: sem pares informativos),
    recorre ao fallback vetorizado em blocos.
    """
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    if len(yt) <= 1:
        return 0.5

    if _HAS_LIFELINES:
        try:
            return float(lifelines.utils.concordance_index(yt, yp))
        except ZeroDivisionError:
            return 0.5
        except Exception:
            pass
    return _compute_ci_vectorized(yt, yp)


def compute_mse(y_true: Any, y_pred: Any) -> float:
    """Calcula o Erro Quadrático Médio (MSE). Retorna float nativo."""
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    if len(yt) == 0:
        return 0.0
    return float(np.mean((yt - yp) ** 2))


def compute_rmse(y_true: Any, y_pred: Any) -> float:
    """Calcula a Raiz do Erro Quadrático Médio (RMSE). Retorna float nativo."""
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    if len(yt) == 0:
        return 0.0
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def compute_pearson(y_true: Any, y_pred: Any) -> float:
    """Calcula o coeficiente de correlação linear de Pearson.

    Retorna 0.0 (float nativo) se o desvio padrão de y_true ou y_pred for zero,
    ou caso o cálculo produza NaN.
    """
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    if len(yt) < 2:
        return 0.0
    if np.std(yt) <= 1e-12 or np.std(yp) <= 1e-12 or np.all(yt == yt[0]) or np.all(yp == yp[0]):
        return 0.0
    res = stats.pearsonr(yt, yp)
    stat = getattr(res, "statistic", res[0])
    val = float(stat)
    return 0.0 if np.isnan(val) else val


def compute_spearman(y_true: Any, y_pred: Any) -> float:
    """Calcula o coeficiente de correlação por postos de Spearman.

    Retorna 0.0 (float nativo) se o desvio padrão de y_true ou y_pred for zero,
    ou caso o cálculo produza NaN.
    """
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    if len(yt) < 2:
        return 0.0
    if np.std(yt) <= 1e-12 or np.std(yp) <= 1e-12 or np.all(yt == yt[0]) or np.all(yp == yp[0]):
        return 0.0
    res = stats.spearmanr(yt, yp)
    stat = getattr(res, "statistic", res[0])
    val = float(stat)
    return 0.0 if np.isnan(val) else val


def compute_rm2(y_true: Any, y_pred: Any) -> float:
    r"""Calcula a métrica r_m^2 de Roy.

    Fórmula canônica de Roy et al.:
        r_m^2 = r^2 * (1 - sqrt(|r^2 - r_0^2|)) se r > 0, sum(yp^2) > 0 e sum((yt - y_bar)^2) > 0,
        onde r é Pearson, k = sum(yt * yp) / sum(yp^2) e
        r_0^2 = 1 - sum((yt - k * yp)^2) / sum((yt - y_bar)^2).

    Nota metodológica e divergência deliberada:
        A guarda r <= 0 -> 0.0 afasta-se deliberadamente da implementação original de
        Roy et al. / DeepDTA (que eleva r ao quadrado sem verificar o sinal, podendo produzir
        r_m^2 > 0 quando as predições estão severamente anticorrelacionadas), prevenindo
        o relato de ajuste positivo espúrio quando o modelo inverte a ordenação biológica.
    """
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    if len(yt) < 2:
        return 0.0

    std_t = np.std(yt)
    std_p = np.std(yp)
    if std_t == 0.0 or std_p == 0.0:
        return 0.0

    r = compute_pearson(yt, yp)
    if r <= 0.0:
        return 0.0

    denom_k = float(np.sum(yp ** 2))
    denom_r0 = float(np.sum((yt - np.mean(yt)) ** 2))

    if denom_k <= 0.0 or denom_r0 <= 0.0:
        return 0.0

    k = float(np.sum(yt * yp) / denom_k)
    r0_sq = 1.0 - float(np.sum((yt - k * yp) ** 2) / denom_r0)

    r_sq = r ** 2
    diff = abs(r_sq - r0_sq)
    rm2_val = float(r_sq * (1.0 - np.sqrt(diff)))
    return float(max(0.0, rm2_val))


def compute_all_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Calcula a suíte completa de 6 métricas de regressão DTA.

    Retorna dicionário com ["ci", "mse", "rmse", "pearson", "spearman", "rm2"],
    assegurando tipos primitivos nativos float do Python.
    """
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)

    return {
        "ci": float(compute_ci(yt, yp)),
        "mse": float(compute_mse(yt, yp)),
        "rmse": float(compute_rmse(yt, yp)),
        "pearson": float(compute_pearson(yt, yp)),
        "spearman": float(compute_spearman(yt, yp)),
        "rm2": float(compute_rm2(yt, yp)),
    }


def compute_uncensored_metrics(
    y_true: Any,
    y_pred: Any,
    censor_value: float = 5.0,
    tol: float = 1e-4,
) -> dict[str, float | int]:
    """Calcula métricas no subconjunto não-censurado (|y_true - censor_value| > tol).

    No Davis, ~69,64% dos pares reportam afinidade constante pKd = 5.0 (limite de
    detecção experimental). Esta função avalia o poder discriminativo do modelo
    exclusivamente nos pares ativos e inclui a contagem 'num_uncensored_pairs'.
    """
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)

    mask = np.abs(yt - censor_value) > tol
    num_uncensored = int(np.count_nonzero(mask))

    if num_uncensored == 0:
        return {
            "ci": 0.5,
            "mse": 0.0,
            "rmse": 0.0,
            "pearson": 0.0,
            "spearman": 0.0,
            "rm2": 0.0,
            "num_uncensored_pairs": 0,
        }

    metrics = compute_all_metrics(yt[mask], yp[mask])
    result: dict[str, float | int] = dict(metrics)
    result["num_uncensored_pairs"] = num_uncensored
    return result


def compute_metrics_by_seq_length(
    y_true: Any,
    y_pred: Any,
    seq_lengths: Any,
    threshold: int = 1000,
) -> dict[str, dict[str, float | int]]:
    """Estratifica as métricas DTA por faixa de comprimento proteico.

    Permite auditar o efeito do truncamento em 1000 resíduos (utilizado no GraphDTA)
    separando o conjunto em 'le_{threshold}' (comprimento <= threshold) e
    'gt_{threshold}' (comprimento > threshold).
    """
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    lengths = np.asarray(seq_lengths, dtype=np.int64).reshape(-1)

    if len(lengths) != len(yt):
        raise ValueError(
            f"Dimensões incompatíveis: y_true ({len(yt)}) vs seq_lengths ({len(lengths)})"
        )

    mask_le = lengths <= threshold
    mask_gt = lengths > threshold

    def _eval_partition(mask: np.ndarray) -> dict[str, float | int]:
        count = int(np.count_nonzero(mask))
        if count == 0:
            return {
                "count": 0,
                "ci": 0.5,
                "mse": 0.0,
                "rmse": 0.0,
                "pearson": 0.0,
                "spearman": 0.0,
                "rm2": 0.0,
            }
        part_m = compute_all_metrics(yt[mask], yp[mask])
        res: dict[str, float | int] = dict(part_m)
        res["count"] = count
        return res

    return {
        f"le_{threshold}": _eval_partition(mask_le),
        f"gt_{threshold}": _eval_partition(mask_gt),
    }


def compute_trivial_baselines(
    y_train: Any,
    y_test: Any,
    X_train_aa: np.ndarray | None = None,
    X_test_aa: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    """Calcula métricas no conjunto de teste para baselines de referência triviais.

    Gera predições para:
      1. mean_predictor: prediz uniformemente a média do conjunto de treino.
      2. ridge_aa_composition (se X_*_aa fornecidos): regressão Ridge(alpha=1.0)
         sobre frequências de aminoácidos e descritores tabulares.
    """
    yt_train = _to_numpy(y_train)
    yt_test = _to_numpy(y_test)

    mean_val = float(np.mean(yt_train)) if len(yt_train) > 0 else 0.0
    y_pred_mean = np.full_like(yt_test, fill_value=mean_val, dtype=np.float64)

    results: dict[str, dict[str, float]] = {
        "mean_predictor": compute_all_metrics(yt_test, y_pred_mean)
    }

    if X_train_aa is not None and X_test_aa is not None:
        X_tr = np.asarray(X_train_aa, dtype=np.float64)
        X_te = np.asarray(X_test_aa, dtype=np.float64)
        clf = Ridge(alpha=1.0)
        clf.fit(X_tr, yt_train)
        y_pred_ridge = clf.predict(X_te)
        results["ridge_aa_composition"] = compute_all_metrics(yt_test, y_pred_ridge)

    return results
