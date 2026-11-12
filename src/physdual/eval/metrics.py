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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    try:
        from rdkit.Chem import rdFingerprintGenerator

        _MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    except Exception:
        _MORGAN_GEN = None

    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False
    _MORGAN_GEN = None

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
        clf = RidgeCV(alphas=np.logspace(-2, 4, 13))
        clf.fit(X_tr, yt_train)
        y_pred_ridge = clf.predict(X_te)
        results["ridge_aa_composition"] = compute_all_metrics(yt_test, y_pred_ridge)

    return results


def compute_binary_classification_metrics(
    y_true: Any,
    y_pred: Any,
    thresholds: list[float] | tuple[float, ...] = (5.0, 7.0),
) -> dict[str, dict[str, float | int]]:
    """Calcula AUROC e AUPR (Average Precision) para limiares binários de afinidade."""
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    results: dict[str, dict[str, float | int]] = {}

    for t in thresholds:
        t_float = float(t)
        if t_float <= 5.0:
            y_bin = (yt > t_float).astype(int)
        else:
            y_bin = (yt >= t_float).astype(int)

        n_pos = int(np.sum(y_bin))
        n_total = int(len(y_bin))

        if n_pos == 0 or n_pos == n_total or len(np.unique(yp)) < 2:
            results[f"threshold_{t_float}"] = {
                "auroc": float("nan"),
                "aupr": float("nan"),
                "num_positives": n_pos,
                "num_total": n_total,
            }
        else:
            try:
                auroc_val = float(roc_auc_score(y_bin, yp))
            except Exception:
                auroc_val = float("nan")
            try:
                aupr_val = float(average_precision_score(y_bin, yp))
            except Exception:
                aupr_val = float("nan")

            results[f"threshold_{t_float}"] = {
                "auroc": auroc_val,
                "aupr": aupr_val,
                "num_positives": n_pos,
                "num_total": n_total,
            }

    return results


def compute_marginal_baselines(
    train_df: Any,
    test_df: Any,
) -> dict[str, dict[str, float]]:
    """Calcula baselines marginais de promiscuidade (Média por Fármaco e Média por Alvo)."""
    y_train = np.asarray(train_df["affinity"], dtype=np.float64)
    y_test = np.asarray(test_df["affinity"], dtype=np.float64)
    global_mean = float(np.mean(y_train)) if len(y_train) > 0 else 0.0

    # Média por fármaco (SMILES)
    drug_means = train_df.groupby("smiles")["affinity"].mean().to_dict()
    pred_drug = np.array([drug_means.get(smi, global_mean) for smi in test_df["smiles"]], dtype=np.float64)

    # Média por alvo (sequência)
    target_means = train_df.groupby("sequence")["affinity"].mean().to_dict()
    pred_target = np.array([target_means.get(seq, global_mean) for seq in test_df["sequence"]], dtype=np.float64)

    return {
        "drug_mean_predictor": compute_all_metrics(y_test, pred_drug),
        "target_mean_predictor": compute_all_metrics(y_test, pred_target),
    }


def compute_knn_baselines(
    train_df: Any,
    test_df: Any,
    k: int = 5,
) -> dict[str, dict[str, float]]:
    """Calcula baselines de homologia e similaridade molecular via kNN ponderado.

    1. knn_tanimoto_drug: Para cada par (droga_teste, alvo_teste), calcula a média ponderada
       dos k fármacos de treino mais similares (Tanimoto ECFP4) testados contra o mesmo alvo.
    2. knn_kmer_target: Para cada par (droga_teste, alvo_teste), calcula a média ponderada
       dos k alvos de treino mais similares (cosseno de 3-mers de aminoácidos) testados com a mesma droga.
    """
    y_test = np.asarray(test_df["affinity"], dtype=np.float64)
    y_train = np.asarray(train_df["affinity"], dtype=np.float64)
    global_mean = float(np.mean(y_train)) if len(y_train) > 0 else 0.0

    # --- 1. kNN Drug Tanimoto ---
    pred_knn_drug = np.full(len(test_df), fill_value=global_mean, dtype=np.float64)
    drug_means = train_df.groupby("smiles")["affinity"].mean().to_dict()
    if _HAS_RDKIT:
        all_smiles = list(set(train_df["smiles"]).union(set(test_df["smiles"])))
        fps: dict[str, Any] = {}
        for smi in all_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                if _MORGAN_GEN is not None:
                    fps[smi] = _MORGAN_GEN.GetFingerprint(mol)
                else:
                    fps[smi] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)

        train_by_target: dict[str, list[tuple[str, float]]] = {}
        for _, row in train_df.iterrows():
            train_by_target.setdefault(row["sequence"], []).append((row["smiles"], float(row["affinity"])))

        for idx, (_, row) in enumerate(test_df.iterrows()):
            t_seq = row["sequence"]
            d_smi = row["smiles"]
            d_fp = fps.get(d_smi)

            candidates = train_by_target.get(t_seq, [])
            if candidates and d_fp is not None:
                sims_and_affs = []
                for c_smi, c_aff in candidates:
                    c_fp = fps.get(c_smi)
                    sim = DataStructs.TanimotoSimilarity(d_fp, c_fp) if c_fp is not None else 0.0
                    sims_and_affs.append((sim, c_aff))

                sims_and_affs.sort(key=lambda x: x[0], reverse=True)
                top_k = sims_and_affs[:k]
                weights = [s for s, _ in top_k]
                affs = [a for _, a in top_k]

                w_sum = sum(weights)
                if w_sum > 1e-6:
                    pred_knn_drug[idx] = sum(w * a for w, a in zip(weights, affs)) / w_sum
                else:
                    pred_knn_drug[idx] = float(np.mean(affs)) if affs else drug_means.get(d_smi, global_mean)
            else:
                pred_knn_drug[idx] = drug_means.get(d_smi, global_mean)
    else:
        pred_knn_drug = np.array([drug_means.get(s, global_mean) for s in test_df["smiles"]], dtype=np.float64)

    # --- 2. kNN Target 3-mer Cosine ---
    pred_knn_target = np.full(len(test_df), fill_value=global_mean, dtype=np.float64)
    target_means = train_df.groupby("sequence")["affinity"].mean().to_dict()
    all_seqs = sorted(list(set(train_df["sequence"]).union(set(test_df["sequence"]))))
    try:
        vec = CountVectorizer(analyzer="char", ngram_range=(3, 3))
        X_seqs = vec.fit_transform(all_seqs)
        seq_to_idx = {s: i for i, s in enumerate(all_seqs)}

        train_by_drug: dict[str, list[tuple[str, float]]] = {}
        for _, row in train_df.iterrows():
            train_by_drug.setdefault(row["smiles"], []).append((row["sequence"], float(row["affinity"])))

        for idx, (_, row) in enumerate(test_df.iterrows()):
            d_smi = row["smiles"]
            t_seq = row["sequence"]
            candidates = train_by_drug.get(d_smi, [])
            if candidates:
                cand_seqs = [c[0] for c in candidates]
                cand_affs = [c[1] for c in candidates]
                t_idx = seq_to_idx[t_seq]
                c_indices = [seq_to_idx[s] for s in cand_seqs]

                c_sims = cosine_similarity(X_seqs[t_idx], X_seqs[c_indices])[0]
                sorted_idx = np.argsort(c_sims)[::-1][:k]
                weights = c_sims[sorted_idx]
                affs = [cand_affs[i] for i in sorted_idx]

                w_sum = float(np.sum(weights))
                if w_sum > 1e-6:
                    pred_knn_target[idx] = float(np.sum(weights * np.array(affs)) / w_sum)
                else:
                    pred_knn_target[idx] = float(np.mean(affs)) if affs else target_means.get(t_seq, global_mean)
            else:
                pred_knn_target[idx] = target_means.get(t_seq, global_mean)
    except Exception:
        pred_knn_target = np.array([target_means.get(s, global_mean) for s in test_df["sequence"]], dtype=np.float64)

    return {
        "knn_tanimoto_drug": compute_all_metrics(y_test, pred_knn_drug),
        "knn_kmer_target": compute_all_metrics(y_test, pred_knn_target),
    }


def bootstrap_grouped_comparison(
    oof_df: Any,
    group_col: str,
    model_col: str = "y_pred_model",
    baseline_col: str = "y_pred_baseline",
    target_col: str = "y_true",
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """Calcula intervalo de confiança por bootstrap agrupado (cluster-level) para Delta MSE e Delta CI."""
    unique_clusters = np.array(sorted(oof_df[group_col].unique()))
    n_clusters = len(unique_clusters)
    if n_clusters < 2:
        return {
            "delta_mse": {"mean": 0.0, "ci_95": [0.0, 0.0]},
            "delta_ci": {"mean": 0.0, "ci_95": [0.0, 0.0]},
            "p_superiority_mse": 0.5,
            "p_superiority_ci": 0.5,
        }

    cluster_indices = oof_df.groupby(group_col).indices

    y_t_all = np.asarray(oof_df[target_col], dtype=np.float64)
    y_m_all = np.asarray(oof_df[model_col], dtype=np.float64)
    y_b_all = np.asarray(oof_df[baseline_col], dtype=np.float64)

    rng = np.random.RandomState(int(seed))
    delta_mses: list[float] = []
    delta_cis: list[float] = []

    for _ in range(n_bootstraps):
        sampled_c = rng.choice(unique_clusters, size=n_clusters, replace=True)
        sample_row_indices = []
        for c in sampled_c:
            sample_row_indices.extend(cluster_indices[c])

        sample_idx = np.array(sample_row_indices, dtype=np.int64)
        yt_s = y_t_all[sample_idx]
        ym_s = y_m_all[sample_idx]
        yb_s = y_b_all[sample_idx]

        mse_m = float(np.mean((yt_s - ym_s) ** 2))
        mse_b = float(np.mean((yt_s - yb_s) ** 2))
        delta_mses.append(mse_m - mse_b)

        ci_m = compute_ci(yt_s, ym_s)
        ci_b = compute_ci(yt_s, yb_s)
        delta_cis.append(ci_m - ci_b)

    arr_mse = np.array(delta_mses, dtype=np.float64)
    arr_ci = np.array(delta_cis, dtype=np.float64)

    low_p = (alpha / 2.0) * 100
    high_p = (1.0 - alpha / 2.0) * 100

    ci_mse = [float(np.percentile(arr_mse, low_p)), float(np.percentile(arr_mse, high_p))]
    ci_ci = [float(np.percentile(arr_ci, low_p)), float(np.percentile(arr_ci, high_p))]

    return {
        "delta_mse": {
            "mean": float(np.mean(arr_mse)),
            "std": float(np.std(arr_mse, ddof=1)),
            "ci_95": ci_mse,
            "wins": int(np.sum(arr_mse < 0)),
            "total_bootstraps": n_bootstraps,
        },
        "delta_ci": {
            "mean": float(np.mean(arr_ci)),
            "std": float(np.std(arr_ci, ddof=1)),
            "ci_95": ci_ci,
            "wins": int(np.sum(arr_ci > 0)),
            "total_bootstraps": n_bootstraps,
        },
        "p_superiority_mse": float(np.mean(arr_mse < 0)),
        "p_superiority_ci": float(np.mean(arr_ci > 0)),
    }
