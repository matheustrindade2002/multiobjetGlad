"""
selection.py
============
Implementa a modelagem multiobjetivo para escolha de pseudo-rótulos:
Tipicidade kNN -> Regressão Isotônica (PAVA) com Bootstrap -> Fronteira de Pareto.
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.isotonic import IsotonicRegression
from sklearn.cluster import KMeans

def get_typicality_and_delta(embeddings, k_knn):
    """Passo 1: Calcula tipicidade kNN e não-centralidade (delta)."""
    embeddings = np.asarray(embeddings)
    n_samples = len(embeddings)
    if n_samples == 0:
        raise ValueError("São necessários embeddings para calcular a tipicidade.")
    if k_knn < 1:
        raise ValueError("k_knn deve ser maior ou igual a 1.")
    if n_samples == 1:
        return np.ones(1), np.zeros(1)

    # Consulta o próprio conjunto, portanto solicita um vizinho extra e remove
    # a distância zero da instância para ela mesma.
    effective_k = min(k_knn, n_samples - 1)
    nn = NearestNeighbors(n_neighbors=effective_k + 1)
    nn.fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)
    mean_dist = distances[:, 1:].mean(axis=1)
    # Evita divisão por zero
    tau = 1.0 / (mean_dist + 1e-9)

    # Normaliza tau para [0, 1] localmente
    tau_norm = (tau - tau.min()) / (tau.max() - tau.min() + 1e-9)
    delta = 1.0 - tau_norm # delta = quão longe do centro
    return tau_norm, delta

def get_robust_expected_confidence(tau, conf, n_bootstraps, rng=None):
    """Passo 2 e 4: Regressão Isotônica com Bootstrap para achar q_hat e w."""
    if n_bootstraps < 1:
        raise ValueError("n_bootstraps deve ser maior ou igual a 1.")
    rng = np.random.default_rng() if rng is None else rng
    preds_boot = np.zeros((n_bootstraps, len(tau)))
    n_samples = len(tau)

    for i in range(n_bootstraps):
        idx = rng.choice(n_samples, n_samples, replace=True)
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(tau[idx], conf[idx])
        preds_boot[i, :] = ir.predict(tau)

    expected_conf = np.mean(preds_boot, axis=0) # q_hat
    w_tau = np.std(preds_boot, axis=0)          # banda de incerteza
    return expected_conf, w_tau

def get_pareto_frontier(scores_a, scores_b):
    """Passo 5: Retorna índices dos pontos não-dominados (maximiza ambos)."""
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)
    is_efficient = np.ones(len(scores_a), dtype=bool)

    for i in range(len(scores_a)):
        # i é dominado quando outro ponto é pelo menos tão bom nos dois
        # objetivos e estritamente melhor em pelo menos um deles.
        dominates_i = (
            (scores_a >= scores_a[i])
            & (scores_b >= scores_b[i])
            & ((scores_a > scores_a[i]) | (scores_b > scores_b[i]))
        )
        if np.any(dominates_i):
            is_efficient[i] = False

    return np.where(is_efficient)[0]

def select_pseudo_labels(pred_class, confidence, embeddings, num_classes, sel_cfg, seed):
    pred_class = np.array(pred_class)
    confidence = np.array(confidence)
    embeddings = np.array(embeddings)

    moo_selected_idx = []
    naive_selected_idx = []

    all_data_frames = []
    rng = np.random.default_rng(seed)

    # Processa classe por classe independentemente
    for c in range(num_classes):
        c_mask = (pred_class == c)
        c_idx = np.where(c_mask)[0]

        if len(c_idx) == 0:
            continue

        c_emb = embeddings[c_idx]
        c_conf = confidence[c_idx]

        # Passo 1
        tau, delta = get_typicality_and_delta(c_emb, sel_cfg.k_knn)

        # Passo 2 e 4
        q_hat, w_tau = get_robust_expected_confidence(
            tau, c_conf, sel_cfg.n_bootstraps, rng=rng
        )

        # Passo 3 e 4
        r = c_conf - q_hat
        r_lower = r - sel_cfg.z_margin * w_tau

        # Dataframe diagnóstico da classe
        df_c = pd.DataFrame({
            "global_idx": c_idx,
            "class": c,
            "confidence": c_conf,
            "typicality": tau,
            "delta": delta,
            "expected_conf": q_hat,
            "uncertainty_w": w_tau,
            "r_lower": r_lower
        })
        df_c["selected_moo"] = False
        df_c["selected_naive"] = False

        # Passo 5: Corta pelo piso absoluto tau_c e acha Pareto
        valid_mask = (c_conf >= sel_cfg.tau_c)
        df_valid = df_c[valid_mask].reset_index(drop=True)

        if len(df_valid) > 0:
            # Percorre fronteiras sucessivas para preencher o orçamento. Se a
            # última fronteira não couber inteira, usa K-Means para espalhar
            # as escolhas dentro dela.
            budget_per_class = sel_cfg.target_selection_size // num_classes
            remaining = df_valid.copy()
            final_c_idx = []

            while len(remaining) > 0 and len(final_c_idx) < budget_per_class:
                pareto_local_idx = get_pareto_frontier(
                    remaining["r_lower"].values, remaining["delta"].values
                )
                df_pareto = remaining.iloc[pareto_local_idx]
                available_slots = budget_per_class - len(final_c_idx)

                if len(df_pareto) <= available_slots:
                    final_c_idx.extend(df_pareto["global_idx"].astype(int).tolist())
                    remaining = remaining.drop(df_pareto.index).reset_index(drop=True)
                    continue

                pareto_global_idx = df_pareto["global_idx"].to_numpy(dtype=int)
                pareto_embeddings = embeddings[pareto_global_idx]
                kmeans = KMeans(
                    n_clusters=available_slots, random_state=seed, n_init="auto"
                ).fit(pareto_embeddings)

                # Seleciona o elemento mais próximo do centro dentro de cada
                # cluster, garantindo uma instância distinta por centroide.
                for cluster_id, center in enumerate(kmeans.cluster_centers_):
                    cluster_members = np.where(kmeans.labels_ == cluster_id)[0]
                    distances = np.linalg.norm(
                        pareto_embeddings[cluster_members] - center, axis=1
                    )
                    selected_local = cluster_members[np.argmin(distances)]
                    final_c_idx.append(int(pareto_global_idx[selected_local]))
                break

            moo_selected_idx.extend(final_c_idx)
            df_c.loc[df_c["global_idx"].isin(final_c_idx), "selected_moo"] = True

        # Baseline Naive: Top-K instâncias mais confiantes (sem olhar Pareto/Espalhamento)
        df_c_sorted = df_c.sort_values("confidence", ascending=False)
        naive_budget = sel_cfg.target_selection_size // num_classes
        naive_c_idx = df_c_sorted.head(naive_budget)["global_idx"].values
        naive_selected_idx.extend(naive_c_idx)
        df_c.loc[df_c["global_idx"].isin(naive_c_idx), "selected_naive"] = True

        all_data_frames.append(df_c)

    full_df = pd.concat(all_data_frames, ignore_index=True)
    return full_df, moo_selected_idx, naive_selected_idx
