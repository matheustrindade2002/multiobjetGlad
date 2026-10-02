"""
plotting.py
===========
Gera as figuras comprobatórias do pipeline multiobjetivo.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from umap import UMAP
from config import PLOTS_DIR

sns.set_theme(style="whitegrid")

def plot_confidence_vs_typicality(sel_df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=sel_df, x="typicality", y="confidence", hue="class", alpha=0.3)

    # Plotar a regressão isotônica ajustada (curva esperada)
    for c in sel_df["class"].unique():
        df_c = sel_df[sel_df["class"] == c].sort_values("typicality")
        plt.plot(df_c["typicality"], df_c["expected_conf"], label=f"Isotonic (Class {c})", linewidth=2)
        plt.fill_between(
            df_c["typicality"],
            df_c["expected_conf"] - df_c["uncertainty_w"],
            df_c["expected_conf"] + df_c["uncertainty_w"],
            alpha=0.2
        )

    plt.title("Confiança vs. Tipicidade kNN (com Curva Isotônica e Incerteza)")
    plt.xlabel("Tipicidade (\u03C4)")
    plt.ylabel("Confiança q(x)")
    plt.legend()
    path = PLOTS_DIR / "confidence_vs_typicality.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path

def plot_pareto_frontier(sel_df):
    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        data=sel_df, x="delta", y="r_lower",
        color="lightgray", alpha=0.5, label="Pool (Não-selecionado)"
    )

    selected_moo = sel_df[sel_df["selected_moo"] == True]
    sns.scatterplot(
        data=selected_moo, x="delta", y="r_lower",
        color="red", edgecolor="black", label="Selecionados (MOO Pareto)", s=60
    )

    plt.title("Espaço Multiobjetivo (Não-centralidade vs. Excedente Robusto)")
    plt.xlabel("Não-centralidade (\u03B4)")
    plt.ylabel("Excedente de Confiança Robusto (r_lower)")
    plt.legend()
    path = PLOTS_DIR / "pareto_frontier.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path

def _plot_2d_embedding_selection(sel_df, emb_2d, method_name, axis_names, filename):
    plot_df = sel_df.copy()
    plot_df["embedding_1"] = emb_2d[:, 0]
    plot_df["embedding_2"] = emb_2d[:, 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    classes = sorted(plot_df["class"].unique())
    colors = sns.color_palette("tab10", n_colors=len(classes))
    palette = dict(zip(classes, colors))

    def draw_selection(ax, selected_column, title):
        for class_id in classes:
            class_points = plot_df[plot_df["class"] == class_id]
            selected_points = class_points[class_points[selected_column] == True]
            color = palette[class_id]

            ax.scatter(
                class_points["embedding_1"],
                class_points["embedding_2"],
                color=color,
                marker="o",
                s=22,
                alpha=0.28,
                edgecolors="none",
                label=f"Classe {class_id}",
            )
            ax.scatter(
                selected_points["embedding_1"],
                selected_points["embedding_2"],
                color=color,
                marker="x",
                s=75,
                linewidths=1.8,
                label=f"Selecionados — classe {class_id}",
            )

        ax.set_title(title)
        ax.set_xlabel(axis_names[0])
        ax.set_ylabel(axis_names[1])
        ax.legend()

    draw_selection(ax1, "selected_naive", "Seleção Naive (Top-Confiança)")
    draw_selection(ax2, "selected_moo", "Seleção MOO (Pareto + Facility Location)")

    fig.suptitle(f"Distribuição das classes e instâncias selecionadas no espaço {method_name}")
    fig.tight_layout()

    path = PLOTS_DIR / filename
    plt.savefig(path, dpi=150)
    plt.close()
    return path

def plot_embedding_selection(sel_df, embeddings):
    emb_2d = PCA(n_components=2).fit_transform(embeddings)
    return _plot_2d_embedding_selection(
        sel_df,
        emb_2d,
        method_name="PCA",
        axis_names=("Primeiro componente principal", "Segundo componente principal"),
        filename="embedding_pca_selected.png",
    )

def plot_umap_selection(sel_df, embeddings, seed=42):
    reducer = UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=seed,
    )
    emb_2d = reducer.fit_transform(embeddings)
    return _plot_2d_embedding_selection(
        sel_df,
        emb_2d,
        method_name="UMAP 2D",
        axis_names=("UMAP 1", "UMAP 2"),
        filename="embedding_umap_selected.png",
    )

def plot_precision_by_typicality_bucket(sel_df, true_labels):
    sel_df = sel_df.copy()
    # `sel_df` é concatenado por classe prevista e, por isso, não preserva a
    # ordem original do pool. O índice global faz a associação correta.
    true_labels = np.asarray(true_labels)
    global_idx = sel_df["global_idx"].to_numpy(dtype=int)
    sel_df["true_label"] = true_labels[global_idx]

    # Q1 (0-25%) é periférico, Q4 (75-100%) é central
    sel_df["typicality_bucket"] = pd.qcut(sel_df["typicality"], q=4, labels=["Q1 (Periférico)", "Q2", "Q3", "Q4 (Central)"])

    results = []
    for method_col, method_name in [("selected_moo", "MOO"), ("selected_naive", "Naive")]:
        subset = sel_df[sel_df[method_col] == True]
        for bucket in sel_df["typicality_bucket"].cat.categories:
            b_data = subset[subset["typicality_bucket"] == bucket]
            n = len(b_data)
            prec = (b_data["class"] == b_data["true_label"]).mean() if n > 0 else np.nan
            results.append({"method": method_name, "bucket": bucket, "precision": prec, "n": n})

    df_res = pd.DataFrame(results)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_res, x="bucket", y="precision", hue="method")
    plt.title("Precisão Offline dos Pseudo-Rótulos por Quartil de Tipicidade")
    plt.ylabel("Precisão")
    plt.xlabel("Bucket de Tipicidade")
    path = PLOTS_DIR / "precision_by_typicality_bucket.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path, df_res

def plot_performance_comparison(metrics_dict):
    names = list(metrics_dict.keys())
    accs = [m["accuracy"] for m in metrics_dict.values()]
    f1s = [m["macro_f1"] for m in metrics_dict.values()]

    df = pd.DataFrame({
        "Model": names * 2,
        "Score": accs + f1s,
        "Metric": ["Accuracy"] * len(names) + ["Macro F1"] * len(names)
    })

    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x="Model", y="Score", hue="Metric")
    plt.title("Desempenho no Hold-Out de Teste")
    plt.ylim(0, 1.0)
    path = PLOTS_DIR / "performance_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path
