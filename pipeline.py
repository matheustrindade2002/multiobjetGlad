"""
pipeline.py
===========
Orquestra o experimento inteiro, estágio por estágio, liberando cada modelo
da memória (modeling.free_model) assim que ele deixa de ser necessário --
importante para caber confortavelmente em 16GB de RAM mantendo só UM BERT
carregado por vez (nunca os três modelos -- baseline/naive/MOO -- ao mesmo
tempo).

Fluxo:
  0) monta os splits (labeled / unlabeled / test)  -- ver data_utils.py
  1) treina o modelo "estágio 0" SÓ com `labeled`                -> avalia em `test`
  2) usa o modelo do estágio 0 para pontuar o pool `unlabeled`
     (confiança + embedding, SEM olhar rótulo verdadeiro)
  3) roda a seleção multiobjetivo (MOO) e o baseline (naive)      -- ver selection.py
  4) re-treina do zero (mesmos hiperparâmetros fixos) com
     `labeled` + pseudo-rótulos (naive)                          -> avalia em `test`
  5) re-treina do zero (mesmos hiperparâmetros fixos) com
     `labeled` + pseudo-rótulos (MOO)                             -> avalia em `test`
  6) análise offline de precisão dos pseudo-rótulos por faixa de tipicidade
     (usa o rótulo verdadeiro do pool SÓ AQUI, para medir -- nunca para treinar)
  7) gera todos os gráficos e salva outputs/results.json e outputs/RESULTS.md

Regra de ouro respeitada em todo o arquivo: `test` só é usado dentro de
`modeling.evaluate(...)`, nunca dentro de `selection.py` nem para construir
o conjunto de re-treino.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import MODEL_CFG, DATA_CFG, SEL_CFG, OUTPUT_DIR, DATA_DIR
import data_utils
import modeling
import selection
import plotting


def _dataset_report(labeled, unlabeled, test):
    """Pequeno relatório dos splits, salvo em disco por transparência/reprodutibilidade."""
    def class_counts(split):
        vals, counts = np.unique(split.labels, return_counts=True)
        return {DATA_CFG.class_names[int(v)]: int(c) for v, c in zip(vals, counts)}

    report = {
        "dataset": DATA_CFG.dataset_name,
        "labeled": {"n": len(labeled.texts), "class_counts": class_counts(labeled)},
        "unlabeled_pool": {"n": len(unlabeled.texts), "class_counts_TRUE_LABEL_FOR_REFERENCE_ONLY": class_counts(unlabeled)},
        "test_holdout": {"n": len(test.texts), "class_counts": class_counts(test)},
        "note": "unlabeled_pool class_counts são o rótulo verdadeiro, mostrado aqui só "
                "para referência de quem lê o relatório -- o pipeline de seleção NUNCA "
                "usa esses rótulos.",
    }
    with open(DATA_DIR / "dataset_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report


def run_full_pipeline():
    t_global = time.time()
    modeling.set_seed(MODEL_CFG.seed)
    device = modeling.get_device()
    print(f"=== dispositivo selecionado: {device} ===")

    tokenizer = modeling.get_tokenizer()

    # ------------------------------------------------------------------
    # 0) splits
    # ------------------------------------------------------------------
    labeled, unlabeled, test = data_utils.build_splits()
    report = _dataset_report(labeled, unlabeled, test)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    labeled_ds = data_utils.make_torch_dataset(labeled, tokenizer)
    unlabeled_ds = data_utils.make_torch_dataset(unlabeled, tokenizer)  # labels aqui NÃO são usados no treino
    test_ds = data_utils.make_torch_dataset(test, tokenizer)

    results = {}

    # ------------------------------------------------------------------
    # 1) estágio 0: treina só com `labeled`
    # ------------------------------------------------------------------
    print("\n=== [Estágio 0] treinando só com o conjunto rotulado inicial ===")
    model0 = modeling.build_model(device)
    modeling.train(model0, tokenizer, labeled_ds, device, log_prefix="stage0")
    metrics0, _ = modeling.evaluate(model0, tokenizer, test_ds, device, DATA_CFG.class_names, log_prefix="stage0/test")
    results["baseline_labeled_only"] = metrics0

    # ------------------------------------------------------------------
    # 2) pontua o pool não rotulado com o modelo do estágio 0
    # ------------------------------------------------------------------
    print("\n=== Pontuando o pool não rotulado (confiança + embedding) ===")
    pool_scores = modeling.score_pool(model0, tokenizer, unlabeled_ds, device)
    modeling.free_model(model0)  # libera RAM antes de treinar o próximo modelo

    # ------------------------------------------------------------------
    # 3) seleção multiobjetivo (MOO) e baseline (naive)
    # ------------------------------------------------------------------
    print("\n=== Rodando seleção multiobjetivo (MOO) e baseline naive ===")
    sel_df, moo_idx, naive_idx = selection.select_pseudo_labels(
        pred_class=pool_scores["pred_class"],
        confidence=pool_scores["confidence"],
        embeddings=pool_scores["embeddings"],
        num_classes=MODEL_CFG.num_labels,
        sel_cfg=SEL_CFG,
        seed=MODEL_CFG.seed,
    )
    sel_df.to_csv(DATA_DIR / "selection_diagnostics.csv", index=False)
    print(f"MOO selecionou {len(moo_idx)} pseudo-rótulos | naive selecionou {len(naive_idx)}")

    # ------------------------------------------------------------------
    # 4) e 5) re-treino do zero, com cada conjunto de pseudo-rótulos
    #    (a classe do pseudo-rótulo é `pred_class` do próprio modelo --
    #     NUNCA o rótulo verdadeiro do pool)
    # ------------------------------------------------------------------
    def build_augmented_dataset(selected_idx):
        pseudo_texts = [unlabeled.texts[i] for i in selected_idx]
        pseudo_labels = pool_scores["pred_class"][selected_idx]           # <- pseudo-rótulo, não o gold
        texts = list(labeled.texts) + pseudo_texts
        labels = np.concatenate([labeled.labels, pseudo_labels])
        return data_utils.TextClassificationDataset(texts, labels, tokenizer, MODEL_CFG.max_seq_length)

    for tag, idx in [("naive_self_training", naive_idx), ("moo_self_training", moo_idx)]:
        print(f"\n=== [Re-treino: {tag}] {len(idx)} pseudo-rótulos + {len(labeled.texts)} rotulados ===")
        aug_ds = build_augmented_dataset(idx)
        # Reinicia as sementes para tornar a inicialização do classificador e
        # o embaralhamento comparáveis entre as duas estratégias.
        modeling.set_seed(MODEL_CFG.seed)
        model_r = modeling.build_model(device)   # sempre parte do checkpoint pré-treinado, do zero
        modeling.train(model_r, tokenizer, aug_ds, device, log_prefix=tag)
        metrics_r, _ = modeling.evaluate(
            model_r, tokenizer, test_ds, device, DATA_CFG.class_names, log_prefix=f"{tag}/test"
        )
        results[tag] = metrics_r
        modeling.free_model(model_r)

    # ------------------------------------------------------------------
    # 6) análise offline de precisão (usa rótulo verdadeiro do pool SÓ AQUI)
    # ------------------------------------------------------------------
    print("\n=== Análise offline: precisão dos pseudo-rótulos por faixa de tipicidade ===")
    overall_precision = {}
    for tag, idx in [("naive", naive_idx), ("moo", moo_idx)]:
        if len(idx) == 0:
            overall_precision[tag] = None
            continue
        correct = pool_scores["pred_class"][idx] == unlabeled.labels[idx]
        overall_precision[tag] = float(correct.float().mean())
    print("Precisão geral dos pseudo-rótulos aceitos:", overall_precision)

    # ------------------------------------------------------------------
    # 7) gráficos
    # ------------------------------------------------------------------
    print("\n=== Gerando gráficos ===")
    p1 = plotting.plot_confidence_vs_typicality(sel_df)
    p2 = plotting.plot_pareto_frontier(sel_df)
    p3 = plotting.plot_embedding_selection(sel_df, pool_scores["embeddings"])
    p4 = plotting.plot_umap_selection(sel_df, pool_scores["embeddings"], MODEL_CFG.seed)
    p5, precision_bucket_df = plotting.plot_precision_by_typicality_bucket(sel_df, unlabeled.labels)
    p6 = plotting.plot_performance_comparison(
        {
            "baseline\n(só rotulado)": results["baseline_labeled_only"],
            "self-training\n(naive)": results["naive_self_training"],
            "self-training\n(MOO)": results["moo_self_training"],
        }
    )
    precision_bucket_df.to_csv(DATA_DIR / "precision_by_bucket.csv", index=False)
    print("Gráficos salvos em:", [str(p) for p in [p1, p2, p3, p4, p5, p6]])

    # ------------------------------------------------------------------
    # salva resultados finais
    # ------------------------------------------------------------------
    final = {
        "device": str(device),
        "config": {
            "model": MODEL_CFG.__dict__,
            "data": {**DATA_CFG.__dict__, "class_names": list(DATA_CFG.class_names)},
            "selection": SEL_CFG.__dict__,
        },
        "dataset_report": report,
        "n_selected": {"moo": int(len(moo_idx)), "naive": int(len(naive_idx))},
        "pseudo_label_precision_overall": overall_precision,
        "test_metrics": {
            k: {
                "accuracy": v["accuracy"],
                "macro_f1": v["macro_f1"],
                "per_class_f1": v["per_class_f1"],
            }
            for k, v in results.items()
        },
        "wall_clock_seconds": time.time() - t_global,
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    _write_results_markdown(final, precision_bucket_df)
    print(f"\n=== Pipeline concluído em {final['wall_clock_seconds']:.1f}s ===")
    return final


def _write_results_markdown(final, precision_bucket_df):
    lines = []
    lines.append("# Resultados da execução\n")
    lines.append(f"- Dispositivo: `{final['device']}`")
    lines.append(f"- Tempo total: {final['wall_clock_seconds']:.1f}s")
    lines.append(f"- Pseudo-rótulos selecionados: MOO={final['n_selected']['moo']}, "
                 f"naive={final['n_selected']['naive']}")
    lines.append(f"- Precisão geral dos pseudo-rótulos (análise offline, vs. rótulo verdadeiro): "
                 f"MOO={final['pseudo_label_precision_overall']['moo']:.3f}, "
                 f"naive={final['pseudo_label_precision_overall']['naive']:.3f}\n")

    lines.append("## Desempenho no hold-out de teste (nunca tocado por pseudo-rotulagem)\n")
    lines.append("| Modelo | Acurácia | Macro-F1 |")
    lines.append("|---|---|---|")
    name_map = {
        "baseline_labeled_only": "Baseline (só rotulado)",
        "naive_self_training": "Self-training (naive top-confidence)",
        "moo_self_training": "Self-training (MOO: excedente de confiança x não-centralidade)",
    }
    for key, name in name_map.items():
        m = final["test_metrics"][key]
        lines.append(f"| {name} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} |")

    lines.append("\n## Precisão dos pseudo-rótulos por quartil de tipicidade (Q1=periférico ... Q4=central)\n")
    lines.append("| Método | Bucket | n | Precisão |")
    lines.append("|---|---|---|---|")
    for _, row in precision_bucket_df.sort_values(["method", "bucket"]).iterrows():
        prec = f"{row['precision']:.3f}" if pd.notna(row["precision"]) else "n/a"
        lines.append(f"| {row['method']} | {row['bucket']} | {int(row['n'])} | {prec} |")

    lines.append("\n## Gráficos gerados\n")
    for fname, desc in [
        ("confidence_vs_typicality.png", "Confiança vs. tipicidade, por classe, com curva isotônica esperada"),
        ("pareto_frontier.png", "Fronteira de Pareto (excedente robusto de confiança x não-centralidade)"),
        ("embedding_pca_selected.png", "Projeção PCA do pool: selecionado (MOO/naive) vs. não selecionado"),
        ("embedding_umap_selected.png", "Projeção UMAP 2D do pool: selecionado (MOO/naive) vs. não selecionado"),
        ("precision_by_typicality_bucket.png", "Precisão dos pseudo-rótulos por quartil de tipicidade"),
        ("performance_comparison.png", "Acurácia/Macro-F1: baseline vs. naive vs. MOO"),
    ]:
        lines.append(f"- `plots/{fname}` -- {desc}")

    with open(OUTPUT_DIR / "RESULTS.md", "w") as f:
        f.write("\n".join(lines))
