"""Rebuild all report figures/tables from the recorded experiment evidence.

No training takes place here. The six held-out test sets and selected checkpoint
are checked before figures are written. Call ``python -m src.reporting --root .``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

CLASSES = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]
RUNS = ["mlp_128", "mlp_128_weighted", "cnn_128", "cnn_128_weighted", "cnn_256", "cnn_512"]
LABELS = {
    "mlp_128": "MLP 128 × 96 · sem pesos",
    "mlp_128_weighted": "MLP 128 × 96 · com pesos",
    "cnn_128": "CNN 128 × 96 · sem pesos",
    "cnn_128_weighted": "CNN 128 × 96 · com pesos",
    "cnn_256": "CNN 256 × 192 · sem pesos",
    "cnn_512": "CNN 512 × 384 · sem pesos",
}
SHORT_LABELS = {k: v.replace(" · sem pesos", "").replace(" · com pesos", " + pesos") for k, v in LABELS.items()}
COLORS = ["#126782", "#e8983c", "#743d89"]


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "savefig.facecolor": "white", "figure.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _save(fig, root: Path, name: str):
    target = root / "artifacts" / "figures"
    target.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(target / f"{name}.{extension}", dpi=240, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


def _read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo necessário ausente: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _image_path(root: Path, path: str) -> Path:
    p = Path(path)
    candidates = [p] if p.is_absolute() else [root / p, root / "data/raw/dataset-resized" / p]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Imagem ausente no projeto: {path}")


def load_data(root: Path):
    root = Path(root).resolve()
    audit = _read_json(root / "data/audit.json")
    path = root / "data/splits.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Divisão ausente: {path}. Execute a preparação de dados primeiro.")
    splits = pd.read_csv(path)
    required = {"path", "label", "class_name", "split", "sha256"}
    if not required.issubset(splits.columns):
        raise ValueError(f"Manifesto sem colunas obrigatórias: {sorted(required - set(splits.columns))}")
    if splits["path"].duplicated().any():
        raise ValueError("O manifesto contém caminhos repetidos.")
    if set(splits["split"]) != {"train", "val", "test"}:
        raise ValueError("O manifesto deve conter treino, validação e teste.")
    expected_labels = splits["class_name"].map({c: i for i, c in enumerate(CLASSES)})
    if expected_labels.isna().any() or not (expected_labels == splits["label"]).all():
        raise ValueError("Classes/índices não seguem a ordem obrigatória.")
    for hash_col in ("sha256", "pixel_sha256"):
        if hash_col in splits:
            if (splits.groupby(hash_col)["split"].nunique() > 1).any():
                raise ValueError(f"Vazamento: o mesmo {hash_col} ocorre em conjuntos distintos.")
    if audit.get("total_images") != len(splits):
        raise ValueError("O número de imagens diverge entre auditoria e divisão.")
    return audit, splits


def generate_eda(root: Path):
    """Create data-only figures; requires preparation but no trained checkpoint."""
    root = Path(root).resolve()
    _style()
    audit, splits = load_data(root)
    tables = root / "artifacts/tables"
    tables.mkdir(parents=True, exist_ok=True)
    counts = pd.crosstab(splits["class_name"], splits["split"]).reindex(index=CLASSES, columns=["train", "val", "test"], fill_value=0)
    counts.assign(total=counts.sum(axis=1)).to_csv(tables / "class_distribution.csv", index_label="class_name")
    fig, ax = plt.subplots(figsize=(7.1, 2.2), layout="constrained")
    x, width = np.arange(6), 0.25
    for i, (split, title, color) in enumerate(zip(["train", "val", "test"], ["Treino", "Validação", "Teste"], COLORS)):
        bars = ax.bar(x + (i - 1) * width, counts[split], width, label=f"{title} (n={counts[split].sum()})", color=color)
        ax.bar_label(bars, fontsize=7, padding=2)
    ax.set(xticks=x, xticklabels=CLASSES, ylabel="Quantidade de imagens", ylim=(0, counts.to_numpy().max() * 1.18))
    ax.legend(loc="upper right", frameon=False)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    _save(fig, root, "class_distribution")

    examples = []
    fig, axes = plt.subplots(2, 6, figsize=(7.1, 2.6), layout="constrained")
    for col, class_name in enumerate(CLASSES):
        rows = splits[(splits["class_name"] == class_name) & (splits["split"] == "train")].sort_values("path").head(2)
        if len(rows) < 2:
            raise ValueError(f"Menos de dois exemplos de treino para {class_name}.")
        for row_idx, (_, row) in enumerate(rows.iterrows()):
            ax = axes[row_idx, col]
            with Image.open(_image_path(root, row["path"])) as im:
                ax.imshow(im.convert("RGB"))
            ax.set_title(class_name if row_idx == 0 else "", fontsize=9)
            ax.set_xlabel(Path(row["path"]).stem, fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_visible(False)
            examples.append({"class_name": class_name, "path": row["path"], "split": "train"})
    _save(fig, root, "class_examples")
    pd.DataFrame(examples).to_csv(tables / "example_images.csv", index=False)

    # Deterministic training example: never chosen using model/test performance.
    example_path = examples[0]["path"]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 1.5), layout="constrained")
    with Image.open(_image_path(root, example_path)) as im:
        original = im.convert("RGB")
        for ax, size in zip(axes, [(128, 96), (256, 192), (512, 384)]):
            ax.imshow(original.resize(size, Image.Resampling.LANCZOS), interpolation="nearest")
            ax.set_title(f"{size[0]} × {size[1]} pixels")
            ax.axis("off")
    _save(fig, root, "resolution_examples")
    (tables / "resolution_example.json").write_text(json.dumps({"path": example_path, "split": "train", "resampling": "Pillow.Resampling.LANCZOS", "width_height": [[128, 96], [256, 192], [512, 384]]}, indent=2), encoding="utf-8")
    return {"audit": audit, "splits": splits, "distribution": counts}


def _labels(series):
    if pd.api.types.is_numeric_dtype(series):
        result = series.to_numpy(dtype=int)
    else:
        result = series.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    if pd.isna(result).any() or not np.isin(result, np.arange(6)).all():
        raise ValueError("Rótulos previstos/reais fora das seis classes.")
    return result.astype(int)


def load_runs(root: Path, splits: pd.DataFrame):
    root = Path(root).resolve()
    selection = _read_json(root / "artifacts/selection.json")
    if selection.get("criterion") != "val_macro_f1":
        raise ValueError("A seleção deve usar somente F1 macro da validação.")
    selected = selection["selected_run"]
    if selected not in RUNS:
        raise ValueError(f"Experimento selecionado desconhecido: {selected}")
    evidence, summary, classwise = {}, [], []
    test = splits[splits["split"] == "test"].set_index("path")
    required_history = {"epoch", "train_loss", "val_loss", "train_accuracy", "val_accuracy", "val_macro_f1", "epoch_seconds"}
    for run_id in RUNS:
        folder = root / "artifacts/runs" / run_id
        required = ["config.json", "metrics.json", "history.csv", "best.pt", "test_predictions.csv", "classification_report.csv", "confusion_matrix.csv"]
        missing = [str(folder / filename) for filename in required if not (folder / filename).is_file()]
        if missing:
            raise FileNotFoundError("Experimento incompleto; execute treinamento/avaliação antes dos resultados:\n" + "\n".join(missing))
        metric = _read_json(folder / "metrics.json")
        history = pd.read_csv(folder / "history.csv")
        if not required_history.issubset(history.columns):
            raise ValueError(f"Histórico incompleto: {run_id}")
        if len(history) == 0 or not np.isfinite(history[list(required_history)].to_numpy()).all():
            raise ValueError(f"Histórico vazio ou não finito: {run_id}")
        predictions = pd.read_csv(folder / "test_predictions.csv")
        if predictions["path"].duplicated().any() or set(predictions["path"]) != set(test.index):
            raise ValueError(f"{run_id} não foi avaliado exatamente no teste fixado.")
        truth, predicted = _labels(predictions["y_true"]), _labels(predictions["y_pred"])
        if not np.array_equal(truth, test.loc[predictions["path"], "label"].to_numpy()):
            raise ValueError(f"Rótulos reais divergentes do manifesto: {run_id}")
        probabilities = predictions[[f"p_{c}" for c in CLASSES]].to_numpy()
        if not np.isfinite(probabilities).all() or (probabilities < -1e-6).any() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
            raise ValueError(f"Probabilidades inválidas: {run_id}")
        if not np.array_equal(probabilities.argmax(axis=1), predicted):
            raise ValueError(f"Previsão difere do argmax das probabilidades: {run_id}")
        if not np.allclose(probabilities[np.arange(len(predicted)), predicted], predictions["confidence"], atol=1e-5):
            raise ValueError(f"Confiança difere da probabilidade da classe prevista: {run_id}")
        accuracy = accuracy_score(truth, predicted)
        macro_f1 = f1_score(truth, predicted, labels=range(6), average="macro", zero_division=0)
        if not np.isclose(accuracy, metric["test_accuracy"], atol=1e-7) or not np.isclose(macro_f1, metric["test_macro_f1"], atol=1e-7):
            raise ValueError(f"Métricas incompatíveis com previsões: {run_id}")
        matrix = confusion_matrix(truth, predicted, labels=range(6))
        saved_matrix = pd.read_csv(folder / "confusion_matrix.csv").set_index("true_class").loc[CLASSES, CLASSES].to_numpy()
        if not np.array_equal(matrix, saved_matrix):
            raise ValueError(f"Matriz de confusão divergente das previsões: {run_id}")
        report = classification_report(truth, predicted, labels=range(6), target_names=CLASSES, output_dict=True, zero_division=0)
        saved_report = pd.read_csv(folder / "classification_report.csv").set_index("class_name")
        for c in CLASSES:
            row = {"run_id": run_id, "class_name": c, "precision": report[c]["precision"], "recall": report[c]["recall"], "f1": report[c]["f1-score"], "support": int(report[c]["support"])}
            if not np.allclose([row[k] for k in ["precision", "recall", "f1", "support"]], saved_report.loc[c, ["precision", "recall", "f1", "support"]].to_numpy(dtype=float), atol=1e-7):
                raise ValueError(f"Métricas de classe divergentes: {run_id}/{c}")
            classwise.append(row)
        predictions["true_class"] = [CLASSES[i] for i in truth]
        predictions["predicted_class"] = [CLASSES[i] for i in predicted]
        evidence[run_id] = {"metrics": metric, "history": history, "predictions": predictions, "matrix": matrix, "config": _read_json(folder / "config.json")}
        summary.append({"run_id": run_id, **metric})
    summary = pd.DataFrame(summary)
    ranked = summary.sort_values(["val_macro_f1", "val_loss", "run_id"], ascending=[False, True, True])
    if selected != ranked.iloc[0]["run_id"]:
        raise ValueError("O modelo em selection.json difere da seleção por F1 macro de validação.")
    return evidence, summary, pd.DataFrame(classwise), selection


def _plot_curve(ax, history, metric, best_epoch, legend=False):
    keys = ("train_loss", "val_loss") if metric == "loss" else ("train_accuracy", "val_accuracy")
    for key, color, label in zip(keys, COLORS[:2], ["Treino", "Validação"]):
        ax.plot(history["epoch"], history[key], color=color, label=label, linewidth=1.5)
    ax.axvline(best_epoch, color="#777777", linestyle=":", linewidth=1, label="Checkpoint salvo")
    ax.set_xlabel("Época")
    ax.set_ylabel("Entropia cruzada" if metric == "loss" else "Acurácia")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    if metric == "accuracy":
        ax.set_ylim(0, 1.02); ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=0.2)
    if legend: ax.legend(frameon=False, fontsize=7, loc="best")


def _plot_matrix(ax, matrix, title, fontsize=7):
    denominators = matrix.sum(axis=1, keepdims=True)
    percent = np.divide(matrix * 100.0, denominators, out=np.zeros(matrix.shape, dtype=float), where=denominators != 0)
    ax.imshow(percent, vmin=0, vmax=100, cmap="Blues", aspect="auto")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{matrix[i, j]}\n{percent[i, j]:.1f}%", ha="center", va="center", fontsize=fontsize, color="white" if percent[i, j] > 55 else "#182329", linespacing=1.15)
    ax.set_xticks(np.arange(-0.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 6, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xticks(range(6), CLASSES, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(6), CLASSES, fontsize=8)
    ax.set_xlabel("Classe prevista", fontsize=8); ax.set_ylabel("Classe real", fontsize=8)
    ax.set_title(title, fontsize=10, pad=7)
    for spine in ax.spines.values(): spine.set_visible(False)


def generate_report(root: Path):
    """Validate recorded evidence and rebuild mandatory figures/CSV tables."""
    root = Path(root).resolve()
    data = generate_eda(root)
    evidence, summary, classwise, selection = load_runs(root, data["splits"])
    tables = root / "artifacts/tables"
    # The canonical training summary is retained; display exports add readable units.
    summary.to_csv(tables / "experiment_comparison.csv", index=False)
    classwise.to_csv(tables / "per_class_comparison.csv", index=False)
    cost_columns = ["run_id", "parameters", "model_size_bytes", "epochs_trained", "best_epoch", "train_seconds", "mean_epoch_seconds", "peak_rss_mb", "hardware", "batch_size"]
    costs = summary[cost_columns].copy()
    costs["model_size_mib"] = costs["model_size_bytes"] / 2**20
    costs["train_minutes"] = costs["train_seconds"] / 60
    costs.to_csv(tables / "computational_cost.csv", index=False)
    metric_columns = ["run_id", "val_accuracy", "val_macro_f1", "test_accuracy", "test_macro_f1"]
    summary[metric_columns].to_csv(tables / "performance.csv", index=False)

    for group_idx, run_group in enumerate([RUNS[:3], RUNS[3:]], start=1):
        fig, axes = plt.subplots(3, 2, figsize=(7.1, 7.5), layout="constrained")
        for row, run_id in enumerate(run_group):
            run = evidence[run_id]
            for col, metric in enumerate(["loss", "accuracy"]):
                _plot_curve(axes[row, col], run["history"], metric, run["metrics"]["best_epoch"], legend=row == 0)
                axes[row, col].set_title(f"{LABELS[run_id]}\n{'Perda' if metric == 'loss' else 'Acurácia'}", fontsize=9)
        _save(fig, root, f"learning_curves_{group_idx}")
    for run_id in RUNS:
        run = evidence[run_id]
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7), layout="constrained")
        for ax, metric in zip(axes, ["loss", "accuracy"]):
            _plot_curve(ax, run["history"], metric, run["metrics"]["best_epoch"], legend=True)
        fig.suptitle(LABELS[run_id], fontsize=11)
        _save(fig, root, f"learning_curve_{run_id}")
        fig, ax = plt.subplots(figsize=(4.6, 4.5), layout="constrained")
        _plot_matrix(ax, run["matrix"], LABELS[run_id], fontsize=9)
        _save(fig, root, f"confusion_{run_id}")
    fig, axes = plt.subplots(3, 2, figsize=(7.1, 8.0), layout="constrained")
    for ax, run_id in zip(axes.flat, RUNS):
        _plot_matrix(ax, evidence[run_id]["matrix"], LABELS[run_id], fontsize=7.2)
    _save(fig, root, "confusion_matrices")

    resolution = summary.set_index("run_id").loc[["cnn_128", "cnn_256", "cnn_512"]].copy()
    resolution.to_csv(tables / "resolution_comparison.csv", index_label="run_id")
    x, labels = np.arange(3), ["128 × 96", "256 × 192", "512 × 384"]
    fig, ax = plt.subplots(figsize=(7.1, 1.9), layout="constrained")
    for key, label, color in zip(["test_accuracy", "test_macro_f1"], ["Acurácia no teste", "F1 macro no teste"], COLORS[:2]):
        values = resolution[key].to_numpy()
        ax.plot(x, values, marker="o", color=color, linewidth=1.7, label=label)
        for j, value in enumerate(values):
            ax.annotate(f"{100*value:.1f}%", (j, value), textcoords="offset points", xytext=(0, 9 if key == "test_accuracy" else -16), ha="center", color=color, fontsize=8)
    ax.set(xticks=x, xticklabels=labels, xlabel="Resolução (largura × altura)", ylabel="Desempenho", ylim=(0, 1.02), xlim=(-0.2, 2.2))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(frameon=False, loc="best"); ax.grid(alpha=0.2)
    _save(fig, root, "resolution_performance")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 1.9), layout="constrained")
    for ax, key, title, divisor, color in zip(axes, ["mean_epoch_seconds", "train_seconds"], ["Tempo médio por época (s)", "Tempo total de treino (min)"], [1, 60], COLORS[:2]):
        values = resolution[key].to_numpy() / divisor
        bars = ax.bar(x, values, width=.55, color=color)
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=3, fontsize=8)
        ax.set(xticks=x, xticklabels=labels, xlabel="Resolução (largura × altura)", ylabel=title, ylim=(0, max(values) * 1.2))
        ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    _save(fig, root, "resolution_cost")

    selected = selection["selected_run"]
    predictions = evidence[selected]["predictions"]
    errors = predictions[predictions["true_class"] != predictions["predicted_class"]].sort_values(["confidence", "path"], ascending=[False, True])
    errors.to_csv(tables / "selected_model_errors.csv", index=False)
    shown = errors.head(6)
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.9), layout="constrained")
    for ax in axes.flat: ax.axis("off")
    for ax, (_, row) in zip(axes.flat, shown.iterrows()):
        with Image.open(_image_path(root, row["path"])) as im: ax.imshow(im.convert("RGB"))
        ax.set_title(f"Real: {row['true_class']} → prevista: {row['predicted_class']}\np(prevista) = {row['confidence']:.1%}", fontsize=8)
        ax.text(.5, -.05, Path(row["path"]).name, transform=ax.transAxes, ha="center", fontsize=7)
    if not len(shown):
        axes[0, 1].text(.5, .5, "Nenhum erro no conjunto de teste.", ha="center", va="center", transform=axes[0, 1].transAxes)
    fig.suptitle(f"{LABELS[selected]} · {len(errors)} erros em {len(predictions)} imagens de teste", fontsize=10)
    _save(fig, root, "selected_model_errors")
    metadata = {"selected_run": selected, "criterion": "val_macro_f1", "test_images": len(predictions), "test_errors": len(errors), "errors_shown": len(shown), "error_order": "confidence descending, then path ascending", "zero_division": 0, "class_order": CLASSES, "loss_note": "Nos experimentos ponderados, a perda de treino usa pesos de classe e a perda de validação usa entropia cruzada sem pesos; suas magnitudes não são diretamente comparáveis."}
    (tables / "report_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return {**data, "evidence": evidence, "summary": summary, "classwise": classwise, "selection": selection, "errors": errors, "costs": costs, "metadata": metadata}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."), help="Raiz do repositório TrashNet")
    parser.add_argument("--eda-only", action="store_true", help="Gerar apenas distribuição e exemplos; dispensa os modelos")
    args = parser.parse_args()
    try:
        result = generate_eda(args.root) if args.eda_only else generate_report(args.root)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        parser.exit(1, f"Falha ao gerar relatório: {exc}\n")
    print(f"Figuras e tabelas verificadas em {(args.root / 'artifacts').resolve()}")
    if not args.eda_only:
        print(f"Modelo escolhido exclusivamente pela validação: {result['selection']['selected_run']}")


if __name__ == "__main__":
    main()
