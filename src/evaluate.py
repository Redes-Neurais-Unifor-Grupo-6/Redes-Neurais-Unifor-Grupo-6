"""Avalia seis checkpoints congelados no mesmo teste, sem novo treinamento."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from .common import (
    CLASS_ORDER, CachedDataset, atomic_json, experiments, predict, prediction_rows,
    read_json, seed_everything, sha256_file, utc_now, write_csv,
)
from .models import build_model


def evaluate(root: Path) -> list[dict]:
    selection_path = root / "artifacts/selection.json"
    if not selection_path.exists():
        raise RuntimeError("Conclua os seis treinos e congele a seleção antes de abrir o teste.")
    selection = read_json(selection_path)
    configs = experiments(root)
    candidates = {row["id"]: row for row in selection["candidates"]}
    if set(candidates) != {config["id"] for config in configs}:
        raise ValueError("A seleção congelada não contém todos os experimentos configurados.")
    if selection["class_order"] != CLASS_ORDER or selection["test_used_for_selection"] is not False:
        raise ValueError("Protocolo de seleção inválido.")
    for split in ["train", "val", "test"]:
        if selection["data_protocol"][f"{split}_manifest_sha256"] != sha256_file(root / f"data/{split}.csv"):
            raise ValueError(f"Manifesto {split} alterado após o congelamento da seleção.")
    # Toda a validação de checkpoints ocorre antes de carregar qualquer imagem de teste.
    for config in configs:
        run = root / "artifacts/runs" / config["id"]
        if sha256_file(run / "best.pt") != candidates[config["id"]]["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint alterado depois da seleção: {config['id']}")
        result = read_json(run / "metrics.json")
        if not result["completed"]:
            raise ValueError(f"Treino incompleto: {config['id']}")
        saved_config = read_json(run / "config.json")
        if any(saved_config.get(key) != value for key, value in config.items()):
            raise ValueError(f"Configuração atual diverge do treinamento salvo: {config['id']}")
        checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
        if checkpoint["config"] != saved_config or checkpoint["class_order"] != CLASS_ORDER:
            raise ValueError(f"Metadados do checkpoint divergem dos registros: {config['id']}")
        del checkpoint
        for split in ["train", "val"]:
            if saved_config[f"{split}_manifest_sha256"] != selection["data_protocol"][f"{split}_manifest_sha256"]:
                raise ValueError("Experimentos não usaram a mesma divisão de dados.")
    summary = []
    for config in configs:
        run = root / "artifacts/runs" / config["id"]
        seed_everything(config["seed"], config["threads"])
        checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
        saved_config = checkpoint["config"]
        if checkpoint["class_order"] != CLASS_ORDER:
            raise ValueError("Ordem de classes divergente no modelo salvo.")
        for split in ["train", "val"]:
            if saved_config[f"{split}_manifest_sha256"] != sha256_file(root / f"data/{split}.csv"):
                raise ValueError("Divisão mudou depois do treinamento.")
        model = build_model(saved_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        dataset = CachedDataset(root, saved_config, "test")
        test, probabilities = predict(model, dataset, saved_config["batch_size"])
        predictions = probabilities.argmax(axis=1)
        write_csv(run / "test_predictions.csv", prediction_rows(dataset, probabilities))
        matrix = confusion_matrix(dataset.labels, predictions, labels=list(range(6)))
        write_csv(run / "confusion_matrix.csv", [
            {"true_class": name, **{predicted: int(matrix[truth, index]) for index, predicted in enumerate(CLASS_ORDER)}}
            for truth, name in enumerate(CLASS_ORDER)
        ])
        percentages = np.divide(matrix * 100.0, matrix.sum(axis=1, keepdims=True), out=np.zeros_like(matrix, dtype=float), where=matrix.sum(axis=1, keepdims=True) != 0)
        write_csv(run / "confusion_matrix_percent.csv", [
            {"true_class": name, **{predicted: float(percentages[truth, index]) for index, predicted in enumerate(CLASS_ORDER)}}
            for truth, name in enumerate(CLASS_ORDER)
        ])
        report = classification_report(dataset.labels, predictions, labels=list(range(6)), target_names=CLASS_ORDER, output_dict=True, zero_division=0)
        write_csv(run / "classification_report.csv", [
            {"class_name": name, "precision": report[name]["precision"], "recall": report[name]["recall"],
             "f1": report[name]["f1-score"], "support": int(report[name]["support"])}
            for name in CLASS_ORDER
        ])
        result = read_json(run / "metrics.json")
        result.update({
            "test_accuracy": test["accuracy"], "test_macro_f1": test["macro_f1"], "test_loss": test["loss"],
            "test_support": len(dataset), "test_manifest_sha256": sha256_file(root / "data/test.csv"),
            "evaluated_at_utc": utc_now(), "selection_sha256": sha256_file(selection_path),
            "zero_division": 0,
        })
        atomic_json(run / "metrics.json", result)
        summary.append({key: result[key] for key in [
            "id", "architecture", "image_width", "image_height", "weighted",
            "val_accuracy", "val_macro_f1", "val_loss", "test_accuracy", "test_macro_f1",
            "parameters", "model_size_bytes", "epochs_trained", "best_epoch", "train_seconds",
            "mean_epoch_seconds", "peak_rss_mb", "batch_size", "seed",
        ]})
        print(f"{config['id']}: teste acc={test['accuracy']:.4f}, F1 macro={test['macro_f1']:.4f}", flush=True)
    write_csv(root / "artifacts/summary.csv", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    evaluate(args.root.resolve())


if __name__ == "__main__":
    main()
