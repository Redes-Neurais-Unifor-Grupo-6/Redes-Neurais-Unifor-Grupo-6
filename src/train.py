"""Treina com treino/validação; congela a seleção antes da avaliação de teste."""

from __future__ import annotations

import argparse
import gc
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .common import (
    CLASS_ORDER, CachedDataset, PeakRSS, atomic_json, atomic_torch_save,
    class_weights, experiments, hardware_info, predict, prediction_rows,
    read_json, seed_everything, select_best, sha256_file, utc_now, write_csv,
)
from .models import build_model, parameter_count


def train_one(root: Path, config: dict) -> dict:
    run = root / "artifacts/runs" / config["id"]
    run.mkdir(parents=True, exist_ok=True)
    config = {
        **config, "class_order": CLASS_ORDER,
        "train_manifest_sha256": sha256_file(root / "data/train.csv"),
        "val_manifest_sha256": sha256_file(root / "data/val.csv"),
        "class_weight_source": "train.csv only",
        "early_stopping": "patience epochs without strictly greater val_macro_f1; lower val_loss breaks checkpoint ties",
        "timing_method": "sum of perf_counter epoch durations: training and validation; excludes setup, checkpoint I/O and final prediction export",
        "memory_method": "process RSS sampled every 100 ms throughout the epoch loop including checkpoint I/O; MiB; includes interpreter and mapped cache pages; no workers; CLI isolates each experiment in a process",
    }
    config_path = run / "config.json"
    if config_path.exists() and read_json(config_path) != config:
        raise ValueError(f"Configuração existente divergente em {run}. Use uma nova pasta para outro protocolo.")
    if (run / "metrics.json").exists():
        previous = read_json(run / "metrics.json")
        if previous.get("completed"):
            if sha256_file(run / "best.pt") != previous["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint alterado: {run}")
            print(f"{config['id']}: já concluído; preservando artefatos.", flush=True)
            return previous
    if (root / "artifacts/selection.json").exists():
        raise RuntimeError("Seleção já congelada. Treinamentos posteriores exigem outro diretório experimental.")
    atomic_json(config_path, config)
    seed_everything(config["seed"], config["threads"])
    train_data = CachedDataset(root, config, "train")
    val_data = CachedDataset(root, config, "val")
    weights = class_weights(train_data.labels) if config["weighted"] else None
    model = build_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    shuffle_generator = torch.Generator().manual_seed(config["seed"])
    history, best_f1, best_loss, best_epoch, bad_epochs, previous_peak = [], -1.0, float("inf"), 0, 0, 0.0
    started_at = utc_now()
    resume_path = run / "last.pt"
    if resume_path.exists():
        # Apenas checkpoints produzidos localmente por este script são retomados.
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if checkpoint["config"] != config:
            raise ValueError("Checkpoint de retomada pertence a outro protocolo.")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history = checkpoint["history"]
        best_f1, best_loss, best_epoch = checkpoint["best_f1"], checkpoint["best_loss"], checkpoint["best_epoch"]
        bad_epochs, previous_peak = checkpoint["bad_epochs"], checkpoint["peak_rss_mb"]
        started_at = checkpoint["started_at_utc"]
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        shuffle_generator.set_state(checkpoint["shuffle_rng"])
        print(f"{config['id']}: retomando após época {len(history)}.", flush=True)
        del checkpoint
    with PeakRSS() as memory:
        for epoch in range(len(history) + 1, config["max_epochs"] + 1):
            if bad_epochs >= config["patience"]:
                break
            started = time.perf_counter()
            model.train()
            order = torch.randperm(len(train_data), generator=shuffle_generator).numpy()
            numerator, denominator, correct = 0.0, 0.0, 0
            for images, labels in train_data.batches(config["batch_size"], order):
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                weight_sum = weights[labels].sum().item() if weights is not None else len(labels)
                numerator += loss.item() * weight_sum
                denominator += weight_sum
                correct += int((logits.argmax(dim=1) == labels).sum())
            validation, _ = predict(model, val_data, config["batch_size"])
            elapsed = time.perf_counter() - started
            row = {
                "epoch": epoch, "train_loss": numerator / denominator,
                "val_loss": validation["loss"], "train_accuracy": correct / len(train_data),
                "val_accuracy": validation["accuracy"], "val_macro_f1": validation["macro_f1"],
                "epoch_seconds": elapsed,
            }
            history.append(row)
            f1_gain = validation["macro_f1"] > best_f1 + config["min_delta"]
            better_checkpoint = f1_gain or (validation["macro_f1"] == best_f1 and validation["loss"] < best_loss)
            bad_epochs = 0 if f1_gain else bad_epochs + 1
            if better_checkpoint:
                best_f1, best_loss, best_epoch = validation["macro_f1"], validation["loss"], epoch
                atomic_torch_save(run / "best.pt", {
                    "model_state_dict": model.state_dict(), "config": config,
                    "class_order": CLASS_ORDER, "epoch": epoch, "seed": config["seed"],
                    "val_macro_f1": best_f1, "val_loss": best_loss,
                })
            write_csv(run / "history.csv", history)
            atomic_torch_save(resume_path, {
                "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "config": config, "class_order": CLASS_ORDER, "epoch": epoch, "seed": config["seed"],
                "history": history, "best_f1": best_f1, "best_loss": best_loss,
                "best_epoch": best_epoch, "bad_epochs": bad_epochs,
                "peak_rss_mb": max(previous_peak, memory.mb), "started_at_utc": started_at,
                "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(), "shuffle_rng": shuffle_generator.get_state(),
            })
            print(f"{config['id']} | época {epoch:02d}/{config['max_epochs']} | "
                  f"F1 val={validation['macro_f1']:.4f} | acc val={validation['accuracy']:.4f} | {elapsed:.2f}s", flush=True)
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    if checkpoint["epoch"] != best_epoch or checkpoint["val_macro_f1"] != best_f1:
        raise RuntimeError("Checkpoint e histórico inconsistentes; não publicar métricas deste treino.")
    model.load_state_dict(checkpoint["model_state_dict"])
    validation, probabilities = predict(model, val_data, config["batch_size"])
    write_csv(run / "validation_predictions.csv", prediction_rows(val_data, probabilities))
    train_seconds = sum(row["epoch_seconds"] for row in history)
    result = {
        "id": config["id"], "architecture": config["architecture"], "weighted": config["weighted"],
        "image_width": config["image_width"], "image_height": config["image_height"],
        "seed": config["seed"], "batch_size": config["batch_size"],
        "val_accuracy": validation["accuracy"], "val_macro_f1": validation["macro_f1"], "val_loss": validation["loss"],
        "parameters": parameter_count(model), "model_size_bytes": (run / "best.pt").stat().st_size,
        "epochs_trained": len(history), "best_epoch": best_epoch,
        "train_seconds": train_seconds, "mean_epoch_seconds": train_seconds / len(history),
        "peak_rss_mb": max(previous_peak, memory.mb), "hardware": hardware_info(),
        "class_weights": weights.tolist() if weights is not None else [1.0] * 6,
        "class_order": CLASS_ORDER, "completed": True,
        "stopping_reason": "early_stopping" if bad_epochs >= config["patience"] else "max_epochs",
        "started_at_utc": started_at, "completed_at_utc": utc_now(),
        "checkpoint_sha256": sha256_file(run / "best.pt"),
        "timing_method": config["timing_method"], "memory_method": config["memory_method"],
    }
    atomic_json(run / "metrics.json", result)
    print(f"{config['id']}: melhor época {best_epoch}; F1 macro val={validation['macro_f1']:.4f}.", flush=True)
    del model, optimizer, train_data, val_data, checkpoint
    gc.collect()
    return result


def freeze_selection(root: Path) -> dict | None:
    destination = root / "artifacts/selection.json"
    if destination.exists():
        return read_json(destination)
    candidates = []
    configs = experiments(root)
    for config in configs:
        run = root / "artifacts/runs" / config["id"]
        if not (run / "metrics.json").exists():
            return None
        result = read_json(run / "metrics.json")
        if not result.get("completed"):
            return None
        if "test_macro_f1" in result:
            raise RuntimeError("Não é permitido selecionar após consultar métricas de teste.")
        candidates.append({key: result[key] for key in ["id", "val_macro_f1", "val_loss", "val_accuracy", "best_epoch", "checkpoint_sha256"]})
    best = select_best(candidates)
    cnn_ids = {config["id"] for config in configs if config["architecture"] == "cnn"}
    data_protocol = read_json(root / "data/protocol.json")
    for split in ["train", "val"]:
        if data_protocol[f"{split}_manifest_sha256"] != sha256_file(root / f"data/{split}.csv"):
            raise ValueError("Os manifests mudaram desde a preparação.")
    selection = {
        "selected_run": best["id"],
        "selected_cnn_run": select_best([row for row in candidates if row["id"] in cnn_ids])["id"],
        "criterion": "val_macro_f1", "tie_break": "lowest_val_loss_then_id",
        "frozen_at_utc": utc_now(), "class_order": CLASS_ORDER, "candidates": candidates,
        "test_used_for_selection": False,
        "data_protocol": data_protocol,
    }
    atomic_json(destination, selection)
    print(f"Seleção congelada pela validação: {best['id']}. Agora src.evaluate pode abrir o teste.", flush=True)
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--experiment", help="Um ID; omitir executa os seis em ordem.")
    args = parser.parse_args()
    root = args.root.resolve()
    configs = experiments(root)
    if args.experiment:
        configs = [config for config in configs if config["id"] == args.experiment]
        if not configs:
            parser.error("ID de experimento desconhecido.")
    if args.experiment:
        train_one(root, configs[0])
    else:
        # Novo processo impede que allocators/cache de um modelo inflem o RSS do seguinte.
        for config in configs:
            subprocess.run([
                sys.executable, "-m", "src.train", "--root", str(root),
                "--experiment", config["id"],
            ], check=True)
    freeze_selection(root)


if __name__ == "__main__":
    main()
