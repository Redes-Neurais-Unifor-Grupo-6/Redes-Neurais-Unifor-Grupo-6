"""Funções comuns; nenhuma delas carrega automaticamente o conjunto de teste."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
import torch
from sklearn.metrics import accuracy_score, f1_score

CLASS_ORDER = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]
EXPECTED_COUNTS = [501, 594, 403, 482, 410, 137]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path: Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def read_json(path: Path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_torch_save(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, temporary)
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict], fieldnames=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def experiments(root: Path) -> list[dict]:
    data = read_json(root / "configs/experiments.json")
    configs = [{**data["protocol"], **item} for item in data["experiments"]]
    # Estes campos descrevem escolhas fixas da implementação, não opções que
    # possam ser trocadas silenciosamente no JSON. Não inserir defaults ausentes:
    # isso preserva a identidade das configurações e checkpoints já registrados.
    supported = {
        "optimizer": "Adam",
        "device": "cpu",
        "augmentation": False,
        "normalization": "uint8 / 255.0",
        "resize": "Pillow.Resampling.LANCZOS",
        "selection_metric": "val_macro_f1",
        "validation_loss": "unweighted_cross_entropy",
        "class_weight_formula": "n_train / (6 * class_count_train)",
    }
    for config in configs:
        for field, expected in supported.items():
            if field in config and config[field] != expected:
                raise ValueError(
                    f"Configuração não implementada em {config.get('id', '<sem ID>')}: "
                    f"{field}={config[field]!r}; valor suportado: {expected!r}."
                )
    return configs


def seed_everything(seed: int, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def hardware_info() -> dict:
    cpu = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    return {
        "cpu": cpu,
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cpus": psutil.cpu_count(logical=True),
        "ram_total_gib": round(psutil.virtual_memory().total / 1024**3, 3),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
    }


def class_weights(labels) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=6)
    if len(counts) != 6 or np.any(counts == 0):
        raise ValueError("O treino deve conter todas as seis classes.")
    return torch.tensor(len(labels) / (6.0 * counts), dtype=torch.float32)


class CachedDataset:
    """Abre apenas o split solicitado. Memmap evita carregar todo cache na RAM."""

    def __init__(self, root: Path, config: dict, split: str):
        if split not in {"train", "val", "test"}:
            raise ValueError(split)
        self.rows = read_rows(root / f"data/{split}.csv")
        self.labels = np.asarray([int(row["label"]) for row in self.rows], dtype=np.int64)
        width, height = config["image_width"], config["image_height"]
        cache = root / f"data/cache/{width}x{height}_{split}.npy"
        self.images = np.load(cache, mmap_mode="r", allow_pickle=False)
        expected_shape = (len(self.rows), height, width, 3)
        if self.images.shape != expected_shape or self.images.dtype != np.uint8:
            raise ValueError(f"Cache incompatível: {cache}")
        metadata = read_json(cache.with_suffix(".json"))
        if metadata["split_manifest_sha256"] != sha256_file(root / f"data/{split}.csv"):
            raise ValueError(f"Manifesto mudou após construção do cache: {cache}")

    def __len__(self):
        return len(self.rows)

    def batches(self, batch_size: int, order=None):
        if order is None:
            order = np.arange(len(self))
        for start in range(0, len(order), batch_size):
            indices = np.asarray(order[start : start + batch_size])
            # Cópia de apenas um lote: tensor não modifica o memmap em disco.
            images = torch.from_numpy(np.array(self.images[indices], copy=True))
            images = images.permute(0, 3, 1, 2).float().div_(255.0)
            labels = torch.from_numpy(self.labels[indices].copy())
            yield images, labels


def metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(6)), average="macro", zero_division=0)),
    }


@torch.inference_mode()
def predict(model, dataset: CachedDataset, batch_size: int) -> tuple[dict, np.ndarray]:
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    probabilities = []
    for images, labels in dataset.batches(batch_size):
        logits = model(images)
        total_loss += criterion(logits, labels).item()
        probabilities.append(logits.softmax(dim=1).cpu().numpy())
    probabilities = np.concatenate(probabilities)
    result = metrics(dataset.labels, probabilities.argmax(axis=1))
    result["loss"] = total_loss / len(dataset)
    return result, probabilities


def prediction_rows(dataset: CachedDataset, probabilities: np.ndarray) -> list[dict]:
    predictions = probabilities.argmax(axis=1)
    return [
        {
            "path": row["path"],
            "y_true": int(row["label"]),
            "y_pred": int(predictions[index]),
            "confidence": float(probabilities[index, predictions[index]]),
            **{f"p_{name}": float(probabilities[index, label]) for label, name in enumerate(CLASS_ORDER)},
        }
        for index, row in enumerate(dataset.rows)
    ]


def select_best(validation_results: list[dict]) -> dict:
    """Empate: menor CE de validação, depois ID. Ignora quaisquer métricas de teste."""
    if not validation_results:
        raise ValueError("Sem resultados de validação.")
    return min(validation_results, key=lambda row: (-row["val_macro_f1"], row["val_loss"], row["id"]))


class PeakRSS:
    """Pico RSS do processo, amostrado a cada 100 ms; não é RAM da máquina."""

    def __init__(self):
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop_event.wait(0.1):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop_event.set()
        self.thread.join()

    @property
    def mb(self):
        return self.peak / 1024**2
