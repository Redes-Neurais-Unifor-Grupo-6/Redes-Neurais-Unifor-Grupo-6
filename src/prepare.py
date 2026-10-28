"""Audita TrashNet, fixa uma divisão sem duplicatas entre conjuntos e cria caches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from .common import CLASS_ORDER, EXPECTED_COUNTS, atomic_json, read_json, sha256_file, write_csv

FIELDS = ["path", "label", "class_name", "split", "sha256", "pixel_sha256"]


def write_manifests(root: Path, rows: list[dict]) -> None:
    """Recusa uma nova divisão incompatível antes de alterar qualquer manifesto."""
    subsets = {split: [row for row in rows if row["split"] == split]
               for split in ["train", "val", "test"]}
    selection = root / "artifacts/selection.json"
    if selection.exists():
        frozen = read_json(selection)["data_protocol"]
        for split, subset in subsets.items():
            buffer = io.StringIO(newline="")
            writer = csv.DictWriter(buffer, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(subset)
            digest = hashlib.sha256(buffer.getvalue().encode("utf-8")).hexdigest()
            if digest != frozen[f"{split}_manifest_sha256"]:
                raise ValueError(
                    f"A preparação geraria um manifesto {split} diferente da seleção congelada. "
                    "Os manifestos publicados foram preservados; confira os dados e as versões."
                )
    write_csv(root / "data/splits.csv", rows, FIELDS)
    for split, subset in subsets.items():
        write_csv(root / f"data/{split}.csv", subset, FIELDS)


def write_protocol(root: Path) -> dict:
    """Metadados opacos permitem congelar o teste sem lê-lo durante o treino."""
    protocol = {f"{split}_manifest_sha256": sha256_file(root / f"data/{split}.csv") for split in ["train", "val", "test"]}
    protocol["class_order"] = CLASS_ORDER
    atomic_json(root / "data/protocol.json", protocol)
    return protocol


def assign_splits(rows: list[dict], seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Estratifica e troca arquivos da mesma classe para agrupar pixels idênticos.

    As trocas usam somente classe, hash e ordem lexicográfica; preservam exatamente
    tamanhos e contagens por classe do split inicial, sem consultar resultados.
    """
    rows = [dict(row) for row in sorted(rows, key=lambda item: item["path"])]
    indices = np.arange(len(rows))
    labels = np.asarray([row["label"] for row in rows])
    train, holdout = train_test_split(indices, test_size=0.3, stratify=labels, random_state=seed)
    val, test = train_test_split(holdout, test_size=0.5, stratify=labels[holdout], random_state=seed)
    for split, subset in [("train", train), ("val", val), ("test", test)]:
        for index in subset:
            rows[index]["split"] = split
    groups = defaultdict(list)
    for row in rows:
        groups[row["pixel_sha256"]].append(row)
    swaps = []
    duplicate_groups = sorted((group for group in groups.values() if len(group) > 1), key=lambda group: group[0]["path"])
    for group in duplicate_groups:
        anchor_split = group[0]["split"]
        for member in group[1:]:
            if member["split"] == anchor_split:
                continue
            old_split = member["split"]
            candidates = [
                row for row in rows
                if row["split"] == anchor_split and row["label"] == member["label"]
                and len(groups[row["pixel_sha256"]]) == 1
            ]
            if not candidates:
                raise ValueError("Não existe troca estratificada segura para agrupar duplicata.")
            replacement = candidates[0]
            swaps.append({
                "duplicate_path": member["path"], "swap_path": replacement["path"],
                "class_name": member["class_name"], "from_split": old_split,
                "to_split": anchor_split, "anchor_path": group[0]["path"],
            })
            member["split"], replacement["split"] = anchor_split, old_split
    if any(len({row["split"] for row in group}) > 1 for group in groups.values()):
        raise AssertionError("Hashes idênticos aparecem em conjuntos diferentes.")
    return rows, swaps


def duplicate_report(rows: list[dict], key: str) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return [
        {"hash": digest, "paths": [row["path"] for row in group],
         "classes": [row["class_name"] for row in group],
         "label_conflict": len({row["label"] for row in group}) > 1}
        for digest, group in sorted(groups.items()) if len(group) > 1
    ]


def extract_zip(path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"Caminho inseguro no ZIP: {member.filename}")
        archive.extractall(destination)


def prepare(root: Path, dataset_root: Path, zip_path: Path | None = None) -> dict:
    if not dataset_root.exists():
        if zip_path is None or not zip_path.exists():
            raise FileNotFoundError("Extraia dataset-resized.zip em data/raw ou informe --zip.")
        extract_zip(zip_path, root / "data/raw")
    rows, corrupt, dimensions, modes = [], [], Counter(), Counter()
    for label, class_name in enumerate(CLASS_ORDER):
        paths = sorted(path for path in (dataset_root / class_name).iterdir() if path.is_file() and not path.name.startswith("."))
        for path in paths:
            relative = path.resolve().relative_to(root.resolve()).as_posix()
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    dimensions[f"{image.width}x{image.height}"] += 1
                    modes[image.mode] += 1
                    rgb = image.convert("RGB")
                    pixels = hashlib.sha256(f"{rgb.width}x{rgb.height}:RGB:".encode() + rgb.tobytes()).hexdigest()
                rows.append({"path": relative, "label": label, "class_name": class_name,
                             "sha256": sha256_file(path), "pixel_sha256": pixels})
            except Exception as error:
                corrupt.append({"path": relative, "error": str(error)})
    counts = Counter(row["class_name"] for row in rows)
    audit = {
        "class_order": CLASS_ORDER, "total_images": len(rows), "counts": dict(counts),
        "dimensions": dict(dimensions), "modes": dict(modes), "corrupt_files": corrupt,
        "byte_duplicates": duplicate_report(rows, "sha256"),
        "pixel_duplicates": duplicate_report(rows, "pixel_sha256"),
        "zip_sha256": sha256_file(zip_path) if zip_path and zip_path.exists() else None,
        "seed": 42,
    }
    atomic_json(root / "data/audit.json", audit)
    if corrupt or [counts[name] for name in CLASS_ORDER] != EXPECTED_COUNTS:
        raise ValueError("A auditoria não corresponde à referência de 2.527 imagens. Veja data/audit.json.")
    if dimensions != Counter({"512x384": 2527}):
        raise ValueError("A referência deve conter somente imagens de 512 × 384 pixels.")
    rows, swaps = assign_splits(rows)
    write_manifests(root, rows)
    audit["split_counts"] = {}
    for split in ["train", "val", "test"]:
        subset = [row for row in rows if row["split"] == split]
        audit["split_counts"][split] = dict(Counter(row["class_name"] for row in subset))
    audit["duplicate_swaps"] = swaps
    audit["duplicate_policy"] = (
        "Arquivos com pixels RGB idênticos são mantidos no mesmo conjunto. Grupos com "
        "rótulos divergentes são preservados para manter as 2.527 imagens e a distribuição "
        "de referência; trocas determinísticas de imagens da mesma classe preservam "
        "as contagens estratificadas. Nenhuma decisão utiliza métricas de modelo."
    )
    audit["label_conflict_groups"] = sum(group["label_conflict"] for group in audit["pixel_duplicates"])
    audit["label_conflict_files"] = sum(len(group["paths"]) for group in audit["pixel_duplicates"] if group["label_conflict"])
    audit["cross_split_pixel_duplicates"] = 0
    audit["manifest_sha256"] = sha256_file(root / "data/splits.csv")
    atomic_json(root / "data/audit.json", audit)
    write_protocol(root)
    cache_dir = root / "data/cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for width, height in [(128, 96), (256, 192), (512, 384)]:
        for split in ["train", "val", "test"]:
            subset = [row for row in rows if row["split"] == split]
            cache = cache_dir / f"{width}x{height}_{split}.npy"
            metadata = {
                "split_manifest_sha256": sha256_file(root / f"data/{split}.csv"),
                "shape": [len(subset), height, width, 3], "dtype": "uint8",
                "resize": "Pillow.Resampling.LANCZOS", "color_mode": "RGB",
            }
            if cache.exists() and cache.with_suffix(".json").exists() and read_json(cache.with_suffix(".json")) == metadata:
                print(f"Cache já existe: {cache.name}", flush=True)
                continue
            temporary = cache.with_suffix(".npy.tmp")
            memory = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint8, shape=tuple(metadata["shape"]))
            for index, row in enumerate(subset):
                with Image.open(root / row["path"]) as image:
                    image = image.convert("RGB")
                    if image.size != (width, height):
                        image = image.resize((width, height), Image.Resampling.LANCZOS)
                    memory[index] = np.asarray(image, dtype=np.uint8)
            memory.flush()
            del memory
            os.replace(temporary, cache)
            atomic_json(cache.with_suffix(".json"), metadata)
            print(f"Cache preparado: {cache.name}", flush=True)
    print(f"Preparação concluída: {len(rows)} imagens; {len(swaps)} trocas contra vazamento.", flush=True)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--zip", dest="zip_path", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    prepare(root, args.dataset_root or root / "data/raw/dataset-resized", args.zip_path or root / "data/raw/dataset-resized.zip")


if __name__ == "__main__":
    main()
