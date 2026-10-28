"""Regressões de preservação dos artefatos congelados, sem imagens reais."""

import copy

import pytest
import torch

from src.common import CLASS_ORDER, atomic_json, sha256_file
from src.prepare import write_manifests, write_protocol


def test_prepare_preserves_frozen_manifests_on_mismatch(tmp_path):
    rows = [
        {"path": f"{split}/{i}.jpg", "label": i, "class_name": name,
         "split": split, "sha256": f"bytes-{split}-{i}", "pixel_sha256": f"rgb-{split}-{i}"}
        for split in ["train", "val", "test"] for i, name in enumerate(CLASS_ORDER)
    ]
    write_manifests(tmp_path, rows)
    frozen = write_protocol(tmp_path)
    atomic_json(tmp_path / "artifacts/selection.json", {"data_protocol": frozen})
    before = {p.name: p.read_bytes() for p in (tmp_path / "data").iterdir()}
    write_manifests(tmp_path, rows)
    assert before == {p.name: p.read_bytes() for p in (tmp_path / "data").iterdir()}
    altered = copy.deepcopy(rows)
    altered[-1]["pixel_sha256"] = "different-image"
    with pytest.raises(ValueError, match="seleção congelada"):
        write_manifests(tmp_path, altered)
    assert before == {p.name: p.read_bytes() for p in (tmp_path / "data").iterdir()}


@pytest.mark.parametrize("changed_file", ["current_config", "saved_config"])
def test_evaluation_refuses_configuration_drift_before_test(tmp_path, monkeypatch, changed_file):
    import src.evaluate as evaluation
    config = {"id": "fixture", "architecture": "cnn", "batch_size": 6, "seed": 42, "threads": 1}
    atomic_json(tmp_path / "configs/experiments.json", {"protocol": {}, "experiments": [config]})
    for split in ["train", "val", "test"]:
        path = tmp_path / f"data/{split}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("path,label\nexample.jpg,0\n")
    protocol = write_protocol(tmp_path)
    saved = {**config, **{f"{split}_manifest_sha256": protocol[f"{split}_manifest_sha256"]
                         for split in ["train", "val"]}}
    run = tmp_path / "artifacts/runs/fixture"
    run.mkdir(parents=True)
    atomic_json(run / "config.json", saved)
    atomic_json(run / "metrics.json", {"completed": True})
    torch.save({"config": saved, "class_order": CLASS_ORDER}, run / "best.pt")
    atomic_json(tmp_path / "artifacts/selection.json", {
        "class_order": CLASS_ORDER, "test_used_for_selection": False,
        "data_protocol": protocol,
        "candidates": [{"id": "fixture", "checkpoint_sha256": sha256_file(run / "best.pt")}],
    })
    if changed_file == "current_config":
        atomic_json(tmp_path / "configs/experiments.json", {
            "protocol": {}, "experiments": [{**config, "batch_size": 12}]})
    else:
        atomic_json(run / "config.json", {**saved, "undeclared_change": True})

    def forbidden_dataset(*args, **kwargs):
        raise AssertionError("Não se pode abrir o teste antes de validar todas as configurações.")

    monkeypatch.setattr(evaluation, "CachedDataset", forbidden_dataset)
    with pytest.raises(ValueError, match="diverge|divergem"):
        evaluation.evaluate(tmp_path)
