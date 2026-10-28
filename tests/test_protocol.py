"""Testes de invariantes do protocolo, sem treinar nos dados reais."""

import ast
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest
import torch

from src.common import CLASS_ORDER, class_weights, metrics, select_best
from src.models import CNN, MLP, parameter_count
from src.prepare import assign_splits


def synthetic_rows():
    return [
        {"path": f"{name}/{index:03d}.jpg", "label": label, "class_name": name,
         "sha256": f"file-{label}-{index}", "pixel_sha256": f"pixel-{label}-{index}"}
        for label, name in enumerate(CLASS_ORDER) for index in range(30)
    ]


def test_class_order_matches_required_order():
    assert CLASS_ORDER == ["glass", "paper", "cardboard", "plastic", "metal", "trash"]


def test_final_experiments_preserve_controlled_pairs_and_resolution_protocol():
    from src.common import experiments
    root = Path(__file__).resolve().parents[1]
    configs = {c['id']: c for c in experiments(root)}
    for base, weighted in [('mlp_128', 'mlp_128_weighted'), ('cnn_128', 'cnn_128_weighted')]:
        comparable = lambda c: {k: v for k, v in c.items() if k not in {'id', 'weighted'}}
        assert comparable(configs[base]) == comparable(configs[weighted])
        assert configs[base]['weighted'] is False
        assert configs[weighted]['weighted'] is True
    comparable = lambda c: {k: v for k, v in c.items() if k not in {'id', 'image_width', 'image_height'}}
    assert comparable(configs['cnn_128']) == comparable(configs['cnn_256']) == comparable(configs['cnn_512'])


def test_stratified_split_is_disjoint_complete_and_reproducible():
    first, swaps = assign_splits(synthetic_rows())
    second, _ = assign_splits(list(reversed(synthetic_rows())))
    assert first == second
    assert swaps == []
    assert Counter(row["split"] for row in first) == {"train": 126, "val": 27, "test": 27}
    assert len({row["path"] for row in first}) == 180
    for split in ["train", "val", "test"]:
        assert {row["label"] for row in first if row["split"] == split} == set(range(6))


def test_duplicate_swaps_keep_stratification_and_prevent_leakage():
    original = synthetic_rows()
    baseline, _ = assign_splits(original)
    first = baseline[0]
    other = next(row for row in baseline if row["split"] != first["split"] and row["label"] != first["label"])
    for row in original:
        if row["path"] in {first["path"], other["path"]}:
            row["pixel_sha256"] = "same-pixels-conflicting-labels"
    repaired, swaps = assign_splits(original)
    assert swaps
    assert Counter((row["split"], row["label"]) for row in baseline) == Counter((row["split"], row["label"]) for row in repaired)
    groups = defaultdict(set)
    for row in repaired:
        groups[row["pixel_sha256"]].add(row["split"])
    assert all(len(splits) == 1 for splits in groups.values())
    assert len(repaired) == len(original)


def test_class_weights_and_weighted_loss_normalization():
    labels = np.repeat(np.arange(6), [2, 3, 4, 5, 6, 10])
    weights = class_weights(labels)
    assert torch.allclose(weights, torch.tensor([30 / (6 * n) for n in [2, 3, 4, 5, 6, 10]]))
    logits = torch.tensor([[2., 0., 1., -1., 0., 0.], [0., 1., 0., 0., 0., 2.]])
    target = torch.tensor([0, 5])
    individual = torch.nn.functional.cross_entropy(logits, target, reduction="none")
    expected = (individual * weights[target]).sum() / weights[target].sum()
    actual = torch.nn.functional.cross_entropy(logits, target, weight=weights)
    assert torch.allclose(actual, expected)


def test_cnn_parameters_and_initialization_are_resolution_independent():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    first = CNN().eval()
    torch.manual_seed(42)
    second = CNN().eval()
    assert parameter_count(first) == 24198
    assert all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters()))
    with torch.inference_mode():
        for height, width in [(96, 128), (192, 256), (384, 512)]:
            assert first(torch.zeros(1, 3, height, width)).shape == (1, 6)
    assert parameter_count(MLP()) == 4727366


def test_selection_ignores_test_and_uses_declared_ties():
    rows = [
        {"id": "a", "val_macro_f1": 0.5, "val_loss": 1.2, "test_macro_f1": 0.99},
        {"id": "b", "val_macro_f1": 0.6, "val_loss": 1.4, "test_macro_f1": 0.01},
        {"id": "c", "val_macro_f1": 0.6, "val_loss": 1.3, "test_macro_f1": 0.02},
    ]
    assert select_best(rows)["id"] == "c"
    rows[0]["test_macro_f1"] = 0.0
    rows[2]["test_macro_f1"] = 1.0
    assert select_best(rows)["id"] == "c"


def test_macro_f1_retains_absent_classes_with_zero_division():
    result = metrics([0, 0], [0, 0])
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1 / 6


def test_training_only_opens_train_and_validation_datasets():
    source = Path(__file__).resolve().parents[1] / "src/train.py"
    tree = ast.parse(source.read_text())
    splits = [
        node.args[2].value for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "CachedDataset"
    ]
    assert sorted(splits) == ["train", "val"]


def test_resume_reproduces_weights_and_evaluation_requires_no_training(tmp_path, monkeypatch):
    """Integração em imagens sintéticas pequenas: interrupção real entre épocas."""
    import src.train as training
    from src.common import atomic_json, read_json, sha256_file, write_csv
    from src.evaluate import evaluate

    config = {
        "id": "cnn_smoke", "architecture": "cnn", "image_width": 32, "image_height": 32,
        "weighted": True, "seed": 42, "threads": 1, "batch_size": 6,
        "learning_rate": 0.001, "weight_decay": 0.0, "max_epochs": 2,
        "patience": 7, "min_delta": 0.0,
    }

    def add_split(root, split, count):
        cache = root / "data/cache"
        cache.mkdir(parents=True, exist_ok=True)
        rows = [{"path": f"{split}/{i}.jpg", "label": i % 6} for i in range(count)]
        write_csv(root / f"data/{split}.csv", rows)
        images = np.random.default_rng(42).integers(0, 256, (count, 32, 32, 3), dtype=np.uint8)
        np.save(cache / f"32x32_{split}.npy", images)
        atomic_json(cache / f"32x32_{split}.json", {"split_manifest_sha256": sha256_file(root / f"data/{split}.csv")})

    complete, resumed = tmp_path / "complete", tmp_path / "resumed"
    for root in [complete, resumed]:
        add_split(root, "train", 18)
        add_split(root, "val", 12)
        atomic_json(root / "configs/experiments.json", {"protocol": {}, "experiments": [config]})
    training.train_one(complete, config)
    original_save = training.atomic_torch_save

    def interrupt_after_first_epoch(path, content):
        original_save(path, content)
        if path.name == "last.pt" and content["epoch"] == 1:
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(training, "atomic_torch_save", interrupt_after_first_epoch)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        training.train_one(resumed, config)
    monkeypatch.setattr(training, "atomic_torch_save", original_save)
    training.train_one(resumed, config)
    uninterrupted = torch.load(complete / "artifacts/runs/cnn_smoke/best.pt", weights_only=True)
    recovered = torch.load(resumed / "artifacts/runs/cnn_smoke/best.pt", weights_only=True)
    assert uninterrupted["epoch"] == recovered["epoch"]
    assert all(torch.equal(value, recovered["model_state_dict"][key]) for key, value in uninterrupted["model_state_dict"].items())
    assert not (resumed / "data/test.csv").exists()
    add_split(resumed, "test", 12)
    from src.prepare import write_protocol
    write_protocol(resumed)
    training.freeze_selection(resumed)

    def forbidden_optimizer(*args, **kwargs):
        raise AssertionError("A avaliação não pode iniciar treinamento.")

    monkeypatch.setattr(torch.optim, "Adam", forbidden_optimizer)
    summary = evaluate(resumed)
    assert len(summary) == 1
    saved = read_json(resumed / "artifacts/runs/cnn_smoke/metrics.json")
    assert saved["test_support"] == 12
    assert saved["epochs_trained"] == 2
