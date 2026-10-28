"""Regressões da revisão: não alteram dados ou resultados experimentais."""

import json

import pytest
import torch
from torch import nn

from src.models import build_model
from src.search import search
from src.common import experiments


@pytest.mark.parametrize("pilot", [None, 0.0001])
def test_search_and_pilot_reject_post_test_before_mutating(tmp_path, monkeypatch, pilot):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "experiments.json").write_text(json.dumps({
        "protocol": {},
        "experiments": [{"id": "mlp_128", "architecture": "mlp"}],
    }))
    run = tmp_path / "artifacts/runs/mlp_128"
    run.mkdir(parents=True)
    predictions = run / "test_predictions.csv"
    predictions.write_text("path,y_true,y_pred\nexample.jpg,0,0\n")
    before = {str(path.relative_to(tmp_path)): path.read_bytes()
              for path in tmp_path.rglob("*") if path.is_file()}

    def forbidden_train(*_args, **_kwargs):
        raise AssertionError("A busca não pode iniciar treino após acessar o teste.")

    monkeypatch.setattr("src.search.train_one", forbidden_train)
    with pytest.raises(RuntimeError, match="teste já foi avaliado"):
        search(tmp_path, pilot=pilot)
    assert not (tmp_path / "artifacts/search").exists()
    assert before == {str(path.relative_to(tmp_path)): path.read_bytes()
                      for path in tmp_path.rglob("*") if path.is_file()}


@pytest.mark.parametrize("architecture,expected_layers", [("mlp", 2), ("cnn", 1)])
def test_build_model_honors_dropout_and_preserves_checkpoint_schema(architecture, expected_layers):
    config = {"architecture": architecture, "image_width": 128, "image_height": 96}
    torch.manual_seed(42)
    default = build_model(config)
    torch.manual_seed(42)
    changed = build_model({**config, "dropout": 0.8})
    assert [layer.p for layer in default.modules() if isinstance(layer, nn.Dropout)] == [0.3] * expected_layers
    assert [layer.p for layer in changed.modules() if isinstance(layer, nn.Dropout)] == [0.8] * expected_layers
    # Dropout não possui tensores: mudar p não altera nomes, formas ou inicialização.
    assert default.state_dict().keys() == changed.state_dict().keys()
    assert all(torch.equal(value, changed.state_dict()[key])
               for key, value in default.state_dict().items())
    changed.load_state_dict(default.state_dict(), strict=True)


@pytest.mark.parametrize("field,unsupported", [
    ("optimizer", "SGD"),
    ("device", "cuda"),
    ("augmentation", True),
    ("normalization", "z-score"),
    ("resize", "Pillow.Resampling.NEAREST"),
    ("selection_metric", "test_accuracy"),
    ("validation_loss", "weighted_cross_entropy"),
    ("class_weight_formula", "inverse_square_root"),
])
def test_unimplemented_declarations_are_rejected(tmp_path, field, unsupported):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "experiments.json").write_text(json.dumps({
        "protocol": {field: unsupported}, "experiments": [{"id": "smoke"}],
    }))
    with pytest.raises(ValueError, match=f"{field}="):
        experiments(tmp_path)


def test_missing_declarations_preserve_config_and_overrides_are_validated(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    path = config_dir / "experiments.json"
    path.write_text(json.dumps({"protocol": {"seed": 42}, "experiments": [{"id": "smoke"}]}))
    assert experiments(tmp_path) == [{"seed": 42, "id": "smoke"}]
    path.write_text(json.dumps({
        "protocol": {"optimizer": "Adam"},
        "experiments": [{"id": "smoke", "optimizer": "SGD"}],
    }))
    with pytest.raises(ValueError, match="optimizer="):
        experiments(tmp_path)
