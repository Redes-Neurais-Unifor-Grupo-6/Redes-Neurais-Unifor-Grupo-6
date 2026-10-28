"""Arquiteturas próprias, sem pré-treinamento e com saídas em logits."""

from torch import nn


class MLP(nn.Sequential):
    def __init__(self, width: int = 128, height: int = 96, classes: int = 6, dropout: float = 0.3):
        super().__init__(
            nn.Flatten(),
            nn.Linear(3 * height * width, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, classes),
        )


class CNN(nn.Sequential):
    """GAP mantém os mesmos 24.198 parâmetros em todas as resoluções."""

    def __init__(self, classes: int = 6, dropout: float = 0.3):
        super().__init__(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, classes),
        )


def build_model(config: dict):
    dropout = config.get("dropout", 0.3)
    if config["architecture"] == "mlp":
        return MLP(config["image_width"], config["image_height"], dropout=dropout)
    if config["architecture"] == "cnn":
        return CNN(dropout=dropout)
    raise ValueError(f"Arquitetura desconhecida: {config['architecture']}")


def parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
