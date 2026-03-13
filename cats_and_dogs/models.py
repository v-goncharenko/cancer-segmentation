from torch import nn

from cats_and_dogs.constants import N_CHANNELS, NUM_CLASSES, SIZE_H, SIZE_W


class LinearModel(nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            nn.Flatten(),
            nn.Linear(N_CHANNELS * SIZE_H * SIZE_W, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_CLASSES),
            nn.Softmax(dim=1),
        )


class SimpleConvModel(nn.Sequential):
    # Last layer (embeddings) size for CNN models
    EMBEDDING_SIZE = 256

    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(5),
            nn.Flatten(),
            nn.LazyLinear(self.EMBEDDING_SIZE),
            nn.ReLU(),
            nn.Linear(self.EMBEDDING_SIZE, NUM_CLASSES, bias=False),
            nn.Softmax(dim=1),
        )
