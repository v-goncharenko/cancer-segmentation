from functools import cache
from pathlib import Path

import torch

project_dir = Path(__file__).resolve().parent
data_dir = project_dir.parent / "data"

# Image size: even though image sizes are bigger than 96, we use this to speed up training
SIZE_H = SIZE_W = 96  # 128 / 192
N_CHANNELS = 3

# Number of classes in the dataset
NUM_CLASSES = 2

# B G R
# Images mean and std channelwise : ImageNet Dataset
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

# Batch size: for batch gradient descent optimization, usually selected as 2**K elements
BATCH_SIZE = 512

# Number of threads for data loader
NUM_WORKERS = 2  # 2 is optimal for google colab, for local machine use 4


@cache
def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    # if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #     return torch.device("mps")
    return torch.device("cpu")
