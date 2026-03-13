import zipfile
from collections.abc import Callable
from enum import Enum
from pathlib import Path

import gdown
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from cats_and_dogs.constants import BATCH_SIZE, NUM_WORKERS, data_dir
from cats_and_dogs.transforms import default_transform


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def download_gdrive(destination: Path = data_dir):
    file_id = "1FPmv21Uew6RXsxxl6LMNJTsz1G0SmzCh"
    downloaded_path = gdown.download(id=file_id, resume=True)

    with zipfile.ZipFile(downloaded_path, "r") as zip_ref:
        zip_ref.extractall(destination / downloaded_path.strip(".zip"))


class CatsDogsDataset(ImageFolder):
    def __init__(self, root: str | Path, transform: Callable | None = None):
        super().__init__(root, transform)


class CatsDogsDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int | None = BATCH_SIZE,
        shuffle: bool | None = None,
        num_workers: int = NUM_WORKERS,
        pin_memory: bool = False,
        drop_last: bool = True,
    ) -> None:
        super().__init__(
            dataset,
            batch_size,
            shuffle,
            num_workers,
            pin_memory,
            drop_last,
        )


def default_dataloader(
    root: str | Path, split: DatasetSplit, transform: Callable | None = None
) -> DataLoader:
    """Returns prepared version of DataLoader

    Args:
        root: directory with unpacked archive data, not particular split
        split: desired split as of Enum
    """
    transform = transform or default_transform()
    if split == DatasetSplit.TRAIN:
        folder_name = "train_11k"
    elif split == DatasetSplit.VAL:
        folder_name = "val"
    elif split == DatasetSplit.TEST:
        folder_name = "test_labeled"
    else:
        raise ValueError(f"Split {split} is not known")

    dataset = CatsDogsDataset(root / folder_name, transform)
    shuffle = split == DatasetSplit.TRAIN

    return DataLoader(dataset, shuffle=shuffle)
