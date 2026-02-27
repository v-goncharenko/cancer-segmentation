from pathlib import Path

from torch.utils.data import Dataset


class CancerSegmentationDataset(Dataset):
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir

    def __getitem__(self, index: int) -> torch.tensor:
        return torch.random((512, 512))

    def __len__(self):
        return 10_000
