import pathlib

import pytorch_lightning as pl
import torchvision.io as io
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class ColorMapDataset(Dataset):

    def __init__(self, dataset_paths):
        self.dataset_paths = dataset_paths

    def __len__(self):
        return len(self.dataset_paths)

    def __getitem__(self, idx):
        image_path, cmap_path = self.dataset_paths[idx]
        image = io.read_image(str(image_path), mode=io.ImageReadMode.RGB)
        cmap = io.read_image(str(cmap_path), mode=io.ImageReadMode.RGB)

        image = (image - 127.5) / 128.0  # normalize to [-1, 1]
        cmap = (cmap - 127.5) / 128.0
        return image, cmap


class ColorMapDatamodule(pl.LightningDataModule):

    def __init__(
            self, seed=420, resolution=256,
            batch_size=16, num_workers=4, val_samples=12, subsample=-1
    ):
        super().__init__()

        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers

        data_dir = pathlib.Path(__file__).parents[2] / "data" / str(resolution)
        raw_dir = data_dir / "raw"
        image_paths = list(sorted(raw_dir.glob("*.png")))
        image_paths = image_paths[:subsample] if (subsample > 0) else image_paths

        def cmap_path(_image_path):
            return data_dir / "cmaps" / f"cmap_{_image_path.stem}.png"

        dataset_paths = [(p, cmap_path(p)) for p in image_paths]

        # (X, 10) train-val split
        pl.seed_everything(seed=seed)
        train_size = len(dataset_paths) - val_samples
        train_paths, val_paths = train_test_split(dataset_paths, train_size=train_size, random_state=seed)

        self.train = ColorMapDataset(train_paths)
        self.val = ColorMapDataset(val_paths)

    def train_dataloader(self):
        return DataLoader(self.train, self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, 1, num_workers=self.num_workers)
