import pathlib

import pytorch_lightning as pl
import torchvision.io as io
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm


class CmapDatamodule(pl.LightningDataModule):

    def __init__(
            self, seed=420, resolution=256,
            batch_size=10, num_workers=4, val_samples=10, subsample=-1
    ):
        super().__init__()

        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers

        data_dir = pathlib.Path(__file__).parents[2] / "data" / str(resolution)
        raw_dir = data_dir / "raw"
        image_paths = list(sorted(raw_dir.glob("*.png")))
        image_paths = image_paths[:subsample] if (subsample > 0) else image_paths

        dataset = []
        for image_path in tqdm(image_paths, desc="Loading Images"):
            cmap_path = data_dir / "cmaps" / f"cmap_{image_path.stem}.png"
            image = io.read_image(str(image_path), mode=io.ImageReadMode.RGB)
            cmap = io.read_image(str(cmap_path), mode=io.ImageReadMode.RGB)

            image = (image - 127.5) / 128.0  # normalize to [-1, 1]
            cmap = (cmap - 127.5) / 128.0
            dataset.append((image, cmap))

        # (X, 10) train-val split
        pl.seed_everything(seed=seed)
        train_size = len(dataset) - val_samples
        self.train, self.val = train_test_split(dataset, train_size=train_size, random_state=seed)

    def train_dataloader(self):
        return DataLoader(self.train, self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, 1, num_workers=self.num_workers)
