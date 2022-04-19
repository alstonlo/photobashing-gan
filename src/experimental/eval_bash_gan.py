import argparse
import os
import pathlib

import numpy as np
import torch
import torchvision.io as io
import wandb
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

from src.cmap.quantize import quantize_colors

os.environ["WANDB_START_METHOD"] = "thread"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJ_DIR = pathlib.Path(__file__).parents[2]


class EvalDataset(Dataset):

    def __init__(self):
        super().__init__()

        data_dir = PROJ_DIR / "data" / "eval"
        self.image_paths = list(sorted(data_dir.glob("*.png")))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = io.read_image(str(image_path), mode=io.ImageReadMode.RGB)
        cmap = self._extract_cmap(image)
        return image, cmap

    def _extract_cmap(self, image):
        image = torch.permute(image, (1, 2, 0))
        image = image.numpy()
        image = image.astype(np.float32) / 255
        cmap = quantize_colors(image, method="k_means")
        cmap = np.round(255 * cmap).astype(np.uint8)
        cmap = torch.from_numpy(cmap)
        return cmap.permute((2, 0, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--model", type=str, default="bash_gan")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    model_path = PROJ_DIR / "results" / f"{args.model}.pt"
    gan = torch.load(str(model_path), map_location=DEVICE)
    gan.eval()

    dataloader = DataLoader(
        dataset=EvalDataset(),
        batch_size=100,
        shuffle=False,
        num_workers=args.num_workers
    )
    args.data_size = len(dataloader.dataset)

    fid = FrechetInceptionDistance().to(DEVICE)
    lpips = LearnedPerceptualImagePatchSimilarity().to(DEVICE)

    lpips_score = 0

    for i, (real_image, cmap) in enumerate(tqdm(dataloader)):
        real_image = real_image.to(DEVICE)
        cmap = cmap.to(DEVICE)
        cmap_batch = torch.concat([cmap] * 2, dim=0)
        cmap_batch = (cmap_batch.float() - 127.5) / 128.0  # normalize to [-1, 1]

        fake_images = gan._sample_images(cmap_batch)
        lpips(fake_images[:100, ...], fake_images[100:, ...])

        fake_images = torch.round(128.0 * fake_images + 127.5).to(torch.uint8)
        fid.update(real_image, real=True)
        fid.update(fake_images, real=False)

        if args.debug and (i == 1):
            break

    fid_score = fid.compute()
    lpips_score = lpips_score / len(dataloader)

    print(f"FID:   {fid_score}")
    print(f"LPIPS: {lpips_score}")

    wandb.init(project="photobash_gan", dir=str(PROJ_DIR / "results"))
    wandb.log({"FID": fid_score, "LPIPS": lpips_score})


if __name__ == "__main__":
    with torch.no_grad():
        main()
