from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.cmap.quantize import quantize_colors

RES_FOLDER = Path(__file__).parent / "256"
RAW_DIR = RES_FOLDER / "raw"
CMAP_DIR = RES_FOLDER / "cmaps"
CMAP_DIR.mkdir(exist_ok=True)

IMAGE_PATHS = list(sorted(RAW_DIR.glob("*.png")))


def extract_cmaps():
    for image_path in tqdm(IMAGE_PATHS, desc="Process Images"):
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255

        cmap = quantize_colors(image, method="k_means")

        cmap = np.round(255 * cmap).astype("uint8")
        cmap = cv2.cvtColor(cmap, cv2.COLOR_RGB2BGR)

        cmap_path = CMAP_DIR / f"cmap_{image_path.stem}.png"
        cv2.imwrite(str(cmap_path), cmap)


if __name__ == "__main__":
    extract_cmaps()
