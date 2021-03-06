# photobashing-gan

This repository implements the PhotobashingGAN (or BashGAN for short), which
was developed for a CSC420H5 (Introduction to Image Understanding at the
University of Toronto)
final project.

## Dependencies

Please create and activate a conda environment with the dependencies listed in
`environment.yml`. Please also install `torchmetrics[image]` via `pip`.

Download and unzip the `256.zip` and `eval.zip` file stored in this
[Google Drive](https://drive.google.com/drive/folders/1vnUFLwjhkTKsRKDEv2MehG9wkMZoxkyv?usp=sharing)
into the `data/` folder, and download the generator weights into the `results/`
folder.

## Repository Organization

The following summarizes the core files and folders of this repository:

- `data/`: the data folder.
    - `256/`: the folder containing the landscape dataset (in `256/raw/`) and
      corresponding color maps (in `256/cmaps/`) on which the BashGAN was
      trained. We initially experimented with a downscaled 128x128 resolution
      dataset, but omitted the code and files from this repository.
    - `extract_cmaps.py`: preprocessing script used to extract the color maps
      from the landscape photos and cache the results.
- `results/`: the folder containing experimental results (plots, logs, etc.).
- `src/`: the source code folder.
    - `cmap/`: the folder containing utility methods used to extract a color
      map from an image. The entry method `quantize_colors()` can be found
      in `cmap/quantize.py`.
    - `experimental/`: the folder containing the training scripts used to train
      the BashGAN and compute or visualize experimental results.
        - `train_bash_gan.py`: script used to train BashGAN.
        - `eval_bash_gan.py`: script used to evaluate BashGAN (computing FID
          and LPIPS scores).
    - `gan/`: the folder containing the implementation of BashGAN. The actual
      high-level model can be found in `gan/gan.py`, while the other files
      implement the submodules (e.g. layers, generator, discriminator).

## Reproducibility

The specific commands used to train and evaluate the GANs are given in the
`scripts/` folder.

## References

- The landscape dataset used to train and evaluate BashGAN created by cleaning
  a subset of LHQ256, which was introduced
  by [ALIS](https://universome.github.io/alis). The full original dataset (90k
  images) can be downloaded on their
  [`universome/alis`](https://github.com/universome/alis) repository.
- The BashGAN architecture was adapted from
  the [SPADE](https://arxiv.org/abs/1903.07291) architecture, found in
  the [`NVlabs/SPADE`](https://github.com/NVlabs/SPADE) repository. We
  reimplement their code in a modified, simplified, and downscaled manner
  in `gan/`.
- The `posterchild` method of `quantize_colors()` uses the pallette extraction
  algorithm used by [PosterChild](https://cragl.cs.gmu.edu/posterchild/). The
  files `cmap/simplify_convexhull.py` and `cmap/trimesh.py` are directly taken
  from the [`tedchao/PosterChild`](https://github.com/tedchao/PosterChild)
  repository.
