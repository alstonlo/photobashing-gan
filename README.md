# photobashing-gan

This repository implements the Photobashing-GAN, as a part of the CSC420H5
final project.

## Dependencies

Please create and activate a conda environment with the dependencies listed in
`environment.yml`.

## Dataset

Please download and unzip the folders stored in
this [Google Drive](https://drive.google.com/drive/folders/1lME1djscAuv7eglOhcf-z4n8_kAuHvge?usp=sharing)
into the `data/` folder.

## File Structure

The following summarizes the core files and folders of this repository:

- `data/`: the dataset folder
    - `cmap/`: contains the extracted color maps from the photos in `raw/`
    - `raw/`: contains the original (raw) landscape photos
    - `extract_cmaps.py`: preprocessing script used to convert the original
      photos into color maps, which are cached
- `results/`: the folder containing experimental results (plots, logs, etc.)
- `src/`: the source code folder
    - `utils/`: contains various helper and convenience functions
        - `cmap.py`: utility functions handling color maps  

    
