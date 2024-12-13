# WassersteinBarycenterTextureMixing

This repository is part of the practical work conducted during my Master's program in Artificial Intelligence at Claude Bernard University Lyon 1. The project is based on the course material [here](https://perso.liris.cnrs.fr/nicolas.bonneel/Lyon1_Transport.pdf).

The objective is to perform texture mixing using the Sliced Wasserstein Distance (SWD). The method enables blending features of two images, referred to as the source and target, while preserving texture consistency.

The `main.py` script implements the following steps:

1. **Image Loading:** Load and normalize source (`f.jpg`) and target (`g.jpg`) images.
2. **Resizing:** Resize both images to a common size for consistent processing.
3. **Texture Mixing:** Apply Sliced Wasserstein Distance (SWD) to iteratively mix textures between the source and target images.
4. **Saving Result:** Save the mixed image in the original size of the target image.

## Setup

### Create a Virtual Environment

To set up the project environment, use the provided `env.yml` file with `conda`:

```bash
conda env create -f env.yml
conda activate color_transfer_env
```

### Run the Script

Execute the script to perform texture mixing:

```bash
python main.py
```
