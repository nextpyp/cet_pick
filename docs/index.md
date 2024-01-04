# Getting Started

MiLoPYP is an open source, dataset-specific contrastive learning-based framework that enables two-step fast molecular pattern visualization followed by accurate protein localization without the need for manual annotation. During the exploration step, it learns an embedding space for 3D macromolecules such that similar structures are grouped together while dissimilar ones are separated. The embedding space is then projected into 2D and 3D which allows easy identification of the distribution of macromolecular structures across an entire dataset. During the refinement step, examples of proteins identified during the exploration step are selected and MiLoPYP learns to localize these proteins with high accuracy.

Each step can be used separately. To use the refinement step only (tomogram particle detection), ground truth particle coordinates need to be provided for training. Typically, around 200 particles from several tomograms are needed to ensure good performance. Training coordinates can be obtained either manually or from the exploration module.

## Installation

The code was tested on CentOS Stream (version 8.0), using [Anaconda](https://www.anaconda.com/download) Python 3.8, [PyTorch]((http://pytorch.org/)) version 1.11.0, and CUDA 10.2. NVIDIA GPUs with 32GB RAMs were used for training. Inference can be performed on either GPUs or CPUs.

After installing Anaconda:

0. Create a new conda environment [optional, but recommended]:

    ```
    conda create --name MiLoPYP python=3.8
    ```

    And activate the environment.

    ```
    conda activate MiLoPYP
    ```

1. Clone the `cet_pick` repo and `cd` to the corresponding directory:

    ```
    git clone https://github.com/nextpyp/cet_pick.git
    cd cet_pick
    ```

2. Install the requirements:

    ```
    pip install -r requirements.txt
    ```

3. Install PyTorch:

    ```
    pip install torch==1.11.0+cu102 torchvision==0.12.0+cu102 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu102
    ```

4. Install `cet_pick` package and dependencies:

    ```
    pip install -e cet_pick
    ```

## Folder structure

MiLoPYP uses the following directory structure:

```

├── data                                # training data
│   ├── sample_train_explore_img.txt
│   ├── sample_train_refine_img.txt
│   ├── training_coordinates.txt
├── datasets                            # dataloading, sampling related code
│   ├── dataset_factory.py              # dataset factory and sampling factory
│   ├── tomo_*.py                       # data factory for different modes
│   ├── particle_*.py                   # sampling factory for different modes
├── trains                              # model training modules
├── models                              # model architectures for different modes
├── utils                               # util functions
├── colormap                            # colormaps for 2D visualization
└── DCNv2                               # deformable convolution related operations
├── opts.py                             # arguments for training
├── main.py                             # training for refinement module
├── simsiam_main.py                     # training for cellular content exploration module
├── simsiam_test_hm_3d.py               # inference for cellular content exploration module
├── test.py                             # inference for refinement/particle detection module
├── interactive_to_training_coords.py   # convert output from interactive session
├── plot_2d.py                          # 2D visualization plots
├── phoenix_visualization.py            # 3D interactive session

```

## Sample datasets

### Globular-shaped particles (EMPIAR-10304)

This dataset contains a subset of tilt-series from EMPIAR-10304 and all the necessary metadata to run MiLoPYP. 
To download and decompress, run:
```
wget https://nextpyp.app/data/milopyp_globular_tutorial.tbz
tar xvfz milopyp_globular_tutorial.tbz
```

### Tubular-shaped particles (EMPIAR-10987)
This dataset contains a subset of tilt-series from EMPIAR-10987 and all the necessary metadata to run MiLoPYP. 
To download and decompress, run:
```
wget https://nextpyp.app/data/milopyp_tubular_tutorial.tbz
tar xvfz milopyp_tubular_tutorial.tbz
```

MiLoPYP consists of two modules:

- [Cellular content exploration](explore.md)

- [Accurate particle localization](refine.md)

The [quick tutorials](quick_tutorial.md) contain a step-by-step guide on how to run MiLoPYP on two sample datasets.

<!-- For full documentation visit [mkdocs.org](https://www.mkdocs.org). -->

<!-- ## Commands

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.
 -->