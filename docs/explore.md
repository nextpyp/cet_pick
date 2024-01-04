## Cellular content exploration module
For the exploration step, MiLoPYP learns an embedding space for 3D macromolecules such that similar structures are grouped together while dissimilar ones are separated. The embedding space is then projected into 2D and 3D which allows easy identification of the distribution of macromolecular structures across an entire dataset.

### Input preparation
There are two different modes for this module:

`2d3d`: Input for this mode requires aligned 2D tilt series (`.mrc`, or `.ali` as in `sample_data`), corresponding tilt-angles (``.tlt`), and 3D reconstructed tomograms (`.rec`).

`3d`: Input for this mode only requires 3D reconstructed tomograms ( `.rec`).

In `2d3d` mode, the training file should be a `.txt` file containing: tomogram name, path to 2D tilt-series, path to corresponding angles, and path to the 3D reconstructed tomograms in the following format:

```
image_name   rec_path    tilt_path   angle_path

tomo1   path_to_rec_1   path_to_tilt_1   path_to_angle_1

tomo2   path_to_rec_2   path_to_tilt_2   path_to_angle_2
...
```

For example, suppose we store tilt-series, corresponding angles, and reconstructed tomograms in the directory: `/data/sample_data`, then the training .txt file will look like this:

```
image_name   rec_path    tilt_path   angle_path

tomo1   /data/sample_data/tomo1.rec   /data/sample_data/tomo1.mrc   /data/sample_data/tomo1.tlt

tomo2   /data/sample_data/tomo2.rec   /data/sample_data/tomo1.mrc   /data/sample_data/tomo1.tlt
...
```

In `3d` mode, the training file only needs the `rec_path` and the tomogram name (`tilt_path` and `angle_path` are not needed). In this case, the file will have the following format:

```
image_name   rec_path

tomo1   path_to_rec_1

tomo2   path_to_rec_2
...
```
It is OK to use the same file formatted for `2d3d` mode as input to the `3d` mode (only `rec_path` column will be used and the rest will be ignored).

!!! warning

    Make sure the 2D tilt-series are *aligned*. They also need to have the same `x-y` dimensions as the 3D tomograms. Typically, we recommend using down-sampled tilt-series and 3D tomograms (with x-y dimensions smaller than 2000 pixels).

__Once files are generated, move all training files to `data/` directory (create `data/` directory if it doesn't exist)__

### Training
To train the exploration module in `2d3d`` mode (with tilt-series and tomograms), run:
```
python simsiam_main.py simsiam2d3d --num_epochs 300 --exp_id test_sample --bbox 36 --dataset simsiam2d3d --arch simsiam2d3d_18 --lr 1e-3 --train_img_txt sample_train_explore_img.txt --batch_size 256 --val_intervals 20 --save_all --gauss 0.8 --dog 3,5
```
In this mode, all training-related files will be saved to the folder `exp/simsiam2d3d/test_sample`, including a log file with the loss and all arguments used for training.

To train the exploration module in `3d` mode (using tomograms only), run:

```
python simsiam_main.py simsiam3d --num_epochs 300 --exp_id test_sample --bbox 36 --dataset simsiam3d --arch simsiam2d_18 --lr 1e-3 --train_img_txt sample_train_explore_img.txt --batch_size 256 --val_intervals 20 --save_all --gauss 0.8 --dog 3,5
```
In this mode, training-related files will be saved to the folder `exp/simsiam3d/test_sample`, including a log file with the loss and all used arguments.

| Arguments   | Purpose                                                                       |
|:-------------|:------------------------------------------------------------------------------|
| `num_epochs` | number of training epochs, recommend 100 to 300                                                  |
| `exp_id`       | experiment id you want to save it as.                                                                   |
| `bbox`  | bounding box size for cropped patches, should be bigger than particle size                                                               |
| `dataset`     | sampling and dataloader mode                                                                   |
| `arch`      | model backbone architecture                                                                  |
| `lr`      | learning rate                                          |
| `training_img_txt`        | input .txt used for training                                                            |
| `batch_size`       | batch size for training                                                             |
| `val_intervals`       | save model every this number of intervals                                                              |
| `save_all`       | whether to save all intermediate models at every interval                                                               |
| `gauss` | preprocessing Gaussian filter to denoise tilt-series and tomogram         |
| `dog`        | kernel sizes for difference of gaussian (DoG) pyramid, comma delimited  |

For more information regarding arguments, look at the file `opts.py`.

### Inference
After training, tomograms/tilt-series can be mapped into the embeddings using the trained model.

For `2d3d` mode, run:
```
python simsiam_test_hm_2d3d.py simsiam2d3d --exp_id test_sample --bbox 36 --dataset simsiam2d3d --arch simsiam2d3d_18 --test_img_txt sample_train_explore_img.txt --load_model exp/simsiam2d3d/test_sample/model_300.pth --gauss 0.8 --dog 3,5
```

For `3d` mode, run:

```
python simsiam_test_hm_3d.py simsiam3d --exp_id test_sample --bbox 36 --dataset simsiam3d --arch simsiam2d_18 --test_img_txt sample_train_explore_img.txt --load_model exp/simsiam3d/test_sample/model_300.pth --gauss 0.8 --dog 3,5
```
In this example, we are using a trained model from the 300th epoch.

???+ tip "Note: Please make sure you use same architecture, bounding box size, gauss, and dog argument for both training and inference and select proper trained model"
    For example, if you use: `--bbox 36 --gauss 0.8 --dog 3,5` during training, make sure the same arguments are used for inference.
    To find the arguments you used for training, go to the output folder and check the file `opts.txt`.
    For trained model selection, check the loss in the file `log.txt` and select models with lower loss.

Output from inference is saved into a `.npz` file that contains the embeddings, corresponding coordinates, original cropped patches from tomograms, and names of corresponding tomograms. The output is saved to the folder `exp/simsiam2d3d/test_sample/all_output_info.npz` (for `2d3d` mode), and to `exp/simsiam3d/test_sample/all_output_info.npz` (for `3d` mode).

### 2D visualization

There are two ways to visualize the results:

=== "2D visualization plot"

    [![2D visualization plot]][2D visualization plot]

=== "2D visualization with labels"

    [![2D visualization with labels]][2D visualization with labels]

  [2D visualization plot]: assets/test_10304_oc_umap3_3d_small.png
  [2D visualization with labels]: assets/2d_vis_with_label.png

To generate the 2D visualization plots, run:
```
python plot_2d.py --input exp/simsiam2d3d/test_sample/all_output_info.npz --n_cluster 48 --num_neighbor 40 --mode umap --path exp/simsiam2d3d/test_sample/ --min_dist_vis 1.3e-3
```
The first 2D visualization plot will be saved to `exp/simsiam2d3d/test_sample/2d_visualization_out.png`, and an additional visualization plot with labels generated using an over-clustering algorithm will be saved to `exp/simsiam2d3d/test_sample/2d_visualization_labels.png` (in the same directory). Additional outputs include the file `all_colors.npy` that will be used as input for plotting the 3D tomogram visualization, and the file `interactive_info_parquet.gzip` that contains the labels from the over-clustering algorithms and can be used as input by the interactive session. PNGs of all cropped patches will be also saved to the folder `exp/simsiam2d3d/test_sample/imgs/`. These files will be used later for by interactive session. When using the `3d` mode, replace `simsiam2d3d` with `simsiam3d`.


| Arguments   | Purpose                                                                       |
|:-------------|:------------------------------------------------------------------------------|
| `input` | output all_output_info.npz from inference                                                   |
| `n_cluster`       | number of clusters for overclustering                                                                   |
| `num_neighbor`  | number of neighbors for both tsne and umap clustering                                                               |
| `mode`     | whether to use tsne or umap for dimensionality reduction                                                                   |
| `path`      | path of directory to save all output and images                                                                  |
| `host`      | local host for images, default is 7000                                         |
| `min_dist_umap`        | min distance in UMAP                                                            |
| `min_dist_vis`       | min distance for patch display on2d visualization                                                             |

### 3D visualization

Results can also be visualized in 3D as shown in this screenshot:

[![3D visualization plot]][3D visualization plot]

  [--dirtyreload]: https://www.mkdocs.org/about/release-notes/#support-for-dirty-builds-990
  [live preview]: http://localhost:8000
  [3D visualization plot]: assets/3d_vis.png

To generate the 3D visualization plots, run:
```
python visualize_3dhm.py --input exp/simsiam2d3d/test_sample/all_output_info.npz --color exp/simsiam2d3d/test_sample/all_colors.npy --dir_simsiam exp/simsiam2d3d/test_sample/ --rec_dir sample_data/
```
This command will produce two numpy arrays: `*rec3d.npy` containing the 3D tomogram, and `*hm3d_simsiam.npy` containing the color heatmaps. To visualize the results, the two arrays can be:

1. Loaded into [napari](https://napari.org/stable/) as two layers and the transparency of each layer can be adjusted using the napari interface.

2. The two arrays can be blended using weighted averaging: `w x rec_3d.npy + (1-w) x hm3d_simsiam.npy`, and the result will be a colored 3D tomogram.

| Arguments   | Purpose                                                                       |
|:-------------|:------------------------------------------------------------------------------|
| `input` | typically the file `all_output_info.npz` (output from the inference step)                                                   |
| `color`       | `.npy` color array, generated by `plot_2d.py`                                                                   |
| `dir_simsiam`  | directory where the current run is stored                                                               |
| `rec_dir`     | directory with corresponding `.rec` files  |

For `3d` mode, replace `simsiam2d3d` with `simsiam3d`.

### 3D interactive session

Interactive sessions require loading of local images which are generated by `plot_2d.py` in the directory `exp/simsiam2d3d/test_sample/imgs/`. To connect images to the localhost and keep it running in the background, initiate a new session from the terminal using `screen`, change directory to `exp/simsiam2d3d/test_sample/` and run the command: `python -m http.server 7000`. The images will be hosted on `7000`. Detach from the `screen`` session.

!!! warning

    Make sure to use the same number as the `host` argument in `plot_2d.py`, default is 7000.

To initiate an interactive session, run:
```
python phoenix_visualization.py --input exp/simsiam2d3d/test_sample/interactive_info_parquet.gzip
```
On the terminal, it should show the headers of the parquet file, including the local host address for the interactive session. For example, here the localhost address is `http://localhost:33203/`:
[![interactive session terminal display]][interactive session terminal display]

  [--dirtyreload]: https://www.mkdocs.org/about/release-notes/#support-for-dirty-builds-990
  [live preview]: http://localhost:8000
  [interactive session terminal display]: assets/interactive_p1.jpg

You should now be able to access the interactive session at the url: `http://localhost:33203/`.

!!! warning

    If you are running everything on a remote cluster and want to visualize everything through your local browser, you will first need to connect remote to local. This needs to be done for both images and the interactive session. To connect images with localhost 7000, use `ssh -N -f -L localhost:7000:localhost:7000 your_remote_login_address` on your local terminal. To connect the interactive session with localhost 33203, use `ssh -N -f -L localhost:33203:localhost:33203 your_remote_login_address`.


In the interactive session, you should be able to visualize clusters of 3D embeddings, you will be able to adjust the number of points to be displayed in the cluster, coloring of each embedding based on the labels, select subclusters based on labels, and export selected subclusters.

=== "homepage for interactive session"

    [![homepage for interactive session]][homepage for interactive session]

=== "adjust number of points to be displayed"

    [![adjust number of points to be displayed]][adjust number of points to be displayed]

=== "coloring of embeddings based on labels"

    [![coloring of embeddings based on labels]][coloring of embeddings based on labels]

=== "select and export subclusters based on labels"

    [![select and export subclusters based on labels]][select and export subclusters based on labels]
  [homepage for interactive session]: assets/interactive_2.jpg
  [adjust number of points to be displayed]: assets/interactive_3.jpg
  [coloring of embeddings based on labels]:assets/interactive_4.jpg
  [select and export subclusters based on labels]: assets/interactive_5.jpg

### Convert exported parquet files to training coordinates file for refinement module
Exported coordinates can be downloaded through the interactive session GUI. In the example below, the downloaded parquet is named: `2023-10-08_19-44-41.parquet`.

[![download exported coordinates]][download exported coordinates]

  [--dirtyreload]: https://www.mkdocs.org/about/release-notes/#support-for-dirty-builds-990
  [live preview]: http://localhost:8000
  [dowloand exported coordinates]: assets/interactive_6.jpg

Convert parquet to training coordinates `.txt` files by running:
```
python interactive_to_training_coords.py --input path_to_dir_of_parquets --output training_coordinates.txt 
```

| Arguments   | Purpose                                                                       |
|:-------------|:------------------------------------------------------------------------------|
| `input` | directory that contains all downloaded parquet files                                                   |
| `output`       | directory where the training coordinates will be saved                                                                   |
| `if_double`  | whether z-coordinates obtained from DoGs in exploration module are downscaled by 2                                                               |

__Now, we can use the same training image file `sample_train_explore_img.txt` and the generated coordinates `training_coordinates.txt` to train the refinement module.__

