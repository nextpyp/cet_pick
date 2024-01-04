### Particle refined localization module
For the refinement step, MiLoPYP learns to localize proteins of interests with high accuracy, when trained using sparsely annotated data. This step can be used without the previous exploration step.

### Input preparation
The training set should include two files:
1. a `.txt` file with the tomogram names and paths to the tomogram files
2. a `.txt` file with the image names and the corresponding x,y,z coordinate files

- __When using the refinement module after the exploration module__, we can use the same train image file and the generated train coordinates file from the exploration module.

- __When using the refinement module alone__, some manual labeling is needed to generate the training coordinates. A corresponding train image file will need to be generated as well.

In the first case, the text file should have the following format:

```
image_name   rec_path

tomo1 	path_to_tomo1

tomo2 	path_to_tomo2
...
```

In the second case, the text file should have the following format:

```
image_name	x_coord	y_coord	z_coord

tomo1	125	38	60

tomo1	130	45	80
...
```

#### Generate training set from manual labels

We provide a simple code to generate the described training set from a selected folder.

First, create a `train/` directory to store all tomograms and their corresponding coordinate files. Each tomogram should have its own coordinate file: e.g., `train_1_img.rec` and `train_1_img_coord.txt`.

For training coordinates, manually picking is performed on selected tomograms using `imod`. For a single tomogram, full annotation is not required. Simply find some subregions and pick around 10\% to 70\% of the particles in that subregion. The subregion does not need to be big.  After manual annotation, `imod` will generate `.mod` files containing the annotated coordinates. Converting `.mod` files to `.txt` files can be done using `imod`'s `model2point` command. For example:
```
model2point input.mod input.txt
```
Once all the `.mod` files are converted to text files, move all the `coordinates.txt` files to the `train/` directory.

!!! warning

	Depends on the x-y-z order of your input tomogram, the output coodinates generated using imod will be in different order. Two most common orders are `x-y-z` and `x-z-y`. Make sure you get the orders correct.


Once all the `.mod` files are converted to text files, move all `coordinates.txt` files to the `train/` directory.

To generate the images and coordinate files for training, run `generate_train_files.py` under the `utils/` folder. Two input arguments are required: `-d/--dir` to indicate the path to `train/` directory, and `-o/--out` to specify the name and location of the output training file. The default order for input coordinates is `x-z-y`, if you want to specify a different order, add the option `-r/--ord`. Possible orders are: `xyz`, `xzy`, or `zxy`.  For example:

```
python generate_train_files.py -e .rec -d sample_data -o sample_run -r xyz
```

| Arguments   | Purpose                                                                       |
|:-------------|:------------------------------------------------------------------------------|
| `ext` | extension of tomogram files                                                  |
| `dir`       | path to all tomograms and labels for training                                                                   |
| `out`  | training file output name                                                               |
| `ord`     | coordinate order (`xyz`, `xzy`, or `zxy`)  |
| `inference`     | generate input for evaluation stage  |

__Once files are generated, move all training files to the `data/` directory (create the `data/` directory if it doesn't exist).__

### Training

#### Globular-shaped proteins
Here is a sample command to train using tomograms from the EMPIAR-10304 dataset assuming a train image file `sample_train_explore_img.txt` and train coordinates file `training_coordinates.txt`, validation image file `sample_val_img.txt`, and validation coordinates file `val_coordinates.txt` (validation files are optional):

```
python main.py semi --down_ratio 2 --num_epochs 10 --bbox 16 --exp_id sample_refinement --dataset semi --arch unet_5 --save_all --debug 4 --val_interval 1 --thresh 0.85 --cr_weight 0.1 --temp 0.07 --tau 0.01 --lr 5e-4 --train_img_txt sample_train_explore_img.txt --train_coord_txt training_coordinates.txt --val_img_txt sample_val_img.txt --val_coord_txt val_coordinates.txt --K 900 --compress --order xzy --gauss 0.8 --contrastive --last_k 3
```

| Arguments   | Purpose                                                                       |
|:-------------|:------------------------------------------------------------------------------|
| `num_epochs` | number of training epochs, 5 to 10 recommended                                                 |
| `exp_id`       | experiment id to use as prefix for saving output files                                                                   |
| `bbox`  | box size for particles, used to generate Guassian kernel during training                                                              |
| `dataset`     | sampling and dataloader mode, defaults to `semi`                                                                   |
| `arch`      | model backbone architecture (`name_numOfLayers` format), `unet_4` or `unet_5` recommended  |
| `lr`      | learning rate, 1e-3 to 5e-4 recommended (for fewer training examples, lower the learning rate)                                         |
| `debug`        | debug mode for visualization, currently only supports mode 4 for easier visualization - output will be saved to 'debug folder' including view of each slice, ground truth heatmap, predicted heatmap, and detection prediction based on heatmap                                                           |
| `val_interval`       | interval to perform validation and save intermediate models                                                             |
| `cr_weight`       | weight for contrastive regularization (smaller values recommended for more samples, larger values for fewer samples)                                                              |
| `save_all`       | whether to save all models for each val_interval                                                               |
| `gauss` | use a Gaussian filter to denoise tilt-series and tomograms during preprocessing         |
| `temp`        | infoNCE temperature  |
| `down_ratio`        | downsampling in x-y direction, default is 2.  |
| `tau`        | class prior probability  |
| `thresh`        | threshold for soft/hard positives  |
| `last_k`        | size of convolution filter for last layer  |
| `compress`        | whether to combine 2 z-slices into 1, recommended  |
| `K`        | maximum number of particles  |
| `fiber`        | turn on for fiber/tubular-shaped particles  |

A more detailed description of arguments is included in the `opt.py` file.

All outputs will be saved to the folder `exp/semi/exp_id`. For this command, outputs will be saved into `exp/semi/sample_refinement` containing the files:

 - `opt.txt` where all the options you used will be saved
 - `debug` folder, where all outputs from validation will be saved
 - `model_xxx.pth` intermediate model checkpoints (weights for the final model will be saved in `modelxxx_last_contrastive.pth`)
 - A directory with specific training/validation loss info for each run

Here are some sample outputs generated in the `debug/` folder:
=== "Predicted heatmap"

    [![predicted heatmap]][predicted heatmap]

=== "Predicted output after NMS"

    [![predicted output after nms]][predicted output after nms]

  [predicted heatmap]: assets/6pred_hm58.png
  [predicted output after nms]: assets/6pred_out58.png


!!! question "How to select the best model and heatmap threshold?"

    The best model and heatmap threshold can be selected based on the validation loss and outputs included in the `debug/` folder.
    === "Model selection"

        When there are fully labeled tomograms for validation, select the model with the lowest validation loss.
        When there are only partially labeled tomograms, select the model that generates the best heatmaps.
        Unless there is severe over-fitting, the model from the last epoch typically generates good results.

    === "Threshold selection"

        Threshold selection can be estimated based on the detection output (`.txt` file that contains x,y,z coordinates and corresponding detection scores). It can also be estimated from `*_pred_out.png` images in the `debug/` folder that marks identified particles above a certain threshold. If there are many false positives, consider using a higher threshold.

#### Tubular-shaped proteins

Here is a sample command to train using tomograms from the EMPIAR-10987 dataset assuming a train image list `sample_train_microtubule_img.txt`, train coordinates `training_coordinates_microtubule.txt`, validation image list `sample_val_microtubule.txt`, and validation coordinates `val_coordinates_microtubule.txt` (validation files are optional):

```
python main.py semi --down_ratio 2 --num_epochs 10 --bbox 12 --contrastive --exp_id fib_test --dataset semi --arch unet_5 --save_all --debug 4 --val_interval 1 --thresh 0.3 --cr_weight 1.0 --temp 0.07 --tau 0.01 --lr 1e-4 --train_img_txt sample_train_microtubule_img.txt --train_coord_txt training_coordinates_microtubule.txt --val_img_txt sample_val_microtubule.txt --val_coord_txt val_coordinates_microtubule.txt --K 550 --compress --gauss 1 --order xzy --last_k 5 --fiber
```

Note that the main difference is the use of the `--fiber` option.

Outputs generated by this command will be the same as those generated for the globular-shaped proteins.

Here are some sample outputs saved in the `debug/` folder:

=== "Predicted heatmap"

    [![predicted microtubule heatmap][predicted microtubule heatmap]

=== "Predicted output after NMS (without postprocessing)"

    [![predicted output after nms without post processing]][predicted output after nms without post processing]

  [predicted microtubule heatmap]: assets/10_L4_ts_03pred_hm39.png
  [predicted output after nms without post processing]: assets/10_L4_ts_03pred_out39.png


### Inference

#### Globular-shaped proteins

Once training is finished, we can use the trained model for testing. `test_img.txt` that contains all tomograms can be generated using `generate_train_files.py` following similar process. To run inference on all tomograms, run:
```
python test.py semi --arch unet_5 --dataset semi --exp_id sample_refinement --load_model exp/semi/sample_refinement/model_4.pth --down_ratio 2 --K 900 --ord xzy --out_thresh 0.2 --test_img_txt test_img.txt --compress --gauss 0.8 --out_id all_out
```
Outputs are saved to the folder `exp/semi/sample_refinement/all_out/`. For each tomogram, 2 outputs are generated:

- `.txt` file with particle coordinates in x,z,y order (if the option `--with_score` was used, a score column will be added to the output)
- `*hm.mrc` a 3D detection heatmap for each tomogram

| Arguments   | Purpose                                                                       |
|:-------------|:------------------------------------------------------------------------------|
| `load_model` | path to the trained model (the command above is using the model from the 4th epoch)                                                  |
| `out_thresh`       | threshold used for detection                                                                   |
| `out_id`  | folder to save all outputs                                                               |
| `ord`     | order of the tomogram  |
| `with_score`     | whether generated output should include score values in addition to x,y,z coordinates  |

!!! warning

	Make sure to use the same `--last_k`, `--gauss`, and `--arch` options to ensure that the correct model is loaded.

#### Tubular-shaped proteins

For tubular-shaped proteins, we just add the option `--fiber` to the command above and we specify a threshold used for curve fitting to the inference command. To run inference on tomograms with fiber-specific post-processing, run:

```
python test.py semi --arch unet_5 --dataset semi --exp_id fib_test --load_model exp/semi/fib_test/model_10.pth --down_ratio 2 --K 550 --order xzy --out_thresh 0.205 --test_img_txt sample_train_microtubule_img.txt --compress --gauss 1 --cutoff_z 10 --out_id new_out --last_k 5 --fiber --curvature_cutoff 0.03 --nms 3
```
| Arguments   | Purpose                                                                       |
|:-------------|:------------------------------------------------------------------------------|
| `curvature_cutoff` | max curvature for fitted curve, segments with higher curvature will be discarded (microtubules should have small curvature) |
| `r2_cutoff` | max residual for fitted curve, discard if above the residual/bad fitting |
| `distance_cutoff` | distance cutoff for whether two points are connected in a graph |

Here are some example outputs:
=== "Before post-processing"

    [![without post-processing]][without post-processing]

=== "After post-processing"

    [![after post-processing]][after post-processing]

  [after post-processing]: assets/l4_ts_03_post.jpg
  [without post-processing]: assets/10_L4_ts_03pred_out39.png



#### Convert output coordinated to `.mod` format for visualization in IMOD
Make sure the output `*.txt` files do not include scores. Then, run:
```
point2model -scat -sphere 5 output.txt output.mod
```
See the documentation of [point2model](https://bio3d.colorado.edu/imod/doc/man/point2model.html) for more options

__Output coordinates obtained from the trained model can be extracted and used for sub-tomogram averaging.__



