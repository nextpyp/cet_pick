## Quick tutorials

### Globular-shaped targets

First, we need to download and decompress the data consisting of a subset of tilt-series and tomograms obtained from EMPIAR-10304:

```
wget https://nextpyp.app/files/data/milopyp_globular_tutorial.tbz
tar xvfz milopyp_globular_tutorial.tbz
```
The sample dataset includes the following files:

- `tilt?.rec`: tomograms (downsampled to size 512x512x256)
- `tilt?.ali`: aligned tilt-series (downsampled to size 512x512x41)
- `tilt?.tlt`: corresponding tilt-angles
- `2023-10-08_19-44-41.parquet`: output from interactive session that includes selected coordinates to train the refinement module
- `sample_train_explore_img.txt`: image file to use as input to train the exploration and refinement modules
- `training_coordinates.txt`: coordinates for training the refinement module (converted from the parquet file above)

Next, go to the folder where `main.py` and `test.py` are located (this is simply to avoid having to enter absolute paths in the image file). Create a folder named `data/` and move the `*.txt` files there. Create another folder named `sample_data/` and move the `*.rec`, `*.ali` and `*.tlt` files there:

```
├── data
│   ├── sample_train_explore_img.txt
│   ├── training_coordinates.txt
├── sample_data
│   ├── *.rec
│   ├── *.ali
│   ├── *.tlt
├── main.py
├── test.py
```

#### Cellular content exploration

To start training, run:

```
python simsiam_main.py simsiam2d3d --num_epochs 20 --exp_id test_sample --bbox 36 --dataset simsiam2d3d --arch simsiam2d3d_18 --lr 1e-3 --train_img_txt sample_train_explore_img.txt --batch_size 256 --val_intervals 20 --save_all --gauss 0.8 --dog 3,5
```

Outputs produced by this command include: the loss for each epoch, trained models saved every 20 epochs, and a file with all program options.

Once trained, we will map tomograms/tilt-series into embeddings by running:

```
python simsiam_test_hm_2d3d.py simsiam2d3d --exp_id test_sample --bbox 36 --dataset simsiam2d3d --arch simsiam2d3d_18 --test_img_txt sample_train_explore_img.txt --load_model exp/simsiam2d3d/test_sample/model_20.pth --gauss 0.8 --dog 3,5
```

In the folder `exp/simsiam2d3d/test_sample/`, you will find the file `all_output_info.npz` containing the embeddings, corresponding coordinates, original cropped patches from tomograms, and the names of corresponding tomograms.

This is what the folder structure should look like:

[![Output snapshot]][Output snapshot]

  [--dirtyreload]: https://www.mkdocs.org/about/release-notes/#support-for-dirty-builds-990
  [live preview]: http://localhost:8000
  [Output snapshot]: assets/outputs_sample.jpg


##### 2D visualization
Run the following command:
```
python plot_2d.py --input exp/simsiam2d3d/test_sample/all_output_info.npz --n_cluster 48 --num_neighbor 40 --mode umap --path exp/simsiam2d3d/test_sample/ --min_dist_vis 1.3e-3 
```

##### 3D visualization
Run the following command:

```
python visualize_3dhm.py --input exp/simsiam2d3d/test_sample/all_output_info.npz --color exp/simsiam2d3d/test_sample/all_colors.npy --dir_simsiam exp/simsiam2d3d/test_sample/ --rec_dir sample_data/
```

##### 3D interactive session

To launch, run:

```
python phoenix_visualization.py --input exp/simsiam2d3d/test_sample/interactive_info_parquet.gzip
```

To save the downloaded parquet files from the interactive session in `txt` format, run:

```
python interactive_to_training_coords.py --input path_to_all_parquet_files --output training_coordinates.txt
```
Note: `--input` should contain the path to where all parquet files are stored (not the path to a single parquet file)

The contents of the output file `training_coordinates.txt` should coincide with the file downloaded above `training_coordinates.txt`.

#### Refined particle localization
To train the model for refined particle localization, run:

```
python main.py semi --down_ratio 2 --num_epochs 10 --bbox 16 --exp_id sample_refinement --dataset semi --arch unet_4 --save_all --debug 4 --val_interval 1 --thresh 0.85 --cr_weight 0.1 --temp 0.07 --tau 0.01 --lr 5e-4 --train_img_txt sample_train_explore_img.txt --train_coord_txt training_coordinates.txt --val_img_txt sample_train_explore_img.txt --val_coord_txt training_coordinates.txt --K 900 --compress --order xzy --gauss 0.8 --contrastive --last_k 3
```

To run inference using the trained model, run:

```
python test.py semi --arch unet_4 --dataset semi --exp_id sample_refinement --load_model exp/semi/sample_refinement/model_4.pth --down_ratio 2 --K 900 --ord xzy --out_thresh 0.25 --test_img_txt test_img.txt --compress --gauss 0.8 --out_id all_out
```

Finally, output coordinates will be saved into `exp/semi/sample_refinement/all_out/*.txt`.

### Tubular-shaped targets

Similar to the globular-shaped case, we download and decompress the data consisting of a subset of tomograms obtained from EMPIAR-10987:
```
wget https://nextpyp.app/files/data/milopyp_tubular_tutorial.tbz
tar xvfz milopyp_tubular_tutorial.tbz
```

The sample dataset includes the following files:

- `L4_ts_??.rec`: tomograms (downsampled to size 512x512x256)
- `10987_microtubule_imgs.txt`: image file to use as input to the exploration and refinement modules
- `10987_microtubule_coordinates.txt`: coordinates for training the refinement module

Assuming you have gone through the first part of the tutorial, now move the `*.txt` files to the `data/` folder. Go to the folder where `main.py` and `test.py` are located, create another folder named `sample_microtubule_data/` and move the `*.rec` files there. The resulting directory structure should look like this:

```
├── data
│   ├── 10987_microtubule_imgs.txt
│   ├── 10987_microtubule_coordinates.txt
├── sample_microtubule_data
│   ├── L4_ts_??.rec
├── main.py
├── test.py
```

#### Cellular content exploration (3D Mode)

This part is similar to the first case (here since we are only using tomograms, we use 3D mode for the first module), reproduced here for completeness:

```
python simsiam_main.py simsiam3d --num_epochs 20 --exp_id fib_test --bbox 36 --dataset simsiam3d --arch simsiam2d_18 --lr 1e-3 --train_img_txt 10987_microtubule_imgs.txt --batch_size 256 --val_intervals 20 --save_all --gauss 0.8 --dog 3,5
```

Once trained, we map tomograms/tilt-series into embeddings by running:

```
python simsiam_test_hm_3d.py simsiam3d --exp_id fib_test --bbox 36 --dataset simsiam3d --arch simsiam2d_18 --test_img_txt 10987_microtubule_imgs.txt --load_model exp/simsiam3d/fib_test/model_20.pth --gauss 0.8 --dog 3,5
```

In the folder `exp/simsiam3d/fib_test/`, you will find the file `all_output_info.npz` containing the embeddings, corresponding coordinates, original cropped patches from tomograms, and the names of corresponding tomograms.

##### 2D visualization
Run the following command:
```
python plot_2d.py --input exp/simsiam3d/fib_test/all_output_info.npz --n_cluster 48 --num_neighbor 40 --mode umap --path exp/simsiam3d/fib_test/ --min_dist_vis 1.3e-3
```

##### 3D visualization
Run the following command:

```
python visualize_3dhm.py --input exp/simsiam3d/fib_test/all_output_info.npz --color exp/simsiam3d/fib_test/all_colors.npy --dir_simsiam exp/simsiam3d/fib_test/ --rec_dir sample_microtubule_data/
```

##### 3D interactive session

To launch, run:

```
python phoenix_visualization.py --input exp/simsiam3d/fib_test/interactive_info_parquet.gzip
```

To save the downloaded parquet files from the interactive session in `txt` format, run:

```
python interactive_to_training_coords.py --input path_to_all_parquet_files --output training_coordinates.txt
```


#### Refined particle localization
To train the model for refined particle localization, run:

```
python main.py semi --down_ratio 2 --num_epochs 10 --bbox 12 --contrastive --exp_id fib_test --dataset semi --arch unet_5 --save_all --debug 4 --val_interval 1 --thresh 0.3 --cr_weight 1.0 --temp 0.07 --tau 0.01 --lr 1e-4 --train_img_txt 10987_microtubule_imgs.txt --train_coord_txt 10987_microtubule_coordinates.txt --val_img_txt 10987_microtubule_imgs.txt --val_coord_txt 10987_microtubule_coordinates.txt --K 550 --compress --gauss 1 --order xzy --last_k 5 --fiber
```

For inference, run:

```
python test.py semi --arch unet_5 --dataset semi --exp_id fib_test --load_model exp/semi/fib_test/model_20.pth --down_ratio 2 --K 550 --order xzy --out_thresh 0.205 --test_img_txt 10987_microtubule_imgs.txt --compress --gauss 1 --cutoff_z 10 --out_id microtubule_out --last_k 5 --fiber --curvature_cutoff 0.03 --nms 3
```

Here we are using the same `.txt` image file for both training and testing. In actual applications, tomograms in training/testing files should be different and the testing file should include all `.rec` files in the dataset.

The final coordinates will be saved into `exp/semi/fib_test/microtubule_out/*.txt`.