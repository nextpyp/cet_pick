### Quick tutorial on globular shaped data
Here is a quick tutorial on the sample globular shaped data [here](https://drive.google.com/drive/folders/1roME4QnAAam1q0D8I53WWGbtAS-D80jk?usp=drive_link). Sample data includes the following files:

- tiltx.rec: 3D downsampled reconstructed tomograms with size 512x512x256
- tiltx.ali: aligned downsampled tilt series with size 512x512x41
- tiltx.tlt: tilt angles 
- 2023-10-08_19-44-41.parquet: output from interactive session that includes selected coordinates for refinement module training
- sample_train_explore_img.txt: train image txt for exploration module and refinement module 
- training_coordinates.txt train coordinates txt for refinement module, converted from parquet file above. 

Once downloaded, go to the same folder as `main.py` and `test.py`. Create a folder for `.rec`, `.ail` and `.tlt` files and name it as `sample_data`. Create another folder and name it as `data` for `.txt` files. 

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
python simsiam_main.py simsiam2d3d --num_epochs 300 --exp_id test_sample --bbox 36 --dataset simsiam2d3d --arch simsiam2d3d_18 --lr 1e-3 --train_img_txt sample_train_explore_img.txt --batch_size 256 --val_intervals 20 --save_all --gauss 0.8 --dog 3,5

```

Outputs include logged loss for each epoch, trained model at each 20 intervals and logged opts. 

Once trained, to map tomograms/tilt series into embeddings, run:

```
python simsiam_test_hm_2d3d.py simsiam2d3d --exp_id test_sample --bbox 36 --dataset simsiam2d3d --arch simsiam2d3d_18 --test_img_txt sample_train_explore_img.txt --load_model exp/simsiam2d3d/test_sample/model_300.pth --gauss 0.8 --dog 3,5
```

Go to `exp/simsiam2d3d/test_sample/`, output is a npz file - `all_output_info.npz` that contains embeddings, corresponding coordinates, original cropped patches from tomogram, names of corresponding tomogram.

[![Output snapshot]][Output snapshot]

  [--dirtyreload]: https://www.mkdocs.org/about/release-notes/#support-for-dirty-builds-990
  [live preview]: http://localhost:8000
  [Output snapshot]: assets/outputs_sample.jpg


- 2D visualization generation:

```
python plot_2d.py --input exp/simsiam2d3d/test_sample/all_output_info.npz --n_cluster 48 --num_neighbor 40 --mode umap --path exp/simsiam2d3d/test_sample/ --min_dist_vis 1.3e-3 

```

- 3D tomogram visualization:

```
python visualize_3dhm.py --input exp/simsiam2d3d/test_sample/all_output_info.npz --color exp/simsiam2d3d/test_sample/all_colors.npy --dir_simsiam exp/simsiam2d3d/test_sample/ --rec_dir sample_data/

```

- 3D interactive session:

To launch, run:

```
python phoenix_visualization.py --input exp/simsiam2d3d/test_sample/interactive_info_parquet.gzip

```

To convert downloaded parquet files from interactive sessions, run:

```
python interactive_to_training_coords.py --input path_to_all_parquet_files --output training_coordinates.txt 

```
Note: `--input` should be the path that contains all parquet files not a single parquet file.

The obtained `training_coordinates.txt` should be the same as downloaded `training_coordinates.txt`. 

#### Refined particle localization
To train model for refined particle localization, run:

```
python main.py semi --down_ratio 2 --num_epochs 10 --bbox 16 --exp_id sample_refinement --dataset semi --arch unet_4 --save_all --debug 4 --val_interval 1 --thresh 0.85 --cr_weight 0.1 --temp 0.07 --tau 0.01 --lr 5e-4 --train_img_txt sample_train_explore_img.txt --train_coord_txt training_coordinates.txt --val_img_txt sample_train_explore_img.txt --val_coord_txt training_coordinates.txt --K 900 --compress --order xzy --gauss 0.8 --contrastive --last_k 3

```

To run inference using trained model, run:

```
python test.py semi --arch unet_4 --dataset semi --exp_id sample_refinement --load_model exp/semi/sample_refinement/model_4.pth --down_ratio 2 --K 900 --ord xzy --out_thresh 0.25 --test_img_txt test_img.txt --compress --gauss 0.8 --out_id all_out

```

Please find all coordinates output in `exp/semi/sample_refinement/all_out/*.txt`




