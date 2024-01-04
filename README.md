# Accurate detection of proteins from cryo-ET tomograms using sparse labels

We propose a novel particle detection framework that uses positive-unlabeled learning and exploits the unique properties of 3D tomograms to improve detection performance. Our end-to-end framework is able to identify particles within minutes when trained using a single partially labeled tomogram.

## Python requirements
This code requires:
- Python 3.7
- Pytorch 1.8.1
- CUDA 10.1
- [Anaconda 2020/02](https://www.anaconda.com/distribution/)


## Installation
Please follow the following steps to install:
1. Create an Anaconda/venv (optional)
2. Install Pytorch: ```pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html```
3. Install required packages using ```requirements.txt``` provided:  ```pip install -r requirements.txt```
4. Install CET package and dependencies: ```pip install -e cet_pick```. Note: if the command shows error, try go to one level above: ```cd ..```

## Preparing datasets

### Training set format

The training set should include two files: 1. a txt file with tomogram name and path to tomogram; 2. a txt file with image name and its corresponding x,y,z coordinates.

For 1, the text file should have the following format:

```
image_name   path

tomo1 	path_to_tomo1

tomo2 	path_to_tomo2
...
```

For 2, the text file should have the following format:

```
image_name	x_coord	y_coord	z_coord

tomo1	125	38	60

tomo1	130	45	80
...
````
etc.


### Generate training set

First, create a ```train/``` directory containing all tomograms and corresponding coordinates for training. Each tomogram should have its own coordinate file: e.g., ```train_1_img.rec``` and ```train_1_img_coord.txt```.

To produce coordinates for training, manual picking can be performed on selected tomograms, for example, using [IMOD](https://bio3d.colorado.edu/imod/). For a given tomogram, select a subregion and pick around 10\% to 70\% of the particles in that area. After manual annotation, IMOD will save the coordinates as an ```.mod``` file. Converting ```.mod``` files to ```.txt``` files can be done using the [model2point](https://bio3d.colorado.edu/imod/doc/man/model2point.html) command from IMOD. For example:
```
model2point input.mod input.txt
```
Once all the ```.mod``` files have been converted to text files, move all the ```coordinate.txt``` files to the ```train/``` directory.

Depending on the x-y-z order of your input tomograms, the coordinates generated using IMOD may be in different order. The two most common orders are 'x-y-z' and 'x-z-y'. Make sure you get the orders correct.

To generate the training files, run ```generate_train_files.py``` file under the ```utils/``` folder. Two input arguments are required: ```-d/--dir``` tp specify the path to the ```train/``` directory, and ```-o/--out``` to specify the output training file name. The default order for the input coordinates is ```x-z-y```, if you want to specify a different order, add ```-r/--ord```. Valid orders are: ```xyz```, ```xzy```, ```zxy```.  For example:
```
python generate_train_files.py -e .mrc -d train_dir -o train_name -r xyz --compress

```


[-e]: extension of the tomogram file, can be .mrc or .rec
[-d]: path to directory that contains all training data
[-o]: output training name 
[-r]: order for input coordinates of the files obtained using imod `point2model`
[--compress]: whether to combine 2 slices into 1 slice, for the detection task this should always be included

For the detection task, we usually combine 2 slices into 1 slice. After completion, two files will be generated: 1. ```out_train_images.txt```; and 2. ```out_train_coords.txt``` (both in the ```train/``` directory). These two files will serve as input to the program.

Validation files can be generated using the same method. Make sure all training files are in one folder and all validation files are in different folder before running the code above.

Once training files/validation files are generated, for simplicity, first create a folder for all training/validation txt files, and move all related files to the data folder.

## Running

To train the model, please use ```main.py```. To run inference using trained model, please use ```test.py```. For help, do ```python main.py -h``` or ```python test.py -h```.

### Training

A sample command to train using the EMPIAR-10304 dataset is:
```
python main.py semi --down_ratio 2 --num_epochs 600 --bbox 16 --contrastive --exp_id test_reprod --dataset semi --arch unet_4 --debug 4 --val_interval 100 --thresh 0.8 --cr_weight 0.1 --temp 0.07 --tau 0.4 --lr 1e-3 --train_img_txt train_img_new_10304.txt --train_coord_txt 10304_train_coord_30p.txt --val_img_txt 10304_new_test_img_all.txt --val_coord_txt 10304_val_coord_new.txt --test_img_txt 10304_new_test_img_all.txt --test_coord_txt 10304_val_coord_new.txt --compress --gauss

```

Below are the parameters we need to specify:

[--down_ratio]: down ratio for output, we use a ratio of 2. This is the default ratio.

[--num_epochs]: number of training epochs, we used 600. If we have <50 labeled coordinates, we recommend <200 epochs. If we have 50-200 labeled coordinates, we recommend 200 - 600 epochs. Generally, less epochs for less data to prevent overfitting. Larger epoch number is more prone to overfitting, especially when there is small amount of labeled data.

[--bbox]: bounding box size for particle, used to generate Guassian kernel during training.This is dependent upon the size of the particle and the pixel size. For a particle that occupies 20*20 pixels on a slice, we set it to 20. If we use larger bbox value, at same threshold level, more pixels will be considered as positive pixel. This will lead to more positive samples during training. However, if bbox is too large, some positive pixels are not located at/near the center of the particle. This will add more noise to feature representation learning (for contrastive module) and may lead to some false positives. Reduce this parameter if false positive rate is large.

[--contrastive]: whether we want to use contrastive regularization (contrastive module). For better performance, please always turn it on. This module also helps with model convergence. 

[--exp_id]: experiment id you want to save it as.

[--dataset]: use 'semi' during training. for semi-supervised/few shot particle detection during training, please use semi. For semi-supervised/few shot particle detection during evaluation, please use semi_test.

[--arch]: We have two main architecture choices available: ResNet and UNet. Format is "name"_"numOfLayers" - for example, we use unet_4/5 here. For resnet, use "ressmall_18". We recommend using UNet architecture. As stated in our paper, UNet generally performs better than ResNet.

[--debug]: debug mode for visualization. Default is 4. Currently only support mode '4' for easier visualization - output will be saved to 'debug folder' including view of each slice, ground truth heatmap, predicted heatmap, and detection prediction based on heatmap.

[--val_interval]: interval to perform validation

[--thresh]: threshold for soft/hard positives. Default is 0.5. This parameter is also related to `--bbox` parameter above. A higher threshold means less pixels will be considered as positive. Therefore, a higher threshold will result in less false negatives and a lower threshold tends to pick more particles (but may lead to some false positives). If false positive rate is too high, decrease the value of this parameter. Generally, we recommend using 0.8. If we have less labels, we recommend lower threshold, such as 0.3. If we set thresh to 1, this means only center coordinates are considered as particles. If we set thresh to 0.8, this means center coordinates and pixels within the radius of 2-3 (depends on bbox) are considered as positive as well.

[--cr_weight]: weight for contrastive regularization module (0 to 1). Default is 0.1. This parameter determines the weight of contrastive loss in the overall loss function (sum of heatmap loss and contrastive loss). If we have very few labeled particles (<50), we recommend using larger values such as 0.5 to 1.0. If we have more labeled particles (>50), we recommend using 0.1. Please refer to our paper for more discussions.

[--temp]: inforNCE temperature. Default is 0.07. For extremely low SNR datasets, we recommend using 0.02. For datasets with relative SNR, we recommend using 0.07. Rule of thumb is lower value for lower SNR datasets and higher value for higher SNR datasets. For more discussion, please refer to our paper and simCLR paper.

[--tau]: class prior probability (0 to 1). Default is 0.1.  This parameter reflects the approximate positive voxel percentage in a tomogram. For densely populated dataset, more voxels are positive voxels that contain particles, we recommend higher values such as 0.7. For sparse dataset, we recommend lower values such as 0.1.

[--lr]: learning rate. Default is 0.001.

[--K]: max number of output objects. Default is 200. This determines the upper limit of the number of particles the algorithm detects. If 200, the algorithm will detect at most 200 particles in a tomogram.

[--gauss]: whether to apply gaussian filter as preprocessing step for tomogram. We recommend on for low SNR data.

[--compress]: whether to combine 2 slice into 1 slice during reading of the dataset. We recommend on for particle detection task.

[--order]: input order for reconstructed tomogram. can be 'xyz', 'xzy' or 'zxy'. This is dependent on the order of z slice.

#### What are the important hyperparameters?

When training from scratch, the most important hyperparameters to adjusts are `--bbox` (this should be based on the particle size directly), `--tau` (this should be based on particle distribution, whether it's densely populated or not), `--K` (this should be based the approximate number of particles in each tomogram). Intermediate results can be viewed when debug mode is 4. If results contain too many false positive, we recommend adjusting hyperparameters `--thresh`. If experiencing overfitting, please decrease the number of epochs for trianing `--num_epochs`.

More description of arguments are in ```opt.py``` file.  Please refer to the paper for detailed parameter selection.

All outputs will be saved to ```exp/exp_id``` folder. Within folder, there will be: 1. ```opt.txt``` which saves all the option you used. 2. ``` debug``` folder, which saves all validation output. 3. ```model_xxx.pth``` model checkpoint, the final model weights will be ```modelxxx_last_contrastive.pth```. 4. A directory with specific training/validation loss info for each run.

### Testing

Once training is finished (this usually takes around 5-6 minutes), we can use the trained model for testing.

A sample testing command to evaluate/produce overall output using a trained model:
```
python test.py semi --arch unet_4 --dataset semi_test --exp_id test_reprod --load_model exp/semi/test_reprod/model_last_contrastive.pth --down_ratio 2 --contrastive --K 1300 --test_img_txt 10304_new_test_img_all.txt --test_coord_txt 10304_val_coord_new.txt

```
[--dataset]: dataset loader for testing, we use semi_test during testing

[--load_model]: path to trained model

[--exp_id]: experiment id you want to save it as

[--K]: max number of output objects. This determines the upper limit of the number of particles the algorithm detects.

[--arch]: the architecture you want to use, it needs to be the same as the architecture you used for training

[--test_img_txt]: name of the file that contains all test images

[--test_coord_txt]: coord file for all test images, the txt file can be empty

[--out_thresh]: output confidence threshold for particles. Detections lower than this threshold will be discarded.

[--with_score]: whether to have score column in output text files

[--out_id]: directory name for all evaluation outputs.

Make sure you have `semi` as this is the task name and is a required parameter. Output will be saved to  ```exp/semi/exp_id/output_xxx```  folder, there will be one `.mrc` file for heatmap and one `.txt` file for each input tomogram with detected coordinates.

### Per-slice visualization

Images showing per-slice detection results are included in the ```visualization/``` folder.

### Demo
We will add code for demo purpose later.
