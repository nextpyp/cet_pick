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
python generate_train_files.py -e .mrc -d train_dir -o train_name -r xyz
```
After completion, two files will be generated: 1. ```out_train_images.txt```; and 2. ```out_train_coords.txt``` (both in the ```train/``` directory). These two files will serve as input to the program.

Validation files can be generated using the same method. Make sure all training files are in one folder and all validation files are in different folder before running the code above.

Once training files/validation files are generated, for simplicity, first create a folder for all training/validation txt files, and move all related files to the data folder. 

## Running

To train the model, please use ```main.py```. To run inference using trained model, please use ```test.py```. For help, do ```python main.py -h``` or ```python test.py -h```. 

### Training

A sample command to train using the EMPIAR-10304 dataset is:
```
python main.py semi --down_ratio 2 --num_epochs 600 --bbox 16 --contrastive --exp_id test_reprod --dataset semi --arch unet_4 --debug 4 --val_interval 100 --thresh 0.8 --cr_weight 0.1 --temp 0.07 --tau 0.4 --lr 1e-3 --train_img_txt train_img_new_10304.txt --train_coord_txt 10304_train_coord_30p.txt --val_img_txt 10304_new_test_img_all.txt --val_coord_txt 10304_val_coord_new.txt --test_img_txt 10304_new_test_img_all.txt --test_coord_txt 10304_val_coord_new.txt
```

[--down_ratio]: down ratio for output, we use a ratio of 2

[--num_epochs]: number of training epochs, we used 600

[--bbox]: bounding box size for particle, used to generate Gaussian kernel during training

[--contrastive]: whether we want to use contrastive regularization. For better performance, please turn it on

[--exp_id]: experiment id used to save results

[--dataset]: use ```semi``` for default during training

[--arch]: We have two main architecture choices available: ResNet and UNet. Format is "name"_"numOfLayers" - for example, we use unet_4 here.

[--debug]: debug mode for visualization: currently only support mode 4 for easier visualization - output will be saved to the ```debug/``` folder including images of each slice, ground truth heatmap, predicted heatmap, and detection prediction based on heatmap

[--val_interval]: interval to perform validation

[--thresh]: threshold for soft/hard positives

[--cr_weight]: weight for contrastive regularization module

[--temp]: inforNCE temperature

[--tau]: class prior probability

[--lr]: learning rate, we start with 0.001

Description of other arguments is included in the ```opt.py``` file.

All output will be saved to the ```exp/exp_id``` folder. Within this folder, there will be: 1. the ```opt.txt``` file containing the options used for training; 2. a ```debug/``` folder containing all validation files; 3. ```model_xxx.pth``` intermediate checkpoint files (the final model weights will be saved as ```modelxxx_last_contrastive.pth```); 4. a directory with specific training/validation loss info for each run.

### Testing

Once training is finished (this usually takes around 5-6 minutes), we can use the trained model for testing.

A sample testing command to evaluate/produce overall output using a trained model is:
```
python test.py semi --arch unet_4 --dataset semi_test --exp_id test_reprod --load_model exp/semi/test_reprod/model_last_contrastive.pth --down_ratio 2 --contrastive --K 1300 --test_img_txt 10304_new_test_img_all.txt --test_coord_txt 10304_val_coord_new.txt
```
[--dataset]: dataset loader for testing, we use semi_test during testing

[--load_model]: path to trained model

[--exp_id]: experiment id used to save results

[--K]: maximum number particles detected

[--arch] the architecture you want to use, it needs to be the same as the architecture you used for training

[--test_img_txt] name of the file that contains all test images

[--test_coord_txt] coordinate file for all test images, the txt file can be empty

Make sure you add ```semi``` as this is the task name and is a required parameter. The output will be saved to a folder named ```exp/semi/exp_id/output_xxx```, and there will be one ```.mrc``` file for heatmap and one ```.txt``` file for each input tomogram with detected coordinates.

### Per-slice visualization

Images showing per-slice detection results are included in the ```visualization/``` folder.
