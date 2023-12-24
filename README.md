# Learning Multi-Scale Photo Exposure Correction

*[Mahmoud Afifi](https://sites.google.com/view/mafifi)*<sup>1,2</sup>, 
*[Konstantinos G. Derpanis](https://www.cs.ryerson.ca/kosta/)*<sup>1</sup>, 
*[Björn Ommer](https://hci.iwr.uni-heidelberg.de/Staff/bommer)*<sup>3</sup>, 
and *[Michael S. Brown](http://www.cse.yorku.ca/~mbrown/)*<sup>1</sup>

<sup>1</sup>Samsung AI Center (SAIC) - Toronto &nbsp;&nbsp;  <sup>2</sup>York University  &nbsp;&nbsp;  <sup>3</sup>Heidelberg University


![teaser](https://user-images.githubusercontent.com/37669469/112195503-69dda280-8be0-11eb-9957-c0f72f18f4d4.jpg)

Project page of the paper [Learning Multi-Scale Photo Exposure Correction.](https://arxiv.org/pdf/2003.11596.pdf) Mahmoud Afifi, Konstantinos G. Derpanis, Björn Ommer, and Michael S. Brown. In CVPR, 2021. If you use this code or our dataset, please cite our paper:
```
@inproceedings{afifi2021learning,
  title={Learning Multi-Scale Photo Exposure Correction},
  author={Afifi, Mahmoud and Derpanis, Konstantinos G, and Ommer, Bj{\"o}rn and Brown, Michael S},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```



## Dataset
![dataset_figure](https://user-images.githubusercontent.com/37669469/112195940-e7091780-8be0-11eb-869d-8a40675beb3a.jpg)

Download our dataset from the following links:
[Training](https://ln2.sync.com/dl/141f68cf0/mrt3jtm9-ywbdrvtw-avba76t4-w6fw8fzj) ([mirror](https://drive.google.com/file/d/1YtsTeUThgD2tzF6RDwQ7Ol9VTSwqFHc_/view?usp=sharing)) | [Validation](https://ln2.sync.com/dl/49a6738c0/3m3imxpe-w6eqiczn-vripaqcf-jpswtcfr) ([mirror](https://drive.google.com/file/d/1k_L2I63NpjDbhFFfHinwF7_2KjTIiipk/view?usp=sharing)) | [Testing](https://ln2.sync.com/dl/098a6c5e0/cienw23w-usca2rgh-u5fxikex-q7vydzkp) ([mirror](https://drive.google.com/file/d/1uxiD6-DOeLnLyI_51DUHMRxORHmUWtgz/view?usp=sharing)) | [Our results](https://ln2.sync.com/dl/36fe0c4e0/d5buy3rd-gkhbcv78-qjj7c2kx-j25u9qk9)


As the dataset was originally rendered using raw images taken from the MIT-Adobe FiveK dataset, our dataset follows the original license of the MIT-Adobe FiveK dataset.



## Code



### Prerequisite
1. Matlab 2019b or higher (tested on Matlab 2019b)
2. Deep Learning Toolbox

### Get Started
Run `install_.m`


#### Demos:

1. Run `demo_single_image.m` or `demo_image_directory.m` to process a single image or image directory, respectively. If you run the demo_single_image.m, it should save the result in `../result_images` and output the following figure:


![exposure_demo](https://user-images.githubusercontent.com/37669469/126022585-8eeef4d3-e020-48a7-912d-14dea2b20399.png)


2. Run `demo_GUI.m` for a gui demo.
<p align="center">
  <img width = 65% src="https://user-images.githubusercontent.com/37669469/126023911-1e00bab8-1ece-4b8a-bf81-08ed443c87e3.gif">
</p>


We provide a way to interactively control the output results by scaling each layer of the Laplacian pyramid before feeding them to the network. This can be controlled from the `S` variable in `demo_single_image.m` or `demo_image_directory.m` or from the GUI demo. Each scale factor in the `S` vector is multiplied by the corresponding pyramid level. 

Additional post-processing options include fusion and histogram adjustment that can be turned on using the `fusion` and `pp` variables, respectively in `demo_single_image.m` or `demo_image_directory.m`. These options are also available in the  GUI demo. Note that none of the `fusion` and `pp` options was used in producing our results in the paper, but they can improve the quality of results in some cases as shown below. 



![fusion](https://user-images.githubusercontent.com/37669469/126023779-f6ae6349-795f-4af1-97af-829385db9933.gif)


#### Training:
We train our model end-to-end to minimize: reconstruction loss, Laplacian pyramid loss, and adversarial loss.  We trained our model on patches randomly
extracted from training images with different dimensions. We first train on patches of size 128×128 pixels. Next, we continue training on 256×256 patches, followed by training on 512×512 patches. 

![exposure_training](https://user-images.githubusercontent.com/37669469/126025119-ce29fe07-bcf0-4384-8a16-424adff4933d.jpg)



Before starting, run `src/patches_extraction.m` to extract random patches with different dimensions -- adjust training/validation image directories before running the code. In the given code, the dataset is supposed to be located in the `exposure_dataset` folder in the root directory. The `exposure_dataset` should include the following directories:
```
- exposure_dataset/
         training/
              INPUT_IMAGES/
              GT_IMAGES/
         validation/
              INPUT_IMAGES/
              GT_IMAGES/
```

The `src/patches_extraction.m` will create subdirectories with patches extracted from each image and its corresponding ground-truth at different resolutions as shown below. 


![patch_extraction](https://user-images.githubusercontent.com/37669469/126023800-b09129bd-a92b-4233-8e23-f20c887ffebb.jpg)




After extracting the training patches, run `main_training.m` to start training -- adjust training/validation image directories before running the code. All training options are available in the `main_training.m`. 


## Results

![exposure_results_1](https://user-images.githubusercontent.com/37669469/126025090-1162f795-8750-4061-81cd-23cda863721a.jpg)
![exposure_results_2](https://user-images.githubusercontent.com/37669469/126025091-821440b4-71d9-4327-a393-8b80c1047517.jpg)
![exposure_results_3](https://user-images.githubusercontent.com/37669469/126025093-1c401b86-bc08-4735-a723-9da0fea6f7b1.jpg)

This software is provided for research purposes only and CANNOT be used for commercial purposes. 


Maintainer: Mahmoud Afifi (m.3afifi@gmail.com)


## Related Research Projects
- [Deep White-Balance Editing](https://github.com/mahmoudnafifi/Deep_White_Balance): A deep learning multi-task framework for white-balance editing (CVPR 2020).


