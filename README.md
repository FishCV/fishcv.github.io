# Finding Nemo: Automated Detection and Classification of Red Roman in Unconstrained Underwater Environments Using Mask R-CNN

[![CC BY 4.0][cc-by-shield]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

Still used today, the most widely used method of analysis for marine ecological data
is manual review by trained human experts. The majority of analysis of this data is
concerned with extracting information on the abundance, distribution and behaviour
of the studied marine organism(s). This task can be broken down into four sub-tasks,
frequently performed on a target species: (i) locate, (ii) identify, (iii) count, and (iv)
track. **This research proposes an object detection and tracking algorithm for red roman (_Chrysoblephus laticeps_) in Cape Town, South Africa.** The model is capable of automating all four sub-tasks above at a test accuracy of mAP<sub>50</sub> = 81.45% on previously unseen footage. This research serves as a proof-of-concept that machine learning based methods of video analysis of marine data can replace or at least supplement human analysis.

[![Red Roman Object Tracking](assets/roman_tracking_sample.gif)](https://www.youtube.com/watch?v=28aIeKxBsrY)

## Installation

The red roman model relies on the [matterport implementation](https://github.com/matterport/Mask_RCNN) of [Mask R-CNN](https://arxiv.org/abs/1703.06870) - **requires "installation"**. The model combines the matterport library with a generic [centroid object tracking](https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/) method - **does not require installation**.

1. Clone the [Matterport Mask R-CNN repository](https://github.com/matterport/Mask_RCNN) and follow the installation instructions. You may be required to install additional software.
2. Download the red roman dataset splits `train`, `test` and `val`, available in this respository [here](https://github.com/FishCV/fishcv.github.io/tree/main/dataset/via). These datasets should be placed in the path: `../Mask_RCNN/datasets/redroman/`. (This will be inside the local matterport directory you've created in 1.)
3. For inference, download `mask_rcnn_redroman.h5` from [here](https://drive.google.com/drive/folders/1OQQOM0r_9_lTYDCh-D4nksKNZFarVFyL?usp=sharing).
4. Download `redroman.py` (for training and inference) and `mAP.ipynb` (for model evaluation) from [here](https://github.com/FishCV/fishcv.github.io/tree/main/model). These should be placed in the path: `../Mask_RCNN/samples/redroman/`

## Training
* The [`detector`](https://github.com/Jess-cah/measure-pineapple/tree/main/detector) folder contains files related to training and evaluation of Mask R-CNN pineapple detectors. 
* Colab notebook `maskpine_160images_coco_resnet50_aug4_ALL_01.ipynb` shows training of Mask R-CNN using COCO starting weights and ResNet50 backbone, and employing data augmentation techniques. Colab was used in order to make use of GPU facilities.
* Use `init_with = "coco"` to initialise with MS COCO starting weights. Can also use `"imagenet"` to initialise with ImageNet starting weights.
* In `pineapple.py`, `BACKBONE = "resnet50"` means that a ResNet50 CNN backbone will be used. Alternatively, this can be set to `"resnet101"`.
* The dataset `datasets/pineapple160` contains 160 images but these are split as 70/20/10 for training/validation/test.
* Detectors are evaluated using AP@0.5 and AP@[0.50:0.05:0.95], as shown in `inspect_model_maskRCNN.ipynb`.

## Inference
* The [`measurement`](https://github.com/Jess-cah/measure-pineapple/tree/main/measurement) folder contains files related to determining pineapple fruit size from images. 
* Detection and measurement of pineapples from images was done using Juypter notebooks on a local machine, in a conda environment that can be replicated using `pineappleEnvironmment.yml`.
* The `predict_measure_04_batch_noAnnot.ipynb` Jupyter notebook is used for extraction of pineapple diameter and length dimensions from the detected masks.
* The `predict_measure_projectedArea.ipynb` Jupter notebook is used to extract the projected area of detected masks.
* Visualisation and comparison of the detected and hand-measured fruit dimensions are shown in `resnet50_fruitsize_distributions_Model4_FlipLR.ipynb`.

