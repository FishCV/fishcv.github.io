# Finding Nemo: Automated Detection and Classification of Red Roman in Unconstrained Underwater Environments Using Mask R-CNN

[![CC BY 4.0][cc-by-shield]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

Still used today, the most widely used method of analysis for marine ecological data is manual review by trained human experts. The majority of analysis of this data is concerned with extracting information on the abundance, distribution and behaviour of the studied marine organism(s). This task can be broken down into four sub-tasks, frequently performed on a target species: (i) locate, (ii) identify, (iii) count, and (iv) track. **This research proposes an object detection and tracking algorithm for red roman (_Chrysoblephus laticeps_) in Cape Town, South Africa.** The model is capable of automating all four sub-tasks above at a test accuracy of mAP<sub>50</sub> = 81.45% on previously unseen footage. This research serves as a proof-of-concept that machine learning based methods of video analysis of marine data can replace or at least supplement human analysis.

[![Red Roman Object Tracking](assets/roman_tracking_sample.gif)](https://www.youtube.com/watch?v=28aIeKxBsrY)

## Installation

The red roman model relies on the [matterport implementation](https://github.com/matterport/Mask_RCNN) of [Mask R-CNN](https://arxiv.org/abs/1703.06870) - **requires "installation"**. The model combines the matterport library with a generic [centroid object tracking](https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/) method - **does not require installation**.

1. Clone the [Matterport Mask R-CNN repository](https://github.com/matterport/Mask_RCNN) and follow the installation instructions. You may be required to install additional software.
2. Download the red roman dataset splits `train`, `test` and `val`, available in this respository [here](https://github.com/FishCV/fishcv.github.io/tree/main/dataset/via). These datasets should be placed in the path: `../Mask_RCNN/datasets/redroman/`. (This will be inside the local matterport directory created in 1.)
3. For inference, download `mask_rcnn_redroman.h5` from [here](https://drive.google.com/drive/folders/1ltqEYAN5qIrL1B_SHkg6SYGlIRaUX7-o?usp=sharing). Save in path: `../Mask_RCNN/weights/redroman/`.
4. Download `redroman.py` (for training and inference) and `mAP.ipynb` (for model evaluation) from [here](https://github.com/FishCV/fishcv.github.io/tree/main/model). These should be placed in the path: `../Mask_RCNN/samples/redroman/`

## Training

1. Train a new model starting from pre-trained COCO weights  
`python redroman.py train --dataset=..\..\datasets\redroman\ --weights=coco`  

2. Resume training a model from last trained weights (or select specific weights file)  
`python redroman.py train --dataset=..\..\datasets\redroman\ --weights=last`  
or  
`python redroman.py train --dataset=..\..\datasets\redroman\ --weights=..\..\weights\redroman\mask_rcnn_redroman.h5`  


## Inference

1.  **(Image)** Detection (bbox, mask, centroid)  
`python fish.py detect --weights=..\..\weights\redroman\mask_rcnn_redroman.h5 --image=..\..\datasets\inference\redroman\images`  
(Note: Inference is performed on a folder of images. If you'd like to run the model on a single image, make a separate folder containing this single image.)  

2. **(Video)** Detection (bbox, mask, centroid)  
`python fish.py detect --weights=..\..\weights\redroman\mask_rcnn_redroman.h5 --video=--image=..\..\datasets\inference\redroman\video\sample_video.MP4`  

3. **(Video with centroid tracking)** Detection (bbox, mask, centroid)  
`python fish.py detect --weights=..\..\weights\redroman\mask_rcnn_redroman.h5 --video=--image=..\..\datasets\inference\redroman\video\sample_video.MP4 --tracking Y`  