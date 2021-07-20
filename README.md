# Finding Nemo: Automated Detection and Classification of Red Roman in Unconstrained Underwater Environments Using Mask R-CNN

[![CC BY 4.0][cc-by-shield]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

Still used today, the most widely used method of analysis for marine ecological data is manual review by trained human experts. The majority of analysis of this data is concerned with extracting information on the abundance, distribution and behaviour of the studied marine organism(s). This task can be broken down into four sub-tasks, frequently performed on a target species: (_i_) locate, (_ii_) identify, (_iii_) count, and (_iv_) track. **This research proposes an object detection and tracking algorithm for red roman (_Chrysoblephus laticeps_) in Cape Town, South Africa.** The model is capable of automating all four sub-tasks above at a test accuracy of mAP<sub>50</sub> = 81.45% on previously unseen footage. This research serves as a proof-of-concept that machine learning based methods of video analysis of marine data can replace or at least supplement human analysis.

[![Red Roman Object Tracking](assets/red_roman_tracking_sample.gif)](https://www.youtube.com/watch?v=28aIeKxBsrY)

## Installation

The red roman model relies on the [matterport implementation](https://github.com/matterport/Mask_RCNN) of [Mask R-CNN](https://arxiv.org/abs/1703.06870) (requires "installation"). The model combines the matterport library with a generic [centroid object tracking](https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/) method (does not require installation).

Note: The files provided in this repository (dataset, weights, model) should be copied into the [Mask_RCNN](https://github.com/matterport/Mask_RCNN) directory created in step 1.

1. Clone the [Matterport Mask R-CNN repository](https://github.com/matterport/Mask_RCNN) and follow the installation instructions. You may be required to install additional software.
2. Download the red roman dataset splits `train`, `test` and `val`, available in this respository [here](https://github.com/FishCV/fishcv.github.io/tree/main/dataset/via). These datasets should be placed in the path: `../Mask_RCNN/datasets/redroman/`. (This will be inside the local matterport directory created in 1.)
3. For inference, download `mask_rcnn_redroman.h5` from [here](https://drive.google.com/drive/folders/1ltqEYAN5qIrL1B_SHkg6SYGlIRaUX7-o?usp=sharing). Save in path: `../Mask_RCNN/weights/redroman/`.
4. Download `redroman.py` python script and `pyimagesearch` lib (for training and inference) and `mAP.ipynb` (for model evaluation) from [here](https://github.com/FishCV/fishcv.github.io/tree/main/model). These should be placed in the path: `../Mask_RCNN/samples/redroman/`
5. Setup a Python environment (an Anaconda virtual environment is recommended). Please use the environment file [here](https://github.com/FishCV/fishcv.github.io/tree/main/model) for this purpose.
6. From the console, `cd` into `../Mask_RCNN/samples/redroman/` to execute sample code (see below) for training/ inference.

## Training

1. Train a new model starting from pre-trained COCO weights  
```
python redroman.py train --dataset=..\..\datasets\redroman\ --weights=coco
```

2. Resume training a model from last trained weights (or select specific weights file)  
```
redroman.py train --dataset=..\..\datasets\redroman\ --weights=last
```
or
``` 
python redroman.py train --dataset=..\..\datasets\redroman\ --weights=..\..\weights\redroman\mask_rcnn_redroman.h5
```

## Inference

1.  **(Image)** Detection (bbox, mask, centroid)
```
python redroman.py detect --weights=..\..\weights\redroman\mask_rcnn_redroman.h5 --image=..\..\datasets\inference\redroman\images
```
(Note: Inference is performed on a folder of images. If you'd like to run the model on a single image, make a separate folder containing this single image.)  

2. **(Video)** Detection (bbox, mask, centroid)
```
python redroman.py detect --weights=..\..\weights\redroman\mask_rcnn_redroman.h5 --video=..\..\datasets\inference\redroman\video\sample_video.MP4
```

3. **(Video with centroid tracking)** Detection (bbox, mask, centroid)
```
python redroman.py detect --weights=..\..\weights\redroman\mask_rcnn_redroman.h5 --video=..\..\datasets\inference\redroman\video\sample_video.MP4 --tracking Y
```

## Model Parameters

There are a number of model parameters that can be tuned during training (see below for some examples). Please see the [Matterport Wiki](https://github.com/matterport/Mask_RCNN/wiki) for help on this.

```python
class FishConfig(Config):
    """
    Configuration for training on your own dataset (red roman dataset).
    Derives from the base Config class and overrides some values.
    """

    # [1]
    BACKBONE = "resnet50"

    # [2]
    IMAGE_MIN_DIM = 460; IMAGE_MAX_DIM = 576

    # [3]
    GPU_COUNT = 1; IMAGES_PER_GPU = 1

    # [4]
    TRAIN_ROIS_PER_IMAGE = 100
    STEPS_PER_EPOCH = 300
    VALIDATION_STEPS = 100
    MAX_GT_INSTANCES = 10
    
```

You can also override some parameters that will apply only when the model is set to inference.

```python
class FishInferenceConfig(FishConfig):
    """
    Configuration for inference on test data (red roman dataset).
    Derives from the FishConfig class (and by extension, Base Config class) and overrides some values.
    """
    
    # [1] e.g. You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7    
```