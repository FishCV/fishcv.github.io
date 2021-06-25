# Finding Nemo: Automated Detection and Classification of Red Roman in Unconstrained Underwater Environments Using Mask R-CNN

[![CC BY 4.0][cc-by-shield]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

Still used today, the most widely used method of analysis for marine ecological data
is manual review by trained human experts. The majority of analysis of this data is
concerned with extracting information on the abundance, distribution and behaviour
of the studied marine organism(s). This task can be broken down into four sub-tasks,
frequently performed on a target species: (i) locate, (ii) identify, (iii) count, and (iv)
track. This research proposes an object detection and tracking algorithm for red roman (_Chrysoblephus laticeps_) in Cape Town, South Africa. The model is capable of automating all four sub-tasks above, at a test accuracy of mAP<sub>50</sub> = 81.45% on previously unseen footage. This research serves as a proof-of-concept that machine learning based methods of video analysis of marine data can replace or at least supplement human analysis.

[Mask R-CNN](https://arxiv.org/abs/1703.06870)
[matterport](https://github.com/matterport/Mask_RCNN)
[centroid object tracking](https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/)



[![Red Roman Object Tracking](assets/roman_tracking_sample.gif)](https://www.youtube.com/watch?v=28aIeKxBsrY)

## Background

  This repository 

