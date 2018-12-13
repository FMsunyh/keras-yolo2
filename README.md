# keras-yolo2
## Introduction
This repo contains the implementation of YOLOv2 in Keras with Tensorflow backend.

For details about YOLO and YOLOv2 please refer to their [project page](https://pjreddie.com/darknet/yolo/) 
and the [paper](https://arxiv.org/abs/1612.08242):
*YOLO9000: Better, Faster, Stronger by Joseph Redmon and Ali Farhadi*.
---

## Requirement
- python 3.5
- keras 2.2.2
- tensorflow 1.12.0

## TODO 
- [x] train model
- [ ] predict model
- [ ] mAP Evaluation
   
## train
```bash
cd tools
python train_yolo2.py pascal 'path_to_data'/VOCdevkit/VOC2007
```