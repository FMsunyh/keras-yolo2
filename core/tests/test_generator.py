#!/usr/bin/python3

"""
Copyright 2018-2019  Firmin.Sun (fmsunyh@gmail.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
# -----------------------------------------------------
# @Time    : 1/9/2019 3:21 PM
# @Author  : Firmin.Sun (fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-

#!/usr/bin/python3
import cv2

from core.utils.visualization import draw_detections, draw_box, draw_caption

"""
Copyright 2018-2019  Firmin.Sun (fmsunyh@gmail.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
# -----------------------------------------------------
# @Time    : 11/9/2018 3:54 PM
# @Author  : Firmin.Sun (fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-
import argparse
import os
import keras
import keras.preprocessing.image
from core.preprocessing import PascalVocGenerator
from core.utils.config import load_setting_cfg
import numpy as np


def create_generators(args):
    # create image data generator objects
    train_image_data_generator = keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0,
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
    )
    valid_image_data_generator = keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0,
    )

    # create a generator for training data
    train_generator = PascalVocGenerator(
        args,
        'trainval',
        batch_size=args.batch_size,
        transform_generator=train_image_data_generator
    )

    # create a generator valid data
    valid_generator = PascalVocGenerator(
        args,
        'test',
        batch_size=args.batch_size,
        transform_generator=valid_image_data_generator
    )

    return train_generator, valid_generator

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')

    parser.add_argument('--pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).', required=False, default=['VOC2007'], type=list)
    parser.add_argument('--root_path', help='Size of the batches.', default= os.path.join(os.path.expanduser('~'), 'keras_yolo2'), type=str)

    parser.add_argument('--args_setting', help='file name of args setting', default='args_setting.cfg', type=str)

    parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi).', default=0, type=int)
    parser.add_argument('--epochs', help='num of the epochs.', default=100, type=int)
    parser.add_argument('--tag', help='filename of the output.', default='voc', type=str)
    parser.add_argument('--classes_path', help='Path to classes directory (ie. /tmp/voc_classes.txt).', default='voc_classes.txt',type=str)

    parser.add_argument('--batch-size', help='Size of the batches.', default=4, type=int)

    parser.add_argument('--save-path', help='Path for saving images with detections.', default=os.path.join(os.path.dirname(__file__), '../../', 'experiments/eval_output'))

    parser.add_argument('--weight_path', help='Path to classes directory (ie. /tmp/voc.h5).', default='', type=str)

    parser.add_argument('--tensorboard-dir', help='Log directory for Tensorboard output', default='./logs')
    parser.add_argument('--no-snapshots', help='Disable saving snapshots.', dest='snapshots', action='store_false')
    parser.add_argument('--snapshot-path',  help='Path to store snapshots of models during training (defaults to \'./snapshots\')',
                        default=os.path.join(os.path.dirname(__file__), '../../snapshots'))
    parser.add_argument('--evaluation', help='', default=True, type=bool)

    # args = parser.parse_args()
    args = check_args(parser.parse_args())
    return args


def check_args(parsed_args):
    # TODO check the args
    # reload parse arguments
    args = load_setting_cfg(parsed_args)

    return args

def main():
    # parse arguments
    args = parse_args()

    train_generator, valid_generator = create_generators(args)

    input ,_= train_generator.next()
    images, gt_boxes, targets = input
    save_path = '/home/syh/keras_yolo2/experiments/test_generator/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(args.batch_size):
        image = images[i]
        box_chw = gt_boxes[i].reshape((-1, 4)) / 13 * 416
        annotations = np.stack([
            (box_chw[:, 0] - box_chw[:, 2] / 2),
            (box_chw[:, 1] - box_chw[:, 3] / 2),
            (box_chw[:, 0] + box_chw[:, 2] / 2),
            (box_chw[:, 1] + box_chw[:, 3] / 2) ], axis=1)

        print(annotations)

        for j in range(len(annotations)):
            c = (0, 255, 0)
            draw_box(image, annotations[j, :], color=c)

            # draw labels
            # caption = (label_to_name(labels[i]) if label_to_name else labels[i]) + ': {0:.2f}'.format(scores[i])
            # draw_caption(image, boxes[i, :], caption)

        cv2.imwrite(os.path.join(save_path, '{}.png'.format(str(i).zfill(4))), image)
    # start training

if __name__ == '__main__':
    main()

    '''
    cd tools
    python train_yolo2.py pascal /home/syh/train_data/VOCdevkit/VOC2007
    '''