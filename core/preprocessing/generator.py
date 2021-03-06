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
# @Time    : 11/8/2018 4:54 PM
# @Author  : Firmin.Sun (fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-
import numpy as np
import random
import threading
import warnings

import keras

from core.preprocessing.utils import bbox_iou
from core.preprocessing.utils import BoundBox
from core.utils.image import (
    resize_image,
    preprocess_image)
from core.utils.image_transform import imageTransfor


class Generator(object):
    def __init__(
        self,
        transform_generator = None,
        batch_size=1,
        group_method='random',  # one of 'none', 'random', 'ratio'
        shuffle_groups=True,
        image_min_side=416,
        image_max_side=416,
        cell_size=13,
        anchors=list([2.56,2.97, 2.77,4.63, 3.71,3.76, 3.93,5.22, 5.13,5.62]),
        gt_box_max_buffer = 100,
        transform_parameters=None,
    ):
        self.transform_generator    = transform_generator
        self.batch_size             = int(batch_size)
        self.group_method           = group_method
        self.shuffle_groups         = shuffle_groups
        self.image_min_side         = image_min_side
        self.image_max_side         = image_max_side
        self.cell_size              = cell_size
        self.transform_parameters   = transform_parameters


        self.anchors                = anchors
        self.num_boxes              = len(self.anchors)//2
        self.anchor_boxes = [BoundBox(0, 0, self.anchors[2*i], self.anchors[2*i+1]) for i in range(len(self.anchors)//2)]

        self.gt_box_max_buffer = gt_box_max_buffer
        self.group_index = 0
        self.lock = threading.Lock()

        self.group_images()

    def get_anchors(self):
        return self.anchors

    def get_batch_size(self):
        return self.batch_size

    def size(self):
        raise NotImplementedError('size method not implemented')

    def num_classes(self):
        raise NotImplementedError('num_classes method not implemented')

    def name_to_label(self, name):
        raise NotImplementedError('name_to_label method not implemented')

    def label_to_name(self, label):
        raise NotImplementedError('label_to_name method not implemented')

    def image_aspect_ratio(self, image_index):
        raise NotImplementedError('image_aspect_ratio method not implemented')

    def load_image(self, image_index):
        raise NotImplementedError('load_image method not implemented')

    def load_annotations(self, image_index):
        raise NotImplementedError('load_annotations method not implemented')

    def load_annotations_group(self, group):
        return [self.load_annotations(image_index) for image_index in group]

    def filter_annotations(self, image_group, annotations_group, group):
        # test all annotations
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            assert(isinstance(annotations, np.ndarray)), '\'load_annotations\' should return a list of numpy arrays, received: {}'.format(type(annotations))

            # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
            invalid_indices = np.where(
                (annotations[:, 2] <= annotations[:, 0]) |
                (annotations[:, 3] <= annotations[:, 1]) |
                (annotations[:, 0] < 0) |
                (annotations[:, 1] < 0) |
                (annotations[:, 2] > image.shape[1]) |
                (annotations[:, 3] > image.shape[0])
            )[0]

            # delete invalid indices
            if len(invalid_indices):
                warnings.warn('Image with id {} (shape {}) contains the following invalid boxes: {}.'.format(
                    group[index],
                    image.shape,
                    [annotations[invalid_index, :] for invalid_index in invalid_indices]
                ))
                annotations_group[index] = np.delete(annotations, invalid_indices, axis=0)

        return image_group, annotations_group

    def load_image_group(self, group):
        return [self.load_image(image_index) for image_index in group]

    def random_transform_group_entry(self, image, annotations):
        # randomly transform both image and annotations
        if self.transform_generator:
            image, annotations = imageTransfor(self.transform_generator)([image, annotations])

        return image, annotations

    def resize_image(self, image):
        return resize_image(image, min_side=self.image_min_side, max_side=self.image_max_side)

    def preprocess_image(self, image):
        return preprocess_image(image)

    def preprocess_group_entry(self, image, annotations):
        # preprocess the image
        image = self.preprocess_image(image)

        # randomly transform image and annotations
        image, annotations = self.random_transform_group_entry(image, annotations)

        # resize image
        image, image_scale = self.resize_image(image)

        # apply resizing to annotations too
        annotations[:, 0:4:2] *= image_scale[1]
        annotations[:, 1:4:2] *= image_scale[0]

        return image, annotations

    def preprocess_group(self, image_group, annotations_group):
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            # preprocess a single group entry
            image, annotations = self.preprocess_group_entry(image, annotations)

            # copy processed data back to group
            image_group[index]       = image
            annotations_group[index] = annotations

        return image_group, annotations_group

    def group_images(self):
        # determine the order of the images
        order = list(range(self.size()))
        if self.group_method == 'random':
            random.shuffle(order)
        elif self.group_method == 'ratio':
            order.sort(key=lambda x: self.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]

    def compute_inputs(self, image_group):
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        # construct an image batch object
        image_batch = np.zeros((self.batch_size,) + max_shape, dtype=keras.backend.floatx())

        # c opy all images to the upper left part of the image batch object
        for image_index, image in enumerate(image_group):
            image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

        return image_batch

    def compute_targets(self, image_group, annotations_group):
        # same size for all batch image
        h, w, _ = image_group[0].shape

        gt_batch = np.zeros((self.batch_size, 1, 1, 1, self.gt_box_max_buffer, 4))
        targets = np.zeros(
            (self.batch_size, self.cell_size, self.cell_size, self.num_boxes, 4 + 1 + self.num_classes()))
        for annotation_index, annotation in enumerate(annotations_group):
            label = np.zeros((self.cell_size, self.cell_size, self.num_boxes, 4 + 1 + self.num_classes()))

            box_chw = np.stack([
                (annotation[:, 0] + annotation[:, 2]) / 2,
                (annotation[:, 1] + annotation[:, 3]) / 2,
                (annotation[:, 2] - annotation[:, 0]),
                (annotation[:, 3] - annotation[:, 1])], axis=1)
            # cls_ind = [self.label_to_name(label) for label in annotation[:, 4]]

            box_chw[:, 0] = box_chw[:, 0] / w * self.cell_size
            box_chw[:, 1] = box_chw[:, 1] / h * self.cell_size
            box_chw[:, 2] = box_chw[:, 2] / w * self.cell_size
            box_chw[:, 3] = box_chw[:, 3] / h * self.cell_size

            cls_ind = np.int32(annotation[:, 4])

            for index, (class_index, box) in enumerate(zip(cls_ind, box_chw)):
                best_anchor = -1
                max_iou = -1

                shifted_box = BoundBox(0,
                                       0,
                                       box[2],
                                       box[3])
                for i in range(self.num_boxes):
                    anchor_box = self.anchor_boxes[i]
                    iou = bbox_iou(shifted_box, anchor_box)

                    if max_iou < iou:
                        best_anchor = i
                        max_iou = iou

                # x_ind， y_ind
                x_ind = np.int32(box[0])
                y_ind = np.int32(box[1])

                # Each grid cell predicts five object
                if label[y_ind, x_ind, best_anchor, 4] == 1:
                    # alreadly existed anchor, skill it
                    continue

                label[y_ind, x_ind, best_anchor, 0:4] = box
                label[y_ind, x_ind, best_anchor, 4] = 1
                label[y_ind, x_ind, best_anchor, 5 + class_index] = 1

                # index must less than self.gt_box_max_buffer
                gt_batch[annotation_index, 0, 0, 0, index] = box
            targets[annotation_index] = label

        return np.asarray(gt_batch), np.asarray(targets)

    def compute_input_output(self, group):
        # load images and annotations
        image_group       = self.load_image_group(group)
        annotations_group = self.load_annotations_group(group)

        # check validity of annotations
        image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        # perform preprocessing steps
        image_group, annotations_group = self.preprocess_group(image_group, annotations_group)

        # compute network inputs
        inputs = self.compute_inputs(image_group)

        # compute network targets
        gt_boxes, targets = self.compute_targets(image_group, annotations_group)

        return [inputs, gt_boxes, targets], None

    def __next__(self):
        return self.next()

    def next(self):
        # advance the group index
        with self.lock:
            if self.group_index == 0 and self.shuffle_groups:
                # shuffle groups at start of epoch
                random.shuffle(self.groups)
            group = self.groups[self.group_index]
            self.group_index = (self.group_index + 1) % len(self.groups)

        return self.compute_input_output(group)