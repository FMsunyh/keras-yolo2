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
# @Time    : 12/5/2018 1:37 PM
# @Author  : Firmin.Sun (fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-

import keras
import core
from core.models import Yolo2


def create_yolo2(inputs, training=True, num_classes=20, weights=None, *args, **kwargs):
    if training:
        image, gt_boxes, targets = inputs
    else:
        image = inputs

    loss, output = Yolo2(training=training, num_classes=num_classes, weights=weights)([image, gt_boxes, targets])

    model = keras.models.Model(inputs=inputs, outputs=[loss,output])
    return model

