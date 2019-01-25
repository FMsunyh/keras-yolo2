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
# @Time    : 11/20/2018 3:22 PM
# @Author  : Firmin.Sun (fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-
import keras
import tensorflow as tf
import numpy as np
from keras import backend as K

from core.tools import Debug


class Loss(keras.layers.Layer):
    def __init__(self, num_classes=20, cell_size=13, boxes_per_cell=5, *args, **kwargs):
        self.num_classes = num_classes

        self.cell_size   = cell_size
        self.boxes_per_cell = boxes_per_cell
        self.batch_size = 4
        self.anchors = list([2.56,2.97, 2.77,4.63, 3.71,3.76, 3.93,5.22, 5.13,5.62])
        self.object_scale = 5.0
        self.noobject_scale = 1.0
        self.class_scale = 1.0
        self.reg_scale = 1.0

        super(Loss, self).__init__(*args, **kwargs)

    def classification_loss(self, y_pred,  y_true, detectors_mask):
        # pred_box_class = tf.nn.softmax(y_pred[..., 5:])
        pred_box_class = y_pred
        true_box_class = y_true[..., 5:]

        # num_class = tf.reduce_sum(tf.to_float(detectors_mask > 0.0))
        num_class = tf.reduce_sum(detectors_mask)

        cls_loss = tf.square(true_box_class - pred_box_class)

        # cls_loss = tf.Print(cls_loss, [tf.shape(cls_loss)], 'cls_loss', summarize=100)

        cls_loss = self.class_scale * detectors_mask * cls_loss
        cls_loss = tf.reduce_sum(cls_loss) / tf.maximum(1.0, num_class)

        if Debug:
            cls_loss =  tf.Print(cls_loss, [tf.shape(cls_loss), cls_loss], 'cls_loss', summarize=100)
        return cls_loss

    def regression_loss(self, y_pred,  y_true, detectors_mask):

        pred_box_xy, pred_box_wh = y_pred
        true_box_xy = y_true[..., 0:2]
        true_box_wh = y_true[..., 2:4]

        num_reg = tf.reduce_sum(detectors_mask)

        loss_xy    = self.reg_scale * detectors_mask  * tf.square(true_box_xy-pred_box_xy)
        loss_xy    = tf.reduce_sum(loss_xy) / tf.maximum(1.0, num_reg * 2.0)

        loss_wh    = self.reg_scale * detectors_mask  * tf.square(true_box_wh-pred_box_wh)
        loss_wh    = tf.reduce_sum(loss_wh) / tf.maximum(1.0, num_reg * 2.0)

        reg_loss = loss_xy + loss_wh
        return reg_loss

    def confidence_loss(self, y_pred,  y_true, gt_boxes, detectors_mask):

        pred_box_xy, pred_box_wh, pred_box_conf = y_pred

        true_box_xy = y_true[..., 0:2]  # relative position to the containing cell

        ### adjust w and h
        true_box_wh = y_true[..., 2:4]  # number of cells accross, horizontally and vertically

        ### adjust confidence
        true_wh_half = true_box_wh / 2.
        true_mins = true_box_xy - true_wh_half
        true_maxes = true_box_xy + true_wh_half

        pred_wh_half = pred_box_wh / 2.
        pred_mins = pred_box_xy - pred_wh_half
        pred_maxes = pred_box_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
        pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)

        true_box_conf = iou_scores

        # pred_box_conf = tf.sigmoid(y_pred[..., 4])

        true_xy = gt_boxes[..., 0:2]
        true_wh = gt_boxes[..., 2:4]

        true_wh_half = true_wh / 2.
        true_mins = true_xy - true_wh_half
        true_maxes = true_xy + true_wh_half

        pred_xy = tf.expand_dims(pred_box_xy, 4)
        pred_wh = tf.expand_dims(pred_box_wh, 4)

        pred_wh_half = pred_wh / 2.
        pred_mins = pred_xy - pred_wh_half
        pred_maxes = pred_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)

        best_ious = tf.reduce_max(iou_scores, axis=4)
        best_ious = tf.expand_dims(best_ious, axis=-1)
        # detectors_mask = y_true[..., 4]
        no_object_conf_mask = (1 - detectors_mask) *  tf.to_float(best_ious < 0.6)

        true_box_conf = tf.expand_dims(true_box_conf, axis=-1)
        pred_box_conf = tf.expand_dims(pred_box_conf, axis=-1)

        # true_box_conf = tf.Print(true_box_conf, [tf.shape(true_box_conf)], 'true_box_conf', summarize=10000)
        # pred_box_conf = tf.Print(pred_box_conf, [tf.shape(pred_box_conf)], 'pred_box_conf', summarize=10000)

        object_conf_mask = detectors_mask
        # object_loss = self.object_scale * object_conf_mask  * tf.square(1 - pred_box_conf)
        object_loss = self.object_scale * object_conf_mask  * tf.square(true_box_conf - pred_box_conf)
        noobject_loss =  self.noobject_scale * no_object_conf_mask  * tf.square(-pred_box_conf)

        conf_loss = object_loss + noobject_loss

        num_conf = tf.reduce_sum(detectors_mask)

        conf_loss  = tf.reduce_sum(conf_loss) / tf.maximum(1.0, num_conf * 2.0)

        return conf_loss

    def compute_output(self, y_pred):
        cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(self.cell_size), [self.cell_size]), (1, self.cell_size, self.cell_size, 1, 1)))
        cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))

        cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [self.batch_size, 1, 1, self.boxes_per_cell, 1])

        pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid

        pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(self.anchors, [1, 1, 1, self.boxes_per_cell, 2])
        pred_box_conf = tf.sigmoid(y_pred[..., 4])
        pred_box_class = tf.nn.softmax(y_pred[..., 5:])

        return pred_box_xy, pred_box_wh, pred_box_conf, pred_box_class


    def call(self, inputs):
        '''

        :param inputs:
        predicts: shape(None, 1470)
        labels shape(None, 7, 7, 25)
        :return:
        '''

        y_pred, gt_boxes, y_true = inputs

        if Debug:
            y_pred = tf.Print(y_pred, [tf.shape(y_pred)], '\ny_pred shape:', summarize=100)
            gt_boxes = tf.Print(gt_boxes, [tf.shape(gt_boxes)], 'gt_boxes shape:', summarize=100)
            y_true = tf.Print(y_true, [tf.shape(y_true)], 'y_pred shape:', summarize=100)

        detectors_mask =tf.expand_dims(y_true[..., 4], -1)

        pred_box_xy, pred_box_wh, pred_box_conf, pred_box_class = self.compute_output(y_pred)

        cls_loss  = self.classification_loss(pred_box_class, y_true, detectors_mask)

        reg_loss  = self.regression_loss([pred_box_xy, pred_box_wh], y_true, detectors_mask)

        conf_loss =  self.confidence_loss([pred_box_xy, pred_box_wh, pred_box_conf], y_true, gt_boxes, detectors_mask)
        loss = cls_loss + reg_loss + conf_loss

        self.add_loss(loss)
        return loss

    def _call(self, inputs):
        '''

        :param inputs:
        y_pred: shape(None, 13 * 13 * 5 * (4 + 1 + self.num_classes)) 《== shape(None, 13, 13, 5, (4 + 1 + self.num_classes))
        y_true: shape(None, 13, 13, 5, 4 + 1 + self.num_classes)
        labels shape(None, 1, 1, 1, 100, 4)
        :return:
        '''

        y_pred, gt_boxes, y_true = inputs

        mask_shape = tf.shape(y_true)[:4]

        cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(self.cell_size), [self.cell_size]), (1, self.cell_size, self.cell_size, 1, 1)))
        cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))

        cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [self.batch_size, 1, 1, 5, 1])

        coord_mask = tf.zeros(mask_shape)
        conf_mask = tf.zeros(mask_shape)
        class_mask = tf.zeros(mask_shape)

        
        index_classification = self.cell_size * self.cell_size * self.boxes_per_cell * self.num_classes
        index_confidence = self.cell_size * self.cell_size * self.boxes_per_cell * (self.num_classes+1)

        pred_classes = tf.reshape(y_pred[:, :index_classification], [-1, self.cell_size, self.cell_size, self.boxes_per_cell, self.num_classes])
        pred_confidence = tf.reshape(y_pred[:, index_classification:index_confidence],[-1, self.cell_size, self.cell_size, self.boxes_per_cell, 1])
        pred_boxes = tf.reshape(y_pred[:, index_confidence:], [-1, self.cell_size, self.cell_size, self.boxes_per_cell, 4])

        pred_boxes_xy = tf.sigmoid(pred_boxes[..., :2]) + cell_grid
        pred_boxes_wh = tf.exp(pred_boxes[..., 2:4]) * tf.reshape(self.anchors, [1, 1, 1, 5, 2])
        pred_confidence = tf.sigmoid(pred_confidence)

        gt_boxes_xy = y_true[..., 0:2]
        gt_boxes_wh = y_true[..., 2:4]
        regression = []
        regression_labels = []

        # # regression loss (localization loss) coord_loss
        # coord_loss = self.regression_loss(regression_target, predict_boxes, object_mask)


        iou_predict_truth = self.calc_iou(regression, regression_labels)

        confidence_labels = iou_predict_truth
        # confidence loss
        object_loss, noobject_loss = self.confidence_loss(pred_confidence, confidence_labels, object_mask)

        classification_labels = tf.argmax(y_true[..., 5:], -1)
        # classification loss
        cls_loss = self.classification_loss(classification_labels, pred_classes, response)

        # predict_boxes = tf.reshape(predicts[:, index_confidence:], [-1, self.cell_size, self.cell_size, self.boxes_per_cell, 4])
        # 
        # response = tf.reshape(labels[:, :, :, 0], [-1, self.cell_size, self.cell_size, 1])
        # regression_labels = tf.reshape(labels[:, :, :, 1:5], [-1, self.cell_size, self.cell_size, 1, 4])
        # regression_labels =tf.div(tf.tile(regression_labels, [1, 1, 1, self.boxes_per_cell, 1]), tf.to_float(image_shape[0]))
        # classification_labels = labels[:, :, :, 5:]
        # 
        # offset = np.transpose(np.reshape(np.array(
        #     [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
        #     (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))
        # offset = tf.constant(offset, dtype=tf.float32)
        # offset = tf.reshape(offset, [1, self.cell_size, self.cell_size, self.boxes_per_cell])
        # 
        # regression = tf.stack([(predict_boxes[:, :, :, :, 0] + offset) / self.cell_size,
        #                                (predict_boxes[:, :, :, :, 1] + tf.transpose(offset, (0, 2, 1, 3))) / self.cell_size,
        #                                tf.square(predict_boxes[:, :, :, :, 2]),
        #                                tf.square(predict_boxes[:, :, :, :, 3])])
        # regression = tf.transpose(regression, [1, 2, 3, 4, 0])
        # 
        # iou_predict_truth = self.calc_iou(regression, regression_labels)
        # 
        # # calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        # object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
        # object_mask = tf.cast((iou_predict_truth >= object_mask), tf.float32) * response
        # 
        # regression_target = tf.stack([regression_labels[:, :, :, :, 0] * self.cell_size - offset,
        #                        regression_labels[:, :, :, :, 1] * self.cell_size - tf.transpose(offset, (0, 2, 1, 3)),
        #                        tf.sqrt(regression_labels[:, :, :, :, 2]),
        #                        tf.sqrt(regression_labels[:, :, :, :, 3])])
        # regression_target = tf.transpose(regression_target, [1, 2, 3, 4, 0])
        # 
        # # regression loss (localization loss) coord_loss
        # coord_loss = self.regression_loss(regression_target, predict_boxes, object_mask)
        # 
        # # confidence loss
        # object_loss, noobject_loss = self.confidence_loss(predict_scales, iou_predict_truth, object_mask)
        # 
        # # classification loss
        # cls_loss = self.classification_loss(classification_labels, predict_classes, response)
        # 
        # self.add_loss(cls_loss)
        # self.add_loss(object_loss)
        # self.add_loss(noobject_loss)
        # self.add_loss(coord_loss)
        # 
        # return [coord_loss, object_loss, noobject_loss, cls_loss]

    def calc_iou(self, boxes1, boxes2, scope='iou'):
        """calculate ious
        Args:
          boxes1: 4-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
          boxes2: 1-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
        Return:
          iou: 3-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """
        with tf.variable_scope(scope):
            boxes1 = tf.stack([boxes1[:, :, :, :, 0] - boxes1[:, :, :, :, 2] / 2.0,
                               boxes1[:, :, :, :, 1] - boxes1[:, :, :, :, 3] / 2.0,
                               boxes1[:, :, :, :, 0] + boxes1[:, :, :, :, 2] / 2.0,
                               boxes1[:, :, :, :, 1] + boxes1[:, :, :, :, 3] / 2.0])
            boxes1 = tf.transpose(boxes1, [1, 2, 3, 4, 0])

            boxes2 = tf.stack([boxes2[:, :, :, :, 0] - boxes2[:, :, :, :, 2] / 2.0,
                               boxes2[:, :, :, :, 1] - boxes2[:, :, :, :, 3] / 2.0,
                               boxes2[:, :, :, :, 0] + boxes2[:, :, :, :, 2] / 2.0,
                               boxes2[:, :, :, :, 1] + boxes2[:, :, :, :, 3] / 2.0])
            boxes2 = tf.transpose(boxes2, [1, 2, 3, 4, 0])

            # calculate the left up point & right down point
            lu = tf.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])
            rd = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])

            # intersection
            intersection = tf.maximum(0.0, rd - lu)
            inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]

            # calculate the boxs1 square and boxs2 square
            square1 = (boxes1[:, :, :, :, 2] - boxes1[:, :, :, :, 0]) * \
                      (boxes1[:, :, :, :, 3] - boxes1[:, :, :, :, 1])
            square2 = (boxes2[:, :, :, :, 2] - boxes2[:, :, :, :, 0]) * \
                      (boxes2[:, :, :, :, 3] - boxes2[:, :, :, :, 1])

            union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)

    def compute_output_shape(self, input_shape):
        return [(1,)]

    def compute_mask(self, inputs, mask=None):
        return [None]

    def get_config(self):
        return {
            'num_classes' : self.num_classes,
            'cell_size'   : self.cell_size,
        }
