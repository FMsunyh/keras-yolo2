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

class Loss(keras.layers.Layer):
    def __init__(self, num_classes=20, cell_size=13, boxes_per_cell=2, *args, **kwargs):
        self.num_classes = num_classes
        self.class_wt = np.ones(self.num_classes, dtype='float32')

        self.cell_size   = cell_size
        self.boxes_per_cell   = boxes_per_cell
        self.batch_size = 4
        self.anchors = list([0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828])
        self.object_scale = 1.0
        self.noobject_scale = 0.5
        self.class_scale = 2.0
        self.coord_scale = 5.0

        super(Loss, self).__init__(*args, **kwargs)

    def classification_loss(self,labels,  classification, response):
        class_delta = response * (labels - classification)
        cls_loss = tf.reduce_mean(tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]), name='cls_loss') * self.class_scale

        return cls_loss

    def regression_loss(self, regression_target, regression,object_mask):
        coord_mask = tf.expand_dims(object_mask, 4)
        boxes_delta = coord_mask * (regression_target - regression)
        coord_loss = tf.reduce_mean(tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]),
                                    name='coord_loss') * self.coord_scale
        reg_loss = tf.constant(0,dtype=tf.float32)
        return reg_loss

    def confidence_loss(self,predict_scales, iou_predict_truth, object_mask):
        # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask

        # object_loss
        object_delta = object_mask * (iou_predict_truth - predict_scales)
        object_loss = tf.reduce_mean(tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]),  name='object_loss') * self.object_scale

        # noobject_loss
        noobject_delta = noobject_mask * predict_scales
        noobject_loss = tf.reduce_mean(tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]), name='noobject_loss') * self.noobject_scale

        return [object_loss, noobject_loss]


    def call(self, inputs):
        '''

        :param inputs:
        predicts: shape(None, 1470)
        labels shape(None, 7, 7, 25)
        :return:
        '''

        y_pred, gt_boxes, y_true = inputs

        mask_shape = tf.shape(y_true)[:4]

        cell_x = tf.to_float(
            tf.reshape(tf.tile(tf.range(self.cell_size), [self.cell_size]), (1, self.cell_size, self.cell_size, 1, 1)))
        cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))

        cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [self.batch_size, 1, 1, 5, 1])

        coord_mask = tf.zeros(mask_shape)
        conf_mask = tf.zeros(mask_shape)
        class_mask = tf.zeros(mask_shape)

        seen = tf.Variable(0.)
        total_recall = tf.Variable(0.)

        """
        Adjust prediction
        """
        ### adjust x and y      
        pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid

        ### adjust w and h
        pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(self.anchors, [1, 1, 1, 5, 2])

        ### adjust confidence
        pred_box_conf = tf.sigmoid(y_pred[..., 4])

        ### adjust class probabilities
        pred_box_class = y_pred[..., 5:]

        """
        Adjust ground truth
        """
        ### adjust x and y
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

        true_box_conf = iou_scores * y_true[..., 4]

        ### adjust class probabilities
        true_box_class = tf.argmax(y_true[..., 5:], -1)

        """
        Determine the masks
        """
        ### coordinate mask: simply the position of the ground truth boxes (the predictors)
        coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * self.coord_scale

        ### confidence mask: penelize predictors + penalize boxes with low IOU
        # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
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
        conf_mask = conf_mask + tf.to_float(best_ious < 0.6) * (1 - y_true[..., 4]) * self.noobject_scale

        # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
        conf_mask = conf_mask + y_true[..., 4] * self.object_scale

        ### class mask: simply the position of the ground truth boxes (the predictors)
        class_mask = y_true[..., 4] * tf.gather(self.class_wt, true_box_class) * self.class_scale


        """
        Finalize the loss
        """
        nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
        nb_conf_box = tf.reduce_sum(tf.to_float(conf_mask > 0.0))
        nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))

        loss_xy = tf.reduce_sum(tf.square(true_box_xy - pred_box_xy) * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_wh = tf.reduce_sum(tf.square(true_box_wh - pred_box_wh) * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_conf = tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * conf_mask) / (nb_conf_box + 1e-6) / 2.
        loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
        loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)

        loss = loss_xy + loss_wh + loss_conf + loss_class

        self.add_loss(loss)
        return loss
        
        # index_classification = tf.multiply(tf.pow(self.cell_size, 2), self.num_classes)
        # index_confidence = tf.multiply(tf.pow(self.cell_size, 2), self.num_classes + self.boxes_per_cell)
        # 
        # 
        # predict_classes = tf.reshape(predicts[:, :index_classification], [-1, self.cell_size, self.cell_size, self.num_classes])
        # predict_scales = tf.reshape(predicts[:, index_classification:index_confidence], [-1, self.cell_size, self.cell_size, self.boxes_per_cell])
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
