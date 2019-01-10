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
# @Time    : 11/9/2018 3:54 PM
# @Author  : Firmin.Sun (fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-
import argparse
import os

import keras
import keras.preprocessing.image
import tensorflow as tf
from keras.callbacks import TensorBoard

from core.callbacks import RedirectModel
from core.callbacks.eval import Evaluate
from core.models.model import create_yolo2
from core.preprocessing import PascalVocGenerator
from core.utils.config import load_setting_cfg

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def get_session():
    cfg = tf.ConfigProto()
    cfg.gpu_options.allocator_type = 'BFC'
    # cfg.gpu_options.per_process_gpu_memory_fraction = 0.90
    cfg.gpu_options.allow_growth = True
    return tf.Session(config=cfg)

def set_gpu(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    sess = get_session()
    import keras.backend.tensorflow_backend as ktf
    ktf.set_session(sess)


def create_model(num_classes=20):
    image = keras.layers.Input((416, 416, 3))
    # image = keras.layers.Input((None, None, 3))
    # gt_boxes = keras.layers.Input((7, 7, 25))
    gt_boxes = keras.layers.Input((1, 1, 1, 100, 4))
    targets = keras.layers.Input((13, 13, 5, 25))
    train_model = create_yolo2([image, gt_boxes,  targets], num_classes=num_classes, weights=None)
    eval_model = create_yolo2(keras.layers.Input((None, None, 3)), training=False, num_classes=num_classes, weights=None)

    return train_model, eval_model

def create_callbacks(model, evaluation_model, validation_generator, args):
    callbacks = []
    # save the model
    if args.snapshots:
        # ensure directory created first; otherwise h5py will error after epoch.
        os.makedirs(args.snapshot_path, exist_ok=True)

        checkpoint_model = keras.callbacks.ModelCheckpoint(
            os.path.join(args.root_path, 'snapshots', args.tag, args.tag + '_{epoch:02d}.h5'),
            verbose=1,
            save_weights_only=True,
        )

        callbacks.append(checkpoint_model)

        weight_path = os.path.join(args.root_path, 'snapshots', args.tag, args.tag + '_weight_evaluation.h5')
        checkpoint_evaluation = keras.callbacks.ModelCheckpoint(
            weight_path,
            verbose=1,
            save_weights_only=True,
        )
        callbacks.append(checkpoint_evaluation)


    if args.tensorboard_dir:
        tensorboard_dir = os.path.abspath(os.path.join(args.root_path, args.tensorboard_dir,args.tag))
        tensorboard_callback = TensorBoard(
            log_dir                = tensorboard_dir,
            batch_size             = args.batch_size,
            write_graph            = True,
            write_grads            = False,
            write_images           = False
        )

        callbacks.append(tensorboard_callback)

    # evaluation
    if args.evaluation and validation_generator:
        evaluation = Evaluate(weight_path, validation_generator, save_path=args.save_path,
                              tensorboard=tensorboard_callback)
        evaluation = RedirectModel(evaluation, evaluation_model)
        callbacks.append(evaluation)

    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.1,
        patience=2,
        verbose=1,
        mode='auto',
        min_delta=0.0001,
        cooldown=0,
        min_lr=0
    ))

    return callbacks

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
        batch_size=1,
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

    parser.add_argument('--save-path', help='Path for saving images with detections.',
                        default=os.path.join(os.path.dirname(__file__), '../../', 'experiments/eval_output'))

    parser.add_argument('--weight_path', help='Path to classes directory (ie. /tmp/voc.h5).', default='', type=str)

    parser.add_argument('--tensorboard-dir', help='Log directory for Tensorboard output', default='./logs')
    parser.add_argument('--no-snapshots', help='Disable saving snapshots.', dest='snapshots', action='store_false')
    parser.add_argument('--snapshot-path',  help='Path to store snapshots of models during training (defaults to \'./snapshots\')',
                        default=os.path.join(os.path.dirname(__file__), '../../snapshots'))
    parser.add_argument('--evaluation', help='', default=False, type=bool)

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
    set_gpu(args)

    # create the model
    print('Creating model, this may take a second...')
    model, eval_model = create_model()

    model.load_weights(filepath=args.weight_path, by_name=True)

    # compile model (note: set loss to None since loss is added inside layer)
    model.compile(loss=None, optimizer=keras.optimizers.adam(lr=1e-5))
    eval_model.compile(loss=None, optimizer=keras.optimizers.adam(lr=1e-5))

    # print model summary
    print(model.summary(line_length=180))

    train_generator, valid_generator = create_generators(args)
    # start training

    callbacks = create_callbacks(model, eval_model, valid_generator, args)

    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=len(train_generator.image_names) // args.batch_size,
        # steps_per_epoch=10,
        epochs=100,
        verbose=1,
        callbacks=callbacks,
    )

if __name__ == '__main__':
    main()

    '''
    cd tools
    python train_yolo2.py pascal /home/syh/train_data/VOCdevkit/VOC2007
    '''