# Code borrows heavily from pix2pix.
# Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017). 
# Image-to-image translation with conditional adversarial networks. 
# In Proceedings of the IEEE conference on computer vision and 
# pattern recognition (pp. 1125-1134).

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import collections


parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="train", choices=["train", "test", "export","real"])
parser.add_argument("--input_dir", default="_dataset")
parser.add_argument("--output_dir", default="_outputs")
parser.add_argument("--parameter_dir", default="CNN_weights")
parser.add_argument("--seed", type=int)
parser.add_argument("--global_iteration_number", type=int, default=1)
parser.add_argument("--max_steps", type=int)
parser.add_argument("--max_epochs", type=int, default=400)
parser.add_argument("--display_freq", type=int, default=100)
parser.add_argument("--output_freq", type=int, default=1)
parser.add_argument("--store_weights_freq", type=int, default=400)
parser.add_argument("--ngf", type=int, default=32)
parser.add_argument("--ndf", type=int, default=32)
parser.add_argument("--batch", type=int, default=1)
parser.add_argument("--nz", type=int, default=256)
parser.add_argument("--nx", type=int, default=256)
parser.add_argument("--nt", type=int, default=2048)
parser.add_argument("--stack_num", type=int, default=20)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--beta1", type=float, default=0.5)
parser.add_argument("--l1_weight", type=float, default=1)
parser.add_argument("--figs", type=bool, default=False)
parser.add_argument("--CNN_num", default='0')
parser.add_argument("--max_image", type=float, default=100)
parser.add_argument("--max_reflectivity", type=float, default=0.2)
parser.add_argument("--max_vp", type=float, default=3000)
parser.add_argument("--mean_vp", type=float, default=2000)

a = parser.parse_args()

if a.mode == 'train':
    a.input_dir = 'train' + a.input_dir
    a.output_dir = 'train' + a.output_dir

if a.mode == 'export':
    a.input_dir = 'train' + a.input_dir
    a.output_dir = 'train' + a.output_dir
    a.figs = True
    a.max_epochs = 1
    a.lr = 0.0

if a.mode == 'test':
    a.input_dir = a.mode + a.input_dir
    a.output_dir = a.mode + a.output_dir
    a.max_epochs = 1
    a.lr = 0.0
    a.figs = True

if a.mode == 'real':
    a.input_dir = a.mode + a.input_dir
    a.output_dir = a.mode + a.output_dir
    a.max_epochs = 1
    a.lr = 0.0
    a.figs = True

if a.input_dir is None or not os.path.exists(a.input_dir):
    raise Exception("input_dir does not exist")

if not os.path.exists(a.output_dir):
    try:
        os.makedirs(a.output_dir)
    except IOError:
        print("Output directory exists")


Model = collections.namedtuple("Model", "predict_real, predict_fake, discrim_loss, gen_loss_GAN, gen_loss_L1, "
                                        "inputs, targets, outputs, train, discrim_grads_and_vars, gen_grads_and_vars")

model_generator = collections.namedtuple("model_generator", "L2_loss, inputs, inputs2, targets, outputs, train, gen_grads_and_vars")