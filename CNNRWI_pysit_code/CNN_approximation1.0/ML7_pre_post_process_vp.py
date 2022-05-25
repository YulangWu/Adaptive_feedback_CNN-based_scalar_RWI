# Code borrows heavily from pix2pix.
# Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017). 
# Image-to-image translation with conditional adversarial networks. 
# In Proceedings of the IEEE conference on computer vision and 
# pattern recognition (pp. 1125-1134).


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import glob
import scipy as sp
import scipy.ndimage
import cmath

import glob
from random import shuffle
import random
import matplotlib.pyplot as plt

def get_name(path):
    fullname, _ = os.path.splitext(os.path.basename(path))
    return fullname

def get_list_files(input_dir,prefix):
    if input_dir is None or not os.path.exists(input_dir):
        raise Exception("input_dir does not exist")

    file_path = glob.glob(os.path.join(input_dir, prefix + "*.dat"))
    numFiles = len(file_path)
  
    file_list = []
    if len(file_path) == 0:
        raise Exception("input_dir contains no image files")

    return file_path


def should(iteration, freq):
    return freq > 0 and ((iteration + 1) % freq == 0)

def load_train_data(input_dir, prefix, nx, nz):
    file_list = get_list_files(input_dir, prefix)
    image_RTM = np.zeros((len(file_list),nx*nz))
    vp_true = np.zeros((len(file_list),nx*nz))
    vp_init = np.zeros((len(file_list),nx*nz))
    

    for i,filename in enumerate(file_list):
        print(" input: " + filename)
        data_set = np.loadtxt(filename)  
        len_data = len(data_set)

        image_RTM[i,:] = data_set[0*int(len_data/3):1*int(len_data/3)] 
        vp_true[i,:] = data_set[1*int(len_data/3):2*int(len_data/3)] 
        vp_init[i,:] = data_set[2*int(len_data/3):3*int(len_data/3)] 

    data = {}
    data['image_RTM'] = image_RTM
    data['vp_true'] = vp_true
    data['vp_init'] = vp_init

    return data

def load_data(input_dir, prefix, nx, nz):
    pair_data = load_train_data(input_dir, prefix, nx, nz)

    return pair_data