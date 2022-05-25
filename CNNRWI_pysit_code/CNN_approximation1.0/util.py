# Code borrows heavily from pix2pix.
# Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017). 
# Image-to-image translation with conditional adversarial networks. 
# In Proceedings of the IEEE conference on computer vision and 
# pattern recognition (pp. 1125-1134).

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def dense(x, inputFeatures, outputFeatures, scope=None, with_w=False):
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [inputFeatures, outputFeatures], tf.float32, tf.random_normal_initializer(stddev=0.02))
        bias = tf.get_variable("bias", [outputFeatures], initializer=tf.constant_initializer(0.0))
        if with_w:
            return tf.matmul(x, matrix) + bias, matrix, bias
        else:
            return tf.matmul(x, matrix) + bias

def lrelu(x, a):
    with tf.name_scope("lrelu"):
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def batchnorm(input):
    with tf.variable_scope("batchnorm"):
        input = tf.identity(input)

        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized


def conv(batch_input, out_channels, stride, filter_size, scope=None):
    with tf.variable_scope(scope or "conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [filter_size, filter_size, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        conv = tf.nn.conv2d(batch_input, filter, [1, stride, stride, 1], padding="SAME")
        return conv

def deconv(batch_input, out_channels, stride, filter_size):
    with tf.variable_scope("deconv"):
        batch, in_height, in_width, in_channels = \
            [int(d) for d in batch_input.get_shape()]
        filter = tf.get_variable("filter",
                                 [filter_size,
                                  filter_size,
                                  out_channels,
                                  in_channels],
                                 dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0, 0.02)) 
        conv = tf.nn.conv2d_transpose(batch_input,
                                      filter,
                                      [batch,
                                       in_height * stride,
                                       in_width * stride,
                                       out_channels],
                                      [1, stride, stride, 1],
                                      padding="SAME")

        return conv
