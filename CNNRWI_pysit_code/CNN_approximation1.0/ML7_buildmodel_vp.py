# Code borrows heavily from pix2pix.
# Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017). 
# Image-to-image translation with conditional adversarial networks. 
# In Proceedings of the IEEE conference on computer vision and 
# pattern recognition (pp. 1125-1134).

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ML7_parse_vp import Model
from ML7_parse_vp import model_generator
from util import *

def create_generator_skip(generator_inputs, outputs_channels, ngf):
    filter_size = 3
    layers = []
    with tf.variable_scope("encoder_1"):
        output = conv(generator_inputs, ngf, 1, filter_size)
        layers.append(output)

    layer_specs = [
        (ngf * 2, 2),  
        (ngf * 4, 2),  
        (ngf * 8, 2),  
        (ngf * 16, 1),  
        (ngf * 8, 1), 
        (ngf * 8, 1),  
        (ngf * 8, 1),  
    ]
    for encoder_layer, (out_channels, stride) in enumerate(layer_specs):
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            convolved = conv(rectified, out_channels, stride, filter_size)
            output = batchnorm(convolved)
            layers.append(output)

    layer_specs = [
        (ngf * 8, 1),  
        (ngf * 8, 1),  
        (ngf * 16, 1),  
        (ngf * 8, 1),  
        (ngf * 8, 2),  
        (ngf * 4, 2),  
        (ngf * 2, 2),  
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, stride) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                input_hidden = layers[-1]
            else:
                input_hidden = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input_hidden)
            output = deconv(rectified, out_channels, stride, filter_size)
            output = batchnorm(output)

            #if dropout > 0.0:
            #    output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    with tf.variable_scope("decoder_1"):
        input_hidden = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input_hidden)
        output = deconv(rectified, outputs_channels, 1, filter_size)
        output = tf.tanh(output)
        layers.append(output)

    for i in range(len(layers)):
        print(i, layers[i])
    return layers

def create_model(inputs, inputs2, targets, nz, nx, batch, lr, beta1, ngf):
    inputs = tf.reshape(inputs, [batch, nx, nz, 1])
    inputs2 = tf.reshape(inputs2, [batch, nx, nz, 1])
    targets = tf.reshape(targets, [batch, nx, nz, 1])
    inputs_combine = tf.concat([inputs,inputs2],axis=3)

    with tf.variable_scope("generator") as scope:
        out_channels = int(targets.get_shape()[-1])
        outputs = create_generator_skip(inputs_combine, out_channels, ngf) # U-net:

    with tf.name_scope("generator_loss"):
        L2_loss = tf.reduce_mean(tf.abs(outputs[-1] - targets)*tf.abs(outputs[-1] - targets))

    with tf.name_scope("generator_train"):
        gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
        gen_optim = tf.train.AdamOptimizer(lr, beta1)
        gen_grads_and_vars = gen_optim.compute_gradients(L2_loss, var_list=gen_tvars)
        gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([L2_loss])

    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step + 1)

    return model_generator(
        inputs=inputs,
        inputs2=inputs2,
        targets=targets,
        outputs=outputs,
        L2_loss= L2_loss, 
        gen_grads_and_vars=gen_grads_and_vars,
        train=tf.group(update_losses, incr_global_step, gen_train),
    )
