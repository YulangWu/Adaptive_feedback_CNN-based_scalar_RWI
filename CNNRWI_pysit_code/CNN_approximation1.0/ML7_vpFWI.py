# Code borrows heavily from pix2pix.
# Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017). 
# Image-to-image translation with conditional adversarial networks. 
# In Proceedings of the IEEE conference on computer vision and 
# pattern recognition (pp. 1125-1134).

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import time
import matplotlib
import matplotlib.pyplot as plt
import os
import subprocess

from ML7_parse_vp import *
from ML7_pre_post_process_vp import *
from ML7_buildmodel_vp import *

import shutil

def main():
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    a.parameter_dir = a.parameter_dir + '_vp'
    if not os.path.exists(a.parameter_dir):
        os.makedirs(a.parameter_dir)
    
    if a.mode != 'train':
        a.parameter_dir = a.parameter_dir + (a.CNN_num)
        print(a.parameter_dir)

    if tf.__version__.split('.')[0] != "1":
        raise Exception("Tensorflow version 1 required")

    if a.seed is None:
        a.seed = random.randint(0, 2 ** 31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    for k, v in a._get_kwargs():
        print(k, " = ", v)

    ##############################################################################
    # 1. Input train/test data:
    if os.path.exists(os.path.join(a.output_dir,'image_RTM.npy')):
        print('load existing data')
        image_RTM = np.load(os.path.join(a.output_dir,'image_RTM.npy'))
        vp_true = np.load(os.path.join(a.output_dir,'vp_true.npy'))
        vp_init = np.load(os.path.join(a.output_dir,'vp_init.npy'))
    else:
        data_set = load_data(a.input_dir, 'CNN_'+ a.mode + '_dataset', a.nx, a.nz)
        print('save data')  
        image_RTM = data_set['image_RTM']
        vp_true = data_set['vp_true'] 
        vp_init = data_set['vp_init'] 
         

        np.save(os.path.join(a.output_dir,'image_RTM.npy'),image_RTM)
        np.save(os.path.join(a.output_dir,'vp_true.npy'),vp_true)
        np.save(os.path.join(a.output_dir,'vp_init.npy'),vp_init)

    image_RTM /= a.max_image
    vp_true = (vp_true - a.mean_vp) / a.max_vp
    vp_init = (vp_init - a.mean_vp) / a.max_vp
    


    # for i in range(len(image_RTM)):
    #     plt.figure(101)
    #     plt.subplot(1,3,1)
    #     vp = image_RTM[i,:].copy()
    #     vp.shape = a.nx,a.nz
    #     plt.imshow(np.transpose(vp),aspect='auto')
    #     plt.title('vz')
    #     plt.clim(-1,1)

    #     plt.subplot(1,3,2)
    #     vp = vp_init[i,:].copy()
    #     vp.shape = a.nx,a.nz
    #     plt.imshow(np.transpose(vp),aspect='auto')
    #     plt.title('smooth vp')
    #     plt.clim(-1,1)

    #     plt.subplot(1,3,3)
    #     vp = vp_true[i,:].copy()
    #     vp.shape = a.nx,a.nz
    #     plt.imshow(np.transpose(vp),aspect='auto')
    #     plt.title('true vp')
    #     plt.clim(-1,1)

    #     plt.show()

    # prepare an index array for shuffling (shuffling index not data every time)
    index_arr = [i for i in range(len(image_RTM))]
    print('Number of stacked data: {}'.format(len(image_RTM))) 



    ##############################################################################
    # 2. Build CNN

    inputs = tf.placeholder(tf.float32, shape=(a.nx, a.nz))  #RTM image
    inputs2 = tf.placeholder(tf.float32, shape=(a.nx, a.nz)) #starting model
    targets = tf.placeholder(tf.float32, shape=(a.nx, a.nz)) #label model

    #=============================================================================
    model_generator = create_model(inputs, inputs2, targets, a.nz, a.nx, a.batch, a.lr, a.beta1, a.ngf)
    #=============================================================================


    ##############################################################################
    # 3. Execute CNN graph

    total_iterations = len(image_RTM) * a.max_epochs #a.batch
    # loss_curve contains all L1 loss for each snapshot of each wavefield
    loss_curve = []

    # Define the epoch
    if a.mode == 'train' or a.mode == 'export':
        max_steps = 2 ** 32
        if a.max_epochs is not None:
            max_steps = total_iterations
            print("max_steps %d" % max_steps)
    else:
        max_steps = 1
        print("max_steps %d" % max_steps)

    # Run Tensorflow in the scope of tf.Session
    saver = tf.train.Saver(max_to_keep=1)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        # Initialize all variables in tensorflow
        sess.run(tf.global_variables_initializer())

        # Compute the total number of the variables in tensorflow
        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v))
                                             for v in tf.trainable_variables()])
        print("parameter_count =", sess.run(parameter_count))

        # Reload the training results
        if a.mode == 'train':
            ckpt = tf.train.get_checkpoint_state(a.parameter_dir + (a.CNN_num))
        else:
            ckpt = tf.train.get_checkpoint_state(a.parameter_dir)
            
        if ckpt and ckpt.model_checkpoint_path:
            print("loading model from checkpoint")
            saver.restore(sess, ckpt.model_checkpoint_path)

        # remove the directory of CNN weights saved at the previous iteration     
        if a.mode == "train":
            try:
                shutil.rmtree(a.parameter_dir + (a.CNN_num))
            except:
                print("No directory found")

        # calculating the time
        t_start = time.time()
        t_used = 0.0

        # 1st loop: (epoch loop)
        nitr = -1  # total iteration number
        for epoch in range(a.max_epochs):

            if a.mode == 'train' and a.max_epochs > 1:
                # shuffle the list of index every epoch
                shuffle(index_arr)

            # 2nd loop : each stacked data
            for index in index_arr:
                nitr += 1
                
                #  Run TensorFlow session:
                fetches = {"train": model_generator.train, "L2_loss": model_generator.L2_loss,
                            "outputs": model_generator.outputs}

                # Without this, CNN weights are not fully stored
                fetches["gen_grads_and_vars"] = model_generator.gen_grads_and_vars

                input_image_RTM = image_RTM[[index_arr[index]],:]
                input_vp_true = vp_true[[index_arr[index]],:]
                input_vp_init = vp_init[[index_arr[index]],:]
                

                input_image_RTM.shape = a.nx,a.nz
                input_vp_true.shape = a.nx,a.nz
                input_vp_init.shape = a.nx,a.nz
                



                if a.mode == 'train' or a.mode == 'export':
                    results = sess.run(fetches,
                                    feed_dict={inputs: input_image_RTM, inputs2: input_vp_init, targets: input_vp_true})
                else:
                    zero_data = np.zeros((a.nx,a.nz))
                    results = sess.run(fetches,
                                    feed_dict={inputs: input_image_RTM, inputs2: input_vp_init, targets: zero_data})


                L2_loss = results["L2_loss"]
                loss_curve.append(L2_loss)

                if a.figs and (a.max_epochs == 1 or (epoch + 1) % a.display_freq == 0):
                    ###############################################################
                    #                          plot
                    ###############################################################
                    if index % int(len(input_image_RTM)/int(len(input_image_RTM))) == 0:
                        output_res = np.zeros((4,a.nx,a.nz))

                        output_res[0,:,:] = input_image_RTM[:,:]*a.max_image 
                        output_res[1,:,:] = input_vp_true[:,:]*a.max_vp + a.mean_vp
                        output_res[2,:,:] = input_vp_init[:,:]*a.max_vp + a.mean_vp
                        output_res[3,:,:] = results['outputs'][-1][0,:,:,0]*a.max_vp + a.mean_vp

                        output_res.shape = -1, 1
                        filename = os.path.join(a.output_dir, a.mode + str(index_arr[index]) + '_vp.dat')
                        np.savetxt(filename, output_res, fmt="%1.8f")

                if index_arr[index] == 0 and (epoch + 1) % a.display_freq == 0:
                    t_end = time.time()
                    t_used = t_end - t_start
                    t_period = t_used / (nitr+1) # average time per data
                    
                    t_remain = t_period * (total_iterations - nitr)
                    print('The running index = {}'.format(index_arr[index]))
                    print('epoch: %3.0f; loss = %17.14f \n (TIME min) used = %3.2f, period = %3.2f, remain = %3.2f' %
                          (epoch, loss_curve[-1], t_used / 60, t_period / 60, t_remain / 60))   #unit is minute

            # Store the CNN weights to the files every 'store_weights_freq' epoch
            if should(epoch, a.store_weights_freq) and a.mode =='train':
                if os.path.exists(a.parameter_dir):
                    print("saving training model at " + str(epoch) + "epoch")
                    saver.save(sess, a.parameter_dir + '/model.ckpt')
                    os.makedirs(a.parameter_dir + str(epoch))

                    subprocess.check_call("mv " + os.path.join(a.parameter_dir,"*") + " " + a.parameter_dir + str(epoch),
                                          shell=True)

                loss_filename = os.path.join(a.output_dir, 'L2_loss_' + a.mode + '_vp.dat')
                if os.path.exists(loss_filename):
                    os.remove(loss_filename)
                np.savetxt(loss_filename, np.array(loss_curve), fmt="%17.8f")

        

main()

