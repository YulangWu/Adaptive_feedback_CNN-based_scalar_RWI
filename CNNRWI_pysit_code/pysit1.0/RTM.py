#=======================================================================================
# The Python Seismic Imaging Toolbox (PySIT) is research-scale platform for 
# developing and prototyping numerical methods and algorithms for seismic 
# imaging and full waveform inversion (FWI), in 1, 2, and 3 dimensions. 
# PySIT is designed to be a common platform which implements the standard 
# seismic imaging methods from the literature and from which the Imaging and 
# Computing Group at MIT, and outside research groups, can quickly develop 
# and prototype new methods, and reproducibly compare or benchmark those 
# new methods against the state of the art in the field.

# PySIT is designed to accelerate the development of seismic imaging algorithms 
# by providing an interface that is consistent with, and in fact reads like, 
# our derivation of the mathematics of seismic imaging. By combining this with
#  the accessibility of the Python programming language (from both the 
# programming and monetary cost perspectives), PySIT provides an efficient 
# environment for incorporating new research, and training new researchers, 
# on short time scales.

# With PySIT, we aim to cleanly and efficiently provide an open platform for 
# future development of seismic imaging methods, both at MIT and elsewhere. 
# Moreover, by embracing an open development model, we aim to encourage 
# cooperation within the research community and to facilitate reproducible research.
# Note: This license has also been called the "Simplified BSD License" 
# and the "FreeBSD License". See also the 3-clause BSD License.

# Copyright (c) 2011-2013, Massachusetts Institute of Technology (MIT)

# All rights reserved.

# Redistribution and use in source and binary forms, with or without modification, 
# are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this 
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice, 
# this list of conditions and the following disclaimer in the documentation and/or 
# other materials provided with the distribution.

# 3. Neither the name of the Massachusetts Institute of Technology nor the names of 
# its contributors may be used to endorse or promote products derived from this 
# software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT 
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR 
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY 
# OF SUCH DAMAGE.
#=======================================================================================

import time
import numpy as np
import matplotlib.pyplot as plt

from pysit import *
from pysit.gallery import horizontal_reflector

#from temporal_least_squares import TemporalLeastSquares # Integral method
import os
import argparse
from scipy.signal import hilbert

from BornModeling import BornModeling
from pysit.modeling.temporal_modeling import TemporalModeling

import glob
from random import choices, shuffle

def rtm(base_model,shots,nshots, norm=True):
    rtm_all_shot = np.zeros((1,a.nx*a.nz))

    if norm:
        rtm_all_shot_normed = np.zeros((1,a.nx*a.nz))

    RTM_time = time.time()
    for num_shot in range(len(shots)):
        # 1. get source wavefield
        source_wavefields = scalar_born_modeling.forward_model(shots[num_shot],base_model,imaging_period=2,return_parameters=['wavefield'])
        source_wavefields = np.array(source_wavefields['wavefield'])
        source_wavefields = np.reshape(source_wavefields, (-1,a.nx*a.nz))
        
        # for i in range(1500):
        #     if i % 100 == 0:
        #         vp = source_wavefields[i,:].copy()
        #         vp.shape = a.nx,a.nz
        #         plt.imshow(np.transpose(vp),aspect='auto')
        #         plt.set_cmap('seismic')
        #         plt.clim(-0.01, 0.01)
        #         plt.title(str(rank)+'source'+str(i))
        #         plt.savefig(str(rank)+'source'+str(i))



        # 2. get reverse-time data
        seis_data = shots[num_shot].receivers._data
        seis_data = np.flip(seis_data,0) #reverse-time data


        nt, _ = seis_data.shape
        data_reverse_time = np.zeros((nt,a.nx,a.nz))
        for i in range(a.nx):
            data_reverse_time[:,i,a.izr] = seis_data[:,i]
        data_reverse_time =np.reshape(data_reverse_time,(nt,a.nx*a.nz))

        # 3. get reverse-time receiver wavefield
        receiver_wavefields = scalar_born_modeling.forward_model_with_2D_source(data_reverse_time,base_model,imaging_period=2,return_parameters=['wavefield'])
        receiver_wavefields = np.array(receiver_wavefields['wavefield'])
        receiver_wavefields = np.reshape(receiver_wavefields, (-1,a.nx*a.nz))

        # for i in range(1500):
        #     if i % 100 == 0:
        #         vp = receiver_wavefields[i,:].copy()
        #         vp.shape = a.nx,a.nz
        #         plt.imshow(np.transpose(vp),aspect='auto')
        #         plt.set_cmap('seismic')
        #         plt.clim(-0.01, 0.01)
        #         plt.title(str(rank)+'receiver'+str(i))
        #         plt.savefig(str(rank)+'receiver'+str(i))

        # 3. RTM imaging to get FWI gradient
        rtm_image = np.zeros((1,a.nx*a.nz))

        if norm:
            rtm_image_normed = np.zeros((1,a.nx*a.nz))
            normalization = np.zeros((1,a.nx*a.nz))

        # # get the first time-derivatives for both source and receiver wavefield
        # source_wavefields = np.diff(source_wavefields, n = 1, axis = 0)
        # receiver_wavefields = np.diff(receiver_wavefields, n = 1, axis = 0)
        nt = len(receiver_wavefields)
        for i in range(nt):
            # conventional cross-corrleation of us and ur
            rtm_image += np.multiply(source_wavefields[i,:],receiver_wavefields[nt-1-i,:])
            if norm:
                normalization += np.multiply(source_wavefields[i,:],source_wavefields[i,:])

        if norm:
            rtm_image_normed = rtm_image.copy()
            rtm_image_normed /= normalization

        rtm_all_shot += rtm_image

        if norm:
            rtm_all_shot_normed += rtm_image_normed

        print('RTM:', num_shot, (time.time() - RTM_time)/(num_shot+1))

    res = {}
    res['rtm'] = rtm_all_shot / nshots
    if norm:
        res['rtm_normed'] = rtm_all_shot_normed / nshots
    return res

if __name__ == '__main__':
    # Setup


    # =============================================================================
    # 1. Parameters
    # -----------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='test', choices=["train","test"])
    parser.add_argument("--nx", type=int, default=256, help="Number of horizontal (x) grid lines in the model.")
    parser.add_argument("--nz", type=int, default=256, help="Number of vertical (z) grid lines in the model.")
    parser.add_argument("--dh", type=float, default=0.0125, help="spatial increment.") 
    parser.add_argument("--nt", type=int, default=4000, help="Number of time steps.")
    parser.add_argument("--dt", type=float, default=0.001, help="temperal increment.")
    parser.add_argument("--order", type=int, default=4, help="spatial order.")
    parser.add_argument("--nshots", type=int, default=24, help="number of shots.")
    parser.add_argument("--izs", type=int, default=5, help="depth of source.")
    parser.add_argument("--izr", type=int, default=5, help="depth of receivers.")
    parser.add_argument("--f_max", type=int, default=20, help="peak frequency.")
    parser.add_argument("--water_depth", type=int, default=0, help="water depth in grids.")
    parser.add_argument("--output_dir", default='pysit1.0', help="output directory.")
    parser.add_argument("--data_dir", default='output0', help="output directory.")
    parser.add_argument("--model_dir", default='given_models0', help="output directory.")
    parser.add_argument("--figs", type=bool, default=False, help="plot figures")
    parser.add_argument("--pml_len", type=int, default=0.1, help="Size of the PML in physical units")
    parser.add_argument("--pml_val", type=int, default=100, help="Scaling factor for the PML coefficient")

    # export options
    a = parser.parse_args()

    print('========================================================')
    for k, v in a._get_kwargs():
        print(k, "=", v)
    print('========================================================')


    #if not os.path.exists(a.output_dir):
    #    os.makedirs(a.output_dir)
    if not os.path.exists(a.data_dir):
        os.makedirs(a.data_dir)
    if not os.path.exists(a.model_dir):
        print('The input directory does not exist')

    # Set up PySIT parameters:
    s_zpos = a.dh*a.izs
    r_zpos = a.dh*a.izr
    pmlx = PML(a.pml_len, a.pml_val) 
    pmlz = PML(a.pml_len, a.pml_val) 
    n_pml = int(a.pml_len/a.dh)
    x_config = (0, a.nx*a.dh, pmlx, pmlx)
    z_config = (0, a.nz*a.dh, pmlz, pmlz)
    d = RectangularDomain(x_config, z_config)
    m = CartesianMesh(d, a.nx, a.nz)
    C, C0, m, d = horizontal_reflector(m)
    trange = (0.0,a.nt*a.dt)

    shots_scalar = equispaced_acquisition(m,
                                   RickerWavelet(a.f_max),
                                   sources=a.nshots,
                                   source_depth=s_zpos,
                                   source_kwargs={},
                                   receivers='max',
                                   receiver_depth=r_zpos,
                                   receiver_kwargs={},
                                   )

    solver_scalar = ConstantDensityAcousticWave(m,
                                         formulation='scalar',
                                         model_parameters={'C': C},
                                         cfl_safety=1/1.4,
                                         spatial_accuracy_order=a.order,
                                         trange=trange,
                                         kernel_implementation='cpp')

    scalar_born_modeling = BornModeling(solver_scalar)        

    # =============================================================================
    # 2. RTM
    # -----------------------------------------------------------------------------

    # Get the file names under the input directory: (true and smooth velocity)
    input_paths = glob.glob(os.path.join(a.model_dir, "true*vp.dat"))
    # Count the number of files under the input directory:
    numFiles = len(input_paths)
    shuffle(input_paths)

    
    for filename in input_paths:

        model_number = int(filename[filename.find('true')+4:filename.find('vp')])

        print(filename,model_number)

        output_file_name = os.path.join(a.data_dir,"rtm" + str(model_number) + ".dat")
        if os.path.exists(output_file_name):
            continue
        
        # =============================================================================
        # 2.1 read true and initial model
        # -----------------------------------------------------------------------------
        vp_smooth = np.zeros((a.nx*a.nz, 1))

        # true and smooth model
        vp_true = np.loadtxt(os.path.join(a.model_dir, 'true' + str(model_number) + 'vp.dat')) #,"true_Marmousi128x256.dat")) #
        vp_true.shape = a.nx*a.nz,1

        # smooth model
        vp_smooth = np.loadtxt(os.path.join(a.model_dir, 'mig' + str(model_number) + 'vp.dat')) #,"true_Marmousi128x256.dat")) #
        vp_smooth.shape = a.nx*a.nz,1

        
        C = vp_true.copy()
        C0 = np.ones((a.nx*a.nz, 1))*vp_true[0,0] #used to remove direct wave

  
        if a.figs == True:
            vp = vp_true.copy()
            vp.shape = a.nx,a.nz
            plt.subplot(1,3,1)
            plt.imshow(np.transpose(vp))
            plt.title('True vp model')

            vp = vp_smooth.copy()
            vp.shape = a.nx,a.nz
            plt.subplot(1,3,2)
            plt.imshow(np.transpose(vp))
            plt.title('Initial vp model')

            vp = C0.copy()
            vp.shape = a.nx,a.nz
            plt.subplot(1,3,3)
            plt.imshow(np.transpose(vp))
            plt.title('homogenous vp model')
            plt.show()


        
        # =============================================================================
        # 2.2 Generate observed data without direct wave
        # -----------------------------------------------------------------------------
        time_start = time.time()

        ### Generate synthetic Seismic data ###
        print('Generating training data without direct wave...')

        scalar_data = []
        true_scalar_model = solver_scalar.ModelParameters(m,{'C': C})
        homo_scalar_model = solver_scalar.ModelParameters(m,{'C': C0})

        forward_time = time.time()
        for num_shot in range(len(shots_scalar)):
            source_wavefields = scalar_born_modeling.forward_model(shots_scalar[num_shot],homo_scalar_model,return_parameters=['simdata'])
            Ddata = source_wavefields['simdata']
            source_wavefields = scalar_born_modeling.forward_model(shots_scalar[num_shot],true_scalar_model,return_parameters=['simdata'])
            shots_scalar[num_shot].receivers._data = source_wavefields['simdata'] - Ddata

            print('data acquisition:', num_shot, (time.time() - forward_time)/(num_shot+1))

        if a.figs == True:
            data = shots_scalar[num_shot].receivers._data.copy()
            plt.imshow(np.transpose(data),aspect='auto')
            plt.set_cmap('seismic')
            plt.clim(-0.01, 0.01)
            plt.title('data')
            plt.show()

        # =============================================================================
        # 2.3 RTM
        # -----------------------------------------------------------------------------
        mig_scalar_model = solver_scalar.ModelParameters(m,{'C': vp_smooth})

        rtm_image = rtm(mig_scalar_model,shots_scalar,a.nshots,norm=False)
        # image_normed = rtm_image['rtm_normed']
        # image_normed.shape = a.nx,a.nz
        # # subtract smoothed model to get high-pass image
        # image -= smooth(image,num_smooth=1)
        # image_normed.shape = a.nx*a.nz,1

        image = rtm_image['rtm']
        # image.shape = a.nx,a.nz
        # # subtract smoothed model to get high-pass image
        # image -= smooth(image,num_smooth=1)
        image.shape = a.nx*a.nz,1


        if a.figs == True:
            plt.figure(101)
            vp = image.copy()
            vp.shape = a.nx,a.nz
            plt.imshow(np.transpose(vp))
            plt.title('RTM')
            plt.show()


        res = np.zeros((a.nz*a.nx*3,1))
        res[0*a.nz*a.nx:1*a.nz*a.nx] = image/(max(image))
        res[1*a.nz*a.nx:2*a.nz*a.nx] = vp_true*1000
        res[2*a.nz*a.nx:3*a.nz*a.nx] = vp_smooth*1000  

            
        np.savetxt(output_file_name,res,fmt="%17.8f")

        time_end = time.time()
        print('one RTM image requires {} min'.format((time_end - time_start)/60))
