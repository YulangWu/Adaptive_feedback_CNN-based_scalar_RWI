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

import numpy as np
from pysit.util.derivatives import build_derivative_matrix, build_permutation_matrix, build_heterogenous_matrices
from numpy.random import uniform

import matplotlib.pyplot as plt
__all__ = ['BornModeling']

__docformat__ = "restructuredtext en"


class BornModeling(object):
    """Class containing a collection of methods needed for seismic inversion in
    the time domain.

    This collection is designed so that a collection of like-methods can be
    passed to an optimization routine, changing how we compute each part, eg, in
    time, frequency, or the Laplace domain, without having to reimplement the
    optimization routines.

    A collection of inversion functions must contain a procedure for computing:
    * the foward model: apply script_F (in our notation)
    * migrate: apply F* (in our notation)
    * demigrate: apply F (in our notation)
    * Hessian?

    Attributes
    ----------
    solver : pysit wave solver object
        A wave solver that inherits from pysit.solvers.WaveSolverBase

    """

    # read only class description
    @property
    def solver_type(self): return "time"

    @property
    def modeling_type(self): return "time"

    def __init__(self, solver):
        """Constructor for the TemporalInversion class.

        Parameters
        ----------
        solver : pysit wave solver object
            A wave solver that inherits from pysit.solvers.WaveSolverBase

        """

        if self.solver_type == solver.supports['equation_dynamics']:
            self.solver = solver
        else:
            raise TypeError("Argument 'solver' type {1} does not match modeling solver type {0}.".format(
                self.solver_type, solver.supports['equation_dynamics']))

    def _setup_forward_rhs(self, rhs_array, data):
        return self.solver.mesh.pad_array(data, out_array=rhs_array)

    def forward_model(self, shot, m0, imaging_period=1, return_parameters=[]):
        """Applies the forward model to the model for the given solver.

        Parameters
        ----------
        shot : pysit.Shot
            Gives the source signal approximation for the right hand side.
        m0 : solver.ModelParameters
            The parameters upon which to evaluate the forward model.
        return_parameters : list of {'wavefield', 'simdata', 'dWaveOp'}

        Returns
        -------
        retval : dict
            Dictionary whose keys are return_parameters that contains the specified data.

        Notes
        -----
        * u is used as the target field universally.  It could be velocity potential, it could be displacement, it could be pressure.
        * utt is used to generically refer to the derivative of u that is needed to compute the imaging condition.

        Forward Model solves:

        For constant density: m*u_tt - lap u = f, where m = 1.0/c**2
        For variable density: m1*u_tt - div(m2 grad)u = f, where m1=1.0/kappa, m2=1.0/rho, and C = (kappa/rho)**0.5
        """

        # Local references
        solver = self.solver
        solver.model_parameters = m0

        mesh = solver.mesh

        d = solver.domain
        dt = solver.dt
        nsteps = solver.nsteps
        source = shot.sources

        # Storage for the field
        if 'wavefield' in return_parameters:
            us = list()

        # Setup data storage for the forward modeled data
        if 'simdata' in return_parameters:
            simdata = np.zeros((solver.nsteps, shot.receivers.receiver_count))

        # Storage for the time derivatives of p
        if 'dWaveOp' in return_parameters:
            dWaveOp = list()

        # Step k = 0
        # p_0 is a zero array because if we assume the input signal is causal
        # and we assume that the initial system (i.e., p_(-2) and p_(-1)) is
        # uniformly zero, then the leapfrog scheme would compute that p_0 = 0 as
        # well. ukm1 is needed to compute the temporal derivative.
        solver_data = solver.SolverData()

        rhs_k = np.zeros(mesh.shape(include_bc=True))
        rhs_kp1 = np.zeros(mesh.shape(include_bc=True))

        for k in range(nsteps):

            uk = solver_data.k.primary_wavefield
            uk_bulk = mesh.unpad_array(uk)

            if 'wavefield' in return_parameters:
                if k % imaging_period == 0:  # Save every 'imaging_period' number of steps
                    us.append(uk_bulk.copy())

            # Record the data at t_k
            if 'simdata' in return_parameters:
                shot.receivers.sample_data_from_array(uk_bulk, k, data=simdata)

            if k == 0:
                rhs_k = self._setup_forward_rhs(rhs_k, source.f(k*dt))
                rhs_kp1 = self._setup_forward_rhs(rhs_kp1, source.f((k+1)*dt))
            else:
                # shift time forward
                rhs_k, rhs_kp1 = rhs_kp1, rhs_k
            rhs_kp1 = self._setup_forward_rhs(rhs_kp1, source.f((k+1)*dt))

            # Note, we compute result for k+1 even when k == nsteps-1.  We need
            # it for the time derivative at k=nsteps-1.
            solver.time_step(solver_data, rhs_k, rhs_kp1)

            # Compute time derivative of p at time k
            # Note that this is is returned as a PADDED array
            if 'dWaveOp' in return_parameters:
                if k % imaging_period == 0:  # Save every 'imaging_period' number of steps
                    dWaveOp.append(solver.compute_dWaveOp('time', solver_data))

            # When k is the nth step, the next step is uneeded, so don't swap
            # any values.  This way, uk at the end is always the final step
            if(k == (nsteps-1)):
                break

            # Don't know what data is needed for the solver, so the solver data
            # handles advancing everything forward by one time step.
            # k-1 <-- k, k <-- k+1, etc
            solver_data.advance()

        retval = dict()

        if 'wavefield' in return_parameters:
            retval['wavefield'] = us
        if 'dWaveOp' in return_parameters:
            retval['dWaveOp'] = dWaveOp
        if 'simdata' in return_parameters:
            retval['simdata'] = simdata

        return retval

    def virtual_source(self, operand_simdata, perturbation_model,dt):
        #### REMEMBER COPY!!! OTHERWISE, ORIGINAL WAVEFIELD IS ERASED!!!
        dt2 = dt*dt
        nt = len(operand_simdata[:,0])
        totaln = len(operand_simdata[0,:])
        # print('size of data is {}x{}'.format(nt,totaln))

        # 1. second-order time-derivatives of background wavefield (operand_simdata)
        temp_wavefield = np.zeros((2,len(operand_simdata[0,:]))) # dim = [2, nx*nz]
        temp_wavefield[0,:] = operand_simdata[0,:]
        temp_wavefield[1,:] = operand_simdata[1,:]
        

        for t in range(1, nt - 1):
            operand_simdata[t,:] = temp_wavefield[0,:] - 2*temp_wavefield[1,:] + operand_simdata[t+1,:]
            temp_wavefield[0,:] = temp_wavefield[1,:]
            temp_wavefield[1,:] = operand_simdata[t+1,:]

        operand_simdata[0,:] = 0
        
        # operand_simdata = np.diff(operand_simdata, n = 2, axis = 0)
        # 2. get virtual source
        for i in range(totaln):
            operand_simdata[:,i] = -np.multiply(operand_simdata[:,i],perturbation_model[i])

        return operand_simdata/dt2 

    def _setup_forward_rhs_with_2D_source(self, rhs_array, k, input_source):
        # basic rhs is always the pseudodata or residual
        rhs_array = self.solver.mesh.pad_array(input_source[k,:], out_array=rhs_array)
        return rhs_array       

    def forward_model_with_2D_source(self, input_source, m0, imaging_period=1, return_parameters=[]):
        """Applies the forward model to the model for the given solver.

        Parameters
        ----------
        shot : pysit.Shot
            Gives the source signal approximation for the right hand side.
        m0 : solver.ModelParameters
            The parameters upon which to evaluate the forward model.
        return_parameters : list of {'wavefield', 'simdata', 'dWaveOp'}

        Returns
        -------
        retval : dict
            Dictionary whose keys are return_parameters that contains the specified data.

        Notes
        -----
        * u is used as the target field universally.  It could be velocity potential, it could be displacement, it could be pressure.
        * utt is used to generically refer to the derivative of u that is needed to compute the imaging condition.

        Forward Model solves:

        For constant density: m*u_tt - lap u = f, where m = 1.0/c**2
        For variable density: m1*u_tt - div(m2 grad)u = f, where m1=1.0/kappa, m2=1.0/rho, and C = (kappa/rho)**0.5
        """

        # Local references
        solver = self.solver
        solver.model_parameters = m0

        mesh = solver.mesh

        d = solver.domain
        dt = solver.dt
        nsteps = len(input_source)
        # print('nsteps=',nsteps)
        # Storage for the field
        if 'wavefield' in return_parameters:
            us = list()

        # Storage for the time derivatives of p
        if 'dWaveOp' in return_parameters:
            dWaveOp = list()

        # Step k = 0
        # p_0 is a zero array because if we assume the input signal is causal
        # and we assume that the initial system (i.e., p_(-2) and p_(-1)) is
        # uniformly zero, then the leapfrog scheme would compute that p_0 = 0 as
        # well. ukm1 is needed to compute the temporal derivative.
        solver_data = solver.SolverData()

        rhs_k = np.zeros(mesh.shape(include_bc=True))
        rhs_kp1 = np.zeros(mesh.shape(include_bc=True))

        for k in range(nsteps-1):  # xrange(int(solver.nsteps)):

            uk = solver_data.k.primary_wavefield
            uk_bulk = mesh.unpad_array(uk)

            if 'wavefield' in return_parameters:
                ## Save every 'imaging_period' number of steps
                if k % imaging_period == 0:  
                    us.append(uk_bulk.copy())


            if k == 0:
                rhs_k = self._setup_forward_rhs_with_2D_source(
                    rhs_k, k,   input_source)
                rhs_kp1 = self._setup_forward_rhs_with_2D_source(
                    rhs_kp1,  k+1, input_source)
            else:
                # shift time forward
                rhs_k, rhs_kp1 = rhs_kp1, rhs_k
            rhs_kp1 = self._setup_forward_rhs_with_2D_source(
                rhs_kp1, k+1, input_source)

            # Note, we compute result for k+1 even when k == nsteps-1.  We need
            # it for the time derivative at k=nsteps-1.
            solver.time_step(solver_data, rhs_k, rhs_kp1)


            # When k is the nth step, the next step is uneeded, so don't swap
            # any values.  This way, uk at the end is always the final step
            if(k == nsteps - 1):
                break

            # Don't know what data is needed for the solver, so the solver data
            # handles advancing everything forward by one time step.
            # k-1 <-- k, k <-- k+1, etc
            solver_data.advance()

        if 'wavefield' in return_parameters:
            k += 1
            if k % imaging_period == 0: 
                us.append(np.empty_like(us[-1]))

        retval = dict()

        if 'wavefield' in return_parameters:
            retval['wavefield'] = us
        return retval

    def forward_model_with_2D_SC(self, input_source, m0, depth, nx, nz, n_pml, return_parameters=[]):
        #This function is the same as the forward_model_with_2D_source, except more input is given
        """Applies the forward model to the model for the given solver.

        Parameters
        ----------
        shot : pysit.Shot
            Gives the source signal approximation for the right hand side.
        m0 : solver.ModelParameters
            The parameters upon which to evaluate the forward model.
        return_parameters : list of {'wavefield', 'simdata', 'dWaveOp'}

        Returns
        -------
        retval : dict
            Dictionary whose keys are return_parameters that contains the specified data.

        Notes
        -----
        * u is used as the target field universally.  It could be velocity potential, it could be displacement, it could be pressure.
        * utt is used to generically refer to the derivative of u that is needed to compute the imaging condition.

        Forward Model solves:

        For constant density: m*u_tt - lap u = f, where m = 1.0/c**2
        For variable density: m1*u_tt - div(m2 grad)u = f, where m1=1.0/kappa, m2=1.0/rho, and C = (kappa/rho)**0.5
        """

        

        # Local references
        solver = self.solver
        solver.model_parameters = m0

        mesh = solver.mesh

        d = solver.domain
        dt = solver.dt

        nsteps = len(input_source)
        # reshape source:
        input_source2 = np.zeros((nsteps,nx,nz))
        input_source2[:,:,depth] = input_source.copy()
        input_source2.shape =nsteps,nx*nz
        input_source = input_source2

        # print('nsteps=',nsteps)
        # Storage for the field
        if 'wavefield' in return_parameters:
            us = list()

        # Storage for the time derivatives of p
        if 'dWaveOp' in return_parameters:
            dWaveOp = list()

        # Step k = 0
        # p_0 is a zero array because if we assume the input signal is causal
        # and we assume that the initial system (i.e., p_(-2) and p_(-1)) is
        # uniformly zero, then the leapfrog scheme would compute that p_0 = 0 as
        # well. ukm1 is needed to compute the temporal derivative.
        solver_data = solver.SolverData()

        rhs_k = np.zeros(mesh.shape(include_bc=True))
        rhs_kp1 = np.zeros(mesh.shape(include_bc=True))

        for k in range(nsteps-1):  # xrange(int(solver.nsteps)):

            uk = solver_data.k.primary_wavefield
            uk_bulk = mesh.unpad_array(uk)

            if 'wavefield' in return_parameters:
                us.append(uk_bulk.copy())

            if k == 0:
                rhs_k = self._setup_forward_rhs_with_2D_source(
                    rhs_k, k,   input_source)
                rhs_kp1 = self._setup_forward_rhs_with_2D_source(
                    rhs_kp1,  k+1, input_source)
            else:
                # shift time forward
                rhs_k, rhs_kp1 = rhs_kp1, rhs_k
                
            rhs_k.shape = -1,1
            
            rhs_kp1 = self._setup_forward_rhs_with_2D_source(
                rhs_kp1, k+1, input_source)

            # Note, we compute result for k+1 even when k == nsteps-1.  We need
            # it for the time derivative at k=nsteps-1.
            solver.time_step(solver_data, rhs_k, rhs_kp1)


            # When k is the nth step, the next step is uneeded, so don't swap
            # any values.  This way, uk at the end is always the final step
            if(k == nsteps - 1):
                break

            # Don't know what data is needed for the solver, so the solver data
            # handles advancing everything forward by one time step.
            # k-1 <-- k, k <-- k+1, etc
            solver_data.advance()

        if 'wavefield' in return_parameters:
            us.append(np.empty_like(us[-1]))

        retval = dict()

        if 'wavefield' in return_parameters:
            retval['wavefield'] = us
        return retval

    def forward_model_with_2D_BC(self, input_data, m0, depth, nx, nz, n_pml, return_parameters=[]):
        """Applies the forward model to the model for the given solver.

        Parameters
        ----------
        shot : pysit.Shot
            Gives the source signal approximation for the right hand side.
        m0 : solver.ModelParameters
            The parameters upon which to evaluate the forward model.
        return_parameters : list of {'wavefield', 'simdata', 'dWaveOp'}

        Returns
        -------
        retval : dict
            Dictionary whose keys are return_parameters that contains the specified data.

        Notes
        -----
        * u is used as the target field universally.  It could be velocity potential, it could be displacement, it could be pressure.
        * utt is used to generically refer to the derivative of u that is needed to compute the imaging condition.

        Forward Model solves:

        For constant density: m*u_tt - lap u = f, where m = 1.0/c**2
        For variable density: m1*u_tt - div(m2 grad)u = f, where m1=1.0/kappa, m2=1.0/rho, and C = (kappa/rho)**0.5
        """

        # Local references
        solver = self.solver
        solver.model_parameters = m0

        mesh = solver.mesh

        d = solver.domain
        dt = solver.dt
        nsteps = len(input_data)
        # print('nsteps=',nsteps)
        # Storage for the field
        if 'wavefield' in return_parameters:
            us = list()

        # Storage for the time derivatives of p
        if 'dWaveOp' in return_parameters:
            dWaveOp = list()

        # Step k = 0
        # p_0 is a zero array because if we assume the input signal is causal
        # and we assume that the initial system (i.e., p_(-2) and p_(-1)) is
        # uniformly zero, then the leapfrog scheme would compute that p_0 = 0 as
        # well. ukm1 is needed to compute the temporal derivative.
        solver_data = solver.SolverData()

        rhs_k = np.zeros(mesh.shape(include_bc=True))
        rhs_kp1 = np.zeros(mesh.shape(include_bc=True))

        for k in range(nsteps-1):  # xrange(int(solver.nsteps)):

            # 1.First get unpad array to insert data as boundary condition:
            orig_u = solver_data.k.primary_wavefield.copy()
            temp_u = mesh.unpad_array(orig_u)
            temp_u.shape = nx, nz
            orig_u.shape = nx+2*n_pml,nz+2*n_pml

            # 2.Then insert data as boundary condition at correct depth in unpad array:
            temp_u[:,depth] = input_data[k,:]

            # 3.Copy unpad array to the copy of the original pad array orig_u
            # notice that only copy of solver_data.k.primary_wavefield can be
            # reshaped, so this copy orig_u is very important and necessary!
            orig_u[n_pml:n_pml+nx,n_pml:n_pml+nz] = temp_u

            # if k % 100 == 0 :
            #     plt.imshow(orig_u,aspect='auto')
            #     plt.set_cmap('seismic')
            #     plt.clim(-0.001,0.001)
            #     plt.show()
            
            # 4.Assign solver_data.k.primary_wavefield the orig_u (both have same shape)
            orig_u.shape = -1, 1
            solver_data.k.primary_wavefield = orig_u 

            uk = solver_data.k.primary_wavefield
            uk_bulk = mesh.unpad_array(uk)

            if 'wavefield' in return_parameters:
                us.append(uk_bulk.copy())

            # Note, we compute result for k+1 even when k == nsteps-1.  We need
            # it for the time derivative at k=nsteps-1.
            solver.time_step(solver_data, rhs_k, rhs_kp1)


            # When k is the nth step, the next step is uneeded, so don't swap
            # any values.  This way, uk at the end is always the final step
            if(k == nsteps - 1):
                break

            # Don't know what data is needed for the solver, so the solver data
            # handles advancing everything forward by one time step.
            # k-1 <-- k, k <-- k+1, etc
            solver_data.advance()

        if 'wavefield' in return_parameters:
            us.append(np.empty_like(us[-1]))

        retval = dict()

        if 'wavefield' in return_parameters:
            retval['wavefield'] = us
        return retval