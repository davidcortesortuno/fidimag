from __future__ import division
from __future__ import print_function


import os
import time
import fidimag.extensions.clib as clib
import fidimag.extensions.micro_clib as micro_clib
import numpy as np
from fidimag.common.fileio import DataSaver, DataReader
from fidimag.common.save_vtk import SaveVTK
from fidimag.common.integrators import CvodeSolver
import fidimag.common.constant as const
import fidimag.common.helper as helper
import re

class Driver(object):
    def __init__(self):
        self.t = 0
        self.step = 0

class LLG_STT_Full(object):

    def __init__(self, mesh, name='unnamed', integrator='sundials', use_jac=False):
        """Simulation object.
        *Arguments*
          name : the Simulation name (used for writing data files, for examples)
        """
        self.t = 0
        self.name = name
        self.mesh = mesh
        self.n = mesh.n
        self.n_nonzero = mesh.n
        self.unit_length = mesh.unit_length
        self._alpha = np.zeros(self.n, dtype=np.float)
        self._Ms = np.zeros(self.n, dtype=np.float)
        self._Ms_inv = np.zeros(self.n, dtype=np.float)
        self.spin = np.ones(3 * self.n, dtype=np.float)
        self.first = np.ones(3 * self.n, dtype=np.float)
        self.second = np.ones(3 * self.n, dtype=np.float)
        self.dm = np.ones(3 * self.n, dtype=np.float)
        self.spin_dm = np.zeros(6 * self.n, dtype=np.float)
        self.field = np.zeros(3 * self.n, dtype=np.float)
        self.field_stt = np.zeros(3 * self.n, dtype=np.float)
        self.field_lap = np.zeros(3 * self.n, dtype=np.float)
        self.dm_dt = np.zeros(3 * self.n, dtype=np.float)
        self._energy__ = np.zeros(self.n, dtype=np.float)
        self.interactions = []
        self.integrator_tolerances_set = False
        self.step = 0
        self.driver = Driver()

        self.dx = self.mesh.dx * mesh.unit_length
        self.dy = self.mesh.dy * mesh.unit_length
        self.dz = self.mesh.dz * mesh.unit_length

        self.integrator = CvodeSolver(self.spin_dm, self.sundials_rhs)

        self.saver = DataSaver(self, name + '.txt')

        self.saver.entities['E_total'] = {
            'unit': '<J>',
            'get': lambda sim: sim.compute_energy(),
            'header': 'E_total'}

        self.saver.entities['m_error'] = {
            'unit': '<>',
            'get': lambda sim: sim.compute_spin_error(),
            'header': 'm_error'}

        self.saver.entities['rhs_evals'] = {
            'unit': '<>',
            'get': lambda sim: self.integrator.rhs_evals(),
            'header': 'rhs_evals'}

        self.saver.entities['real_time'] = {
            'unit': '<s>',
            'get': lambda _: time.time(),  # seconds since epoch
            'header': 'real_time'}

        self.saver.update_entity_order()

        # This is for old C files codes using the xperiodic variables
        self.xperiodic, self.yperiodic, self.zperiodic = mesh.periodicity

        self.vtk = SaveVTK(self.mesh, name=name)

        self.set_default_options()
        self.set_parameters()

    def set_parameters(self, u=1,  D=2.5e-4, lambda_sf=5e-9, lambda_J=1e-9, speedup=1):

        self.D = D / speedup
        self.lambda_sf = lambda_sf
        self.lambda_J = lambda_J
        self.tau_sf = lambda_sf ** 2 / D * speedup
        self.tau_sd = lambda_J ** 2 / D * speedup
        #self.u0 = const.g_e * const.mu_B / (2 * const.c_e)
        self.u = u

    def set_default_options(self, gamma=2.21e5, Ms=8.0e5, alpha=0.1):
        self.default_c = 1e11
        self._alpha[:] = alpha
        self._Ms[:] = Ms
        self.gamma = gamma
        self.do_precession = True

    def reset_integrator(self, t=0):
        self.integrator.reset(self.spin, t)
        self.t = t # also reinitialise the simulation time and step
        self.step = 0

    def set_tols(self, rtol=1e-8, atol=1e-10, max_ord=None, reset=True):
        if max_ord is not None:
            self.integrator.set_options(rtol=rtol, atol=atol, max_ord=max_ord)
        else:
            # not all integrators have max_ord (only VODE does)
            # and we don't want to encode a default value here either
            self.integrator.set_options(rtol=rtol, atol=atol)
        if reset:
            self.reset_integrator(self.t)

    def set_m(self, m0=(1, 0, 0), normalise=True):

        self.spin[:] = helper.init_vector(m0, self.mesh, normalise)

        # TODO: carefully checking and requires to call set_mu first
        self.spin.shape = (-1, 3)
        for i in range(self.spin.shape[0]):
            if self._Ms[i] == 0:
                self.spin[i, :] = 0
        self.spin.shape = (-1,)
        self.spin_dm.shape = (2,-1)
        self.spin_dm[0,:] = self.spin[:]
        self.spin_dm.shape = (-1,)
        self.integrator.set_initial_value(self.spin_dm, self.t)


    def get_alpha(self):
        return self._alpha

    def set_alpha(self, value):
        self._alpha[:] = helper.init_scalar(value, self.mesh)

    alpha = property(get_alpha, set_alpha)

    def get_Ms(self):
        return self._Ms

    def set_Ms(self, value):
        self._Ms[:] = helper.init_scalar(value, self.mesh)
        nonzero = 0
        for i in range(self.n):
            if self._Ms[i] > 0.0:
                self._Ms_inv[i] = 1.0 / self._Ms[i]
                nonzero += 1

        self.n_nonzero = nonzero

        self.Ms_const = np.max(self._Ms)

    Ms = property(get_Ms, set_Ms)

    def add(self, interaction, save_field=False):
        interaction.setup(self.mesh, self.spin, Ms=self._Ms)

        # TODO: FIX
        for i in self.interactions:
            if i.name == interaction.name:
                interaction.name = i.name + '_2'

        self.interactions.append(interaction)

        energy_name = 'E_{0}'.format(interaction.name)
        self.saver.entities[energy_name] = {
            'unit': '<J>',
            'get': lambda sim: sim.get_interaction(interaction.name).compute_energy(),
            'header': energy_name}

        if save_field:
            fn = '{0}'.format(interaction.name)
            self.saver.entities[fn] = {
                'unit': '<>',
                'get': lambda sim: sim.get_interaction(interaction.name).average_field(),
                'header': ('%s_x' % fn, '%s_y' % fn, '%s_z' % fn)}

        self.saver.update_entity_order()

    def get_interaction(self, name):
        for interaction in self.interactions:
            if interaction.name == name:
                return interaction
        else:
            raise ValueError("Failed to find the interaction with name '{0}', "
                             "available interactions: {1}.".format(
                                 name, [x.name for x in self.interactions]))

    def run_until(self, t):
        if t <= self.t:
            if t == self.t and self.t == 0.0:
                self.compute_effective_field(t)
                self.saver.save()
            return

        flag = self.integrator.run_until(t)
        if flag < 0:
            raise Exception("Run run_until failed!!!")

        self.spin_dm[:] = self.integrator.y[:]
        self.spin_dm.shape = (2,-1)
        self.spin[:] = self.spin_dm[0,:]
        self.dm[:] = self.spin_dm[1,:]
        self.spin_dm.shape = (-1,)

        self.step += 1
        self.driver.t = t
        self.driver.step = self.step

        self.compute_effective_field(t) # update fields before saving data
        self.saver.save()

    def compute_lap_field(self):
        clib.compute_derivatives_1d(self.dm, self.first, self.field_lap,
                                    self.mesh.dx*self.mesh.unit_length,
                                    self.mesh.nx, self.mesh.ny, self.mesh.nz)


    def compute_effective_field(self, t):

        #self.spin[:] = y[:]

        self.field[:] = 0

        for obj in self.interactions:
            self.field += obj.compute_field(t)

    def compute_effective_field_jac(self, t, spin):
        self.field[:] = 0
        for obj in self.interactions:
            if obj.jac:
                self.field += obj.compute_field(t, spin=spin)

    def sundials_rhs(self, t, y, ydot):

        self.t = t

        # already synchronized when call this funciton
        # self.spin[:]=y[:]
        y.shape=(2,-1)
        self.spin[:] = y[0,:]
        self.dm[:] = y[1,:]
        y.shape = (-1,)

        self.compute_effective_field(t)
        self.compute_lap_field()

        clib.compute_derivatives_1d(self.spin, self.field_stt, self.second,
                                    self.mesh.dx*self.mesh.unit_length,
                                    self.mesh.nx, self.mesh.ny, self.mesh.nz)

        clib.compute_llg_stt_nonlocal_rhs(ydot,
                             self.spin,
                             self.dm,
                             self.field,
                             self.field_stt,
                             self.field_lap,
                             self._alpha,
                             self.D,
                             self.tau_sd,
                             self.tau_sf,
                             -1.0*self.u, #self.u0 * self.p / self.Ms_const *self.jx,
                             self.gamma,
                             self.n)
        #ydot[:] = self.dm_dt[:]

        return 0

    def compute_average(self):
        self.spin.shape = (-1, 3)
        average = np.sum(self.spin, axis=0) / self.n_nonzero
        self.spin.shape = (3 * self.n)
        return average

    def compute_energy(self):
        energy = 0
        for obj in self.interactions:
            energy += obj.compute_energy()

        return energy

    def save_vtk(self):
        self.vtk.save_vtk(self.spin.reshape(-1, 3), self.Ms, step=self.step)

    def save_m(self):
        if not os.path.exists('%s_npys' % self.name):
            os.makedirs('%s_npys' % self.name)
        name = '%s_npys/m_%g.npy' % (self.name, self.step)
        np.save(name, self.spin)

    def stat(self):
        return self.integrator.stat()

    def spin_length(self):
        self.spin.shape = (-1, 3)
        length = np.sqrt(np.sum(self.spin**2, axis=1))
        self.spin.shape = (-1,)
        return length

    def compute_spin_error(self):
        length = self.spin_length() - 1.0
        return np.max(abs(length))
