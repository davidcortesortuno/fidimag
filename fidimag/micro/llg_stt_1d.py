from __future__ import division

import numpy as np
import fidimag.extensions.clib as clib

from .micro_driver import MicroDriver
import fidimag.common.helper as helper
import fidimag.common.constant as const


class LLG_STT_1D(MicroDriver):

    """

    This class is the driver to solve the Landau Lifshitz Gilbert equation
    with a Spin Transfer Torque term, which has the form:


      dm        -gamma
     ---- =    --------  ( m X H_eff  + a * m X ( m x H_eff ) + ... )
      dt             2
              ( 1 + a  )

    by using the Sundials library with CVODE.

    This class inherits common methods to evolve the system using CVODE, from
    the micro_driver.MicroDriver class. Arrays with the system information
    are taken as references from the main micromagnetic Simulation class

        """

    def __init__(self, mesh, spin, Ms, field, pins,
                 interactions,
                 name,
                 data_saver,
                 integrator='sundials',
                 use_jac=False
                 ):

        # Inherit from the driver class
        super(LLG_STT_1D, self).__init__(mesh, spin, Ms, field,
                                      pins, interactions, name,
                                      data_saver,
                                      integrator=integrator,
                                      use_jac=use_jac
                                      )

        self.first = np.ones(3 * self.n, dtype=np.float)
        self.second = np.ones(3 * self.n, dtype=np.float)
        self.field_stt = np.zeros(3 * self.n)


        self.p = 0.5
        self.beta = 0
        self.update_j_fun = None

        # FIXME: change the u0 to spatial
        self.u0 = const.g_e * const.mu_B / (2 * const.c_e)
        self.u = 0


    def sundials_rhs(self, t, y, ydot):

        self.t = t

        # already synchronized when call this funciton
        # self.spin[:]=y[:]

        self.compute_effective_field(t)

        clib.compute_derivatives_1d(self.spin, self.field_stt, self.second,
                                            self.mesh.dx*self.mesh.unit_length,
                                            self.mesh.nx, self.mesh.ny, self.mesh.nz)

        clib.compute_llg_stt_rhs(ydot,
                                 self.spin,
                                 self.field,
                                 self.field_stt,
                                 self._alpha,
                                 self.beta,
                                 -1.0*self.u, #this is due to the defintion of u
                                 self.gamma,
                                 self.n)
