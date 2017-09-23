import fidimag.extensions.micro_clib as micro_clib
import fidimag.extensions.clib as clib
from .energy import Energy
import numpy as np
#from constant import mu_0


class UniformExchange(Energy):

    """
    Compute the exchange field in micromagnetics.
    """

    def __init__(self, A, name='UniformExchange'):
        self.A = A
        self.name = name
        self.jac = True

    def compute_field(self, t=0, spin=None):
        if spin is not None:
            m = spin
        else:
            m = self.spin

        micro_clib.compute_exchange_field_micro(m,
                                                self.field,
                                                self.energy,
                                                self.Ms_inv,
                                                self.A,
                                                self.dx,
                                                self.dy,
                                                self.dz,
                                                self.n,
                                                self.neighbours
                                                )

        return self.field


class UniformExchange1D(Energy):

    """
    Compute the exchange field in micromagnetics.
    """

    def __init__(self, A, name='UniformExchange'):
        self.A = A
        self.name = name
        self.jac = True

    def setup(self, mesh, spin, Ms):
        super(UniformExchange1D, self).setup(mesh, spin, Ms)
        self.first = np.zeros(3*self.n)
        self.Ms_const = self.Ms[0]


    def compute_field(self, t=0, spin=None):
        if spin is not None:
            m = spin
        else:
            m = self.spin

        clib.compute_derivatives_1d(m, self.first, self.field,
                                    self.dx, self.nx, self.ny, self.nz)

        mu0 = np.pi*4e-7
        self.field[:] *= 2*self.A/(mu0*self.Ms_const)

        return self.field
