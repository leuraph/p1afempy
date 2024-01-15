import unittest
import numpy as np
import p1afempy.mesh as mesh
from p1afempy.solvers import solve_laplace
from pathlib import Path


OMEGA = 7./4. * np.pi
N_REFINEMENTS = 3


def u(r: np.ndarray) -> float:
    """analytical solution"""
    return np.sin(OMEGA*2.*r[:, 0])*np.sin(OMEGA*r[:, 1])


def f(r: np.ndarray) -> float:
    """volume force corresponding to analytical solution"""
    return 5. * OMEGA**2 * np.sin(OMEGA*2.*r[:, 0]) * np.sin(OMEGA*r[:, 1])


def uD(r: np.ndarray) -> float:
    """solution value on the Dirichlet boundary"""
    return u(r)


def g_right(r: np.ndarray) -> float:
    return -2.*OMEGA*np.sin(OMEGA*r[:, 1])*np.cos(OMEGA*2.*r[:, 0])


def g_upper(r: np.ndarray) -> float:
    return OMEGA*np.sin(OMEGA*2.*r[:, 0]) * np.cos(OMEGA*r[:, 1])


def g(r: np.ndarray) -> float:
    out = np.zeros(r.shape[0])
    right_indices = r[:, 0] == 1
    upper_indices = r[:, 1] == 1
    out[right_indices] = g_right(r[right_indices])
    out[upper_indices] = g_upper(r[upper_indices])
    return out


class SolverTest(unittest.TestCase):

    def test_solve_laplace(self) -> None:
        path_to_coordinates = Path(
            'tests/data/laplace_example/coordinates.dat')
        path_to_elements = Path('tests/data/laplace_example/elements.dat')
        path_to_neumann = Path('tests/data/laplace_example/neumann.dat')
        path_to_dirichlet = Path('tests/data/laplace_example/dirichlet.dat')
        path_to_matlab_x = Path('tests/data/laplace_example/x.dat')
        path_to_matlab_energy = Path('tests/data/laplace_example/energy.dat')

        square_mesh = mesh.read_mesh(path_to_coordinates=path_to_coordinates,
                                     path_to_elements=path_to_elements)
        neumann_bc = mesh.read_boundary_condition(
            path_to_boundary=path_to_neumann)
        dirichlet_bc = mesh.read_boundary_condition(
            path_to_boundary=path_to_dirichlet)
        boundary_conditions = [dirichlet_bc, neumann_bc]

        for _ in range(N_REFINEMENTS):
            marked_elements = np.arange(square_mesh.elements.shape[0])
            square_mesh, boundary_conditions = mesh.refineNVB(
                mesh=square_mesh,
                marked_elements=marked_elements,
                boundary_conditions=boundary_conditions)

        x, energy = solve_laplace(
            mesh=square_mesh,
            dirichlet=boundary_conditions[0],
            neumann=boundary_conditions[1],
            f=f, g=g, uD=uD)

        x_matlab = np.loadtxt(path_to_matlab_x)
        energy_matlab = np.loadtxt(path_to_matlab_energy).reshape((1,))[0]

        self.assertTrue(np.allclose(x, x_matlab))
        self.assertTrue(np.isclose(energy, energy_matlab))


if __name__ == "__main__":
    unittest.main()
