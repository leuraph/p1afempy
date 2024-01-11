import unittest
from p1afempy import mesh, solvers, indicators
from pathlib import Path
import numpy as np


def u(r: np.ndarray, omega: float = 7./4. * np.pi) -> float:
    """analytical solution"""
    return np.sin(omega*2.*r[:, 0])*np.sin(omega*r[:, 1])


def f(r: np.ndarray, omega: float = 7./4. * np.pi) -> float:
    """volume force corresponding to analytical solution"""
    return 5. * omega**2 * np.sin(omega*2.*r[:, 0]) * np.sin(omega*r[:, 1])


def uD(r: np.ndarray, omega: float = 7./4. * np.pi) -> float:
    """solution value on the Dirichlet boundary"""
    return u(r, omega=omega)


def g_right(r: np.ndarray, omega: float = 7./4. * np.pi) -> float:
    return -2.*omega*np.sin(omega*r[:, 1])*np.cos(omega*2.*r[:, 0])


def g_upper(r: np.ndarray, omega: float = 7./4. * np.pi) -> float:
    return omega*np.sin(omega*2.*r[:, 0]) * np.cos(omega*r[:, 1])


def g(r: np.ndarray, omega: float = 7./4. * np.pi) -> float:
    out = np.zeros(r.shape[0])
    right_indices = r[:, 0] == 1
    upper_indices = r[:, 1] == 1
    out[right_indices] = g_right(r[right_indices], omega)
    out[upper_indices] = g_upper(r[upper_indices], omega)
    return out


class IndicatorTest(unittest.TestCase):

    def test_compute_eta_r(self) -> None:
        path_to_coordinates = Path(
            'tests/data/laplace_example/coordinates.dat')
        path_to_elements = Path('tests/data/laplace_example/elements.dat')
        path_to_neumann = Path('tests/data/laplace_example/neumann.dat')
        path_to_dirichlet = Path('tests/data/laplace_example/dirichlet.dat')
        path_to_matlab_indicators = Path(
            'tests/data/laplace_example/indicators.dat')

        square_mesh = mesh.read_mesh(path_to_coordinates=path_to_coordinates,
                                     path_to_elements=path_to_elements)
        neumann_bc = mesh.read_boundary_condition(
            path_to_boundary=path_to_neumann)
        dirichlet_bc = mesh.read_boundary_condition(
            path_to_boundary=path_to_dirichlet)
        boundary_conditions = [dirichlet_bc, neumann_bc]

        n_refinements = 2
        for _ in range(n_refinements):
            marked_elements = np.arange(square_mesh.elements.shape[0])
            square_mesh, boundary_conditions = mesh.refineNVB(
                mesh=square_mesh,
                marked_elements=marked_elements,
                boundary_conditions=boundary_conditions)

        x, energy = solvers.solve_laplace(
            mesh=square_mesh,
            dirichlet=boundary_conditions[0],
            neumann=boundary_conditions[1],
            f=f, g=g, uD=uD)

        ref_indicators = indicators.compute_eta_r(
            x, square_mesh,
            dirichlet=boundary_conditions[0],
            neumann=boundary_conditions[1], f=f, g=g)

        indicators_matlab = np.loadtxt(path_to_matlab_indicators)

        self.assertTrue(np.isclose(ref_indicators, indicators_matlab))
        pass


if __name__ == '__main__':
    unittest.main()
