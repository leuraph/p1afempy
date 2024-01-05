import unittest
import numpy as np
import mesh
import solve_laplace
from pathlib import Path

# TODO change the functions to be applied to all coordinates


def u(r: np.ndarray) -> float:
    """analytical solution"""
    x, y = r[0], r[1]
    omega = 7./8. * np.pi
    return np.sin(omega*2.*x)*np.sin(omega*y)


def f(r: np.ndarray) -> float:
    """volume force corresponding to analytical solution"""
    x, y = r[0], r[1]
    omega = 7./8. * np.pi
    return 5. * omega**2 * np.sin(omega*2.*x) * np.sin(omega*y)


def uD(r: np.ndarray) -> float:
    """solution value on the Dirichlet boundary"""
    return u(r)


def g_right(r: np.ndarray) -> float:
    x, y = r[0], r[1]
    omega = 7./8. * np.pi
    return -2.*omega*np.sin(omega*y)*np.cos(omega*2.*x)


def g_upper(r: np.ndarray) -> float:
    x, y = r[0], r[1]
    omega = 7./8. * np.pi
    return omega*np.sin(omega*2.*x) * np.cos(omega*y)


def g(r: np.ndarray) -> float:
    if r[0] == 1.:
        return g_right(r)
    elif r[1] == 1.:
        return g_upper(r)
    else:
        raise RuntimeError("Non-boundary evaluation of g")


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

        n_refinements = 3
        for _ in range(n_refinements):
            marked_elements = np.arange(square_mesh.elements.shape[0])
            square_mesh, boundary_conditions = mesh.refineNVB(
                mesh=square_mesh,
                marked_elements=marked_elements,
                boundary_conditions=boundary_conditions)

        x, energy = solve_laplace.solve_laplace(
            mesh=square_mesh,
            dirichlet=boundary_conditions[0],
            neumann=boundary_conditions[1],
            f=f, g=g, uD=uD)

        x_matlab = np.loadtxt(path_to_matlab_x)
        energy_matlab = np.loadtxt(path_to_matlab_energy)

        self.assertTrue(np.all(x == x_matlab))
        self.assertTrue(energy == energy_matlab)


if __name__ == "__main__":
    unittest.main()
