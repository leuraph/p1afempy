import unittest
import numpy as np
import p1afempy.mesh as mesh
from p1afempy.solvers import solve_laplace
from pathlib import Path

# TODO change the functions to be applied to all coordinates

# TODO split this performance test in refining/solving


def u(r: np.ndarray, omega: float = 7./4. * np.pi) -> float:
    """analytical solution"""
    x, y = r[0], r[1]
    return np.sin(omega*2.*x)*np.sin(omega*y)


def f(r: np.ndarray, omega: float = 7./4. * np.pi) -> float:
    """volume force corresponding to analytical solution"""
    x, y = r[0], r[1]
    return 5. * omega**2 * np.sin(omega*2.*x) * np.sin(omega*y)


def uD(r: np.ndarray, omega: float = 7./4. * np.pi) -> float:
    """solution value on the Dirichlet boundary"""
    return u(r, omega=omega)


def g_right(r: np.ndarray, omega: float = 7./4. * np.pi) -> float:
    x, y = r[0], r[1]
    return -2.*omega*np.sin(omega*y)*np.cos(omega*2.*x)


def g_upper(r: np.ndarray, omega: float = 7./4. * np.pi) -> float:
    x, y = r[0], r[1]
    return omega*np.sin(omega*2.*x) * np.cos(omega*y)


def g(r: np.ndarray, omega: float = 7./4. * np.pi) -> float:
    if r[0] == 1.:
        return g_right(r, omega=omega)
    elif r[1] == 1.:
        return g_upper(r, omega=omega)
    else:
        raise RuntimeError("Non-boundary evaluation of g")


def main() -> None:
    
    path_to_coordinates = Path(
        'tests/data/laplace_example/coordinates.dat')
    path_to_elements = Path('tests/data/laplace_example/elements.dat')
    path_to_neumann = Path('tests/data/laplace_example/neumann.dat')
    path_to_dirichlet = Path('tests/data/laplace_example/dirichlet.dat')

    square_mesh = mesh.read_mesh(path_to_coordinates=path_to_coordinates,
                                 path_to_elements=path_to_elements)
    neumann_bc = mesh.read_boundary_condition(
        path_to_boundary=path_to_neumann)
    dirichlet_bc = mesh.read_boundary_condition(
        path_to_boundary=path_to_dirichlet)
    boundary_conditions = [dirichlet_bc, neumann_bc]

    n_refinements = 9
    for _ in range(n_refinements):
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


if __name__ == "__main__":
    main()
