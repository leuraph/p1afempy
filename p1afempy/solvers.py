import p1afempy.mesh as mesh
from p1afempy.mesh import get_directional_vectors, get_area
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
from typing import Callable


def get_stiffness_matrix(coordinates: np.ndarray,
                         elements: np.ndarray) -> coo_matrix:
    """
    returns the stiffness matrix for the P1 FEM
    with Legendre basis

    parameters
    ----------
    coordinates: np.ndarray
    elements: np.ndarray

    returns
    -------
    scipy.sparse.coo_matrix: the sparse stiffness matrix
    """
    # vector of element areas 4*|T|
    area4 = 4. * get_area(coordinates=coordinates,
                          elements=elements)

    I = (elements[:, [0, 1, 2, 0, 1, 2, 0, 1, 2]].T).flatten(order='F')
    J = (elements[:, [0, 0, 0, 1, 1, 1, 2, 2, 2]].T).flatten(order='F')

    d21, d31 = get_directional_vectors(coordinates=coordinates,
                                       elements=elements)
    a = (np.sum(d21*d31, axis=1)/area4)
    b = (np.sum(d31*d31, axis=1)/area4)
    c = (np.sum(d21*d21, axis=1)/area4)

    A = np.vstack([-2.*a+b+c, a-b, a-c, a-b, b, -a, a-c, -a, c])
    return coo_matrix((A.flatten(order='F'), (I, J)))


def get_mass_matrix(mesh: mesh.Mesh) -> coo_matrix:
    """
    returns the mass matrix of the mesh provided
    for the P1 FEM with Legendre basis
    """
    I, J, D = get_mass_matrix_elements(mesh=mesh)
    return coo_matrix((D, (I, J)))


def get_mass_matrix_elements(
        mesh: mesh.Mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    returns the mass matrix of the mesh provided
    for the P1 FEM with Legendre basis

    returns
    -------
    I: np.ndarray
    J: np.ndarray
    D: np.ndarray
        D[m] represents a mass matrix contribution
        belonging to its (I[m], J[m]) coordinate
    """
    I = (mesh.elements[:, [0, 1, 2, 0, 1, 2, 0, 1, 2]].T).flatten(order='F')
    J = (mesh.elements[:, [0, 0, 0, 1, 1, 1, 2, 2, 2]].T).flatten(order='F')

    area = get_area(coordinates=mesh.coordinates,
                    elements=mesh.elements)
    D = np.vstack(
        [area/6., area/12., area/12.,
         area/12., area/6., area/12.,
         area/12., area/12., area/6.]).flatten(order='F')
    return I, J, D


def get_right_hand_side(mesh: mesh.Mesh,
                        f: Callable[[np.ndarray], float]):
    """
    returns the load vector for the P1 FEM with Legendre basis

    parameters
    ----------
    mesh: mesh.Mesh
        the mesh which to evaluate the load vector on
    f: Callable[[np.ndarray], float]
        the function for which to evaluate the load vector

    returns
    -------
    b: np.ndarray
        the P1 FEM load vector of f on the mesh at hand

    notes
    -----
    the load vector F_i := int f(x)phi_i(x) dx
    is approximated as sum_T |T| * f(sT) * phi_i(sT),
    where sT denotes the center of mass of triangle T
    """
    # vector of element areas 4*|T|
    area4 = 4. * get_area(coordinates=mesh.coordinates,
                          elements=mesh.elements)

    # assembly of right-hand side
    d21, d31 = get_directional_vectors(coordinates=mesh.coordinates,
                                       elements=mesh.elements)
    fsT = f((mesh.coordinates[mesh.elements[:, 0], :]+(d21+d31) / 3))
    b = np.bincount(
        mesh.elements.flatten(order='F'),
        weights=np.tile(area4*fsT/12., (3, 1)).flatten(),
        minlength=mesh.coordinates.shape[0])
    return b


def apply_neumann(neumann_bc: mesh.BoundaryCondition,
                  mesh: mesh.Mesh,
                  g: Callable[[np.ndarray], float],
                  b: np.ndarray):
    """
    applies neuman boundary conditions to b and returns new b
    """
    # TODO channge b in place, do not return it or
    # at least check if this version generates computational overhead
    cn1 = mesh.coordinates[neumann_bc.boundary[:, 0], :]
    cn2 = mesh.coordinates[neumann_bc.boundary[:, 1], :]
    gmE = g((cn1+cn2)/2)
    b = b + np.bincount(
        neumann_bc.boundary.flatten(order='F'),
        weights=np.tile(
            np.sqrt(np.sum(np.square(cn2-cn1), axis=1))*gmE/2.,
            (2, 1)).flatten(), minlength=b.size)
    return b


def solve_laplace(mesh: mesh.Mesh,
                  dirichlet: mesh.BoundaryCondition,
                  neumann: mesh.BoundaryCondition,
                  f: Callable[[np.ndarray], float],
                  g: Callable[[np.ndarray], float],
                  uD: Callable[[np.ndarray], float]
                  ) -> tuple[np.ndarray, float]:
    """
    solves the laplace equation, i.e.

    -Delta u = f, on Omega
    u = uD, on Gamma_D
    du/dn = g, on Gamma_N

    on the provided mesh using P1 FEM with Legendre basis

    parameters
    ----------
    mesh: mesh.Mesh
        the mesh on which to calculate the solution on
    dirichlet: mesh.BoundaryCondition
        the dirichlet boundary of the problem
    neumann: mesh.BoundaryCondition
        the neumann boundary of the problem
    f: Callable[[np.ndarray], float]
        the right-hand-side function (volume force) of the problem
    g: Callable[[np.ndarray], float]
        the neumann boundary function, i.e.
        u(x) = g(x) on Gamma_N
    uD: Callable[[np.ndarray], float]
        the dirichlet boundary function, i.e.
        du/dn(x) = uD(x) on Gamma_D

    returns
    -------
    x: np.ndarray
        the solution of the given laplace problem
        on the mesh provided
    energy: float
        the energy (A-norm: x.T*A*x) of the solution found

    notes
    -----
    the functions f, g, and uD are all expected to be callable like
    f(coordinates), where coordinates is an (n_coordinates x 2) array
    """
    n_coordinates = mesh.coordinates.shape[0]
    x = np.zeros(n_coordinates)

    A = get_stiffness_matrix(coordinates=mesh.coordinates,
                             elements=mesh.elements)

    # prescribe values at dirichlet nodes
    unique_dirichlet = np.unique(dirichlet.boundary)
    x[unique_dirichlet] = uD((mesh.coordinates[unique_dirichlet, :]))

    b = get_right_hand_side(mesh=mesh, f=f) - A.dot(x)
    if neumann.boundary.size > 0:
        b = apply_neumann(neumann_bc=neumann, mesh=mesh,
                          g=g, b=b)

    # computation of P1-FEM approximation
    freenodes = np.setdiff1d(
        np.arange(0, n_coordinates), unique_dirichlet, assume_unique=True)
    A = csc_matrix(A)
    x[freenodes] = spsolve(A[freenodes, :][:, freenodes],
                           b[freenodes],
                           use_umfpack=True)
    # compute energy || grad(uh) ||^2 of discrete solution
    energy = x.dot(A.dot(x))

    return x, energy
