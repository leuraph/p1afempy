import p1afempy.mesh as mesh
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
from typing import Callable


def get_stiffness_matrix(mesh: mesh.Mesh):
    """
    returns the stiffness matrix for the P1 FEM
    with Legendre basis

    parameters
    ----------
    mesh: mesh.Mesh
        the mesh used for the construction of the
        stiffness matrix

    returns
    -------
    scipy.sparse.coo_matrix: the sparse stiffness matrix
    """
    c1 = mesh.coordinates[mesh.elements[:, 0], :]
    d21 = mesh.coordinates[mesh.elements[:, 1], :] - c1
    d31 = mesh.coordinates[mesh.elements[:, 2], :] - c1

    # vector of element areas 4*|T|
    area4 = 2 * (d21[:, 0]*d31[:, 1] - d21[:, 1] * d31[:, 0])

    I = (mesh.elements[:, [0, 1, 2, 0, 1, 2, 0, 1, 2]].T).flatten(order='F')
    J = (mesh.elements[:, [0, 0, 0, 1, 1, 1, 2, 2, 2]].T).flatten(order='F')
    a = (np.sum(d21*d31, axis=1)/area4)
    b = (np.sum(d31*d31, axis=1)/area4)
    c = (np.sum(d21*d21, axis=1)/area4)
    A = np.vstack([-2.*a+b+c, a-b, a-c, a-b, b, -a, a-c, -a, c])
    return coo_matrix((A.flatten(order='F'), (I, J)))


def get_right_hand_side(mesh: mesh.Mesh,
                        f: Callable[[np.ndarray], float]):
    c1 = mesh.coordinates[mesh.elements[:, 0], :]
    d21 = mesh.coordinates[mesh.elements[:, 1], :] - c1
    d31 = mesh.coordinates[mesh.elements[:, 2], :] - c1

    # vector of element areas 4*|T|
    area4 = 2 * (d21[:, 0]*d31[:, 1] - d21[:, 1] * d31[:, 0])
    # assembly of right-hand side
    fsT = np.apply_along_axis(f, 1, c1+(d21+d31) / 3)
    # TODO pad the result of np.bincount to the same size as `x`,
    # i.e. add zeros, if necessary
    b = np.bincount(
        mesh.elements.flatten(order='F'),
        weights=np.tile(area4*fsT/12., (3, 1)).flatten())
    return b


def solve_laplace(mesh: mesh.Mesh,
                  dirichlet: mesh.BoundaryCondition,
                  neumann: mesh.BoundaryCondition,
                  f: Callable[[np.ndarray], float],
                  g: Callable[[np.ndarray], float],
                  uD: Callable[[np.ndarray], float]
                  ) -> tuple[np.ndarray, float]:
    # n_elements = mesh.elements.shape[0] unused in python context
    n_coordinates = mesh.coordinates.shape[0]
    x = np.zeros(n_coordinates)

    # first vertex of elements and corresponding edge vectors
    c1 = mesh.coordinates[mesh.elements[:, 0], :]
    d21 = mesh.coordinates[mesh.elements[:, 1], :] - c1
    d31 = mesh.coordinates[mesh.elements[:, 2], :] - c1

    # vector of element areas 4*|T|
    area4 = 2 * (d21[:, 0]*d31[:, 1] - d21[:, 1] * d31[:, 0])

    I = (mesh.elements[:, [0, 1, 2, 0, 1, 2, 0, 1, 2]].T).flatten(order='F')
    J = (mesh.elements[:, [0, 0, 0, 1, 1, 1, 2, 2, 2]].T).flatten(order='F')
    a = (np.sum(d21*d31, axis=1)/area4)
    b = (np.sum(d31*d31, axis=1)/area4)
    c = (np.sum(d21*d21, axis=1)/area4)
    A = np.vstack([-2.*a+b+c, a-b, a-c, a-b, b, -a, a-c, -a, c])
    A = coo_matrix((A.flatten(order='F'), (I, J)))

    # prescribe values at dirichlet nodes
    unique_dirichlet = np.unique(dirichlet.boundary)
    x[unique_dirichlet] = np.apply_along_axis(
        uD, 1, mesh.coordinates[unique_dirichlet, :])

    # assembly of right-hand side
    fsT = np.apply_along_axis(f, 1, c1+(d21+d31) / 3)
    # TODO pad the result of np.bincount to the same size as `x`,
    # i.e. add zeros, if necessary
    b = np.bincount(
        mesh.elements.flatten(order='F'),
        weights=np.tile(area4*fsT/12., (3, 1)).flatten()) - A.dot(x)
    if neumann.boundary.size > 0:
        cn1 = mesh.coordinates[neumann.boundary[:, 0], :]
        cn2 = mesh.coordinates[neumann.boundary[:, 1], :]
        gmE = np.apply_along_axis(g, 1, (cn1+cn2)/2)
        # TODO pad if necessary
        b = b + np.bincount(
            neumann.boundary.flatten(order='F'),
            weights=np.tile(
                np.sqrt(np.sum(np.square(cn2-cn1), axis=1))*gmE/2.,
                (2, 1)).flatten())

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
