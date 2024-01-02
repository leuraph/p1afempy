import mesh
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def solve_laplace(mesh: mesh.Mesh,
                  dirichlet: mesh.BoundaryCondition,
                  neumann: mesh.BoundaryCondition,
                  f: callable,
                  g: callable,
                  uD: callable) -> tuple[np.ndarray, float]:
    n_elements = mesh.elements.shape[0]
    n_coordinates = mesh.coordinates.shape[0]
    x = np.zeros(n_elements)

    # first vertex of elements and corresponding edge vectors
    c1 = mesh.coordinates[mesh.elements[:, 0], :]
    d21 = mesh.coordinates[mesh.elements[:, 1], :] - c1
    d31 = mesh.coordinates[mesh.elements[:, 2], :] - c1

    # vector of element areas 4*|T|
    area4 = 2 * (d21[:, 0]*d31[:, 1] - d21[:, 1] * d31[:, 0])

    # assembly of stiffness matrix
    I = np.reshape(mesh.elements[:, [0, 1, 2, 0, 1, 2, 0, 1, 2]].T,
                   (9*n_elements, 1), order='F')
    J = np.reshape(mesh.elements[:, [0, 0, 0, 1, 1, 1, 2, 2, 2]].T,
                   (9*n_elements, 1), order='F')
    a = (np.sum(d21*d31, axis=1)/area4)
    b = (np.sum(d31*d31, axis=1)/area4)
    c = (np.sum(d21*d21, axis=1)/area4)
    A = np.vstack([-2.*a+b+c, a-b, a-c, a-b, b, -a, a-c, -a, c])
    A = coo_matrix((A.flatten(order='F'), (I, J)))

    # prescribe values at dirichlet nodes
    unique_dirichlet = np.unique(dirichlet.boundary)
    x[unique_dirichlet] = np.apply_along_axis(
        uD, 1, mesh.coordinates[dirichlet.boundary, :])

    # assembly of right-hand side
    fsT = np.apply_along_axis(f, 1, c1+(d21+d31) / 3)
    b = np.bincount(
        mesh.elements.flatten(order='F'),
        weights=np.tile(area4*fsT/12., (3, 1))) - A.dot(x)
    if neumann.boundary.size > 0:
        cn1 = mesh.coordinates[neumann.boundary[:, 1], :]
        cn2 = mesh.coordinates[neumann.boundary[:, 2], :]
        gmE = np.apply_along_axis(g, 1, (cn1+cn2)/2)
        b = b + np.bincount(
            neumann.boundary.flatten(order='F'),
            weights=np.tile(
                np.sqrt(np.sum(np.square(cn2-cn1), axis=1))*gmE/2.,
                (2, 1)))

    # computation of P1-FEM approximation
    freenodes = np.setdiff1d(
        np.arange(0, n_coordinates), unique_dirichlet, assume_unique=True)
    x[freenodes] = spsolve(A[freenodes, freenodes], b[freenodes])
    # compute energy || grad(uh) ||^2 of discrete solution
    energy = x.dot(A.dot(x))

    return x, energy
