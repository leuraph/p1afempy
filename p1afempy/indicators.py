import numpy as np
from p1afempy.mesh import Mesh, provide_geometric_data, BoundaryCondition
from typing import Callable


def compute_eta_r(x: np.ndarray, mesh: Mesh,
                  dirichlet: BoundaryCondition,
                  neumann: BoundaryCondition,
                  f: Callable[[np.ndarray], float],
                  g: Callable[[np.ndarray], float]) -> np.ndarray:
    """
    computes residual-based error estimator for finite element
    solution of Laplace problem with mixed Dirichlet-Neumann
    boundary conditions.

    parameters
    ----------
    x: np.ndarray
        the current solution iterate
    mesh: Mesh
        the mesh on which the laplace problems should be solved
    dirichlet: BoundaryCondition
    neumann: BoundaryCondition
    f: Callable[[np.ndarray], float]
        Function correspnding to Dirichlet BC.
        Expected to be callable like
        f(np.array([[x1, y1], [x2, y2], [x3, y3]])).
    g: Callable[[np.ndarray], float]
        Function correspnding to Neumann BC.
        Expected to be callable like
        g(np.array([[x1, y1], [x2, y2], [x3, y3]])).
    returns
    -------
    etaR: np.ndarray
        the residual-based error estimator
    """
    boundary_conditions = [dirichlet, neumann]
    element2edges, edge2nodes, boundaries_to_edges = \
        provide_geometric_data(domain=mesh,
                               boundaries=boundary_conditions)

    # first vertex of elements and corresponding edge vectors
    c1 = mesh.coordinates[mesh.elements[:, 0], :]
    d21 = mesh.coordinates[mesh.elements[:, 1], :] - c1
    d31 = mesh.coordinates[mesh.elements[:, 2], :] - c1

    # vector of element volumes 2*|T|
    area2 = d21[:, 0] * d31[:, 1] - d21[:, 1] * d31[:, 0]

    # compute curl
    tmp1 = x[mesh.elements[:, 1]] - x[mesh.elements[:, 0]]
    tmp2 = x[mesh.elements[:, 2]] - x[mesh.elements[:, 0]]
    u21 = np.column_stack([tmp1, tmp1])
    u31 = np.column_stack([tmp2, tmp2])
    curl = (d31 * u21 - d21 * u31) / np.column_stack([area2, area2])

    # compute edge terms hE*(duh/dn) for uh
    dudn21 = np.sum(d21 * curl, axis=1)
    dudn13 = -np.sum(d31 * curl, axis=1)
    dudn32 = -(dudn13 + dudn21)
    etaR = np.bincount(
        element2edges.flatten(order='F'),
        np.hstack([dudn21, dudn32, dudn13]),
        minlength=edge2nodes.shape[0])

    # incorporate Neumann data
    if neumann.boundary.size > 0:
        cn1 = mesh.coordinates[neumann.boundary[:, 0], :]
        cn2 = mesh.coordinates[neumann.boundary[:, 1], :]
        gmE = g((cn1+cn2) / 2.)
        neumann2edges = boundaries_to_edges[1]
        etaR[neumann2edges] = etaR[neumann2edges] - np.sqrt(np.sum(
            np.square(cn2-cn1),
            axis=1)) * gmE

    # incorporate Dirichlet data
    dirichlet2edges = boundaries_to_edges[0]
    etaR[dirichlet2edges] = 0
    # assemble edge contributions of indicators
    etaR = np.sum(np.square(etaR[element2edges]), axis=1)
    # add volume residual to indicators
    fsT = f((c1+(d21+d31) / 3.))
    etaR = etaR + np.square(0.5 * area2 * fsT)
    return etaR