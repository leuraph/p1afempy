import numpy as np
from pathlib import Path
from scipy.sparse import coo_matrix, find
from matplotlib import pyplot as plt


def get_area(coordinates: np.ndarray,
             elements: np.ndarray) -> np.ndarray:
    """
    calculates and returns the area of each element as numpy array

    parameters
    ----------
    coordinates: np.ndarray
        vertices of the mesh
    elements: np.ndarray
        triangular elements of the mesh

    returns
    -------
    np.ndarray: the area of each element as Mx1 array
    """
    d21, d31 = get_directional_vectors(coordinates=coordinates,
                                       elements=elements)
    return 0.5 * (d21[:, 0]*d31[:, 1] - d21[:, 1] * d31[:, 0])


def get_directional_vectors(coordinates: np.ndarray,
                            elements: np.ndarray) -> tuple[np.ndarray,
                                                           np.ndarray]:
    """
    returns the vectors pointing from vertex[0] to vertex[1]
    and vertex[2], respectively, for each element

    Returns
    -------
    d21: np.ndarray (M x 2)
        d21[k, :] points from vertex 0 to vertex 1 in the k-th element
    d31: np.ndarray (M x 2)
        d31[k, :] points from vertex 0 to vertex 2 in the k-th element
    """
    c1 = coordinates[elements[:, 0], :]
    d21 = coordinates[elements[:, 1], :] - c1
    d31 = coordinates[elements[:, 2], :] - c1

    return d21, d31


def show_mesh(coordinates: np.ndarray,
              elements: np.ndarray) -> None:
    """displays the mesh at hand"""
    for element in elements:
        r0, r1, r2 = coordinates[element, :]
        plt.plot(
            [r0[0], r1[0], r2[0], r0[0]],
            [r0[1], r1[1], r2[1], r0[1]],
            'black', linewidth=0.5)
    plt.show()


def provide_geometric_data(
        elements: np.ndarray,
        boundaries: list[np.ndarray]) -> tuple[np.ndarray,
                                               np.ndarray,
                                               list[np.ndarray]]:
    """
    provides geometric data about the mesh (elements and boundaries) at hand

    Parameters
    ----------
    elements: np.ndarray
        the elements of the mesh, i.e.
        an Mx3 array where each row represents an element
    boundaries: list[np.ndarray]
        boundary conditions defined on the mesh

    Returns
    -------
    element_to_edges: np.ndarray
        element_to_edges[k] holds the edges' indices of
        the k-th element (counter-clockwise)
    edge_to_nodes: np.ndarray
        edge_to_nodes[k] holds the nodes' indices (i, j)
        of the k-th edge s.t. i < j
    boundaries_to_edges: list[np.ndarray]
        boundaries_to_edges[k] holds the mapping
        s.t. boundaries_to_edges[k][n] gives the indices
        (i, j) of the n-th edge of the k-th boundary.
    """
    n_elements = elements.shape[0]
    n_boundaries = len(boundaries)

    # Extracting all directed edges E_l:=(I[l], J[l])
    # (interior edges appear twice)
    I = elements.flatten(order='F')
    J = elements[:, [1, 2, 0]].flatten(order='F')

    # Symmetrize I and J (so far boundary edges appear only once)
    pointer = np.concatenate(([0, 3*n_elements-1],
                              np.zeros(n_boundaries, dtype=int)), dtype=int)
    for k, boundary in enumerate(boundaries):
        if boundary.size:
            I = np.concatenate((I, boundary[:, 1]), dtype=int)
            J = np.concatenate((J, boundary[:, 0]), dtype=int)
        pointer[k+2] = pointer[k+1] + boundary.shape[0]

    # Fixing an edge number for all edges, where i<j
    idx_IJ = np.where(I < J)[0]
    n_unique_edges = idx_IJ.size
    edge_number = np.zeros(I.size, dtype=int)
    edge_number[idx_IJ] = np.arange(n_unique_edges)

    # Ensuring the same numbering for all edges, where j<i
    idx_JI = np.where(J < I)[0]
    number_to_edges = coo_matrix(
        (np.arange(n_unique_edges) + 1, (I[idx_IJ], J[idx_IJ])))
    # NOTE In Matlab, the returned order is different
    _, _, numbering_IJ = find(number_to_edges)
    # NOTE In Matlab, the returned order is different
    _, _, idx_JI2IJ = find(coo_matrix((idx_JI + 1, (J[idx_JI], I[idx_JI]))))
    # NOTE Here, it coincides with Matlab again, though.
    edge_number[idx_JI2IJ - 1] = numbering_IJ - 1

    element_to_edges = edge_number[0:3*n_elements].reshape(n_elements, 3,
                                                        order='F')
    edge_to_nodes = np.column_stack((I[idx_IJ], J[idx_IJ]))
    # Provide boundary2edges
    boundaries_to_edges = []
    for j in np.arange(n_boundaries):
        boundaries_to_edges.append(
            edge_number[np.arange(pointer[j+1]+1, pointer[j+2]+1, dtype=int)])
    return element_to_edges, edge_to_nodes, boundaries_to_edges
