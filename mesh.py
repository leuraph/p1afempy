import numpy as np
from pathlib import Path
from scipy.sparse import coo_matrix, find


class Mesh:

    coordinates: np.ndarray
    elements: np.ndarray

    def __init__(self, coordinates: np.ndarray, elements: np.ndarray) -> None:
        self.coordinates = coordinates
        self.elements = elements


def read_mesh(path_to_coordinates: Path, path_to_elements: Path) -> Mesh:
    coordinates = np.loadtxt(path_to_coordinates)
    elements = np.loadtxt(path_to_elements, dtype=int)
    return Mesh(coordinates=coordinates, elements=elements)


def provide_geometric_data(domain: Mesh, *boundaries):
    n_elements = domain.elements.shape[0]
    n_boundaries = len(boundaries)

    # Extracting all directed edges E_l:=(I[l], J[l]) (interior edges appear twice)
    I = domain.elements.flatten()
    J = domain.elements[:, [1, 2, 0]].flatten()

    # Symmetrize I and J (so far boundary edges appear only once)
    pointer = np.concatenate(([0, 3.*n_elements-1], 
                              np.zeros(n_boundaries)))
    for k, boundary in enumerate(boundaries):
        if boundary.size:
            I = np.concatenate(I, boundary[:, 1])
            J = np.concatenate(J, boundary[:, 0])
        pointer[k+2] = pointer[k+1] + boundary.shape[0]

    # Fixing an edge number for all edges, where i<j
    idx_IJ = np.where(I < J)[0]
    n_unique_edges = idx_IJ.size
    edge_number = np.zeros(I.size)
    edge_number[idx_IJ] = np.arange(n_unique_edges)

    # Ensuring the same numbering for all edges, where j<i
    idx_JI = np.where(J < I)[0]
    number_to_edges = coo_matrix((np.arange(n_unique_edges), (I[idx_IJ],J[idx_IJ])))
    _, _, numbering_IJ = find(number_to_edges)
    _, _, idx_JI2IJ = find(coo_matrix((idx_JI, (J(idx_JI), I(idx_JI)))))
    edge_number[idx_JI2IJ] = numbering_IJ
    
    element2edges = edge_number[0:3*n_elements].reshape(n_elements,3)
    edge2nodes = np.column_stack((I[idx_IJ], J[idx_IJ]))
    # Provide boundary2edges
    boundarie_to_edges = []
    for j in np.arange(n_boundaries):
        boundarie_to_edges.append(edge_number[np.arange(pointer[j+1]+1,pointer[j+2])])
    return element2edges, edge2nodes, boundarie_to_edges