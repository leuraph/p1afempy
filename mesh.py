import numpy as np
from pathlib import Path
from scipy.sparse import coo_matrix, find


class Mesh:
    """
    A triangular mesh.

    instance variables
    ------------------
    coordinates: np.ndarray(dtype=float)
        the coordinates of the mesh's vertices.
        The k-th vertex is accessed via coordinates[k, :]
        and returns np.array([x_k, y_k]), i.e.
        its (x, y) coordinates. 
    elements: np.ndarray(dtype=int)
        the elements of the mesh, i.e. the triangles,
        where the k-th element is accessed via elements[k, :]
        and returns np.array([i, j, k]), i.e.
        the indices (starting at 0) of the corresponding 
        vertices.
    """
    coordinates: np.ndarray
    elements: np.ndarray

    def __init__(self, coordinates: np.ndarray, elements: np.ndarray) -> None:
        self.coordinates = coordinates
        self.elements = elements

#TODO add a class for boundary conditions

def read_mesh(path_to_coordinates: Path, path_to_elements: Path) -> Mesh:
    coordinates = np.loadtxt(path_to_coordinates)
    elements = np.loadtxt(path_to_elements, dtype=int)
    return Mesh(coordinates=coordinates, elements=elements)


def provide_geometric_data(domain: Mesh, *boundaries: tuple[np.ndarray]):
    """
    #TODO add complete docstring

    Returns
    -------
    element2edges: np.ndarray
        element2edges[k] holds the edges' indices of
        the k-th element (counter-clockwise)
    edge2nodes: np.ndarray
        edge2nodes[k] holds the nodes' indices (i, j)
        of the k-th edge s.t. i < j
    boundaries_to_edges: list[np.ndarray]
        #TODO describe...
    """
    n_elements = domain.elements.shape[0]
    n_boundaries = len(boundaries)

    # Extracting all directed edges E_l:=(I[l], J[l])
    # (interior edges appear twice)
    I = domain.elements.flatten()
    J = domain.elements[:, [1, 2, 0]].flatten()

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
    _, _, numbering_IJ = find(number_to_edges)
    _, _, idx_JI2IJ = find(coo_matrix((idx_JI, (J[idx_JI], I[idx_JI]))))
    edge_number[idx_JI2IJ] = numbering_IJ - 1

    element2edges = edge_number[0:3*n_elements].reshape(n_elements, 3)
    edge2nodes = np.column_stack((I[idx_IJ], J[idx_IJ]))
    # Provide boundary2edges
    boundaries_to_edges = []
    for j in np.arange(n_boundaries):
        boundaries_to_edges.append(
            edge_number[np.arange(pointer[j+1]+1, pointer[j+2]+1, dtype=int)])
    return element2edges, edge2nodes, boundaries_to_edges


def refineNVB(coordinates: np.ndarray,
              elements: np.ndarray,
              marked_elements: np.ndarray,
              *boundaries):
    n_elements = elements.shape[0]

    # obtain geometric information on edges
    element2edges, edge2nodes, boundaries_to_edges = provide_geometric_data(
        Mesh(coordinates=coordinates, elements=elements), *boundaries
    )

    # mark all edges of marked elements for refinement
    # TODO can this be replaced with `np.zeros(edges2nodes.shape[0])`?
    edge2newNode = np.zeros(np.max(element2edges)+1, dtype=int)
    edge2newNode[element2edges[marked_elements].flatten()] = 1

    # closure of edge marking, i.e.
    # if any edge in T is marked, make sure that the reference
    # edge in T is marked, as well
    swap = 1
    while swap:
        element2marked_edges = edge2newNode[element2edges]
        swap = np.nonzero(
            np.logical_and(
                # elements, whose reference edge is not marked
                np.logical_not(element2marked_edges[:, 0]),
                # elements, having any non-reference edge marked
                np.logical_or(element2marked_edges[:, 1],
                              element2marked_edges[:, 2])))[0]
        edge2newNode[element2edges[swap, 0]] = 1

    # generate new nodes
    n_new_nodes = np.count_nonzero(edge2newNode)  # number of new nodes
    # assigning indices to new nodes
    edge2newNode[edge2newNode != 0] = np.arange(
        coordinates.shape[0] - 1,
        coordinates.shape[0] - 1 + n_new_nodes)
    idx = np.nonzero(edge2newNode)[0]
    new_node_coordinates = (
        coordinates[edge2nodes[idx, 0], :] +
        coordinates[edge2nodes[idx, 1], :]) / 2.
    coordinates = np.vstack([coordinates, new_node_coordinates])

    # refine boundary conditions
    for k, boundary in enumerate(boundaries):
        if boundary.size:
            new_nodes_on_boundary = edge2newNode[boundaries_to_edges[k]]
            marked_edges = np.nonzero(new_nodes_on_boundary)[0]
            if marked_edges.size:
                boundary = np.vstack(
                    [boundary[np.logical_not(new_nodes_on_boundary), :],
                     np.hstack(boundary[marked_edges, 0],
                               new_nodes_on_boundary[marked_edges]),
                     np.hstack(new_nodes_on_boundary[marked_edges],
                               boundary[marked_edges, 1])])

    # provide new nodes for refinement of elements
    new_nodes = edge2newNode[element2edges]

    # Determine type of refinement for each element
    marked_edges = new_nodes != 0

    ref_marked = marked_edges[:, 0]
    first_marked = marked_edges[:, 1]
    second_marked = marked_edges[:, 2]

    none = np.logical_not(ref_marked)
    bisec1 = ref_marked & np.logical_not(first_marked) & np.logical_not(second_marked)
    bisec12 = ref_marked & first_marked & np.logical_not(second_marked)
    bisec13 = ref_marked & np.logical_not(first_marked) & second_marked
    bisec123 = ref_marked & first_marked & second_marked

    # generate element numbering for refined mesh
    idx = np.ones(n_elements)
    idx[bisec1] = 2  # bisec(1): newest vertex bisection of 1st edge
    idx[bisec12] = 3  # bisec(2): newest vertex bisection of 1st and 2nd edge
    idx[bisec13] = 3  # bisec(2): newest vertex bisection of 1st and 3rd edge
    idx[bisec123] = 4  # bisec(3): newest vertex bisection of all edges
    idx = np.vstack([np.array([0]), np.cumsum(idx)])  # TODO maybe bug source

    # TODO continue here
    # generate new elements
    newElements = np.zeros(idx[-1] - 1, 3)
    newElements[idx(none), :] = elements[none, :]
    newElements[np.hstack([idx[bisec1], 1+idx[bisec1]]), :] \
        = np.vstack(
            [np.hstack([
                elements[bisec1, 3],
                elements[bisec1, 1],
                new_nodes[bisec1, 1]]),
             np.hstack([
                 elements[bisec1, 2],
                 elements[bisec1, 3],
                 new_nodes[bisec1, 1]])])
    newElements[np.hstack([idx[bisec12], 1+idx[bisec12], 2+idx[bisec12]]), :] \
        = np.vstack(
            [np.hstack([elements[bisec12, 3],
                        elements[bisec12, 1],
                        new_nodes[bisec12, 1]]),
             np.hstack([new_nodes[bisec12, 1],
                        elements[bisec12, 2],
                        new_nodes[bisec12, 2]]),
             np.hstack([elements[bisec12, 3],
                        new_nodes[bisec12, 1],
                        new_nodes[bisec12, 2]])])
    newElements[np.hstack([idx[bisec13], 1+idx[bisec13], 2+idx[bisec13]]), :] \
        = np.vstack(
            [np.hstack([new_nodes[bisec13, 1],
                        elements[bisec13, 3],
                        new_nodes[bisec13, 3]]),
             np.hstack([elements[bisec13, 1],
                        new_nodes[bisec13, 1],
                        new_nodes[bisec13, 3]]),
             np.hstack([elements[bisec13, 2],
                        elements[bisec13, 3],
                        new_nodes[bisec13, 1]])])
    newElements[np.hstack([idx(bisec123), 1+idx(bisec123),
                           2+idx(bisec123), 3+idx(bisec123)]), :] \
        = np.vstack([
            np.hstack([new_nodes[bisec123, 1],
                       elements[bisec123, 3],
                       new_nodes[bisec123, 3]]),
            np.hstack([elements[bisec123, 1],
                       new_nodes[bisec123, 1],
                       new_nodes[bisec123, 3]]),
            np.hstack([new_nodes[bisec123, 1],
                       elements[bisec123, 2],
                       new_nodes[bisec123, 2]]),
            np.hstack([elements[bisec123, 3],
                       new_nodes[bisec123, 1],
                       new_nodes[bisec123, 2]])])

    return coordinates, newElements, boundaries
