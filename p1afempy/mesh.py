import numpy as np
from pathlib import Path
from scipy.sparse import coo_matrix, find
from matplotlib import pyplot as plt


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


def plot_mesh(mesh: Mesh) -> None:
    for element in mesh.elements:
        r0, r1, r2 = mesh.coordinates[element, :]
        plt.plot(
            [r0[0], r1[0], r2[0], r0[0]],
            [r0[1], r1[1], r2[1], r0[1]],
            'black', linewidth=0.5)
    plt.show()


class BoundaryCondition:
    name: str
    boundary: np.ndarray

    def __init__(self, name: str, boundary: np.ndarray) -> None:
        self.name = name
        self.boundary = boundary


def read_boundary_condition(path_to_boundary: Path) -> BoundaryCondition:
    data = np.loadtxt(path_to_boundary, dtype=int, converters=float)
    name = path_to_boundary.stem
    return BoundaryCondition(name=name, boundary=data)


def read_mesh(path_to_coordinates: Path, path_to_elements: Path) -> Mesh:
    coordinates = np.loadtxt(path_to_coordinates)
    elements = np.loadtxt(path_to_elements, dtype=int, converters=float)
    return Mesh(coordinates=coordinates, elements=elements)


def provide_geometric_data(domain: Mesh, boundaries: list[BoundaryCondition]):
    """
    Provides geometric data about the mesh at hand.

    Returns
    -------
    element2edges: np.ndarray
        element2edges[k] holds the edges' indices of
        the k-th element (counter-clockwise)
    edge2nodes: np.ndarray
        edge2nodes[k] holds the nodes' indices (i, j)
        of the k-th edge s.t. i < j
    boundaries_to_edges: list[np.ndarray]
        boundaries_to_edges[k] holds the mapping
        s.t. boundaries_to_edges[k][n] gives the indices
        (i, j) of the n-th edge of the k-th boundary.
    """
    n_elements = domain.elements.shape[0]
    n_boundaries = len(boundaries)

    # Extracting all directed edges E_l:=(I[l], J[l])
    # (interior edges appear twice)
    I = domain.elements.flatten(order='F')
    J = domain.elements[:, [1, 2, 0]].flatten(order='F')

    # Symmetrize I and J (so far boundary edges appear only once)
    pointer = np.concatenate(([0, 3*n_elements-1],
                              np.zeros(n_boundaries, dtype=int)), dtype=int)
    for k, boundary_condition in enumerate(boundaries):
        boundary = boundary_condition.boundary
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
    _, _, numbering_IJ = find(number_to_edges)  # NOTE In Matlab, the returned order is different
    _, _, idx_JI2IJ = find(coo_matrix((idx_JI + 1, (J[idx_JI], I[idx_JI]))))  # NOTE In Matlab, the returned order is different
    edge_number[idx_JI2IJ - 1] = numbering_IJ - 1 # NOTE Here, it coincides with Matlab again, though.

    element2edges = edge_number[0:3*n_elements].reshape(n_elements, 3, order='F')
    edge2nodes = np.column_stack((I[idx_IJ], J[idx_IJ]))
    # Provide boundary2edges
    boundaries_to_edges = []
    for j in np.arange(n_boundaries):
        boundaries_to_edges.append(
            edge_number[np.arange(pointer[j+1]+1, pointer[j+2]+1, dtype=int)])
    return element2edges, edge2nodes, boundaries_to_edges


def refineNVB(mesh: Mesh, marked_elements: np.ndarray,
              boundary_conditions: list[BoundaryCondition]
              ) -> tuple[Mesh, list[BoundaryCondition]]:
    """
    Refines the mesh based on marked elements and updates boundary conditions.

    Parameters
    ----------
    mesh: Mesh
        The initial mesh to be refined.
    marked_elements: np.ndarray
        Indices of the elements to be refined.
    boundary_conditions: list[BoundaryCondition]
        List of boundary conditions to update.

    Returns
    -------
    refined_mesh: Mesh
        The refined mesh
    new_boundaries: list[BoundaryCondition]
        The refined boundary conditions

    Example
    -------
    >>> mesh = Mesh(...)  # Initialize a mesh
    >>> marked_elements = np.array([0, 2, 3, 4])
    >>> boundary_conditions = [BC1, BC2, BC3]  # Assuming BC1, BC2, BC3 are instances of BoundaryCondition
    >>> new_mesh, new_boundary_conditions = refineNVB(mesh, marked_elements, boundary_conditions)
    """
    elements = mesh.elements
    coordinates = mesh.coordinates
    n_elements = elements.shape[0]

    # obtain geometric information on edges
    element2edges, edge2nodes, boundaries_to_edges = provide_geometric_data(
        domain=mesh, boundaries=boundary_conditions)

    # mark all edges of marked elements for refinement
    edge2newNode = np.zeros(edge2nodes.shape[0], dtype=int)
    edge2newNode[element2edges[marked_elements].flatten()] = 1

    # closure of edge marking, i.e.
    # if any edge in T is marked, make sure that the reference
    # edge in T is marked, as well
    swap = np.array([1])
    while swap.size > 0:
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
        coordinates.shape[0],
        coordinates.shape[0] + n_new_nodes)
    idx = np.nonzero(edge2newNode)[0]
    new_node_coordinates = (
        coordinates[edge2nodes[idx, 0], :] +
        coordinates[edge2nodes[idx, 1], :]) / 2.
    coordinates = np.vstack([coordinates, new_node_coordinates])

    # refine boundary conditions
    new_boundaries = []
    for k, boundary_condition in enumerate(boundary_conditions):
        boundary = boundary_condition.boundary
        if boundary.size:
            new_nodes_on_boundary = edge2newNode[boundaries_to_edges[k]]
            marked_edges = np.nonzero(new_nodes_on_boundary)[0]
            if marked_edges.size:
                boundary = np.vstack(
                    [boundary[np.logical_not(new_nodes_on_boundary), :],
                     np.column_stack([boundary[marked_edges, 0],
                                      new_nodes_on_boundary[marked_edges]]),
                     np.column_stack([new_nodes_on_boundary[marked_edges],
                                      boundary[marked_edges, 1]])])
        new_boundaries.append(
            BoundaryCondition(name=boundary_condition.name, boundary=boundary))

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
    idx = np.ones(n_elements, dtype=int)
    idx[bisec1] = 2  # bisec(1): newest vertex bisection of 1st edge
    idx[bisec12] = 3  # bisec(2): newest vertex bisection of 1st and 2nd edge
    idx[bisec13] = 3  # bisec(2): newest vertex bisection of 1st and 3rd edge
    idx[bisec123] = 4  # bisec(3): newest vertex bisection of all edges
    idx = np.hstack([0, np.cumsum(idx)])  # TODO maybe bug source

    # generate new elements
    newElements = np.zeros((idx[-1], 3), dtype=int)
    newElements[idx[np.hstack((none, False))], :] = elements[none, :]
    newElements[np.hstack([idx[np.hstack((bisec1, False))],
                           1+idx[np.hstack((bisec1, False))]]), :] \
        = np.vstack(
            [np.column_stack([
                elements[bisec1, 2],
                elements[bisec1, 0],
                new_nodes[bisec1, 0]]),
             np.column_stack([
                 elements[bisec1, 1],
                 elements[bisec1, 2],
                 new_nodes[bisec1, 0]])])
    newElements[np.hstack([idx[np.hstack((bisec12, False))],
                           1+idx[np.hstack((bisec12, False))],
                           2+idx[np.hstack((bisec12, False))]]), :] \
        = np.vstack(
            [np.column_stack([elements[bisec12, 2],
                              elements[bisec12, 0],
                              new_nodes[bisec12, 0]]),
             np.column_stack([new_nodes[bisec12, 0],
                              elements[bisec12, 1],
                              new_nodes[bisec12, 1]]),
             np.column_stack([elements[bisec12, 2],
                              new_nodes[bisec12, 0],
                              new_nodes[bisec12, 1]])])
    newElements[np.hstack([idx[np.hstack((bisec13, False))],
                           1+idx[np.hstack((bisec13, False))],
                           2+idx[np.hstack((bisec13, False))]]), :] \
        = np.vstack(
            [np.column_stack([new_nodes[bisec13, 0],
                              elements[bisec13, 2],
                              new_nodes[bisec13, 2]]),
             np.column_stack([elements[bisec13, 0],
                              new_nodes[bisec13, 0],
                              new_nodes[bisec13, 2]]),
             np.column_stack([elements[bisec13, 1],
                              elements[bisec13, 2],
                              new_nodes[bisec13, 0]])])
    newElements[np.hstack([idx[np.hstack((bisec123, False))],
                           1+idx[np.hstack((bisec123, False))],
                           2+idx[np.hstack((bisec123, False))],
                           3+idx[np.hstack((bisec123, False))]]), :] \
        = np.vstack([
            np.column_stack([new_nodes[bisec123, 0],
                             elements[bisec123, 2],
                             new_nodes[bisec123, 2]]),
            np.column_stack([elements[bisec123, 0],
                             new_nodes[bisec123, 0],
                             new_nodes[bisec123, 2]]),
            np.column_stack([new_nodes[bisec123, 0],
                             elements[bisec123, 1],
                             new_nodes[bisec123, 1]]),
            np.column_stack([elements[bisec123, 2],
                             new_nodes[bisec123, 0],
                             new_nodes[bisec123, 1]])])

    refined_mesh = Mesh(coordinates=coordinates, elements=newElements)
    return refined_mesh, new_boundaries
