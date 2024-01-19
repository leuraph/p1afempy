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


def get_area(coordinates: np.ndarray,
             elements: np.ndarray) -> np.ndarray:
    """
    returns the area of each element
    """
    d21, d31 = get_directional_vectors(coordinates=coordinates,
                                       elements=elements)

    # vector of element areas 4*|T|
    return 0.5 * (d21[:, 0]*d31[:, 1] - d21[:, 1] * d31[:, 0])


def get_directional_vectors(coordinates: np.ndarray,
                            elements: np.ndarray) -> tuple[np.ndarray,
                                                           np.ndarray]:
    """
    returns the vectors pointing from vertex[0] to vertex[1]
    and vertex[2], respectively, for each triangular element,
    i.e. for each row in elements

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


class BoundaryCondition:
    """
    A boundary condition.

    instance variables
    ------------------
    name: str
        The name of the boundary condition
    boundary: np.ndarray
        boundary[k] holds the indices (i, j) of the vertices making up the
        k-th edge of the boundary.
    """
    name: str
    boundary: np.ndarray

    def __init__(self, name: str, boundary: np.ndarray) -> None:
        self.name = name
        self.boundary = boundary


def plot_mesh(coordinates: np.ndarray,
              elements: np.ndarray) -> None:
    for element in elements:
        r0, r1, r2 = coordinates[element, :]
        plt.plot(
            [r0[0], r1[0], r2[0], r0[0]],
            [r0[1], r1[1], r2[1], r0[1]],
            'black', linewidth=0.5)
    plt.show()


def read_boundary_condition(
        path_to_boundary: Path, name: str = "") -> BoundaryCondition:
    """
    Reads boundary condition data from a file.

    Parameters
    ----------
    path_to_boundary: pathlib.Path
        Path to the file containing boundary condition data.
    name: str, optional
        Name to assign to the BoundaryCondition instance.
        If not provided, the name will be derived from
        the filename (excluding the extension).

    Returns
    -------
    BoundaryCondition:
        An instance of the BoundaryCondition class representing the read data.

    Example
    -------
    >>> file_path = Path("path/to/boundary_data.dat")
    >>> boundary_condition = read_boundary_condition(file_path, name="BC1")
    """
    data = np.loadtxt(path_to_boundary, dtype=int, converters=float)
    if not name:
        name = path_to_boundary.stem
    return BoundaryCondition(name=name, boundary=data)


def read_mesh(path_to_coordinates: Path, path_to_elements: Path,
              shift_indices: bool = False) -> Mesh:
    """
    Reads vertices and elements from files and
    returns the corresponding Mesh.

    Parameters
    ----------
    path_to_coordinates : pathlib.Path
        Path to the file containing mesh coordinates data.
    path_to_elements : pathlib.Path
        Path to the file containing mesh elements data.
    shift_indices: bool (default=False)
        If true, shifts the elements' indices according to i':=i-1.
        This can come in handy, if, e.g., one wants to read data
        that is compatible with Matlab/Fortran/Julia indexing
        (starting at 1, instead of 0).

    Returns
    -------
    Mesh:
        An instance of the Mesh class representing the read mesh data.

    Example
    -------
    >>> coordinates_path = Path("path/to/coordinates.txt")
    >>> elements_path = Path("path/to/elements.txt")
    >>> mesh_instance = read_mesh(coordinates_path, elements_path)
    """
    coordinates = np.loadtxt(path_to_coordinates)
    elements = np.loadtxt(path_to_elements, dtype=int, converters=float)
    if shift_indices:
        elements = elements - 1
    return Mesh(coordinates=coordinates, elements=elements)


def provide_geometric_data(elements: np.ndarray,
                           boundaries: list[BoundaryCondition]):
    """
    Provides geometric data about the mesh at hand.

    Parameeters
    -----------
    elements: np.ndarray
        the elements of the mesh, i.e.
        an Mx3 array where each row represents an element
    boundaries: list[BoundaryCondition]
        boundary conditions defined on the mesh

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
    n_elements = elements.shape[0]
    n_boundaries = len(boundaries)

    # Extracting all directed edges E_l:=(I[l], J[l])
    # (interior edges appear twice)
    I = elements.flatten(order='F')
    J = elements[:, [1, 2, 0]].flatten(order='F')

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
    # NOTE In Matlab, the returned order is different
    _, _, numbering_IJ = find(number_to_edges)
    # NOTE In Matlab, the returned order is different
    _, _, idx_JI2IJ = find(coo_matrix((idx_JI + 1, (J[idx_JI], I[idx_JI]))))
    # NOTE Here, it coincides with Matlab again, though.
    edge_number[idx_JI2IJ - 1] = numbering_IJ - 1

    element2edges = edge_number[0:3*n_elements].reshape(n_elements, 3,
                                                        order='F')
    edge2nodes = np.column_stack((I[idx_IJ], J[idx_IJ]))
    # Provide boundary2edges
    boundaries_to_edges = []
    for j in np.arange(n_boundaries):
        boundaries_to_edges.append(
            edge_number[np.arange(pointer[j+1]+1, pointer[j+2]+1, dtype=int)])
    return element2edges, edge2nodes, boundaries_to_edges


def refineRGB(coordinates: np.ndarray,
              elements: np.ndarray,
              marked_elements: np.ndarray,
              boundary_conditions: list[BoundaryCondition]
              ) -> tuple[np.ndarray, np.ndarray, list[BoundaryCondition]]:
    """
    Refines the mesh and boundary conditions based on the
    red-green-blue (RGB) refinement strategy.

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
    new_coordinates: np.ndarray
        the coordinates of the refined mesh
    new_elements: np.ndarray
        the elements of the refined mesh
    new_boundaries: list[BoundaryCondition]
        The refined boundary conditions

    Example
    -------
    >>> mesh = Mesh(...)  # Initialize a mesh
    >>> marked_elements = np.array([0, 2, 3, 4])
    >>> boundary_conditions = [BC1, BC2, BC3]  # instances of BoundaryCondition
    >>> new_mesh, new_boundary_conditions = refineRGB(
            mesh,
            marked_elements, boundary_conditions)
    """
    return refineNVB(coordinates,
                     elements,
                     marked_elements,
                     boundary_conditions,
                     sort_for_longest_egde=True)


def refineNVB(coordinates: np.ndarray,
              elements: np.ndarray,
              marked_elements: np.ndarray,
              boundary_conditions: list[BoundaryCondition],
              sort_for_longest_egde: bool = False
              ) -> tuple[np.ndarray, np.ndarray, list[BoundaryCondition]]:
    """
    Refines the mesh based on newest vertex bisection (NVB).

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
    new_coordinates: np.ndarray
        the coordinates of the refined mesh
    new_elements: np.ndarray
        the elements of the refined mesh
    new_boundaries: list[BoundaryCondition]
        The refined boundary conditions

    Example
    -------
    >>> mesh = Mesh(...)  # Initialize a mesh
    >>> marked_elements = np.array([0, 2, 3, 4])
    >>> boundary_conditions = [BC1, BC2, BC3]  # instances of BoundaryCondition
    >>> new_mesh, new_boundary_conditions = refineNVB(
            mesh.coordinates, mesh.elements,
            marked_elements, boundary_conditions)
    """
    n_elements = elements.shape[0]

    if sort_for_longest_egde:
        # Sort elements such that first edge is longest
        dx = (coordinates[elements[:, [1, 2, 0]], 0]
              - coordinates[elements, 0]).flatten(order='F')
        dy = (coordinates[elements[:, [1, 2, 0]], 1]
              - coordinates[elements, 1]).flatten(order='F')
        idxMax = np.argmax(
            (np.square(dx)+np.square(dy)).reshape((n_elements, 3), order='F'),
            axis=1)
        idx = idxMax == 1
        elements[idx, :] = elements[idx][:, [1, 2, 0]]
        idx = idxMax == 2
        elements[idx, :] = elements[idx][:, [2, 0, 1]]

    # obtain geometric information on edges
    element2edges, edge2nodes, boundaries_to_edges = provide_geometric_data(
        elements=elements,
        boundaries=boundary_conditions)

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
    new_coordinates = np.vstack([coordinates, new_node_coordinates])

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
    not_first = np.logical_not(first_marked)
    not_second = np.logical_not(second_marked)
    bisec1 = ref_marked & not_first & not_second
    bisec12 = ref_marked & first_marked & not_second
    bisec13 = ref_marked & not_first & second_marked
    bisec123 = ref_marked & first_marked & second_marked

    # generate element numbering for refined mesh
    idx = np.ones(n_elements, dtype=int)
    idx[bisec1] = 2  # bisec(1): newest vertex bisection of 1st edge
    idx[bisec12] = 3  # bisec(2): newest vertex bisection of 1st and 2nd edge
    idx[bisec13] = 3  # bisec(2): newest vertex bisection of 1st and 3rd edge
    idx[bisec123] = 4  # bisec(3): newest vertex bisection of all edges
    idx = np.hstack([0, np.cumsum(idx)])  # TODO maybe bug source

    # generate new elements
    new_elements = np.zeros((idx[-1], 3), dtype=int)
    new_elements[idx[np.hstack((none, False))], :] = elements[none, :]
    new_elements[np.hstack([idx[np.hstack((bisec1, False))],
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
    new_elements[np.hstack([idx[np.hstack((bisec12, False))],
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
    new_elements[np.hstack([idx[np.hstack((bisec13, False))],
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
    if sort_for_longest_egde:
        new_elements[np.hstack([idx[np.hstack([bisec123, False])],
                               1+idx[np.hstack([bisec123, False])],
                               2+idx[np.hstack([bisec123, False])],
                               3+idx[np.hstack([bisec123, False])]]), :] \
            = np.vstack([
                np.column_stack([elements[bisec123, 0],
                                 new_nodes[bisec123, 0],
                                 new_nodes[bisec123, 2]]),
                np.column_stack([new_nodes[bisec123, 0],
                                 elements[bisec123, 1],
                                 new_nodes[bisec123, 1]]),
                np.column_stack([new_nodes[bisec123, 2],
                                 new_nodes[bisec123, 1],
                                 elements[bisec123, 2]]),
                np.column_stack([new_nodes[bisec123, 1],
                                 new_nodes[bisec123, 2],
                                 new_nodes[bisec123, 0]])])
    else:
        new_elements[np.hstack([idx[np.hstack((bisec123, False))],
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

    return new_coordinates, new_elements, new_boundaries
