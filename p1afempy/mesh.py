import numpy as np
from p1afempy import data_structures
from scipy.sparse import coo_matrix, find
from matplotlib import pyplot as plt


def get_area(coordinates: data_structures.CoordinatesType,
             elements: data_structures.ElementsType) -> np.ndarray:
    """
    calculates and returns the area of each element as numpy array

    parameters
    ----------
    coordinates: data_structures.CoordinatesType
    elements: data_structures.ElementsType

    returns
    -------
    np.ndarray:
        area of each element as Mx1 array,
        where M represents the number ot elements.
    """
    d21, d31 = get_directional_vectors(coordinates=coordinates,
                                       elements=elements)
    return 0.5 * (d21[:, 0]*d31[:, 1] - d21[:, 1] * d31[:, 0])


def get_directional_vectors(coordinates: data_structures.CoordinatesType,
                            elements: data_structures.ElementsType
                            ) -> tuple[np.ndarray, np.ndarray]:
    """
    returns the vectors pointing from vertex[0] to vertex[1]
    and vertex[2], respectively, for each element

    Returns
    -------
    d21: np.ndarray
        Mx2 array, where d21[k, :] points from vertex 0 to vertex 1
        in the k-th element and M represents the number of elements
    d31: np.ndarray
        Mx2 array, where d31[k, :] points from vertex 0 to vertex 2
        in the k-th element and M represents the number of elements
    """
    c1 = coordinates[elements[:, 0], :]
    d21 = coordinates[elements[:, 1], :] - c1
    d31 = coordinates[elements[:, 2], :] - c1

    return d21, d31


def show_mesh(coordinates: data_structures.CoordinatesType,
              elements: data_structures.ElementsType) -> None:
    """displays the mesh at hand"""
    for element in elements:
        r0, r1, r2 = coordinates[element, :]
        plt.plot(
            [r0[0], r1[0], r2[0], r0[0]],
            [r0[1], r1[1], r2[1], r0[1]],
            'black', linewidth=0.5)
    plt.show()


def provide_geometric_data(elements: data_structures.ElementsType,
                           boundaries: list[data_structures.BoundaryType]
                           ) -> tuple[np.ndarray,
                                      np.ndarray,
                                      list[np.ndarray]]:
    """
    provides geometric data about the mesh (elements and boundaries) at hand

    Parameters
    ----------
    elements: data_structures.ElementsType
    boundaries: list[data_structures.BoundaryType

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
    element_indices_i = elements.flatten(order='F')
    element_indices_j = elements[:, [1, 2, 0]].flatten(order='F')

    # Symmetrize I and J (so far boundary edges appear only once)
    pointer = np.concatenate(([0, 3*n_elements-1],
                              np.zeros(n_boundaries, dtype=int)), dtype=int)
    for k, boundary in enumerate(boundaries):
        if boundary.size:
            element_indices_i = np.concatenate(
                (element_indices_i, boundary[:, 1]), dtype=int)
            element_indices_j = np.concatenate(
                (element_indices_j, boundary[:, 0]), dtype=int)
        pointer[k+2] = pointer[k+1] + boundary.shape[0]

    # Fixing an edge number for all edges, where i<j
    idx_IJ = np.where(element_indices_i < element_indices_j)[0]
    n_unique_edges = idx_IJ.size
    edge_number = np.zeros(element_indices_i.size, dtype=int)
    edge_number[idx_IJ] = np.arange(n_unique_edges)

    # Ensuring the same numbering for all edges, where j<i
    idx_JI = np.where(element_indices_j < element_indices_i)[0]
    number_to_edges = coo_matrix(
        (np.arange(n_unique_edges) + 1, (element_indices_i[idx_IJ],
                                         element_indices_j[idx_IJ])))
    # NOTE In Matlab, the returned order is different
    _, _, numbering_IJ = find(number_to_edges)
    # NOTE In Matlab, the returned order is different
    _, _, idx_JI2IJ = find(
        coo_matrix((idx_JI + 1, (element_indices_j[idx_JI],
                                 element_indices_i[idx_JI]))))
    # NOTE Here, it coincides with Matlab again, though.
    edge_number[idx_JI2IJ - 1] = numbering_IJ - 1

    element_to_edges = edge_number[0:3*n_elements].reshape(n_elements, 3,
                                                           order='F')
    edge_to_nodes = np.column_stack((element_indices_i[idx_IJ],
                                     element_indices_j[idx_IJ]))
    # Provide boundary2edges
    boundaries_to_edges = []
    for j in np.arange(n_boundaries):
        boundaries_to_edges.append(
            edge_number[np.arange(pointer[j+1]+1, pointer[j+2]+1, dtype=int)])
    return element_to_edges, edge_to_nodes, boundaries_to_edges


def get_global_to_local_index_mapping(unique_idxs: np.ndarray):
    """
    given a set of n unique, non-negative integers I,
    this function returns a vectorized and order preserving mapping
    `perform_transform` : I -> [0, 1, 2, ..., n-1] =: I'

    notes
    -----
    - the returned mapping may be used, e.g. to transform global indexation
      to local indexation after extraction of a local patch
    - if global[i, j, ...] <= global[i', j', ...], then we also have
      local[i, j, ...]  <= local[i', j', ...]
    - if N is the number of distinct indices in global_indices,
      then local_indices contains the integers 0, 1, ..., N-1

    example
    -------
    >>> global_indices = np.array([[12, 3, 6],
    >>>                            [12, 2, 6],
    >>>                            [12, 3, 15],
    >>>                            [66, 77, 88]])
    >>> unique_indices = np.unique(global_indices)
    >>> perform_transform = get_global_to_local_index_mapping(unique_indices)
    >>> local_indices = perform_transform(global_indices)
    >>> local_indices
        array([[3, 1, 2],
               [3, 0, 2],
               [3, 1, 4],
               [5, 6, 7]])
    """
    local_idx = np.arange(unique_idxs.size)
    transform = dict(zip(unique_idxs, local_idx))
    perform_transform = np.vectorize(lambda old: transform[old])
    return perform_transform


def complete_boundaries(elements: data_structures.ElementsType,
                        boundaries: list[data_structures.BoundaryType]
                        ) -> list[data_structures.BoundaryType]:
    """
    given a possibly incomplete list of boundaries,
    returns a complete list of boundaries, i.e. adds
    an artificial boundary where needed

    parameters
    ----------
    elements: data_structures.ElementsType
        elements of the mesh at hand
    boundaries: list[data_structures.BoundaryType]
        a possibly incomplete list of boundaries
        of the elements given

    returns
    -------
    completed_boundaries: list[data_structures.BoundaryType]
        a coomplete list of boundaries of the elements given

    example
    -------
    >>> elements = np.array([[0, 1, 2], [0, 2, 3]])
    >>> boundary = np.array([[0, 1], [1, 2]])
    >>> completed_boundaries = complete_boundaries(
    >>>     elements, [boundary])
    >>> completed_boundaries
        [array([[0, 1], [1, 2]]), array([[2, 3], [3, 0]])]
    """
    element_indices_i = elements.flatten()
    element_indices_j = elements[:, [1, 2, 0]].flatten()
    all_edges = np.column_stack([element_indices_i, element_indices_j])

    exterior_edges = []
    for edge in all_edges:
        # mark all rows where edge appears
        edge_is_here = np.sum(np.isin(all_edges, edge), axis=1) == 2
        # if edge appears twice, it is shared by two elements and therefore
        # must be interior
        is_interior = np.sum(edge_is_here) == 2
        if not is_interior:
            exterior_edges.append(edge)

    artificial_boundary = []
    for exterior_edge in exterior_edges:
        covered = False
        for boundary in boundaries:
            if np.any(np.sum(np.isin(boundary, exterior_edge), axis=1) == 2):
                covered = True
                break
        if not covered:
            artificial_boundary.append(exterior_edge)

    if artificial_boundary:
        boundaries.append(np.array(artificial_boundary))
    return boundaries


def get_local_patch(coordinates: data_structures.CoordinatesType,
                    elements: data_structures.ElementsType,
                    boundaries: list[data_structures.BoundaryType],
                    which_for: int,
                    global_values: np.ndarray = np.array([])
                    ) -> tuple[data_structures.CoordinatesType,
                               data_structures.ElementsType,
                               list[data_structures.BoundaryType]]:
    """
    returns the local mesh corresponding to the k-th element
    and its immediate neightbours, i.e. elements that share an edge
    with the selected k-th element

    parameters
    ----------
    coordinates: data_structures.CoordinatesType
        coordinates of the mesh at hand
    elements: data_structures.ElementsType
        elements of the mesh at hand
    boundaries: list[data_structures.BoundaryType]
        boundaries of the mesh at hand
    which_for: int
        index of the element for which to extract the local patch
    global_values: np.ndarray = np.array([])
        an array of values defined on the global coordinates

    returns
    -------
    local_coordinates: data_structures.CoordinatesType
        coordinates of the selected element and its neighbours,
        where neighbours are elements that share an edge with
        the selected element
    local_elements: data_structures.ElementsType
        neighbours of the selected element, i.e.
        elements that share an edge with
        the selected element
    local_boundaries: list[data_structures.BoundaryType]
        boundaries of the local patch inherited from
        the given boundaries, i.e. if none of the elements
        in the local patch touch the boundary at hand,
        an empty list is returned
    local_values: np.ndarray
        the global values given on the local coordinates

    notes
    -----
    the returned local patch is indexed in a local fashion, i.e.
    - the indices in local_elements refer to the entries in local_coordinates
    - the indices in all elements in local_boundaries refer to entries in
      local_coordinates
    """
    # global indices of the marked element's nodes
    nodes = elements[which_for]

    # identifying the marked element's nieghbours
    neighbours = np.sum(np.isin(elements, nodes), axis=1) == 2

    # global indices of the local patch's elements
    local_elements = np.vstack([elements[neighbours], nodes])

    # unique sorted global indices of all nodes in global patch
    unique_idxs = np.unique(local_elements)

    # retreiving the transformation (global -> local indices)
    perform_transform = get_global_to_local_index_mapping(unique_idxs)

    local_boundaries = []
    for boundary in boundaries:
        edge_in_local_patch = np.sum(
            np.isin(boundary, local_elements), axis=1) == 2
        local_boundary = boundary[edge_in_local_patch]
        if local_boundary.size > 0:
            local_boundaries.append(perform_transform(local_boundary))

    # local patch's elements in local indices
    local_elements = perform_transform(local_elements)

    # local patch's coordinates
    local_coordinates = coordinates[unique_idxs]

    # local value extraction, if given
    local_values = np.array([])
    if global_values.size > 0:
        local_values = global_values[unique_idxs]

    return local_coordinates, local_elements, local_boundaries, local_values
