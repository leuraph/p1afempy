import numpy as np
from p1afempy import data_structures
from ismember import ismember, is_row_in
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
    `perform_transform` : I -> [-1, 0, 1, 2, ..., n-1] =: I'

    notes
    -----
    - the returned mapping may be used, e.g. to transform global indexation
      to local indexation after extraction of a local patch
    - if global[i, j, ...] <= global[i', j', ...], then we also have
      local[i, j, ...]  <= local[i', j', ...]
    - if N is the number of distinct indices in global_indices,
      then local_indices contains the integers 0, 1, ..., N-1
    - values of -1 get mapped to -1 as they indicate a boundary in
      element to neighbour mapping arrays

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
    transform = dict(zip(
        np.hstack((unique_idxs, -1)),
        np.hstack((local_idx, -1))))
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
    all_edges_ij = np.column_stack([element_indices_i, element_indices_j])

    is_interior = is_row_in(all_edges_ij, all_edges_ij[:, [1, 0]])

    exterior_edges = all_edges_ij[np.logical_not(is_interior)]

    for boundary in boundaries:
        covered = is_row_in(exterior_edges, boundary)
        exterior_edges = exterior_edges[np.logical_not(covered)]

    if exterior_edges.size > 0:
        boundaries.append(exterior_edges)
    return boundaries


def get_local_boundaries(boundaries: list[data_structures.BoundaryType],
                         local_elements: data_structures.ElementsType,
                         perform_transform: np.vectorize
                         ) -> list[data_structures.BoundaryType]:
    nodes_idx_i = local_elements[:, [0, 1, 2]].flatten()
    nodes_idx_j = local_elements[:, [1, 2, 0]].flatten()
    local_edges = np.column_stack((nodes_idx_i, nodes_idx_j))

    local_boundaries = []
    for boundary in boundaries:
        shared_edges = is_row_in(boundary, local_edges)
        if np.any(shared_edges):
            local_boundaries.append(
                perform_transform(boundary[shared_edges]))
    return local_boundaries


def get_neighbouring_elements(elements: data_structures.ElementsType,
                              which_for: int,
                              element_to_neighbours: np.ndarray
                              ) -> tuple[
                                data_structures.ElementsType,
                                np.ndarray]:
    """
    identifies and returns neighbouring elements for a marked element

    Parameters
    ----------
    elements: data_structures.ElementsType
        a 2D array-like structure containing all elements.
    which_for: int
        index of the marked element for which neighbouring
        elements are to be found.
    element_to_neighbours: np.ndarray
        array mapping each element to its neighbours

    Returns
    -------
    local_elements: data_structures.ElementsType
        a 2D array-like structure containing neighbouring
        elements along with the marked element
    local_element_to_neighbours: np.ndarray
        all rows of `element_to_neighbours` that correspond
        to an element included in `local_elements`
    """
    local_neighbours = element_to_neighbours[which_for]
    has_neighbour = local_neighbours >= 0

    local_elements = np.vstack([
        elements[local_neighbours[has_neighbour], :],
        elements[which_for]])

    local_element_to_neighbours = np.vstack([
        element_to_neighbours[local_neighbours[has_neighbour], :],
        element_to_neighbours[which_for]
    ])

    return local_elements, local_element_to_neighbours


def get_local_patch_edge_based(
        elements: data_structures.ElementsType,
        coordinates: data_structures.CoordinatesType,
        current_iterate: np.ndarray,
        edge: np.ndarray) -> tuple[
            data_structures.ElementsType,
            data_structures.CoordinatesType,
            np.ndarray,
            np.ndarray]:
    """
    parameters
    ----------
    elements: data_structures.ElementsType
        the global mesh's elements
    coordinates: data_structures.CoordinatesType
        the global mesh's coordinates
    current_iterate: np.ndarray
        the global curret iterate
    edge: np.ndarray
        indices of the edge to be refined, i.e.
        edge = np.array([i, j]), where (i, j)
        represent the indices of the coordintes
        that make up the edge to be refined

    returns
    -------
    local_elements: np.ndarray
        local patch's elements in local indexing
    local_coordinates: np.ndarray
        local patch's coordinates
    local_iterate: np.ndarray
        current iterate on local patch's nodes
    local_edge_indices: np.ndarray
        local patch's edge shared by the two neighbouring
        elements in local indexing
    """

    local_elements_indices = np.sum(np.isin(elements, edge), axis=1) == 2
    local_elements = np.copy(elements[local_elements_indices])

    # local, unique, sorted indices of nodes
    local_indices = np.unique(local_elements.flatten())

    local_coordinates = np.copy(coordinates[local_indices])
    local_iterate = np.copy(current_iterate[local_indices])

    # retreiving the transformation (global -> local indices)
    perform_transform = get_global_to_local_index_mapping(local_indices)

    # local patch's elements in local indices
    local_elements = perform_transform(local_elements)
    local_edge_indices = perform_transform(edge)

    if local_elements.shape[0] != 2:
        # if there are less than two elements extracted, the provided edge
        # is a boundary edge. As this special case remains
        # unnecessary to consider for now (we only consider homogenous
        # dirichlet), we exclude this possibility.
        raise ValueError(
            'this function must not be called with boundary edges')

    return local_elements, local_coordinates, local_iterate, \
        local_edge_indices


def get_local_patch(coordinates: data_structures.CoordinatesType,
                    elements: data_structures.ElementsType,
                    boundaries: list[data_structures.BoundaryType],
                    which_for: int,
                    element_to_neighbours: np.ndarray,
                    global_values: np.ndarray = np.array([]),
                    ) -> tuple[data_structures.CoordinatesType,
                               data_structures.ElementsType,
                               list[data_structures.BoundaryType],
                               np.ndarray, np.ndarray, int]:
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
    element_to_neighbours: np.ndarray
        array mapping elements to their neighbours
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
    local_element_to_neighbours: np.ndarray
        array mapping local elements to local neighbours
    local_which: int
        the local index of the marked element `which_for`

    notes
    -----
    the returned local patch is indexed in a local fashion, i.e.
    - the indices in local_elements refer to the entries in local_coordinates
    - the indices in all elements in local_boundaries refer to entries in
      local_coordinates
    """
    # indices of all elements neighbouring the selected element
    # (including the index of the selected element)
    global_elements_idx = np.hstack((
        element_to_neighbours[which_for],
        which_for))
    # remove -1 as index
    global_elements_idx = global_elements_idx[global_elements_idx != -1]
    local_elements_idx = np.arange(global_elements_idx.size)
    transform = dict(zip(
        np.hstack((global_elements_idx, -1)),
        np.hstack((local_elements_idx, -1))))
    global_to_local_element_index_mapping = np.vectorize(
        lambda old: transform[old])

    local_element_to_neighbours = element_to_neighbours[global_elements_idx, :]
    exceeding_elements = np.logical_not(np.isin(
        local_element_to_neighbours, global_elements_idx))
    local_element_to_neighbours[exceeding_elements] = -1

    local_elements = elements[global_elements_idx, :]
    # unique sorted global indices of all nodes in global patch
    unique_idxs = np.unique(local_elements)

    # retreiving the transformation (global -> local indices)
    perform_transform = get_global_to_local_index_mapping(unique_idxs)

    # retreive the local boundaries
    local_boundaries = get_local_boundaries(
        boundaries=boundaries,
        local_elements=local_elements,
        perform_transform=perform_transform)

    # local patch's elements in local indices
    local_elements = perform_transform(local_elements)
    # local patch's neighbour map in local indices
    # note that we map neighbours of the patch as -1,
    # indicating a local boundary
    local_element_to_neighbours = global_to_local_element_index_mapping(
        local_element_to_neighbours)

    # local patch's coordinates
    local_coordinates = coordinates[unique_idxs]

    # local value extraction, if given
    local_values = np.array([])
    if global_values.size > 0:
        local_values = global_values[unique_idxs]

    local_which = global_to_local_element_index_mapping(which_for)

    # TODO return a local which as well
    # makes code more maintainable in the future,
    # because then, outside of the function, we do not
    # need to assume a specific order of anything
    # (only in the unit tests, of course)
    return local_coordinates, local_elements, \
        local_boundaries, local_values, local_element_to_neighbours, \
        local_which


def get_element_to_neighbours(
        elements: data_structures.ElementsType) -> np.ndarray:
    """
    calculates and returns a map element to neighbours

    parameters
    ----------
    elements: data_structures.ElementsType
        all elements at hand

    returns
    -------
    element2neighbours: np.ndarray
        element2neighbours[k] holds the indices of thk-th
        element's neighbours, where element2neighbours[k, i]
        corresponds to the index of the element sharing the i-th
        edge of the k-th element.
        a value of `-1` resembles a boundary, i.e. an edge without a neighbour
    """
    n_elements = elements.shape[0]
    I = elements.flatten(order='F')
    J = elements[:, [1, 2, 0]].flatten(order='F')
    nodes2edge = coo_matrix((np.arange(1, 3*n_elements+1), (I, J)))
    mask = nodes2edge > 0
    _, _, idxIJ = find(nodes2edge)
    aranged_element_indices = np.arange(1, n_elements + 1)
    tmp = np.hstack((
        aranged_element_indices,
        aranged_element_indices,
        aranged_element_indices))
    _, _, neighbourIJ = find(mask + mask.multiply(coo_matrix((tmp, (J, I)))))
    element2neighbours = np.zeros(3 * n_elements, dtype=int)
    element2neighbours[idxIJ-1] = neighbourIJ - 1
    return element2neighbours.reshape((n_elements, 3), order='F') - 1
