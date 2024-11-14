import numpy as np
import ismember
from p1afempy.mesh import provide_geometric_data
from p1afempy.data_structures import \
    CoordinatesType, ElementsType, BoundaryType


def generate_new_nodes(edge2newNode: np.ndarray,
                       edge2nodes: np.ndarray,
                       coordinates: CoordinatesType,
                       to_embed: np.ndarray = np.array([])) -> tuple[
                           np.ndarray,
                           np.ndarray]:
    n_new_nodes = np.count_nonzero(edge2newNode)  # number of new nodes
    # assigning indices to new nodes
    edge2newNode[edge2newNode != 0] = np.arange(
        coordinates.shape[0],
        coordinates.shape[0] + n_new_nodes)
    edges_with_new_nodes_indices = np.nonzero(edge2newNode)[0]
    new_node_coordinates = (
        coordinates[edge2nodes[edges_with_new_nodes_indices, 0], :] +
        coordinates[edge2nodes[edges_with_new_nodes_indices, 1], :]) / 2.

    # interpolate values, if given
    if to_embed.size:
        new_embedded_values = (
            to_embed[edge2nodes[edges_with_new_nodes_indices, 0]] +
            to_embed[edge2nodes[edges_with_new_nodes_indices, 1]]) / 2.
        to_embed = np.hstack([to_embed, new_embedded_values])

    return np.vstack([coordinates, new_node_coordinates]), to_embed


def refine_boundaries(elements, which,
                      boundaries, n_nodes,
                      element_to_neighbours) -> list[BoundaryType]:
    edges_to_split_bool = element_to_neighbours[which] == -1

    if not np.any(edges_to_split_bool):
        return boundaries

    # splitting the boundary, where necessary
    # ---------------------------------------
    # 3x2 array of all edges of k-th element
    possible_edges = np.column_stack((
        elements[which, [0, 1, 2]].flatten(),
        elements[which, [1, 2, 0]].flatten()))
    # isolate edges to be split, i.e. edges
    # of the k-th element lying on the
    # domain's boundary

    new_boundaries = []
    for boundary in boundaries:
        if boundary.size > 0:
            boundary_to_split_bool, idx = \
                ismember.ismember(boundary, possible_edges, method='rows')
            # indices of new nodes to be inserted in the bundary at hand
            idx_new_nodes = idx + n_nodes
            boundary = np.vstack([
                boundary[np.logical_not(boundary_to_split_bool)],
                np.column_stack([
                    boundary[boundary_to_split_bool, 0], idx_new_nodes
                ]),
                np.column_stack([
                    idx_new_nodes, boundary[boundary_to_split_bool, 1]
                ])
            ])
        new_boundaries.append(boundary)
    return new_boundaries


def refine_elements(elements,
                    element_to_neighbours,
                    which,
                    n_nodes) -> ElementsType:
    index_new_node_1 = n_nodes  # indexing starts at zero
    index_new_node_2 = n_nodes + 1
    index_new_node_3 = n_nodes + 2

    # building new elements
    # ---------------------
    n_elements = elements.shape[0]
    # element whose first neighbour is `which`
    green_1 = np.isin(element_to_neighbours[:, 0], which)
    # element whose second neighbour is `which`
    green_2 = np.isin(element_to_neighbours[:, 1], which)
    # element whose third neighbour is `which`
    green_3 = np.isin(element_to_neighbours[:, 2], which)
    # marking `which` as being red-refined
    red = np.zeros(n_elements, dtype=bool)
    red[which] = True
    # the rest of the elements is not marked for refinement
    # TODO check if this can be improved without reduce
    none = np.logical_not(
        np.logical_or.reduce((green_1, green_2,
                              green_3, red)))
    # retreiving new elements' indices
    idx = np.ones(n_elements, dtype=int)
    idx[green_1] = 2
    idx[green_2] = 2
    idx[green_3] = 2
    idx[red] = 4
    idx = np.hstack([0, np.cumsum(idx)])
    new_elements = np.zeros((idx[-1], 3), dtype=int)
    idx = idx[:-1]
    # generate new elements

    # no refinement
    new_elements[idx[none]] = elements[none]

    # TODO check if/how we can remove these `np.any`
    # green refinement (1)
    if np.any(green_1):
        # get the indices of the elements to be green (1) refined
        idx_element_green_1 = np.nonzero(green_1)[0]
        # get the indices of the corresponding new nodes
        _, iloc = ismember.ismember(
            idx_element_green_1, element_to_neighbours[which, :])
        idx_new_node_green_1 = n_nodes + iloc
        # creating and adding the green (1) refined element
        new_elements[
            np.hstack((idx[green_1], idx[green_1]+1)), :] = np.vstack((
                np.column_stack([elements[green_1, 0],
                                idx_new_node_green_1,
                                elements[green_1, 2]]),
                np.column_stack([idx_new_node_green_1,
                                elements[green_1, 1],
                                elements[green_1, 2]])
            ))

    # green refinement (2)
    if np.any(green_2):
        # get the index of the element to be green (2) refined
        idx_element_green_2 = np.nonzero(green_2)[0]
        # get the indices of the corresponding new nodes
        _, iloc = ismember.ismember(
            idx_element_green_2, element_to_neighbours[which, :])
        idx_new_node_green_2 = n_nodes + iloc
        # creating and adding the green (2) refined element
        new_elements[
            np.hstack((idx[green_2], idx[green_2]+1)), :] = np.vstack((
                np.column_stack([elements[green_2, 1],
                                idx_new_node_green_2,
                                elements[green_2, 0]]),
                np.column_stack([idx_new_node_green_2,
                                elements[green_2, 2],
                                elements[green_2, 0]])
            ))

    # green refinement (3)
    if np.any(green_3):
        # get the index of the element to be green (3) refined
        idx_element_green_3 = np.nonzero(green_3)[0]
        # get the indices of the corresponding new nodes
        _, iloc = ismember.ismember(
            idx_element_green_3, element_to_neighbours[which, :])
        idx_new_node_green_3 = n_nodes + iloc
        # creating and adding the green (3) refined element
        new_elements[
            np.hstack((idx[green_3], idx[green_3]+1)), :] = np.vstack((
                np.column_stack([elements[green_3, 2],
                                idx_new_node_green_3,
                                elements[green_3, 1]]),
                np.column_stack([idx_new_node_green_3,
                                elements[green_3, 0],
                                elements[green_3, 1]])
                    ))

    # red refinement
    new_elements[
        np.hstack((
            idx[red], idx[red]+1,
            idx[red]+2, idx[red]+3)), :] = np.vstack((
                np.column_stack([elements[red, 0],
                                 index_new_node_1,
                                 index_new_node_3]),
                np.column_stack([index_new_node_1,
                                 elements[red, 1],
                                 index_new_node_2]),
                np.column_stack([index_new_node_3,
                                 index_new_node_2,
                                 elements[red, 2]]),
                np.column_stack([index_new_node_1,
                                 index_new_node_2,
                                 index_new_node_3])
            ))

    return new_elements


def refineRG_with_element_to_neighbours(coordinates: CoordinatesType,
                                        elements: ElementsType,
                                        which: int,
                                        boundaries: list[BoundaryType],
                                        element_to_neighbours,
                                        to_embed: np.ndarray = np.array([])
                                        ) -> tuple[CoordinatesType,
                                                   ElementsType,
                                                   list[BoundaryType],
                                                   np.ndarray]:
    """
    red refines a single specified element and removes
    hanging nodes by green refining neighbouring elements

    Parameters
    ----------
    coordinates: CoordinatesType
        coordinates of the mesh at hand
    elements: ElementsType
        elements of the mesh at hand
    which: int
        index of the element to be red refined
    boundaries: list[BoundaryType]
        list of boundaries at hand
    element_to_neighbours: np.ndarray
        array whose j-th entry in the i-th row represents the
        index of the element sharing the i-th edge of the j-th element
    to_embed: np.ndarray, optional
        array containing data on the nodes to be linearly interpolated,
        defaults to an empty array

    Returns
    -------
    new_coordinates: CoordinatesType
        coordinates of the refined mesh
    new_elements: ElementsType
        elements of the refined mesh
    new_boundaries: list[BoundaryType]
        boundaries of the refined mesh
    to_embed: np.ndarray
        linearly interpolated data on the refined mesh

    Notes
    -----
    - this routine does not assume the list of boundary
      conditions to be complete.
    - during refinement, three new coordinates are generated,
      these get simply appended to the existing coordinates
    """
    # building (three) new coordinates
    # --------------------------------
    x_y_1 = coordinates[elements[which, [0, 1, 2]], :]
    x_y_2 = coordinates[elements[which, [1, 2, 0]], :]
    new_coordinates = np.vstack([coordinates,
                                 (x_y_1 + x_y_2)/2.])
    # retrieving new nodes' indices
    n_nodes = coordinates.shape[0]

    # interpolating `to_embed`
    # ------------------------
    if to_embed.size > 0:
        interpolated = (to_embed[elements[which, [0, 1, 2]].flatten()] +
                        to_embed[elements[which, [1, 2, 0]].flatten()])/2.
        to_embed = np.hstack([to_embed, interpolated])

    new_elements = refine_elements(elements=elements,
                                   element_to_neighbours=element_to_neighbours,
                                   which=which,
                                   n_nodes=n_nodes)

    new_boundaries = refine_boundaries(
        elements=elements,
        boundaries=boundaries,
        n_nodes=n_nodes,
        which=which,
        element_to_neighbours=element_to_neighbours)

    return new_coordinates, new_elements, new_boundaries, to_embed


# NOTE This functionn is based on the technologies found in [1].
# However, it must not be used by passinng several marked elements
# as elements that have two marked egdes are ingored by this implementation.
#
# [1] S. Funken, D. Praetorius, and P. Wissgott.
#     Efficient Implementation of Adaptive P1-FEM in Matlab
#     http://dx.doi.org/10.2478/cmam-2011-0026
def refineRG_without_element_to_neighbours(
    coordinates: CoordinatesType,
    elements: ElementsType,
    marked_element: int,
    boundaries: list[BoundaryType],
    to_embed: np.ndarray = np.array([])) -> tuple[CoordinatesType,
                                                  ElementsType,
                                                  list[BoundaryType]]:
    """
    refines the mesh according to
    red-green refinement of one single element

    parameters
    ----------
    coordinates: CoordinatesType
    elements: ElementsType
    marked_element: int
        the index of the element to be red refined
    boundaries: list[BoundaryType]
    to_embed: np.ndarray = np.array([])
        vector of values on coordinates to be interpolated
        (canonically embedded) onto the refined mesh

    returns
    -------
    new_coordinates: CoordinatesType
        coordinates of the refined mesh
    new_elements: ElementsType
        elements of the refined mesh
    new_boundaries: list[BoundaryType]
        boundaries of the refines mesh
    embedded_values: np.ndarray
        to_embed interpolated onto the refined mesh

    notes
    -----
    - this function requires to be passed a complete list of
      boundaries, i.e. each edge of the mesh at hand must be
      part of a boundary in `boundaries` (due to implementation
      details in `provide_geometric_data`)

    - this method does the following:
        1. given an element k=marked_element marked for
           refinement
        2. red refine element k
        3. green refine all neighbouring elements (at most three),
           meaning that the new vertex on the edge shared by k
           and its neighbour k' gets connected to the vertex
           in k' opposite to the shared edge
        4. if any new vertex lies on a boundary,
           refine the boundary accordingly
    """
    nE = elements.shape[0]

    # Obtain geometric information on edges
    element2edges, edge2nodes, boundary2edges = \
        provide_geometric_data(elements=elements, boundaries=boundaries)

    # Mark edges for refinement
    edge2newNode = np.zeros(edge2nodes.shape[0], dtype=int)
    edge2newNode[element2edges[marked_element].flatten()] = 1

    new_coordinates, embedded_values = generate_new_nodes(
        edge2newNode=edge2newNode,
        edge2nodes=edge2nodes,
        coordinates=coordinates,
        to_embed=to_embed)

    # refine boundary conditions
    new_boundaries = []
    for k, boundary in enumerate(boundaries):
        if boundary.size:
            new_nodes_on_boundary = edge2newNode[boundary2edges[k]]
            marked_edges = np.nonzero(new_nodes_on_boundary)[0]
            if marked_edges.size:
                boundary = np.vstack(
                    [boundary[np.logical_not(new_nodes_on_boundary), :],
                     np.column_stack([boundary[marked_edges, 0],
                                      new_nodes_on_boundary[marked_edges]]),
                     np.column_stack([new_nodes_on_boundary[marked_edges],
                                      boundary[marked_edges, 1]])])
        new_boundaries.append(boundary)

    # Provide new nodes for refinement of elements
    newNodes = edge2newNode[element2edges]

    # Determine type of refinement for each element
    marked_edges = newNodes != 0

    first_marked = marked_edges[:, 0]
    second_marked = marked_edges[:, 1]
    third_marked = marked_edges[:, 2]

    not_first = np.logical_not(first_marked)
    not_second = np.logical_not(second_marked)
    not_third = np.logical_not(third_marked)

    none = not_first & not_second & not_third
    green1 = first_marked & not_second & not_third
    green2 = not_first & second_marked & not_third
    green3 = not_first & not_second & third_marked
    red = first_marked & second_marked & third_marked

    # generate element numbering for refined mesh
    idx = np.ones(nE, dtype=int)
    idx[green1] = 2  # green(1): green refinement of 1st edge
    idx[green2] = 2  # green(2): green refinement of 2nd edge
    idx[green3] = 2  # green(3): green refinement of 3rd edge
    idx[red] = 4  # red: red refinement
    idx = np.hstack([0, np.cumsum(idx)])

    # generate new elements
    # ---------------------
    new_elements = np.zeros((idx[-1], 3), dtype=int)
    idx = idx[:-1]

    # no refinement
    new_elements[idx[none], :] = elements[none, :]

    # green refinement (1)
    new_elements[np.hstack([idx[green1], 1+idx[green1]]), :] \
        = np.vstack(
            [np.column_stack([
                elements[green1, 0],
                newNodes[green1, 0],
                elements[green1, 2]]),
             np.column_stack([
                 newNodes[green1, 0],
                 elements[green1, 1],
                 elements[green1, 2]])])

    # green refinement (2)
    new_elements[np.hstack([idx[green2], 1+idx[green2]]), :] \
        = np.vstack(
            [np.column_stack([
                elements[green2, 1],
                newNodes[green2, 1],
                elements[green2, 0]]),
             np.column_stack([
                 newNodes[green2, 1],
                 elements[green2, 2],
                 elements[green2, 0]])])

    # green refinement (3)
    new_elements[np.hstack([idx[green3], 1+idx[green3]]), :] \
        = np.vstack(
            [np.column_stack([
                elements[green3, 2],
                newNodes[green3, 2],
                elements[green3, 1]]),
             np.column_stack([
                 newNodes[green3, 2],
                 elements[green3, 0],
                 elements[green3, 1]])])

    # red refinement
    new_elements[np.hstack([idx[red], 1+idx[red],
                            2+idx[red], 3+idx[red]]), :] \
        = np.vstack([
            np.column_stack([elements[red, 0],
                             newNodes[red, 0],
                             newNodes[red, 2]]),
            np.column_stack([newNodes[red, 0],
                             elements[red, 1],
                             newNodes[red, 1]]),
            np.column_stack([newNodes[red, 2],
                             newNodes[red, 1],
                             elements[red, 2]]),
            np.column_stack([newNodes[red, 0],
                             newNodes[red, 1],
                             newNodes[red, 2]])])

    return new_coordinates, new_elements, new_boundaries, embedded_values


def refineRGB(coordinates: CoordinatesType,
              elements: ElementsType,
              marked_elements: np.ndarray,
              boundary_conditions: list[BoundaryType],
              to_embed: np.ndarray = np.array([])
              ) -> tuple[CoordinatesType,
                         ElementsType,
                         list[BoundaryType]]:
    """
    Refines the mesh and boundary conditions based on the
    red-green-blue (RGB) refinement strategy.

    Parameters
    ----------
    coordinates: CoordinatesType
    elements: ElementsType
    marked_elements: np.ndarray
        Indices of the elements to be refined.
    boundary_conditions: list[BoundaryType]
        List of boundary conditions to update.
    to_embed: np.ndarray = np.array([])
        vector of values on coordinates to be interpolated
        (canonically embedded) onto the refined mesh

    Returns
    -------
    new_coordinates: CoordinatesType
        the coordinates of the refined mesh
    new_elements: ElementsType
        the elements of the refined mesh
    new_boundaries: list[BoundaryType]
        The refined boundary conditions
    embedded_values: np.ndarray
        to_embed interpolated onto the refined mesh

    Example
    -------
    >>> coordinates, elements = ...  # Initialize a mesh
    >>> marked_elements = np.array([0, 2, 3, 4])
    >>> boundary_conditions = [BC1, BC2, BC3]  # BCs as `np.ndarray`s
    >>> new_coordinates, new_elements, new_boundary_conditions, _ = refineRGB(
            coordinates, elements,
            marked_elements, boundary_conditions)
    """
    new_coordinates, new_elements, new_boundaries, embedded_values = refineNVB(
        coordinates,
        elements,
        marked_elements,
        boundary_conditions=boundary_conditions,
        sort_for_longest_egde=True,
        to_embed=to_embed)
    return new_coordinates, new_elements, new_boundaries, embedded_values


# TODO refactor s.t. sort_for_longest_egde vanishes, this is an ugly solution
def refineNVB(coordinates: CoordinatesType,
              elements: ElementsType,
              marked_elements: np.ndarray,
              boundary_conditions: list[BoundaryType],
              to_embed: np.ndarray = np.array([]),
              sort_for_longest_egde: bool = False
              ) -> tuple[CoordinatesType,
                         ElementsType,
                         list[BoundaryType],
                         np.ndarray]:
    """
    refines the mesh based on newest vertex bisection (NVB)

    Parameters
    ----------
    coordinates: CoordinatesType
    elements: ElementsType
    marked_elements: np.ndarray
        indices of the elements to be refined
    boundary_conditions: list[BoundaryType]
        list of boundaries to be refined
    to_embed: np.ndarray = np.array([])
        vector of values on coordinates to be interpolated
        (canonically embedded) onto the refined mesh

    Returns
    -------
    new_coordinates: CoordinatesType
        the coordinates of the refined mesh
    new_elements: ElementsType
        the elements of the refined mesh
    new_boundaries: list[BoundaryType]
        the refined boundary conditions
    embedded_values: np.ndarray
        to_embed interpolated onto the refined mesh

    Example
    -------
    >>> coordinates, elements = Mesh(...)  # Initialize a mesh
    >>> marked_elements = np.array([0, 2, 3, 4])
    >>> boundary_conditions = [BC1, BC2, BC3]  # BC's as np.ndarray
    >>> new_coordinates, new_elements, new_boundary_conditions, _ = refineNVB(
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
    element2edges, edge_to_nodes, boundaries_to_edges = provide_geometric_data(
        elements=elements,
        boundaries=boundary_conditions)

    # mark all edges of marked elements for refinement
    edge2newNode = np.zeros(edge_to_nodes.shape[0], dtype=int)
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

    new_coordinates, embedded_values = generate_new_nodes(
        edge2newNode=edge2newNode,
        edge2nodes=edge_to_nodes,
        coordinates=coordinates,
        to_embed=to_embed)

    # refine boundary conditions
    new_boundaries = []
    for k, boundary in enumerate(boundary_conditions):
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
        new_boundaries.append(boundary)

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
    idx = idx[:-1]
    new_elements[idx[none], :] = elements[none, :]
    new_elements[np.hstack([idx[bisec1],
                           1+idx[bisec1]]), :] \
        = np.vstack(
            [np.column_stack([
                elements[bisec1, 2],
                elements[bisec1, 0],
                new_nodes[bisec1, 0]]),
             np.column_stack([
                 elements[bisec1, 1],
                 elements[bisec1, 2],
                 new_nodes[bisec1, 0]])])
    new_elements[np.hstack([idx[bisec12],
                           1+idx[bisec12],
                           2+idx[bisec12]]), :] \
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
    new_elements[np.hstack([idx[bisec13],
                           1+idx[bisec13],
                           2+idx[bisec13]]), :] \
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
        new_elements[np.hstack([idx[bisec123],
                               1+idx[bisec123],
                               2+idx[bisec123],
                               3+idx[bisec123]]), :] \
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
        new_elements[np.hstack([idx[bisec123],
                               1+idx[bisec123],
                               2+idx[bisec123],
                               3+idx[bisec123]]), :] \
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

    return new_coordinates, new_elements, new_boundaries, embedded_values


def refineNVB_edge_based(
    coordinates: CoordinatesType,
    elements: ElementsType,
    boundary_conditions: list[BoundaryType],
    element2edges: np.ndarray,
    edge_to_nodes: np.ndarray,
    boundaries_to_edges: np.ndarray,
    edge2newNode: np.ndarray,
    to_embed: np.ndarray = np.array([])
        ) -> tuple[
            CoordinatesType,
            ElementsType,
            list[BoundaryType],
            np.ndarray]:
    """
    refines the mesh using an edge-based newest vertex bisection (NVB)

    Parameters
    ----------
    coordinates: CoordinatesType
    elements: ElementsType
    boundary_conditions: list[BoundaryType]
        list of boundaries to be refined
    element2edges: np.ndarray
        mapping from elements to edges
    edge_to_nodes: np.ndarray
        mapping from edges to nodes
    boundaries_to_edges: np.ndarray
        mapping from boundaries to edges
    edge2newNode: np.ndarray
        if edge2newNode[k] = 1, the edge corresponding to
        edge_to_nodes[k] is markd for refinement.
        if edge2newNode[k] = 0, the edge corresponding to
        edge_to_nodes[k] is not markd for refinement.
    to_embed: np.ndarray = np.array([])
        vector of values on coordinates to be interpolated
        (canonically embedded) onto the refined mesh, i.e.
        linearly interpolated

    Returns
    -------
    new_coordinates: CoordinatesType
        the coordinates of the refined mesh
    new_elements: ElementsType
        the elements of the refined mesh
    new_boundaries: list[BoundaryType]
        the refined boundary conditions
    embedded_values: np.ndarray
        `to_embed` interpolated onto the refined mesh

    Example
    -------
    >>> element2edges, edge_to_nodes, boundaries_to_edges =
        provide_geometric_data(elements=elements,
        boundaries=boundary_conditions)
    >>> n_unique_edges = edge_to_nodes.shape[0]
    >>> random_marked_edges_indices =
        np.random.randint(0, n_unique_edges, int(n_unique_edges/2))
    >>> marked_edges = np.zeros(n_unique_edges, dtype=int)
    >>> marked_edges[random_marked_edges_indices] = 1
    >>> new_coordinates, new_elements, new_boundaries, embedded_iterate=\
        refineNVB_edge_based(
            coordinates=coordinates,
            elements=elements,
            boundary_conditions=boundaries,
            element2edges=element2edges,
            edge_to_nodes=edge_to_nodes,
            boundaries_to_edges=boundaries_to_edges,
            edge2newNode=marked_edges,
            to_embed=current_iterate)
    """
    n_elements = elements.shape[0]

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

    new_coordinates, embedded_values = generate_new_nodes(
        edge2newNode=edge2newNode,
        edge2nodes=edge_to_nodes,
        coordinates=coordinates,
        to_embed=to_embed)

    # refine boundary conditions
    new_boundaries = []
    for k, boundary in enumerate(boundary_conditions):
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
        new_boundaries.append(boundary)

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
    idx = np.hstack([0, np.cumsum(idx)])

    # generate new elements
    new_elements = np.zeros((idx[-1], 3), dtype=int)
    idx = idx[:-1]
    new_elements[idx[none], :] = elements[none, :]
    new_elements[np.hstack([idx[bisec1],
                           1+idx[bisec1]]), :] \
        = np.vstack(
            [np.column_stack([
                elements[bisec1, 2],
                elements[bisec1, 0],
                new_nodes[bisec1, 0]]),
             np.column_stack([
                 elements[bisec1, 1],
                 elements[bisec1, 2],
                 new_nodes[bisec1, 0]])])
    new_elements[np.hstack([idx[bisec12],
                           1+idx[bisec12],
                           2+idx[bisec12]]), :] \
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
    new_elements[np.hstack([idx[bisec13],
                           1+idx[bisec13],
                           2+idx[bisec13]]), :] \
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
    new_elements[np.hstack([idx[bisec123],
                            1+idx[bisec123],
                            2+idx[bisec123],
                            3+idx[bisec123]]), :] \
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

    return new_coordinates, new_elements, new_boundaries, embedded_values


def refine_single_edge(
        coordinates: CoordinatesType,
        elements: ElementsType,
        edge: np.ndarray,
        to_embed: np.ndarray = np.array([])
        ) -> tuple[CoordinatesType, ElementsType, np.ndarray]:
    """
    refines a single non-boundary edge

    parameters
    ----------
    coordinates: CoordinatesType
        coordinates of the mesh to be refined
    elements: ElementsType
        elements of the mesh to be refined
    edge: np.ndarray
        indices of the edge to be refined, i.e.
        edge = np.array([i, j]), where (i, j)
        represent the indices of the coordintes
        that make up the edge to be refined
    to_embed: np.ndarray = np.array([])

    returns
    -------
    new_coordinates: CoordinatesType
        coordinates of the refined mesh
    new_elements: ElementsType
        elements of the refined mesh
    embedded_values: np.ndarray
        `to_embed` embedded in the refined mesh, i.e.
        linearly interpolated

    details
    -------
    - this method must not be called with boundary edges
    - the new coordinate is located at coordinates[-1]
    """
    i, j = edge[0], edge[1]
    new_coordinate = 0.5 * (coordinates[i, :] + coordinates[j, :])
    new_coordinates = np.vstack([coordinates, new_coordinate])

    marked_elements_indices = np.sum(np.isin(elements, edge), axis=1) == 2
    not_marked_elements_indices = np.logical_not(marked_elements_indices)
    marked_elements = elements[marked_elements_indices]

    if marked_elements.shape[0] != 2:
        # if there are less than two elements marked, the edge marked
        # for refinement is a boundary edge. As this special case remains
        # unnecessary to consider for now (we only consider homogenous
        # dirichlet), we exclude this possibility. Otherwise, we would
        # need to pass the boundary connditions, too, and refine them, too.
        raise ValueError(
            'this function must not be called with boundary edges')

    first_element = marked_elements[0]
    k_1 = first_element[np.logical_not(np.isin(first_element, edge))][0]
    second_element = marked_elements[1]
    k_2 = second_element[np.logical_not(np.isin(second_element, edge))][0]

    first_is_left = False
    for k in range(3):
        if (first_element[k % 3] == i and first_element[(k+1) % 3] == j):
            first_is_left = True
    if first_is_left:
        k_L = k_1
        k_R = k_2
    else:
        k_L = k_2
        k_R = k_1

    # generate element numbering for refined mesh
    n_vertices = coordinates.shape[0]
    untouched_elements = np.copy(elements[not_marked_elements_indices, :])
    TL_1 = np.array([n_vertices, k_L, i])
    TL_2 = np.array([n_vertices, j, k_L])
    TR_1 = np.array([n_vertices, i, k_R])
    TR_2 = np.array([n_vertices, k_R, j])

    new_elements = np.vstack([
        untouched_elements,
        TL_1,
        TL_2,
        TR_1,
        TR_2,
    ])

    if to_embed.size:
        interpolated_value = 0.5*(to_embed[i] + to_embed[j])
        embedded_values = np.hstack([to_embed, interpolated_value])
        return new_coordinates, new_elements, embedded_values

    return new_coordinates, new_elements, to_embed
