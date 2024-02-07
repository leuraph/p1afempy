import numpy as np
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


def refineRG(coordinates: CoordinatesType,
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
    this method does the following
    - given an element k=marked_element marked for
      refinement
    - red refine element k
    - green refine all neighbouring elements (at most three),
      meaning that the new vertex on the edge shared by k
      and its neighbour k' gets connected to the vertex
      in k' opposite to the shared edge
    - if any new vertex lies on a boundary,
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


# TODO refactor s.t. boundary_conditions is optional
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
# TODO refactor s.t. boundary_conditions is optional
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
