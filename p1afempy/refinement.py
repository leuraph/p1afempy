import numpy as np
from p1afempy.mesh import provide_geometric_data
from p1afempy.data_structures import \
    CoordinatesType, ElementsType, BoundaryType


def refineRG(coordinates: CoordinatesType,
             elements: ElementsType,
             marked_element: int,
             boundaries: list[BoundaryType] = []) -> tuple[CoordinatesType,
                                                           ElementsType,
                                                           np.ndarray]:
    """
    refines the mesh according to red-green refinement of one single element
    """
    nE = elements.shape[0]
    markedElements = marked_element

    # Obtain geometric information on edges
    element2edges, edge2nodes, boundary2edges = \
        provide_geometric_data(elements=elements, boundaries=boundaries)

    # Count number of green sibling elements;
    # if isempty(nG)
    #     nG = 0;
    # end
    # nR = nE-nG;

    # Mark edges for refinement
    edge2newNode = np.zeros(edge2nodes.shape[0], dtype=int)
    edge2newNode[element2edges[marked_element].flatten()] = 1

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

    %*** Refine boundary conditions
    varargout = cell(nargout-2,1);
    for j = 1:nargout-2
        boundary = varargin{j};
        if ~isempty(boundary)
            newNodes = edge2newNode(boundary2edges{j})';
            markedEdges = find(newNodes);
            if ~isempty(markedEdges)
                boundary = [boundary(~newNodes,:); ...
                    boundary(markedEdges,1),newNodes(markedEdges); ...
                    newNodes(markedEdges),boundary(markedEdges,2)];
            end
        end
        varargout{j} = boundary;
    end
    %*** Provide new nodes for refinement of elements
    newNodes = edge2newNode(element2edges);
    %*** Determine type of refinement for each red element
    none = find(valR == 0);
    r2red = find(valR == 1);
    r2green1 = find(valR == 2);
    r2green2 = find(valR == 3);
    r2green3 = find(valR == 4);
    %*** Determine type of refinement for each green element
    g2green = nR + find(valG == 0);
    g2red = nR + find(valG == 1);
    g2red1 = nR + find(valG == 2);
    g2red2 = nR + find(valG == 3);
    g2red12 = nR + find(valG == 4);
    %*** Generate element numbering for refined mesh
    rdx = zeros(nR+nG/2,1);
    rdx(none)    = 1;
    rdx([r2red,g2red])   = 4;
    rdx([g2red1,g2red2])  = 3;
    rdx(g2red12) = 2;
    rdx = [1;1+cumsum(rdx)];
    gdx = zeros(size(rdx));
    gdx([r2green1, r2green2, r2green3, g2green, g2red1, g2red2]) = 2;
    gdx(g2red12) = 4;
    gdx = rdx(end)+[0;0+cumsum(gdx)];
    %*** Generate new red elements
    newElements = 1+zeros(gdx(end)-1,3);
    newElements(rdx(none),:) = elements(none,:);
    tmp = [elements(1:nR,:),newNodes(1:nR,:),zeros(nR,2);...
        elements(nR+1:2:end,2),elements(nR+2:2:end,[2,3,1])...
        newNodes(nR+2:2:end,2), newNodes(nR+1:2:end,1:2),...
        newNodes(nR+2:2:end,1)];
    newElements([rdx(r2red),1+rdx(r2red),2+rdx(r2red),3+rdx(r2red)],:) ...
        = [tmp(r2red,[4,5,6]);tmp(r2red,[1,4,6]);tmp(r2red,[2,5,4]);...
        tmp(r2red,[3,6,5])];
    newElements([rdx(g2red),1+rdx(g2red),2+rdx(g2red),3+rdx(g2red)],:) ...
        = [tmp(g2red,[4,5,6]);tmp(g2red,[1,4,6]);...
        tmp(g2red,[4,2,5]);tmp(g2red,[3,6,5])];
    newElements([rdx(g2red1),1+rdx(g2red1),2+rdx(g2red1)],:) ...
        = [tmp(g2red1,[4,5,6]);tmp(g2red1,[4,2,5]);tmp(g2red1,[3,6,5])];
    newElements([rdx(g2red2),1+rdx(g2red2),2+rdx(g2red2)],:) ...
        = [tmp(g2red2,[4,5,6]);tmp(g2red2,[1,4,6]);tmp(g2red2,[3,6,5])];
    newElements([rdx(g2red12),1+rdx(g2red12)],:) ...
        = [tmp(g2red12,[4,5,6]);tmp(g2red12,[3,6,5])];
    %*** New green elements
    newElements([gdx(r2green1),1+gdx(r2green1)],:) ...
        = [tmp(r2green1,[3,1,4]);tmp(r2green1,[4,2,3])];
    newElements([gdx(r2green2),1+gdx(r2green2)],:) ...
        = [tmp(r2green2,[1,2,5]);tmp(r2green2,[5,3,1])];
    newElements([gdx(r2green3),1+gdx(r2green3)],:) ...
        = [tmp(r2green3,[2,3,6]);tmp(r2green3,[6,1,2])];
    newElements([gdx(g2green),1+gdx(g2green)],:) ...
        = [tmp(g2green,[3,1,4]);tmp(g2green,[4,2,3]);];
    newElements([gdx(g2red1),1+gdx(g2red1)],:) ...
        = [tmp(g2red1,[6,1,7]);tmp(g2red1,[7,4,6])];
    newElements([gdx(g2red2),1+gdx(g2red2)],:) ...
        = [tmp(g2red2,[5,4,8]);tmp(g2red2,[8,2,5])];
    newElements([gdx(g2red12),1+gdx(g2red12),2+gdx(g2red12),...
        3+gdx(g2red12)],:) = [tmp(g2red12,[6,1,7]);tmp(g2red12,[7,4,6]);...
        tmp(g2red12,[5,4,8]);tmp(g2red12,[8,2,5])];
    nG = size(newElements,1)-rdx(end)+1;

    return coordinates, elements, boundaries


# TODO refactor s.t. boundary_conditions is optional
def refineRGB(coordinates: CoordinatesType,
              elements: ElementsType,
              marked_elements: np.ndarray,
              boundary_conditions: list[BoundaryType]
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

    Returns
    -------
    new_coordinates: CoordinatesType
        the coordinates of the refined mesh
    new_elements: ElementsType
        the elements of the refined mesh
    new_boundaries: list[BoundaryType]
        The refined boundary conditions

    Example
    -------
    >>> coordinates, elements = ...  # Initialize a mesh
    >>> marked_elements = np.array([0, 2, 3, 4])
    >>> boundary_conditions = [BC1, BC2, BC3]  # BCs as `np.ndarray`s
    >>> new_coordinates, new_elements, new_boundary_conditions = refineRGB(
            coordinates, elements,
            marked_elements, boundary_conditions)
    """
    new_coordinates, new_elements, new_boundaries = refineNVB(
        coordinates,
        elements,
        marked_elements,
        boundary_conditions=boundary_conditions,
        sort_for_longest_egde=True)
    return new_coordinates, new_elements, new_boundaries


# TODO refactor s.t. sort_for_longest_egde vanishes, this is an ugly solution
# TODO refactor s.t. boundary_conditions is optional
def refineNVB(coordinates: CoordinatesType,
              elements: ElementsType,
              marked_elements: np.ndarray,
              boundary_conditions: list[BoundaryType],
              sort_for_longest_egde: bool = False
              ) -> tuple[CoordinatesType,
                         ElementsType,
                         list[BoundaryType]]:
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

    Returns
    -------
    new_coordinates: CoordinatesType
        the coordinates of the refined mesh
    new_elements: ElementsType
        the elements of the refined mesh
    new_boundaries: list[BoundaryType]
        the refined boundary conditions

    Example
    -------
    >>> coordinates, elements = Mesh(...)  # Initialize a mesh
    >>> marked_elements = np.array([0, 2, 3, 4])
    >>> boundary_conditions = [BC1, BC2, BC3]  # BC's as np.ndarray
    >>> new_coordinates, new_elements, new_boundary_conditions = refineNVB(
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

    # generate new nodes
    n_new_nodes = np.count_nonzero(edge2newNode)  # number of new nodes
    # assigning indices to new nodes
    edge2newNode[edge2newNode != 0] = np.arange(
        coordinates.shape[0],
        coordinates.shape[0] + n_new_nodes)
    idx = np.nonzero(edge2newNode)[0]
    new_node_coordinates = (
        coordinates[edge_to_nodes[idx, 0], :] +
        coordinates[edge_to_nodes[idx, 1], :]) / 2.
    new_coordinates = np.vstack([coordinates, new_node_coordinates])

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
