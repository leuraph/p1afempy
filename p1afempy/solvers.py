import numpy as np
from p1afempy.mesh import get_directional_vectors, get_area
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
from p1afempy.data_structures import \
    CoordinatesType, ElementsType, BoundaryConditionType, BoundaryType
from triangle_cubature.cubature_rule import CubatureRuleEnum
from triangle_cubature.rule_factory import get_rule
from itertools import product
from triangle_cubature.rule_factory import get_rule
from collections.abc import Callable


def get_stiffness_matrix(coordinates: CoordinatesType,
                         elements: ElementsType) -> coo_matrix:
    """
    returns the stiffness matrix for the P1 FEM
    with Legendre basis

    parameters
    ----------
    coordinates: CoordinatesType
    elements: ElementsType

    returns
    -------
    scipy.sparse.coo_matrix: the sparse stiffness matrix
    """
    # vector of element areas 4*|T|
    area4 = 4. * get_area(coordinates=coordinates,
                          elements=elements)

    indices_i = (elements[:, [0, 1, 2, 0, 1, 2, 0, 1, 2]].T).flatten(
        order='F')
    indices_j = (elements[:, [0, 0, 0, 1, 1, 1, 2, 2, 2]].T).flatten(
        order='F')

    d21, d31 = get_directional_vectors(coordinates=coordinates,
                                       elements=elements)
    a = (np.sum(d21*d31, axis=1)/area4)
    b = (np.sum(d31*d31, axis=1)/area4)
    c = (np.sum(d21*d21, axis=1)/area4)

    A = np.vstack([-2.*a+b+c, a-b, a-c, a-b, b, -a, a-c, -a, c])
    return coo_matrix((A.flatten(order='F'),
                       (indices_i, indices_j)))


def get_weighted_mass_matrix(
        coordinates: CoordinatesType,
        elements: ElementsType,
        current_iterate: np.ndarray,
        phi: BoundaryConditionType,
        cubature_rule: CubatureRuleEnum) -> coo_matrix:
    """
    returns a weighted mass matrix in the sense of
    M_ij := int phi(current_iterate) phi_i phi_j,
    where current iterate is a P1FEM function
    given by its values on the nodes
    and phi_j are the standard Lagrange basis functions

    parameters
    ----------
    coordinates: CoordinatesType
    elements: ElementsType
    current_iterate: np.ndarray
    phi: BoundaryConditionType
    cubature_rule: CubatureRuleEnum

    returns
    -------
    weighted_mass_matrix: coo_matrix
        the weighted mass matrix as described above
    """
    n_vertices = coordinates.shape[0]
    n_elements = elements.shape[0]

    z0 = coordinates[elements[:, 0]]
    z1 = coordinates[elements[:, 1]]
    z2 = coordinates[elements[:, 2]]

    dz1 = z1 - z0
    dz2 = z2 - z0

    areas = 0.5 * (dz1[:, 0]*dz2[:, 1] - dz1[:, 1]*dz2[:, 0])

    wip = get_rule(rule=cubature_rule).weights_and_integration_points
    weights, integration_points = wip.weights, wip.integration_points

    # preparing empty array
    a_11 = np.zeros(n_elements)
    a_12 = np.zeros(n_elements)
    a_13 = np.zeros(n_elements)
    a_21 = np.zeros(n_elements)
    a_22 = np.zeros(n_elements)
    a_23 = np.zeros(n_elements)
    a_31 = np.zeros(n_elements)
    a_32 = np.zeros(n_elements)
    a_33 = np.zeros(n_elements)

    u_1 = current_iterate[elements[:, 0]]
    u_2 = current_iterate[elements[:, 1]]
    u_3 = current_iterate[elements[:, 2]]

    for weight, integration_point in zip(weights, integration_points):
        eta, xi = integration_point

        phi_1_hat = 1 - eta - xi
        phi_2_hat = eta
        phi_3_hat = xi

        u_on_integration_points = (
            u_1 * phi_1_hat
            + u_2 * phi_2_hat
            + u_3 * phi_3_hat
        )

        phi_of_u_on_integration_points = phi(u_on_integration_points)

        a_11 += 2*areas*weight*phi_of_u_on_integration_points*phi_1_hat*phi_1_hat
        a_12 += 2*areas*weight*phi_of_u_on_integration_points*phi_1_hat*phi_2_hat
        a_13 += 2*areas*weight*phi_of_u_on_integration_points*phi_1_hat*phi_3_hat
        a_21 += 2*areas*weight*phi_of_u_on_integration_points*phi_2_hat*phi_1_hat
        a_22 += 2*areas*weight*phi_of_u_on_integration_points*phi_2_hat*phi_2_hat
        a_23 += 2*areas*weight*phi_of_u_on_integration_points*phi_2_hat*phi_3_hat
        a_31 += 2*areas*weight*phi_of_u_on_integration_points*phi_3_hat*phi_1_hat
        a_32 += 2*areas*weight*phi_of_u_on_integration_points*phi_3_hat*phi_2_hat
        a_33 += 2*areas*weight*phi_of_u_on_integration_points*phi_3_hat*phi_3_hat

    I_loc = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    J_loc = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])

    A = np.column_stack(
        [a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32, a_33])
    data = A.flatten()
    row = elements[:, I_loc].flatten()
    col = elements[:, J_loc].flatten()

    weighted_mass_matrix = coo_matrix(
        (data, (row, col)), shape=(n_vertices, n_vertices))
    return weighted_mass_matrix


def get_mass_matrix(coordinates: CoordinatesType,
                    elements: ElementsType) -> coo_matrix:
    """
    returns the mass matrix of the mesh provided
    for the P1 FEM with Legendre basis
    """
    I, J, D = get_mass_matrix_elements(coordinates=coordinates,
                                       elements=elements)
    return coo_matrix((D, (I, J)))


def get_mass_matrix_elements(
        coordinates: CoordinatesType,
        elements: ElementsType) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    returns the mass matrix of the mesh provided
    for the P1 FEM with Legendre basis

    returns
    -------
    indices_i: np.ndarray
    indices_j: np.ndarray
    D: np.ndarray
        D[m] represents a mass matrix contribution
        belonging to its (indices_i[m], indices_j[m]) coordinate
    """
    indices_i = (elements[:, [0, 1, 2, 0, 1, 2, 0, 1, 2]].T).flatten(
        order='F')
    indices_j = (elements[:, [0, 0, 0, 1, 1, 1, 2, 2, 2]].T).flatten(
        order='F')

    area = get_area(coordinates=coordinates,
                    elements=elements)
    D = np.vstack(
        [area/6., area/12., area/12.,
         area/12., area/6., area/12.,
         area/12., area/12., area/6.]).flatten(order='F')
    return indices_i, indices_j, D


def get_general_stiffness_matrix_inefficient(
        coordinates: CoordinatesType,
        elements: ElementsType,
        a_11: BoundaryConditionType,
        a_12: BoundaryConditionType,
        a_21: BoundaryConditionType,
        a_22: BoundaryConditionType,
        cubature_rule: CubatureRuleEnum) -> coo_matrix:
    n_vertices = coordinates.shape[0]

    data = []
    row = []
    col = []

    def get_ref_gradient(
            global_index_of_vertex: int,
            element: np.ndarray) -> np.ndarray:
        local_index = np.where(element == global_index_of_vertex)[0]
        if local_index == 0:
            return np.array([-1., -1.])
        if local_index == 1:
            return np.array([1., 0.])
        if local_index == 2:
            return np.array([0., 1.])

    def get_gradient_on_element(
            global_index_of_vertex: int,
            element: np.ndarray,
            DPhi: np.ndarray) -> np.ndarray:
        ref_gradient = get_ref_gradient(
            global_index_of_vertex=global_index_of_vertex,
            element=element)
        ref_gradient_on_element = np.linalg.solve(
            DPhi.transpose(),
            ref_gradient)
        return ref_gradient_on_element

    for i, j in product(range(n_vertices), repeat=2):
        # get the relevant triangles
        i_inside = (elements == i).astype(int)
        j_inside = (elements == j).astype(int)
        i_or_j_inside = i_inside + j_inside

        relevant_elements_bool = np.sum(i_or_j_inside, axis=1) == 2
        relevant_elements = elements[relevant_elements_bool]

        if len(relevant_elements) == 0:
            continue

        for element in relevant_elements:
            z0 = coordinates[element[0]].reshape(1, 2)
            z1 = coordinates[element[1]].reshape(1, 2)
            z2 = coordinates[element[2]].reshape(1, 2)

            DPhi = np.column_stack([(z1 - z0).flatten(), (z2 - z0).flatten()])

            area = np.linalg.det(DPhi) / 2.

            dphi_i = get_gradient_on_element(
                global_index_of_vertex=i, element=element, DPhi=DPhi)
            dphi_j = get_gradient_on_element(
                global_index_of_vertex=j, element=element, DPhi=DPhi)

            A = np.zeros((2, 2))

            wip = get_rule(rule=cubature_rule).weights_and_integration_points
            weights, integration_points = wip.weights, wip.integration_points
            for weight, integration_point in zip(weights, integration_points):
                eta, xi = integration_point
                transformed_integration_point = z0 + eta*(z1-z0) + xi*(z2-z0)
                a_11_on_transformed_point = a_11(transformed_integration_point)
                a_12_on_transformed_point = a_12(transformed_integration_point)
                a_21_on_transformed_point = a_21(transformed_integration_point)
                a_22_on_transformed_point = a_22(transformed_integration_point)
                A[0, 0] += a_11_on_transformed_point[0] * weight
                A[0, 1] += a_12_on_transformed_point[0] * weight
                A[1, 0] += a_21_on_transformed_point[0] * weight
                A[1, 1] += a_22_on_transformed_point[0] * weight

            row.append(i)
            col.append(j)

            data.append(-2. * area * dphi_i.dot(A.dot(dphi_j)))

    data = np.array(data)
    row = np.array(row)
    col = np.array(col)

    general_stiffness_matrix = coo_matrix(
        (data, (row, col)), shape=(n_vertices, n_vertices))
    return general_stiffness_matrix


def get_general_stiffness_matrix(
        coordinates: CoordinatesType,
        elements: ElementsType,
        a_11: BoundaryConditionType,
        a_12: BoundaryConditionType,
        a_21: BoundaryConditionType,
        a_22: BoundaryConditionType,
        cubature_rule: CubatureRuleEnum) -> coo_matrix:
    """
    returns the stiffness matrix corresponding to the term
    nabla( A(x) nabla u(x)),
    where
    A(x) = [
        [a_11(x), a_12(x)],
        [a_21(x), a_22(x)]]

    note
    ----
    we understand that this code is not readable.
    here, we trade readability for speed.
    """
    n_vertices = coordinates.shape[0]
    n_elements = elements.shape[0]

    z0 = coordinates[elements[:, 0]]
    z1 = coordinates[elements[:, 1]]
    z2 = coordinates[elements[:, 2]]

    dz1 = z1 - z0
    dz2 = z2 - z0

    alpha = dz2[:, 1]
    beta = -dz2[:, 0]
    gamma = -dz1[:, 1]
    delta = dz1[:, 0]

    areas = 0.5 * (dz1[:, 0]*dz2[:, 1] - dz1[:, 1]*dz2[:, 0])

    wip = get_rule(rule=cubature_rule).weights_and_integration_points
    weights, integration_points = wip.weights, wip.integration_points

    a = np.zeros(n_elements)
    b = np.zeros(n_elements)
    c = np.zeros(n_elements)
    d = np.zeros(n_elements)

    for weight, integration_point in zip(weights, integration_points):
        eta, xi = integration_point
        transformed_integration_points = z0 + eta*dz1 + xi*dz2
        a += weight * a_11(transformed_integration_points)
        b += weight * a_12(transformed_integration_points)
        c += weight * a_21(transformed_integration_points)
        d += weight * a_22(transformed_integration_points)

    b11 = (-1./(2.*areas))*(
        a * alpha * alpha +
        b * beta * alpha +
        c * alpha * beta +
        d * beta * beta)
    b12 = (-1./(2.*areas))*(
        a * alpha * gamma +
        b * delta * alpha +
        c * gamma * beta +
        d * beta * delta
    )
    b21 = (-1./(2.*areas))*(
        a * alpha * gamma +
        b * beta * gamma +
        c * alpha * delta +
        d * beta * delta
    )
    b22 = (-1./(2.*areas))*(
        a * gamma * gamma +
        b * delta * gamma +
        c * gamma * delta +
        d * delta * delta
    )

    A = np.column_stack([
        b11 + b12 + b21 + b22,
        -(b11 + b21),
        -(b12 + b22),
        -(b11 + b12),
        b11,
        b12,
        -(b21 + b22),
        b21,
        b22
    ])

    I_loc = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    J_loc = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])

    data = A.flatten()
    row = elements[:, I_loc].flatten()
    col = elements[:, J_loc].flatten()

    general_stiffness_matrix = coo_matrix(
        (data, (row, col)), shape=(n_vertices, n_vertices))
    return general_stiffness_matrix


def get_right_hand_side(coordinates: CoordinatesType,
                        elements: ElementsType,
                        f: BoundaryConditionType,
                        cubature_rule: CubatureRuleEnum = None):
    """
    returns the load vector for the P1 FEM with Legendre basis

    parameters
    ----------
    coordinates: CoordinatesType
    elements: ElementsType
    f: BoundaryConditionType
        the function for which to evaluate the load vector

    returns
    -------
    b: np.ndarray
        the P1 FEM load vector of f on the mesh at hand

    notes
    -----
    if `cubature_rule` is `None` (default value),
    the load vector F_i := int f(x)phi_i(x) dx
    is approximated as sum_T |T| * f(sT) * phi_i(sT),
    where sT denotes the center of mass of triangle T.
    otherwise, the integral is approximated using the
    cubature rule specified.
    """
    if cubature_rule is not None:
        return get_right_hand_side_using_quadrature_rule(
            coordinates=coordinates,
            elements=elements,
            f=f,
            cubature_rule=cubature_rule)
    # vector of element areas 4*|T|
    area4 = 4. * get_area(coordinates=coordinates,
                          elements=elements)

    # assembly of right-hand side
    d21, d31 = get_directional_vectors(coordinates=coordinates,
                                       elements=elements)
    fsT = f((coordinates[elements[:, 0], :]+(d21+d31) / 3))
    b = np.bincount(
        elements.flatten(order='F'),
        weights=np.tile(area4*fsT/12., (3, 1)).flatten(),
        minlength=coordinates.shape[0])
    return b


def integrate_composition_nonlinear_with_fem(
        f: Callable[[float], float],
        u: np.ndarray,
        coordinates: CoordinatesType,
        elements: ElementsType,
        cubature_rule: CubatureRuleEnum) -> float:
    """
    numerically approximates the integral
    int_Omega f(u(x)) dx,
    where u lives in the P1 FEM space of the
    mesh at hand and f:R -> R is any
    (non-linear) function

    parameters
    ----------
    f: BoundaryConditionType
        a general function f:R->R
    u: np.ndarray
        P1FEM function represented as array
        of its values on the `coordinates`
    coordinates: CoordinatesType
        coordinates of the mesh at hand
    elements: ElementsType
        elements of the mesh at hand
    cubature_rule: CubatureRuleEnum
        cubature rule used to numerically
        approximate the integral
    """
    areas = get_area(coordinates=coordinates, elements=elements)
    n_elements = elements.shape[0]

    wip = get_rule(rule=cubature_rule).weights_and_integration_points
    weights, integration_points = wip.weights, wip.integration_points

    # initializing empty container
    L = np.zeros(n_elements, dtype=float)
    for weight, integration_point in zip(weights, integration_points):
        eta, xi = integration_point

        u_on_transformed_interation_points \
            = u[elements[:, 0]]*(1-eta-xi) + u[elements[:, 1]]*eta + u[elements[:, 2]]*xi
        L += weight * f(u_on_transformed_interation_points) * 2. * areas
    
    return np.sum(L)


def get_load_vector_of_composition_nonlinear_with_fem(
        f: Callable[[float], float],
        u: np.ndarray,
        coordinates: CoordinatesType,
        elements: ElementsType,
        cubature_rule: CubatureRuleEnum) -> float:
    """
    numerically approximates an array F with entries
    F_i := int_Omega f(u(x)) phi_i(x) dx,
    where u lives in the P1 FEM space of the
    mesh at hand, f:R -> R is any
    (non-linear) function, and phi_i
    are the standard Lagrange hat functions

    parameters
    ----------
    f: BoundaryConditionType
        a general function f:R->R
    u: np.ndarray
        P1FEM function represented as array
        of its values on the `coordinates`
    coordinates: CoordinatesType
        coordinates of the mesh at hand
    elements: ElementsType
        elements of the mesh at hand
    cubature_rule: CubatureRuleEnum
        cubature rule used to numerically
        approximate the integral
    """
    areas = get_area(coordinates=coordinates, elements=elements)
    n_elements = elements.shape[0]
    n_coordinates = coordinates.shape[0]

    wip = get_rule(rule=cubature_rule).weights_and_integration_points
    weights, integration_points = wip.weights, wip.integration_points

    u_0 = u[elements[:, 0]]
    u_1 = u[elements[:, 1]]
    u_2 = u[elements[:, 2]]

    # initializing empty container
    L = np.zeros_like(elements, dtype=float)
    for weight, integration_point in zip(weights, integration_points):
        eta, xi = integration_point

        phi = np.array([1.-eta-xi, eta, xi])

        u_at_transformed_integration_points = u_0*(1.-eta-xi) + u_1*eta + u_2*xi
        f_on_integration_points = f(u_at_transformed_integration_points)

        L += (
            2. *
            weight *
            areas.reshape((n_elements, 1)) *
            phi *
            f_on_integration_points.reshape((n_elements, 1)))

        Phi = np.bincount(
            elements.flatten(),
            weights=L.flatten(),
            minlength=n_coordinates)

    return Phi


def get_right_hand_side_using_quadrature_rule(
        coordinates: CoordinatesType,
        elements: ElementsType,
        f: BoundaryConditionType,
        cubature_rule: CubatureRuleEnum):
    """
    returns the load vector for the P1 FEM with Legendre basis
    using the specified cubature rule

    parameters
    ----------
    coordinates: CoordinatesType
    elements: ElementsType
    f: BoundaryConditionType
        the function for which to evaluate the load vector
    cubature_rule: CubatureRuleEnum
        cubature rule used to approximate integrals
        such as $\int f(x) phi_i(x) dx$

    returns
    -------
    b: np.ndarray
        the P1 FEM load vector of f on the mesh at hand
    """
    areas = get_area(coordinates=coordinates, elements=elements)
    n_elements = elements.shape[0]
    n_coordinates = coordinates.shape[0]

    wip = get_rule(rule=cubature_rule).weights_and_integration_points
    weights, integration_points = wip.weights, wip.integration_points

    z_0 = coordinates[elements[:, 0], :]
    z_1 = coordinates[elements[:, 1], :]
    z_2 = coordinates[elements[:, 2], :]

    # initializing empty container
    L = np.zeros_like(elements, dtype=float)
    for weight, integration_point in zip(weights, integration_points):
        eta, xi = integration_point

        phi = np.array([1.-eta-xi, eta, xi])

        transformed_integration_points = (
            z_0 + eta * (z_1 - z_0) + xi * (z_2 - z_0))
        f_on_integration_points = f(transformed_integration_points)

        L += (
            2. *
            weight *
            areas.reshape((n_elements, 1)) *
            phi *
            f_on_integration_points.reshape((n_elements, 1)))

        b = np.bincount(
            elements.flatten(),
            weights=L.flatten(),
            minlength=n_coordinates)

    return b


def apply_neumann(neumann_bc: BoundaryType,
                  coordinates: CoordinatesType,
                  g: BoundaryConditionType,
                  b: np.ndarray):
    """
    applies neuman boundary conditions to b and returns new b
    """
    # TODO channge b in place, do not return it or
    # at least check if this version generates computational overhead
    cn1 = coordinates[neumann_bc[:, 0], :]
    cn2 = coordinates[neumann_bc[:, 1], :]
    gmE = g((cn1+cn2)/2)
    b = b + np.bincount(
        neumann_bc.flatten(order='F'),
        weights=np.tile(
            np.sqrt(np.sum(np.square(cn2-cn1), axis=1))*gmE/2.,
            (2, 1)).flatten(), minlength=b.size)
    return b


def solve_laplace(coordinates: CoordinatesType,
                  elements: ElementsType,
                  dirichlet: BoundaryType,
                  neumann: BoundaryType,
                  f: BoundaryConditionType,
                  g: BoundaryConditionType,
                  uD: BoundaryConditionType,
                  cubature_rule: CubatureRuleEnum = None
                  ) -> tuple[np.ndarray, float]:
    """
    solves the laplace equation, i.e.

    -Delta u = f, on Omega
    u = uD, on Gamma_D
    du/dn = g, on Gamma_N

    on the provided mesh using P1 FEM with Legendre basis

    parameters
    ----------
    coordinates: CoordinatesType
    elements: ElementsType
    dirichlet: BoundaryType
        the dirichlet boundary of the problem
    neumann: BoundaryType
        the neumann boundary of the problem
    f: BoundaryConditionType
        the right-hand-side function (volume force) of the problem
    g: BoundaryConditionType
        the neumann boundary function, i.e.
        u(x) = g(x) on Gamma_N
    uD: BoundaryConditionType
        the dirichlet boundary function, i.e.
        du/dn(x) = uD(x) on Gamma_D

    returns
    -------
    x: np.ndarray
        Nx2 array representing the solution of the
        given laplace problem on the mesh provided
    energy: float
        the energy (A-norm: x.T*A*x) of the solution found

    notes
    -----
    the functions f, g, and uD are all expected to be callable like
    f(coordinates), where coordinates is an (n_coordinates x 2) array
    """
    n_coordinates = coordinates.shape[0]
    x = np.zeros(n_coordinates)

    A = get_stiffness_matrix(coordinates=coordinates,
                             elements=elements)

    # prescribe values at dirichlet nodes
    unique_dirichlet = np.unique(dirichlet)
    x[unique_dirichlet] = uD((coordinates[unique_dirichlet, :]))

    b = get_right_hand_side(coordinates=coordinates,
                            elements=elements, f=f,
                            cubature_rule=cubature_rule) - A.dot(x)
    if neumann.size > 0:
        b = apply_neumann(neumann_bc=neumann,
                          coordinates=coordinates,
                          g=g, b=b)

    # computation of P1-FEM approximation
    freenodes = np.setdiff1d(
        np.arange(0, n_coordinates), unique_dirichlet, assume_unique=True)
    A = csc_matrix(A)
    x[freenodes] = spsolve(A[freenodes, :][:, freenodes],
                           b[freenodes],
                           use_umfpack=True)
    # compute energy || grad(uh) ||^2 of discrete solution
    energy = x.dot(A.dot(x))

    return x, energy


def evaluate_on_coordinates(
        u: np.ndarray,
        elements: ElementsType,
        coordinates: CoordinatesType,
        r: CoordinatesType
) -> np.ndarray:
    """
    evaluates the P1FEM function `u`
    living on the mesh given by
    `coordinates` and `elements`
    on the coordinates given by `r`

    parameters
    ----------
    u: np.ndarray
        a P1 FEM vector living on the mesh
        given by `coordinates_H` and `elements_H`
    elements: ElementsType
        the elements of the mesh we wish to transfer from
    coordinates: CoordinatesType
        the coordinates of the mesh we wish to transfer from
    r: CoordinatesType
        the coordinates on which we wish
        to evaluate u_H in P1(T_H)
    
    notes
    -----
    let N denote the number of coordinates in `r`.
    then, this routine returns a vector `u_tilde`
    of length N, given by u_tilde_j := u(z_j).
    """

    determinants = 2.*get_area(
        coordinates=coordinates,
        elements=elements)
    
    x1 = coordinates[elements[:, 0], 0]
    x2 = coordinates[elements[:, 1], 0]
    x3 = coordinates[elements[:, 2], 0]
    y1 = coordinates[elements[:, 0], 1]
    y2 = coordinates[elements[:, 1], 1]
    y3 = coordinates[elements[:, 2], 1]

    u_tilde = np.zeros(r.shape[0])
    for k, z in enumerate(r):
        x, y = z

        lambdas_1 = (
            (y2 - y3)*(x - x3)
            +
            (x3 - x2)*(y - y3)
        )/determinants

        lambdas_2 = (
            (y3 - y1)*(x - x3)
            +
            (x1 - x3)*(y - y3)
        )/determinants

        lambdas_3 = 1. - lambdas_1 - lambdas_2

        lambdas = np.column_stack([
            lambdas_1, lambdas_2, lambdas_3
        ])

        indicators = np.sum(lambdas >= 0., axis=1)

        index_of_element = np.argmax(indicators)

        l1, l2, l3 = lambdas[index_of_element, :]

        u_1, u_2, u_3 = u[elements[index_of_element, :]]

        u_tilde[k] = (
            l1 * u_1
            +
            l2 * u_2
            +
            l3 * u_3
        )

    return u_tilde
