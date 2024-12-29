import numpy as np
from p1afempy.mesh import get_directional_vectors, get_area
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
from p1afempy.data_structures import \
    CoordinatesType, ElementsType, BoundaryConditionType, BoundaryType
from triangle_cubature.cubature_rule import CubatureRuleEnum
from triangle_cubature.rule_factory import get_rule


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
