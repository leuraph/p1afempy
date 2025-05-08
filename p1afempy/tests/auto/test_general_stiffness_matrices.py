import numpy as np
import unittest
from p1afempy.data_structures import ElementsType, CoordinatesType
from p1afempy.refinement import refineNVB
from p1afempy.solvers import \
    get_stiffness_matrix, get_general_stiffness_matrix, \
        get_general_stiffness_matrix_inefficient
from triangle_cubature.cubature_rule import CubatureRuleEnum
import random


class GeneralStiffnessMatrixTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        random.seed(42)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_general_stiffness_matrix_with_identity(self) -> None:
        max_n_vertices = 100

        elements, coordinates = get_small_mesh(
            max_n_vertices=max_n_vertices)

        def a_11(coordinates: CoordinatesType) -> np.ndarray:
            n_coordinates = coordinates.shape[0]
            return -np.ones(n_coordinates, dtype=float)

        def a_22(coordinates: CoordinatesType) -> np.ndarray:
            n_coordinates = coordinates.shape[0]
            return -np.ones(n_coordinates, dtype=float)

        def a_12(coordinates: CoordinatesType) -> np.ndarray:
            n_coordinates = coordinates.shape[0]
            return np.zeros(n_coordinates, dtype=float)

        def a_21(coordinates: CoordinatesType) -> np.ndarray:
            n_coordinates = coordinates.shape[0]
            return np.zeros(n_coordinates, dtype=float)

        general_stiffness_matrix = get_general_stiffness_matrix(
            coordinates=coordinates,
            elements=elements,
            a_11=a_11,
            a_12=a_12,
            a_21=a_21,
            a_22=a_22,
            cubature_rule=CubatureRuleEnum.DAYTAYLOR)

        stiffness_matrix = get_stiffness_matrix(
            coordinates=coordinates, elements=elements)

        # ------------------------------------------------------------
        # sanity check: we can compare sparse matrices in this way
        # ref: https://stackoverflow.com/a/30685839/15004717
        sanity_check = (stiffness_matrix != stiffness_matrix).nnz == 0
        self.assertTrue(sanity_check)
        # ------------------------------------------------------------

        matrices_agree = \
            (stiffness_matrix != general_stiffness_matrix).nnz == 0
        self.assertTrue(matrices_agree)

    def test_inefficient_general_stiffness_matrix_with_identity(self) -> None:
        max_n_vertices = 100

        elements, coordinates = get_small_mesh(
            max_n_vertices=max_n_vertices)

        def a_11(coordinates: CoordinatesType) -> np.ndarray:
            n_coordinates = coordinates.shape[0]
            return -np.ones(n_coordinates, dtype=float)

        def a_22(coordinates: CoordinatesType) -> np.ndarray:
            n_coordinates = coordinates.shape[0]
            return -np.ones(n_coordinates, dtype=float)

        def a_12(coordinates: CoordinatesType) -> np.ndarray:
            n_coordinates = coordinates.shape[0]
            return np.zeros(n_coordinates, dtype=float)

        def a_21(coordinates: CoordinatesType) -> np.ndarray:
            n_coordinates = coordinates.shape[0]
            return np.zeros(n_coordinates, dtype=float)

        general_stiffness_matrix = get_general_stiffness_matrix_inefficient(
            coordinates=coordinates,
            elements=elements,
            a_11=a_11,
            a_12=a_12,
            a_21=a_21,
            a_22=a_22,
            cubature_rule=CubatureRuleEnum.DAYTAYLOR)

        stiffness_matrix = get_stiffness_matrix(
            coordinates=coordinates, elements=elements)

        # ------------------------------------------------------------
        # sanity check: we can compare sparse matrices in this way
        # ref: https://stackoverflow.com/a/30685839/15004717
        sanity_check = (stiffness_matrix != stiffness_matrix).nnz == 0
        self.assertTrue(sanity_check)
        # ------------------------------------------------------------

        matrices_agree = np.allclose(
            stiffness_matrix.toarray(),
            general_stiffness_matrix.toarray()
        )
        self.assertTrue(matrices_agree)

    def test_general_stiffness_matrix_inefficient_vs_vectorized(self) -> None:
        max_n_vertices = 100

        elements, coordinates = get_small_mesh(
            max_n_vertices=max_n_vertices)

        def a_11(coordinates: CoordinatesType) -> np.ndarray:
            xs, ys = coordinates[:, 0], coordinates[:, 1]
            return np.sin(xs) * np.cos(ys)

        def a_22(coordinates: CoordinatesType) -> np.ndarray:
            xs, ys = coordinates[:, 0], coordinates[:, 1]
            return np.sin(xs) * np.sin(ys)

        def a_12(coordinates: CoordinatesType) -> np.ndarray:
            xs, ys = coordinates[:, 0], coordinates[:, 1]
            return np.cos(xs) * np.cos(ys)

        def a_21(coordinates: CoordinatesType) -> np.ndarray:
            xs, ys = coordinates[:, 0], coordinates[:, 1]
            return np.sin(xs) * np.cos(ys) + np.sin(ys) * np.cos(xs)

        general_stiffness_matrix_vect = get_general_stiffness_matrix(
            coordinates=coordinates,
            elements=elements,
            a_11=a_11,
            a_12=a_12,
            a_21=a_21,
            a_22=a_22,
            cubature_rule=CubatureRuleEnum.DAYTAYLOR)

        general_stiffness_matrix_ineff = \
            get_general_stiffness_matrix_inefficient(
                coordinates=coordinates,
                elements=elements,
                a_11=a_11,
                a_12=a_12,
                a_21=a_21,
                a_22=a_22,
                cubature_rule=CubatureRuleEnum.DAYTAYLOR)

        matrices_agree = np.allclose(
            general_stiffness_matrix_vect.toarray(),
            general_stiffness_matrix_ineff.toarray()
        )
        self.assertTrue(matrices_agree)


def get_small_mesh(max_n_vertices: int = 100) -> tuple[
        ElementsType, CoordinatesType]:
    """
    returns a relatively coarse mesh

    notes
    -----
    the mesh is generated by a non-homogenoues
    NVB mesh refinement

    returns
    -------
    elements: ElementsType
    coordinates: CoordinatesType
    """
    coordinates = np.array([
        [0., 0.],
        [1., 0.],
        [1., 1.],
        [0., 1.]
    ])

    elements = np.array([
        [0, 1, 2],
        [0, 2, 3]
    ])

    boundary = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0]
    ])
    boundaries = [boundary]
    while True:
        # randomly mark 10% of elements for refinement in each step
        fraction_of_elements_to_refine = 0.1
        n_elements = elements.shape[0]
        # at least mark one element for refinement
        n_elements_to_refine = max(
            [int(n_elements*fraction_of_elements_to_refine), 1])
        marked = random.sample(
            list(np.arange(n_elements)), k=n_elements_to_refine)

        tmp_coordinates, tmp_elements, tmp_boundaries, _ = refineNVB(
            coordinates=coordinates,
            elements=elements,
            marked_elements=marked,
            boundary_conditions=boundaries)

        n_vertices = tmp_coordinates.shape[0]
        max_n_vertices_exceeded = n_vertices > max_n_vertices
        if max_n_vertices_exceeded:
            return elements, coordinates

        coordinates = tmp_coordinates
        elements = tmp_elements
        boundaries = tmp_boundaries


if __name__ == '__main__':
    unittest.main()
