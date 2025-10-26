import numpy as np
import unittest
from p1afempy.data_structures import BoundaryType, ElementsType, CoordinatesType
from p1afempy.refinement import refineNVB
from p1afempy.solvers import get_stiffness_matrix
from p1afempy.solvers import evaluate_on_coordinates
from scipy.sparse import csr_matrix


class GeneralStiffnessMatrixTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_evaluate_on_mesh(self) -> None:
        n_initial_refinements = 3
        n_test_refinements = 5

        elements, coordinates, boundaries = get_initial_mesh()

        for _ in range(n_initial_refinements):
            marked_elements = np.arange(elements.shape[0])
            coordinates, elements, boundaries, _ = \
                refineNVB(coordinates=coordinates,
                                        elements=elements,
                                        marked_elements=marked_elements,
                                        boundary_conditions=boundaries)

        old_coordinates = coordinates
        old_elements = elements
        old_boundaries = boundaries

        n_initial_coordinates = coordinates.shape[0]
        u = np.random.rand(n_initial_coordinates) * 10.

        stiffness_matrix = csr_matrix(get_stiffness_matrix(
            coordinates=coordinates, elements=elements))
        initial_a_norm_squared = u.dot(stiffness_matrix.dot(u))

        for _ in range(n_test_refinements):
            marked_elements = np.arange(elements.shape[0])
            new_coordinates, new_elements, new_boundaries, _ = \
                refineNVB(
                    coordinates=old_coordinates,
                    elements=old_elements,
                    marked_elements=marked_elements,
                    boundary_conditions=old_boundaries)
            stiffness_matrix = csr_matrix(get_stiffness_matrix(
                coordinates=new_coordinates, elements=new_elements))
            
            u = evaluate_on_coordinates(
                u=u,
                elements=old_elements,
                coordinates=old_coordinates,
                r=new_coordinates,
                display_progress_bar=False)
            
            current_a_norm_squared = u.dot(stiffness_matrix.dot(u))

            self.assertAlmostEqual(
                initial_a_norm_squared,
                current_a_norm_squared
            )

            old_coordinates = new_coordinates
            old_elements = new_elements
            old_boundaries = new_boundaries


def get_initial_mesh() -> tuple[
        ElementsType, CoordinatesType, BoundaryType]:
    """
    returns a coarse initial unit square mesh
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

    return elements, coordinates, boundaries


if __name__ == '__main__':
    unittest.main()
