import numpy as np
import unittest
from p1afempy.refinement import refineNVB
from triangle_cubature.cubature_rule import CubatureRuleEnum
import random
from p1afempy.solvers import get_stiffness_matrix, get_weighted_stiffness_matrix
from scipy.sparse import csr_matrix


class WeightedStiffnessMatrixTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        random.seed(42)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_weighted_stiffness_matrix(self) -> None:

        a, b = 5.34, 90.32

        def mu(t):
            return 1. + np.exp(-t)
        
        def u_analytical(x, y):
            return a*x + b*y

        # square mesh with side length 2
        # centered at the origin
        coordinates = np.array([
            [-1., -1.],
            [ 1., -1.],
            [ 1.,  1.],
            [-1.,  1.],
            [ 0.,  0.]
        ], dtype=float)
        elements = np.array([
            [0,4,3],
            [0,1,4],
            [1,2,4],
            [4,2,3]
        ], dtype=int)
        dirichlet = np.array([
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0]
        ])
        boundary_conditions = [dirichlet]

        current_iterate = np.array([u_analytical(x, y) for x, y in coordinates])

        n_mesh_refinements = 5
        for _ in range(n_mesh_refinements):
            marked_elements = np.arange(elements.shape[0])
            coordinates, elements, boundary_conditions, current_iterate = \
                refineNVB(
                    coordinates=coordinates,
                    elements=elements,
                    marked_elements=marked_elements,
                    boundary_conditions=boundary_conditions,
                    to_embed=current_iterate)

            weighted_stiffness_matrix = csr_matrix(get_weighted_stiffness_matrix(
                coordinates=coordinates,
                elements=elements,
                u=current_iterate,
                mu=mu
            ))

            stiffness_matrix = csr_matrix(get_stiffness_matrix(
                coordinates=coordinates,
                elements=elements
            ))

            A = weighted_stiffness_matrix
            B = mu(a**2 + b**2)*stiffness_matrix

            np.testing.assert_allclose(A.toarray(), B.toarray(), rtol=1e-12, atol=1e-14)


if __name__ == '__main__':
    unittest.main()
