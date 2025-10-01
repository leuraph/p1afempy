import numpy as np
import unittest
from p1afempy.refinement import refineNVB
from triangle_cubature.cubature_rule import CubatureRuleEnum
import random
from p1afempy.solvers import get_weighted_mass_matrix
from scipy.sparse import csr_matrix


class WeightedMassMatrixTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        random.seed(42)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_weighted_mass_matrix(self) -> None:

        coefficients = [
            34.3,
            12.4,
            1.4,
            0.45,
            0.332]

        Uh_star = 12.

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

        current_iterate = np.array([0., 0., 0., 0., Uh_star])

        def phi(x):
            # polynomial
            n_entries = x.shape[0]
            result = np.zeros(n_entries)
            for (k, a) in enumerate(coefficients):
                result += a * x**k
            return result

        def get_exact_result(coefficients: list[float], u_star: float):
            # see hand-written notes from 30.09.2025 for a derivation
            result = 0.
            for (k, a) in enumerate(coefficients):
                result += (a * u_star**k)/((k + 3.)*(k + 4.))
            return 8. * u_star**2 * result

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

            weighted_mass_matrix = csr_matrix(get_weighted_mass_matrix(
                coordinates=coordinates,
                elements=elements,
                current_iterate=current_iterate,
                phi=phi,
                cubature_rule=CubatureRuleEnum.DAYTAYLOR))

            numerical_result = current_iterate.dot(
                weighted_mass_matrix.dot(current_iterate))
            
            exact_result = get_exact_result(
                coefficients=coefficients,
                u_star=Uh_star)
            print(exact_result)

            self.assertAlmostEqual(numerical_result, exact_result)


if __name__ == '__main__':
    unittest.main()
