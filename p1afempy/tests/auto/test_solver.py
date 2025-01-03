import unittest
import numpy as np
import p1afempy.io_helpers as io_helpers
from p1afempy.solvers import solve_laplace, get_mass_matrix_elements, \
    get_right_hand_side
import p1afempy.refinement as refinement
from pathlib import Path
import example_setup
from tests.auto.example_setup import f
from triangle_cubature.cubature_rule import CubatureRuleEnum


class SolverTest(unittest.TestCase):

    def test_solve_laplace(self) -> None:
        path_to_coordinates = Path(
            'tests/data/laplace_example/coordinates.dat')
        path_to_elements = Path('tests/data/laplace_example/elements.dat')
        path_to_neumann = Path('tests/data/laplace_example/neumann.dat')
        path_to_dirichlet = Path('tests/data/laplace_example/dirichlet.dat')
        path_to_matlab_x = Path('tests/data/laplace_example/x.dat')
        path_to_matlab_energy = Path('tests/data/laplace_example/energy.dat')

        coordinates, elements = io_helpers.read_mesh(
            path_to_coordinates=path_to_coordinates,
            path_to_elements=path_to_elements)
        neumann_bc = io_helpers.read_boundary_condition(
            path_to_boundary=path_to_neumann)
        dirichlet_bc = io_helpers.read_boundary_condition(
            path_to_boundary=path_to_dirichlet)
        boundary_conditions = [dirichlet_bc, neumann_bc]

        n_refinements = 3
        for _ in range(n_refinements):
            marked_elements = np.arange(elements.shape[0])
            coordinates, elements, boundary_conditions, _ = refinement.refineNVB(
                coordinates=coordinates,
                elements=elements,
                marked_elements=marked_elements,
                boundary_conditions=boundary_conditions)

        x, energy = solve_laplace(
            coordinates=coordinates, elements=elements,
            dirichlet=boundary_conditions[0],
            neumann=boundary_conditions[1],
            f=example_setup.f, g=example_setup.g, uD=example_setup.uD)

        x_matlab = np.loadtxt(path_to_matlab_x)
        energy_matlab = np.loadtxt(path_to_matlab_energy).reshape((1,))[0]

        self.assertTrue(np.allclose(x, x_matlab))
        self.assertTrue(np.isclose(energy, energy_matlab))

    def test_get_mass_matrix_elements(self) -> None:
        path_to_coordinates = Path(
            'tests/data/ahw_codes_example_mesh/coordinates.dat')
        path_to_elements = Path(
            'tests/data/ahw_codes_example_mesh/elements.dat')
        mesh_ahw_coordinates, mesh_ahw_elements = io_helpers.read_mesh(
            path_to_coordinates=path_to_coordinates,
            path_to_elements=path_to_elements,
            shift_indices=True)

        path_to_expected_I = Path(
            'tests/data/ahw_codes_example_mesh/mass_matrix_I.dat')
        path_to_expected_J = Path(
            'tests/data/ahw_codes_example_mesh/mass_matrix_J.dat')
        path_to_expected_D = Path(
            'tests/data/ahw_codes_example_mesh/mass_matrix_D.dat')
        expected_I = np.loadtxt(path_to_expected_I).astype(np.uint32)
        expected_J = np.loadtxt(path_to_expected_J).astype(np.uint32)
        expected_D = np.loadtxt(path_to_expected_D)


        I, J, D = get_mass_matrix_elements(
            coordinates=mesh_ahw_coordinates,
            elements=mesh_ahw_elements)

        self.assertTrue(np.allclose(I+1, expected_I))
        self.assertTrue(np.allclose(J+1, expected_J))
        self.assertTrue(np.allclose(D, expected_D))

    def test_get_right_hand_side(self):
        """
        This test is a sanity check to test
        two implementations of retreiving the right
        hand side (load) vector, i.e. we test the
        "original" P1AFEM implementation (midpoint rule) against
        an adapted version based on custom quadrature rules.
        """
        path_to_coordinates = Path(
            'tests/data/simple_square_mesh/coordinates.dat')
        path_to_elements = Path(
            'tests/data/simple_square_mesh/elements.dat')
        path_to_boundary_0 = Path(
            'tests/data/simple_square_mesh/square_boundary_0.dat')
        path_to_boundary_1 = Path(
            'tests/data/simple_square_mesh/square_boundary_1.dat')
        coordinates, elements = io_helpers.read_mesh(
            path_to_coordinates=path_to_coordinates,
            path_to_elements=path_to_elements,
            shift_indices=False)
        boundary_0 = io_helpers.read_boundary_condition(
            path_to_boundary=path_to_boundary_0)
        boundary_1 = io_helpers.read_boundary_condition(
            path_to_boundary=path_to_boundary_1)
        boundaries = [boundary_0, boundary_1]

        n_refinements = 5
        for _ in range(n_refinements):
            marked_elements = np.arange(elements.shape[0])
            coordinates, elements, boundaries, _ = refinement.refineNVB(
                coordinates=coordinates,
                elements=elements,
                marked_elements=marked_elements,
                boundary_conditions=boundaries)

        rhs_vector_P1AFEM = get_right_hand_side(
            coordinates=coordinates, elements=elements, f=f)
        rhs_vector_midpoint = get_right_hand_side(
            coordinates=coordinates, elements=elements, f=f,
            cubature_rule=CubatureRuleEnum.MIDPOINT)

        self.assertTrue(np.allclose(rhs_vector_midpoint, rhs_vector_P1AFEM))


if __name__ == "__main__":
    unittest.main()
