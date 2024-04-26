import unittest
import p1afempy.refinement as refinement
import p1afempy.mesh as mesh
import p1afempy.io_helpers as io_helpers
from pathlib import Path
import numpy as np


class RefinementTest(unittest.TestCase):

    def test_refineNVB(self) -> None:
        # Square Domain
        boundary_condition_0 = io_helpers.read_boundary_condition(
            Path('tests/data/simple_square_mesh/square_boundary_0.dat'))
        boundary_condition_1 = io_helpers.read_boundary_condition(
            Path('tests/data/simple_square_mesh/square_boundary_1.dat'))
        boundary_conditions = [boundary_condition_0,
                               boundary_condition_1]

        path_to_coordinates = Path(
            'tests/data/simple_square_mesh/coordinates.dat')
        path_to_elements = Path(
            'tests/data/simple_square_mesh/elements.dat')

        coordinates, elements = io_helpers.read_mesh(
            path_to_coordinates=path_to_coordinates,
            path_to_elements=path_to_elements)

        refined_coordinates, refined_elements, new_boundaries, _ = \
            refinement.refineNVB(coordinates=coordinates,
                                 elements=elements,
                                 marked_elements=np.array([0, 1]),
                                 boundary_conditions=boundary_conditions)

        expected_refined_coordinates = np.array([[0., 0.],
                                                 [1., 0.],
                                                 [1., 1.],
                                                 [0., 1.],
                                                 [0.5, 0.],
                                                 [0.5, 0.5],
                                                 [1., 0.5],
                                                 [0.5, 1.],
                                                 [0., 0.5]])
        expected_refined_elements = np.array([[4, 2, 5],
                                              [0, 4, 5],
                                              [4, 1, 6],
                                              [2, 4, 6],
                                              [5, 3, 8],
                                              [0, 5, 8],
                                              [5, 2, 7],
                                              [3, 5, 7]], dtype=int)
        expected_refined_bc_0 = np.array([[0, 4],
                                          [1, 6],
                                          [4, 1],
                                          [6, 2]], dtype=int)
        expected_refined_bc_1 = np.array([[2, 7],
                                          [3, 8],
                                          [7, 3],
                                          [8, 0]], dtype=int)
        self.assertTrue(
            np.all(expected_refined_coordinates == refined_coordinates))
        self.assertTrue(
            np.all(expected_refined_elements == refined_elements))
        self.assertTrue(
            np.all(expected_refined_bc_0 == new_boundaries[0]))
        self.assertTrue(
            np.all(expected_refined_bc_1 == new_boundaries[1]))

        # L-shaped Domain
        path_to_coordinates = Path(
            'tests/data/l_shape_mesh/l_shape_coordinates.dat')
        path_to_elements = Path(
            'tests/data/l_shape_mesh/l_shape_elements.dat')
        path_to_bc_0 = Path('tests/data/l_shape_mesh/l_shape_bc_0.dat')
        path_to_bc_1 = Path('tests/data/l_shape_mesh/l_shape_bc_1.dat')
        path_to_bc_2 = Path('tests/data/l_shape_mesh/l_shape_bc_2.dat')
        coordinates, elements = io_helpers.read_mesh(
            path_to_coordinates=path_to_coordinates,
            path_to_elements=path_to_elements)
        marked_elements = np.array([0, 1, 3, 5])
        refined_coordinates, refined_elements, new_boundaries, _ = \
            refinement.refineNVB(
                coordinates=coordinates,
                elements=elements,
                marked_elements=marked_elements,
                boundary_conditions=[
                    io_helpers.read_boundary_condition(path_to_bc_0),
                    io_helpers.read_boundary_condition(path_to_bc_1),
                    io_helpers.read_boundary_condition(path_to_bc_2)])

        path_to_refined_coordinates = Path(
            'tests/data/refined_nvb/l_shape_coordinates_refined.dat')
        path_to_refined_elements = Path(
            'tests/data/refined_nvb/l_shape_elements_refined.dat')
        expected_coordinates, expected_elements = io_helpers.read_mesh(
            path_to_coordinates=path_to_refined_coordinates,
            path_to_elements=path_to_refined_elements)
        refined_bc_0 = io_helpers.read_boundary_condition(
            Path('tests/data/refined_nvb/l_shape_boundary_0_refined.dat'))
        refined_bc_1 = io_helpers.read_boundary_condition(
            Path('tests/data/refined_nvb/l_shape_boundary_1_refined.dat'))
        refined_bc_2 = io_helpers.read_boundary_condition(
            Path('tests/data/refined_nvb/l_shape_boundary_2_refined.dat'))

        self.assertTrue(np.all(
            refined_coordinates == expected_coordinates))
        self.assertTrue(np.all(
            refined_elements == expected_elements - 1))
        self.assertTrue(np.all(
            new_boundaries[0] == refined_bc_0 - 1))
        self.assertTrue(np.all(
            new_boundaries[1] == refined_bc_1 - 1))
        self.assertTrue(np.all(
            new_boundaries[2] == refined_bc_2 - 1))

    def test_refineRGB(self) -> None:
        # L-shaped Domain
        path_to_coordinates = Path(
            'tests/data/l_shape_mesh/l_shape_coordinates.dat')
        path_to_elements = Path(
            'tests/data/l_shape_mesh/l_shape_elements.dat')
        path_to_bc_0 = Path('tests/data/l_shape_mesh/l_shape_bc_0.dat')
        path_to_bc_1 = Path('tests/data/l_shape_mesh/l_shape_bc_1.dat')
        path_to_bc_2 = Path('tests/data/l_shape_mesh/l_shape_bc_2.dat')
        coordinates, elements = io_helpers.read_mesh(
            path_to_coordinates=path_to_coordinates,
            path_to_elements=path_to_elements)
        marked_elements = np.array([0, 1, 3, 5])
        refined_coordinates, refined_elements, new_boundaries, _ = \
            refinement.refineRGB(
                coordinates=coordinates,
                elements=elements,
                marked_elements=marked_elements,
                boundary_conditions=[
                    io_helpers.read_boundary_condition(path_to_bc_0),
                    io_helpers.read_boundary_condition(path_to_bc_1),
                    io_helpers.read_boundary_condition(path_to_bc_2)])

        path_to_refined_coordinates = Path(
            'tests/data/refined_rgb/l_shape_coordinates_refined.dat')
        path_to_refined_elements = Path(
            'tests/data/refined_rgb/l_shape_elements_refined.dat')
        expected_coordinates, expected_elements = io_helpers.read_mesh(
            path_to_coordinates=path_to_refined_coordinates,
            path_to_elements=path_to_refined_elements)
        refined_bc_0 = io_helpers.read_boundary_condition(
            Path('tests/data/refined_rgb/l_shape_boundary_0_refined.dat'))
        refined_bc_1 = io_helpers.read_boundary_condition(
            Path('tests/data/refined_rgb/l_shape_boundary_1_refined.dat'))
        refined_bc_2 = io_helpers.read_boundary_condition(
            Path('tests/data/refined_rgb/l_shape_boundary_2_refined.dat'))

        self.assertTrue(np.all(
            refined_coordinates == expected_coordinates))
        self.assertTrue(np.all(
            refined_elements == expected_elements - 1))
        self.assertTrue(np.all(
            new_boundaries[0] == refined_bc_0 - 1))
        self.assertTrue(np.all(
            new_boundaries[1] == refined_bc_1 - 1))
        self.assertTrue(np.all(
            new_boundaries[2] == refined_bc_2 - 1))

    def test_refineRG(self) -> None:
        # ------------------------
        # reading the initial mesh
        # ------------------------
        path_to_coordinates = Path(
            'tests/data/refine_rg/coordinates.dat')
        path_to_elements = Path(
            'tests/data/refine_rg/elements.dat')
        path_to_dirichlet = Path('tests/data/refine_rg/dirichlet.dat')
        path_to_neumann = Path('tests/data/refine_rg/neumann.dat')
        coordinates, elements = io_helpers.read_mesh(
            path_to_coordinates=path_to_coordinates,
            path_to_elements=path_to_elements,
            shift_indices=False)
        dirichlet = io_helpers.read_boundary_condition(
            path_to_boundary=path_to_dirichlet,
            shift_indices=False)
        neumann = io_helpers.read_boundary_condition(
            path_to_boundary=path_to_neumann,
            shift_indices=False)
        boundaries = [dirichlet, neumann]

        # ----------------
        # case no_boundary
        # ----------------
        marked_element = 3

        new_coordinates, new_elements, new_boundaries, _ = \
            refinement.refineRG_without_element_to_neighbours(
                coordinates=coordinates,
                elements=elements,
                marked_element=marked_element,
                boundaries=boundaries)

        path_to_refined_coordinates = Path(
            'tests/data/refine_rg/case_no_boundary/new_coordinates.dat')
        path_to_refined_elements = Path(
            'tests/data/refine_rg/case_no_boundary/new_elements.dat')
        expected_coordinates, expected_elements = io_helpers.read_mesh(
            path_to_coordinates=path_to_refined_coordinates,
            path_to_elements=path_to_refined_elements,
            shift_indices=False)
        expected_dirichlet = io_helpers.read_boundary_condition(
            Path('tests/data/refine_rg/case_no_boundary/new_dirichlet.dat'),
            shift_indices=False)
        expected_neumann = io_helpers.read_boundary_condition(
            Path('tests/data/refine_rg/case_no_boundary/new_neumann.dat'),
            shift_indices=False)

        self.assertTrue(np.all(
            new_coordinates == expected_coordinates))
        self.assertTrue(np.all(
            new_elements == expected_elements))
        self.assertTrue(np.all(
            new_boundaries[0] == expected_dirichlet))
        self.assertTrue(np.all(
            new_boundaries[1] == expected_neumann))

        # --------------
        # case dirichlet
        # --------------
        marked_element = 0

        new_coordinates, new_elements, new_boundaries, _ = \
            refinement.refineRG_without_element_to_neighbours(
                coordinates=coordinates,
                elements=elements,
                marked_element=marked_element,
                boundaries=boundaries)

        path_to_refined_coordinates = Path(
            'tests/data/refine_rg/case_dirichlet/new_coordinates.dat')
        path_to_refined_elements = Path(
            'tests/data/refine_rg/case_dirichlet/new_elements.dat')
        expected_coordinates, expected_elements = io_helpers.read_mesh(
            path_to_coordinates=path_to_refined_coordinates,
            path_to_elements=path_to_refined_elements,
            shift_indices=False)
        expected_dirichlet = io_helpers.read_boundary_condition(
            Path('tests/data/refine_rg/case_dirichlet/new_dirichlet.dat'),
            shift_indices=False)
        expected_neumann = io_helpers.read_boundary_condition(
            Path('tests/data/refine_rg/case_dirichlet/new_neumann.dat'),
            shift_indices=False)

        self.assertTrue(np.all(
            new_coordinates == expected_coordinates))
        self.assertTrue(np.all(
            new_elements == expected_elements))
        self.assertTrue(np.all(
            new_boundaries[0] == expected_dirichlet))
        self.assertTrue(np.all(
            new_boundaries[1] == expected_neumann))

        # ------------
        # case neumann
        # ------------
        marked_element = 5

        new_coordinates, new_elements, new_boundaries, _ = \
            refinement.refineRG_without_element_to_neighbours(
                coordinates=coordinates,
                elements=elements,
                marked_element=marked_element,
                boundaries=boundaries)

        path_to_refined_coordinates = Path(
            'tests/data/refine_rg/case_neumann/new_coordinates.dat')
        path_to_refined_elements = Path(
            'tests/data/refine_rg/case_neumann/new_elements.dat')
        expected_coordinates, expected_elements = io_helpers.read_mesh(
            path_to_coordinates=path_to_refined_coordinates,
            path_to_elements=path_to_refined_elements,
            shift_indices=False)
        expected_dirichlet = io_helpers.read_boundary_condition(
            Path('tests/data/refine_rg/case_neumann/new_dirichlet.dat'),
            shift_indices=False)
        expected_neumann = io_helpers.read_boundary_condition(
            Path('tests/data/refine_rg/case_neumann/new_neumann.dat'),
            shift_indices=False)

        self.assertTrue(np.all(
            new_coordinates == expected_coordinates))
        self.assertTrue(np.all(
            new_elements == expected_elements))
        self.assertTrue(np.all(
            new_boundaries[0] == expected_dirichlet))
        self.assertTrue(np.all(
            new_boundaries[1] == expected_neumann))

        # TODO add case with many more elements exploting possible edge-cases

    def test_refineRG_single(self) -> None:
        # ------------------------
        # reading the initial mesh
        # ------------------------
        path_to_coordinates = Path(
            'tests/data/refine_rg/coordinates.dat')
        path_to_elements = Path(
            'tests/data/refine_rg/elements.dat')
        path_to_dirichlet = Path('tests/data/refine_rg/dirichlet.dat')
        path_to_neumann = Path('tests/data/refine_rg/neumann.dat')
        coordinates, elements = io_helpers.read_mesh(
            path_to_coordinates=path_to_coordinates,
            path_to_elements=path_to_elements,
            shift_indices=False)
        dirichlet = io_helpers.read_boundary_condition(
            path_to_boundary=path_to_dirichlet,
            shift_indices=False)
        neumann = io_helpers.read_boundary_condition(
            path_to_boundary=path_to_neumann,
            shift_indices=False)
        boundaries = [dirichlet, neumann]
        element_to_neighbours = mesh.get_element_to_neighbours(
            elements=elements)

        # ----------------
        # case no_boundary
        # ----------------
        marked_element = 3

        new_coordinates, new_elements, new_boundaries, _ = \
            refinement.refineRG_with_element_to_neighbours(
                coordinates=coordinates,
                elements=elements,
                which=marked_element,
                boundaries=boundaries,
                element_to_neighbours=element_to_neighbours)

        path_to_refined_coordinates = Path(
            'tests/data/refine_rg/case_no_boundary/new_coordinates.dat')
        path_to_refined_elements = Path(
            'tests/data/refine_rg/case_no_boundary/new_elements.dat')
        expected_coordinates, expected_elements = io_helpers.read_mesh(
            path_to_coordinates=path_to_refined_coordinates,
            path_to_elements=path_to_refined_elements,
            shift_indices=False)
        expected_dirichlet = io_helpers.read_boundary_condition(
            Path('tests/data/refine_rg/case_no_boundary/new_dirichlet.dat'),
            shift_indices=False)
        expected_neumann = io_helpers.read_boundary_condition(
            Path('tests/data/refine_rg/case_no_boundary/new_neumann.dat'),
            shift_indices=False)

        self.assertTrue(np.all(
            new_coordinates == expected_coordinates))
        self.assertTrue(np.all(
            new_elements == expected_elements))
        self.assertTrue(np.all(
            new_boundaries[0] == expected_dirichlet))
        self.assertTrue(np.all(
            new_boundaries[1] == expected_neumann))

        # --------------
        # case dirichlet
        # --------------
        marked_element = 0

        new_coordinates, new_elements, new_boundaries, _ = \
            refinement.refineRG_with_element_to_neighbours(
                coordinates=coordinates,
                elements=elements,
                which=marked_element,
                boundaries=boundaries,
                element_to_neighbours=element_to_neighbours)

        path_to_refined_coordinates = Path(
            'tests/data/refine_rg/case_dirichlet/new_coordinates.dat')
        path_to_refined_elements = Path(
            'tests/data/refine_rg/case_dirichlet/new_elements.dat')
        expected_coordinates, expected_elements = io_helpers.read_mesh(
            path_to_coordinates=path_to_refined_coordinates,
            path_to_elements=path_to_refined_elements,
            shift_indices=False)
        expected_dirichlet = io_helpers.read_boundary_condition(
            Path('tests/data/refine_rg/case_dirichlet/new_dirichlet.dat'),
            shift_indices=False)
        expected_neumann = io_helpers.read_boundary_condition(
            Path('tests/data/refine_rg/case_dirichlet/new_neumann.dat'),
            shift_indices=False)

        self.assertTrue(np.all(
            new_coordinates == expected_coordinates))
        self.assertTrue(np.all(
            new_elements == expected_elements))
        self.assertTrue(np.all(
            new_boundaries[0] == expected_dirichlet))
        self.assertTrue(np.all(
            new_boundaries[1] == expected_neumann))

        # ------------
        # case neumann
        # ------------
        marked_element = 5

        new_coordinates, new_elements, new_boundaries, _ = \
            refinement.refineRG_with_element_to_neighbours(
                coordinates=coordinates,
                elements=elements,
                which=marked_element,
                boundaries=boundaries,
                element_to_neighbours=element_to_neighbours)

        path_to_refined_coordinates = Path(
            'tests/data/refine_rg/case_neumann/new_coordinates.dat')
        path_to_refined_elements = Path(
            'tests/data/refine_rg/case_neumann/new_elements.dat')
        expected_coordinates, expected_elements = io_helpers.read_mesh(
            path_to_coordinates=path_to_refined_coordinates,
            path_to_elements=path_to_refined_elements,
            shift_indices=False)
        expected_dirichlet = io_helpers.read_boundary_condition(
            Path('tests/data/refine_rg/case_neumann/new_dirichlet.dat'),
            shift_indices=False)
        expected_neumann = io_helpers.read_boundary_condition(
            Path('tests/data/refine_rg/case_neumann/new_neumann.dat'),
            shift_indices=False)

        self.assertTrue(np.all(
            new_coordinates == expected_coordinates))
        self.assertTrue(np.all(
            new_elements == expected_elements))
        self.assertTrue(np.all(
            new_boundaries[0] == expected_dirichlet))
        self.assertTrue(np.all(
            new_boundaries[1] == expected_neumann))


if __name__ == '__main__':
    unittest.main()
