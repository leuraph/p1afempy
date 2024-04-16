import unittest
import p1afempy.refinement as refinement
import p1afempy.mesh as mesh
import p1afempy.io_helpers as io_helpers
from pathlib import Path
import numpy as np


class RefinementTest(unittest.TestCase):

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
            refinement.refineRG(
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
            refinement.refineRG(
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
            refinement.refineRG(
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
            refinement.refineRG_single(
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


if __name__ == '__main__':
    unittest.main()
