import unittest
import p1afempy.refinement as refinement
import p1afempy.io_helpers as io_helpers
from pathlib import Path
import numpy as np


class RefinementTest(unittest.TestCase):

    def test_refineRG(self) -> None:
        # ------------------------
        # reading the initial mesh
        # ------------------------
        path_to_coordinates = Path(
            'tests/data/trefined_rg/coordinates.dat')
        path_to_elements = Path(
            'tests/data/trefined_rg/elements.dat')
        path_to_dirichlet = Path('tests/data/trefined_rg/dirichlet.dat')
        path_to_neumann = Path('tests/data/trefined_rg/neumann.dat')
        coordinates, elements = io_helpers.read_mesh(
            path_to_coordinates=path_to_coordinates,
            path_to_elements=path_to_elements,
            shift_indices=True)
        dirichlet = io_helpers.read_boundary_condition(
            path_to_boundary=path_to_dirichlet,
            shift_indices=True)
        neumann = io_helpers.read_boundary_condition(
            path_to_boundary=path_to_neumann,
            shift_indices=True)
        boundaries = [dirichlet, neumann]

        # ----------------
        # case no_boundary
        # ----------------
        marked_element = 3

        new_coordinates, new_elements, new_boundaries = \
            refinement.refineRG(
                coordinates=coordinates,
                elements=elements,
                marked_element=marked_element,
                boundaries=boundaries)

        path_to_refined_coordinates = Path(
            'tests/data/trefined_rg/case_no_boundary/new_coordinates.dat')
        path_to_refined_elements = Path(
            'tests/data/trefined_rg/case_no_boundary/new_elements.dat')
        expected_coordinates, expected_elements = io_helpers.read_mesh(
            path_to_coordinates=path_to_refined_coordinates,
            path_to_elements=path_to_refined_elements,
            shift_indices=False)
        expected_dirichlet = io_helpers.read_boundary_condition(
            Path('tests/data/trefined_rg/case_no_boundary/new_dirichlet.dat'),
            shift_indices=False)
        expected_neumann = io_helpers.read_boundary_condition(
            Path('tests/data/trefined_rg/case_no_boundary/new_neumann.dat'),
            shift_indices=False)

        self.assertTrue(np.all(
            new_coordinates == expected_coordinates))
        self.assertTrue(np.all(
            new_elements == expected_elements))
        self.assertTrue(np.all(
            new_boundaries[0] == expected_dirichlet))
        self.assertTrue(np.all(
            new_boundaries[1] == expected_neumann))

        # TODO add case dirichlet
        # TODO add case neumann

        # TODO add case with many more elements exploting possible edge-cases


if __name__ == '__main__':
    unittest.main()
