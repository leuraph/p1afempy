import numpy as np
import unittest
from pathlib import Path
from p1afempy import io_helpers, solvers
from p1afempy.data_structures import CoordinatesType, ElementsType


def test_function(x: float, y: float) -> float:
    x_inside_domain = -1. < x < 1.
    y_inside_domain = -1. < y < 1.
    if (not x_inside_domain) or (not y_inside_domain):
        return 0.
    return min(1. - abs(x), 1. - abs(y))


def evaluate_energy_on_mesh(coordinates: CoordinatesType,
                            elements: ElementsType) -> float:

    stiffness_matrix = solvers.get_stiffness_matrix(
        coordinates=coordinates,
        elements=elements)
    test_function_vector = np.array([
        test_function(x, y) for (x, y) in coordinates])

    return test_function_vector.dot(stiffness_matrix.dot(test_function_vector))


class SanityChecks(unittest.TestCase):

    def test_stiffness_assembly(self) -> None:
        path_to_coordinates = Path('tests/data/sanity_check/coordinates.dat')
        path_to_elements = Path('tests/data/sanity_check/elements.dat')
        coordinates, elements = io_helpers.read_mesh(
            path_to_coordinates=path_to_coordinates,
            path_to_elements=path_to_elements)

        expected_energy = 4.
        computed_energy = evaluate_energy_on_mesh(
            coordinates=coordinates, elements=elements)

        self.assertEquals(expected_energy, computed_energy)


if __name__ == '__main__':
    unittest.main()
