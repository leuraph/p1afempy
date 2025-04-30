import numpy as np
import unittest
from p1afempy.data_structures import ElementsType, CoordinatesType
from p1afempy.refinement import refineNVB


class GeneralStiffnessMatrixTest(unittest.TestCase):

    def test_identity(self) -> None:
        elements, coordinates = get_small_mesh()


def get_small_mesh() -> tuple[ElementsType, CoordinatesType]:
    """
    returns a relatively coarse mesh

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

    n_refinement_steps = 3
    for _ in range(n_refinement_steps):
        n_elements = elements.shape[0]
        marked = np.arange(n_elements)
        coordinates, elements, boundaries, _ = refineNVB(
            coordinates=coordinates,
            elements=elements,
            marked_elements=marked,
            boundary_conditions=boundaries)

    return elements, coordinates


if __name__ == '__main__':
    unittest.main()
