import numpy as np
import unittest
from p1afempy.data_structures import ElementsType, CoordinatesType


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

    n_refinement_steps = 3
    for _ in range(n_refinement_steps):
        pass
        # TODO refine the mesh

    return elements, coordinates


if __name__ == '__main__':
    unittest.main()
