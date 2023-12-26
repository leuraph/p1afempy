import unittest
import mesh
from pathlib import Path
import numpy as np


class MeshTest(unittest.TestCase):
    def test_read_mesh(self):
        path_to_coordinates = Path('tests/data/coordinates.dat')
        path_to_elements = Path('tests/data/elements.dat')

        z0, z1, z2, z3 = [0., 0.], [1., 0.], [1., 1.], [0., 1.]
        e0, e1 = [0, 1, 2], [0, 2, 3]
        expected_coordinates = np.array([z0, z1, z2, z3])
        expected_elements = np.array([e0, e1])

        domain = mesh.read_mesh(path_to_coordinates=path_to_coordinates,
                                path_to_elements=path_to_elements)
        
        self.assertTrue(np.all(expected_coordinates == domain.coordinates))
        self.assertTrue(np.all(expected_elements == domain.elements))


if __name__ == '__main__':
    unittest.main()
