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

    def test_provide_geometric_data(self):
        path_to_coordinates = Path('tests/data/coordinates.dat')
        path_to_elements = Path('tests/data/elements.dat')

        boundary_0 = np.array([[0, 1], [1, 2]], dtype=int)
        boundary_1 = np.array([[2, 3], [3, 0]], dtype=int)
        
        domain = mesh.read_mesh(path_to_coordinates=path_to_coordinates,
                                path_to_elements=path_to_elements)
        element2edges, edge2nodes, boundaries_to_edges = mesh.provide_geometric_data(
            domain, boundary_0, boundary_1)

        self.assertTrue( np.all(element2edges[0] == [0, 1, 2] ))
        self.assertTrue( np.all(element2edges[1] == [2, 3, 4] ))
        self.assertEqual(len(element2edges), 2)

        self.assertTrue( np.all( edge2nodes[0] == [0, 1] ) )
        self.assertTrue( np.all( edge2nodes[1] == [1, 2] ) )
        self.assertTrue( np.all( edge2nodes[2] == [0, 2] ) )
        self.assertTrue( np.all( edge2nodes[3] == [2, 3] ) )
        self.assertTrue( np.all( edge2nodes[4] == [0, 3] ) )
        self.assertEqual(len(edge2nodes), 5)

        self.assertTrue(np.all(boundaries_to_edges[0] == [0, 1]))
        self.assertTrue(np.all(boundaries_to_edges[1] == [3, 4]))


if __name__ == '__main__':
    unittest.main()
