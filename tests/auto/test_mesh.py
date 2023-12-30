import unittest
import mesh
from pathlib import Path
import numpy as np


class MeshTest(unittest.TestCase):

    @staticmethod
    def get_test_mesh() -> mesh.Mesh:
        path_to_coordinates = Path('tests/data/coordinates.dat')
        path_to_elements = Path('tests/data/elements.dat')
        return mesh.read_mesh(path_to_coordinates=path_to_coordinates,
                              path_to_elements=path_to_elements)

    def test_read_mesh(self):
        z0, z1, z2, z3 = [0., 0.], [1., 0.], [1., 1.], [0., 1.]
        e0, e1 = [0, 1, 2], [0, 2, 3]
        expected_coordinates = np.array([z0, z1, z2, z3])
        expected_elements = np.array([e0, e1])

        domain = MeshTest.get_test_mesh()

        self.assertTrue(np.all(expected_coordinates == domain.coordinates))
        self.assertTrue(np.all(expected_elements == domain.elements))

    def test_provide_geometric_data(self):
        # square-shaped testing domain
        boundary_condition_0 = mesh.read_boundary_condition(
            Path('tests/data/square_boundary_0.dat'))
        boundary_condition_1 = mesh.read_boundary_condition(
            Path('tests/data/square_boundary_1.dat'))
        boundary_conditions = [boundary_condition_0, boundary_condition_1]

        domain = MeshTest.get_test_mesh()
        element2edges, edge2nodes, boundaries_to_edges = \
            mesh.provide_geometric_data(domain, boundary_conditions)

        self.assertTrue(np.all(element2edges[0] == [0, 2, 1]))
        self.assertTrue(np.all(element2edges[1] == [1, 3, 4]))
        self.assertEqual(len(element2edges), 2)

        self.assertTrue(np.all(edge2nodes[0] == [0, 1]))
        self.assertTrue(np.all(edge2nodes[1] == [0, 2]))
        self.assertTrue(np.all(edge2nodes[2] == [1, 2]))
        self.assertTrue(np.all(edge2nodes[3] == [2, 3]))
        self.assertTrue(np.all(edge2nodes[4] == [0, 3]))
        self.assertEqual(len(edge2nodes), 5)

        self.assertTrue(np.all(boundaries_to_edges[0] == [0, 2]))
        self.assertTrue(np.all(boundaries_to_edges[1] == [3, 4]))

        # L-shaped testing domain
        boundary_condition_0 = mesh.read_boundary_condition(
            Path('tests/data/l_shape_bc_0.dat'))
        boundary_condition_1 = mesh.read_boundary_condition(
            Path('tests/data/l_shape_bc_1.dat'))
        boundary_condition_2 = mesh.read_boundary_condition(
            Path('tests/data/l_shape_bc_2.dat'))
        boundary_conditions = [boundary_condition_0,
                               boundary_condition_1,
                               boundary_condition_2]
        path_to_coordinates = Path('tests/data/l_shape_coordinates.dat')
        path_to_elements = Path('tests/data/l_shape_elements.dat')
        l_shape_domain = mesh.read_mesh(
            path_to_coordinates=path_to_coordinates,
            path_to_elements=path_to_elements)
        element2edges, edge2nodes, boundaries_to_edges = \
            mesh.provide_geometric_data(l_shape_domain, boundary_conditions)

        self.assertTrue(np.all(element2edges[0] == [0, 5, 1]))
        self.assertTrue(np.all(element2edges[1] == [1, 6, 12]))
        self.assertTrue(np.all(element2edges[2] == [2, 7, 5]))
        self.assertTrue(np.all(element2edges[3] == [3, 8, 7]))
        self.assertTrue(np.all(element2edges[4] == [6, 9, 11]))
        self.assertTrue(np.all(element2edges[5] == [4, 10, 9]))
        self.assertEqual(len(element2edges), 6)

        self.assertTrue(np.all(edge2nodes[0] == [0, 1]))
        self.assertTrue(np.all(edge2nodes[1] == [0, 4]))
        self.assertTrue(np.all(edge2nodes[2] == [1, 2]))
        self.assertTrue(np.all(edge2nodes[3] == [2, 3]))
        self.assertTrue(np.all(edge2nodes[4] == [4, 5]))
        self.assertTrue(np.all(edge2nodes[5] == [1, 4]))
        self.assertTrue(np.all(edge2nodes[6] == [4, 7]))
        self.assertTrue(np.all(edge2nodes[7] == [2, 4]))
        self.assertTrue(np.all(edge2nodes[8] == [3, 4]))
        self.assertTrue(np.all(edge2nodes[9] == [4, 6]))
        self.assertTrue(np.all(edge2nodes[10] == [5, 6]))
        self.assertTrue(np.all(edge2nodes[11] == [6, 7]))
        self.assertTrue(np.all(edge2nodes[12] == [0, 7]))
        self.assertEqual(len(edge2nodes), 13)

        self.assertTrue(np.all(boundaries_to_edges[0] == [0, 2, 3]))
        self.assertTrue(np.all(boundaries_to_edges[1] == [8, 4, 10]))
        self.assertTrue(np.all(boundaries_to_edges[2] == [11, 12]))
        self.assertEqual(len(boundaries_to_edges), 3)

    # def test_refineNVB(self) -> None:
    #     domain = MeshTest.get_test_mesh()
    #     boundary_0 = np.array([[0, 1], [1, 2]], dtype=int)
    #     boundary_1 = np.array([[2, 3], [3, 0]], dtype=int)

    #     marked_elements = np.array([0, 1])

    #     coordinates, newElements, boundaries = mesh.refineNVB(
    #         domain.coordinates, domain.elements,
    #         marked_elements, boundary_0, boundary_1)


if __name__ == '__main__':
    unittest.main()
