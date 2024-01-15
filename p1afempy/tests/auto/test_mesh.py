import unittest
import p1afempy.mesh as mesh
from pathlib import Path
import numpy as np


class MeshTest(unittest.TestCase):

    @staticmethod
    def get_simple_square_mesh() -> mesh.Mesh:
        path_to_coordinates = Path(
            'tests/data/simple_square_mesh/coordinates.dat')
        path_to_elements = Path(
            'tests/data/simple_square_mesh/elements.dat')
        return mesh.read_mesh(path_to_coordinates=path_to_coordinates,
                              path_to_elements=path_to_elements)

    def test_read_mesh(self):
        z0, z1, z2, z3 = [0., 0.], [1., 0.], [1., 1.], [0., 1.]
        e0, e1 = [0, 1, 2], [0, 2, 3]
        expected_coordinates = np.array([z0, z1, z2, z3])
        expected_elements = np.array([e0, e1])

        domain = MeshTest.get_simple_square_mesh()

        self.assertTrue(np.all(expected_coordinates == domain.coordinates))
        self.assertTrue(np.all(expected_elements == domain.elements))

    def test_provide_geometric_data(self):
        # square-shaped testing domain
        boundary_condition_0 = mesh.read_boundary_condition(
            Path('tests/data/simple_square_mesh/square_boundary_0.dat'))
        boundary_condition_1 = mesh.read_boundary_condition(
            Path('tests/data/simple_square_mesh/square_boundary_1.dat'))
        boundary_conditions = [boundary_condition_0, boundary_condition_1]

        domain = MeshTest.get_simple_square_mesh()
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
            Path('tests/data/l_shape_mesh/l_shape_bc_0.dat'))
        boundary_condition_1 = mesh.read_boundary_condition(
            Path('tests/data/l_shape_mesh/l_shape_bc_1.dat'))
        boundary_condition_2 = mesh.read_boundary_condition(
            Path('tests/data/l_shape_mesh/l_shape_bc_2.dat'))
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

    def test_refineNVB(self) -> None:
        # Square Domain
        boundary_condition_0 = mesh.read_boundary_condition(
            Path('tests/data/simple_square_mesh/square_boundary_0.dat'))
        boundary_condition_1 = mesh.read_boundary_condition(
            Path('tests/data/simple_square_mesh/square_boundary_1.dat'))
        boundary_conditions = [boundary_condition_0, boundary_condition_1]
        domain = MeshTest.get_simple_square_mesh()

        refined_mesh, new_boundaries = mesh.refineNVB(
            mesh=domain,
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
            np.all(expected_refined_coordinates == refined_mesh.coordinates))
        self.assertTrue(
            np.all(expected_refined_elements == refined_mesh.elements))
        self.assertTrue(
            np.all(expected_refined_bc_0 == new_boundaries[0].boundary))
        self.assertTrue(
            np.all(expected_refined_bc_1 == new_boundaries[1].boundary))

        # L-shaped Domain
        path_to_coordinates = Path('tests/data/l_shape_coordinates.dat')
        path_to_elements = Path('tests/data/l_shape_elements.dat')
        path_to_bc_0 = Path('tests/data/l_shape_mesh/l_shape_bc_0.dat')
        path_to_bc_1 = Path('tests/data/l_shape_mesh/l_shape_bc_1.dat')
        path_to_bc_2 = Path('tests/data/l_shape_mesh/l_shape_bc_2.dat')
        domain = mesh.read_mesh(path_to_coordinates=path_to_coordinates,
                                path_to_elements=path_to_elements)
        marked_elements = np.array([0, 1, 3, 5])
        refined_mesh, new_boundaries = \
            mesh.refineNVB(mesh=domain,
                           marked_elements=marked_elements,
                           boundary_conditions=[
                               mesh.read_boundary_condition(path_to_bc_0),
                               mesh.read_boundary_condition(path_to_bc_1),
                               mesh.read_boundary_condition(path_to_bc_2)])

        path_to_refined_coordinates = Path(
            'tests/data/l_shape_coordinates_refined.dat')
        path_to_refined_elements = Path(
            'tests/data/l_shape_elements_refined.dat')
        expected_refined_mesh = mesh.read_mesh(
            path_to_coordinates=path_to_refined_coordinates,
            path_to_elements=path_to_refined_elements)
        refined_bc_0 = mesh.read_boundary_condition(
            Path('tests/data/l_shape_boundary_0_refined.dat'))
        refined_bc_1 = mesh.read_boundary_condition(
            Path('tests/data/l_shape_boundary_1_refined.dat'))
        refined_bc_2 = mesh.read_boundary_condition(
            Path('tests/data/l_shape_boundary_2_refined.dat'))

        self.assertTrue(np.all(
            refined_mesh.coordinates == expected_refined_mesh.coordinates))
        self.assertTrue(np.all(
            refined_mesh.elements == expected_refined_mesh.elements - 1))
        self.assertTrue(np.all(
            new_boundaries[0].boundary == refined_bc_0.boundary - 1))
        self.assertTrue(np.all(
            new_boundaries[1].boundary == refined_bc_1.boundary - 1))
        self.assertTrue(np.all(
            new_boundaries[2].boundary == refined_bc_2.boundary - 1))

    def test_refineRGB(self) -> None:
        # L-shaped Domain
        path_to_coordinates = Path('tests/data/l_shape_coordinates.dat')
        path_to_elements = Path('tests/data/l_shape_elements.dat')
        path_to_bc_0 = Path('tests/data/l_shape_mesh/l_shape_bc_0.dat')
        path_to_bc_1 = Path('tests/data/l_shape_mesh/l_shape_bc_1.dat')
        path_to_bc_2 = Path('tests/data/l_shape_mesh/l_shape_bc_2.dat')
        domain = mesh.read_mesh(path_to_coordinates=path_to_coordinates,
                                path_to_elements=path_to_elements)
        marked_elements = np.array([0, 1, 3, 5])
        refined_mesh, new_boundaries = \
            mesh.refineRGB(mesh=domain,
                           marked_elements=marked_elements,
                           boundary_conditions=[
                               mesh.read_boundary_condition(path_to_bc_0),
                               mesh.read_boundary_condition(path_to_bc_1),
                               mesh.read_boundary_condition(path_to_bc_2)])

        path_to_refined_coordinates = Path(
            'tests/data/refined_rgb/l_shape_coordinates_refined.dat')
        path_to_refined_elements = Path(
            'tests/data/refined_rgb/l_shape_elements_refined.dat')
        expected_refined_mesh = mesh.read_mesh(
            path_to_coordinates=path_to_refined_coordinates,
            path_to_elements=path_to_refined_elements)
        refined_bc_0 = mesh.read_boundary_condition(
            Path('tests/data/refined_rgb/l_shape_boundary_0_refined.dat'))
        refined_bc_1 = mesh.read_boundary_condition(
            Path('tests/data/refined_rgb/l_shape_boundary_1_refined.dat'))
        refined_bc_2 = mesh.read_boundary_condition(
            Path('tests/data/refined_rgb/l_shape_boundary_2_refined.dat'))

        self.assertTrue(np.all(
            refined_mesh.coordinates == expected_refined_mesh.coordinates))
        self.assertTrue(np.all(
            refined_mesh.elements == expected_refined_mesh.elements - 1))
        self.assertTrue(np.all(
            new_boundaries[0].boundary == refined_bc_0.boundary - 1))
        self.assertTrue(np.all(
            new_boundaries[1].boundary == refined_bc_1.boundary - 1))
        self.assertTrue(np.all(
            new_boundaries[2].boundary == refined_bc_2.boundary - 1))


if __name__ == '__main__':
    unittest.main()
