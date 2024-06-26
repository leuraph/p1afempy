import unittest
import p1afempy.mesh as mesh
import p1afempy.refinement as refinement
import p1afempy.io_helpers as io_helpers
from pathlib import Path
import numpy as np


class MeshTest(unittest.TestCase):

    def test_read_mesh(self):
        z0, z1, z2, z3 = [0., 0.], [1., 0.], [1., 1.], [0., 1.]
        e0, e1 = [0, 1, 2], [0, 2, 3]
        expected_coordinates = np.array([z0, z1, z2, z3])
        expected_elements = np.array([e0, e1])

        path_to_coordinates = Path(
            'tests/data/simple_square_mesh/coordinates.dat')
        path_to_elements = Path(
            'tests/data/simple_square_mesh/elements.dat')

        coordinates, elements = io_helpers.read_mesh(
            path_to_coordinates=path_to_coordinates,
            path_to_elements=path_to_elements)

        self.assertTrue(np.all(expected_coordinates == coordinates))
        self.assertTrue(np.all(expected_elements == elements))

    def test_provide_geometric_data(self):
        # square-shaped testing domain
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

        _, elements = io_helpers.read_mesh(
            path_to_coordinates=path_to_coordinates,
            path_to_elements=path_to_elements)

        element2edges, edge2nodes, boundaries_to_edges = \
            mesh.provide_geometric_data(elements=elements,
                                        boundaries=boundary_conditions)

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
        boundary_condition_0 = io_helpers.read_boundary_condition(
            Path('tests/data/l_shape_mesh/l_shape_bc_0.dat'))
        boundary_condition_1 = io_helpers.read_boundary_condition(
            Path('tests/data/l_shape_mesh/l_shape_bc_1.dat'))
        boundary_condition_2 = io_helpers.read_boundary_condition(
            Path('tests/data/l_shape_mesh/l_shape_bc_2.dat'))
        boundary_conditions = [boundary_condition_0,
                               boundary_condition_1,
                               boundary_condition_2]
        path_to_coordinates = Path(
            'tests/data/l_shape_mesh/l_shape_coordinates.dat')
        path_to_elements = Path(
            'tests/data/l_shape_mesh/l_shape_elements.dat')
        _, l_shape_elements = io_helpers.read_mesh(
            path_to_coordinates=path_to_coordinates,
            path_to_elements=path_to_elements)
        element2edges, edge2nodes, boundaries_to_edges = \
            mesh.provide_geometric_data(elements=l_shape_elements,
                                        boundaries=boundary_conditions)

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

    def test_get_area(self) -> None:
        path_to_coordinates = Path(
            'tests/data/ahw_codes_example_mesh/coordinates.dat')
        path_to_elements = Path(
            'tests/data/ahw_codes_example_mesh/elements.dat')
        ahw_coordinates, ahw_elements = io_helpers.read_mesh(
            path_to_coordinates=path_to_coordinates,
            path_to_elements=path_to_elements,
            shift_indices=True)

        path_to_expected_area = Path(
            'tests/data/ahw_codes_example_mesh/area.dat')
        expected_area = np.loadtxt(path_to_expected_area)

        self.assertTrue(np.allclose(
            expected_area, mesh.get_area(coordinates=ahw_coordinates,
                                         elements=ahw_elements)))

    def test_relabel_global_indices(self):
        global_indices = np.array([
            [12, 3, 6],
            [12, 2, 6],
            [12, 3, 15],
            [66, 77, 88],
        ])
        expected_local_indices = np.array([
            [3, 1, 2],
            [3, 0, 2],
            [3, 1, 4],
            [5, 6, 7]
        ])

        unique_indices = np.unique(global_indices)
        global_to_local_index_mapping = mesh.get_global_to_local_index_mapping(
            unique_indices)
        computed_local_indices = global_to_local_index_mapping(global_indices)

        self.assertTrue(
            np.all(expected_local_indices == computed_local_indices))

    def test_complete_boundaries(self) -> None:
        path_to_elements = Path(
            'tests/data/simple_square_mesh/elements.dat')
        elements = io_helpers.read_elements(
            path_to_elements=path_to_elements)

        # test with one missing edge
        # --------------------------
        input_boundary = np.array([
            [0, 1],
            [1, 2],
            [2, 3]
        ])
        expected_artificial_boundary = np.array([
            [3, 0]
        ])

        expected_boundaries = [input_boundary,
                               expected_artificial_boundary]
        calculated_completed_boundaries = mesh.complete_boundaries(
            elements=elements,
            boundaries=[input_boundary])

        for calc, exp in zip(calculated_completed_boundaries,
                             expected_boundaries):
            self.assertTrue(np.all(calc == exp))

        # test with two missing edges
        # ---------------------------
        input_boundary_0 = np.array([
            [0, 1]
        ])
        input_boundary_1 = np.array([
            [2, 3]
        ])
        expected_artificial_boundary = np.array([
            [1, 2],
            [3, 0]
        ])

        expected_boundaries = [input_boundary_0,
                               input_boundary_1,
                               expected_artificial_boundary]
        calculated_completed_boundaries = mesh.complete_boundaries(
            elements=elements,
            boundaries=[input_boundary_0, input_boundary_1])
        for calc, exp in zip(calculated_completed_boundaries,
                             expected_boundaries):
            self.assertTrue(np.all(calc == exp))

        # test with already complete edge
        # -------------------------------
        input_boundary = np.array([
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0]
        ])

        expected_boundaries = [input_boundary]
        calculated_completed_boundaries = mesh.complete_boundaries(
            elements=elements,
            boundaries=[input_boundary])

        for calc, exp in zip(calculated_completed_boundaries,
                             expected_boundaries):
            self.assertTrue(np.all(calc == exp))

        # test with all edges missing
        # ---------------------------
        expected_artificial_boundary = np.array([
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0]
        ])

        expected_boundaries = [expected_artificial_boundary]
        calculated_completed_boundaries = mesh.complete_boundaries(
            elements=elements,
            boundaries=[])

        for calc, exp in zip(calculated_completed_boundaries,
                             expected_boundaries):
            self.assertTrue(np.all(calc == exp))

    def test_get_local_patch(self) -> None:
        base_path = Path('tests/data/get_local_patch')
        path_to_coordinates = base_path / Path('coordinates.dat')
        path_to_elements = base_path / Path('elements.dat')
        path_to_dirichlet = base_path / Path('dirichlet.dat')
        path_to_neumann = base_path / Path('neumann.dat')
        path_to_global_values = base_path / Path('global_values.dat')

        global_coordinates, global_elements = io_helpers.read_mesh(
            path_to_coordinates=path_to_coordinates,
            path_to_elements=path_to_elements)
        global_dirichlet = io_helpers.read_boundary_condition(
            path_to_boundary=path_to_dirichlet)
        global_neumann = io_helpers.read_boundary_condition(
            path_to_boundary=path_to_neumann)
        global_boundaries = [global_dirichlet, global_neumann]
        global_values = np.loadtxt(path_to_global_values)

        element_to_neighbours = mesh.get_element_to_neighbours(
            elements=global_elements)

        # local patch of element not touching any boundary
        # ------------------------------------------------
        local_coordinates, local_elements, \
            local_boundaries, local_values, local_element_to_neighbours, \
            local_which = \
            mesh.get_local_patch(
                coordinates=global_coordinates,
                elements=global_elements,
                boundaries=global_boundaries,
                which_for=3,
                global_values=global_values,
                element_to_neighbours=element_to_neighbours)
        expected_local_coordinates = np.array([
            [0., 0.],
            [1., 0.],
            [2., 0.],
            [1., 1.],
            [2., 1.],
            [2., 2.]
        ])
        expected_local_elements = np.array([
            [3, 4, 5],
            [0, 1, 3],
            [1, 2, 4],
            [4, 3, 1]
        ])
        expected_local_element_to_neighbours = np.array([
            [3, -1, -1],
            [-1, 3, -1],
            [-1, -1, 3],
            [0, 1, 2]
        ])
        expected_local_boundaries = [
            np.array([
                [0, 1], [1, 2]
            ]),
            np.array([
                [2, 4], [4, 5]
            ])]
        expected_local_values = np.array([0.0, 0.1, 0.2, 0.4, 0.5, 0.8])
        self.assertEqual(local_which, 3)
        self.assertTrue(np.all(
            local_coordinates == expected_local_coordinates))
        self.assertTrue(np.all(
            local_elements == expected_local_elements))
        self.assertTrue(
            len(expected_local_boundaries) == len(local_boundaries))
        self.assertTrue(np.all(
            local_element_to_neighbours == expected_local_element_to_neighbours
        ))
        for k in range(len(local_boundaries)):
            local_boundary = local_boundaries[k]
            expected_local_boundary = expected_local_boundaries[k]
            self.assertTrue(np.all(local_boundary == expected_local_boundary))
        self.assertTrue(np.all(local_values == expected_local_values))

        # local patch of element touching both boundaries
        # --------------------------------------------------
        local_coordinates, local_elements, \
            local_boundaries, local_values, local_element_to_neighbours, \
            local_which = \
            mesh.get_local_patch(coordinates=global_coordinates,
                                 elements=global_elements,
                                 boundaries=global_boundaries,
                                 which_for=1,
                                 global_values=global_values,
                                 element_to_neighbours=element_to_neighbours)
        expected_local_elements = np.array([
            [3, 2, 0],
            [0, 1, 3],
        ])
        expected_local_coordinates = np.array([
            [1, 0],
            [2, 0],
            [1, 1],
            [2, 1]
        ])
        expected_local_element_to_neighbours = np.array([
            [-1, -1, 1],
            [-1, -1, 0]
        ])
        expected_local_boundaries = [
            np.array([0, 1]),
            np.array([1, 3])
        ]
        expected_local_values = np.array([0.1, 0.2, 0.4, 0.5])
        self.assertEqual(local_which, 1)
        self.assertTrue(np.all(
            local_coordinates == expected_local_coordinates))
        self.assertTrue(np.all(
            local_elements == expected_local_elements))
        self.assertTrue(np.all(
            expected_local_element_to_neighbours == local_element_to_neighbours
            ))
        self.assertTrue(
            len(expected_local_boundaries) == len(local_boundaries))
        for k in range(len(local_boundaries)):
            local_boundary = local_boundaries[k]
            expected_local_boundary = expected_local_boundaries[k]
            self.assertTrue(np.all(local_boundary == expected_local_boundary))
        self.assertTrue(np.all(local_values == expected_local_values))

        # local patch of element touching dirichlet boundaries
        # -----------------------------------------------
        local_coordinates, local_elements, \
            local_boundaries, local_values, local_element_to_neighbours, \
            local_which = \
            mesh.get_local_patch(coordinates=global_coordinates,
                                 elements=global_elements,
                                 boundaries=global_boundaries,
                                 which_for=0,
                                 global_values=global_values,
                                 element_to_neighbours=element_to_neighbours)
        expected_local_elements = np.array([
            [4, 3, 1],
            [3, 2, 0],
            [0, 1, 3]
        ])
        expected_local_coordinates = np.array([
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
            [2, 1]
        ])
        expected_local_element_to_neighbours = np.array([
            [-1, 2, -1],
            [-1, -1, 2],
            [-1, 0, 1]
        ])
        expected_local_boundaries = [
            np.array([
                [2, 0],
                [0, 1]
            ])]
        expected_local_values = np.array([0.0, 0.1, 0.3, 0.4, 0.5])
        self.assertEqual(local_which, 2)
        self.assertTrue(np.all(
            local_coordinates == expected_local_coordinates))
        self.assertTrue(np.all(
            local_elements == expected_local_elements))
        self.assertTrue(np.all(
            expected_local_element_to_neighbours == local_element_to_neighbours
        ))
        self.assertTrue(
            len(expected_local_boundaries) == len(local_boundaries))
        for k in range(len(local_boundaries)):
            local_boundary = local_boundaries[k]
            expected_local_boundary = expected_local_boundaries[k]
            self.assertTrue(np.all(local_boundary == expected_local_boundary))
        self.assertTrue(np.all(expected_local_values == local_values))

        # local patch of element s.t. local patch inherits no boundary
        # here, we only test that the returned list of boundaries
        # is indeed an empty list
        # ------------------------------------------------------------
        base_path = Path(
            'tests/data/local_patch_touching_no_boundary')
        path_to_coordinates = base_path / Path('coordinates.dat')
        path_to_elements = base_path / Path('elements.dat')
        path_to_boundary = base_path / Path('boundary.dat')
        path_to_global_values = base_path / Path('global_values.dat')
        coordinates, elements = io_helpers.read_mesh(
            path_to_coordinates=path_to_coordinates,
            path_to_elements=path_to_elements)
        boundary = io_helpers.read_boundary_condition(
            path_to_boundary=path_to_boundary)
        global_values = np.loadtxt(path_to_global_values)
        element_to_neighbours = mesh.get_element_to_neighbours(
            elements=elements)

        expected_local_values = np.array([0.1, 0.3, 0.4, 0.5, 0.6, 0.8])

        local_coordinates, local_elements, \
            local_boundaries, local_values, _, local_which = \
            mesh.get_local_patch(coordinates=coordinates,
                                 elements=elements,
                                 boundaries=[boundary],
                                 which_for=6,
                                 element_to_neighbours=element_to_neighbours,
                                 global_values=global_values)

        self.assertTrue(np.all(expected_local_values == local_values))
        self.assertFalse(local_boundaries)

        # local patch edge case:
        # an edge of the actual boundary is not covered by the local patch
        # but the nodes making up the edge are both part of the local patch
        # we consider two cases fo the same setup
        # -----------------------------------------------------------------
        base_path = Path('tests/data/centered_square_mesh')
        path_to_coordinates = base_path / Path('coordinates.dat')
        path_to_elements = base_path / Path('elements.dat')
        path_to_dirichlet_full = base_path / Path('dirichlet_full.dat')
        path_to_dirichlet_single = base_path / Path('dirichlet_single.dat')

        global_coordinates, global_elements = io_helpers.read_mesh(
            path_to_coordinates=path_to_coordinates,
            path_to_elements=path_to_elements)
        global_dirichlet_full = io_helpers.read_boundary_condition(
            path_to_boundary=path_to_dirichlet_full)
        global_dirichlet_single = io_helpers.read_boundary_condition(
            path_to_boundary=path_to_dirichlet_single)
        element_to_neighbours = mesh.get_element_to_neighbours(
            elements=global_elements)

        # we expect the same local elements in both cases
        expected_local_elements = np.array([
            [1, 4, 2],
            [0, 2, 3],
            [0, 1, 2]])

        expected_local_boundary_full = np.array([
            [0, 1],
            [1, 4],
            [3, 0]])

        local_coordinates, local_elements, local_boundaries, _, _, _ = \
            mesh.get_local_patch(
                coordinates=global_coordinates,
                elements=global_elements,
                boundaries=[global_dirichlet_full],
                which_for=0,
                element_to_neighbours=element_to_neighbours)
        # all global coordinates are in the local patch
        self.assertTrue(np.all(local_coordinates == global_coordinates))
        self.assertTrue(np.all(local_elements == expected_local_elements))
        self.assertEqual(len(local_boundaries), 1)
        self.assertTrue(np.all(
            local_boundaries[0] == expected_local_boundary_full))

        local_coordinates, local_elements, local_boundaries, _, _, _ = \
            mesh.get_local_patch(
                coordinates=global_coordinates,
                elements=global_elements,
                boundaries=[global_dirichlet_single],
                which_for=0,
                element_to_neighbours=element_to_neighbours)
        # all global coordinates are in the local patch
        self.assertTrue(np.all(local_coordinates == global_coordinates))
        self.assertTrue(np.all(local_elements == expected_local_elements))
        self.assertEqual(len(local_boundaries), 0)

    def test_get_element_to_neighbours(self):
        base_path = Path('tests/data/get_neighbours')
        path_to_elements = base_path / Path('elements_matlab.dat')
        path_to_expected_output = base_path / Path(
            'element2neighbours_matlab.dat')

        elements = io_helpers.read_elements(path_to_elements=path_to_elements,
                                            shift_indices=True)

        expected_element_to_neighbours = np.loadtxt(
            fname=path_to_expected_output)
        element_to_neighbours = mesh.get_element_to_neighbours(
            elements=elements)

        self.assertEqual(expected_element_to_neighbours.size,
                         element_to_neighbours.size)
        self.assertTrue(np.all(
            expected_element_to_neighbours - 1 == element_to_neighbours))


if __name__ == '__main__':
    unittest.main()
