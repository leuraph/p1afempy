import unittest
import p1afempy.mesh as mesh
import p1afempy.refinement as refinement
import p1afempy.io_helpers as io_helpers
from pathlib import Path
import numpy as np


class MeshTest(unittest.TestCase):

    @staticmethod
    # TODO Refactor the calls to get_simple_square_mesh
    def get_simple_square_mesh() -> tuple[np.ndarray, np.ndarray]:
        path_to_coordinates = Path(
            'tests/data/simple_square_mesh/coordinates.dat')
        path_to_elements = Path(
            'tests/data/simple_square_mesh/elements.dat')
        return io_helpers.read_mesh(path_to_coordinates=path_to_coordinates,
                                    path_to_elements=path_to_elements)

    def test_read_mesh(self):
        z0, z1, z2, z3 = [0., 0.], [1., 0.], [1., 1.], [0., 1.]
        e0, e1 = [0, 1, 2], [0, 2, 3]
        expected_coordinates = np.array([z0, z1, z2, z3])
        expected_elements = np.array([e0, e1])

        coordinates, elements = MeshTest.get_simple_square_mesh()

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

        _, elements = MeshTest.get_simple_square_mesh()
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

    def test_refineNVB(self) -> None:
        # Square Domain
        boundary_condition_0 = io_helpers.read_boundary_condition(
            Path('tests/data/simple_square_mesh/square_boundary_0.dat'))
        boundary_condition_1 = io_helpers.read_boundary_condition(
            Path('tests/data/simple_square_mesh/square_boundary_1.dat'))
        boundary_conditions = [boundary_condition_0,
                               boundary_condition_1]
        coordinates, elements = MeshTest.get_simple_square_mesh()

        refined_coordinates, refined_elements, new_boundaries, _ = \
            refinement.refineNVB(coordinates=coordinates,
                                 elements=elements,
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
            np.all(expected_refined_coordinates == refined_coordinates))
        self.assertTrue(
            np.all(expected_refined_elements == refined_elements))
        self.assertTrue(
            np.all(expected_refined_bc_0 == new_boundaries[0]))
        self.assertTrue(
            np.all(expected_refined_bc_1 == new_boundaries[1]))

        # L-shaped Domain
        path_to_coordinates = Path(
            'tests/data/l_shape_mesh/l_shape_coordinates.dat')
        path_to_elements = Path(
            'tests/data/l_shape_mesh/l_shape_elements.dat')
        path_to_bc_0 = Path('tests/data/l_shape_mesh/l_shape_bc_0.dat')
        path_to_bc_1 = Path('tests/data/l_shape_mesh/l_shape_bc_1.dat')
        path_to_bc_2 = Path('tests/data/l_shape_mesh/l_shape_bc_2.dat')
        coordinates, elements = io_helpers.read_mesh(
            path_to_coordinates=path_to_coordinates,
            path_to_elements=path_to_elements)
        marked_elements = np.array([0, 1, 3, 5])
        refined_coordinates, refined_elements, new_boundaries, _ = \
            refinement.refineNVB(
                coordinates=coordinates,
                elements=elements,
                marked_elements=marked_elements,
                boundary_conditions=[
                    io_helpers.read_boundary_condition(path_to_bc_0),
                    io_helpers.read_boundary_condition(path_to_bc_1),
                    io_helpers.read_boundary_condition(path_to_bc_2)])

        path_to_refined_coordinates = Path(
            'tests/data/refined_nvb/l_shape_coordinates_refined.dat')
        path_to_refined_elements = Path(
            'tests/data/refined_nvb/l_shape_elements_refined.dat')
        expected_coordinates, expected_elements = io_helpers.read_mesh(
            path_to_coordinates=path_to_refined_coordinates,
            path_to_elements=path_to_refined_elements)
        refined_bc_0 = io_helpers.read_boundary_condition(
            Path('tests/data/refined_nvb/l_shape_boundary_0_refined.dat'))
        refined_bc_1 = io_helpers.read_boundary_condition(
            Path('tests/data/refined_nvb/l_shape_boundary_1_refined.dat'))
        refined_bc_2 = io_helpers.read_boundary_condition(
            Path('tests/data/refined_nvb/l_shape_boundary_2_refined.dat'))

        self.assertTrue(np.all(
            refined_coordinates == expected_coordinates))
        self.assertTrue(np.all(
            refined_elements == expected_elements - 1))
        self.assertTrue(np.all(
            new_boundaries[0] == refined_bc_0 - 1))
        self.assertTrue(np.all(
            new_boundaries[1] == refined_bc_1 - 1))
        self.assertTrue(np.all(
            new_boundaries[2] == refined_bc_2 - 1))

    def test_refineRGB(self) -> None:
        # L-shaped Domain
        path_to_coordinates = Path(
            'tests/data/l_shape_mesh/l_shape_coordinates.dat')
        path_to_elements = Path(
            'tests/data/l_shape_mesh/l_shape_elements.dat')
        path_to_bc_0 = Path('tests/data/l_shape_mesh/l_shape_bc_0.dat')
        path_to_bc_1 = Path('tests/data/l_shape_mesh/l_shape_bc_1.dat')
        path_to_bc_2 = Path('tests/data/l_shape_mesh/l_shape_bc_2.dat')
        coordinates, elements = io_helpers.read_mesh(
            path_to_coordinates=path_to_coordinates,
            path_to_elements=path_to_elements)
        marked_elements = np.array([0, 1, 3, 5])
        refined_coordinates, refined_elements, new_boundaries, _ = \
            refinement.refineRGB(
                coordinates=coordinates,
                elements=elements,
                marked_elements=marked_elements,
                boundary_conditions=[
                    io_helpers.read_boundary_condition(path_to_bc_0),
                    io_helpers.read_boundary_condition(path_to_bc_1),
                    io_helpers.read_boundary_condition(path_to_bc_2)])

        path_to_refined_coordinates = Path(
            'tests/data/refined_rgb/l_shape_coordinates_refined.dat')
        path_to_refined_elements = Path(
            'tests/data/refined_rgb/l_shape_elements_refined.dat')
        expected_coordinates, expected_elements = io_helpers.read_mesh(
            path_to_coordinates=path_to_refined_coordinates,
            path_to_elements=path_to_refined_elements)
        refined_bc_0 = io_helpers.read_boundary_condition(
            Path('tests/data/refined_rgb/l_shape_boundary_0_refined.dat'))
        refined_bc_1 = io_helpers.read_boundary_condition(
            Path('tests/data/refined_rgb/l_shape_boundary_1_refined.dat'))
        refined_bc_2 = io_helpers.read_boundary_condition(
            Path('tests/data/refined_rgb/l_shape_boundary_2_refined.dat'))

        self.assertTrue(np.all(
            refined_coordinates == expected_coordinates))
        self.assertTrue(np.all(
            refined_elements == expected_elements - 1))
        self.assertTrue(np.all(
            new_boundaries[0] == refined_bc_0 - 1))
        self.assertTrue(np.all(
            new_boundaries[1] == refined_bc_1 - 1))
        self.assertTrue(np.all(
            new_boundaries[2] == refined_bc_2 - 1))

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
        path_to_coordinates = Path(
            'tests/data/get_local_patch/coordinates.dat')
        path_to_elements = Path(
            'tests/data/get_local_patch/elements.dat')
        path_to_dirichlet = Path('tests/data/get_local_patch/dirichlet.dat')
        path_to_neumann = Path('tests/data/get_local_patch/neumann.dat')

        global_coordinates, global_elements = io_helpers.read_mesh(
            path_to_coordinates=path_to_coordinates,
            path_to_elements=path_to_elements)
        global_dirichlet = io_helpers.read_boundary_condition(
            path_to_boundary=path_to_dirichlet)
        global_neumann = io_helpers.read_boundary_condition(
            path_to_boundary=path_to_neumann)
        global_boundaries = [global_dirichlet, global_neumann]

        # local patch of element not touching any boundary
        # ------------------------------------------------
        local_coordinates, local_elements, local_boundaries = \
            mesh.get_local_patch(coordinates=global_coordinates,
                                 elements=global_elements,
                                 boundaries=global_boundaries,
                                 which_for=3)
        expected_local_coordinates = np.array([
            [0., 0.],
            [1., 0.],
            [2., 0.],
            [1., 1.],
            [2., 1.],
            [2., 2.]
        ])
        expected_local_elements = np.array([
            [0, 1, 3],
            [1, 2, 4],
            [3, 4, 5],
            [4, 3, 1]
        ])
        expected_local_boundaries = [
            np.array([
                [0, 1], [1, 2]
            ]),
            np.array([
                [2, 4], [4, 5]
            ])]
        self.assertTrue(np.all(
            local_coordinates == expected_local_coordinates))
        self.assertTrue(np.all(
            local_elements == expected_local_elements))
        self.assertTrue(
            len(expected_local_boundaries) == len(local_boundaries))
        for k in range(len(local_boundaries)):
            local_boundary = local_boundaries[k]
            expected_local_boundary = expected_local_boundaries[k]
            self.assertTrue(np.all(local_boundary == expected_local_boundary))

        # local patch of element touching both boundaries
        # --------------------------------------------------
        local_coordinates, local_elements, local_boundaries = \
            mesh.get_local_patch(coordinates=global_coordinates,
                                 elements=global_elements,
                                 boundaries=global_boundaries,
                                 which_for=1)
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
        expected_local_boundaries = [
            np.array([0, 1]),
            np.array([1, 3])
        ]
        self.assertTrue(np.all(
            local_coordinates == expected_local_coordinates))
        self.assertTrue(np.all(
            local_elements == expected_local_elements))
        self.assertTrue(
            len(expected_local_boundaries) == len(local_boundaries))
        for k in range(len(local_boundaries)):
            local_boundary = local_boundaries[k]
            expected_local_boundary = expected_local_boundaries[k]
            self.assertTrue(np.all(local_boundary == expected_local_boundary))

        # local patch of element touching dirichlet boundaries
        # -----------------------------------------------
        local_coordinates, local_elements, local_boundaries = \
            mesh.get_local_patch(coordinates=global_coordinates,
                                 elements=global_elements,
                                 boundaries=global_boundaries,
                                 which_for=0)
        expected_local_elements = np.array([
            [3, 2, 0],
            [4, 3, 1],
            [0, 1, 3]
        ])
        expected_local_coordinates = np.array([
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
            [2, 1]
        ])
        expected_local_boundaries = [
            np.array([
                [2, 0],
                [0, 1]
            ])]
        self.assertTrue(np.all(
            local_coordinates == expected_local_coordinates))
        self.assertTrue(np.all(
            local_elements == expected_local_elements))
        self.assertTrue(
            len(expected_local_boundaries) == len(local_boundaries))
        for k in range(len(local_boundaries)):
            local_boundary = local_boundaries[k]
            expected_local_boundary = expected_local_boundaries[k]
            self.assertTrue(np.all(local_boundary == expected_local_boundary))

        # local patch of element s.t. local patch inherits no boundary
        # ------------------------------------------------------------
        base_path = Path(
            'tests/data/local_patch_touching_no_boundary')
        path_to_coordinates = base_path / Path('coordinates.dat')
        path_to_elements = base_path / Path('elements.dat')
        path_to_boundary = base_path / Path('boundary.dat')
        coordinates, elements = io_helpers.read_mesh(
            path_to_coordinates=path_to_coordinates,
            path_to_elements=path_to_elements)
        boundary = io_helpers.read_boundary_condition(
            path_to_boundary=path_to_boundary)

        local_coordinates, local_elements, local_boundaries = \
            mesh.get_local_patch(coordinates=coordinates,
                                 elements=elements,
                                 boundaries=[boundary],
                                 which_for=6)

        self.assertFalse(local_boundaries)


if __name__ == '__main__':
    unittest.main()
