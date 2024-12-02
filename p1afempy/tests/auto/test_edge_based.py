import unittest
import p1afempy.mesh as mesh
from p1afempy.refinement import refine_single_edge
import numpy as np


class MeshTest(unittest.TestCase):

    def test_get_local_patch_edge_based(self) -> None:
        elements, coordinates, solution = get_state()

        # TEST edge=(4, 8), edge=(8, 4)
        # -----------------------------
        expected_elements = np.array([[0, 1, 3], [3, 2, 0]], dtype=int)
        expected_coordinates = np.array(
            [[1, 1], [2, 1], [1, 2], [2, 2]], dtype=float)
        expected_solution = np.array([11, 13, 19, 23], dtype=float)

        edge = np.array([4, 8])
        expected_edge = np.array([0, 3], dtype=int)

        local_elements, local_coordinates, local_solution, local_edge =\
            mesh.get_local_patch_edge_based(
                elements=elements, coordinates=coordinates,
                current_iterate=solution, edge=edge)

        self.assertTrue(np.all(local_elements == expected_elements))
        self.assertTrue(np.all(local_coordinates == expected_coordinates))
        self.assertTrue(np.all(local_solution == expected_solution))
        self.assertTrue(np.all(local_edge == expected_edge))

        edge = np.array([8, 4])
        expected_edge = np.array([3, 0], dtype=int)

        local_elements, local_coordinates, local_solution, local_edge =\
            mesh.get_local_patch_edge_based(
                elements=elements, coordinates=coordinates,
                current_iterate=solution, edge=edge)

        self.assertTrue(np.all(local_elements == expected_elements))
        self.assertTrue(np.all(local_coordinates == expected_coordinates))
        self.assertTrue(np.all(local_solution == expected_solution))
        self.assertTrue(np.all(local_edge == expected_edge))

        # Raises when called with boundary edge
        # -------------------------------------
        boundary_edges = [
            np.array([0, 1]),
            np.array([1, 2]),
            np.array([2, 5]),
            np.array([5, 8]),
            np.array([8, 7]),
            np.array([7, 6]),
            np.array([6, 3]),
            np.array([3, 0])]
        for boundary_edge in boundary_edges:
            self.assertRaises(
                ValueError, mesh.get_local_patch_edge_based,
                elements, coordinates, solution, boundary_edge)

    def test_refine_single_edge(self) -> None:
        # initial setup
        # -------------
        elements = np.array([
            [0, 1, 3],
            [3, 2, 0]
        ], dtype=int)
        coordinates = np.array([
            [0., 0.],
            [1., 0.],
            [0., 1.],
            [1., 1.]
        ], dtype=float)
        solution = np.array([1., 5., 7., 11.], dtype=float)

        # refined coords and solution are equal for flipped edges
        # -------------------------------------------------------
        expected_refined_solution = np.array(
            [1., 5., 7., 11., 6.], dtype=float)
        expected_refined_coordinates = np.array([
            [0., 0.],
            [1., 0.],
            [0., 1.],
            [1., 1.],
            [0.5, 0.5]
        ], dtype=float)

        # refine edge=(0, 3)
        # ------------------
        edge = np.array([0, 3], dtype=int)
        expected_refined_elements = np.array([
            [4, 2, 0],
            [4, 3, 2],
            [4, 0, 1],
            [4, 1, 3]
        ], dtype=int)

        refined_coordinates, refined_elements, refined_solution = \
            refine_single_edge(
                coordinates=coordinates,
                elements=elements,
                edge=edge,
                to_embed=solution)
        self.assertTrue(np.all(
            expected_refined_coordinates == refined_coordinates))
        self.assertTrue(np.all(
            expected_refined_elements == refined_elements))
        self.assertTrue(np.all(
            expected_refined_solution == refined_solution))

        # refine edge=(0, 3)
        # ------------------
        edge = np.array([3, 0], dtype=int)
        expected_refined_elements = np.array([
            [4, 1, 3],
            [4, 0, 1],
            [4, 3, 2],
            [4, 2, 0]
        ], dtype=int)

        refined_coordinates, refined_elements, refined_solution = \
            refine_single_edge(
                coordinates=coordinates,
                elements=elements,
                edge=edge,
                to_embed=solution)
        self.assertTrue(np.all(
            expected_refined_coordinates == refined_coordinates))
        self.assertTrue(np.all(
            expected_refined_elements == refined_elements))
        self.assertTrue(np.all(
            expected_refined_solution == refined_solution))


def get_state() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    elements = np.array([
        [0, 1, 4],
        [1, 2, 5],
        [3, 4, 7],
        [4, 5, 8],
        [4, 3, 0],
        [5, 4, 1],
        [7, 6, 3],
        [8, 7, 4]
    ], dtype=int)

    coordinates = np.array([
        [0, 0],
        [1, 0],
        [2, 0],
        [0, 1],
        [1, 1],
        [2, 1],
        [0, 2],
        [1, 2],
        [2, 2]
    ], dtype=float)

    solution = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23], dtype=float)

    return elements, coordinates, solution


if __name__ == '__main__':
    unittest.main()
