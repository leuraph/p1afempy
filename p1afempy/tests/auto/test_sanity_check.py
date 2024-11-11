import numpy as np
import unittest
import random
from pathlib import Path
from p1afempy import io_helpers, solvers, mesh
from p1afempy.data_structures import \
    CoordinatesType, ElementsType, BoundaryType
from p1afempy import refinement
from p1afempy.refinement import refine_single_edge
from ismember import is_row_in


def test_function(x: float, y: float) -> float:
    x_inside_domain = -1. < x < 1.
    y_inside_domain = -1. < y < 1.
    if (not x_inside_domain) or (not y_inside_domain):
        return 0.
    return min(1. - abs(x), 1. - abs(y))


def evaluate_energy_on_mesh(
        coordinates: CoordinatesType,
        elements: ElementsType,
        test_function_vector: np.ndarray = np.array([])) -> float:

    stiffness_matrix = solvers.get_stiffness_matrix(
        coordinates=coordinates,
        elements=elements)

    # if test function vector was not passed, calculate the exact value
    if not test_function_vector.size:
        test_function_vector = np.array([
            test_function(x, y) for (x, y) in coordinates])

    return test_function_vector.dot(stiffness_matrix.dot(test_function_vector))


class SanityChecks(unittest.TestCase):
    """
    These tests constitute some sort of sanity checks to check
    - stiffness matrix assembly
    - mesh refinement strategies

    idea
    ----
    The idea is to compute the discrete version of the energy
    E(u) := a(u, u)
    of a function u(x, y) that is exactly represented already
    on the initial mesh. In this way, we can check the interplay
    of stiffness matrix assembly and mesh refinement by checking
    the computed energy E:= x.T * A * x with the exact value for
    the initial mesh and all subsequent refined meshes thereof.

    implementation
    --------------
    we choose
    - Omega := {(x, y) | -1 < x, y < 1}
    - u(x, y) := min{1-|x|, 1-|y|}
    - an initial mesh that already allows for exact approximation
      of u on Omega
      (for details, see the input data in `tests/data/sanity_check`).
    - the expected energy of u is then given by E(u) = 4.
    """

    @staticmethod
    def get_initial_mesh() -> tuple[CoordinatesType,
                                    ElementsType,
                                    BoundaryType]:
        path_to_coordinates = Path('tests/data/sanity_check/coordinates.dat')
        path_to_elements = Path('tests/data/sanity_check/elements.dat')
        path_to_dirichlet = Path('tests/data/sanity_check/dirichlet.dat')
        coordinates, elements = io_helpers.read_mesh(
            path_to_coordinates=path_to_coordinates,
            path_to_elements=path_to_elements)
        dirichlet = io_helpers.read_boundary_condition(
            path_to_boundary=path_to_dirichlet)
        return coordinates, elements, dirichlet

    def test_simple_stiffness_assembly(self) -> None:
        coordinates, elements, _ = SanityChecks.get_initial_mesh()

        expected_energy = 4.
        computed_energy = evaluate_energy_on_mesh(
            coordinates=coordinates, elements=elements)

        self.assertEqual(expected_energy, computed_energy)

    def test_refine_nvb(self) -> None:
        coordinates, elements, dirichlet = SanityChecks.get_initial_mesh()

        # initial vector of values to be interpolated on refined meshes
        to_embed = np.array([test_function(x, y) for (x, y) in coordinates])

        boundaries = [dirichlet]
        n_refinements = 5
        for _ in range(n_refinements):
            # mark all elements for refinement
            marked_elements = np.arange(elements.shape[0])

            # perform refinement
            coordinates, elements, boundaries, to_embed = refinement.refineNVB(
                coordinates=coordinates,
                elements=elements,
                marked_elements=marked_elements,
                boundary_conditions=boundaries,
                to_embed=to_embed)

            # in each step, compare the computed vs. expected eenergy
            expected_energy = 4.

            computed_energy = evaluate_energy_on_mesh(
                coordinates=coordinates, elements=elements)
            self.assertEqual(expected_energy, computed_energy)

            computed_interpolated_energy = evaluate_energy_on_mesh(
                coordinates=coordinates, elements=elements,
                test_function_vector=to_embed)
            self.assertEqual(expected_energy, computed_interpolated_energy)

    def test_refine_rgb(self) -> None:
        coordinates, elements, dirichlet = SanityChecks.get_initial_mesh()

        # initial vector of values to be interpolated on refined meshes
        to_embed = np.array([test_function(x, y) for (x, y) in coordinates])

        boundaries = [dirichlet]
        n_refinements = 5
        for _ in range(n_refinements):
            # mark all elements for refinement
            marked_elements = np.arange(elements.shape[0])

            # perform refinement
            coordinates, elements, boundaries, to_embed = refinement.refineRGB(
                coordinates=coordinates,
                elements=elements,
                marked_elements=marked_elements,
                boundary_conditions=boundaries,
                to_embed=to_embed)

            # in each step, compare the computed vs. expected eenergy
            expected_energy = 4.

            computed_energy = evaluate_energy_on_mesh(
                coordinates=coordinates, elements=elements)
            self.assertEqual(expected_energy, computed_energy)

            computed_interpolated_energy = evaluate_energy_on_mesh(
                coordinates=coordinates, elements=elements,
                test_function_vector=to_embed)
            self.assertEqual(expected_energy, computed_interpolated_energy)

    def test_refine_rg(self) -> None:
        random.seed(42)
        coordinates, elements, dirichlet = SanityChecks.get_initial_mesh()

        boundaries = [dirichlet]
        n_refinements = 100
        for _ in range(n_refinements):
            # mark a random element for refinement
            n_elements = elements.shape[0]
            marked_element = random.randrange(n_elements)

            # perform refinement
            coordinates, elements, boundaries, _ = \
                refinement.refineRG_without_element_to_neighbours(
                    coordinates=coordinates,
                    elements=elements,
                    marked_element=marked_element,
                    boundaries=boundaries)

            # in each step, compare the computed vs. expected eenergy
            expected_energy = 4.
            computed_energy = evaluate_energy_on_mesh(
                coordinates=coordinates, elements=elements)

            self.assertEqual(expected_energy, computed_energy)

    def test_refine_rg_single(self) -> None:
        random.seed(42)
        coordinates, elements, dirichlet = SanityChecks.get_initial_mesh()

        boundaries = [dirichlet]
        n_refinements = 100
        for _ in range(n_refinements):
            # mark a random element for refinement
            n_elements = elements.shape[0]
            marked_element = random.randrange(n_elements)

            # perform refinement
            coordinates, elements, boundaries, _ = \
                refinement.refineRG_with_element_to_neighbours(
                    coordinates=coordinates,
                    elements=elements,
                    which=marked_element,
                    boundaries=boundaries,
                    element_to_neighbours=mesh.get_element_to_neighbours(
                        elements))

            # in each step, compare the computed vs. expected eenergy
            expected_energy = 4.
            computed_energy = evaluate_energy_on_mesh(
                coordinates=coordinates, elements=elements)

            self.assertEqual(expected_energy, computed_energy)

    def test_refine_single_edge(self) -> None:
        coordinates, elements, dirichlet = SanityChecks.get_initial_mesh()

        element_indices_i = elements.flatten()
        element_indices_j = elements[:, [1, 2, 0]].flatten()
        edges = np.column_stack([element_indices_i, element_indices_j])
        unique_edges = edges[element_indices_i < element_indices_j]

        for edge in unique_edges:
            # skip if edge is on boundary
            if np.any(is_row_in(edge.reshape((1, 2)), dirichlet)):
                continue

            # perform refinement
            new_coordinates, new_elements = \
                refine_single_edge(
                    coordinates=coordinates,
                    elements=elements,
                    edge=edge)

            # checking for correct orientation of elements,
            # i.e. counter-clockwise
            for element in new_elements:
                z0 = new_coordinates[element[0]]
                z1 = new_coordinates[element[1]]
                z2 = new_coordinates[element[2]]
                DPhi = np.column_stack([z1 - z0, z2 - z0])
                self.assertGreater(np.linalg.det(DPhi), 0.0)

            # for each refined edge, compare the computed vs. expected eenergy
            expected_energy = 4.
            computed_energy = evaluate_energy_on_mesh(
                coordinates=coordinates, elements=elements)

            expected_new_coordinate = 0.5*(
                coordinates[edge[0], :] + coordinates[edge[1], :])

            self.assertEqual(expected_energy, computed_energy)
            self.assertAlmostEqual(
                expected_new_coordinate[0], new_coordinates[-1, 0])
            self.assertAlmostEqual(
                expected_new_coordinate[1], new_coordinates[-1, 1])

    def test_refine_rg_with_solution_interpolation(self) -> None:
        random.seed(42)
        coordinates, elements, dirichlet = SanityChecks.get_initial_mesh()

        to_embed = np.array([test_function(x, y) for (x, y) in coordinates])

        boundaries = [dirichlet]
        n_refinements = 100
        for _ in range(n_refinements):
            # mark a random element for refinement
            n_elements = elements.shape[0]
            marked_element = random.randrange(n_elements)

            # perform refinement
            coordinates, elements, boundaries, to_embed = \
                refinement.refineRG_without_element_to_neighbours(
                    coordinates=coordinates,
                    elements=elements,
                    marked_element=marked_element,
                    boundaries=boundaries,
                    to_embed=to_embed)

            # in each step, compare the computed vs. expected eenergy
            expected_energy = 4.
            computed_energy = evaluate_energy_on_mesh(
                coordinates=coordinates, elements=elements,
                test_function_vector=to_embed)

            self.assertEqual(expected_energy, computed_energy)

    def test_refine_rg_single_with_solution_interpolation(self) -> None:
        random.seed(42)
        coordinates, elements, dirichlet = SanityChecks.get_initial_mesh()

        to_embed = np.array([test_function(x, y) for (x, y) in coordinates])

        boundaries = [dirichlet]
        n_refinements = 100
        for _ in range(n_refinements):
            # mark a random element for refinement
            n_elements = elements.shape[0]
            marked_element = random.randrange(n_elements)

            # perform refinement
            coordinates, elements, boundaries, to_embed = \
                refinement.refineRG_with_element_to_neighbours(
                    coordinates=coordinates,
                    elements=elements,
                    which=marked_element,
                    boundaries=boundaries,
                    to_embed=to_embed,
                    element_to_neighbours=mesh.get_element_to_neighbours(
                        elements=elements))

            # in each step, compare the computed vs. expected eenergy
            expected_energy = 4.
            computed_energy = evaluate_energy_on_mesh(
                coordinates=coordinates, elements=elements,
                test_function_vector=to_embed)

            self.assertEqual(expected_energy, computed_energy)


if __name__ == '__main__':
    unittest.main()
