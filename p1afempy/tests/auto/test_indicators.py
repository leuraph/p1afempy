import unittest
from p1afempy import mesh, solvers, indicators
from mesh import BoundaryCondition
from pathlib import Path
import numpy as np
import example_setup


class IndicatorTest(unittest.TestCase):

    def test_compute_eta_r(self) -> None:
        path_to_coordinates = Path(
            'tests/data/laplace_example/coordinates.dat')
        path_to_elements = Path('tests/data/laplace_example/elements.dat')
        path_to_neumann = Path('tests/data/laplace_example/neumann.dat')
        path_to_dirichlet = Path('tests/data/laplace_example/dirichlet.dat')
        path_to_matlab_indicators = Path(
            'tests/data/laplace_example/indicators.dat')

        coordinates, elements = mesh.read_mesh(
            path_to_coordinates=path_to_coordinates,
            path_to_elements=path_to_elements)
        neumann_bc = mesh.read_boundary_condition(
            path_to_boundary=path_to_neumann)
        dirichlet_bc = mesh.read_boundary_condition(
            path_to_boundary=path_to_dirichlet)
        boundary_conditions = [dirichlet_bc.boundary, neumann_bc.boundary]

        n_refinements = 2
        for _ in range(n_refinements):
            marked_elements = np.arange(elements.shape[0])
            coordinates, elements, boundary_conditions = \
                mesh.refineNVB(coordinates=coordinates,
                               elements=elements,
                               marked_elements=marked_elements,
                               boundary_conditions=boundary_conditions)

        boundary_conditions = [
            BoundaryCondition(
                name='', boundary=bc) for bc in boundary_conditions]
        x, energy = solvers.solve_laplace(
            coordinates=coordinates, elements=elements,
            dirichlet=boundary_conditions[0],
            neumann=boundary_conditions[1],
            f=example_setup.f, g=example_setup.g, uD=example_setup.uD)

        ref_indicators = indicators.compute_eta_r(
            x=x,
            coordinates=coordinates,
            elements=elements,
            dirichlet=boundary_conditions[0],
            neumann=boundary_conditions[1],
            f=example_setup.f, g=example_setup.g)

        indicators_matlab = np.loadtxt(path_to_matlab_indicators)

        self.assertTrue(np.allclose(ref_indicators, indicators_matlab))
        pass


if __name__ == '__main__':
    unittest.main()
