from pickle import NONE
import unittest
import numpy as np
import p1afempy.io_helpers as io_helpers
from p1afempy.solvers import solve_laplace, get_mass_matrix_elements, \
    get_right_hand_side, integrate_nonlinear_fem
import p1afempy.refinement as refinement
from pathlib import Path
import example_setup
from tests.auto.example_setup import f
from triangle_cubature.cubature_rule import CubatureRuleEnum
import sympy as sp


class SolverTest(unittest.TestCase):

    def test_solve_laplace(self) -> None:
        path_to_coordinates = Path(
            'tests/data/laplace_example/coordinates.dat')
        path_to_elements = Path('tests/data/laplace_example/elements.dat')
        path_to_neumann = Path('tests/data/laplace_example/neumann.dat')
        path_to_dirichlet = Path('tests/data/laplace_example/dirichlet.dat')
        path_to_matlab_x = Path('tests/data/laplace_example/x.dat')
        path_to_matlab_energy = Path('tests/data/laplace_example/energy.dat')

        coordinates, elements = io_helpers.read_mesh(
            path_to_coordinates=path_to_coordinates,
            path_to_elements=path_to_elements)
        neumann_bc = io_helpers.read_boundary_condition(
            path_to_boundary=path_to_neumann)
        dirichlet_bc = io_helpers.read_boundary_condition(
            path_to_boundary=path_to_dirichlet)
        boundary_conditions = [dirichlet_bc, neumann_bc]

        n_refinements = 3
        for _ in range(n_refinements):
            marked_elements = np.arange(elements.shape[0])
            coordinates, elements, boundary_conditions, _ = refinement.refineNVB(
                coordinates=coordinates,
                elements=elements,
                marked_elements=marked_elements,
                boundary_conditions=boundary_conditions)

        x, energy = solve_laplace(
            coordinates=coordinates, elements=elements,
            dirichlet=boundary_conditions[0],
            neumann=boundary_conditions[1],
            f=example_setup.f, g=example_setup.g, uD=example_setup.uD)

        x_matlab = np.loadtxt(path_to_matlab_x)
        energy_matlab = np.loadtxt(path_to_matlab_energy).reshape((1,))[0]

        self.assertTrue(np.allclose(x, x_matlab))
        self.assertTrue(np.isclose(energy, energy_matlab))

    def test_get_mass_matrix_elements(self) -> None:
        path_to_coordinates = Path(
            'tests/data/ahw_codes_example_mesh/coordinates.dat')
        path_to_elements = Path(
            'tests/data/ahw_codes_example_mesh/elements.dat')
        mesh_ahw_coordinates, mesh_ahw_elements = io_helpers.read_mesh(
            path_to_coordinates=path_to_coordinates,
            path_to_elements=path_to_elements,
            shift_indices=True)

        path_to_expected_I = Path(
            'tests/data/ahw_codes_example_mesh/mass_matrix_I.dat')
        path_to_expected_J = Path(
            'tests/data/ahw_codes_example_mesh/mass_matrix_J.dat')
        path_to_expected_D = Path(
            'tests/data/ahw_codes_example_mesh/mass_matrix_D.dat')
        expected_I = np.loadtxt(path_to_expected_I).astype(np.uint32)
        expected_J = np.loadtxt(path_to_expected_J).astype(np.uint32)
        expected_D = np.loadtxt(path_to_expected_D)


        I, J, D = get_mass_matrix_elements(
            coordinates=mesh_ahw_coordinates,
            elements=mesh_ahw_elements)

        self.assertTrue(np.allclose(I+1, expected_I))
        self.assertTrue(np.allclose(J+1, expected_J))
        self.assertTrue(np.allclose(D, expected_D))

    def test_get_right_hand_side(self):
        """
        This test is a sanity check to test
        two implementations of retreiving the right
        hand side (load) vector, i.e. we test the
        "original" P1AFEM implementation (midpoint rule) against
        an adapted version based on custom quadrature rules.
        """
        path_to_coordinates = Path(
            'tests/data/simple_square_mesh/coordinates.dat')
        path_to_elements = Path(
            'tests/data/simple_square_mesh/elements.dat')
        path_to_boundary_0 = Path(
            'tests/data/simple_square_mesh/square_boundary_0.dat')
        path_to_boundary_1 = Path(
            'tests/data/simple_square_mesh/square_boundary_1.dat')
        coordinates, elements = io_helpers.read_mesh(
            path_to_coordinates=path_to_coordinates,
            path_to_elements=path_to_elements,
            shift_indices=False)
        boundary_0 = io_helpers.read_boundary_condition(
            path_to_boundary=path_to_boundary_0)
        boundary_1 = io_helpers.read_boundary_condition(
            path_to_boundary=path_to_boundary_1)
        boundaries = [boundary_0, boundary_1]

        n_refinements = 5
        for _ in range(n_refinements):
            marked_elements = np.arange(elements.shape[0])
            coordinates, elements, boundaries, _ = refinement.refineNVB(
                coordinates=coordinates,
                elements=elements,
                marked_elements=marked_elements,
                boundary_conditions=boundaries)

        rhs_vector_P1AFEM = get_right_hand_side(
            coordinates=coordinates, elements=elements, f=f)
        rhs_vector_midpoint = get_right_hand_side(
            coordinates=coordinates, elements=elements, f=f,
            cubature_rule=CubatureRuleEnum.MIDPOINT)

        self.assertTrue(np.allclose(rhs_vector_midpoint, rhs_vector_P1AFEM))
    
    def test_integrate_nonlinear_fem(self):
        """
        This is a sanity check to verify
        the numerical integration of the
        composition of any non-linear function
        with a P1FEM function
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
        dirichlet = np.array([
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0]
        ])
        boundaries = [dirichlet]

        # initial refinement
        n_refinements = 5
        for _ in range(n_refinements):
            marked_elements = np.arange(elements.shape[0])
            coordinates, elements, boundaries, _ = refinement.refineNVB(
                coordinates=coordinates,
                elements=elements,
                marked_elements=marked_elements,
                boundary_conditions=boundaries)
        
        c_0 = 1.0
        c_1 = 2.3
        c_2 = 1.6

        def f(u: float):
            return c_0 + c_1*u + c_2*u**2
        
        a = 1.4
        b = 6.3
        c = 4.7

        def u(r: np.ndarray) -> np.ndarray:
            return a + b * r[:, 0] + c * r[:, 1]
        
        u_on_nodes = u(coordinates)

        def get_exact_integral(
                a:float, 
                b:float,
                c:float,
                c_0:float,
                c_1:float,
                c_2:float,
                c_3:float) -> float:
            """
            computes the exact value of the integral
            int_Omega f(u(x)) dx, where
            Omega = (0, 1)^2
            f(u) = c_0 + c_1 u + c_2 u^2 + c_3 u^3
            u(x) = a + bx + cy
            """
            # Define symbols
            a_sym, b_sym, c_sym = sp.symbols('a b c', real=True)
            c0_sym, c1_sym, c2_sym, c3_sym = sp.symbols('c0 c1 c2 c3', real=True)
            x_sym, y_sym = sp.symbols('x y', real=True)

            # Define u(x,y)
            u_sym = a_sym + b_sym*x_sym + c_sym*y_sym

            # Define f(u)
            f_sym = c0_sym + c1_sym*u_sym + c2_sym*u_sym**2 + c3_sym*u_sym**3

            # Compute exact integral over (0,1)^2
            I_sym = sp.integrate(sp.integrate(f_sym, (x_sym, 0, 1)), (y_sym, 0, 1))

            # Substitute numerical values for parameters (example values)
            subs_dict = {
                a_sym: a,
                b_sym: b,
                c_sym: c,
                c0_sym: c_0,
                c1_sym: c_1,
                c2_sym: c_2,
                c3_sym: c_3}
            I_numeric = I_sym.subs(subs_dict).evalf()

            return I_numeric
        
        exact_integral = get_exact_integral(a, b, c, c_0, c_1, c_2, c_3=0.)
        numerical_integral = integrate_nonlinear_fem(
            f=f, u=u_on_nodes,
            coordinates=coordinates,
            elements=elements,
            cubature_rule=CubatureRuleEnum.DAYTAYLOR)

        self.assertAlmostEqual(exact_integral, numerical_integral)



if __name__ == "__main__":
    unittest.main()
