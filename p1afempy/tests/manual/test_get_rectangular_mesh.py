from scipy.__config__ import show
from p1afempy.mesh import show_mesh, get_rectangular_mesh
import numpy as np
from pathlib import Path
from p1afempy.refinement import refineNVB
from p1afempy.solvers import get_stiffness_matrix
from scipy.sparse import csr_matrix


def main():

    lower_left = np.array([0., 0.])
    upper_right = np.array([1., 1.])
    n_elements_x = 5
    n_elements_y = 10

    coordinates, elements, boundaries\
        = get_rectangular_mesh(
            lower_left=lower_left,
            upper_right=upper_right,
            n_elements_x=n_elements_x,
            n_elements_y=n_elements_y)

    path_to_save = Path('tests/manual/test.pdf')
    show_mesh(
        coordinates=coordinates,
        elements=elements,
        boundaries=boundaries,
        path_to_save=path_to_save)
    
    # REFINEMENT
    # ----------
    lower_left = np.array([0., 0.])
    upper_right = np.array([1., 1.])
    n_elements_x = 2
    n_elements_y = 2

    coordinates, elements, boundaries\
        = get_rectangular_mesh(
            lower_left=lower_left,
            upper_right=upper_right,
            n_elements_x=n_elements_x,
            n_elements_y=n_elements_y)
    
    to_embed = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=float)

    stiffness_matrix = csr_matrix(get_stiffness_matrix(coordinates, elements))
    initial_energy_norm = np.sqrt(to_embed.dot(stiffness_matrix.dot(to_embed)))

    for _ in range(5):
        n_elements = elements.shape[0]
        marked_elements = np.arange(n_elements)
        coordinates, elements, boundaries, to_embed = \
            refineNVB(
                coordinates=coordinates,
                elements=elements,
                marked_elements=marked_elements,
                boundary_conditions=boundaries,
                to_embed=to_embed)
    
    stiffness_matrix = csr_matrix(get_stiffness_matrix(coordinates, elements))
    final_energy_norm = np.sqrt(to_embed.dot(stiffness_matrix.dot(to_embed)))

    print('the following values should agree!')
    print(f'\t|u_init|_a = {initial_energy_norm}')
    print(f'\t|u_final|_a = {final_energy_norm}')
    
    show_mesh(coordinates, elements, linewidth=1.0, boundaries=boundaries)


if __name__ == '__main__':
    main()
