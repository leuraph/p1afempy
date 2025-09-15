from scipy.__config__ import show
from p1afempy.mesh import show_mesh, get_rectangular_mesh
import numpy as np


def main():

    lower_left = np.array([0., 0.])
    upper_right = np.array([1., 1.])
    n_elements_x = 3
    n_elements_y = 3

    coordinates, elements, boundaries\
        = get_rectangular_mesh(
            lower_left=lower_left,
            upper_right=upper_right,
            n_elements_x=n_elements_x,
            n_elements_y=n_elements_y)

    show_mesh(coordinates=coordinates, elements=elements)


if __name__ == '__main__':
    main()
