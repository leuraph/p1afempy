from pathlib import Path
import numpy as np


def main():
    path_to_coordinates = Path('p1afempy/tests/data/coordinates.dat')
    path_to_elements = Path('p1afempy/tests/data/elements.dat')

    coordinates = np.loadtxt(path_to_coordinates)
    elements = np.loadtxt(path_to_elements, dtype=int)

    print(coordinates)
    print(elements)

if __name__ == '__main__':
    main()
