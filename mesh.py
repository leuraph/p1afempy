import numpy as np
from pathlib import Path


class Mesh:

    coordinates: np.ndarray
    elements: np.ndarray

    def __init__(self, coordinates: np.ndarray, elements: np.ndarray) -> None:
        self.coordinates = coordinates
        self.elements = elements


def read_mesh(path_to_coordinates: Path, path_to_elements: Path) -> Mesh:
    coordinates = np.loadtxt(path_to_coordinates)
    elements = np.loadtxt(path_to_elements, dtype=int)
    return Mesh(coordinates=coordinates, elements=elements)
