from p1afempy import io_helpers, refinement, mesh
from pathlib import Path
from tqdm import tqdm
import numpy as np


# specifying statistics
N_INITIAL_REFINEMENTS = 6
N_RED_REFINEMENT_LOOPS = 1


def main() -> None:
    # ---------------------------------------
    # specifiying paths to read data from
    path_to_elements = Path('tests/data/simple_square_mesh/elements.dat')
    path_to_coordinates = Path('tests/data/simple_square_mesh/coordinates.dat')
    path_to_boundary_0 = Path(
        'tests/data/simple_square_mesh/square_boundary_0.dat')
    path_to_boundary_1 = Path(
        'tests/data/simple_square_mesh/square_boundary_1.dat')
    # ---------------------------------------

    # ---------------------------------------
    # reading initial data
    coordinates, elements = io_helpers.read_mesh(
        path_to_coordinates=path_to_coordinates,
        path_to_elements=path_to_elements)
    boundary_0 = io_helpers.read_boundary_condition(path_to_boundary_0)
    boundary_1 = io_helpers.read_boundary_condition(path_to_boundary_1)
    boundaries = [boundary_0, boundary_1]
    # ---------------------------------------

    # ---------------------------------------
    # initial refinements
    for _ in range(N_INITIAL_REFINEMENTS):
        marked_elements = np.arange(elements.shape[0])
        coordinates, elements, boundaries, _ = refinement.refineNVB(
            coordinates=coordinates,
            elements=elements,
            marked_elements=marked_elements,
            boundary_conditions=boundaries)
    # ---------------------------------------

    # ---------------------------------------
    n_elements = elements.shape[0]
    print(f"number of elements = {n_elements}")
    element_to_neighbours = mesh.get_element_to_neighbours(elements)
    for _ in range(N_RED_REFINEMENT_LOOPS):
        for marked_element in tqdm(range(n_elements)):
            # refine and throw away the result
            _, _, _, _ = refinement.refineRG_with_element_to_neighbours(
                coordinates=coordinates,
                elements=elements,
                which=marked_element,
                boundaries=boundaries,
                element_to_neighbours=element_to_neighbours)
    # ---------------------------------------


if __name__ == '__main__':
    main()
