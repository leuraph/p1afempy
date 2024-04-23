from p1afempy import io_helpers, refinement, mesh
from pathlib import Path


def main() -> None:
    # ------------------------
    # reading the initial mesh
    # ------------------------
    path_to_coordinates = Path(
        'tests/data/refine_rg/coordinates.dat')
    path_to_elements = Path(
        'tests/data/refine_rg/elements.dat')
    path_to_dirichlet = Path('tests/data/refine_rg/dirichlet.dat')
    path_to_neumann = Path('tests/data/refine_rg/neumann.dat')
    coordinates, elements = io_helpers.read_mesh(
        path_to_coordinates=path_to_coordinates,
        path_to_elements=path_to_elements,
        shift_indices=False)
    dirichlet = io_helpers.read_boundary_condition(
        path_to_boundary=path_to_dirichlet,
        shift_indices=False)
    neumann = io_helpers.read_boundary_condition(
        path_to_boundary=path_to_neumann,
        shift_indices=False)
    boundaries = [dirichlet, neumann]

    # ----------------
    # case no_boundary
    # ----------------
    marked_element = 3

    new_coordinates, new_elements, new_boundaries, _ = \
        refinement.refineRG_without_element_to_neighbours(
            coordinates=coordinates,
            elements=elements,
            marked_element=marked_element,
            boundaries=boundaries)

    mesh.show_mesh(coordinates=coordinates,
                   elements=elements)

    mesh.show_mesh(coordinates=new_coordinates,
                   elements=new_elements)


if __name__ == '__main__':
    main()
