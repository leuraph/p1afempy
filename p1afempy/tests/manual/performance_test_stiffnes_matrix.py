import numpy as np
from p1afempy import io_helpers
from p1afempy import solvers
from p1afempy import refinement
from pathlib import Path


class TestResult:
    n_elements: int
    times: list[float]

    def __init__(self,
                 n_elements: int = 0,
                 times: list[float] = []) -> None:
        self.n_elements = n_elements
        self.times = times

    def add_time(self, time: float) -> None:
        self.times.append(time)

    def get_statistics(self) -> tuple[float, float]:
        """returns mean and standard deviation of results"""
        return np.mean(self.times), np.std(self.times)


def main() -> None:
    # specifiying paths to read data from
    path_to_elements = Path('tests/data/simple_square_mesh/elements.dat')
    path_to_coordinates = Path('tests/data/simple_square_mesh/coordinates.dat')
    path_to_boundary_0 = Path(
        'tests/data/simple_square_mesh/square_boundary_0.dat')
    path_to_boundary_1 = Path(
        'tests/data/simple_square_mesh/square_boundary_1.dat')

    # reading initial data
    coordinates, elements = io_helpers.read_mesh(
        path_to_coordinates=path_to_coordinates,
        path_to_elements=path_to_elements)
    boundary_0 = io_helpers.read_boundary_condition(path_to_boundary_0)
    boundary_1 = io_helpers.read_boundary_condition(path_to_boundary_1)
    boundaries = [boundary_0, boundary_1]

    n_refinements = 5

    for _ in range(n_refinements):
        n_elements = elements.shape[0]

        # TODO measure
        _ = solvers.get_stiffness_matrix(coordinates=coordinates,
                                         elements=elements)
        # TODO measure

        marked_elements = np.arange(n_elements)  # refine all elements
        coordinates, elements, boundaries = refinement.refineNVB(
            coordinates=coordinates,
            elements=elements,
            marked_elements=marked_elements,
            boundary_conditions=boundaries)


if __name__ == '__main__':
    main()
