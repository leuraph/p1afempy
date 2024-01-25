import numpy as np
import p1afempy.refinement as refinement
import p1afempy.io_helpers as io_helpers
from p1afempy.solvers import solve_laplace
from pathlib import Path
from p1afempy.tests.manual.test_result import TestResult
import time
import matplotlib.pyplot as plt
from scipy.stats import linregress
import copy
from p1afempy.tests.auto.example_setup import uD, g, f


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

    # specifying statistics
    n_refinements = 8
    n_repetetitions_each = 10

    test_results: list[TestResult] = []
    for _ in range(n_refinements):
        # refine all elements
        marked_elements = np.arange(elements.shape[0])
        coordinates, elements, boundaries = refinement.refineNVB(
            coordinates=coordinates,
            elements=elements,
            marked_elements=marked_elements,
            boundary_conditions=boundaries)

        n_elements = elements.shape[0]
        test_result = copy.deepcopy(TestResult(n_elements=n_elements))
        for _ in range(n_repetetitions_each):
            start_time = time.process_time_ns()
            # solve and throw away the result
            _, _ = solve_laplace(coordinates=coordinates,
                                 elements=elements,
                                 dirichlet=boundaries[0],
                                 neumann=boundaries[1],
                                 f=f, g=g, uD=uD)
            end_time = time.process_time_ns()
            test_result.add_time(time=(end_time - start_time)*1e-9)

        test_results.append(copy.deepcopy(test_result))

    n_elements = np.array([result.n_elements for result in test_results])
    means = np.array([np.mean(result.times) for result in test_results])
    stdvs = np.array([np.std(result.times) for result in test_results])

    # Fit (y = mx + q) using linregress to the last four measurements
    res = linregress(n_elements[-4:], means[-4:])

    # read matlab results
    path_to_matlab_means = Path(
        'tests/data/matlab_performance/'
        'solve_laplace/means.dat')
    path_to_matlab_stdevs = Path(
        'tests/data/matlab_performance/'
        'solve_laplace/stdevs.dat')
    path_to_matlab_n_elements = Path(
        'tests/data/matlab_performance/'
        'solve_laplace/n_elements.dat')
    matlab_means = np.loadtxt(path_to_matlab_means)
    matlab_stdevs = np.loadtxt(path_to_matlab_stdevs)
    matlab_n_elements = np.loadtxt(path_to_matlab_n_elements)

    fig, ax = plt.subplots()
    ax.set_title('P1AFEM(PY): solve laplace')

    # indices below this will be ignores because Matlab's
    # cputime method simply returns zero for values 'too small'
    cut_at = 4

    # plot python results
    ax.errorbar(n_elements[cut_at:], means[cut_at:], stdvs[cut_at:],
                fmt='x', label='Python', color='green',
                capsize=2, capthick=1, elinewidth=1, markersize=4)
    ax.loglog(n_elements[cut_at:],
              res.slope * n_elements[cut_at:],
              linestyle='--', color='black', linewidth=0.5,
              label=r'$\propto M$')

    # plot matlab results
    ax.errorbar(matlab_n_elements[cut_at:],
                matlab_means[cut_at:], matlab_stdevs[cut_at:],
                fmt='s', label='Matlab', color='red',
                capsize=2, capthick=1, elinewidth=1, markersize=4)

    ax.set_xlabel('number of elements')
    ax.set_ylabel('cpu time $/s$')

    # Set the axis to log-log scale
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlim(left=1e3)

    ax.grid()
    ax.legend(loc='best')
    fig.savefig('solve_laplace.png')


if __name__ == "__main__":
    main()
