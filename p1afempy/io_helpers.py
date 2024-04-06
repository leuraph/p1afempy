import numpy as np
from pathlib import Path


def read_boundary_condition(path_to_boundary: Path,
                            shift_indices: bool = False) -> np.ndarray:
    """
    Reads boundary condition data from a file.

    Parameters
    ----------
    path_to_boundary: pathlib.Path
        Path to the file containing boundary condition data.
    shift_indices: bool (default=False)
        If true, shifts the elements' indices according to i':=i-1.
        This can come in handy if, e.g., one wants to read data
        that is compatible with Matlab/Fortran/Julia indexing
        (starting at 1, instead of 0).

    Returns
    -------
    boundary_indices: np.ndarray:
        boundary conditions as np.nndarray, i.e.
        boundary_indices[k] gives the indices (i_k, j_k) of the
        vertices belonging to the k-th edge of the boundary condition at hand

    Example
    -------
    >>> file_path = Path("path/to/boundary_data.dat")
    >>> boundary_condition = read_boundary_condition(file_path)
    """
    boundary_indices = np.loadtxt(path_to_boundary).astype(np.uint32)
    if shift_indices:
        boundary_indices = boundary_indices - 1
    # make sure we match the expected shape even if (in an edge case)
    # the specified boundary condition is only valid on one single edge
    if len(boundary_indices.shape) == 1:
        return boundary_indices[None, :]
    return boundary_indices


def read_coordinates(path_to_coordinates: Path) -> np.ndarray:
    """
    reads and returns vertices from file

    Parameters
    ----------
    path_to_coordinates : pathlib.Path
        Path to the file containing mesh coordinates data.
    path_to_elements : pathlib.Path

    Returns
    -------
    coordinates: CoordinatesType
        coordinates of the mesh's vertices

    Example
    -------
    >>> coordinates_path = Path("path/to/coordinates.txt")
    >>> coordinates = read_coordinates(coordinates_path)
    """
    coordinates = np.loadtxt(path_to_coordinates)
    return coordinates


def read_elements(path_to_elements: Path,
                  shift_indices: bool = False) -> np.ndarray:
    """
    reads and returns elements from file.

    Parameters
    ----------
    path_to_elements : pathlib.Path
        Path to the file containing mesh elements data.
    shift_indices: bool (default=False)
        If true, shifts the elements' indices according to i':=i-1.
        This can come in handy if, e.g., one wants to read data
        that is compatible with Matlab/Fortran/Julia indexing
        (starting at 1, instead of 0).

    Returns
    -------
    elements: ElementsType
        elements of the mesh

    Example
    -------
    >>> elements_path = Path("path/to/elements_from_matlab_code.txt")
    >>> elements = read_elements(elements_path, shift_indices=True)
    """
    elements = np.loadtxt(path_to_elements).astype(np.uint32)
    if shift_indices:
        elements = elements - 1
    return elements


def read_mesh(path_to_coordinates: Path,
              path_to_elements: Path,
              shift_indices: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads vertices and elements from files and
    returns the corresponding Mesh.

    Parameters
    ----------
    path_to_coordinates : pathlib.Path
        Path to the file containing mesh coordinates data.
    path_to_elements : pathlib.Path
        Path to the file containing mesh elements data.
    shift_indices: bool (default=False)
        If true, shifts the elements' indices according to i':=i-1.
        This can come in handy if, e.g., one wants to read data
        that is compatible with Matlab/Fortran/Julia indexing
        (starting at 1, instead of 0).

    Returns
    -------
    coordinates: CoordinatesType
        coordinates of the mesh's vertices
    elements: ElementsType
        elements of the mesh

    Example
    -------
    >>> coordinates_path = Path("path/to/coordinates.txt")
    >>> elements_path = Path("path/to/elements.txt")
    >>> coordinates, elements = read_mesh(coordinates_path, elements_path)
    """
    coordinates = read_coordinates(path_to_coordinates=path_to_coordinates)
    elements = read_elements(path_to_elements=path_to_elements,
                             shift_indices=shift_indices)
    return coordinates, elements
