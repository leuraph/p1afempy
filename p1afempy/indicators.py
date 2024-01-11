import numpy as np
from p1afempy import mesh
from typing import Callable


def compute_eta_r(x: np.ndarray, mesh: mesh.Mesh,
                  dirichlet: mesh.BoundaryCondition,
                  neumann: mesh.BoundaryCondition,
                  f: Callable[[np.ndarray], float],
                  g: Callable[[np.ndarray], float]) -> np.ndarray:
    """
    # TODO doc
    """
    # TODO implement
    pass
