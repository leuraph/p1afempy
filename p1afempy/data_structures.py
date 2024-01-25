import numpy as np
from typing import Callable

"""
This files serves as a lookup dictionary on the data structures
used throughout this package. Note that the data structres used
here coincide with the data structures defined in [1].

References
----------
- [1] S. Funken, D. Praetorius, and P. Wissgott.
      Efficient Implementation of Adaptive P1-FEM in Matlab,
      (http://dx.doi.org/10.2478/cmam-2011-0026).
      Computational Methods in Applied Mathematics,
      Vol. 11 (2011), No. 4, pp. 460–490.
"""

ElementsType = np.ndarray
"""
The triangulation T is represented by the Mx3 integer array elements.
The l-th triangle T_l = conv{zi, zj , zk} ∈ T with vertices zi, zj , zk ∈ N
is stored as elements[l, :] = [i, j, k],
where the nodes are given in counterclockwise order,
i.e. the parametrization of the boundary is mathematically positive.
"""

CoordinatesType = np.ndarray
"""
The set of all nodes is represented by the Nx2 array coordinates.
The k-th row of coordinates stores the coordinates of the k-th node, i.e.
z_l = (x_l, y_l) ∈ R2 as coordinates[k, :] = [x_k, y_k].
"""

BoundaryType = np.ndarray
"""
The Dirichlet boundary is split into K affine boundary pieces,
which are edges of triangles T ∈ T.
It is represented by a Kx2 integer array dirichlet.
The l-th edge E_l = conv{zi, zj} on the Dirichlet boundary is stored
in the form dirichlet[l, :] = [i, j].
"""

BoundaryConditionType = Callable[[CoordinatesType], np.ndarray]
"""
Any function specifying a boundary condition is expected to
be implemented such that it can be called on the Nx2 array coordinates
and returns an Nx1 array, where the function was applied row-wise.
"""
