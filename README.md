# P1AFEM-PY

This package is the pythonic adaption of the p1afem Matlab package,
whose code can be found
[here (ZIP)](https://www.tuwien.at/index.php?eID=dumpFile&t=f&f=180536&token=1b5f89369acab20d59455e42569bf1e0b2db8b41)
and whose details are described in the paper (open access) [[1]](#1).

## What is (not) provided

In the following, we provide a list indicating which functions from P1AFEM
are implemented in this repo as well (ticked boxes) and which are not (yet) (unticked boxes).

- [ ] `adaptiveAlgorithm.m`
- [x] `coarsenNVB.m`
- [ ] `computeEtaH.m`
- [x] `computeEtaR.m`
- [ ] `computeEtaZ.m`
- [ ] `example1/`
- [ ] `example2/`
- [x] `provideGeometricData.m`
- [ ] `refineMRGB.m`
- [x] `refineNVB.m`
- [ ] `refineNVB1.m`
- [ ] `refineNVB5.m`
- [x] `refineRGB.m`
- [x] `solveLaplace.m`
- [ ] `solveLaplace0.m`
- [ ] `solveLaplace1.m`

Also, this repo includes some functionalities that were not provided in the original MATLAB code.
The most important of which is the assembly of the mass matrix,
which is implemented along the same lines as the assembly of the stiffness matrix.

## Data structures

Regarding the underlying data structures used, we follow the original code as closely as possible, i.e.
elements, coordinates, and boundary conditions are all handled as simple `numpy` arrays.
We abstained from implementing any additional data structures,
e.g. classes like `Mesh` or `BoundaryCondition`, in order to remain the original "low-level" usability of the code.
In this way, any user can decide whether to implement additional data structures and, possibly, wrappers thereof.

As a quick reference, we refer to figure 3.1 below (copied from [[1]](#1)).
For more details about the expected format of the data structures
we refer to chapter 3.1 of [[1]](#1).

![](https://raw.githubusercontent.com/leuraph/p1afempy/main/figures/newest_vertex_bisection.png "Figure 3.1 from ref. [1]")

## Performance tests

In order to perform a profiled performance test, you can use the existing scripts in
the manual tests directory, i.e. `p1afempy/tests/manual`.
For example, to perform a profiled test, you can do

```sh
cd p1afempy
python -m cProfile -s time -m tests.manual.<script> > benchmark.out
```

Below, you can find some performance test results found on a reference machine [[2]](#2).
The error bars in the plots represent the standard deviation of measured CPU time.

### Stiffness Matrix Assembly

The script used to measure and compare python performance is located at
`p1afempy/tests/manual/performance_test_stiffnes_matrix.py`.
On each mesh, we performed $20$ measurements.
For more information, see
`p1afempy/tests/data/matlab_performance/stiffness_matrix_assembly/readme.md`.

![](https://raw.githubusercontent.com/leuraph/p1afempy/main/figures/stiffness_matrix_assembly.png "Stiffness Matrix Assembly Performance Comparison")

### Newest Vertex Bisection

The script used to measure and compare python performance is located at
`p1afempy/tests/manual/performance_test_refineNVB.py`.
In every iteration, we marked all elements for refinement and measured the CPU time needed
for the refinement $10$ times.
For more information, see
`p1afempy/tests/data/matlab_performance/newest_vertex_bisection/readme.md`.

![](https://raw.githubusercontent.com/leuraph/p1afempy/main/figures/newest_vertex_bisection.png "Newest Vertex Bisection refinement performance comparison")


### Solve Laplace

The script used to measure and compare python performance is located at
`p1afempy/tests/manual/performance_test_solve_laplace.py`.
In every iteration, i.e. on each mesh, we measured the CPU time needed for solving $4$ times.
For more information, see
`p1afempy/tests/data/matlab_performance/solve_laplace/readme.md`.

![](https://raw.githubusercontent.com/leuraph/p1afempy/main/figures/solve_laplace.png "solve laplace performance comparison")

## Performance upgrade

In the following, we describe how to get more (the most) performance out of `solve_laplace`.

### Use UMFPACK

**TL;DR;**
*Make sure you have `scikit.umfpack` installed (can be found on [pypi](https://pypi.org/project/scikit-umfpack/)).*

In the `solve_laplace` function, we make use of `scipy.sparse.linalg.spsolve` and explicitly set `use_umfpack` to `True`.
However, in the documentation (`scipy==1.11.4`) of this function, we read the following.

> if True (default) then use UMFPACK for the solution.
> This is only referenced if b is a vector and ``scikit.umfpack`` is installed.

Therefore, make sure you have `scikit.umfpack` installed (can be found on [pypi](https://pypi.org/project/scikit-umfpack/)).
In case your installation can not figure out where to find the UMFPACK (Suite-Sparse) headers and library
or you want to make use of your own Suite-Sparse version, 

### <a name="openblas"></a>Do not link Suite-Sparse against OpenBLAS

**TL;DR;**
*Make sure the Suite-Sparse library your scikit-umfpack is pointing to does not link against OpenBLAS but rather against either Intel MKL BLAS or, if you are on a mac, the BLAS and LAPACK libraries under the Accelerate framework.*

Note that Suite-Sparse makes use of BLAS routines.
As can be read in 
[this issue](https://github.com/DrTimothyAldenDavis/SuiteSparse/issues/1) 
and 
[this part of the readme](https://github.com/DrTimothyAldenDavis/SuiteSparse?tab=readme-ov-file#about-the-blas-and-lapack-libraries),
in a 2019 test, OpenBLAS caused severe performance degradation.
Therefore, it is recommended that your Suite-Sparse library (used by scikit-umfpack) links against the corresponding BLAS library.
Hence, you need to:

- Ensure that the Suite-Sparse library used by scikit-umfpack is pointing to the correct BLAS library. Instructions on how to link Suite-Sparse to a custom BLAS library
can be found [in the very same part of the readme](https://github.com/DrTimothyAldenDavis/SuiteSparse?tab=readme-ov-file#about-the-blas-and-lapack-libraries) as mentioned above.
- Make sure your installation of scikit-umfpack is using the correct Suite-Sparse library, i.e. one that points to the correct BLAS library.
To install scikit-umfpack and make it use a custom Suite-Sparse library, follow the steps mentioned in the [troubleshooting](#troubleshooting) section below.

### <a name="troubleshooting"></a>Troubleshooting

#### Installing scikit-umfpack on a mac

It seems that using the `suite-sparse` version shipped via Homebrew conflicts with the `scikit-umfpack` version installed via pip.
For reference, check the following [issue](https://github.com/scikit-umfpack/scikit-umfpack/issues/98) on GitHub.
An easy way around this would be to install `suite-sparse` via `conda`, as it ships an older version that seems to be compatible.
However, conda comes with OpenBLAS, which causes a dramatic performance degredation (as mentioned [above](openblas)).
In order to resolve the issue and not fall into a performance degredation pitfall, make sure you have a compatible version of Suite-Sparse (as mentioned in [this isse](https://github.com/scikit-umfpack/scikit-umfpack/issues/98); at least v5.10.1 seems to work) available, linking against the correct BLAS library.
Finally, install scikit-umfpack making use of this Suite-Sparse installation (instructions on how to install scikit-umfpack with a custom Suite-Sparse are described [below](#custom)).

#### <a name="custom"></a>Installing scikit-umfpack with custom Suite-Sparse

In order to install scikit-umfpack pointing to a custom Suite-Sparse, you first create a `nativefile.ini` with the content as listed further below and then do:

```sh
pip install --config-settings setup-args=--native-file=$PWD/nativefile.ini scikit-umfpack
```

The `nativefile.ini` should look like this:
```ini
[properties]
umfpack-libdir = 'path/to/suite-sparse/lib'
umfpack-includedir = 'path/to/suite-sparse/include'
```

## References

<a id="1">[1]</a> 
S. Funken, D. Praetorius, and P. Wissgott.
[Efficient Implementation of Adaptive P1-FEM in Matlab](http://dx.doi.org/10.2478/cmam-2011-0026).
Computational Methods in Applied Mathematics, Vol. 11 (2011), No. 4, pp. 460–490.

<a id="2">[2]</a>
Reference Machine:
| **Device**       | MacBook Pro 15-inch, 2018       |
|-------------------|---------------------------------|
| **Processor**    | 2.6 GHz 6-Core Intel Core i7    |
| **Graphics**     | Radeon Pro 560X 4 GB            |
|                  | Intel UHD Graphics 630 1536 MB |
| **Memory**       | 16 GB 2400 MHz DDR4             |
| **Operating System** | MacOS 13.6.3 (22G436)         |
| **Matlab Version**   | R2023b                          |