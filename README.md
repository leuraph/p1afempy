# P1AFEM-PY

This package is the pythonic adaption of the p1afem Matlab package,
whose code can be found
[here (ZIP)](https://www.tuwien.at/index.php?eID=dumpFile&t=f&f=180536&token=1b5f89369acab20d59455e42569bf1e0b2db8b41)
and whose details are described in the paper (open access)
[Computational Methods in Applied Mathematics, 11 (2011)](http://dx.doi.org/10.2478/cmam-2011-0026).

## Performance tests

In order to perform a profiled performance test, you can use the existing sccripts in
the manual tests directory, i.e. `p1afempy/tests/manual`.
For example, to perform a profiled test, you can do

```sh
cd p1afempy
python -m cProfile -s time -m tests.manual.<script> > benchmark.out
```

### Notes

First, note the following.
In the `solve_laplace` function, we make use of `scipy.sparse.linalg.spsolve`.
Note that we explicitly set `use_umfpack` to `True`.
In the documentation (`scipy==1.11.4`) of this function, we read the following.

> if True (default) then use UMFPACK for the solution.
> This is only referenced if b is a vector and ``scikits.umfpack`` is installed.

Therefore, in order to make use of the performance upgrade, you should make sure you have
`scikits.umfpack` installed.
Secondly, we point out the following problem when trying to install `scikits.umfpack` on a mac.
In orderto install `scikits.umfpack` via pip, you (along other things) need to make sure you
have a working version of `suite-sparse` installed on your machine (`scikits.umfpack` will use
its headers and link against its library).
However, it seems that using the `suite-sparse` version shipped via homebrew does conflict
with the `scikits.umfpack` version installed via pip.
For a reference, check the following [issue](https://github.com/scikit-umfpack/scikit-umfpack/issues/98) on github.
The problem is resolved when installing `suite-sparse` via `conda`.
Thirdly, when installing `scikits.umfpack`, your machine may not automatically detect `suite-sparse`'s
header and library files.
In order to resolve this issue, you can install `scikits.umfpack` by first creating a `nativefile.ini`
file with the content as listed further below and then do:

```sh
pip install --config-settings setup-args=--native-file=$PWD/nativefile.ini scikit-umfpack
```

The `nativefile.ini` file should look like this:
```ini
[properties]
umfpack-libdir = 'path/to/umfpack/lib/files'
umfpack-includedir = 'path/to/umfpack/include/files'
```
