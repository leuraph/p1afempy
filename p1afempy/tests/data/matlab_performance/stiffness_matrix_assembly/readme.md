# Performance Test: Stiffness Matrix Assembly

In this directory, you can find everything needed to perform
a performance test on the Matlab implementation of the stiffness matrix
assembly.
For completeness' sake (e.g. if you do not posess a Matlab license),
we added test results found on our machine,
whose specs can be found below.

| **Device**       | MacBook Pro 15-inch, 2018       |
|-------------------|---------------------------------|
| **Processor**    | 2.6 GHz 6-Core Intel Core i7    |
| **Graphics**     | Radeon Pro 560X 4 GB            |
|                  | Intel UHD Graphics 630 1536 MB |
| **Memory**       | 16 GB 2400 MHz DDR4             |
| **Operating System** | MacOS 13.6.3 (22G436)         |
| **Matlab Version**   | R2023b                          |



## Setup
The test case considered is a simple unit square.
In each refinement step, we refine the mesh using `RefineNVB`,
where all elements are marked for refinement.
