Here, we hardcoded the input and expected output data
for our implementation of the red-green refinement of a sinngle
element.

## Test Cases

Starting from the same initial mesh,
we generate the following data for tests:

- case_no_boundary:
  - we mark one element for refinenemt
  - the marked element's edges do not lie on any boundary
- case_dirichlet
  - we mark one element for refinenemt
  - one of the element's edges lies on the dirichlet boundary
- case_neumann
  - we mark one element for refinenemt
  - one of the element's edges lies on the neumann boundary