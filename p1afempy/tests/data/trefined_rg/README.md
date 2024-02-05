Here, we generate the data used to test our python
implementation of the reg-green refinement strategy
against the MATLAB function `TrefineRG` found in this repo:
https://github.com/aschmidtuulm/ameshref

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