load elements.dat
load coordinates.dat
load dirichlet.dat
load neumann.dat

% adapt this path to a clone of https://github.com/aschmidtuulm/ameshref
addpath("path/to/ameshref/refinement/")

% -------------------------------------------------------------------------
% case no_boundary:
% -------------------------------------------------------------------------
clear TrefineRG
markedElements = [4];

[new_coordinates, new_elements, new_dirichlet, new_neumann] ...
    = TrefineRG(coordinates, elements, dirichlet, neumann, markedElements);

save("case_no_boundary/new_coordinates.dat", "new_coordinates", "-ascii")
save("case_no_boundary/new_elements.dat", "new_elements", "-ascii")
save("case_no_boundary/new_dirichlet.dat", "new_dirichlet", "-ascii")
save("case_no_boundary/new_neumann.dat", "new_neumann", "-ascii")

% visualize mesh
figure(1)
trisurf(new_elements, new_coordinates(:,1), new_coordinates(:,2), zeros(size(new_coordinates, 1), 1),'facecolor','interp')
title(sprintf('# Elements = %s',int2str(size(new_elements,1))),'FontSize',20);
axis([-0.1 3.1 -0.1 3.1])
axis equal
axis off
view(2)

% -------------------------------------------------------------------------
% case dirichlet:
% -------------------------------------------------------------------------
clear TrefineRG
markedElements = [1];

[new_coordinates, new_elements, new_dirichlet, new_neumann] ...
    = TrefineRG(coordinates, elements, dirichlet, neumann, markedElements);

save("case_dirichlet/new_coordinates.dat", "new_coordinates", "-ascii")
save("case_dirichlet/new_elements.dat", "new_elements", "-ascii")
save("case_dirichlet/new_dirichlet.dat", "new_dirichlet", "-ascii")
save("case_dirichlet/new_neumann.dat", "new_neumann", "-ascii")

% visualize mesh
figure(1)
trisurf(new_elements, new_coordinates(:,1), new_coordinates(:,2), zeros(size(new_coordinates, 1), 1),'facecolor','interp')
title(sprintf('# Elements = %s',int2str(size(new_elements,1))),'FontSize',20);
axis([-0.1 3.1 -0.1 3.1])
axis equal
axis off
view(2)

% -------------------------------------------------------------------------
% case neumann:
% -------------------------------------------------------------------------
clear TrefineRG
markedElements = [8];

[new_coordinates, new_elements, new_dirichlet, new_neumann] ...
    = TrefineRG(coordinates, elements, dirichlet, neumann, markedElements);

save("case_neumann/new_coordinates.dat", "new_coordinates", "-ascii")
save("case_neumann/new_elements.dat", "new_elements", "-ascii")
save("case_neumann/new_dirichlet.dat", "new_dirichlet", "-ascii")
save("case_neumann/new_neumann.dat", "new_neumann", "-ascii")

% visualize mesh
figure(1)
trisurf(new_elements, new_coordinates(:,1), new_coordinates(:,2), zeros(size(new_coordinates, 1), 1),'facecolor','interp')
title(sprintf('# Elements = %s',int2str(size(new_elements,1))),'FontSize',20);
axis([-0.1 3.1 -0.1 3.1])
axis equal
axis off
view(2)