load elements.dat
load coordinates.dat
load dirichlet.dat
load neumann.dat

addpath('path/to/p1afem')

n_refinements = 11;
n_repetitions_each = 4;
n_elements = zeros(n_refinements, 1);
means = zeros(n_refinements, 1);
stdevs = zeros(n_refinements, 1);

for k = 1:n_refinements

    % do a refinement first (otherwisem we would start with boundary only)
    marked = 1:size(elements, 1); % mark all elements for refinement
    [coordinates,elements,dirichlet,neumann] = ...
           refineNVB(coordinates,elements,dirichlet,neumann,marked);
    
    tmp_times = zeros(n_repetitions_each, 1); % used to compute mean and stdev
    for rep = 1:n_repetitions_each
        tStart = cputime;
        % -----------------------------------------------------------------
        % running p1afem's solveLaplace
        % -----------------------------------------------------------------
        [x, ~] = solveLaplace(coordinates, elements, dirichlet, neumann, @f, @g, @uD);
        % -----------------------------------------------------------------
        % end of copied block
        % -----------------------------------------------------------------
        tEnd = cputime;

        dt = tEnd - tStart;
        tmp_times(rep) = dt;
    end
    
    means(k) = mean(tmp_times);
    stdevs(k) = std(tmp_times);
    n_elements(k) = size(elements, 1);
end

% trisurf(elements(:,1:3),coordinates(:,1),coordinates(:,2),x','facecolor','interp')
% view(15,22)

save("means.dat", "means", "-ascii")
save("stdevs.dat", "stdevs", "-ascii")
save("n_elements.dat", "n_elements", "-ascii")
