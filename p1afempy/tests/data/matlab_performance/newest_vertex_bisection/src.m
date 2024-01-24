load elements.dat
load coordinates.dat
load dirichlet.dat
load neumann.dat

addpath 'path/to/p1afem'

n_refinements = 11;
n_repetitions_each = 10;
n_elements = zeros(n_refinements, 1);
means = zeros(n_refinements, 1);
stdevs = zeros(n_refinements, 1);

for k = 1:n_refinements
    nE = size(elements, 1);
    marked = 1:size(elements,1);

    tmp_times = zeros(n_repetitions_each, 1);

    for rep = 1:n_repetitions_each
        tStart = cputime;
        % -----------------------------------------------------------------
        % code block copied from refineNVB.m
        % -----------------------------------------------------------------
        [~, ~ , ~, ~] = ...
            refineNVB(coordinates,elements,dirichlet,neumann,marked);
        % -----------------------------------------------------------------
        % end of copied block
        % -----------------------------------------------------------------
        tEnd = cputime;
        dt = tEnd - tStart;
        tmp_times(rep) = dt;
    end
    
    means(k) = mean(tmp_times);
    stdevs(k) = std(tmp_times);
    n_elements(k) = nE;

    [coordinates,elements,dirichlet,neumann] = ...
           refineNVB(coordinates,elements,dirichlet,neumann,marked);
end

loglog(n_elements, means);

save("means.dat", "means", "-ascii")
save("stdevs.dat", "stdevs", "-ascii")
save("n_elements.dat", "n_elements", "-ascii")