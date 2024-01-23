load elements.dat
load coordinates.dat
load dirichlet.dat
load neumann.dat

n_refinements = 11;
n_repetitions_each = 20;
n_elements = zeros(n_refinements, 1);
means = zeros(n_refinements, 1);
stdevs = zeros(n_refinements, 1);

for k = 1:n_refinements
    nE = size(elements, 1);

    tmp_times = zeros(n_repetitions_each, 1);
    for rep = 1:n_repetitions_each
        tStart = cputime;
        % -----------------------------------------------------------------
        % code block copied from solvelaplace.m
        % -----------------------------------------------------------------
        %*** First vertex of elements and corresponding edge vectors 
        c1 = coordinates(elements(:,1),:);
        d21 = coordinates(elements(:,2),:) - c1;
        d31 = coordinates(elements(:,3),:) - c1;
        %*** Vector of element areas 4*|T|
        area4 = 2*(d21(:,1).*d31(:,2)-d21(:,2).*d31(:,1));
        %*** Assembly of stiffness matrix
        I = reshape(elements(:,[1 2 3 1 2 3 1 2 3])',9*nE,1);
        J = reshape(elements(:,[1 1 1 2 2 2 3 3 3])',9*nE,1);
        a = (sum(d21.*d31,2)./area4)';
        b = (sum(d31.*d31,2)./area4)';
        c = (sum(d21.*d21,2)./area4)';
        A = [-2*a+b+c;a-b;a-c;a-b;b;-a;a-c;-a;c];
        A = sparse(I,J,A(:));
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

    marked = 1:size(elements,1);
    [coordinates,elements,dirichlet,neumann] = ...
           refineNVB(coordinates,elements,dirichlet,neumann,marked);
end

loglog(n_elements, means);

save("means.dat", "means", "-ascii")
save("stdevs.dat", "stdevs", "-ascii")
save("n_elements.dat", "n_elements", "-ascii")