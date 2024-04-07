load elements_matlab.dat

element2neighbours = get_element_to_neighbour(elements)

save("element2neighbours_matlab.dat", "element2neighbours", "-ascii")

% the following is taken from:
% S. Funken, D. Praetorius, and P. Wissgott, Efficient Implementation of 
% Adaptive P1-FEM in Matlab, lines 5-12 in listing 5.3.
function [element2neighbours] = get_element_to_neighbour(elements)
    %*** Obtain geometric information on neighbouring elements
    nE = size(elements, 1);
    I = elements(:);
    J = reshape(elements(:,[2,3,1]),3*nE,1);
    nodes2edge = sparse(I,J,1:3*nE);
    mask = nodes2edge>0;
    [foo{1:2},idxIJ] = find( nodes2edge );
    [foo{1:2},neighbourIJ] = find( mask + mask.*sparse(J,I,[1:nE,1:nE,1:nE]') );
    element2neighbours(idxIJ) = neighbourIJ - 1;
    element2neighbours = reshape(element2neighbours,nE,3);
end