function out = uD(x)
%Dirichlet boundary data: u = u_D

% initialized as 'ones' to detect bugs easier 
% (all of out should eventually become 0)
out = ones(size(x,1),1);
out(x(:,1) == 0) = 0;
out(x(:,2) == 0) = 0;
