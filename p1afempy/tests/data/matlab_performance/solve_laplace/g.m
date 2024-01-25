function out = g(x)
%Neumann boundary data: d/dn u = g

g_right_indices = (x(:, 1) == 1);
g_upper_indices = (x(:, 2) == 1);

omega = 7./4.*pi;

out = zeros(size(x, 1), 1);
out(g_right_indices) = omega*2.*sin(omega*x(g_right_indices, 2)).*cos(omega*2.*x(g_right_indices, 1));
out(g_upper_indices) = omega*sin(omega*2.*x(g_upper_indices, 1)).*cos(omega*x(g_upper_indices, 2));