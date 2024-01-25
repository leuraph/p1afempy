function out = f(x)
%Volume force, right hand side of Laplace equation: - div(grad(u)) = f

omega = 7./4.*pi;

out = 5. * omega^2 * sin(omega*2.*x(:,1)).*sin(omega*x(:,2));
