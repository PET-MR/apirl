function g = Gaussian2(x,X)

% This function is called by LSQNONLIN.
% x is a vector which contains the coefficients of the
% equation. X and Y are the data

A1 = x(1);
A2 = x(2);
A3 = x(3);
A4 = x(4);
A5 = x(5);
A6 = x(6);


g1 = A1.*exp(-0.5*(X-A2).^2/A3^2);
g2 = A4.*exp(-0.5*(X-A5).^2/A6^2);

g = g1 + g2;

end
