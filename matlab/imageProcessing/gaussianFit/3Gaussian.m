function gauss = Gaussian3(x,X)

% This function is called by LSQNONLIN.
% x is a vector which contains the coefficients of the
% equation. X and Y are the data

A1 = x(1);
A2 = x(2);
A3 = x(3);
A4 = x(4);
A5 = x(5);
A6 = x(6);
A7 = x(7);
A8 = x(8);
A9 = x(9);

g1 = A1.*exp(-0.5*(X-A2).^2/A3^2);
g2 = A4.*exp(-0.5*(X-A5).^2/A6^2);
g3 = A7.*exp(-0.5*(X-A8).^2/A9^2);

g = g1 + g2 + g3 ;

end
