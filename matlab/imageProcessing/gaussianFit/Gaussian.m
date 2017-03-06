function gauss = Gaussian(x,X)

% This function is called by LSQNONLIN.
% x is a vector which contains the coefficients of the
% equation. X and Y are the data

A1 = x(1);
A2 = x(2);
A3 = x(3);

gauss = A1.*exp(-0.5*(X-A2).^2/A3^2);


end
