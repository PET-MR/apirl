function diff = diff_3Gaussian(x,X,Y)

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

diff1 = A1.*exp(-0.5*(X-A2).^2/A3^2);
diff2 = A4.*exp(-0.5*(X-A5).^2/A6^2);
diff3 = A7.*exp(-0.5*(X-A8).^2/A9^2);

diff = diff1 + diff2 + diff3 - Y;

end
