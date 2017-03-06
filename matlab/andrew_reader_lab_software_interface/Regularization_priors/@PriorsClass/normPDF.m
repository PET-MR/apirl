function x = normPDF(ObjPrior,x,y,sigma)

x = 1./sigma./sqrt(2*pi).*exp(-0.5*(x-y).^2./sigma.^2);

end