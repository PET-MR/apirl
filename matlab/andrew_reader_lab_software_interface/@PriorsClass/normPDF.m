function x = normPDF(ObjPrior,x,y,sigma)

x = 1./sqrt(2*pi*sigma^2).*exp(-0.5*(x-y).^2./sigma.^2);

end