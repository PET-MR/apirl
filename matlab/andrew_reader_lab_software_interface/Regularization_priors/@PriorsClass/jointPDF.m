function p = jointPDF(ObjPrior, x,y,imgF,imgA,sigma_x,sigma_y)
% joint pdfs using the Parzen window method
p  = ObjPrior.normPDF(x,imgF(:),sigma_x).*ObjPrior.normPDF(y,imgA(:),sigma_y);
p = sum(p)./numel(p);
end