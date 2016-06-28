function Wje = W_JointEntropy(ObjPrior,Img,sigma)

imgSize = size(Img);
if ~all(ObjPrior.CropedImageSize(1:2) == imgSize(1:2))
    Img = ObjPrior.imCrop(Img);
end

Wje = ObjPrior.normPDF(ObjPrior.GraphGrad(Img),0,sigma);

end