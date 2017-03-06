function N  = normalVectors(ObjPrior,Img)

imgSize = size(Img);
if ~all(ObjPrior.CropedImageSize(1:2) == imgSize(1:2))
    Img = ObjPrior.imCrop(Img);
end

imgGrad = ObjPrior.GraphGrad(Img);

N = imgGrad./repmat(ObjPrior.L2Norm(imgGrad)+1e-1,[1,ObjPrior.nS]);

end