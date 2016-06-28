function Wb = W_Bowsher(ObjPrior,Img,B)
% B: user defined number of the neighboring voxels that have the
% highest similarity on the anatomical image based on thier
% absoulte intensity differences
imgSize = size(Img);
if ~all(ObjPrior.CropedImageSize(1:2) == imgSize(1:2))
    Img = ObjPrior.imCrop(Img);
end
if B > ObjPrior.nS
    error ('B can be in maximum %d\n',ObjPrior.nS)
end
abs_imgGrad = abs(ObjPrior.GraphGrad(Img));
Wb = 0*abs_imgGrad;

for i = 1:size(abs_imgGrad,1)
    [~,idx] = sort(abs_imgGrad(i,:));
    
    Wb(i,idx(1:B)) = 1;
end

end