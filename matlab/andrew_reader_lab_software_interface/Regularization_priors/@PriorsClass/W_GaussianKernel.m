function Wg = W_GaussianKernel(ObjPrior,Img,KernelSigma)

imgSize = size(Img);
if ~all(ObjPrior.CropedImageSize(1:2) == imgSize(1:2))
    Img = ObjPrior.imCrop(Img);
end

nVoxels = prod(ObjPrior.CropedImageSize);
Wg = zeros(nVoxels,ObjPrior.nS,'single');

normalize = 1;
for i = 1:ObjPrior.chunkSize:nVoxels
    voxels = i: min(i+ObjPrior.chunkSize-1,nVoxels);
    
    imgPatch = Img(ObjPrior.LocalWindow(ObjPrior.SearchWindow(voxels,:),:));
    imgLocalWindow = Img(ObjPrior.LocalWindow(voxels,:));
    
    
    imgLocalWindow = repmat(imgLocalWindow,[ObjPrior.nS,1]);
    D = reshape(sum( (imgPatch - imgLocalWindow).^2, 2 ),length(voxels),ObjPrior.nS);
    clear imgPatch imgLocalWindow
    
    if normalize
        D_norm = repmat(max(sqrt(sum(D.^2,2)),eps),[1,ObjPrior.nS]);
    else
        D_norm = 1;
    end
    Wg(voxels,:) = exp( -D/KernelSigma^2 ./D_norm);%
    clear D D_norm
end
% check for errors
sumWg = sum(Wg,2);
i = sumWg==0 | isinf(sumWg) | isnan(sumWg);
Wg(i,:) = 1./ObjPrior.nS;

end