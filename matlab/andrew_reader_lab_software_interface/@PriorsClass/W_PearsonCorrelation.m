function Wg = W_PearsonCorrelation(ObjPrior,Img,T)

imgSize = size(Img);
if ~all(ObjPrior.CropedImageSize(1:2) == imgSize(1:2))
    Img = ObjPrior.imCrop(Img);
end

nVoxels = prod(ObjPrior.CropedImageSize);
Wg = zeros(nVoxels,ObjPrior.nS,'single');

for i = 1:ObjPrior.chunkSize:nVoxels
    voxels = i: min(i+ObjPrior.chunkSize-1,nVoxels);
    
    imgPatch = Img(ObjPrior.LocalWindow(ObjPrior.SearchWindow(voxels,:),:));
    imgPatch = imgPatch - repmat(mean(imgPatch,2),[1,ObjPrior.nL]);
    imgLocalWindow = Img(ObjPrior.LocalWindow(voxels,:));
    imgLocalWindow = imgLocalWindow - repmat(mean(imgLocalWindow,2),[1,ObjPrior.nL]);
    
    imgLocalWindow = repmat(imgLocalWindow,[ObjPrior.nS,1]);
%     MeanImgPatch = repmat(mean(imgPatch,3),[1,1,)
    
D = sum(imgPatch.*imgLocalWindow,2)./sqrt(sum(imgPatch.^2,2).*sum(imgLocalWindow.^2,2));

    
    D = reshape(D,length(voxels),ObjPrior.nS);
    clear imgPatch imgLocalWindow
    

    Wg(voxels,:) = D;%
    clear D D_norm
end
% check for errors
Wg = Wg>=T;
% sumWg = sum(Wg,2);
% i = sumWg==0 | isinf(sumWg) | isnan(sumWg);
% Wg(i,:) = 1./ObjPrior.nS;

end