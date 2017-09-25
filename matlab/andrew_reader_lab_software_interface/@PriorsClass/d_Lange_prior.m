function dP = d_Lange_prior(ObjPrior,Img, params)

% imgGrad = ObjPrior.GraphGrad(Img);
% Norm = repmat(sqrt(sum(ObjPrior.Wd.*imgGrad.^2,2)+ beta),[1,ObjPrior.nS]);
% imgGrad = imgGrad./Norm.*(1 + delta./(delta + Norm));
% dP = -1* sum(ObjPrior.Wd.*imgGrad,2);
if strcmp(ObjPrior.PriorImplementation, 'matlab')
    imgGrad = ObjPrior.GraphGrad(Img);
    imgGrad = imgGrad./(params.LangeDeltaParameter + abs(imgGrad));
    dP = -1* sum(ObjPrior.Wd.*imgGrad,2);
elseif strcmp(ObjPrior.PriorImplementation, 'mex-cuda')
    enableSpatialWeight = 1; % For TV, in my opinion needs to be included, but it wasn't in the matlab implementation.
    typeOfLange = 1; % Local
    dP =  mexGPULange(single(Img), ObjPrior.sWindowSize, ObjPrior.sWindowSize, ObjPrior.sWindowSize, params.LangeDeltaParameter, enableSpatialWeight, typeOfLange); % This function is not part of the prior class because is a mex file.
end


end