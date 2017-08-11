function dP = d_smoothed_TV_prior(ObjPrior,Img,beta)
% beta, TV smoothness parameter

% imgGrad = ObjPrior.GraphGrad(Img);
% Norm = repmat(sqrt(sum(imgGrad.^2,2)+ beta),[1,ObjPrior.nS]);
% imgGrad = imgGrad./Norm;
% dP = -1* sum(ObjPrior.Wd.*imgGrad,2);

Norm = ObjPrior.MagnitudGraphGradWithSpatialWeight(Img, beta);
dP = -1* ObjPrior.GraphGradWithSpatialWeight(Img)./Norm;
dP(Norm == 0) = 0;
end