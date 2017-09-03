function dP = d_nonlocal_Tikhonov_prior(ObjPrior,Img,nl_weights, params) % params only for compatibility, but it's not really needed

% params is not necessary for 'matlab' implementation because the bowsher
% coefficients have been already computed at this stage but 
% nl_weights, non-local weights clculated from ObjPrior.W_GaussianKernel or W_Bowsher()

dP = -2*ObjPrior.GraphGradWithSpatialWeightAndSimilarity(Img, nl_weights, params); %sum(nl_weights.*ObjPrior.Wd.*ObjPrior.GraphGrad(Img),2);

end