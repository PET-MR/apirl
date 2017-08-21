function dP = d_nonlocal_Tikhonov_prior(ObjPrior,Img,nl_weights, params) % params only for compatibility, but it's not really needed

% nl_weights, non-local weights clculated from ObjPrior.W_GaussianKernel or W_Bowsher()

dP = -2*ObjPrior.GraphGradWithSpatialWeightAndSimilarity(Img, nl_weights); %sum(nl_weights.*ObjPrior.Wd.*ObjPrior.GraphGrad(Img),2);

end