function dP = d_nonlocal_Tikhonov_prior(ObjPrior,Img,nl_weights)

% nl_weights, non-local weights clculated from ObjPrior.W_GaussianKernel or W_Bowsher()

dP = -2* sum(nl_weights.*ObjPrior.Wd.*ObjPrior.GraphGrad(Img),2);

end