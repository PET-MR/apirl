function dP = d_smoothed_nonlocal_TV_prior(ObjPrior,Img,beta,nl_weights)
% beta, TV smoothness parameter
% nl_weights, non-local weights calculated from ObjPrior.W_GaussianKernel() or W_Bowsher()


imgGrad = ObjPrior.GraphGrad(Img).*nl_weights;
Norm = repmat(sqrt(sum(imgGrad.^2,2)+ beta),[1,ObjPrior.nS]);
imgGrad = imgGrad./Norm;

dP = -1* sum(ObjPrior.Wd.*imgGrad,2);

end