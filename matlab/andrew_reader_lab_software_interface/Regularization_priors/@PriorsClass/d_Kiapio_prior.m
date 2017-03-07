function dP = d_Kiapio_prior(ObjPrior,Img,normVectors,alpha)
% normVectors: normal vectors pre-calculated by ObjPrior.normalVectors(mri_image)
% alpha: weights the imapct of the normal vectors

% dP = Dt(D(x) - alpha*N*Nt*D(x))

imgGrad = ObjPrior.GraphGrad(Img);
inprod = sum(normVectors.*imgGrad,2);
K = normVectors.*(repmat(inprod,[1,ObjPrior.nS]));
dP = -1* sum(ObjPrior.Wd.*(imgGrad -alpha.*K ),2);

end

% Refernce: Ehrhardt,et al PET Reconstruction with an Anatomical MRI Prior
% using Parallel Level Sets, IEEE TMI, 2016
% Eq.(10)