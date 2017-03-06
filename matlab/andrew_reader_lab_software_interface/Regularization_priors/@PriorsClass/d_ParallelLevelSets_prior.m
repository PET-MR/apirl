function dP = d_ParallelLevelSets_prior(ObjPrior,Img,normVectors,alpha,beta)

% normVectors: normal vectors pre-calculated by ObjPrior.normalVectors(mri_image)
% alpha: weights the imapct of the normal vectors

% dP = Dt(W(D(x) - alpha*N*Nt*D(x)))

imgGrad = ObjPrior.GraphGrad(Img);
Norm = repmat(sqrt(sum(imgGrad.^2,2)+ beta),[1,ObjPrior.nS]);

inprod = sum(normVectors.*imgGrad,2);
K = normVectors.*(repmat(inprod,[1,ObjPrior.nS]));
num = imgGrad./Norm -alpha.*K;
denum = repmat(sqrt(sum(imgGrad.^2,2)+ beta -alpha*inprod.^2 ),[1,ObjPrior.nS]);
dP = -1* sum(ObjPrior.Wd.*(num./denum ),2);

end


% Refernce: Ehrhardt,et al PET Reconstruction with an Anatomical MRI Prior
% using Parallel Level Sets, IEEE TMI, 2016
% Eq.(9)