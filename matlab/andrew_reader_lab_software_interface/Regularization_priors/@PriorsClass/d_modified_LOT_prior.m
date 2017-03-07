function dP = d_modified_LOT_prior(ObjPrior,Img,normVectors,alpha,beta)
% beta: TV smoothness parameter
% alpha: weights the imapct of the normal vectors
% normVectors: normal vectors pre-calculated by ObjPrior.normalVectors(mri_image)


imgGrad = GraphGrad(ObjPrior,Img);

inprod = sum(normVectors.*imgGrad,2);
Sign = repmat(inprod./abs(inprod + beta),[1,ObjPrior.nS]);

Norm = repmat(sqrt(sum(imgGrad.^2,2)+ beta),[1,ObjPrior.nS]);
imgGrad = imgGrad./Norm - alpha*normVectors .* Sign;

dP = -1* sum(ObjPrior.Wd.*imgGrad,2);


end

% Refernce: Ehrhardt,et al PET Reconstruction with an Anatomical MRI Prior
% using Parallel Level Sets, IEEE TMI, 2016
% modified Eq.(11)