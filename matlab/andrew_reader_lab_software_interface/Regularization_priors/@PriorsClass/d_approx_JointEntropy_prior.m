function dP = d_approx_JointEntropy_prior(ObjPrior,Img,sigma_f,Wje_imgA)
% sigma_f
% Wje_imgA: pre-calculated weights obtained from ObjPrior.W_JointEntropy(imgA,opt.sigma_a)
 

W = ObjPrior.W_JointEntropy(Img,sigma_f).*Wje_imgA;

W = W./repmat(sum(W,2),[1,ObjPrior.nS]);
W = W./(sigma_f^2)/prod(ObjPrior.ImageSize);

W = W./repmat(sum(W,2),[1,ObjPrior.nS]);

imgGrad = ObjPrior.GraphGrad(Img).*W; 
dP = -2* sum(ObjPrior.Wd .*imgGrad,2);%

end


% Refernce: Somayajula,et al PET Image Reconstruction Using Information
% Theoretic Anatomical Priors, IEEE TMI, 2011
% Eq.(37)
