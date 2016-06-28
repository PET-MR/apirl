function [dH,jointPDF,H] = d_JointEntropy_prior(ObjPrior,imgF,imgA,sigma_x,sigma_y,M,N)
% N and M number of histogram bins of imgF and imgA

imgF = ObjPrior.imCrop(single(imgF));
imgA = ObjPrior.imCrop(single(imgA));

Ns = numel(imgF);

[x,M,dx] = ObjPrior.JE_binCentersOfJointPDF(imgF,M);
[y,N,dy] = ObjPrior.JE_binCentersOfJointPDF(imgA,N);


jointPDF = zeros(M,N);
dH = zeros(Ns,M,N);
for i = 1:M
    
    Gxi = ObjPrior.normPDF(x(i),imgF(:),sigma_x);
    for j = 1:N
        Gyj = ObjPrior.normPDF(y(j),imgA(:),sigma_y);
        Gxi_Gyj = Gxi.*Gyj;
        pij  = sum(Gxi_Gyj,1)/(Ns);
        jointPDF(i,j) = pij;
        dpij = 1/Ns*Gxi_Gyj.*((x(i)-imgF(:))./sigma_x.^2);
        dH(:,i,j) = (1 + log(pij+eps)).*dpij;
    end
end
H = jointPDF.*log(jointPDF+eps);
dH = -(dx*dy)*sum(sum(dH,3),2);

end

% Refernce: Somayajula,et al PET Image Reconstruction Using Information
% Theoretic Anatomical Priors, IEEE TMI, 2011
% Eq.(19)