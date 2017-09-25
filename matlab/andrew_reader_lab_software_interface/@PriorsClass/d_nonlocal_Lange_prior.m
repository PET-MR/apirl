function dP = d_nonlocal_Lange_prior(ObjPrior,Img,nl_weights, params)

%imgGrad = ObjPrior.GraphGrad(Img);
%Norm = repmat(sqrt(sum(ObjPrior.Wd.*imgGrad.^2,2)+ beta),[1,ObjPrior.nS]);
%imgGrad = imgGrad./Norm.*(1 + delta./(delta + Norm));
%dP = -1* sum(ObjPrior.Wd.*imgGrad,2);

Norm = MagnitudGraphGradWithSpatialWeightAndSimilarity(ObjPrior, Img, nl_weights, params);
% dP = -1*ObjPrior.GraphGradWithSpatialWeightAndSimilarity(Img, nl_weights, params)./Norm.*(1+params.LangeDeltaParameter./(params.LangeDeltaParameter+Norm)); %sum(nl_weights.*ObjPrior.Wd.*ObjPrior.GraphGrad(Img),2);
% dP(Norm==0) = 0;
dP = -1*ObjPrior.GraphGradWithSpatialWeightAndSimilarity(Img, nl_weights, params)./(params.LangeDeltaParameter+Norm); %sum(nl_weights.*ObjPrior.Wd.*ObjPrior.GraphGrad(Img),2);
%dP(Norm==0) = 0;
end