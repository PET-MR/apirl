function dP = d_Lange_withnonlocalweights_prior(ObjPrior,Img, nl_weights, params)

% imgGrad = ObjPrior.GraphGrad(Img);
% Norm = repmat(sqrt(sum(ObjPrior.Wd.*imgGrad.^2,2)+ beta),[1,ObjPrior.nS]);
% imgGrad = imgGrad./Norm.*(1 + delta./(delta + Norm));
% dP = -1* sum(ObjPrior.Wd.*imgGrad,2);

imgGrad = ObjPrior.GraphGrad(Img);
imgGrad = imgGrad./(params.LangeDeltaParameter + abs(imgGrad));
dP = -1* sum(nl_weights.*ObjPrior.Wd.*imgGrad,2);

end