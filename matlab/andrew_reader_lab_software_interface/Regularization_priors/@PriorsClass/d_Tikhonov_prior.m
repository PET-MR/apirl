function dP = d_Tikhonov_prior(ObjPrior,Img)

dP = -2* sum(ObjPrior.Wd.*ObjPrior.GraphGrad(Img),2);

end