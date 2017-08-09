function dP = d_Tikhonov_prior(ObjPrior, Img, params) % params should be empty, is only kept to 

dP = -2*ObjPrior.GraphGradWithSpatialWeight(Img);

end