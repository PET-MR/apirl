function n = L2Norm(ObjPrior,G)

n = sqrt(sum(abs(G).^2,2));

end