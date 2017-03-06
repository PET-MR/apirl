function X = LASSO(DicObj,D,P,X)



% DicObj.SC_TargetError = 1e-2;
% DicObj.SC_LASSO_niter = 300;
% DicObj.SC_LASSO_lambda =  1;

[~,nTrainiSets] = size(P);
nAtoms = size(D,2);

if isempty(X)
    % X = pinv(D)*P;   % initial values
    X = zeros(nAtoms,nTrainiSets);
end
lambda = zeros(nTrainiSets,1) + DicObj.SC_LASSO_lambda; 

Norm =@(x) sqrt(sum(x(:).^2));
mu = 1.9/Norm(D)^2;% step size

L2NormError =  Norm(P-D*X);

k1 = 0;
while (L2NormError > DicObj.SC_TargetError)&& (k1< DicObj.SC_LASSO_niter)
    k1 = k1 + 1;
    X = X + mu*D'*(P-D*X);
    X = max( 1 - repmat(lambda(:)', [nAtoms 1]) ./ max(abs(X),1e-10), 0 ) .* X;
    
    % adaptive lambda selection
    if mod(k1,5)==0
        lambda = lambda * DicObj.SC_TargetError ./ sqrt( sum( (P-D*X).^2 ) )';
    end
    L2NormError = Norm(P-D*X);
end


end