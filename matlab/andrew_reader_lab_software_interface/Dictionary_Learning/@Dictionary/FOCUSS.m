function X = FOCUSS(DicObj,D,P,X)

% DicObj.SC_FOCUSS_niter  = 3;
% DicObj.SC_FOCUSS_p      = 1;
% DicObj.SC_FOCUSS_lambda = 0;
% DicObj.SC_TargetError    = 1e-3;


[N,nTrainiSets] = size(P);
nAtoms = size(D,2);

if isempty(X)
    % X = pinv(D)*P;   % initial values
    X = ones(nAtoms,nTrainiSets);
end

for ii = 1:nTrainiSets

    x = X(:,ii);
    x0 = x;
    p = P(:,ii);
    for i = 1:DicObj.SC_FOCUSS_niter
        Qdiagonal = abs(x).^(1-DicObj.SC_FOCUSS_p/2);
        F = D.*(ones(N,1)*(Qdiagonal'));  
        if (DicObj.SC_FOCUSS_lambda > 0)  % Regularized FOCUSS
            q = F'*( ((F*F'+DicObj.SC_FOCUSS_lambda*eye(N)) \ p) );
        else  % original FOCUSS
            q = pinv(F) * p;
        end
        x = Qdiagonal.*q;
        change = norm(x-x0);
        if (change < DicObj.SC_TargetError); break; end;
        x0 = x;
    end
    X(:,ii) = x;
end