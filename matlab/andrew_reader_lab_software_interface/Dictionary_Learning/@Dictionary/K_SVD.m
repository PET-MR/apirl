    
function D = K_SVD(DicObj, P, X,D )

% DicObj.nAtoms

if 0
R = P - D*X;
for k=1:DicObj.nAtoms
    I = find(X(k,:));
    Ri = double(R(:,I) + D(:,k)*X(k,I));
    [U,S,V] = svds(Ri,1,'L');
    % U is normalized
    D(:,k) = U;
    X(k,I) = S*V';
    R(:,I) = Ri - D(:,k)*X(k,I);
    
end
% D = D.*repmat(sign(D(1,:)),size(D,1),1);
elseif 0

    R = P - D*X;
    for j=1:2
    for k=1:DicObj.nAtoms
        I = find(X(k,:));
        Ri = double(R(:,I) + D(:,k)*X(k,I));
        dk = Ri * X(k,I)';
        dk = dk/sqrt(dk'*dk);  % normalize
        D(:,k) = dk;
        X(k,I) = dk'*Ri;
        R(:,I) = Ri - D(:,k)*X(k,I);
    end
    end
elseif 1
    T = 1e-3;
% the first atoms is supposed to be constant
for k=2:DicObj.nAtoms
    % find the exemplar that are using dictionnary basis k
    I = find( abs(X(k,:))>T );
    if ~isempty(I)
        % compute redisual
        if 1
            D0 = D; D0(:,k) = 0;
            E0 = P - D0*X;
            % restrict to element actually using k
            E = E0(:,I);
        else
            S = X(:,I);
            S(k,:) = 0;
            E = P(:,I) - D*S;
        end
        % perform SVD
        [U,S,V] = svd(E);
        D(:,k) = U(:,1);
        X(k,I) = S(1) * V(:,1)';
    end
end
    
end