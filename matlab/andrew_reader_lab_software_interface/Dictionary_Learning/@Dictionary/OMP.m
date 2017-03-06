function X = OMP(DicObj,D,P)

% Sparse coding of the Traning dataset, P, given the
% dictionary D
% input : D - the dictionary
%         P - Traning dataset
% output: X - sparse coefficient matrix.


% DicObj.SC_TargetError = 1e-3;
% DicObj.SC_algorithm = 'OMP';
% DicObj.SC_MP_method = 'Built-In-MATLAB';


[~,nTraningSets]=size(P);
X = zeros(size(D,2),size(P,2));

if strcmpi(DicObj.SC_OMP_method , 'Matlab')
    for k=1:nTraningSets
        [~,~,Coeff,Indx] = wmpalg('OMP',P(:,k),D,'maxerr',{'L2',DicObj.SC_TargetError});
        X(Indx,k)=Coeff';
    end
    
elseif strcmpi(DicObj.SC_OMP_method , 'Script') 
    
    for k=1:nTraningSets
        x=P(:,k);
        residual=x;
        indx = [];
        alpha = [];
        L2NormError = sqrt(sum((abs(residual)).^2));
        j = 0;
        while L2NormError > DicObj.SC_TargetError
            j = j+1;
            proj=D'*residual;
            [~,pos] = max(abs(proj));
            indx(j)= pos;
            alpha = pinv(D(:,indx(1:j)))*x;
            residual = x - D(:,indx(1:j))*alpha;
            L2NormError = sqrt(sum((abs(residual)).^2));
        end;
        if ~isempty(indx)
            X(indx,k)=alpha;
        end
        
    end
elseif strcmpi(DicObj.SC_OMP_method , 'mex')
    D = double(D);
    P = double(P);
    Gramm_matrix = D'*D;
    if 1
%         display('error-constrained OMP')
        X = omp2(D,P,Gramm_matrix,DicObj.SC_TargetError,'gammamode','full');
    else
        display('sparsity-constrained OMP')
        X = omp(D,P,Gramm_matrix,DicObj.SC_TargetError,'gammamode','full');
    end
else
    error('Unknown dictionary update method');
end

