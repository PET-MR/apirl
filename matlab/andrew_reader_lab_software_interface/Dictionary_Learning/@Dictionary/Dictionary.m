classdef Dictionary < handle 
    properties (SetAccess = public)
        
        ImageSize
        PatchSize
        PatchNumel
        PatchPitch     % overlapping factor of patches, if == PatchSize, no overlap
        nPatches       % No. regularly spaced pathces with PatchPitch
        nTrainingSets  % No. training sets selected from randomly chosen Patches
        nAtoms
        is3D
        NdPatches      % Neighborhood of regularly spaced patches
        % sparse coding
        SC_algorithm
        SC_TargetError
        SC_LASSO_lambda
        SC_LASSO_niter
        SC_FOCUSS_lambda
        SC_FOCUSS_p
        SC_FOCUSS_niter
        SC_OMP_method
        % dictionary update
        DU_algorithm
        DU_GD_niter
        DU_AK_SVD_niter
        % dictionary learning
        DL_niter
    end
    
    methods
        function DicObj = Dictionary(varargin)
            DicObj.ImageSize        = [344,344,1];
            DicObj.PatchSize        = 12;
            DicObj.PatchPitch       = 1;
            DicObj.is3D             = 0;
            % sparse coding
            DicObj.SC_algorithm     = 'LASSO'; % OMP, LASSO, FOCUSS
            DicObj.SC_LASSO_niter   = 50;
            DicObj.SC_LASSO_lambda  = 1;
            DicObj.SC_FOCUSS_niter  = 3;
            DicObj.SC_FOCUSS_p      = 1;
            DicObj.SC_FOCUSS_lambda = 0;
            DicObj.SC_TargetError    = 1e-1;
            DicObj.SC_OMP_method    = 'mex'; %'Script','mex'
            % dictionary update
            DicObj.DU_algorithm     = 'MOD'; % AK-SVD, K-SVD, GD
            DicObj.DU_GD_niter      = 100;
            DicObj.DU_AK_SVD_niter  = 3;
            % dictionary learning
            DicObj.DL_niter         = 50;
            
            if isempty(varargin)
            elseif isstruct(varargin{1})
                % get fields from user's input
                vfields = fieldnames(varargin{1});
                prop = properties(DicObj);
                for i = 1:length(vfields)
                    field = vfields{i};
                    if sum(strcmpi(prop, field )) > 0
                        DicObj.(field) = varargin{1}.(field);
                    end
                end
            end
            DicObj.PatchNumel       = DicObj.PatchSize^2;
            DicObj.nAtoms           = 5*DicObj.PatchNumel;

            
            if isempty(DicObj.ImageSize)
                error('Image size should be specified');
            end
            if DicObj.ImageSize(3)>1
                DicObj.is3D = 1;
            end
            
            [DicObj.NdPatches, DicObj.nPatches ] = NeighborhoodRegularGridVoxels(DicObj);
            
            if DicObj.is3D
                DicObj.PatchNumel = DicObj.PatchSize^3;
                DicObj.nPatches   = DicObj.nPatches * DicObj.PatchSize;
                DicObj.nAtoms     = DicObj.nAtoms * DicObj.PatchSize;
            end
            DicObj.nTrainingSets    = min(100*DicObj.PatchNumel,DicObj.nPatches);

        end
    end
    
    methods (Access = private)
        function N = NeighborhoodRandomVoxels(DicObj)
            % returns the Neighborhood of m randomly chosen voxels

            nRows    = DicObj.ImageSize(1);
            nColumns = DicObj.ImageSize(2);
            nSlices  = DicObj.ImageSize(3);
            w = DicObj.PatchSize;
            m = DicObj.nTrainingSets;
            
            if DicObj.is3D
                x = single(floor( rand(1,1,1,m)*(nRows-w) )+1);
                y = single(floor( rand(1,1,1,m)*(nColumns-w) )+1);
                z = single(floor( rand(1,1,1,m)*(nSlices-w) )+1);
                
                [dY,dX,dZ] = meshgrid(0:w-1,0:w-1,0:w-1);
                dX = single(dX);
                dY = single(dY);
                dZ = single(dZ);
                
                dX = repmat(dX, [1 1 1 m]) + repmat(x, [w w w 1]);
                dY = repmat(dY, [1 1 1 m]) + repmat(y, [w w w 1]);
                dZ = repmat(dZ, [1 1 1 m]) + repmat(z, [w w w 1]);
                
                N = dX + (dY-1)*nRows + (dZ-1)*nRows*nColumns; 
            else
                x = floor( rand(1,1,m)*(nRows-w) )+1;
                y = floor( rand(1,1,m)*(nColumns-w) )+1;
                [dY,dX] = meshgrid(0:w-1,0:w-1);
                dX = repmat(dX,[1 1 m]) + repmat(x, [w w 1]);
                dY = repmat(dY,[1 1 m]) + repmat(y, [w w 1]);
                
                N = dX+(dY-1)*nRows;
            end
            
        end
        
        function [N,nPatches] = NeighborhoodRegularGridVoxels(DicObj)
            % returns the Neighborhood of the voxels in a regular grid with
            % pitch of q
            nRows    = DicObj.ImageSize(1);
            nColumns = DicObj.ImageSize(2);
            nSlices  = DicObj.ImageSize(3);
            w = DicObj.PatchSize;
            q = DicObj.PatchPitch;
            
            if DicObj.is3D
                [dY,dX,dZ] = meshgrid(0:w-1,0:w-1,0:w-1);
                dX = single(dX);
                dY = single(dY);
                dZ = single(dZ);
                
                [x,y,z] = meshgrid(1:q:nRows-w/2, 1:q:nColumns-w/2,1:q:nSlices-w/2);
                x = single(x);
                y = single(y);
                z = single(z);
                
                n = size(x(:),1);
                m = size(y(:),1);
                h = size(z(:),1);
                
                dX = repmat(dX,[1 1 1 n]) + repmat( reshape(x(:),[1 1 1 n]), [w w w 1]);
                dY = repmat(dY,[1 1 1 m]) + repmat( reshape(y(:),[1 1 1 m]), [w w w 1]);
                dZ = repmat(dZ,[1 1 1 h]) + repmat( reshape(z(:),[1 1 1 h]), [w w w 1]);
                
                idx = dX>nRows;
                dX(idx) = 2*nRows-dX(idx);
                
                idx = dY>nColumns;
                dY(idx) = 2*nColumns-dY(idx);
                
                idx = dZ>nSlices;
                dZ(idx) = 2*nSlices-dZ(idx);                
                
                N = dX + (dY-1)*nRows + (dZ-1)*nRows*nColumns; 
                
                nPatches = size(N,4);
            else
                [dY,dX] = meshgrid(0:w-1,0:w-1);

                [x,y] = meshgrid(1:q:nRows-w/2, 1:q:nColumns-w/2);
                n = size(x(:),1);
                m = size(y(:),1);
                Xp = repmat(dX,[1 1 n]) + repmat( reshape(x(:),[1 1 n]), [w w 1]);
                Yp = repmat(dY,[1 1 m]) + repmat( reshape(y(:),[1 1 m]), [w w 1]);
                
                idx = Xp>nRows;
                Xp(idx) = 2*nRows-Xp(idx);
                
                idx = Yp>nColumns;
                Yp(idx) = 2*nColumns-Yp(idx);
                N = Xp+(Yp-1)*nRows;
                
                nPatches = size(N,3);
            end
            N = reshape(N, [DicObj.PatchNumel,nPatches]);
        end
    end
    methods (Access = public)
        
        function Revise(DicObj,opt)
            % to revise the properties of a given dictionary wihtout
            % re-instantiation
            vfields = fieldnames(opt);
            prop = properties(DicObj);
            for i = 1:length(vfields)
                field = vfields{i};
                if sum(strcmpi(prop, field )) > 0
                    DicObj.(field) = opt.(field);
                end
            end
        end
        
        function [imgPatch, meanPatch] = RemoveMeanOfPatch(DicObj,imgPatch)
            meanPatch = repmat( mean(imgPatch), [DicObj.PatchNumel,1] );
            imgPatch = imgPatch - meanPatch;
        end
        
        function [imgPatch, meanPatch] = Patch(DicObj,Img)
            % Extracts patches using the Neighborhood NdPatches and calculates the
            % mean of each Patch
            imgPatch = single(Img(DicObj.NdPatches));
            [imgPatch, meanPatch] = RemoveMeanOfPatch(DicObj,imgPatch);
        end
        

        function TS = TrainingSet(DicObj,P)            
            % Exclude zero pathes from the training set
            id = sum(P)==0;
            idN = 1:size(P,2);
            idN(id)=[]; % non-zero patches
            n_nonZeros_Pathces = length(idN);
            
            if DicObj.nTrainingSets < n_nonZeros_Pathces
                sel = randperm(n_nonZeros_Pathces,DicObj.nTrainingSets);
            else
                sel = randperm(n_nonZeros_Pathces);
                DicObj.nTrainingSets = n_nonZeros_Pathces; 
            end
            TS = P(:,idN(sel));

        end
 
        
        function Img = TransPatch(DicObj,imgPatch)

            [W,Img] = deal(zeros(DicObj.ImageSize));
            for i = 1:DicObj.nPatches
                idx = DicObj.NdPatches(:,i);
                Img(idx) = Img(idx) + imgPatch(:,i);
                W(idx)   = W(idx) + 1;
            end
            Img = Img ./ W;
            Img(isinf(Img)) = 0;
        end
        
        function D = initialDictionary(DicObj,P0)
            % P0: mean subtarcted training set (from Patches)

            sel = randperm(DicObj.nTrainingSets);
            sel = sel(1:DicObj.nAtoms);
            D = P0(:,sel);
            D = normalize_dictionary(DicObj,D);
            
        end

        
        function D = normalize_dictionary(DicObj,D)
            
            D = D ./ repmat( sqrt(sum(D.^2)), [DicObj.PatchNumel, 1] );
           
           % workaround for persicons errors on normaizated atoms
           D(isnan(D)) = 1e-5;
           D(isinf(D)) = 1e-5;
           id= abs(sqrt(sum(D.^2))-1) > 1e-1;
           
           D(:,id) = 1e-5./sqrt(sum((1e-5*ones(1,DicObj.PatchNumel)).^2));
           id= abs(sqrt(sum(D.^2))-1) > 1e-1;
           if any(id)
               error('|d_i|_2 = 1 is not met')
           end
        end
        
        function plot_dictionary(DicObj,D,x)
            if nargin==2
                x =[20,20];
            end
            if ~DicObj.is3D
                figure
                plot_dictionary_(DicObj,D, [], x);
            else
                disp('Only 2D Dictionaries')
            end
        end
        
        function X = sparse_coding(DicObj,P,D,X0)
            % X0 = initial estimate of the sparse, or [];
            if nargin==3
                X0 =[];
            end
            if strcmpi(DicObj.SC_algorithm,'OMP')

                X = OMP(DicObj,D,P);
                
            elseif strcmpi(DicObj.SC_algorithm,'LASSO')

                X = LASSO(DicObj,D,P,X0);

            elseif strcmpi(DicObj.SC_algorithm,'FOCUSS')

                X = FOCUSS(DicObj,D,P,X0);
            else
                error('Unknown sparse coding algorithm');
            end
        end
        
        function [D,X] = dictionary_update(DicObj,P,D,X)
            % D = initial estimate of the sparse

            if strcmpi(DicObj.DU_algorithm,'MOD')
                % Method of Optimal Directions
                % D = P * pinv(X);
                D = (P * X')/(X*X'+1e-5*eye(size(X,1)));
                D = normalize_dictionary(DicObj,D);
            elseif strcmpi(DicObj.DU_algorithm,'K-SVD')
                % Standard K-SVD
                R = P - D*X;
                for k = 1:DicObj.nAtoms
                    I = find(X(k,:));
                    Ri = double(R(:,I) + D(:,k)*X(k,I));
                    [U,S,V] = svds(Ri,1,'L');
                    % U is normalized
                    D(:,k) = U;
                    X(k,I) = S*V';
                    R(:,I) = Ri - D(:,k)*X(k,I);
                end
            elseif strcmpi(DicObj.DU_algorithm,'AK-SVD')
                % Approximate K-SVD
                R = P - D*X;
                for j= 1 : DicObj.DU_AK_SVD_niter
                    for k= 1:DicObj.nAtoms
                        I = find(X(k,:));
                        Ri = double(R(:,I) + D(:,k)*X(k,I));
                        dk = Ri * X(k,I)';
                        dk = dk/sqrt(dk'*dk);  % normalize
                        D(:,k) = dk;
                        X(k,I) = dk'*Ri;
                        R(:,I) = Ri - D(:,k)*X(k,I);
                    end
                end
            elseif strcmpi(DicObj.DU_algorithm,'ODL')
                % Online Dictionary Learning
                
            elseif strcmpi(DicObj.DU_algorithm,'GD')

                tau = 1.9./norm(X*X');
                for i = 1 : DicObj.DU_GD_niter
                    D = D + tau*(P-D*X)*X';
                    D = normalize_dictionary(DicObj,D);
                end
            else
                error('Unknown dictionary learning algorithm');
            end

            D = normalize_dictionary(DicObj,D);
        end
        
        function [D,X] = DictionaryLearning(DicObj,TS,D,X)
            if nargin == 2 
                %if TS is full imgPatch
                TS = DicObj.TrainingSet(TS);
                D = DicObj.initialDictionary(TS);
                X = [];
            end
            if nargin == 3
                X = [];
            end
            for i = 1:DicObj.DL_niter
                X = sparse_coding(DicObj,TS,D,X);
                D = dictionary_update(DicObj,TS,D,X);
            end
        end

        function [dD,Img_hat] = derivative_DL(DicObj,Img,D)

            % 2 *PhiT*Phi*v - 2*PhiT*D*X
            [P,imgPatchMean] = DicObj.Patch(Img);
            
            X = DicObj.sparse_coding(P,D);
            
            imgPatch_hat = DicObj.RemoveMeanOfPatch(D*X) + imgPatchMean;
            Img_hat = DicObj.TransPatch(imgPatch_hat);
            
            dD = 2* (Img - Img_hat);
        end
        

    end
    
    
end




