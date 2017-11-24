classdef PriorsClass < handle
    properties (SetAccess = public)
        
        ImageSize
        CropedImageSize
        imCropFactor
        sWindowSize % search window size
        lWindowSize % local window size (local neighborhood)
        SearchWindow
        LocalWindow
        PriorImplementation % two options: matlab(default), mex-cuda
        PriorType % type of prior: 'Lange';%'Quadratic'
        SimilarityKernel % type of similarity kernel: 'local'(no-similarity), 'Bowsher' 'JointBurgEntropy'
        PriorImage % Image for the similarity kernel
        dPhandle % handle of the derivative of the Prior function
        dWhandle % handle of the function that computes the similarity weights
        dPNLhandle % Handle for priors with non local similarity kernels
        PriorParams % parameters for the prior, is a cell array witha s amny elements as parameters has the prior. (e.g. quadratic [], tv smoothparametr, lange: delta and smoothparameter
        PreCompWeights % Pre computed weight for the similarity kernels, if empty theya re computed
        chunkSize
        Wd
        nS
        nL
        is3D
    end
    
    methods
        
        function ObjPrior = PriorsClass(varargin)
            ObjPrior.ImageSize = [344,344,1];
            ObjPrior.CropedImageSize = [];
            ObjPrior.sWindowSize = 7;
            ObjPrior.lWindowSize = 3;
            ObjPrior.SearchWindow = [];
            ObjPrior.LocalWindow =[];
            ObjPrior.Wd = [];
            ObjPrior.nS = [];
            ObjPrior.nL = [];
            ObjPrior.chunkSize = 1;
            ObjPrior.is3D = 0;
            ObjPrior.imCropFactor = 3;
            ObjPrior.chunkSize = 5e6;
            ObjPrior.PriorImplementation = 'matlab'; % by default matlab
            ObjPrior.SimilarityKernel = 'local';
            ObjPrior.PreCompWeights = [];
            ObjPrior.PriorImage = [];
            if isempty(varargin{1}.ImageSize)
                error('Image size should be specified');
            end

            % get fields from user's input
            ObjPrior = getFiledsFromUsersOpt(ObjPrior,varargin{1});
            if length(varargin{1}.ImageSize)==3 && varargin{1}.ImageSize(3)>1
                ObjPrior.is3D = 1;
            end
            % Sometimes as input parameter, we can have PetPriorType and
            % PetSimilarityKernel, or the Mr version:
            if isfield(varargin{1}, 'PetPriorType')
                ObjPrior.PriorType = varargin{1}.PetPriorType;
            elseif isfield(varargin{1}, 'MrPriorType')
                ObjPrior.PriorType = varargin{1}.MrPriorType;
            end
            if isfield(varargin{1}, 'PetSimilarityKernel')
                ObjPrior.SimilarityKernel = varargin{1}.PetSimilarityKernel;
            elseif isfield(varargin{1}, 'MrSimilarityKernel')
                ObjPrior.SimilarityKernel = varargin{1}.MrSimilarityKernel;
            end
            InitializePriors(ObjPrior);
        end
    end
    
    methods (Access = private)
        
        function [N,D] = Neighborhood(ObjPrior,w)
            
            % patch-size
            m = ObjPrior.CropedImageSize(1);
            n = ObjPrior.CropedImageSize(2);
            if ObjPrior.is3D
                h = ObjPrior.CropedImageSize(3);
            else
                h = 1;
            end
            
            
            wlen = 2*floor(w/2); % length of neighborhood window
            widx = -wlen/2:wlen/2;
            xidx = widx; yidx = widx;
            
            if h==1
                zidx = 0;
                nN = w*w;
            else
                zidx = widx;
                nN = w*w*w;
            end
            
            % image grid
            [Y, X, Z] = ndgrid(1:m,1:n,1:h); % ndgrid behaves different to meshgrid, first parameter are columns and not x
            Y = single(Y);
            X = single(X);
            Z = single(Z);
            % index and distance
            
            N = zeros(n*m*h, nN,'single');
            D = N;
            l = 1;
            for x = xidx
                Xnew = ObjPrior.setBoundary1(X + x, n);
                for y = yidx
                    Ynew = ObjPrior.setBoundary1(Y + y, m);
                    for z = zidx
                        Znew = ObjPrior.setBoundary1(Z + z, h);
                        N(:,l) = Ynew + (Xnew-1).*m + (Znew-1)*n*m; % col-wise
                        D(:,l) = sqrt(x^2+y^2+z^2);
                        l = l + 1;
                    end
                end
            end
            
            D = 1./D;
            D(isinf(D))= 0;
            D = D./repmat(sum(D,2),[1,nN]);
            
            %             for local first-order neighborhood 3x3x3
            if ObjPrior.sWindowSize ==3 && ObjPrior.lWindowSize ==1 && size(N,2)>1
                if h>1 %3D
                    nearsetVoxels = [5,11,13,14,15,17,23];
                    ObjPrior.nS = 7;
                else
                    nearsetVoxels = [2,4,5,6,8];
                    ObjPrior.nS = 5;
                end
                N = N(:,nearsetVoxels);
                D = D(:,nearsetVoxels);
            end
        end
        
        function X = setBoundary1(~,X, n)
            
            idx = X<1;
            X(idx) = 2-X(idx);
            idx = X>n;
            X(idx) = 2*n-X(idx);
            X=X(:);
        end
        
        %         function ObjPrior = getFiledsFromUsersOpt(ObjPrior,opt)
        %             vfields = fieldnames(opt);
        %             prop = properties(ObjPrior);
        %             for i = 1:length(vfields)
        %                 field = vfields{i};
        %                 if sum(strcmpi(prop, field )) > 0
        %                     ObjPrior.(field) = opt.(field);
        %                 end
        %             end
        %         end
    end
    methods (Access = public)
        
        function InitializePriors(ObjPrior)
            [~,ObjPrior.CropedImageSize] = imCrop(ObjPrior);
            
            if ~rem(ObjPrior.sWindowSize,2)
                error('The size of search window should be odd');
            end
            if ~rem(ObjPrior.lWindowSize,2)
                error('The size of local window should be odd');
            end
            
            if ObjPrior.is3D
                ObjPrior.nS = ObjPrior.sWindowSize^3;
                ObjPrior.nL = ObjPrior.lWindowSize^3;
            else
                ObjPrior.nS = ObjPrior.sWindowSize^2;
                ObjPrior.nL = ObjPrior.lWindowSize^2;
            end
            % Get the prior function into a handle:
            if strcmpi(ObjPrior.SimilarityKernel,'local') % local
                switch lower(ObjPrior.PriorType)
                    case 'quadratic'
                        ObjPrior.dPhandle = @d_Tikhonov_prior;
                    case 'tikhonov'
                        ObjPrior.dPhandle = @d_Tikhonov_prior;
                    case 'lange'
                        ObjPrior.dPhandle = @d_Lange_prior;
                    case 'huber'
                        ObjPrior.dPhandle = @d_Huber_prior;
                    case 'tv'
                        ObjPrior.dPhandle = @d_smoothed_TV_prior;
                    case 'approx_joint_entropy'
                        ObjPrior.dPhandle = @d_approx_JointEntropy_prior;
                    case 'joint_entropy'
                        ObjPrior.dPhandle = @d_JointEntropy_prior;
                    case 'kiapio'
                        ObjPrior.dPhandle = @d_Kiapio_prior;
                    case 'modified_lot'
                        ObjPrior.dPhandle = @d_modified_LOT_prior;
                    case 'parallel_level_set'
                        ObjPrior.dPhandle = @d_ParallelLevelSets_prior;
                end
            else % non-local
                switch lower(ObjPrior.PriorType)
                    case 'tikhonov'
                        ObjPrior.dPNLhandle = @d_nonlocal_Tikhonov_prior;
                    case 'quadratic'
                        ObjPrior.dPNLhandle = @d_nonlocal_Tikhonov_prior;
                    case 'lange'
                        ObjPrior.dPNLhandle = @d_Lange_withnonlocalweights_prior;
                    case 'nonlocal_lange'
                        ObjPrior.dPNLhandle = @d_nonlocal_Lange_prior;
                    case 'huber'
                        ObjPrior.dPNLhandle = @d_nonlocal_Huber_prior;
                    case 'tv'
                        ObjPrior.dPNLhandle = @d_smoothed_TV_prior;
                end
                % Similarity functions:
                if strcmpi(ObjPrior.SimilarityKernel,'nl_self_similarity')
                    ObjPrior.dWhandle = @W_GaussianKernel;
                elseif strcmpi(ObjPrior.SimilarityKernel,'nl_side_similarity') || strcmpi(ObjPrior.SimilarityKernel,'bowsher')
                    ObjPrior.dWhandle = @W_Bowsher;
                elseif strcmpi(ObjPrior.SimilarityKernel,'nl_joint_similarity')
                    ObjPrior.dWhandle = @W_JointGaussianKernel;
                end
            end
            % The search window is only needed for the matlab version:
            if strcmp(ObjPrior.PriorImplementation, 'matlab')
                [ObjPrior.SearchWindow, ObjPrior.Wd] = Neighborhood(ObjPrior,ObjPrior.sWindowSize);
                ObjPrior.LocalWindow = Neighborhood(ObjPrior,ObjPrior.lWindowSize);
            end
        end
        
        function [Img,newSize] = imCrop(ObjPrior,Img)
            % 0, [0 0 0]
            % 2,3,...
            % [2, 2, 0]
            if all(ObjPrior.imCropFactor==0)
                newSize = ObjPrior.ImageSize;
                if nargin==1
                    Img = [];
                end
            else
                if length(ObjPrior.imCropFactor)== 1
                    if ObjPrior.is3D
                        ObjPrior.imCropFactor = ObjPrior.imCropFactor*[1 1 0];
                    else
                        ObjPrior.imCropFactor = ObjPrior.imCropFactor*[1 1];
                    end
                end
                
                J = 0;
                if ObjPrior.imCropFactor(1)
                    ObjPrior.imCropFactor(1) = max(2.5, ObjPrior.imCropFactor(1));
                    J = floor(ObjPrior.ImageSize(1)/ObjPrior.imCropFactor(1));
                end
                
                I = 0;
                if ObjPrior.imCropFactor(2)
                    ObjPrior.imCropFactor(2) = max(2.5, ObjPrior.imCropFactor(2));
                    I = floor(ObjPrior.ImageSize(2)/ObjPrior.imCropFactor(2));
                end
                
                if ObjPrior.is3D
                    K = 0;
                    if ObjPrior.imCropFactor(3)
                        ObjPrior.imCropFactor(3) = max(2.5, ObjPrior.imCropFactor(3));
                        K = floor(ObjPrior.ImageSize(3)/ObjPrior.imCropFactor(3));
                    end
                    newSize = [length((J:(ObjPrior.ImageSize(1)-J-1))+1),length((I:(ObjPrior.ImageSize(2)-I-1))+1),length((K:(ObjPrior.ImageSize(3)-K-1))+1)];
                else
                    newSize = [length((J:(ObjPrior.ImageSize(1)-J-1))+1),length((I:(ObjPrior.ImageSize(2)-I-1))+1)];
                    if length(ObjPrior.ImageSize)==3
                        newSize = [ newSize ,1];
                    end
                end
                if nargin==1
                    Img = [];
                else
                    if ObjPrior.is3D
                        Img = Img((J:(ObjPrior.ImageSize(1)-J-1))+1,(I:(ObjPrior.ImageSize(2)-I-1))+1,(K:(ObjPrior.ImageSize(3)-K-1))+1);
                    else
                        Img = Img((J:(ObjPrior.ImageSize(1)-J-1))+1,(I:(ObjPrior.ImageSize(2)-I-1))+1);
                    end
                end
            end
        end
        
        function ImgNew = UndoImCrop(ObjPrior,Img)
            if all(ObjPrior.imCropFactor==0)
                ImgNew = Img;
                return
            end
            ImgNew = zeros(ObjPrior.ImageSize,'single');
            
            S = (ObjPrior.ImageSize - ObjPrior.CropedImageSize)/2;
            J = S(1); I = S(2);
            if ObjPrior.is3D
                K = S(3);
                ImgNew((J:(ObjPrior.ImageSize(1)-S(1)-1))+1,(I:(ObjPrior.ImageSize(2)-I-1))+1,(K:(ObjPrior.ImageSize(3)-K-1))+1) = Img;
            else
                ImgNew((J:(ObjPrior.ImageSize(1)-S(1)-1))+1,(I:(ObjPrior.ImageSize(2)-I-1))+1) = Img;
            end
        end
        
        function ObjPrior = RevisePrior(ObjPrior,opt)
            % to revise the properties of the object
            
            ObjPrior = getFiledsFromUsersOpt(ObjPrior,opt);
            if isfield(opt,'sWindowSize') || isfield(opt,'lWindowSize') ...
                    || isfield(opt,'imCropFactor') || isfield(opt,'ImageSize')
                InitializePriors(ObjPrior);
            end
        end
        
        function imgGrad = GraphGrad(ObjPrior,Img)
            imgGrad = (Img(ObjPrior.SearchWindow)-repmat(Img(:),[1,ObjPrior.nS]));
        end
        
        function imgGrad = GraphGradWithSpatialWeight(ObjPrior,Img)
            if strcmp(ObjPrior.PriorImplementation, 'matlab')
                imgGrad = sum(ObjPrior.Wd.*ObjPrior.GraphGrad(Img),2);
            elseif strcmp(ObjPrior.PriorImplementation, 'mex-cuda')
                enableSpatialWeight = 1;
                typeOfLocalDifferences = 'LinearSum';
                imgGrad = ObjPrior.GpuGraphGrad(Img, ObjPrior.sWindowSize, ObjPrior.sWindowSize, ObjPrior.sWindowSize, enableSpatialWeight, typeOfLocalDifferences); % Possible values: 'LinearSum', 'Magnitud'
            end
        end
        
        function imgGrad = GraphGradWithSpatialWeightAndSimilarity(ObjPrior,Img, nl_weights, opt)
            if strcmp(ObjPrior.PriorImplementation, 'matlab')
                imgGrad = sum(nl_weights.*ObjPrior.Wd.*ObjPrior.GraphGrad(Img),2);
            elseif strcmp(ObjPrior.PriorImplementation, 'mex-cuda')
                enableSpatialWeight = 1;
                typeOfLocalDifferences = 'LinearSum';
                % So far this only works with bowsher:
                imgGrad = ObjPrior.GpuGraphGradWithSimilarity(Img, ObjPrior.imCrop(single(ObjPrior.PriorImage)), ObjPrior.sWindowSize, ObjPrior.sWindowSize, ObjPrior.sWindowSize, enableSpatialWeight, typeOfLocalDifferences, opt); % Possible values: 'LinearSumWithBowsher', 'MagnitudWithBowsher'
            end
        end
        
        function magGrad = MagnitudGraphGradWithSpatialWeight(ObjPrior, Img, smooth) % Parameter needed sometimes if the magnitud is used in the denominator.
            if nargin == 2
                smooth = 0;
            end
            if strcmp(ObjPrior.PriorImplementation, 'matlab')
                imgGrad = ObjPrior.GraphGrad(Img);
                magGrad = sqrt(sum(imgGrad.^2,2)+ smooth);
            elseif strcmp(ObjPrior.PriorImplementation, 'mex-cuda')
                enableSpatialWeight = 0; % For TV, in my opinion needs to be included, but it wasn't in the matlab implementation.
                typeOfLocalDifferences = 'Magnitud';
                magGrad = ObjPrior.GpuGraphGrad(Img, ObjPrior.sWindowSize, ObjPrior.sWindowSize, ObjPrior.sWindowSize, enableSpatialWeight, typeOfLocalDifferences); % Possible values: 'LinearSum', 'Magnitud'
            end
        end      
        
        function magGrad = MagnitudGraphGradWithSpatialWeightAndSimilarity(ObjPrior, Img, nl_weights, opt) % Parameter needed sometimes if the magnitud is used in the denominator.
            if nargin == 2
                smooth = 0;
            end
            if strcmp(ObjPrior.PriorImplementation, 'matlab')
                imgGrad = ObjPrior.GraphGrad(Img);
                magGrad = sqrt(sum(nl_weights.*ObjPrior.Wd.*imgGrad.^2,2)+ opt.TVsmoothingParameter);
            elseif strcmp(ObjPrior.PriorImplementation, 'mex-cuda')
                enableSpatialWeight = 1; % For TV, in my opinion needs to be included, but it wasn't in the matlab implementation.
                typeOfLocalDifferences = 'Magnitud';
                magGrad = ObjPrior.GpuGraphGradWithSimilarity(Img, ObjPrior.imCrop(single(ObjPrior.PriorImage)), ObjPrior.sWindowSize, ObjPrior.sWindowSize, ObjPrior.sWindowSize, enableSpatialWeight, typeOfLocalDifferences, opt); % Possible values: 'LinearSum', 'Magnitud'
            end
        end 
        
        function imgGrad = GpuGraphGrad(ObjPrior, image, Kx, Ky, Kz, enableSpatialWeight, typeOfLocalDifferences)
            % typeOfLocalDifferences: 
            % 1:Sum of linear differences,
            % 2:Magnitud of linear differences(sqrt(square_sums))
            switch typeOfLocalDifferences
                case 'LinearSum'
                    imgGrad = mexGPUGradient(single(image), Kx, Ky, Kz, enableSpatialWeight, 1); % This function is not part of the prior class because is a mex file.
                case 'Magnitud'
                    imgGrad = mexGPUGradient(single(image), Kx, Ky, Kz, enableSpatialWeight, 2); % This function is not part of the prior class because is a mex file.
            end
        end
        
        function imgGrad = GpuGraphGradWithSimilarity(ObjPrior, image, similarityImage, Kx, Ky, Kz, enableSpatialWeight, typeOfLocalDifferences, opt)
            % typeOfLocalDifferences: 
            % 1:Sum of linear differences,
            % 2:Magnitud of linear differences(sqrt(square_sums))
            switch typeOfLocalDifferences
                case 'LinearSum'
                    imgGrad = mexGPUGradientWithSimilarityKernel(single(image), single(similarityImage), Kx, Ky, Kz, enableSpatialWeight, 1, opt.BowsherB); % This function is not part of the prior class because is a mex file.
                case 'Magnitud'
                    imgGrad = mexGPUGradientWithSimilarityKernel(single(image), single(similarityImage), Kx, Ky, Kz, enableSpatialWeight, 2, opt.BowsherB);
            end
        end
        
        function imgGrad = GraphGradCrop(ObjPrior,Img)
            Img = imCrop(ObjPrior,single(Img));
            imgGrad = (Img(ObjPrior.SearchWindow)-repmat(Img(:),[1,ObjPrior.nS]));
        end
        
        function imgDiv = GraphDivCrop(ObjPrior,Img)
            Img = imCrop(ObjPrior,single(Img));
            imgDiv = (Img(ObjPrior.SearchWindow)+repmat(Img(:),[1,ObjPrior.nS]));
        end
        function dP = TransGraphGradUndoCrop(ObjPrior,imgGrad)
            dP = -2* sum(ObjPrior.Wd.*imgGrad,2);
            dP = reshape(dP,ObjPrior.CropedImageSize);
            dP = UndoImCrop(ObjPrior,dP);
        end
        
        % Computes the gradient of an image in gpu
        %function [SumGrad] = gpuGradient(ObjPrior,Img);
            
        function imgGradW = TV_weights(ObjPrior,imgGrad,beta)
            Norm = repmat(sqrt(sum(abs(imgGrad).^2,2)+ beta.^2),[1,ObjPrior.nS]);
            imgGradW = imgGrad./Norm/2;
        end
        
        function dP = dPrior(ObjPrior,Img,opt)
            % opt.prior: 'tikhonov', 'tv', 'approx_joint_entropy',
            % 'joint_entropy','kiapio', 'modified_lot', 'parallel_level_set'
            
            %-------- Tikhonov or TV ----------
            % opt.weight_method:
            % 'local','nl_self_similarity','nl_side_similarity','nl_joint_similarity'
            % opt.nl_weights: Gaussian or Bowsher weights calculated from one/more anatomical images
            % opt.sigma_ker
            % opt.n_modalities: number of anatomical images +1 for 'nl_joint_similarity' method
            % opt.beta
            %-------- approx_joint_entropy ----------
            % opt.sigma_x
            % opt.je_weights: weights calculated from one/more anatomical images
            %-------- joint_entropy ----------
            % opt.imgA
            % opt.sigma_x
            % opt.sigma_y
            % opt.M
            % opt.N
            %-------- Kiapio_prior, modified_LOT_prior,ParallelLevelSets_prior  ----------
            % opt.normVectors
            % opt.alpha
            % opt.beta
            
            Img = imCrop(ObjPrior,single(Img));
            
            % Get prior:
            if strcmpi(ObjPrior.SimilarityKernel,'local') % local
                dP = ObjPrior.dPhandle(ObjPrior, Img, opt);
                dP = dP./ObjPrior.nS; % to use the same regualrization as non-local methods
            else % non-local
                if strcmp(ObjPrior.PriorImplementation, 'matlab')
                    if isempty(ObjPrior.PreCompWeights)
                        W = ObjPrior.dWhandle(ObjPrior, ObjPrior.imCrop(ObjPrior.PriorImage), opt);
                        W = W./repmat(sum(W,2),[1,ObjPrior.nS]);
                        if strcmpi(ObjPrior.SimilarityKernel,'bowsher')
                            % bowsher doesn't change with iterations, so
                            % once computed, reuse it
                            ObjPrior.PreCompWeights = W;
                        end
                    else
                        W = ObjPrior.PreCompWeights;
                    end
                else
                    W = [];
                end
                dP = ObjPrior.dPNLhandle(ObjPrior, Img,W,opt);
            end
            dP = reshape(dP,ObjPrior.CropedImageSize);
            dP = UndoImCrop(ObjPrior,dP);
        end
        
        % helper functions
        p = jointPDF(ObjPrior, x,y,imgF,imgA,sigma_f,sigma_a);
        x = normPDF(ObjPrior,x,y,sigma);
        N = normalVectors(ObjPrior,Img);
        n = L2Norm(ObjPrior,imgGrad);
        %[binCenter,nBins,binwidth,binBoundery,fB] = binCentersOfJointPDF(ObjPrior,f,nBins,maxbin,d);
        plot_histogram(ObjPrior,f,M);
        
        % neighborhood weights
        W = W_GaussianKernel(ObjPrior,Img,KernelSigma);
        W = W_Bowsher(ObjPrior,Img,B);
        W = W_JointEntropy(ObjPrior,Img,sigma);
        %W = W_PearsonCorrelation(ObjPrior,Img)
        % derivative of priors
        %dP = d_JointEntropy_prior(ObjPrior,imgF,imgA,sigma_f,sigma_a,M,N)
        dP = d_approx_JointEntropy_prior(ObjPrior,ImgF,sigma_f,Wje_imgA);
        %dP = d_modified_LOT_prior(ObjPrior,Img,normVectors,alpha,beta);
        dP = d_Kiapio_prior(ObjPrior,Img,normVectors,alpha);
        %dP = d_ParallelLevelSets_prior(ObjPrior,Img,normVectors,alpha,beta);
        dP = d_smoothed_TV_prior(ObjPrior,Img,beta);
        dP = d_smoothed_nonlocal_TV_prior(ObjPrior,Img,beta,nl_weights);
        dP = d_Tikhonov_prior(ObjPrior,Img, params);
        dP = d_nonlocal_Tikhonov_prior(ObjPrior,Img,nl_weights,params);
        
        % reconstruction
        img = MAP_OSEM(ObjPrior,PET,Prompts,RS, SensImg,opt,Img)
        
        function display(ObjPrior)
            disp(ObjPrior)
            methods(ObjPrior)
        end
    end
end
