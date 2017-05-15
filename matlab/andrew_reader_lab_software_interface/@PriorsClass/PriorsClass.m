classdef PriorsClass < handle
    properties (SetAccess = public)
        
        ImageSize
        CropedImageSize
        imCropFactor
        sWindowSize % search window size
        lWindowSize % local window size (local neighborhood)
        SearchWindow
        LocalWindow
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
                
            if isempty(varargin{1}.ImageSize)
                error('Image size should be specified');
            end

            % get fields from user's input
            ObjPrior = getFiledsFromUsersOpt(ObjPrior,varargin{1});
            
            if length(varargin{1}.ImageSize)==3 && varargin{1}.ImageSize(3)>1
                ObjPrior.is3D = 1;
            end
            InitializePriors(ObjPrior);
        end
    end
    
    methods (Access = private)
        
        function [N,D] = Neighborhood(ObjPrior,w)
            
            % patch-size
            n = ObjPrior.CropedImageSize(1);
            m = ObjPrior.CropedImageSize(2);
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
            [X, Y, Z] = ndgrid(1:n,1:m,1:h);
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
                        N(:,l) = Xnew + (Ynew-1).*n + (Znew-1)*n*m;
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
            
            [ObjPrior.SearchWindow, ObjPrior.Wd] = Neighborhood(ObjPrior,ObjPrior.sWindowSize);
            ObjPrior.LocalWindow = Neighborhood(ObjPrior,ObjPrior.lWindowSize);
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
            
            switch lower(opt.prior)
                case 'tikhonov'
                    if strcmpi(opt.weight_method,'local') % local
                        dP = d_Tikhonov_prior(ObjPrior,Img);
                        dP = dP./ObjPrior.nS; % to use the same regualrization as non-local methods
                    else % non-local
                        if strcmpi(opt.weight_method,'nl_self_similarity')
                            W = W_GaussianKernel(ObjPrior,Img,opt.sigma_ker);
                        elseif strcmpi(opt.weight_method,'nl_side_similarity') || strcmpi(opt.weight_method,'bowsher')
                            W = opt.nl_weights;
                        elseif strcmpi(opt.weight_method,'nl_joint_similarity')
                            W = (W_GaussianKernel(ObjPrior,Img,opt.sigma_ker).*opt.nl_weights).^1/(opt.n_modalities);
                        end
                        W = W./repmat(sum(W,2),[1,ObjPrior.nS]);
                        dP = d_nonlocal_Tikhonov_prior(ObjPrior,Img,W);
                    end
                case 'tv'
                    if strcmpi(opt.weight_method,'local') % local
                        dP = d_smoothed_TV_prior(ObjPrior,Img,opt.beta);
                    else % non-local
                        if strcmpi(opt.weight_method,'nl_self_similarity')
                            W = W_GaussianKernel(ObjPrior,Img,opt.sigma_ker);
                        elseif strcmpi(opt.weight_method,'nl_side_similarity') || strcmpi(opt.weight_method,'bowsher')
                            W = opt.nl_weights;
                        elseif strcmpi(opt.weight_method,'nl_joint_similarity')
                            W = (W_GaussianKernel(ObjPrior,Img,opt.sigma_ker).*opt.nl_weights).^1/(opt.n_modalities);
                        end
                        W = W./repmat(sum(W,2),[1,ObjPrior.nS]);
                        dP = d_smoothed_nonlocal_TV_prior(ObjPrior,Img,opt.beta,W);
                    end
                case 'approx_joint_entropy'
                    dP = d_approx_JointEntropy_prior(ObjPrior,Img,opt.sigma_f,opt.je_weights);
                case 'joint_entropy'
                    dP = d_JointEntropy_prior(ObjPrior,Img,opt.imgA,opt.sigma_f,opt.sigma_a,opt.M,opt.N);
                case 'kiapio'
                    dP = d_Kiapio_prior(ObjPrior,Img,opt.normVectors,opt.alpha);
                case 'modified_lot'
                    dP = d_modified_LOT_prior(ObjPrior,Img,opt.normVectors,opt.alpha,opt.beta);
                case 'parallel_level_set'
                    dP = d_ParallelLevelSets_prior(ObjPrior,Img,opt.normVectors,opt.alpha,opt.beta);
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
        dP = d_Tikhonov_prior(ObjPrior,Img);
        dP = d_nonlocal_Tikhonov_prior(ObjPrior,Img,nl_weights);
        
        % reconstruction
        img = MAP_OSEM(ObjPrior,PET,Prompts,RS, SensImg,opt,Img)
        
        function display(ObjPrior)
            disp(ObjPrior)
            methods(ObjPrior)
        end
    end
end
