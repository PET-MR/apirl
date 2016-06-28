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
    end
    
    methods
        % Constructurs
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
            
            
            if isempty(varargin{1}.ImageSize)
                error('Image size should be specified');
            end
            
            if varargin{1}.ImageSize(3)>1
                ObjPrior.imCropFactor = 4;
                ObjPrior.chunkSize = 5e5;
            end
            % get fields from user's input
            ObjPrior = Revise(ObjPrior,varargin{1});
            
            [~,ObjPrior.CropedImageSize] = imCrop(ObjPrior);
            
            if ~rem(ObjPrior.sWindowSize,2)
                error('The size of search window should be odd');
            end
            if ~rem(ObjPrior.lWindowSize,2)
                error('The size of local window should be odd');
            end
            
            if ObjPrior.ImageSize(3)>1
                ObjPrior.nS = ObjPrior.sWindowSize^3;
                ObjPrior.nL = ObjPrior.lWindowSize^3;
            else
                ObjPrior.nS = ObjPrior.sWindowSize^2;
                ObjPrior.nL = ObjPrior.lWindowSize^2;
            end
            
            [ObjPrior.SearchWindow, ObjPrior.Wd] = Neighborhood(ObjPrior,ObjPrior.sWindowSize);
            ObjPrior.LocalWindow = Neighborhood(ObjPrior,ObjPrior.lWindowSize);
            
        end
    end
    
    methods (Access = private)
        
        function [N,D] = Neighborhood(ObjPrior,w)
            
            % patch-size
            n = ObjPrior.CropedImageSize(1);
            m = ObjPrior.CropedImageSize(2);
            h = ObjPrior.CropedImageSize(3);
            
            
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
        end
        
        function X = setBoundary1(~,X, n)
            
            idx = X<1;
            X(idx) = 2-X(idx);
            idx = X>n;
            X(idx) = 2*n-X(idx);
            X=X(:);
        end
                
    end
    methods (Access = public)
        
        function [Img,newSize] = imCrop(ObjPrior,Img)
            
            if ObjPrior.imCropFactor==0
                newSize = ObjPrior.ImageSize;
                if nargin==1
                    Img = [];
                end
            else
                ObjPrior.imCropFactor = max(3,ObjPrior.imCropFactor);
                J = floor(ObjPrior.ImageSize(1)/ObjPrior.imCropFactor);
                I = floor(ObjPrior.ImageSize(2)/ObjPrior.imCropFactor);
                % Crop the matrix by imCropFactor in transverse plane
                newSize = [length(J:(ObjPrior.ImageSize(1)-J)), ...
                    length(I:(ObjPrior.ImageSize(1)-I)), ObjPrior.ImageSize(3)];
                
                if nargin==1
                    Img = [];
                else
                    Img = Img(J:(ObjPrior.ImageSize(1)-J),I:(ObjPrior.ImageSize(1)-I),:);
                end
            end
        end
        
        function ImgNew = UndoImCrop(ObjPrior,Img)
            if ObjPrior.imCropFactor==0
                ImgNew = Img;
                return
            end
            ImgNew = zeros(ObjPrior.ImageSize,'single');
            J = floor(ObjPrior.ImageSize(1)/ObjPrior.imCropFactor);
            I = floor(ObjPrior.ImageSize(2)/ObjPrior.imCropFactor);
            
            ImgNew(J:(ObjPrior.ImageSize(1)-J),I:(ObjPrior.ImageSize(1)-I),:) = Img;
        end
        
        function ObjPrior = Revise(ObjPrior,opt)
            % to revise the properties of a given dictionary wihtout
            % re-instantiation
            vfields = fieldnames(opt);
            prop = properties(ObjPrior);
            for i = 1:length(vfields)
                field = vfields{i};
                if sum(strcmpi(prop, field )) > 0
                    ObjPrior.(field) = opt.(field);
                end
            end
        end
       
        function imgGrad = GraphGrad(ObjPrior,Img)           
            imgGrad = (Img(ObjPrior.SearchWindow)-repmat(Img(:),[1,ObjPrior.nS]));
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
        [binCenter,nBins,binwidth] = binCentersOfJointPDF(ObjPrior,f,nBins);
        
        % neighborhood weights
        W = W_GaussianKernel(ObjPrior,Img,KernelSigma);
        W = W_Bowsher(ObjPrior,Img,B);
        W = W_JointEntropy(ObjPrior,Img,sigma);
        
        % derivative of priors
        dP = d_JointEntropy_prior(ObjPrior,imgF,imgA,sigma_f,sigma_a,M,N)
        dP = d_approx_JointEntropy_prior(ObjPrior,ImgF,sigma_f,Wje_imgA);
        dP = d_modified_LOT_prior(ObjPrior,Img,normVectors,alpha,beta);
        dP = d_Kiapio_prior(ObjPrior,Img,normVectors,alpha);
        dP = d_ParallelLevelSets_prior(ObjPrior,Img,normVectors,alpha,beta);
        dP = d_smoothed_TV_prior(ObjPrior,Img,beta);
        dP = d_smoothed_nonlocal_TV_prior(ObjPrior,Img,beta,nl_weights);
        dP = d_Tikhonov_prior(ObjPrior,Img);
        dP = d_nonlocal_Tikhonov_prior(ObjPrior,Img,nl_weights);
        
        % reconstruction
        img = MAP_OSEM(ObjPrior,PET,Prompts,RS, SensImg,opt,Img)
    end
end
