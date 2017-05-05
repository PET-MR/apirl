function [x,S] = SENSE_PCG(ObjMRI,arg,initialEstimate,RHS)

if nargin==1, arg.save = 0; end

if nargin>=3
    img = initialEstimate;
else
    img = zeros(ObjMRI.nkSamples,'single');
end


% Default values
opt.SENSE_niter = 10;
opt.ReconUnderSampledkSpace = 0;
opt.MrRegularizationParameter = 0;
opt.MrPriorType = 'Quadratic'; % 'sTV', 'Gaussian', 'JBE'
opt.MrPreCompWeights = 1;
opt.TVsmoothingParameter = 0.02;
opt.display = 0;
opt.save = 0;
opt.JBE_sigma_p = 10; % prior image
opt.JBE_sigma_i = 10; % initial image
opt.MrSigma = 30;
opt.PriorMrImage =[];
opt.message = [];


% Docs
% opt.display: 0 (default, no display), m the coronal slice number
% for 3D images
%
% opt.save: 0 (default, no save), {n,m} saves every n iterates in
% output S, if image is 3D, m is the coronal slice number to be
% saved, if m = 0, all slices will be saved, {n,'dir'} saves
% every n iterates in the directory 'dir'
%
% initialEstimate
% pre-computed RHS = ObjMRI.FH(Data)


opt = getFiledsFromUsersOpt(opt,arg);

if nargin==1
    if ObjMRI.isUnderSampled
        if opt.display, fprintf('Reconstruction of undesampled data...\n');end
        Data = ObjMRI.kSpaceUnderSampled;
        underSampling = 1;
    else
        if opt.display, fprintf('Reconstruction of fully sampled data...\n');end
        Data = ObjMRI.kSpace;
        underSampling = 0;
    end
else
    if opt.ReconUnderSampledkSpace && ObjMRI.isUnderSampled
        if opt.display, fprintf('Reconstruction of undesampled data...\n');end
        Data = ObjMRI.kSpaceUnderSampled;
        underSampling = 1;
    else
        if opt.display, fprintf('Reconstruction of fully sampled data...\n');end
        Data = ObjMRI.kSpace;
        underSampling = 0;
        opt.MrRegularizationParameter =0;
    end
end


if nargin ==4
    y = RHS;
else
    y = ObjMRI.FH(Data,underSampling);
end




if strcmpi(opt.MrPriorType,'Gaussian') 
    if isempty(opt.PriorMrImage)
        error('Registered prior image should be provided')
    else
        W = ObjMRI.Prior.W_GaussianKernel(opt.PriorMrImage,opt.MrSigma);
        W = W./repmat(sum(W,2),[1,ObjMRI.Prior.nS]);
    end
end

if strcmpi(opt.MrPriorType,'JBE') 
    if isempty(opt.PriorMrImage)
        error('Registered prior image should be provided')
    else

        W = ObjMRI.Prior.W_JointEntropy(opt.PriorMrImage,opt.JBE_sigma_p) .* ObjMRI.Prior.W_JointEntropy(initialEstimate,opt.JBE_sigma_i);
        W = W./repmat(sum(W,2),[1,ObjMRI.Prior.nS]);
    end
end

if strcmpi(opt.MrPriorType,'Quadratic')
    % By default MrPreCompWeights = 1, but can be used to apply precomputed
    % weights,e.g. in Prior-image-guided MRI reconstruction
    if size(opt.MrPreCompWeights,1) ==1
        W = opt.MrPreCompWeights./ObjMRI.Prior.nS;
    else
        W = opt.MrPreCompWeights./repmat(sum(opt.MrPreCompWeights,2),[1,ObjMRI.Prior.nS]);
    end
end


if opt.MrRegularizationParameter
    if ~strcmpi(opt.MrPriorType,'sTV')
        A = @(x)ObjMRI.FHF(x,underSampling) + ...
            opt.MrRegularizationParameter*ObjMRI.Prior.TransGraphGradUndoCrop(W.*ObjMRI.Prior.GraphGradCrop(x));
    else % TV
        A = @(x)ObjMRI.FHF(x,underSampling) + ...
            opt.MrRegularizationParameter*...
            ObjMRI.Prior.TransGraphGradUndoCrop(ObjMRI.Prior.TV_weights(ObjMRI.Prior.GraphGradCrop(x),opt.TVsmoothingParameter));
    end
else
    A = @(x)ObjMRI.FHF(x,underSampling);
end

[x,S] = ObjMRI.PCG(y,A,img,opt.SENSE_niter,1,opt);
end
