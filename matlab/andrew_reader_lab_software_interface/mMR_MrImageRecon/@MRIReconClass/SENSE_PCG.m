function [x,S] = SENSE_PCG2(ObjMRI,arg,initialEstimate,RHS,PriorImage)

if nargin==1, arg.save = 0; end

if nargin>=3
    img = initialEstimate;
else
    img = zeros(ObjMRI.nkSamples,'single');
end

if nargin==5
   PriorImage = [];
end

% Default values
opt.SENSE_niter = 10;
opt.ReconUnderSampledkSpace = 0;
opt.RegularizationParameter = 0;
opt.PriorType = 'QP'; % 'sTV', 'Bowsher', 'JBE'
opt.PreCompWeights = 1;
opt.TVsmoothingParameter = 1e-2;
opt.display = 0;
opt.save = 0;
opt.JBE_sigma_p = 10; % prior image
opt.JBE_sigma_i = 10; % initial image
opt.Bowhser_B = 30;


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
        fprintf('Reconstruction of undesampled data...\n');
        Data = ObjMRI.kSpaceUnderSampled;
        underSampling = 1;
    else
        fprintf('Reconstruction of fully sampled data...\n');
        Data = ObjMRI.kSpace;
        underSampling = 0;
    end
else
    if opt.ReconUnderSampledkSpace && ObjMRI.isUnderSampled
        fprintf('Reconstruction of undesampled data...\n');
        Data = ObjMRI.kSpaceUnderSampled;
        underSampling = 1;
    else
        fprintf('Reconstruction of fully sampled data...\n');
        Data = ObjMRI.kSpace;
        underSampling = 0;
        opt.RegularizationParameter =0;
    end
end


if nargin >=4
    y = RHS;
else
    y = ObjMRI.FH(Data,underSampling);
end




if strcmpi(opt.PriorType,'Bowsher') 
    if isempty(PriorImage)
        error('Registered prior image should be provided')
    else
        W = ObjMRI.W_Bowsher(PriorImage,opt.Bowhser_B);
    end
end

if strcmpi(opt.PriorType,'JBE') 
    if isempty(PriorImage)
        error('Registered prior image should be provided')
    else

        W = ObjMRI.W_JointEntropy(PriorImage,opt.JBE_sigma_p) .* ObjMRI.W_JointEntropy(initialEstimate,opt.JBE_sigma_i);
        W = W./repmat(sum(W,2),[1,ObjMRI.nS]);
    end
end

if strcmpi(opt.PriorType,'QP')
    % By default PreCompWeights = 1, but can be used to apply precomputed
    % weights,e.g. in Prior-image-guided MRI reconstruction
    if opt.PreCompWeights==1
        W = 1;
    else
        W = opt.PreCompWeights./repmat(sum(opt.PreCompWeights,2),[1,ObjMRI.nS]);
    end
end


if opt.RegularizationParameter
    if ~strcmpi(opt.PriorType,'sTV')
        A = @(x)ObjMRI.FHF(x,underSampling) + ...
            opt.RegularizationParameter*ObjMRI.TransGraphGradUndoCrop(W.*ObjMRI.GraphGradCrop(x));
    else % TV
        A = @(x)ObjMRI.FHF(x,underSampling) + ...
            opt.RegularizationParameter*...
            ObjMRI.TransGraphGradUndoCrop(ObjMRI.TV_weights(ObjMRI.GraphGradCrop(x),opt.TVsmoothingParameter));
    end
else
    A = @(x)ObjMRI.FHF(x,underSampling);
end

[x,S] = ObjMRI.PCG(y,A,img,opt.SENSE_niter,1,opt);
end
