
%% PET object initialization
opt.method =  'otf_siddon_gpu';

opt.scanner = 'mMR';
opt.PSF.Width = 4;
opt.sinogram_size.span = -1;
opt.nSubsets = 1;
opt.random_algorithm = 'from_ML_singles_matlab';
PET = classGpet(opt);
%% SIMULATE A BRAIN PHANTOM WITH ATTENUATION, NORMALIZATION, RANDOMS AND SCATTER
%[sinogram, delayedSinogram, structSizeSino3d] = interfileReadSino('E:\PatientData\FDG\PETSinoPlusUmap-Converted\PETSinoPlusUmap-00\PETSinoPlusUmap-00-sino-uncomp.s.hdr');
load BrainMultiMaps_mMR.mat;

tAct = permute(MultiMaps_Ref.PET, [2 1 3]);
tAct = tAct(end:-1:1,:,:);
tMu = permute(MultiMaps_Ref.uMap, [2 1 3]);
tMu = tMu(end:-1:1,:,:);
pixelSize_mm = [2.08625 2.08625 2.03125];
xLimits = [-size(tAct,2)/2*pixelSize_mm(2) size(tAct,2)/2*pixelSize_mm(2)];
yLimits = [-size(tAct,1)/2*pixelSize_mm(1) size(tAct,1)/2*pixelSize_mm(1)];
zLimits = [-size(tAct,3)/2*pixelSize_mm(3) size(tAct,3)/2*pixelSize_mm(3)];
refAct = imref3d(size(tAct),xLimits,yLimits,zLimits);
refAt  = imref3d(size(tMu),xLimits,yLimits,zLimits);

% Change the image sie, to the one of the phantom:
PET.init_image_properties(refAct);

% Change the span size:
span = 11;
numRings = 64;
maxRingDifference = 60;
PET.init_sinogram_size(span, numRings, maxRingDifference);

% Counts to simulate:
counts = 1e9;
randomsFraction = 0.1;
scatterFraction = 0.35;
truesFraction = 1 - randomsFraction - scatterFraction;

% Geometrical projection:
y = PET.P(tAct);

% Multiplicative correction factors:
ncf = PET.NCF;
acf= PET.ACF(tMu, refAct);
% Convert into factors:
n = ncf; a = acf;
n(n~=0) = 1./ n(n~=0); a(a~=0) = 1./ a(a~=0);
% Introduce poission noise:
y = y.*n.*a;
scale_factor = counts*truesFraction/sum(y(:));
y_poisson = poissrnd(y.*scale_factor);

%% Simulation set-up (2D)
% y is simulated sinogram
SensImg = PET.Sensitivity(ones(size(y)));
%% Prior object initilization (2D)
opt.ImageSize= [344,344,127];
opt.sWindowSize = 3; % search window size
opt.lWindowSize = 1; % loca
opt.imCropFactor = 3.5;
G = PriorsClass(opt);
%% Reconstruction options

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
            
opt.save = 0;
opt.nIter = 200;
opt.display = 1;
%opt.prior = 'modified_lot' ;%approx_joint_entropy, kiapio, tikhonov 
opt.prior = 'approx_joint_entropy';
opt.weight_method = 'nl_side_similarity';%'bowsher';
opt.lambda = 60;


if strcmpi(opt.prior, 'tikhonov')
    if strcmpi(opt.weight_method,'nl_side_similarity')
        opt.sigma_ker = 0.08;
        opt.nl_weights = G.W_GaussianKernel(T1,opt.sigma_ker);
    elseif strcmpi(opt.weight_method,'bowsher')
        opt.B = 7;
        opt.nl_weights = G.W_Bowsher(T1,opt.B);
    end
elseif strcmpi(opt.prior, 'approx_joint_entropy')
    opt.sigma_a = 10;
    opt.sigma_f = 800;
    opt.je_weights = G.W_JointEntropy(T1,opt.sigma_a);
elseif strcmpi(opt.prior, 'kiapio')
    opt.alpha = 1;
    opt.normVectors = G.normalVectors(T1);
elseif strcmpi(opt.prior, 'modified_lot')
    opt.alpha = 1;
    opt.beta = 0.05;
    opt.normVectors = G.normalVectors(T1);
end

recon = G.MAP_OSEM(PET,y,0, SensImg,opt);