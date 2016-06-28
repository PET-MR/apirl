
%% PET object initialization
opt.method =  'otf_siddon_cpu';

opt.scanner = 'mMR';
opt.PSF.Width = 4;
opt.sinogram_size.span = -1;
opt.nSubsets = 1;
opt.random_algorithm = 'from_ML_singles_matlab';
PET = classGpet(opt);
%% Simulation set-up (2D)
% y is simulated sinogram
SensImg = PET.Sensitivity(ones(size(y)));
%% Prior object initilization (2D)
opt.ImageSize= [344,344,1];
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
opt.prior = 'modified_lot' ;%approx_joint_entropy, kiapio, tikhonov 
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