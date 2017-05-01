% load data
load('/data/Scans/PatientData/AD_patients/P02/e7/data-Converted/data-LM-00/sino_rawdata_100/rawdata_matlab.mat', 'Prompts','RS','AN','T1');

% initialize PET object
set_framework_environment();

opt.method =  'otf_siddon_gpu';
opt.nSubsets = 1;
opt.nIter = 100;
opt.PSF.Width = 2.5;

PET = classGpet(opt);
SensImg = PET.Sensitivity(AN);

%% MAP PET image reconstruction
opt.OptimizationMethod = 'DePierro';%'OSL'
opt.PriorType = 'Bowsher';%'Quadratic'; %'JointBurgEntropy', 
opt.MrImage = T1./max(T1(:));
opt.display = 40;
% parameters experimentally chosen for \AD_patients\P02 data
opt.BowsherB = 40;
opt.MrSigma = 0.01;
opt.PetSigma  = 1;

if strcmpi(opt.PriorType,'Quadratic')
    opt.RegualrizationParameter = 3.0e+03;
else
    opt.RegualrizationParameter = 2.0e+05;
end

Img = PET.MAPEM(Prompts,RS, SensImg,PET.ones,PET.nIter,opt);

