%% MODIFED VERSION OF ABOFAZL EXAMPLE
clear all 
close all
apirlPath = '/home/mab15/workspace/apirl-code/trunk/';
addpath([apirlPath 'matlab/andrew_reader_lab_software_interface/']);
set_framework_environment(apirlPath);

% load data
load('/data/Scans/PatientData/AD_patients/P02/e7/data-Converted/data-LM-00/sino_rawdata_100/rawdata_matlab.mat', 'Prompts','RS','AN','T1');

% initialize PET object
set_framework_environment();

opt.method =  'otf_siddon_gpu';
opt.nSubsets = 1;
opt.nIter = 100;
opt.PSF.Width = 2.5;
PET = classGpet(opt);
[T1, refImageT1, T1InPetFov, refImagePetFov] = PET.getMrInNativeImageSpace('/data/Scans/PatientData/AD_patients/P02/T1/');
clear T1;

%% HIGH RESOLUTION OBJECT
highresImage.voxelSize_mm = [refImagePetFov.PixelExtentInWorldX refImagePetFov.PixelExtentInWorldY 2.03125]; % CHANGE THE SIZE OF T1 BECAUSE OF NOT ENOUGH MEMORY
highresImage.matrixSize = round(PET.image_size.matrixSize.*PET.image_size.voxelSize_mm./highresImage.voxelSize_mm);
paramPET.scanner = 'mMR';
paramPET.method =  'otf_siddon_gpu';
paramPET.PSF.type = 'none';
paramPET.radialBinTrim = 0;
paramPET.Geom = '';
paramPET.sinogram_size.span = 11;
paramPET.nSubsets = 1;
paramPET.random_algorithm = 'from_ML_singles_matlab';
paramPET.image_size = highresImage;
paramPET.verbosity = 1;
PET_highres = classGpet(paramPET);
% Now with PSF:
% With PSF
psfWidth_mm = 4.5;
paramPET.PSF.type = 'shift-invar';
paramPET.PSF.Width = psfWidth_mm;
PET_highres_psf = classGpet(paramPET);
%% RESAMPLE THE MR
[T1InPetFov, refImagePetFov] = ImageResample(T1InPetFov, refImagePetFov, PET_highres.ref_image);
%% MAP PET image reconstruction
opt.OptimizationMethod = 'DePierro';%'OSL'
opt.PriorType = 'Bowsher';%'Quadratic'; %'JointBurgEntropy', 
opt.MrImage = T1InPetFov./max(T1InPetFov(:));
opt.display = 40;
% parameters experimentally chosen for \AD_patients\P02 data
opt.BowsherB = 40;
opt.MrSigma = 0.01;
opt.PetSigma  = 1;
if strcmpi(opt.PriorType,'Quadratic')
    opt.RegualrizationParameter = 3.0e+03;
else
    opt.RegualrizationParameter = 1.0e+05;
end
outputPath = sprintf('/data/Scans/PatientData/AD_patients/P02/e7/data-Converted/data-LM-00/sino_rawdata_100/MR_guided_OPMLEM_hr_param%.2e/', opt.RegualrizationParameter);
if ~isdir(outputPath)
    mkdir(outputPath);
end
interfilewrite(T1InPetFov, [outputPath 'prior'], PET_highres.image_size.voxelSize_mm);
numIterations = 150;
saveInterval = 10;



additive = RS.*AN;
clear RS;
Img = PET_highres.MAPEM_DS(Prompts, AN, additive, PET_highres.ones, numIterations, opt, outputPath, saveInterval);% MAPEM(Prompts,RS, SensImg,PET.ones,PET.nIter,opt);

