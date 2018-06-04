%% TEST RECONSTRUCTION WITH MEX PROJECTOR AND WITH STANDARD WITH I/O THROUGH HD
apirlPath = '/home/mab15/workspace/apirl-code/trunk/';
addpath([apirlPath 'matlab/andrew_reader_lab_software_interface/']);
set_framework_environment(apirlPath);

%% GENERATE A SINOGRAM  WITH STANDARD METHOD AND RECONTRUCT
opt.method =  'otf_siddon_gpu';
opt.PSF.type = 'shift-invar';
opt.PSF.Width = 2.5;
opt.nSubsets = 1;
opt.verbosity = 1;
PET = classGpet(opt);

image = zeros(PET.image_size.matrixSize);
image(120:160, 220:240, 30:50)=1;
sino = PET.P(image);

% time for projection
tic; a = PET.P(image); time_proj_standard = toc
% time for backprojection
tic; b = PET.PT(a); time_backproj_standard = toc

% time for mlem:
nIter = 60;
tic
sens = PET.Sensitivity(ones(size(sino)));
recon_image_standard = PET.OPMLEM(sino,zeros(size(sino)), sens,PET.ones, nIter);
time_mlem60_standard = toc

opt.nSubsets = 21;
PET = classGpet(opt);
tic;
sens = PET.Sensitivity(ones(size(sino)));
osem_image_standard = PET.OPOSEM(sino,zeros(size(sino)), sens,PET.ones, 3);
time_osem21_3_standard = toc

%% NOW RECONSTRUCT WITH MEX
opt.method =  'mex_otf_siddon_gpu';
opt.PSF.type = 'shift-invar';
opt.PSF.Width = 2.5;
opt.nSubsets = 1;
opt.verbosity = 1;
PET = classGpet(opt);


% time for projection
tic; c = PET.P(image); time_proj_mex = toc
% time for backprojection
tic; d = PET.PT(sino); time_backproj_mex = toc

% time for mlem:
nIter = 60;
tic
sens = PET.Sensitivity(ones(size(sino)));
recon_image_mex = PET.OPMLEM(sino,zeros(size(sino)), sens,PET.ones, nIter);
time_mlem60_mex = toc

opt.nSubsets = 21;
PET = classGpet(opt);
tic;
sens = PET.Sensitivity(ones(size(sino)));
osem_image_mex = PET.OPOSEM(sino,zeros(size(sino)), sens,PET.ones, 3);
time_osem21_3_mex = toc