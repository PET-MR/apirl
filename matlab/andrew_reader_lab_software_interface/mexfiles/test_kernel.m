%% TEST RECONSTRUCTION WITH MEX PROJECTOR AND WITH STANDARD WITH I/O THROUGH HD
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
nIter = 20;
tic
sens = PET.Sensitivity(ones(size(sino)));
recon_image_standard = PET.OPMLEM(sino,zeros(size(sino)), sens,PET.ones, nIter);
time_mlem60_standard = toc
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
tic; d = PET.PT(a); time_backproj_mex = toc

% time for mlem:
nIter = 20;
tic
sens = PET.Sensitivity(ones(size(sino)));
recon_image_standard = PET.OPMLEM(sino,zeros(size(sino)), sens,PET.ones, nIter);
time_mlem60_standard = toc