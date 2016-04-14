%% EXAMPLE MLEM MARTIN PROJECTOR (ANY SPAN)
clear all 
close all
%% CONFIGURE PATHS
% Check what OS I am running on:
if(strcmp(computer(), 'GLNXA64'))
    os = 'linux';
    pathBar = '/';
    sepEnvironment = ':';
elseif(strcmp(computer(), 'PCWIN') || strcmp(computer(), 'PCWIN64'))
    os = 'windows';
    pathBar = '\';
    sepEnvironment = ';';
else
    disp('OS not compatible');
    return;
end
% CUDA PATH
cudaPath = '/usr/local/cuda/';
setenv('PATH', [getenv('PATH') sepEnvironment cudaPath pathBar 'bin']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') sepEnvironment cudaPath pathBar 'lib64']);
% APIRL PATH
apirlPath = '/workspaces/Martin/apirl-code/trunk/';
addpath(genpath([apirlPath pathBar 'matlab']));
setenv('PATH', [getenv('PATH') sepEnvironment apirlPath pathBar 'build' pathBar 'bin']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') sepEnvironment apirlPath pathBar 'build' pathBar 'bin']);
%% SIMULATE A BRAIN PHANTOM WITH ATTENUATION, NORMALIZATION, RANDOMS AND SCATTER
[sinogram, delayedSinogram, structSizeSino3d] = interfileReadSino('/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/BRAIN_PETMR/SINOGRAMS/PET_ACQ_68_20150610155347-0uncomp.s.hdr');
load subject_4_tpm.mat;
load brainWeb3D.mat;
% % tAct: activity image.
% % tMu: attenuationMap. Is in different scale and size.
% % Register both images:
% [optimizer,metric] = imregconfig('multimodal');
% optimizer. InitialRadius = 0.01;
% refAt  = imref3d(size(tMu));
% refAct = imref3d(size(tAct),2.086250,2.086250,2.031250);
% [movingRegistered refAt] = imregister(tMu, refAt,tAct, refAct, 'Rigid', optimizer, metric);
% tAct: activity image. Need to transpose and invert the y axis.
% tMu: attenuationMap. Is in different scale and size.
tAct = permute(tAct, [2 1 3]);
tAct = tAct(end:-1:1,:,:);
tMu = permute(tMu, [2 1 3]);
tMu = tMu(end:-1:1,:,:);
% Register both images:
[optimizer,metric] = imregconfig('multimodal');
optimizer.InitialRadius = 0.01;
xLimitsAt = [-size(tMu,2)/2*1.8 size(tMu,2)/2*1.8];
yLimitsAt = [-size(tMu,1)/2*1.8 size(tMu,1)/2*1.8];
zLimitsAt = [-size(tMu,3)/2*2.5 size(tMu,3)/2*2.5];
refAt  = imref3d(size(tMu),xLimitsAt,yLimitsAt,zLimitsAt);
xLimits = [-size(tAct,2)/2*2.08625 size(tAct,2)/2*2.08625];
yLimits = [-size(tAct,1)/2*2.08625 size(tAct,1)/2*2.08625];
zLimits = [-size(tAct,3)/2*2.03125 size(tAct,3)/2*2.03125];
refAct = imref3d(size(tAct),xLimits,yLimits,zLimits);
%[tMu refAt] = imregister(tMu, refAt,tAct, refAct, 'affine', optimizer, metric);
[tMu, refAt] = ImageResample(tMu, refAt, refAct);

PET.scanner = 'mMR';
PET.method =  'otf_siddon_gpu';
PET.PSF.type = 'none';
PET.radialBinTrim = 0;
PET.Geom = '';
PET.random_algorithm = 'from_ML_singles_matlab';
PET = classGpet(PET);

% Change the image sie, to the one of the phantom:
PET.init_image_properties(refAct);
% Change the span size:
span = 1;
numRings = 64;
maxRingDifference = 60;
PET.init_sinogram_size(span, numRings, maxRingDifference);

% Geometrical projection:
y = PET.P(tAct);
% Multiplicative correction factors:
ncf = PET.NCF;
acf= PET.ACF(tMu, refAct);
% Convert into factors:
n = ncf; a = acf;
n(n~=0) = 1./ n(n~=0); a(a~=0) = 1./ a(a~=0);
% Introduce poission noise:
counts = 1e9;
y = y.*n.*a;
scale_factor = counts/sum(y(:));
y_poisson = poissrnd(y.*scale_factor);

% Additive factors:
randomsFractions = 0.3;
%r = PET.R(counts*randomsFractions); 
r = PET.R(delayedSinogram);  % Without a delayed sinograms, just
scale_factor_randoms = counts*randomsFractions./sum(r);
% Poisson distribution:
r = poissrnd(r.*scale_factor_randoms);

scatterFraction = 0.5;
counts_scatter = counts*scatterFraction;
s_withoutNorm = PET.S(y);
scale_factor_scatter = counts_scatter/sum(s_withoutNorm(:));
s_withoutNorm = s_withoutNorm .* scale_factor_scatter;
% noise for the scatter:
s = poissrnd(s_withoutNorm.*n);
% Add randoms and scatter@
simulatedSinogram = y_poisson + s + r;


