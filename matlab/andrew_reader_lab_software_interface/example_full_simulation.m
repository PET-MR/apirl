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
load subject_4_tpm.mat;
load brainWeb3D.mat;
% tAct: activity image.
% tMu: attenuationMap. Is in different scale and size.
% Register both images:
[optimizer,metric] = imregconfig('multimodal');
optimizer. InitialRadius = 0.01;
refAt  = imref3d(size(tMu));
refAct = imref3d(size(tAct),2.086250,2.086250,2.031250);
[movingRegistered refAt] = imregister(tMu, refAt,tAct, refAct, 'Rigid', optimizer, metric);

PET.scanner = 'mMR';
PET.method =  'otf_siddon_gpu';
PET.PSF.type = 'none';
PET.radialBinTrim = 0;
PET.Geom = '';
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
counts = 1e9/sum(y(:));
y = y.*n.*a;
y_poisson = poissrnd(y.*counts)./counts;

% Additive factors:
%r = PET.R;  % Without a delayed sinograms, just
s = PET.S(y);
% noise for the scatter:
scatterFraction = 0.3;
s = poissrnd(s.*n.*counts.*scatterFraction)./(counts);
% Add randoms and scatter@
y_poisson = y_poisson + s + r;

