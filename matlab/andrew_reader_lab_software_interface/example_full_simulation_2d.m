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

% APIRL PATH
apirlPath = 'E:\apirl-code\trunk\';
addpath(genpath([apirlPath pathBar 'matlab']));
setenv('PATH', [getenv('PATH') sepEnvironment apirlPath pathBar 'build' pathBar 'bin']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') sepEnvironment apirlPath pathBar 'build' pathBar 'bin']);
%% INIT CLASS GPET
PET.scanner = '2D_mMR';
PET.method =  'otf_siddon_cpu';
PET.PSF.type = 'none';
PET.radialBinTrim = 0;
PET.Geom = '';
PET.random_algorithm = 'from_ML_singles_matlab';
PET = classGpet(PET);
%% SIMULATE A BRAIN PHANTOM WITH ATTENUATION, NORMALIZATION, RANDOMS AND SCATTER
[sinogram, delayedSinogram, structSizeSino3d] = interfileReadSino('E:\PatientData\FDG\PETSinoPlusUmap-Converted\PETSinoPlusUmap-00\PETSinoPlusUmap-00-sino-uncomp.s.hdr');
delayedSinogram_2d = delayedSinogram(:,:,60);
load BrainMultiMaps_mMR.mat;
% Get only one slice of the phantom:
tAct = permute(MultiMaps_Ref.PET(:,:,60), [2 1 3]);
tAct = tAct(end:-1:1,:);
tMu = permute(MultiMaps_Ref.uMap(:,:,60), [2 1 3]);
tMu = tMu(end:-1:1,:);
pixelSize_mm = [2.08625 2.08625 2.03125];
xLimits = [-size(tAct,2)/2*pixelSize_mm(2) size(tAct,2)/2*pixelSize_mm(2)];
yLimits = [-size(tAct,1)/2*pixelSize_mm(1) size(tAct,1)/2*pixelSize_mm(1)];
zLimits = [-size(tAct,3)/2*pixelSize_mm(3) size(tAct,3)/2*pixelSize_mm(3)];
refAct = imref3d([size(tAct) 1],xLimits,yLimits,zLimits);
refAt  = imref3d([size(tMu) 1],xLimits,yLimits,zLimits);

% Change the image sie, to the one of the phantom:
PET.init_image_properties(refAct);

% Counts to simulate:
counts = 1e8;
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

% Additive factors:
%r = PET.R(counts*randomsFractions); 
r = PET.R(delayedSinogram_2d);  % Without a delayed sinograms, just
scale_factor_randoms = counts*randomsFraction./sum(r(:));
% Poisson distribution:
r = poissrnd(r.*scale_factor_randoms);

scatterFraction = 0.35;
counts_scatter = counts*scatterFraction;
s_withoutNorm = PET.S(y);
scale_factor_scatter = counts_scatter/sum(s_withoutNorm(:));
s_withoutNorm = s_withoutNorm .* scale_factor_scatter;
% noise for the scatter:
s = poissrnd(s_withoutNorm.*n);
% Add randoms and scatter@
simulatedSinogram = y_poisson + s + r;


