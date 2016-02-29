
%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 28/094/2015
%  *********************************************************************
% Example that uses simulatePhantom. It is also used to see normalization
% factors for span-n sinograms.
clear all 
close all
%% APIRL PATH
apirlPath = '/home/mab15/workspace/apirl-code/trunk/';

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

%% CUDA PATH
cudaPath = '/usr/local/cuda/';
setenv('PATH', [getenv('PATH') sepEnvironment cudaPath pathBar 'bin']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') sepEnvironment cudaPath pathBar 'lib64']);
%% STIR PATH
stirPath = '/usr/local/stir3.0/';
stirMatlabPath = '/home/mab15/workspace/KCL/apirl-kcl/trunk/stir/';
scriptsPath = [stirMatlabPath 'scripts/'];
%% SET ENVIRONMENT AND MATLAB PATHS
addpath(genpath([apirlPath pathBar 'matlab']));
addpath(genpath(stirMatlabPath));
setenv('PATH', [getenv('PATH') sepEnvironment apirlPath pathBar 'build' pathBar 'bin' ':' stirPath pathBar 'bin/']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') sepEnvironment apirlPath pathBar 'build' pathBar 'bin' ':' stirPath pathBar 'lib/' ]);
%% OUTPUT PATH
%outputPath = '/home/mab15/workspace/KCL/AxialCompression/Simulations/PointsPhantom/';
outputPath = '/home/mab15/workspace/KCL/xtal_efficiencies/simulated_phantoms/brain/';
mkdir(outputPath);
%% SIZE OF RECONSTRUCTED IMAGE
% Create image from the same size than used by siemens:
% Size of the pixels:
pixelSize_mm = [2.08625 2.08625 2.03125];
% The size in pixels:
imageSize_pixels = [288 288 127]; % For cover the full Fov: 596/4.1725=142.84
% Size of the image to cover the full fov:
sizeImage_mm = pixelSize_mm .* imageSize_pixels;
%% CREATE PHANTOM
% [phantom, refImage, attenMap_1_cm] = CreateDefrisePhantom(imageSize_pixels, pixelSize_mm);
binaryFilename = '/home/mab15/workspace/KCL/xtal_efficiencies/subject04_crisp_v.rawb';
[phantom, attenMap_1_cm, refImage] = CreateBrainPhantom(binaryFilename, imageSize_pixels, pixelSize_mm);
interfilewrite( phantom, [outputPath 'phantom'], pixelSize_mm);
interfilewrite( attenMap_1_cm, [outputPath 'attenMap'], pixelSize_mm);
sigma_pixels = 2;
%phantom=imgaussian(phantom,sigma_pixels);
% Convert to single:
phantom = single(phantom);
%% SINOGRAM'S SIZE
% Size of mMr Sinogram's
numTheta = 252; numR = 344; numRings = 64; maxAbsRingDiff = 60; rFov_mm = 594/2; zFov_mm = 258;
span_choice = 1;
structSizeSino3d_span1 = getSizeSino3dFromSpan(numR, numTheta, numRings, rFov_mm, zFov_mm, span_choice, maxAbsRingDiff);
structSizeSino3d_span51 = getSizeSino3dFromSpan(numR, numTheta, numRings, rFov_mm, zFov_mm, 51, maxAbsRingDiff);
%% SIMULATE PHANTOM
% span 51:
outputPath = '/home/mab15/workspace/KCL/AxialCompression/Simulations/Defrise/testSpan51Norm/';
normFilename = '/home/mab15/workspace/KCL/Biograph_mMr/Normalization/NormFiles/Norm_20150130133645.n';
%% WITHOUT NORM
[emissionSinogramSpan51NoNorm, structSizeSino3d, overall_nf_3d, afsSinogram, randomsSinogram, scatterSinogram] = simulatePhantom(phantom, [], refImage, 51, 5, [], [], [], outputPath, 1);
interfileWriteSino(emissionSinogramSpan51NoNorm,[outputPath 'sinoSpan51'],structSizeSino3d_span51);
%% WITH NORM
[emissionSinogramSpan51, structSizeSino3d, overall_nf_3d, afsSinogram, randomsSinogram, scatterSinogram] = simulatePhantom(phantom, [], refImage, 51, 5, normFilename, [], [], outputPath, 1);
interfileWriteSino(emissionSinogramSpan51,[outputPath 'sinoSpan51_wNorm'],structSizeSino3d_span51);
%% WITH NORM AND ATTENUATION
[emissionSinogramSpan51, structSizeSino3d, overall_nf_3d, afsSinogram, randomsSinogram, scatterSinogram] = simulatePhantom(phantom, attenMap_1_cm, refImage, 51, 5, normFilename, [], [], outputPath, 1);
interfileWriteSino(emissionSinogramSpan51,[outputPath 'sinoSpan51_wNormAtten'],structSizeSino3d_span51);
%% WITH NORM AND ATTENUATION AND RANDOMS
[emissionSinogramSpan51, structSizeSino3d, overall_nf_3d, afsSinogram, randomsSinogram, scatterSinogram] = simulatePhantom(phantom, attenMap_1_cm, refImage, 51, 5, normFilename, 1, [], outputPath, 1);
interfileWriteSino(emissionSinogramSpan51,[outputPath 'sinoSpan51_wNormAttenRand'],structSizeSino3d_span51);
% 
% 
% overall_ncf_3d = zeros(size(emissionSinogramSpan51NoNorm));
% % Now the normalzaition factors must be the compression factor(only if there were no normalization):
% for i = 1 : sum(structSizeSino3d_span51.sinogramsPerSegment)
%     overall_ncf_3d(:,:,i) = 1./structSizeSino3d_span51.numSinosMashed(i);
% end

