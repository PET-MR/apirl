%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 06/05/2015
%  *********************************************************************
%  This function projects an image into an span 1 sinogram. It receives
%  also as a parameter the subset that it is wanted to be projeted. It must
%  be left empty or in zero for projecting the complete sinogram.
% The span of the sinogram is received as a parameter.
%  When a subset is projected, it returns a sinogram of the original size,
%  but only filled in the bins of the subset. This was desgined this way to
%  be more trnsparent for the user.
% Examples:
%   [sinogram, structSizeSinogram] = ProjectMmr(image, pixelSize_mm, outputPath, span, numberOfSubsets, subsetIndex, useGpu)

function [sinogram, structSizeSino3d] = ProjectMmr(image, pixelSize_mm, outputPath, span, numberOfSubsets, subsetIndex, useGpu)

mkdir(outputPath);
% Check what OS I am running on:
if(strcmp(computer(), 'GLNXA64'))
    os = 'linux';
    pathBar = '/';
elseif(strcmp(computer(), 'PCWIN') || strcmp(computer(), 'PCWIN64'))
    os = 'windows';
    pathBar = '\';
else
    disp('OS not compatible');
    return;
end

if nargin == 6
    useGpu = 0;
elseif nargin < 6
    error('Invalid number of parameters: [sinogram, structSizeSinogram] = ProjectMmrSpan1(image, pixelSize_mm, outputPath, useGpu)');
end
% Handle the number of subsets:
if isempty(numberOfSubsets)
    numberOfSubsets = 0;
end
if(isempty(subsetIndex))
    subsetIndex = 0;
end

% Create output sample sinogram:
% Size of mMr Sinogram's
numTheta = 252; numR = 344; numRings = 64; maxAbsRingDiff = 60; rFov_mm = 594/2; zFov_mm = 258; 
structSizeSino3d = getSizeSino3dFromSpan(numR, numTheta, numRings, rFov_mm, zFov_mm, span, maxAbsRingDiff);
% constant sinogram:
sinogram = ones(numR, numTheta, sum(structSizeSino3d.sinogramsPerSegment), 'single');
sinogramSampleFilename = [outputPath 'sinogramSample'];
interfileWriteSino(sinogram, sinogramSampleFilename, structSizeSino3d);

% Write image in interfile:
filenameImage = [outputPath 'inputImage'];
interfilewrite(single(image), filenameImage, pixelSize_mm);


% Generate projecte sinogram:
filenameProjectionConfig = [outputPath 'projectPhantom.par'];
projectionFilename = [outputPath 'projectedSinogram'];
CreateProjectConfigFileForMmr(filenameProjectionConfig, [filenameImage '.h33'], [sinogramSampleFilename '.h33'], projectionFilename, numberOfSubsets, subsetIndex, useGpu);
status = system(['project ' filenameProjectionConfig])

% Read the projected sinogram:
% if is a subset, get the new size:
if numberOfSubsets ~= 0
    structSizeSino3dSubset = structSizeSino3d;
    structSizeSino3dSubset.numTheta = ceil(structSizeSino3d.numTheta/numberOfSubsets);
    
    fid = fopen([projectionFilename '.i33'], 'r');
    numSinos = sum(structSizeSino3dSubset.sinogramsPerSegment);
    [subset, count] = fread(fid, structSizeSino3dSubset.numTheta*structSizeSino3dSubset.numR*numSinos, 'single=>single');
    fclose(fid);
    subset = reshape(subset, [structSizeSino3dSubset.numR structSizeSino3dSubset.numTheta numSinos]);
    % Fille a sinogram of the original size
    sinogram = zeros(structSizeSino3d.numR, structSizeSino3d.numTheta, numSinos);
    sinogram(:,subsetIndex : numberOfSubsets : end, :) = subset;
else
    fid = fopen([projectionFilename '.i33'], 'r');
    numSinos = sum(structSizeSino3d.sinogramsPerSegment);
    [sinogram, count] = fread(fid, structSizeSino3d.numTheta*structSizeSino3d.numR*numSinos, 'single=>single');
    fclose(fid);
    sinogram = reshape(sinogram, [structSizeSino3d.numR structSizeSino3d.numTheta numSinos]);
end
