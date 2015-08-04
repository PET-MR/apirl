%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 13/07/2015
%  *********************************************************************
%  This function projects an image into sinogram. It uses a 2d projection.
%  It can project only one sinogram 2d or a set of sinogram 2d for a
%  volume. The number of slices must be the same than sinograms2d.
% Examples:
%   [sinogram, structSizeSinogram] = ProjectMmr2d(image, pixelSize_mm, outputPath, numberOfSubsets, subsetIndex, useGpu)

function [sinogram, structSizeSino] = ProjectMmr2d(image, pixelSize_mm, outputPath, numberOfSubsets, subsetIndex, useGpu)

if ~isdir(outputPath)
    mkdir(outputPath);
end
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

if nargin == 5
    useGpu = 0;
elseif nargin < 5
    error('Invalid number of parameters: [sinogram, structSizeSino] = ProjectMmr2d(image, pixelSize_mm, outputPath, numberOfSubsets, subsetIndex, useGpu)');
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
numTheta = 252; numR = 344; numRings = 64; rFov_mm = 594/2; zFov_mm = 258;
structSizeSino = getSizeSino2dStruct(numR, numTheta, numRings, rFov_mm, zFov_mm);

% Check the size of the image:
if (size(image,3) ~= numRings) && (size(image,3) ~= 1)
    warning('The input image for the projection2d for the Siemens mMR needs to have 64 slices or be only one slice. Image will be resized.');
    [image] = ImageResize(image, [size(image,1) size(image,2) structSizeSino.numZ]);
    pixelSize_mm(3) = zFov_mm ./ numRings;
else
    if (size(image,3) == 1)
        structSizeSino.numZ = 1;
    end
end
% constant sinogram:
sinogram = ones(numR, numTheta, structSizeSino.numZ, 'single');
sinogramSampleFilename = [outputPath 'sinogramSample'];
interfileWriteSino(sinogram, sinogramSampleFilename, structSizeSino);

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
    structSizeSinoSubset = structSizeSino;
    structSizeSinoSubset.numTheta = ceil(structSizeSino.numTheta/numberOfSubsets);
    
    fid = fopen([projectionFilename '.i33'], 'r');
    numSinos = sum(structSizeSinoSubset.sinogramsPerSegment);
    [subset, count] = fread(fid, structSizeSinoSubset.numTheta*structSizeSinoSubset.numR*numSinos, 'single=>single');
    fclose(fid);
    subset = reshape(subset, [structSizeSinoSubset.numR structSizeSinoSubset.numTheta numSinos]);
    % Fille a sinogram of the original size
    sinogram = zeros(structSizeSino.numR, structSizeSino.numTheta, numSinos);
    sinogram(:,subsetIndex : numberOfSubsets : end, :) = subset;
else
    fid = fopen([projectionFilename '.i33'], 'r');
    numSinos = sum(structSizeSino.numZ);
    [sinogram, count] = fread(fid, structSizeSino.numTheta*structSizeSino.numR*numSinos, 'single=>single');
    fclose(fid);
    sinogram = reshape(sinogram, [structSizeSino.numR structSizeSino.numTheta numSinos]);
end
