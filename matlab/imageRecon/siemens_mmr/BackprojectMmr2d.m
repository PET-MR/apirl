%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 19/05/2015
%  *********************************************************************
%  This function backprojects a sinogram into an image. It works with only
%  one sinogram or with a set of direct sinograms (64 for the Siemens mMR).
%  The imageSize and the pixel size are vectors with two elements. The z
%  dimensions are fixed (64 rings for a volvume or 1 for one slice). 
% Examples:
%   [image, pixelSize_mm] = BackprojectMmr2d(sinogram, imageSize_pixels, pixelSize_mm, outputPath, numberOfSubsets, subsetIndex, useGpu)

function [image, pixelSize_mm] = BackprojectMmr2d(sinogram, imageSize_pixels, pixelSize_mm, outputPath, numberOfSubsets, subsetIndex, useGpu)

if ~isdir(outputPath)
    mkdir(outputPath);
end

if nargin == 6
    useGpu = 0;
elseif nargin < 6
    error('Invalid number of parameters: [image, pixelSize_mm] = BackprojectMmr2d(sinogram, imageSize_pixels, pixelSize_mm, outputPath, numberOfSubsets, subsetIndex, useGpu)');
end

% Handle the number of subsets:
if isempty(numberOfSubsets)
    numberOfSubsets = 0;
end
if(isempty(subsetIndex))
    subsetIndex = 0;
end
% % This shouldnt be necessary:
% if numberOfSubsets ~= 0
%     subset = zeros(size(sinogram));
%     subset(:,subsetIndex : numberOfSubsets : end) = sinogram(:,subsetIndex : numberOfSubsets : end);
%     sinogram=subset;
% end
if numel(pixelSize_mm) ~= 2
    % EDIT: Sam Ellis - 05.01.2016
    if numel(pixelSize_mm) == 3
        pixelSize_mm = pixelSize_mm(1:2);
    else
        error('The image size (imageSize_pixels) and pixel size (pixelSize_mm) parameters must be a two-elements vector with the x-y sizes.');
    end
end

if (size(sinogram,3) ~= 1) && (size(sinogram,3) ~= 64)
    error('The sinogram for the projection2d for the Siemens mMR needs to have 64 sinograms (direct sinograms) or be only one sinogram 2d.');
end

% Create output sample sinogram.
% Size of mMr Sinogram's
numTheta = 252; numR = 344; maxAbsRingDiff = 60; rFov_mm = 594/2; zFov_mm = 258; 

% if several sinograms 2d:
if size(sinogram,3) == 64
    numRings = 64;
    imageSize_pixels(3) = size(sinogram,3);
    pixelSize_mm(3) =  zFov_mm ./ numRings;
else
    numRings = 1;
end
structSizeSino = getSizeSino2dStruct(numR, numTheta, numRings, rFov_mm, zFov_mm);
% Generate a constant image:
image = ones(imageSize_pixels);
% Write image in interfile:
filenameImage = [outputPath 'imageSample'];
interfilewrite(single(image), filenameImage, pixelSize_mm);

% constant sinogram:
sinogramFilename = [outputPath 'inputSinogram'];
interfileWriteSino(single(sinogram), sinogramFilename, structSizeSino);

% Generate backprojected image:
filenameBackprojectionConfig = [outputPath 'backprojectSinogram.par'];
backprojectionFilename = [outputPath 'backprojectedImage'];
CreateBackprojectConfigFileForMmr(filenameBackprojectionConfig, [sinogramFilename '.h33'], [filenameImage '.h33'], backprojectionFilename, numberOfSubsets, subsetIndex, useGpu);
status = system(['backproject "' filenameBackprojectionConfig '"'])

% Read the image:
image = interfileRead([backprojectionFilename '.h33']);
