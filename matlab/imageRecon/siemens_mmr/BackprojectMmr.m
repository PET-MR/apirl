%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 19/05/2015
%  *********************************************************************
%  This function backprojects an span-x sinogram that is a three dimensional
%  matrix with the size fixed for the Siemens mMr: (344,252,4084). The
%  output image is defined by two paramters:
%   -imageSize_pixels: vector with three elemnts with the size in pixels of
%   the image. e.g. [172 172 127].
%   -pixelSize_mm: pixel size for each corrdinate, a three elements vector.
%   e.g.: [4.1725 4.1725 2.03125]
% The span of the sinogram is received as a parameter.
% It receives also as a parameter the subset that it is wanted to be backprojeted. 
% It must be left empty or in zero for projecting the complete sinogram.
%  11/12/2015: The span parameter, now can be replaced by a
%  structSizeSino3d
% Examples:
%   [image, pixelSize_mm] = BackprojectMmr(sinogram, imageSize_pixels, pixelSize_mm, outputPath, structSizeSino3d_span, numberOfSubsets, subsetIndex, useGpu)

function [image, pixelSize_mm] = BackprojectMmr(sinogram, imageSize_pixels, pixelSize_mm, outputPath, structSizeSino3d_span, numberOfSubsets, subsetIndex, useGpu)

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

if nargin == 7
    useGpu = 0;
elseif nargin < 7
    error('Invalid number of parameters: [image, pixelSize_mm] = BackprojectMmrSpan1(sinogram, imageSize_pixels, pixelSize_mm, outputPath, useGpu)');
end

% Handle the number of subsets:
if isempty(numberOfSubsets)
    numberOfSubsets = 0;
end
if(isempty(subsetIndex))
    subsetIndex = 0;
end

if numel(pixelSize_mm) ~= 3
    error('The image size (imageSize_pixels) and pixel size (pixelSize_mm) parameters must be a three-elements vector.');
end

% Check if is an struct or the span value:
if(isstruct(structSizeSino3d_span))
    structSizeSino3d = structSizeSino3d_span;
else
    % The parameter has only the span value:
    % Size of mMr Sinogram's
    numTheta = 252; numR = 344; numRings = 64; maxAbsRingDiff = 60; rFov_mm = 594/2; zFov_mm = 258; 
    structSizeSino3d = getSizeSino3dFromSpan(numR, numTheta, numRings, rFov_mm, zFov_mm, structSizeSino3d_span, maxAbsRingDiff);
end

% Create output sample sinogram.
% Generate a constant image:
image = ones(imageSize_pixels);
% Write image in interfile:
filenameImage = [outputPath 'imageSample'];
interfilewrite(single(image), filenameImage, pixelSize_mm);

% constant sinogram:
sinogramFilename = [outputPath 'inputSinogram'];
% I replace writing a real sinogram for an empty sinogram to save space and
% time:
interfileWriteSino(single(sinogram), sinogramFilename, structSizeSino3d);

% Generate backprojected image:
filenameBackprojectionConfig = [outputPath 'backprojectSinogram.par'];
backprojectionFilename = [outputPath 'backprojectedImage'];
CreateBackprojectConfigFileForMmr(filenameBackprojectionConfig, [sinogramFilename '.h33'], [filenameImage '.h33'], backprojectionFilename, numberOfSubsets, subsetIndex, useGpu);
status = system(['backproject ' filenameBackprojectionConfig])

% Remove the input sinogram:
delete([sinogramFilename '.*']);

% Read the image:
image = interfileRead([backprojectionFilename '.h33']);
