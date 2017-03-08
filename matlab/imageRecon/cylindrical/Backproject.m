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
% The sinogram size is received as a parameter: structSizeSino. It can be a
% 2d, 2dmultislice or 3d sinogram.
% It receives also as a parameter the subset that it is wanted to be backprojeted. 
% It must be left empty or in zero for projecting the complete sinogram.

% Examples:
%   [image, pixelSize_mm] = Backproject(sinogram, imageSize_pixels, pixelSize_mm, outputPath, scanner, scanner_parameters, structSizeSino, numberOfSubsets, subsetIndex, useGpu, numSamples, numAxialSamples)

function [image, pixelSize_mm, output_message] = Backproject(sinogram, imageSize_pixels, pixelSize_mm, outputPath, scanner, scanner_parameters, structSizeSino, numberOfSubsets, subsetIndex, useGpu, numSamples, numAxialSamples)

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

if nargin == 9
    useGpu = 0;
    numSamples = 1;
    numAxialSamples = 1;
elseif nargin == 10
    numSamples = 1;
    numAxialSamples = 1;
elseif nargin == 11
    numAxialSamples = 1;
elseif nargin < 9
    error('Invalid number of parameters: [image, pixelSize_mm] = BackprojectMmrSpan1(sinogram, imageSize_pixels, pixelSize_mm, outputPath, useGpu)');
end

% Handle the number of subsets:
if isempty(numberOfSubsets)
    numberOfSubsets = 0;
end
if(isempty(subsetIndex))
    subsetIndex = 0;
end

if (numel(pixelSize_mm) ~= 3) && (imageSize_pixels(3) ~= 1)
    error('The image size (imageSize_pixels) and pixel size (pixelSize_mm) parameters must be a three-elements vector.');
end

% Generate a constant image:
image = ones(imageSize_pixels);

% Check if is an struct or the span value:
if isfield(structSizeSino, 'sinogramsPerSegment')
    if numel(structSizeSino.sinogramsPerSegment) == 1
        if structSizeSino.numZ == 1
            % sinogram 2d.
            if size(image,3) ~= 1
                error('Backprojecct: an image with multiple slices can not be backprojected into a sinogram 2d.');
            end
        else
            if size(image,3) ~= structSizeSino.sinogramsPerSegment(1)
                error('Backprojecct: the number of slices of the image is different to the number of rings of the output sinogram.');
            end
        end
    else
        % 3d sinogram
    end
elseif structSizeSino.numZ == 1
    % sinogram 2d.
    if size(image,3) ~= 1
        error('Backprojecct: an image with multiple slices can not be backprojected into a sinogram 2d.');
    else
        if size(image,3) ~= structSizeSino.numZ
            error('Backprojecct: the number of slices of the image is different to the number of rings of the output sinogram.');
        end
    end
end

% Write image in interfile:
filenameImage = [outputPath 'imageSample'];
interfilewrite(single(image), filenameImage, pixelSize_mm);

% constant sinogram:
sinogramFilename = [outputPath 'inputSinogram'];
% I replace writing a real sinogram for an empty sinogram to save space and
% time:
interfileWriteSino(single(sinogram), sinogramFilename, structSizeSino);

% Generate backprojected image:
filenameBackprojectionConfig = [outputPath 'backprojectSinogram.par'];
backprojectionFilename = [outputPath 'backprojectedImage'];
CreateBackprojectConfigFile(filenameBackprojectionConfig, [sinogramFilename '.h33'], [filenameImage '.h33'], scanner, scanner_parameters, backprojectionFilename, numberOfSubsets, subsetIndex, useGpu, numSamples, numAxialSamples);
[status, output_message] = system(['backproject ' filenameBackprojectionConfig]);
if status > 0
    error(output_message);
end
% Remove the input sinogram:
delete([sinogramFilename '.*']);

% Read the image:
image = interfileRead([backprojectionFilename '.h33']);
image(isnan(image)) = 0;