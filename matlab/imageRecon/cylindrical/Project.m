%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 15/05/2016
%  *********************************************************************
%  This function projects an image into an span 1 sinogram. It receives
%  also as a parameter the subset that it is wanted to be projeted. It must
%  be left empty or in zero for projecting the complete sinogram.
% The span of the sinogram is received as a parameter.
%  When a subset is projected, it returns a sinogram of the original size,
%  but only filled in the bins of the subset. This was desgined this way to
%  be more trnsparent for the user.
% Examples:
%   [sinogram, structSizeSinogram] = Project(image, pixelSize_mm, outputPath, scanner, scanner_parameters, structSizeSino3d_span, numberOfSubsets, subsetIndex, useGpu)

function [sinogram, structSizeSino, output_message] = Project(image, pixelSize_mm, outputPath, scanner, scanner_parameters, structSizeSino, numberOfSubsets, subsetIndex, useGpu, numSamples, numAxialSamples)

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

if nargin == 8
    useGpu = 0;
    numSamples = 1;
    numAxialSamples = 1;
elseif nargin == 9
    numSamples = 1;
    numAxialSamples = 1;
elseif nargin == 10
    numAxialSamples = 1;
elseif nargin < 6
    error('Invalid number of parameters: [sinogram, structSizeSinogram] = Project(image, pixelSize_mm, outputPath, scanner, scanner_parameters, structSizeSino3d_span, numberOfSubsets, subsetIndex, useGpu, numSamples, numAxialSamples)');
end
% Handle the number of subsets:
if isempty(numberOfSubsets)||(numberOfSubsets<=1) % 1 subset is the same to not using any subset.
    numberOfSubsets = 0;
end
if(isempty(subsetIndex))||(numberOfSubsets<=1)
    subsetIndex = 0;
end

% Check if is an struct or the span value:
if isfield(structSizeSino, 'sinogramsPerSegment')
    if (numel(structSizeSino.sinogramsPerSegment) == 1) && (structSizeSino.sinogramsPerSegment(1)==structSizeSino.numZ)
        numSinos = structSizeSino.numZ; % to be used later.
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
        numSinos = sum(structSizeSino.sinogramsPerSegment); % to be used later.
    end
else
    numSinos = structSizeSino.numZ; % to be used later.
    % sinogram 2d.
    if structSizeSino.numZ == 1
        % sinogram 2d.
        if size(image,3) ~= 1
            error('Backprojecct: an image with multiple slices can not be backprojected into a sinogram 2d.');
        end
    else
        if size(image,3) ~= structSizeSino.numZ
            error('Backprojecct: the number of slices of the image is different to the number of rings of the output sinogram.');
        end
    end
end

% Create output sample sinogram:
% empty sinogram:
% sinogram = ones(numR, numTheta, sum(structSizeSino3d.sinogramsPerSegment), 'single');
sinogramSampleFilename = [outputPath 'sinogramSample'];
interfileWriteSino(single([]), sinogramSampleFilename, structSizeSino);

% Write image in interfile:
filenameImage = [outputPath 'inputImage'];
interfilewrite(single(image), filenameImage, pixelSize_mm);


% Generate projecte sinogram:
filenameProjectionConfig = [outputPath 'projectPhantom.par'];
projectionFilename = [outputPath 'projectedSinogram'];
CreateProjectConfigFile(filenameProjectionConfig, [filenameImage '.h33'], [sinogramSampleFilename '.h33'],  scanner, scanner_parameters, projectionFilename, numberOfSubsets, subsetIndex, useGpu, numSamples, numAxialSamples);
[status, output_message] = system(['project ' filenameProjectionConfig]);
if status > 0
    error(output_message);
end
% Read the projected sinogram:
% if is a subset, get the new size:
if numberOfSubsets > 1
    structSizeSino3dSubset = structSizeSino;
    structSizeSino3dSubset.numTheta = ceil(structSizeSino.numTheta/numberOfSubsets);
    
    fid = fopen([projectionFilename '.i33'], 'r');
    
    [subset, count] = fread(fid, structSizeSino3dSubset.numTheta*structSizeSino3dSubset.numR*numSinos, 'single=>single');
    fclose(fid);
    subset = reshape(subset, [structSizeSino3dSubset.numR structSizeSino3dSubset.numTheta numSinos]);
    % Fille a sinogram of the original size
    sinogram = zeros(structSizeSino.numR, structSizeSino.numTheta, numSinos);
    sinogram(:,subsetIndex : numberOfSubsets : end, :) = subset;
else
    fid = fopen([projectionFilename '.i33'], 'r');
    [sinogram, count] = fread(fid, structSizeSino.numTheta*structSizeSino.numR*numSinos, 'single=>single');
    fclose(fid);
    sinogram = reshape(sinogram, [structSizeSino.numR structSizeSino.numTheta numSinos]);
end
sinogram(isnan(sinogram)) = 0;