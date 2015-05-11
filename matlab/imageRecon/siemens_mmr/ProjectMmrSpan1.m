%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 06/05/2015
%  *********************************************************************
%  This function projects
%
% Examples:
%   [sinogram, structSizeSinogram] = ProjectMmrSpan1(image, pixelSize_mm, outputPath)

function [sinogram, structSizeSinogram] = ProjectMmrSpan1(image, pixelSize_mm, outputPath)

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

% Create output sample sinogram:
% Size of mMr Sinogram's
numTheta = 252; numR = 344; numRings = 64; maxAbsRingDiff = 60; rFov_mm = 594/2; zFov_mm = 258; span = 1;
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
projectionFilename = [outputPath 'projectedImage'];
CreateProjectConfigFileForMmr(filenameProjectionConfig, [filenameImage '.h33'], [sinogramSampleFilename '.h33'], projectionFilename);
status = system(['project ' filenameProjectionConfig])

% Read the projected sinogram:
fid = fopen([projectionFilename '.i33'], 'r');
numSinos = sum(structSizeSino3d.sinogramsPerSegment);
[sinogram, count] = fread(fid, structSizeSino3d.numTheta*structSizeSino3d.numR*numSinos, 'single=>single');
fclose(fid);
sinogram = reshape(sinogram, [structSizeSino3d.numR structSizeSino3d.numTheta numSinos]);
