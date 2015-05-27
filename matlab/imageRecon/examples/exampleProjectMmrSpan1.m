%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 06/05/2015
%  *********************************************************************
%  Example of how to use OsemMmrSpan1:
clear all 
close all
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
%% APIRL PATH
apirlPath = 'E:\apirl-code\trunk\';
addpath(genpath([apirlPath pathBar 'matlab']));
setenv('PATH', [getenv('PATH') sepEnvironment apirlPath pathBar 'build' pathBar 'bin']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') sepEnvironment apirlPath pathBar 'build' pathBar 'bin']);
%% READ IMAGE
fullFilename = 'E:\NemaReconstruction\umap\AttenMapCtManuallyRegistered.h33';
image = interfileRead(fullFilename); 
infoVolumeSpan11 = interfileinfo(fullFilename); 
pixelSize_mm = [infoVolumeSpan11.ScalingFactorMmPixel1 infoVolumeSpan11.ScalingFactorMmPixel2 infoVolumeSpan11.ScalingFactorMmPixel3];
%% PROJECT CPU
outputPath = 'E:\NemaReconstruction\testProject\';
[sinogram, structSizeSinogram] = ProjectMmrSpan1(image, pixelSize_mm, outputPath, 0);
%% PROJECT GPU
outputPath = 'E:\NemaReconstruction\testProjectCuda\';
[sinogram, structSizeSinogram] = ProjectMmrSpan1(image, pixelSize_mm, outputPath, 1);
