%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 06/05/2015
%  *********************************************************************
%  Example of how to use OsemMmrSpan1:
clear all 
close all
%% APIRL PATH
apirlPath = '/home/mab15/workspace/apirl-code/trunk/';
cudaPath = '/usr/local/cuda/';
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


%% SET ENVIRONMENT AND MATLAB PATHS
addpath(genpath([apirlPath pathBar 'matlab']));
setenv('PATH', [getenv('PATH') sepEnvironment apirlPath pathBar 'build' pathBar 'bin']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') sepEnvironment apirlPath pathBar 'build' pathBar 'bin']);
setenv('PATH', [getenv('PATH') sepEnvironment cudaPath pathBar 'bin']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') sepEnvironment cudaPath pathBar 'lib64']);
%% READ IMAGE
fullFilename = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/NEMA_IQ_20_02_2014/umap/AttenMapCtManuallyRegistered.h33';
image = interfileRead(fullFilename); 
infoVolumeSpan11 = interfileinfo(fullFilename); 
pixelSize_mm = [infoVolumeSpan11.ScalingFactorMmPixel1 infoVolumeSpan11.ScalingFactorMmPixel2 infoVolumeSpan11.ScalingFactorMmPixel3];
%% PROJECT CPU
outputPath = '/fast/NemaReconstruction/Project/';
[sinogram, structSizeSinogram] = ProjectMmrSpan1(image, pixelSize_mm, outputPath, 0);
%% PROJECT GPU
outputPath = '/fast/NemaReconstruction/ProjectCuda/';
[sinogram, structSizeSinogram] = ProjectMmrSpan1(image, pixelSize_mm, outputPath, 1);
