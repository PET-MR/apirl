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
%% READ SINOGRAM
% Read the sinograms:
sinogramsPath = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/NEMA_IQ_20_02_2014/';
filenameUncompressedMmr = [sinogramsPath 'PET_ACQ_194_20150220154553-0uncomp.s'];
outFilenameIntfSinograms = [sinogramsPath 'NemaIq20_02_2014_ApirlIntf_Span1.s'];
[sinogramSpan1, delaySinogramSpan1, structSizeSino3dSpan1] = getIntfSinogramsFromUncompressedMmr(filenameUncompressedMmr, outFilenameIntfSinograms);
%% IMAGE SIZE
imageSize_pixels = [286 286 127];
pixelSize_mm = [2.08625 2.08625 2.03125];
%% PROJECT CPU
outputPath = '/fast/NemaReconstruction/Backproject/';
[image, pixelSize_mm] = BackprojectMmrSpan1(sinogramSpan1, imageSize_pixels, pixelSize_mm, outputPath, 0);
%% PROJECT GPU
outputPath = '/fast/NemaReconstruction/BackprojectCuda/';
[image, pixelSize_mm] = BackprojectMmrSpan1(sinogramSpan1, imageSize_pixels, pixelSize_mm, outputPath, 1);
