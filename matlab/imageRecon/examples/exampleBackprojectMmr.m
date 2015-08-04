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
apirlPath = '/home/mab15/workspace/apirl-code/trunk/';
apirlPath = '/workspaces/Martin/KCL/apirl-code/trunk/';
addpath(genpath([apirlPath pathBar 'matlab']));
setenv('PATH', [getenv('PATH') sepEnvironment apirlPath pathBar 'build' pathBar 'bin']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') sepEnvironment apirlPath pathBar 'build' pathBar 'bin']);
%% READ SINOGRAM
% Read the sinograms:
sinogramsPath = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/NEMA_IQ_20_02_2014/';
filenameUncompressedMmr = [sinogramsPath 'PET_ACQ_194_20150220154553-0uncomp.s'];
outFilenameIntfSinograms = [sinogramsPath 'NemaIq20_02_2014_ApirlIntf_Span1.s'];
[sinogramSpan1, delaySinogramSpan1, structSizeSino3dSpan1] = getIntfSinogramsFromUncompressedMmr(filenameUncompressedMmr, outFilenameIntfSinograms);
%% IMAGE SIZE
imageSize_pixels = [286 286 127];
pixelSize_mm = [2.08625 2.08625 2.03125];
% %% BACKPROJECT CPU
% outputPath = '/fast/NemaReconstruction/Backproject/';
% [image, pixelSize_mm] = BackprojectMmrSpan1(sinogramSpan1, imageSize_pixels, pixelSize_mm, outputPath, 0);
% figure;
% slice = 80;
% imshow(image(:,:,slice), [0 max(max(image(:,:,slice)))]);
%% BACKPROJECT GPU Span1
span = 1;
% outputPath = '/fast/NemaReconstruction/BackprojectCuda/';
% [image, pixelSize_mm] = BackprojectMmr(sinogramSpan1, imageSize_pixels, pixelSize_mm, outputPath, span, [], [], 1);
% figure;
% slice = 80;
% imshow(image(:,:,slice), [0 max(max(image(:,:,slice)))]);
%% BACKPROJECT SUBSET GPU
% numberOfSubsets = 21;
% subsetIndex = 5;
% outputPath = '/fast/NemaReconstruction/BackprojectSubsetCuda/';
% [image, pixelSize_mm] = BackprojectMmr(sinogramSpan1, imageSize_pixels, pixelSize_mm, outputPath, span, numberOfSubsets, subsetIndex, 1);
% figure;
% slice = 80;
% imshow(image(:,:,slice), [0 max(max(image(:,:,slice)))]);
%% BACKPROJECT 2D
sinogramFilename = '/home/martin/Project2D/projectedSinogram.h33';
sinogram = interfileReadSino(sinogramFilename);
imageSize_pixels = [344 344];
pixelSize_mm = [2.08625 2.08625];
outputPath = '/home/martin/Backproject2D/';
[image, pixelSize_mm] = BackprojectMmr2d(sinogram, imageSize_pixels, pixelSize_mm, outputPath, [], [], 0);
showSlices(image);
