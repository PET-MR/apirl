%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 06/05/2015
%  *********************************************************************
%  Example of how to use OsemMmrSpan1:
clear all 
close all

apirlPath = [fileparts(mfilename('fullpath')) filesep '..' filesep '..' filesep '..'];
%% CUDA PATH
cudaPath = '/usr/local/cuda/';
setenv('PATH', [getenv('PATH') pathsep cudaPath filesep 'bin']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') pathsep cudaPath filesep 'lib64']);
%% APIRL PATH
apirlPath = '/home/mab15/workspace/apirl-code/trunk/';
addpath(genpath([apirlPath filesep 'matlab']));
setenv('PATH', [getenv('PATH') pathsep apirlPath filesep 'build' filesep 'bin']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') pathsep apirlPath filesep 'build' filesep 'bin']);
%% READ SINOGRAM
% Read the sinograms:
% sinogramFilename = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/LineSources/Line_Source_sinograms/PET_ACQ_91_20150313115152-0uncomp.s.hdr';
% [sinogramSpan1, delayedSinograms, structSizeSino3dSpan1] = interfileReadSino(sinogramFilename);

sinogramFilename = '/fast/Defrise/exampleProject/projectedSinogram.h33';
[sinogramSpan1, delayedSinograms, structSizeSino3dSpan1] = interfileReadSino(sinogramFilename);
%% IMAGE SIZE
imageSize_pixels = [288 288 127];
pixelSize_mm = [2.08625 2.08625 2.03125];
% %% BACKPROJECT CPU
% outputPath = '/fast/NemaReconstruction/Backproject/';
% [image, pixelSize_mm] = BackprojectMmrSpan1(sinogramSpan1, imageSize_pixels, pixelSize_mm, outputPath, 0);
% figure;
% slice = 80;
% imshow(image(:,:,slice), [0 max(max(image(:,:,slice)))]);
%% SENSITIVITY GPU Span1
span = 1;
outputPath = '/fast/Defrise/Sensitivity/';
constSino = ones(size(sinogramSpan1));
[sensitivity, pixelSize_mm] = BackprojectMmr(constSino, imageSize_pixels, pixelSize_mm, outputPath, span, [], [], 1);
figure;
slice = 80;
imshow(sensitivity(:,:,slice), [0 max(max(sensitivity(:,:,slice)))]);
%% BACKPROJECT GPU Span1
span = 1;
outputPath = '/fast/Defrise/BackprojectCuda/';
[image, pixelSize_mm] = BackprojectMmr(sinogramSpan1, imageSize_pixels, pixelSize_mm, outputPath, span, [], [], 1);
figure;
slice = 80;
imshow(image(:,:,slice), [0 max(max(image(:,:,slice)))]);
% NORM TO SENSITIVTY:
normImage = image;
normImage(sensitivity~=0) = normImage(sensitivity~=0) ./ sensitivity(sensitivity~=0);
figure;
imshow(normImage(:,:,slice), [0 max(max(normImage(:,:,slice)))]);
%% BACKPROJECT SUBSET GPU
% numberOfSubsets = 21;
% subsetIndex = 5;
% outputPath = '/fast/NemaReconstruction/BackprojectSubsetCuda/';
% [image, pixelSize_mm] = BackprojectMmr(sinogramSpan1, imageSize_pixels, pixelSize_mm, outputPath, span, numberOfSubsets, subsetIndex, 1);
% figure;
% slice = 80;
% imshow(image(:,:,slice), [0 max(max(image(:,:,slice)))]);
% %% BACKPROJECT MULTI 2D
% sinogramFilename = '/fast/ProjectMulti2D/projectedSinogram.h33';
% sinogram = interfileReadSino(sinogramFilename);
% imageSize_pixels = [344 344];
% pixelSize_mm = [2.08625 2.08625];
% outputPath = '/fast/BackprojectMulti2D/';
% [image, pixelSize_mm] = BackprojectMmr2d(sinogram, imageSize_pixels, pixelSize_mm, outputPath, [], [], 1);
% showSlices(image);
% 
% %% BACKPROJECT 2D
% sinogramFilename = '/fast/Project2D/projectedSinogram.h33';
% sinogram = interfileReadSino(sinogramFilename);
% imageSize_pixels = [344 344];
% pixelSize_mm = [2.08625 2.08625];
% outputPath = '/fast/Backproject2D/';
% [image, pixelSize_mm] = BackprojectMmr2d(sinogram, imageSize_pixels, pixelSize_mm, outputPath, [], [], 1);
% showSlices(image);