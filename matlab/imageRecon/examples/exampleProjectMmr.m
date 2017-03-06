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
addpath(genpath([apirlPath filesep 'matlab']));
setenv('PATH', [getenv('PATH') pathsep apirlPath filesep 'build' filesep 'bin']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') pathsep apirlPath filesep 'build' filesep 'bin']);
%% READ IMAGE
fullFilename = '/home/mab15/workspace/KCL/Aboflazl/Martin/sino_e7tools/psf/nema.v.hdr';
fullFilename = '/workspaces/Martin/KCL/Biograph_mMr/Mediciones/BRAIN_PETMR/SINOGRAMS/PET_ACQ_68_20150610155347_ima_AC_000_000.v.hdr';
[image, refImage, bedPosition_mm, info]  = interfileReadSiemensImage(fullFilename); 
%fullFilename = '/media/mab15/DATA/Reconstructions/LineSource_2015_03_13/span1/reconImage_final.h33';
%[image, refImage] = interfileRead (fullFilename);
%fullFilename = '/media/martin/My Book/BackupWorkspace/KCL/Biograph_mMr/Mediciones/NEMA_IQ_20_02_2014/umap/AttenMapCtManuallyRegistered.h33';

pixelSize_mm = [refImage.PixelExtentInWorldY refImage.PixelExtentInWorldX refImage.PixelExtentInWorldZ];

% Defrise phantom:
[image refImage] = CreateDefrisePhantom(size(image), pixelSize_mm);


% %% PROJECT CPU
% outputPath = 'E:\NemaReconstruction\testProject\';
% [sinogram, structSizeSinogram] = ProjectMmrSpan1(image, pixelSize_mm, outputPath, 0);
% % Show one sinogram:
% figure;
% indiceSino = 1000;
% imshow(sinogram(:,:,indiceSino), [0 max(max(sinogram(:,:,indiceSino)))]);
%% PROJECT GPU SPAN 1
useGpu = 0;
span = 11;
outputPath = '/workspaces/Martin/KCL/AxialCompression/exampleProject/';
numTheta = 252; numR = 344; numRings = 32; maxAbsRingDiff = 60; rFov_mm = 594/2; zFov_mm = 258; 
structSizeSino3d = getSizeSino3dFromSpan(numR, numTheta, numRings, rFov_mm, zFov_mm, span, maxAbsRingDiff);
%outputPath = '/home/mab15/workspace/KCL/Aboflazl/Martin/sino_e7tools/psf/apirl_span1/';
tic
[sinogram, structSizeSinogram] = ProjectMmr(image, pixelSize_mm, outputPath, structSizeSino3d, [], [], useGpu);
toc
% Show one sinogram:
figure;
indiceSino = 30;
imshow(sinogram(:,:,indiceSino), [0 max(max(sinogram(:,:,indiceSino)))]);
% %% PROJECT GPU SPAN 11
% useGpu = 1;
% span = 11;
% outputPath = '/home/mab15/workspace/KCL/Aboflazl/Martin/sino_e7tools/psf/apirl_span11/';
% tic
% [sinogram, structSizeSinogram] = ProjectMmr(image, pixelSize_mm, outputPath, span, [], [], useGpu);
% toc
% % Show one sinogram:
% figure;
% indiceSino = 250;
% imshow(sinogram(:,:,indiceSino), [0 max(max(sinogram(:,:,indiceSino)))]);

% %% PROJECT SUBSET GPU SPAN 1
% numberOfSubsets = 21;
% subsetIndex = 1;
% % outputPath = 'E:\NemaReconstruction\testProjectCudaSubset\';
% outputPath = '/fast/Defrise/ProjectCudaSubset/';
% tic
% [sinogram, structSizeSinogram] = ProjectMmr(image, pixelSize_mm, outputPath, span, numberOfSubsets, subsetIndex, useGpu);
% toc
% % Show one sinogram:
% figure;
% indiceSino = 1000;
% imshow(sinogram(:,:,indiceSino), [0 max(max(sinogram(:,:,indiceSino)))]);
% %% PROJECT ALL SUBSET GPU SPAN 1
% span = 1;
% numberOfSubsets = 21;
% sinogramAllSubsets = zeros(size(sinogram));
% figure;
% indiceSino = 1000;
% for subsetIndex = 1 : numberOfSubsets
%     % outputPath = 'E:\NemaReconstruction\testProjectCudaSubset\';
%     outputPath = sprintf('/fast/Defrise/ProjectCudaSubset_%d/', subsetIndex);
%     [sinogram, structSizeSinogram] = ProjectMmr(image, pixelSize_mm, outputPath, span, numberOfSubsets, subsetIndex, useGpu);
%     sinogramAllSubsets = sinogramAllSubsets + sinogram;
% end
% % Show one sinogram:
% figure;
% indiceSino = 1000;
% imshow(sinogramAllSubsets(:,:,indiceSino), [0 max(max(sinogramAllSubsets(:,:,indiceSino)))]);
% outputPath = '/fast/Defrise/ProjectCudaSubset_SummSubsets/';
% % Write the input sinogram:
% interfileWriteSino(single(sinogramAllSubsets), [outputPath 'SumSinogram'], structSizeSinogram);

% % PROJECT DIRECT SINOGRAMS
% useGpu = 0;
% span = 1;
% %outputPath = 'E:\NemaReconstruction\testProjectCuda\';
% outputPath = '/fast/ProjectMulti2D/';
% [sinogram, structSizeSinogram] = ProjectMmr2d(image, pixelSize_mm, outputPath, [], [], useGpu);
% % Show one sinogram:
% showSlices(sinogram);
% % PROJECT DIRECT SINOGRAMS
% useGpu = 0;
% pixelSize_mm = [refImage.PixelExtentInWorldY refImage.PixelExtentInWorldX];
% %outputPath = 'E:\NemaReconstruction\testProjectCuda\';
% outputPath = '/fast/Project2D/';
% [sinogram, structSizeSinogram] = ProjectMmr2d(image(:,:,81), pixelSize_mm, outputPath, [], [], useGpu);
% % Show one sinogram:
% h = showSlices(sinogram);