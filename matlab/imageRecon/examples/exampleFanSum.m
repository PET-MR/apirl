%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 09/03/2015
%  *********************************************************************
%  Test different fan sum methods.

clear all 
close all
%% PATHS FOR EXTERNAL FUNCTIONS AND RESULTS
addpath('/home/mab15/workspace/KCL/Biograph_mMr/mmr');
apirlPath = '/home/mab15/workspace/apirl-code/trunk/';
addpath(genpath([apirlPath '/matlab']));
setenv('PATH', [getenv('PATH') ':/home/mab15/workspace/apirl-code/trunk/build/bin']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') ':/home/mab15/workspace/apirl-code/trunk/build/bin']);
outputPath = '/home/mab15/workspace/KCL/Biograph_mMr/Randoms/';
mkdir(outputPath);
%setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') ':/usr/lib/x86_64-linux-gnu/']);
%% READING THE SINOGRAMS
disp('Read input sinogram...');
% Read the sinograms:
sinogramFilename = '/home/mab15/workspace/KCL/Biograph_mMr/mmr/5hr_ge68/cylinder_5hours.s.hdr';
%sinogramFilename = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/BRAIN_PETMR/SINOGRAMS/PET_ACQ_68_20150610155347-0uncomp.s.hdr';
[sinogram, delayedSinogram, structSizeSino3d] = interfileReadSino(sinogramFilename);

%% FANSUM
disp('Method 1');
tic
[crystalsCounts_1] = FansumMmr(sinogram, structSizeSino3d, 1);
toc
disp('Method 2');
tic
[crystalsCounts_2] = FansumMmr(sinogram, structSizeSino3d, 2);
toc
%% WITH SYSTEM MATRIX
disp('Creation System Matrix');
tic
[detector1SystemMatrix, detector2SystemMatrix] = createDetectorSystemMatrix3d(1, 0);
combinedNormalization =  sum(detector1SystemMatrix',2) + sum(detector2SystemMatrix',2);
combSystemMatrix = (detector1SystemMatrix+detector2SystemMatrix);
toc
disp('Fansum computation');
tic
crystalsCounts_3=double(sinogram(:))'*combSystemMatrix;
toc
%% BIN-DRIVEN METHOD
tic
[crystalsCounts_4] = FansumMmr(sinogram, structSizeSino3d, 3);
toc
%% TEST INDIVIDUAL CRYSTAL
tic
[crystalCounts] = FansumPerCrystalMmr(sinogram, structSizeSino3d, 105, 1)
toc

tic
[crystalCounts] = FansumPerCrystalMmr(sinogram, structSizeSino3d, 105, 3)
toc

tic
aux1=double(sinogram(:))'*detector1SystemMatrix(:,105);
toc