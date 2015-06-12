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
addpath(genpath([apirlPath pathBar 'matlab']));
setenv('PATH', [getenv('PATH') sepEnvironment apirlPath pathBar 'build' pathBar 'bin']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') sepEnvironment apirlPath pathBar 'build' pathBar 'bin']);
%% RECONSTRUCTION
sinogramFilename = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/NEMA_IQ_20_02_2014/PET_ACQ_194_20150220154553-0uncomp.s.hdr';
normFilename = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/NEMA_IQ_20_02_2014/norm/Norm_20150210112413.n';
attMapBaseFilename = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/NEMA_IQ_20_02_2014/umap/PET_ACQ_194_20150220154553';
pixelSize_mm = [2.08625 2.08625 2.03125];

%% MLEM
%outputPath = 'E:\NemaReconstruction\testMlemCuda\';
outputPath = '/fast/NemaReconstruction/cumlem/';
numIterations = 40;
useGpu = 1;
volume = MlemMmr(sinogramFilename, normFilename, attMapBaseFilename, outputPath, pixelSize_mm, numIterations, useGpu);
%% MLEM WITH APIRL SINOGRAM AND REGISTERED ATTENUATION MAP
outputPath = '/fast/NemaReconstruction/cumlem_aprildata/';
sinogramFilename = '/fast/NemaReconstruction/cumlem/sinogram.h33';
attMapBaseFilename = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/NEMA_IQ_20_02_2014/umap/AttenMapCtManuallyRegistered.h33';
volume = MlemMmr(sinogramFilename, normFilename, attMapBaseFilename, outputPath, pixelSize_mm, numIterations, useGpu);


