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
%% SET ENVIRONMENT AND MATLAB PATHS
addpath(genpath([apirlPath pathBar 'matlab']));
setenv('PATH', [getenv('PATH') sepEnvironment apirlPath pathBar 'build' pathBar 'bin']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') sepEnvironment apirlPath pathBar 'build' pathBar 'bin']);
%% RECONSTRUCTION
% sinogramFilename = 'E:\workspace\KCL\Biograph_mMr\Mediciones\NEMA_IQ_20_02_2014\PET_ACQ_194_20150220154553-0uncomp.s';
% normFilename = 'E:\workspace\KCL\Biograph_mMr\Mediciones\NEMA_IQ_20_02_2014\norm\Norm_20150210112413.n';
% attMapBaseFilename = 'E:\workspace\KCL\Biograph_mMr\Mediciones\NEMA_IQ_20_02_2014\umap\PET_ACQ_194_20150220154553';
sinogramFilename = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/BRAIN_PETMR/SINOGRAMS/PET_ACQ_68_20150610155347-0uncomp.s.hdr';
normFilename = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/BRAIN_PETMR/SINOGRAMS/norm/Norm_20150609084317.n';
attMapBaseFilename = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/BRAIN_PETMR/SINOGRAMS/PET_ACQ_68_20150610155347';
pixelSize_mm = [2.08625 2.08625 2.03125];
%% OSEM
outputPath = '/fast/BrainPhantom/cuosem/';
numSubsets = 21;
numIterations = 3;
optUseGpu = 1;
volume = OsemMmr(sinogramFilename, normFilename, attMapBaseFilename, outputPath, pixelSize_mm, numSubsets, numIterations, optUseGpu);
%% OSEM WITH APIRL SINOGRAM AND REGISTERED ATTENUATION MAP
outputPath = '/fast/NemaReconstruction/cuosem_aprildata/';
sinogramFilename = '/fast/NemaReconstruction/cumlem/sinogram.h33';
attMapBaseFilename = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/NEMA_IQ_20_02_2014/umap/AttenMapCtManuallyRegistered.h33';
numSubsets = 21;
numIterations = 3;
optUseGpu = 1;
volume = OsemMmr(sinogramFilename, normFilename, attMapBaseFilename, outputPath, pixelSize_mm, numSubsets, numIterations, optUseGpu);
