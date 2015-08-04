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
apirlPath = '/workspaces/Martin/KCL/apirl-code/trunk/';
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
%% STIR PATH
stirPath = '/usr/local/stir3.0/';
stirMatlabPath = '/home/mab15/workspace/KCL/apirl-kcl/trunk/stir/';
scriptsPath = [stirMatlabPath 'scripts/'];
%% SET ENVIRONMENT AND MATLAB PATHS
addpath(genpath([apirlPath pathBar 'matlab']));
setenv('PATH', [getenv('PATH') sepEnvironment apirlPath pathBar 'build' pathBar 'bin']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') sepEnvironment apirlPath pathBar 'build' pathBar 'bin']);
%% RECONSTRUCTION
% sinogramFilename = 'E:\UncompressedInterfile\NEMA_IF\PET_ACQ_194_20150220154553-0uncomp.s.hdr';
% normFilename = 'E:\workspace\KCL\Biograph_mMr\Normalization\NormFiles\Norm_20150210112413.n';
% attMapBaseFilename = 'E:\UncompressedInterfile\NEMA_IF\umap\PET_ACQ_194_20150220154553';
sinogramFilename = '/media/martin/My Book/BackupWorkspace/KCL/Biograph_mMr/Mediciones/BRAIN_PETMR/SINOGRAMS/PET_ACQ_68_20150610155347-0uncomp.s.hdr';
normFilename = '/media/martin/My Book/BackupWorkspace/KCL/Biograph_mMr/Mediciones/BRAIN_PETMR/norm/Norm_20150609084317.n';
attMapBaseFilename = '/media/martin/My Book/BackupWorkspace/KCL/Biograph_mMr/Mediciones/BRAIN_PETMR/SINOGRAMS/PET_ACQ_68_20150610155347';
pixelSize_mm = [2.08625 2.08625 2.03125];
%% MATLAB MLEM
outputPath = '/workspaces/Martin/KCL/Reconstructions/NEMA/matlab_osem_no_scatter_randoms/';
numIterations = 3;
numSubsets = 21;
saveInterval = 3;
useGpu = 0;
span = 11;
correctRandoms = 0;
correctScatter = 0;
[volume randoms scatter] = MatlabOsemMmr(sinogramFilename, span, normFilename, attMapBaseFilename, correctRandoms, correctScatter, outputPath, pixelSize_mm, numSubsets, numIterations, saveInterval, useGpu, stirMatlabPath);



