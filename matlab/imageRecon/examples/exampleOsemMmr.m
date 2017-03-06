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
%% STIR PATH
stirPath = '/usr/local/stir3.0/';
stirMatlabPath = '/home/mab15/workspace/KCL/apirl-kcl/trunk/stir/';
scriptsPath = [stirMatlabPath 'scripts/'];
%% SET ENVIRONMENT AND MATLAB PATHS
addpath(genpath([apirlPath filesep 'matlab']));
addpath(genpath(stirMatlabPath));
setenv('PATH', [getenv('PATH') pathsep apirlPath filesep 'build' filesep 'bin' pathsep stirPath filesep 'bin/']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') pathsep apirlPath filesep 'build' filesep 'bin' pathsep stirPath filesep 'lib/' ]);
%% RECONSTRUCTION
% sinogramFilename = 'E:\workspace\KCL\Biograph_mMr\Mediciones\NEMA_IQ_20_02_2014\PET_ACQ_194_20150220154553-0uncomp.s';
% normFilename = 'E:\workspace\KCL\Biograph_mMr\Mediciones\NEMA_IQ_20_02_2014\norm\Norm_20150210112413.n';
% attMapBaseFilename = 'E:\workspace\KCL\Biograph_mMr\Mediciones\NEMA_IQ_20_02_2014\umap\PET_ACQ_194_20150220154553';
sinogramFilename = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/BRAIN_PETMR/SINOGRAMS/PET_ACQ_68_20150610155347-0uncomp.s.hdr';
normFilename = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/BRAIN_PETMR/norm/Norm_20150609084317.n';
attMapBaseFilename = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/BRAIN_PETMR/SINOGRAMS/PET_ACQ_68_20150610155347';
pixelSize_mm = [2.08625 2.08625 2.03125];
%% OSEM
outputPath = '/fast/BrainPhantom/osem_scatter_randoms/';
numSubsets = 21;
numIterations = 3;
optUseGpu = 1;
span = 11;
[volume randoms scatter] = OsemMmr(sinogramFilename, span, normFilename, attMapBaseFilename, 1, 1, outputPath, pixelSize_mm, numSubsets, numIterations, optUseGpu, stirMatlabPath);

