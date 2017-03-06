%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 06/05/2015
%  *********************************************************************
%  Example of how to use OsemMmrSpan1:
clear all 
close all

apirlPath = [fileparts(mfilename('fullpath')) filesep '..' filesep '..' filesep '..'];
addpath(genpath([apirlPath filesep 'matlab']));
addpath(genpath([apirlPath filesep 'matlab']));
setenv('PATH', [getenv('PATH') pathsep apirlPath filesep 'build' filesep 'bin']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') pathsep apirlPath filesep 'build' filesep 'bin']);
%% RECONSTRUCTION
% sinogramFilename = 'E:\UncompressedInterfile\NEMA_IF\PET_ACQ_194_20150220154553-0uncomp.s.hdr';
% normFilename = 'E:\workspace\KCL\Biograph_mMr\Normalization\NormFiles\Norm_20150210112413.n';
% attMapBaseFilename = 'E:\UncompressedInterfile\NEMA_IF\umap\PET_ACQ_194_20150220154553';
sinogramFilename = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/NEMA_IQ_20_02_2014/PET_ACQ_194_20150220154553-0uncomp.s.hdr';
normFilename = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/NEMA_IQ_20_02_2014/norm/Norm_20150210112413.n';
attMapBaseFilename = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/NEMA_IQ_20_02_2014/umap/PET_ACQ_194_20150220154553';
pixelSize_mm = [2.08625 2.08625 2.03125];
%% MATLAB MLEM
outputPath = '/fast/NemaReconstruction/testMatlabMlemMmr/';
numIterations = 60;
saveInterval = 5;
useGpu = 1;
volume = MatlabMlemMmr(sinogramFilename, normFilename, attMapBaseFilename, outputPath, pixelSize_mm, numIterations, saveInterval, useGpu);



