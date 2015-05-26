%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 06/05/2015
%  *********************************************************************
%  Example of how to use OsemMmrSpan1:
clear all 
close all
%% APIRL PATH
apirlPath = 'E:\apirl-code\trunk\';

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
%% RECONSTRUCTION

sinogramFilename = 'E:\UncompressedInterfile\NEMA_IF\PET_ACQ_194_20150220154553-0uncomp.s';
normFilename = 'E:\workspace\KCL\Biograph_mMr\Normalization\NormFiles\Norm_20150210112413.n';
attMapBaseFilename = 'E:\UncompressedInterfile\NEMA_IF\umap\PET_ACQ_194_20150220154553';
outputPath = 'E:\Recontructions\NEMA\';

pixelSize_mm = [2.08625 2.08625 2.03125];
pixelSize_mm(1:2) = pixelSize_mm(1:2).*2;
%% OSEM
numSubsets = 21;
numIterations = 3;
volume = OsemMmrSpan1(sinogramFilename, normFilename, attMapBaseFilename, outputPath, pixelSize_mm, numSubsets, numIterations);
%% MLEM
numIterations = 60;
volume = MlemMmrSpan1(sinogramFilename, normFilename, attMapBaseFilename, outputPath, pixelSize_mm, numIterations);
