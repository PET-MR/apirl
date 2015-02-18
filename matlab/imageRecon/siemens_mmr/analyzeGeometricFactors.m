%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 16/02/2015
%  *********************************************************************
%  This scripts analyze the geometric factors. It comapres the one in the .n
%  file
clear all 
close all
%% PATHS FOR EXTERNAL FUNCTIONS AND RESULTS
addpath('/home/mab15/workspace/Biograph_mMr/mmr');
apirlPath = '/home/mab15/workspace/apirl-code/trunk/';
addpath(genpath([apirlPath '/matlab']));
setenv('PATH', [getenv('PATH') ':/home/mab15/workspace/apirl-code/trunk/build/bin']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') ':/home/mab15/workspace/apirl-code/trunk/build//bin']);
normPath = '/home/mab15/workspace/KCL/Biograph_mMr/Normalization/NormFiles/';
%setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') ':/usr/lib/x86_64-linux-gnu/']);
%% GET ALL THE NORM FILES IN THE PATH AND READ THEM
normFiles = dir([normPath '*.n']);
for i = 1 : numel(normFiles)
    componentFactors{i}  = readmMrComponentBasedNormalization([normPath normFiles(i).name], 0);
end

%% GENERATE THE STRUCT OF A SINO3D SPAN 11
% In that structure I have the amount of sinograms axially compressed per
% stored sinogram.
numR = 344; numTheta = 252; numZ = 64; maxAbsRingDiff = 60; rFov_mm = 594/2; zFov_mm = 258;
structSizeSino3dSpan11 = getSizeSino3dFromSpan(numR, numTheta, numZ, ...
    rFov_mm, zFov_mm, 11, maxAbsRingDiff);
%% GET GEOMETRIC FACTORS AND CRYSTAL INTERFERENCES
% Get the crystal efficiencies:
geometricFactor = zeros(size(componentFactors{1}{1},1), size(componentFactors{1}{1},2), numel(normFiles));   % A 3d maitrx
crystalInterfFactor = zeros(size(componentFactors{1}{2},1), size(componentFactors{1}{2},2), numel(normFiles));   % A 3d maitrx
for i = 1 : numel(normFiles)
    geometricFactor(:,:,i) = componentFactors{i}{1};
    crystalInterfFactor(:,:,i) = componentFactors{i}{2};
end

%% GEOMETRIC FACTORS
% Check if they vary with the time:
geometricRef = geometricFactor(:,:,1);
numDiffFactors = 0;
for i = 1 : numel(normFiles)
    diff = geometricFactor(:,:,i) ~= geometricRef;
    if(sum(diff)>0)
        disp(sprintf('The geometric factors for the %s file was different from %s.', normFiles(i).name, normFiles(1).name));
        numDiffFactors = numDiffFactors +1;
    end
end
if (numDiffFactors == 0)
    disp('The Geometric Factors are the same for all the norm files.');
end

%% CRYSTAL INTERFERENCE FACTORS
% Check if they vary with the time:
crystalInterfRef = crystalInterfFactor(:,:,1);
numDiffFactors = 0;
for i = 1 : numel(normFiles)
    diff = crystalInterfFactor(:,:,i) ~= crystalInterfRef;
    if(sum(diff) > 0)
        disp(sprintf('The crystal interference factors for the %s file was different from %s.', normFiles(i).name, normFiles(1).name));
    end
end

if (numDiffFactors  == 0)
    disp('The Crystal Interference Factors are the same for all the norm files.');
end