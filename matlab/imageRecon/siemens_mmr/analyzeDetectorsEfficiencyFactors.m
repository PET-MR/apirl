%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 16/02/2015
%  *********************************************************************
%  This scripts analyze the axial factors. It comapres the one in the .n
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
numCrystalsPerRing = size(componentFactors{1}{3},1);
numRings = size(componentFactors{1}{3},2);
%% GET CRYSTAL EFFICIENCIES
% Get the crystal efficiencies:
crystalEff = zeros(size(componentFactors{1}{3},1), size(componentFactors{1}{3},2), numel(normFiles));   % A 3d maitrx: crystal element, ring, norm file
for i = 1 : numel(normFiles)
    crystalEff(:,:,i) = componentFactors{i}{3};
end;

%% STATS OF THE CRYSTAL EFFICIENCIES
% Global variations in a normalization file:
for i = 1 : numel(normFiles)
    devGlobal(i) = std2(crystalEff(:,:,i));
    meanGlobal(i) = mean2(crystalEff(:,:,i));
end
figure;
plot(devGlobal);
title('Standard Deviation for Each Normalization File');
ylabel('Std Dev of Efficiency Factors');
xlabel('Norm File');

% Get std dev for each pixel position:
devCrystalEff = std(crystalEff,0,3);
meanCrystalEff = mean(crystalEff,3);
% Plot for each rings:
for i = 1 : 9 : numRings
    figure;
    h=errorbar(meanCrystalEff(:,i),devCrystalEff(:,i));
    title(sprintf('Standard Deviation of Crystal Efficiencies in Ring %d during 1 1/2 year', i));
    xlim([0 numCrystalsPerRing]);
    ylabel('Efficiency Factor');
    xlabel('Crystal Element');
    set(gcf, 'position', [1 25 1920 1069]);
end
%% GET SOME EXAMPLES
% Maximum deviation in ring:
[maxDevValue, maxDevPos] = max(devCrystalEff);
figure;
plot(permute(crystalEff(maxDevPos(1),1,:), [3 1 2]));
title(sprintf('MAX EXAMPLE: Variation of Crystal %d in ring 1 in 1 1/2 year', maxDevPos(1)));
ylabel('Efficiency Factor');
xlabel('Crystal Element');
set(gcf, 'position', [1 25 1920 1069]);

% Minimum deviation in ring:
[minDevValue, minDevPos] = min(devCrystalEff);
figure;
plot(permute(crystalEff(minDevPos(1),1,:), [3 1 2]));
title(sprintf('MIN EXAMPLE: Variation of Crystal %d in ring 1 in 1 1/2 year', minDevPos(1)));

%% REMOVE NORMALIZATION FILES WITH BAD VALUES
% Put a threshold to consider a failed norm file or failed detector:
threshold = 0.5;
[c, r, n] = ind2sub([size(crystalEff,1), size(crystalEff,2), size(crystalEff,3)],find((crystalEff < (1-threshold)) | (crystalEff > (1+threshold))));
crystalEff(:,:,n) = [];
% Recompute standard deviations:
% Global variations in a normalization file:
devGlobal = zeros(1, size(crystalEff,3));
meanGlobal = zeros(1, size(crystalEff,3));
for i = 1 : size(crystalEff,3)
    devGlobal(i) = std2(crystalEff(:,:,i));
    meanGlobal(i) = mean2(crystalEff(:,:,i));
end
figure;
plot(devGlobal);
title('Standard Deviation for Each Normalization File');
ylabel('Std Dev of Efficiency Factors');
xlabel('Norm File');
set(gcf, 'position', [1 25 1920 1069]);

% Update std values:
devCrystalEff = std(crystalEff,0,3);
meanCrystalEff = mean(crystalEff,3);
% Plot for each rings:
for i = 1 : 9 : numRings
    figure;
    h=errorbar(meanCrystalEff(:,i),devCrystalEff(:,i));
    title(sprintf('Standard Deviation of Crystal Efficiencies in Ring %d during 1 1/2 year (without failed efficencies)', i));
    xlim([0 numCrystalsPerRing]);
    ylabel('Efficiency Factor');
    xlabel('Crystal Element');
    set(gcf, 'position', [1 25 1920 1069]);
end
% Maximum deviation in ring:
[maxDevValue, maxDevPos] = max(devCrystalEff);
figure;
plot(permute(crystalEff(maxDevPos(1),1,:), [3 1 2]));
title(sprintf('MAX EXAMPLE: Variation of Crystal %d in ring 1 in 1 1/2 year', maxDevPos(1)));
ylabel('Efficiency Factor');
xlabel('Crystal Element');
set(gcf, 'position', [1 25 1920 1069]);