%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 19/02/2015
%  *********************************************************************
%  Creates the system matrix for the mlem algorithm
clear all
close all

%% GET THE NORM FACTORS TO GET EFFINCENCIES
normFile = '/home/mab15/workspace/STIR/KCL/STIR_mMR_KCL/IM/NORM.n';
componentFactors = readmMrComponentBasedNormalization(normFile, 0);
crystalEff = componentFactors{3};
crystalEffRing1 = crystalEff(:,1);
% This would be to include the gaps:
crystalEffRing1(1:9:end) = 0;
%%
% Size of mMr Sinogram's
numTheta = 252; numR = 344; numRings = 64; maxAbsRingDiff = 60; rFov_mm = 594/2; zFov_mm = 258; span = 1;
structSizeSino3d = getSizeSino3dFromSpan(numR, numTheta, numRings, rFov_mm, zFov_mm, span, maxAbsRingDiff);
numDetectors = 504;

[mapaDet1Ids, mapaDet2Ids] = createMmrDetectorsIdInSinogram();

% Image with as many rows as bins in the sinogram, and as many cols as
% detectors:
detector1SystemMatrix = zeros(numR*numTheta, numDetectors);
for i = 1 : numDetectors
    detector1SystemMatrix(:,i) = mapaDet1Ids(:) == i;
end

detector2SystemMatrix = zeros(numR*numTheta, numDetectors);
for i = 1 : numDetectors
    detector2SystemMatrix(:,i) = mapaDet2Ids(:) == i;
end

% Generate sinograms with system matrix:
sinoEfficienciesSystemMatrix = (detector1SystemMatrix * crystalEffRing1) .* (detector2SystemMatrix * crystalEffRing1);
sinoEfficienciesSystemMatrix = reshape(sinoEfficienciesSystemMatrix, numR, numTheta);
% Generate with function:
sinoEfficencies = createSinogram2dFromDetectorsEfficency(crystalEffRing1, structSizeSino3d, 1, 0);

numDiff = sum(sum(sinoEfficienciesSystemMatrix ~= sinoEfficencies));

disp(sprintf('Hay %d bins diferentes en los sinogramas.', numDiff));
%% TRANSPOSE TO GET THE EFFICENCIES FROM THE SINOGRAM
effDet1 = detector1SystemMatrix'*sinoEfficencies(:) ./ (sum(detector1SystemMatrix',2));
effDet1(isnan(effDet1)) = 0;
figure;
plot(effDet1);
title('Estimated efficencies from Tranpose of System Matrix 1');
effDet2 = detector2SystemMatrix'*sinoEfficencies(:) ./ (sum(detector2SystemMatrix',2));
effDet2(isnan(effDet2)) = 0;
figure;
plot(effDet2);
title('Estimated efficencies from Tranpose of System Matrix 2');

% Combination of both efficencies:
combEff = effDet1;
combEff(effDet1 == 0) = effDet2(effDet1 == 0);
combEff((effDet1 == 0)&(effDet2 == 0)) = (effDet1((effDet1 == 0)&(effDet2 == 0)) + effDet2((effDet1 == 0)&(effDet2 == 0))) / 2;
figure;
plot(1:numDetectors, crystalEffRing1, 1:numDetectors, combEff);

% Need to normalize, the efficencies are always normalized to the media:
meanEfficency = mean(combEff(combEff~=0))
combEffNorm = combEff / meanEfficency;
figure;
plot(1:numDetectors, crystalEffRing1, 1:numDetectors, combEffNorm);

%% TRANSPOSE TO GET THE EFFICENCIES FROM THE SINOGRAM
% Method 2 average normalizing over the sum:
effDet1 = detector1SystemMatrix'*sinoEfficencies(:);
figure;
plot(effDet1);
title('Estimated efficencies from Tranpose of System Matrix 1');
effDet2 = detector2SystemMatrix'*sinoEfficencies(:);
effDet2(isnan(effDet2)) = 0;
figure;
plot(effDet2);
title('Estimated efficencies from Tranpose of System Matrix 2');

% Combination of both efficencies:
combEff = effDet1 + effDet2;
combinedNormalization =  sum(detector1SystemMatrix',2) + sum(detector2SystemMatrix',2);
combEff = combEff / combinedNormalization;
figure;
plot(1:numDetectors, crystalEffRing1, 1:numDetectors, combEff);

% Need to normalize, the efficencies are always normalized to the media:
meanEfficency = mean(combEff(combEff~=0))
combEffNorm = combEff / meanEfficency;
figure;
plot(1:numDetectors, crystalEffRing1, 1:numDetectors, combEffNorm);
%% MATRIX VERSION
% Forward:
forwardSino = diag(detector1SystemMatrix * crystalEffRing1)*(crystalEffRing1'*detector2SystemMatrix');
%% MATRIX VERSION 2
forwardSino2 = detector1SystemMatrix*(crystalEffRing1*crystalEffRing1');