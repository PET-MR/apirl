%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 19/02/2015
%  *********************************************************************
%  Creates the system matrix for the mlem algorithm for crystal
%  efficiencies. Goes from detectors to sinograms. This one is for he 3d
%  case.
clear all
close all
%% PATHS FOR EXTERNAL FUNCTIONS AND RESULTS
addpath('/home/mab15/workspace/KCL/Biograph_mMr/mmr');
apirlPath = '/home/mab15/workspace/apirl-code/trunk/';
addpath(genpath([apirlPath '/matlab']));
setenv('PATH', [getenv('PATH') ':/home/mab15/workspace/apirl-code/trunk/build/bin']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') ':/home/mab15/workspace/apirl-code/trunk/build/bin']);
%% GET THE NORM FACTORS TO GET EFFINCENCIES
normFile = '/home/mab15/workspace/STIR/KCL/STIR_mMR_KCL/IM/NORM.n';
componentFactors = readmMrComponentBasedNormalization(normFile, 0);
crystalEff = componentFactors{3};
% This would be to include the gaps:
crystalEff(1:9:end,:) = 0;
% Transform in it in a vector:
crystalEffAllRings = crystalEff(:);

%% SYSTEM MATRIX FOR SPAN1 SINOGRAMS
numTheta = 252; numR = 344; numRings = 64; maxAbsRingDiff = 60; rFov_mm = 594/2; zFov_mm = 258; span = 1;
structSizeSino3d = getSizeSino3dFromSpan(numR, numTheta, numRings, rFov_mm, zFov_mm, span, maxAbsRingDiff);
% [detector1SystemMatrix, detector2SystemMatrix] = createDetectorSystemMatrix3d(1, 0);
% save('/home/mab15/workspace/KCL/xtal_efficiencies/AnalyzeCrystalEfficienciesImpact/detector1SystemMatrixSpan1.mat', 'detector1SystemMatrix', '-v7.3')
% save('/home/mab15/workspace/KCL/xtal_efficiencies/AnalyzeCrystalEfficienciesImpact/detector2SystemMatrixSpan1.mat', 'detector2SystemMatrix', '-v7.3')
detector1SystemMatrix = load('/home/mab15/workspace/KCL/xtal_efficiencies/AnalyzeCrystalEfficienciesImpact/detector1SystemMatrixSpan1.mat');
detector1SystemMatrix = detector1SystemMatrix.detector1SystemMatrix;
detector2SystemMatrix = load('/home/mab15/workspace/KCL/xtal_efficiencies/AnalyzeCrystalEfficienciesImpact/detector2SystemMatrixSpan1.mat');
detector2SystemMatrix = detector2SystemMatrix.detector2SystemMatrix;
% Generate sinograms with system matrix:
sinoEfficienciesSystemMatrix = (detector1SystemMatrix * double(crystalEffAllRings)) .* (detector2SystemMatrix * double(crystalEffAllRings));
sinoEfficienciesSystemMatrix = reshape(sinoEfficienciesSystemMatrix, [numR numTheta sum(structSizeSino3d.sinogramsPerSegment)]);
% Generate with function:
sinoEfficencies = createSinogram3dFromDetectorsEfficency(double(crystalEff), structSizeSino3d, 0);

numDiff = sum(sum(sum(single(sinoEfficienciesSystemMatrix) ~= sinoEfficencies)));

disp(sprintf('Hay %d bins diferentes en los sinogramas.', numDiff));
% TRANSPOSE TO GET THE EFFICENCIES FROM THE SINOGRAM
% Method 2 average normalizing over the sum:
effDet1 = detector1SystemMatrix'*double(sinoEfficencies(:));
% Self normalized matrixes:
% detector1SystemMatrix_aux = double(detector1SystemMatrix);
% for i = 1 : size(detector1SystemMatrix,2)
%     detector1SystemMatrix_aux(:,i) = double(detector1SystemMatrix(:,i))./combinedNormalization(i)';
% end
% 
% detector2SystemMatrix_aux = double(detector2SystemMatrix);
% for i = 1 : size(detector2SystemMatrix,2)
%     detector2SystemMatrix_aux(:,i) = double(detector2SystemMatrix(:,i))./combinedNormalization(i)';
% end
% figure;
% plot(effDet1);
% title('Estimated efficencies from Tranpose of System Matrix 1');
effDet2 = detector2SystemMatrix'*double(sinoEfficencies(:));
effDet2(isnan(effDet2)) = 0;
% figure;
% plot(effDet2);
% title('Estimated efficencies from Tranpose of System Matrix 2');

% Combination of both efficencies:
combEff = effDet1 + effDet2;
combinedNormalization =  sum(detector1SystemMatrix',2) + sum(detector2SystemMatrix',2);
combEff = combEff ./ combinedNormalization;
% figure;
% plot(1:numDetectors, crystalEffAllRings, 1:numDetectors, combEff);

% Need to normalize, the efficencies are always normalized to the media:
meanEfficency = mean(combEff(combEff~=0));
combEffNorm = combEff ./ meanEfficency;
figure;
numDetectors = 504*64;
plot(1:numDetectors, crystalEffAllRings, 1:numDetectors, combEffNorm);
%% SPAN 11 SYSTEM MATRIX
numTheta = 252; numR = 344; numRings = 64; maxAbsRingDiff = 60; rFov_mm = 594/2; zFov_mm = 258; span = 11;
structSizeSino3d = getSizeSino3dFromSpan(numR, numTheta, numRings, rFov_mm, zFov_mm, span, maxAbsRingDiff);
[detector1SystemMatrix, detector2SystemMatrix] = createDetectorSystemMatrix3d(11, 0);
save('/home/mab15/workspace/KCL/xtal_efficiencies/AnalyzeCrystalEfficienciesImpact/detector1SystemMatrixSpan11.mat', 'detector1SystemMatrix', '-v7.3')
save('/home/mab15/workspace/KCL/xtal_efficiencies/AnalyzeCrystalEfficienciesImpact/detector2SystemMatrixSpan11.mat', 'detector2SystemMatrix', '-v7.3')
[detector1SystemMatrix_norm, detector2SystemMatrix_norm] = createDetectorSystemMatrix3d(11, 1);
save('/home/mab15/workspace/KCL/xtal_efficiencies/AnalyzeCrystalEfficienciesImpact/detector1SystemMatrixSpan11_norm.mat', 'detector1SystemMatrix', '-v7.3')
save('/home/mab15/workspace/KCL/xtal_efficiencies/AnalyzeCrystalEfficienciesImpact/detector2SystemMatrixSpan11_norm.mat', 'detector2SystemMatrix', '-v7.3')
detector1SystemMatrix = load('/home/mab15/workspace/KCL/xtal_efficiencies/AnalyzeCrystalEfficienciesImpact/detector1SystemMatrixSpan11.mat');
detector1SystemMatrix = detector1SystemMatrix.detector1SystemMatrix;
detector2SystemMatrix = load('/home/mab15/workspace/KCL/xtal_efficiencies/AnalyzeCrystalEfficienciesImpact/detector2SystemMatrixSpan11.mat');
detector2SystemMatrix = detector2SystemMatrix.detector2SystemMatrix;
% Generate sinograms with system matrix:
sinoEfficienciesSystemMatrix = (detector1SystemMatrix * double(crystalEffAllRings)) .* (detector2SystemMatrix * double(crystalEffAllRings));
sinoEfficienciesSystemMatrix = reshape(sinoEfficienciesSystemMatrix, [numR numTheta sum(structSizeSino3d.sinogramsPerSegment)]);
% Generate with function:
sinoEfficencies = createSinogram3dFromDetectorsEfficency(double(crystalEff), structSizeSino3d, 0);

numDiff = sum(sum(sum(single(sinoEfficienciesSystemMatrix) ~= sinoEfficencies)));

disp(sprintf('Hay %d bins diferentes en los sinogramas.', numDiff));
% TRANSPOSE TO GET THE EFFICENCIES FROM THE SINOGRAM
% Method 2 average normalizing over the sum:
effDet1 = detector1SystemMatrix'*double(sinoEfficencies(:));
% Self normalized matrixes:
% detector1SystemMatrix_aux = double(detector1SystemMatrix);
% for i = 1 : size(detector1SystemMatrix,2)
%     detector1SystemMatrix_aux(:,i) = double(detector1SystemMatrix(:,i))./combinedNormalization(i)';
% end
% 
% detector2SystemMatrix_aux = double(detector2SystemMatrix);
% for i = 1 : size(detector2SystemMatrix,2)
%     detector2SystemMatrix_aux(:,i) = double(detector2SystemMatrix(:,i))./combinedNormalization(i)';
% end
% figure;
% plot(effDet1);
% title('Estimated efficencies from Tranpose of System Matrix 1');
effDet2 = detector2SystemMatrix'*double(sinoEfficencies(:));
effDet2(isnan(effDet2)) = 0;
% figure;
% plot(effDet2);
% title('Estimated efficencies from Tranpose of System Matrix 2');

% Combination of both efficencies:
combEff = effDet1 + effDet2;
combinedNormalization =  sum(detector1SystemMatrix',2) + sum(detector2SystemMatrix',2);
combEff = combEff ./ combinedNormalization;
% figure;
% plot(1:numDetectors, crystalEffAllRings, 1:numDetectors, combEff);

% Need to normalize, the efficencies are always normalized to the media:
meanEfficency = mean(combEff(combEff~=0));
combEffNorm = combEff ./ meanEfficency;
figure;
plot(1:numDetectors, crystalEffAllRings, 1:numDetectors, combEffNorm);