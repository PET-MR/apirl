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
%% STIR PATH
stirPath = '/usr/local/stir3.0/';
stirMatlabPath = '/home/mab15/workspace/KCL/apirl-kcl/trunk/stir/';
scriptsPath = [stirMatlabPath 'scripts/'];
%% SET ENVIRONMENT AND MATLAB PATHS
addpath(genpath([apirlPath pathBar 'matlab']));
addpath(genpath(stirMatlabPath));
setenv('PATH', [getenv('PATH') sepEnvironment apirlPath pathBar 'build' pathBar 'bin' ':' stirPath pathBar 'bin/']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') sepEnvironment apirlPath pathBar 'build' pathBar 'bin' ':' stirPath pathBar 'lib/' ]);
%% GET THE NORM FACTORS TO GET EFFINCENCIES
normFile = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/BRAIN_PETMR/norm/Norm_20150609084317.n';
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
%% TEST OPERATOR AND GENERATE IMAGES FOR PRESENTATION
outputPath = '/home/mab15/workspace/KCL/xtal_efficiencies/testSystemMatrix/';
mkdir(outputPath);
useGpu = 1;
% Select one detector and project into sinogram:
indRing = 10;
pixelSize_mm = [2.08626 2.08626 2.03125];
imageSize_pixels = [344 344 127];
detectorsInsideRing = [1 10 68 121 189 243 322 458];
for detInsideRing = detectorsInsideRing
    indDet = detInsideRing + (indRing-1)*504;
    crystals = zeros(size(crystalEffAllRings));
    crystals(indDet) = 1;
    sinoDet1 = detector1SystemMatrix*double(crystals);
    sinoDet1 = reshape(sinoDet1, [numR numTheta sum(structSizeSino3d.sinogramsPerSegment)]);
    sinoDet2 = detector2SystemMatrix*double(crystals);
    sinoDet2 = reshape(sinoDet2, [numR numTheta sum(structSizeSino3d.sinogramsPerSegment)]);
    imshow(sinoDet1(:,:,indRing)');
    % Save for publication:
    fullFilename = [outputPath sprintf('DirectSino_Det1_%d_%d', detInsideRing, indRing)];
    frame = getframe(gca);
    imwrite(frame.cdata, [fullFilename '.png']);
    
    imshow(sinoDet2(:,:,indRing)');
    % Save for publication:
    fullFilename = [outputPath sprintf('DirectSino_Det2_%d_%d', detInsideRing, indRing)];
    frame = getframe(gca);
    imwrite(frame.cdata, [fullFilename '.png']);

        figure;
    subplot(1,2,1);
    imshow(sinoDet1(:,:,indRing)');
    title(sprintf('Detector 1: %d in ring %d', detInsideRing, indRing));
    subplot(1,2,2);
    imshow(sinoDet2(:,:,indRing)');
    title(sprintf('Detector 2: %d in ring %d', detInsideRing, indRing));
    % Save for publication:
    set(gcf,'PaperPositionMode','auto');    % Para que lo guarde en el tamaño modificado.
    fullFilename = [outputPath sprintf('DirectSino_Det_%d_%d', detInsideRing, indRing)];
    saveas(gca, [fullFilename], 'tif');
    saveas(gca, [fullFilename], 'epsc');
    
    % Backproject of detector id:
    outputBackprojectPath = [outputPath sprintf('Backproject_%d_%d/', detInsideRing, indRing)];
    [backprojImage, pixelSize_mm] = BackprojectMmr(sinoDet1+sinoDet2, imageSize_pixels, pixelSize_mm, outputBackprojectPath, structSizeSino3d.span, [],[], useGpu);
end
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