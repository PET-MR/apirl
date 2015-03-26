
%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 11/03/2015
%  *********************************************************************
%  This scripts compares two volume images with different metrics.
clear all 
close all
%% PATHS FOR EXTERNAL FUNCTIONS AND RESULTS
addpath('/home/mab15/workspace/KCL/Biograph_mMr/mmr');
apirlPath = '/home/mab15/workspace/apirl-code/trunk/';
addpath(genpath([apirlPath '/matlab']));
setenv('PATH', [getenv('PATH') ':' apirlPath '/build/bin']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') ':' apirlPath '/build/bin']);
outputPath = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/NEMA_IQ_20_02_2014/ReconstructionEvaluation/';
mkdir(outputPath);
%setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') ':/usr/lib/x86_64-linux-gnu/']);
%% IMAGES PARAMETERS
numIterations = 60;
% Read interfile reconstructed image:
% Span 11 APIRL:
span11Path = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/NEMA_IQ_20_02_2014/Reconstruction/span11/';
filenameSpan11 = 'Nema_Mlem21_HR_iter';
infoVolumeSpan11 = interfileinfo([span11Path filenameSpan11 '_0.h33']); 

% Span 1 APIRL:
span1Path = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/NEMA_IQ_20_02_2014/Reconstruction/span1/';
filenameSpan1 = 'Nema_Mlem21_HR_iter';

% Stir
stirPath = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/NEMA_IQ_20_02_2014/Stir/';
filenameStir = 'pet_im_1subset';

%% LOAD PHANTOM IMAGE FOR MASKS
% We use the attenuation map:
attenPath = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/NEMA_IQ_20_02_2014/umap/';
filename = 'AttenMapCtManuallyRegistered.h33';
attenMap = interfileRead([attenPath filename]);
infoAttenMap = interfileinfo([attenPath filename]);
% Resample to the emission images space (centered axially):
coordXct = (-infoAttenMap.ScalingFactorMmPixel2*infoAttenMap.MatrixSize2/2+infoAttenMap.ScalingFactorMmPixel2/2) : infoAttenMap.ScalingFactorMmPixel2 : (infoAttenMap.ScalingFactorMmPixel2*infoAttenMap.MatrixSize2/2);
coordYct = (-infoAttenMap.ScalingFactorMmPixel1*infoAttenMap.MatrixSize1/2+infoAttenMap.ScalingFactorMmPixel1/2) : infoAttenMap.ScalingFactorMmPixel1 : (infoAttenMap.ScalingFactorMmPixel1*infoAttenMap.MatrixSize1/2);
coordZct = (-infoAttenMap.ScalingFactorMmPixel3*infoAttenMap.MatrixSize3/2+infoAttenMap.ScalingFactorMmPixel3/2) : infoAttenMap.ScalingFactorMmPixel3 : (infoAttenMap.ScalingFactorMmPixel3*infoAttenMap.MatrixSize3/2);
[Xct, Yct, Zct] = meshgrid(coordXct, coordYct, coordZct);
% Idem for the dixon attenuation map:
coordXpet = (-infoVolumeSpan11.ScalingFactorMmPixel2*infoVolumeSpan11.MatrixSize2/2 + infoVolumeSpan11.ScalingFactorMmPixel2/2) : infoVolumeSpan11.ScalingFactorMmPixel2 : (infoVolumeSpan11.ScalingFactorMmPixel2*infoVolumeSpan11.MatrixSize2/2);
coordYpet = (-infoVolumeSpan11.ScalingFactorMmPixel1*infoVolumeSpan11.MatrixSize1/2 + infoVolumeSpan11.ScalingFactorMmPixel1/2) : infoVolumeSpan11.ScalingFactorMmPixel1 : (infoVolumeSpan11.ScalingFactorMmPixel1*infoVolumeSpan11.MatrixSize1/2);
coordZpet = (-infoVolumeSpan11.ScalingFactorMmPixel3*infoVolumeSpan11.MatrixSize3/2 + infoVolumeSpan11.ScalingFactorMmPixel3/2) : infoVolumeSpan11.ScalingFactorMmPixel3 : (infoVolumeSpan11.ScalingFactorMmPixel3*infoVolumeSpan11.MatrixSize3/2);
[Xpet, Ypet, Zpet] = meshgrid(coordXpet, coordYpet, coordZpet);
% Interpolate the ct image to the mr coordinates:
attenMap_rescaled = interp3(Xct,Yct,Zct,attenMap,Xpet,Ypet,Zpet); 
attenMap_rescaled(isnan(attenMap_rescaled)) = 0;
attenMap_rescaled = imdilate(attenMap_rescaled, ones(5));
SE = strel('disk',8);
attenMap_rescaled = imerode(attenMap_rescaled, SE);
maskPhantom = (attenMap_rescaled > 0.09) & (attenMap_rescaled < 0.11);
% Remove bottom and top:
maskPhantom(:,:,1:14) = 0;
maskPhantom(:,:,118:end) = 0;
% Show slices of mask:
figure;
title('Mask for computing the values inside the Phantom');
imageMask = getImageFromSlices(maskPhantom, 12);
imshow(imageMask);
%% READ ALL IMAGES FOR EACH ITERATION AND COMPUTE PARAMETERS
centralSlice = 81;
sizePixel_mm = [infoVolumeSpan11.ScalingFactorMmPixel1 infoVolumeSpan11.ScalingFactorMmPixel2 infoVolumeSpan11.ScalingFactorMmPixel3];
center_mm = [2 2];
numSpheres = 6;
contrastRecoverySpan11 = zeros(numIterations, numSpheres);
contrastRecoverySpan1 = zeros(numIterations, numSpheres);
contrastRecoveryStir = zeros(numIterations, numSpheres);
desvioNormBackgroundSpan11  = zeros(numIterations, numSpheres);
desvioNormBackgroundSpan1  = zeros(numIterations, numSpheres);
desvioNormBackgroundStir  = zeros(numIterations, numSpheres);
centralSlices = zeros(infoVolumeSpan11.MatrixSize1, infoVolumeSpan11.MatrixSize2, numIterations, 3);
for i = 1 : numIterations
    fullFilename = [span11Path sprintf('%s_%d.h33', filenameSpan11, i-1)];
    reconVolumeSpan11 = interfileRead(fullFilename); 
    % Mean Value of Slice / Std In Slice. For the whole volume.
    aux2dArray = reshape(reconVolumeSpan11, [size(reconVolumeSpan11,1)*size(reconVolumeSpan11,2) size(reconVolumeSpan11,3)]);
    snrSpan11(i) = mean(aux2dArray(:))./std(aux2dArray(:));
    snrSpan11Masked(i) = mean(reconVolumeSpan11(maskPhantom))./std(reconVolumeSpan11(maskPhantom));
    [contrastRecoverySpan11(i,:), desvioBackgroundSpan11, desvioNormBackgroundSpan11(i,:), meanLungRoiSpan11, relativeLungErrorSpan11] = procImagesQualityPhantomColdSpheres(reconVolumeSpan11, sizePixel_mm, center_mm, centralSlice, 0);
    centralSlices(:,:,i,1) =  reconVolumeSpan11(:,:,centralSlice);
    
    fullFilename = [span1Path sprintf('%s_%d.h33', filenameSpan1, i-1)];
    reconVolumeSpan1 = interfileRead(fullFilename); 
    % Mean Value of Slice / Std In Slice. For the whole volume.
    aux2dArray = reshape(reconVolumeSpan1, [size(reconVolumeSpan1,1)*size(reconVolumeSpan1,2) size(reconVolumeSpan1,3)]);
    snrSpan1(i) = mean(aux2dArray(:))./std(aux2dArray(:));
    snrSpan1Masked(i) = mean(reconVolumeSpan1(maskPhantom))./std(reconVolumeSpan1(maskPhantom));
    [contrastRecoverySpan1(i,:), desvioBackgroundSpan1, desvioNormBackgroundSpan1(i,:), meanLungRoiSpan1, relativeLungErrorSpan1] = procImagesQualityPhantomColdSpheres(reconVolumeSpan1, sizePixel_mm, center_mm, centralSlice, 0);
    centralSlices(:,:,i,2) =  reconVolumeSpan1(:,:,centralSlice);
    
    fullFilename = [stirPath sprintf('%s_%d.hv', filenameStir, i)];
    reconVolumeStir = interfileRead(fullFilename); 
    % Mean Value of Slice / Std In Slice. For the whole volume.
    aux2dArray = reshape(reconVolumeStir, [size(reconVolumeStir,1)*size(reconVolumeStir,2) size(reconVolumeStir,3)]);
    snrStir(i) = mean(aux2dArray(:))./std(aux2dArray(:));
    snrStirMasked(i) = mean(reconVolumeStir(maskPhantom))./std(reconVolumeStir(maskPhantom));
    [contrastRecoveryStir(i,:), desvioBackgroundStir, desvioNormBackgroundStir(i,:), meanLungRoiStir, relativeLungErrorStir, radioEsferas_mm, centers_pixels] = procImagesQualityPhantomColdSpheres(reconVolumeStir, sizePixel_mm, center_mm, centralSlice, 0);
    centralSlices(:,:,i,3) =  reconVolumeStir(:,:,centralSlice);
end
%% PLOT RESULTS
figure;
set(gcf, 'Name', 'SNR');
set(gcf, 'Position', [50 50 1600 1200]);
plot([snrSpan11; snrSpan1; snrStir]')
xlabel('Iterations');
ylabel('SNR (mean/std)');
legend('Span 11', 'Span 1', 'Stir', 'Location', 'SouthEast');

figure;
set(gcf, 'Name', 'SNR Masked');
set(gcf, 'Position', [50 50 1600 1200]);
plot([snrSpan11Masked; snrSpan1Masked; snrStirMasked]')
xlabel('Iterations');
ylabel('SNR (mean/std)');
legend('Span 11', 'Span 1', 'Stir', 'Location', 'SouthEast');

% Recovery contrast of all spheres for each type of reconstruction:
figure;
set(gcf, 'Name', 'Recovery Contrast for each Reconstruction');
set(gcf, 'Position', [50 50 1000 1200]);
subplot(3,1,1);
plot(contrastRecoverySpan11);
xlabel('Iterations');
ylabel('Recovery Contrast [%]');
for i = 1 : numel(radioEsferas_mm)-1 % Lug not used
    legends{i} = sprintf('Sphere Radius %.2f\n', radioEsferas_mm(i));
end
legend(legends, 'Location', 'SouthEast');
subplot(3,1,2);
plot(contrastRecoverySpan1);
xlabel('Iterations');
ylabel('Recovery Contrast [%]');
legend(legends, 'Location', 'SouthEast');
subplot(3,1,3);
plot(contrastRecoveryStir);
xlabel('Iterations');
ylabel('Recovery Contrast [%]');
legend(legends, 'Location', 'SouthEast');

% Recovery contrast. Intercomparison for each sphere betweeen the
% reconstruction methods.
figure;
set(gcf, 'Name', 'Recovery Contrast for each Reconstruction');
set(gcf, 'Position', [50 50 1600 1200]);
for i = 1 : numel(radioEsferas_mm)-1 % Lug not used
    subplot(3,2,i);
    plot([contrastRecoverySpan11(:,i) contrastRecoverySpan1(:,i) contrastRecoveryStir(:,i)], 'LineWidth', 2);
    xlabel('Iterations');
    ylabel('Recovery Contrast [%]');
    title(sprintf('Sphere Radius %.2f\n', radioEsferas_mm(i)));
    legend('Span 11', 'Span 1', 'Stir', 'Location', 'SouthEast');
end

% Recovery contrast. Intercomparison for each sphere betweeen the
% reconstruction methods.
figure;
set(gcf, 'Name', 'Normalized Std Dev');
set(gcf, 'Position', [50 50 1600 1200]);
for i = 1 : numel(radioEsferas_mm)-1 % Lug not used
    subplot(3,2,i);
    plot([desvioNormBackgroundSpan11(:,i) desvioNormBackgroundSpan1(:,i) desvioNormBackgroundStir(:,i)], 'LineWidth', 2);
    xlabel('Iterations');
    ylabel('Norm Std Dev');
    title(sprintf('Sphere Radius %.2f\n', radioEsferas_mm(i)));
    legend('Span 11', 'Span 1', 'Stir', 'Location', 'SouthEast');
end
%% OVERRAL RESULT
figure;
set(gcf, 'Name', 'SNR');
set(gcf, 'Position', [50 50 1600 1200]);
i = 1;
plot(contrastRecoverySpan11(:,i), desvioNormBackgroundSpan11(:,i), contrastRecoverySpan1(:,i), desvioNormBackgroundSpan1(:,i), contrastRecoveryStir(:,i), desvioNormBackgroundStir(:,i));
xlabel('Recovery Contrast');
ylabel('Standard Deviation');
legend('Span 11', 'Span 1', 'Stir', 'Location', 'SouthEast');

figure;
set(gcf, 'Name', 'SNR');
set(gcf, 'Position', [50 50 1600 1200]);
i = 2;
plot(contrastRecoverySpan11(:,i), desvioNormBackgroundSpan11(:,i), contrastRecoverySpan1(:,i), desvioNormBackgroundSpan1(:,i), contrastRecoveryStir(:,i), desvioNormBackgroundStir(:,i));
xlabel('Recovery Contrast');
ylabel('Standard Deviation');
legend('Span 11', 'Span 1', 'Stir', 'Location', 'SouthEast');

figure;
set(gcf, 'Name', 'SNR');
set(gcf, 'Position', [50 50 1600 1200]);
i = 3;
plot(contrastRecoverySpan11(:,i), desvioNormBackgroundSpan11(:,i), contrastRecoverySpan1(:,i), desvioNormBackgroundSpan1(:,i), contrastRecoveryStir(:,i), desvioNormBackgroundStir(:,i));
xlabel('Recovery Contrast');
ylabel('Standard Deviation');
legend('Span 11', 'Span 1', 'Stir', 'Location', 'SouthEast');
%% PLOT SLICES
iterationsPerFigure = 10;
for i = 1 : ceil(numIterations / iterationsPerFigure)
    figure;
    set(gcf, 'Position', [50 50 1600 1200]);
    set(gcf, 'Name', 'Central Slice for Each Iteration');
    subplot(3,1,1);
    imshow(getImageFromSlices(centralSlices(:,:,iterationsPerFigure*(i-1)+1:iterationsPerFigure*i,1),iterationsPerFigure));
    colormap(hot);
    title(sprintf('Iterations %d to %d', iterationsPerFigure*(i-1)+1, iterationsPerFigure*i), 'FontWeight','Bold');
    ylabel('Span 11');
    
    subplot(3,1,2);
    imshow(getImageFromSlices(centralSlices(:,:,iterationsPerFigure*(i-1)+1:iterationsPerFigure*i,2),iterationsPerFigure));
    colormap(hot);
    ylabel('Span 1');
    
    subplot(3,1,3);
    imshow(getImageFromSlices(centralSlices(:,:,iterationsPerFigure*(i-1)+1:iterationsPerFigure*i,3),iterationsPerFigure));
    colormap(hot);
    ylabel('Stir');
end
%% PLOT SLICES REMOVING ODD VALUES
iterationsPerFigure = 10;
for i = 1 : ceil(numIterations / iterationsPerFigure)
    figure;
    set(gcf, 'Position', [50 50 1600 1200]);
    set(gcf, 'Name', 'Central Slice for Each Iteration with Border Noise Removed');
    subplot(3,1,1);
    aux = getImageFromSlices(centralSlices(:,:,iterationsPerFigure*(i-1)+1:iterationsPerFigure*i,1),iterationsPerFigure);
    meanValue = mean(mean(aux));
    aux(aux>(35*meanValue)) = meanValue;
    aux = aux ./ max(max(aux));
    imshow(aux);
    colormap(hot);
    title(sprintf('Iterations %d to %d', iterationsPerFigure*(i-1)+1, iterationsPerFigure*i), 'FontWeight','Bold');
    ylabel('Span 11');
    
    subplot(3,1,2);
    aux = getImageFromSlices(centralSlices(:,:,iterationsPerFigure*(i-1)+1:iterationsPerFigure*i,2),iterationsPerFigure);
    meanValue = mean(mean(aux));
    aux(aux>(35*meanValue)) = meanValue;
    aux = aux ./ max(max(aux));
    imshow(aux);
    colormap(hot);
    ylabel('Span 1');
    
    subplot(3,1,3);
    aux = getImageFromSlices(centralSlices(:,:,iterationsPerFigure*(i-1)+1:iterationsPerFigure*i,3),iterationsPerFigure);
    meanValue = mean(mean(aux));
    aux(aux>(35*meanValue)) = meanValue;
    aux = aux ./ max(max(aux));
    imshow(aux);
    colormap(hot);
    ylabel('Stir');
end