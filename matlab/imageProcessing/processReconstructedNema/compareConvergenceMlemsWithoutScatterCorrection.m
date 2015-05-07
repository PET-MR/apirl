
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

centralSlice = 81;
sizePixel_mm = [infoVolumeSpan11.ScalingFactorMmPixel1 infoVolumeSpan11.ScalingFactorMmPixel2 infoVolumeSpan11.ScalingFactorMmPixel3];
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
% attenMap_rescaled = imdilate(attenMap_rescaled, ones(5));
% SE = strel('disk',8);
% attenMap_rescaled = imerode(attenMap_rescaled, SE);
maskPhantom = (attenMap_rescaled > 0.09) & (attenMap_rescaled < 0.11);
% Remove bottom and top:
maskPhantom(:,:,1:14) = 0;
maskPhantom(:,:,118:end) = 0;
% Create an eroded mask:
SE = strel('disk',3);
maskPhantomEroded = imerode(maskPhantom, SE);

% Show slices of mask:
figure;
title('Mask for computing the values inside the Phantom');
imageMask = getImageFromSlices(maskPhantom, 12);
imshow(imageMask);
%% CREATE MEAN PROFILES
% Create spatially variant mean values, to be able to compute snr despite the scatter present in the image (and the subsquent bias).
% I plot each step of the generation of this mean spatially variant values.
rowProfile = round(infoVolumeSpan11.MatrixSize1/2);
% Read first and last images to analyze the low frequency estimate:
fullFilename = [span11Path sprintf('%s_0.h33', filenameSpan11)];
reconVolumeSpan11_0 = interfileRead(fullFilename); 
fullFilename = [span11Path sprintf('%s_5.h33', filenameSpan11)];
reconVolumeSpan11_5 = interfileRead(fullFilename); 
fullFilename = [span11Path sprintf('%s_10.h33', filenameSpan11)];
reconVolumeSpan11_10 = interfileRead(fullFilename); 
fullFilename = [span11Path sprintf('%s_59.h33', filenameSpan11)];
reconVolumeSpan11_60 = interfileRead(fullFilename); 

% Mask scaled to mean value of reconstructed image:
maskPhantomScaled = maskPhantom .* mean(reconVolumeSpan11_60(maskPhantom));
maskPhantomErodedScaled = maskPhantomEroded .* mean(reconVolumeSpan11_60(maskPhantom));
figure;
set(gcf, 'Name', 'Generation of  Spatially Variant Mean Value');
set(gcf, 'Position', [50 50 1600 1200]);
subplot('position',[0.1 0.55 0.40 0.40]);
plot(coordXpet, [reconVolumeSpan11_60(rowProfile,:,81)' maskPhantomScaled(rowProfile,:,81)' maskPhantomErodedScaled(rowProfile,:,81)'], 'LineWidth', 2);
legend('Reconstructed Image', 'Mask scaled to Mean Value of Recon Image', 'Eroded Mask scaled to Mean Value of Recon Image');
title('Step 1 - Reconstructed Image and Mask')
ylim([0 0.1]);
ylabel('Image Value');
xlabel('X [mm]')
% Filter for SNR:
filter = fspecial('gaussian',[25 25],11);
for i = 1 : size(reconVolumeSpan11_60,3)
    reconVolumeSpan11_60_filtered(:,:,i) = imfilter(maskPhantom(:,:,i).*reconVolumeSpan11_60(:,:,i), filter);
    maskPhantom_filtered(:,:,i) =  imfilter(single(maskPhantom(:,:,i)), filter);
end
maskPhantomScaled_filtered = maskPhantom_filtered.* mean(reconVolumeSpan11_60(maskPhantom));
subplot('position',[0.55 0.55 0.40 0.40]);
plot(coordXpet, [reconVolumeSpan11_60_filtered(rowProfile,:,81)' maskPhantomScaled_filtered(rowProfile,:,81)'], 'LineWidth', 2);
legend('Filtered Masked Reconstructed Image', 'Filtered Mask scaled to Mean Value of Recon Image', 'Location', 'SouthEast');
title('Step 2 - Filtering Image and Mask')
ticklabels = get(gca, 'XtickLabel');
set(gca, 'XtickLabel', ticklabels, 'FontSize',10);
ylabel('Image Value');
xlabel('X [mm]')

% Filtered normalized to mask Phantom: 
reconVolumeSpan11_60_filtered_masked = zeros(size(maskPhantom));
reconVolumeSpan11_60_filtered_masked(maskPhantom_filtered ~= 0) = reconVolumeSpan11_60_filtered(maskPhantom_filtered ~= 0) ./ maskPhantomScaled_filtered(maskPhantom_filtered ~= 0).* mean(reconVolumeSpan11_60(maskPhantom));
subplot('position',[0.1 0.08 0.40 0.40]);
plot(coordXpet, [reconVolumeSpan11_60_filtered(rowProfile,:,81)' maskPhantomScaled_filtered(rowProfile,:,81)' reconVolumeSpan11_60_filtered_masked(rowProfile,:,81)'], 'LineWidth', 2);
ylim([0 mean(reconVolumeSpan11_60_filtered(maskPhantom))*2]);
legend('1 - Filtered Masked Reconstructed Image', '2 - Filtered Mask scaled to Mean Value of Recon Image', 'Ratio between 1 and 2', 'Location', 'SouthEast', 'FontWeight','bold');
title('Step 3 - Dividing Filtered Image with the Filtered Mask')
% ticklabels = get(gca, 'XtickLabel');
% set(gca, 'XtickLabel', ticklabels, 'FontSize',10);
ylabel('Image Value');
xlabel('X [mm]')


% Apply original mask:
reconVolumeSpan11_60_filtered_masked = reconVolumeSpan11_60_filtered_masked .* maskPhantomEroded;
subplot('position',[0.55 0.08 0.40 0.40]);
plot(coordXpet, [reconVolumeSpan11_60(rowProfile,:,81)' reconVolumeSpan11_60_filtered_masked(rowProfile,:,81)'], 'LineWidth', 2);
legend('Reconstructed Image', 'Final Result: Spatillay Variant Mean Value', 'Location', 'NorthEast');
title('Step 4 - Apply Mask')
ylim([0 0.1]);
ticklabels = get(gca, 'XtickLabel');
set(gca, 'XtickLabel', ticklabels, 'FontSize',10);
ylabel('Image Value');
xlabel('X [mm]')

% Save for publication:
fullFilename = [outputPath 'SpatiallyVariantMeanValue'];
saveas(gca, [fullFilename], 'tif');
set(gcf,'PaperPositionMode','auto');    % Para que lo guarde en el tamaño modificado.
frame = getframe(gca);
imwrite(frame.cdata, [fullFilename '.png']);
saveas(gca, [fullFilename], 'epsc');

% Without eroded mask:
subplot('position',[0.1 0.55 0.40 0.40]);
plot(coordXpet, [reconVolumeSpan11_60(rowProfile,:,81)' maskPhantomScaled(rowProfile,:,81)' ], 'LineWidth', 2);
legend('Reconstructed Image', 'Mask scaled to Mean Value of Recon Image');
title('Step 1 - Reconstructed Image and Mask')
ticklabels = get(gca, 'XtickLabel');
set(gca, 'XtickLabel', ticklabels, 'FontSize',10);
ylim([0 0.1]);
ylabel('Image Value');
xlabel('X [mm]')

% Save for publication:
fullFilename = [outputPath 'SpatiallyVariantMeanValue_withoutErodedMask'];
saveas(gca, [fullFilename], 'tif');
set(gcf,'PaperPositionMode','auto');    % Para que lo guarde en el tamaño modificado.
frame = getframe(gca);
imwrite(frame.cdata, [fullFilename '.png']);
saveas(gca, [fullFilename], 'epsc');

interfilewrite(reconVolumeSpan11_60_filtered_masked, [outputPath 'meanValueImage'], sizePixel_mm);  
% [snr_5 snr_per_slice_5 meanValue_5 mean_per_slice_5 stdValue_5 std_per_slice_5] = getSnrWithSpatiallyVariantMean(reconVolumeSpan11_5, maskPhantom);
% [snr_10 snr_per_slice_10 meanValue_10 mean_per_slice_10 stdValue_10 std_per_slice_10] = getSnrWithSpatiallyVariantMean(reconVolumeSpan11_10, maskPhantom);
[snr_60 snr_per_slice_60 meanValue_60 mean_per_slice_60 stdValue_60 std_per_slice_60] = getSnrWithSpatiallyVariantMean(reconVolumeSpan11_60, maskPhantom);
%% EXAMPLE OF CONTRAST RECOVERY
% Example to show the ROIs used:
center_mm = [2 2];
[contrastRecovery, desvioBackground, desvioNormBackground, radioEsferas_mm, centers_pixels] = procImagesQualityPhantomColdSpheresWithoutSC(reconVolumeSpan11_60, sizePixel_mm, center_mm, centralSlice, 1);
%% READ ALL IMAGES FOR EACH ITERATION AND COMPUTE PARAMETERS
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
    [snrSpan11(i) snrPerSliceSpan11(i,:) meanSpan11(i) meanPerSliceSpan11(i,:) stdSpan11(i) stdPerSliceSpan11(i,:)] = getSnrWithSpatiallyVariantMean(reconVolumeSpan11, maskPhantom);
    [contrastRecoverySpan11(i,:), desvioBackgroundSpan11(i,:), desvioNormBackgroundSpan11(i,:) radioEsferas_mm, centers_pixels] = procImagesQualityPhantomColdSpheresWithoutSC(reconVolumeSpan11, sizePixel_mm, center_mm, centralSlice, 0);
    centralSlices(:,:,i,1) =  reconVolumeSpan11(:,:,centralSlice);

    fullFilename = [span1Path sprintf('%s_%d.h33', filenameSpan1, i-1)];
    reconVolumeSpan1 = interfileRead(fullFilename); 
    % Mean Value of Slice / Std In Slice. For the whole volume.
    [snrSpan1(i) snrPerSliceSpan1(i,:) meanSpan1(i) meanPerSliceSpan1(i,:) stdSpan1(i) stdPerSliceSpan1(i,:)] = getSnrWithSpatiallyVariantMean(reconVolumeSpan1, maskPhantom);
    [contrastRecoverySpan1(i,:), desvioBackgroundSpan1(i,:), desvioNormBackgroundSpan1(i,:)  radioEsferas_mm, centers_pixels] = procImagesQualityPhantomColdSpheresWithoutSC(reconVolumeSpan1, sizePixel_mm, center_mm, centralSlice, 0);
    centralSlices(:,:,i,2) =  reconVolumeSpan1(:,:,centralSlice);
    
    fullFilename = [stirPath sprintf('%s_%d.hv', filenameStir, i)];
    reconVolumeStir = interfileRead(fullFilename); 
    % Mean Value of Slice / Std In Slice. For the whole volume.
    [snrStir(i) snrPerSliceStir(i,:) meanStir(i) meanPerSliceStir(i,:) stdStir(i) stdPerSliceStir(i,:)] = getSnrWithSpatiallyVariantMean(reconVolumeStir, maskPhantom);
    [contrastRecoveryStir(i,:), desvioBackgroundStir(i,:), desvioNormBackgroundStir(i,:) radioEsferas_mm, centers_pixels] = procImagesQualityPhantomColdSpheresWithoutSC(reconVolumeStir, sizePixel_mm, center_mm, centralSlice, 0);
    centralSlices(:,:,i,3) =  reconVolumeStir(:,:,centralSlice);
end

%% PLOT RESULTS
figure;
set(gcf, 'Name', 'STD DEV');
set(gcf, 'Position', [50 50 1600 1200]);
plot([stdSpan11; stdSpan1; stdStir]', 'LineWidth', 2)
xlabel('Iterations');
ylabel('Std Dev');
legend('Span 11', 'Span 1', 'Stir', 'Location', 'SouthEast');

figure;
set(gcf, 'Name', 'MEAN');
set(gcf, 'Position', [50 50 1600 1200]);
plot([meanSpan11; meanSpan1; meanStir]', 'LineWidth', 2)
xlabel('Iterations');
ylabel('Mean Value');
legend('Span 11', 'Span 1', 'Stir', 'Location', 'SouthEast');

figure;
set(gcf, 'Name', 'SNR');
set(gcf, 'Position', [50 50 800 600]);
plot([snrSpan11; snrSpan1; snrStir]', 'LineWidth', 2)
xlabel('Iterations', 'FontSize', 14);
ylabel('SNR (mean/std)', 'FontSize', 14);
legend('Span 11', 'Span 1', 'Stir', 'Location', 'NorthEast', 'FontSize', 14);
% Save for publication:
fullFilename = [outputPath 'SNR'];
saveas(gca, [fullFilename], 'tif');
set(gcf,'PaperPositionMode','auto');    % Para que lo guarde en el tamaño modificado.
frame = getframe(gca);
imwrite(frame.cdata, [fullFilename '.png']);
saveas(gca, [fullFilename], 'epsc');

figure;
set(gcf, 'Name', 'SNR Per Slice');
set(gcf, 'Position', [50 50 1600 1200]);
i = 20;
plot([snrPerSliceSpan11(i,:); snrPerSliceSpan1(i,:); snrPerSliceStir(i,:)]');
xlabel('Slice');
ylabel('SNR (mean/std)');
legend('Span 11', 'Span 1', 'Stir', 'Location', 'SouthEast');
title('SNR per Slice for Iteration 20');

figure;
set(gcf, 'Name', 'SNR Per Slice');
set(gcf, 'Position', [50 50 1600 1200]);
i = 60;
plot([snrPerSliceSpan11(i,:); snrPerSliceSpan1(i,:); snrPerSliceStir(i,:)]');
xlabel('Slice');
ylabel('SNR (mean/std)');
legend('Span 11', 'Span 1', 'Stir', 'Location', 'SouthEast');
title('SNR per Slice for Iteration 60');

% Contrast Recovery Coefficient of all spheres for each type of reconstruction:
figure;
set(gcf, 'Name', 'Contrast Recovery Coefficient for each Reconstruction');
set(gcf, 'Position', [50 50 1000 1200]);
subplot(3,1,1);
plot(contrastRecoverySpan11);
xlabel('Iterations');
ylabel('Contrast Recovery Coefficient [%]');
for i = 1 : numel(radioEsferas_mm)
    legends{i} = sprintf('Sphere Radius %.2f\n', radioEsferas_mm(i));
end
legend(legends, 'Location', 'SouthEast');
subplot(3,1,2);
plot(contrastRecoverySpan1);
xlabel('Iterations');
ylabel('Contrast Recovery Coefficient [%]');
legend(legends, 'Location', 'SouthEast');
subplot(3,1,3);
plot(contrastRecoveryStir);
xlabel('Iterations');
ylabel('Contrast Recovery Coefficient [%]');
legend(legends, 'Location', 'SouthEast');

% Contrast Recovery Coefficient. Intercomparison for each sphere betweeen the
% reconstruction methods.
figure;
set(gcf, 'Name', 'Contrast Recovery Coefficient for each Reconstruction');
set(gcf, 'Position', [50 50 1600 1200]);
for i = 1 : numel(radioEsferas_mm)-1 % Lug not used
    subplot(3,2,i);
    plot([contrastRecoverySpan11(:,i) contrastRecoverySpan1(:,i) contrastRecoveryStir(:,i)], 'LineWidth', 2);
    xlabel('Iterations');
    ylabel('Contrast Recovery Coefficient [%]');
    title(sprintf('Sphere Radius %.2f\n', radioEsferas_mm(i)));
    legend('Span 11', 'Span 1', 'Stir', 'Location', 'SouthEast');
end

% Contrast Recovery Coefficient. Intercomparison for each sphere betweeen the
% reconstruction methods.
figure;
set(gcf, 'Name', 'Normalized Std Dev');
set(gcf, 'Position', [50 50 1600 1200]);
for i = 1 : numel(radioEsferas_mm)
    subplot(3,2,i);
    plot([desvioNormBackgroundSpan11(:,i) desvioNormBackgroundSpan1(:,i) desvioNormBackgroundStir(:,i)], 'LineWidth', 2);
    xlabel('Iterations');
    ylabel('Norm Std Dev');
    title(sprintf('Sphere Radius %.2f\n', radioEsferas_mm(i)));
    legend('Span 11', 'Span 1', 'Stir', 'Location', 'SouthEast');
end
%% OVERRAL RESULT WITH STD DV IN SPHERES
% Separate plots:
for i = 1 : numel(radioEsferas_mm)
    figure;
    set(gcf, 'Name', 'CRC VS STD (in neighbours rois)');
    set(gcf, 'Position', [50 50 1600 1200]);
    plot(contrastRecoverySpan11(:,i), desvioNormBackgroundSpan11(:,i), contrastRecoverySpan1(:,i), desvioNormBackgroundSpan1(:,i), contrastRecoveryStir(:,i), desvioNormBackgroundStir(:,i), 'LineWidth', 2);
    xlabel('Contrast Recovery Coefficient');
    ylabel('Standard Deviation');
    legend('Span 11', 'Span 1', 'Stir', 'Location', 'SouthEast');
    title(sprintf('Sphere with Diameter of %.2f mm\n', 2*radioEsferas_mm(i)));
    fullFilename = [outputPath sprintf('crc_vs_std_sphere_%d', i)];
    saveas(gca, [fullFilename], 'tif');
    set(gcf,'PaperPositionMode','auto');    % Para que lo guarde en el tamaño modificado.
    frame = getframe(gca);
    imwrite(frame.cdata, [fullFilename '.png']);
    saveas(gca, [fullFilename], 'epsc');
end

% All in one figure:
figure;
set(gcf, 'Name', 'CRC VS STD (in neighbours rois)');
set(gcf, 'Position', [50 50 1600 1200]);
for i = 1 : numel(radioEsferas_mm)   
    subplot(2,3,i);
    plot(contrastRecoverySpan11(:,i), desvioNormBackgroundSpan11(:,i), contrastRecoverySpan1(:,i), desvioNormBackgroundSpan1(:,i), contrastRecoveryStir(:,i), desvioNormBackgroundStir(:,i), 'LineWidth', 2);
    xlabel('Contrast Recovery Coefficient');
    ylabel('Standard Deviation');
    legend('Span 11', 'Span 1', 'Stir', 'Location', 'NorthWest');
    title(sprintf('Sphere with Diameter of %.2f mm\n', 2*radioEsferas_mm(i)));
    
end
fullFilename = [outputPath 'crc_vs_std_all_spheres'];
saveas(gca, [fullFilename], 'tif');
set(gcf,'PaperPositionMode','auto');    % Para que lo guarde en el tamaño modificado.
frame = getframe(gca);
imwrite(frame.cdata, [fullFilename '.png']);
saveas(gca, [fullFilename], 'epsc');

% All in one figure:
figure;
set(gcf, 'Name', 'CRC VS STD (in neighbours rois)');
set(gcf, 'Position', [50 50 1200 1400]);
for i = 1 : numel(radioEsferas_mm)   
    subplot(3,2,i);
    plot(contrastRecoverySpan11(:,i), desvioNormBackgroundSpan11(:,i), contrastRecoverySpan1(:,i), desvioNormBackgroundSpan1(:,i), contrastRecoveryStir(:,i), desvioNormBackgroundStir(:,i), 'LineWidth', 2);
    xlabel('Contrast Recovery Coefficient');
    ylabel('Standard Deviation');
    legend('Span 11', 'Span 1', 'Stir', 'Location', 'NorthWest');
    title(sprintf('Sphere with Diameter of %.2f mm\n', 2*radioEsferas_mm(i)));
    
end
fullFilename = [outputPath 'crc_vs_std_all_spheres_bis'];
saveas(gca, [fullFilename], 'tif');
set(gcf,'PaperPositionMode','auto');    % Para que lo guarde en el tamaño modificado.
frame = getframe(gca);
imwrite(frame.cdata, [fullFilename '.png']);
saveas(gca, [fullFilename], 'epsc');
%% OVERRAL RESULT WITH GLOBAL STD
normStdSpan11 = stdSpan11 ./ meanSpan11;
normStdSpan1 = stdSpan1 ./ meanSpan1;
normStdStir = stdStir ./ meanStir;
for i = 1 : numel(radioEsferas_mm)
    figure;
    set(gcf, 'Name', 'CRC VS Global STD ');
    set(gcf, 'Position', [50 50 1600 1200]);
    plot(contrastRecoverySpan11(:,i), normStdSpan11, contrastRecoverySpan1(:,i), normStdSpan1, contrastRecoveryStir(:,i), normStdStir, 'LineWidth', 2);
    xlabel('Contrast Recovery Coefficient');
    ylabel('Standard Deviation');
    legend('Span 11', 'Span 1', 'Stir', 'Location', 'SouthEast');
    title(sprintf('Sphere with Diameter of %.2f mm\n', 2*radioEsferas_mm(i)));
    fullFilename = [outputPath sprintf('crc_vs_global_std_sphere_%d', i)];
    saveas(gca, [fullFilename], 'tif');
    set(gcf,'PaperPositionMode','auto');    % Para que lo guarde en el tamaño modificado.
    frame = getframe(gca);
    imwrite(frame.cdata, [fullFilename '.png']);
    saveas(gca, [fullFilename], 'epsc');
end

% All in one figure:
figure;
set(gcf, 'Name', 'CRC VS Global STD');
set(gcf, 'Position', [50 50 1800 1400]);
for i = 1 : numel(radioEsferas_mm)   
    subplot(2,3,i);
    plot(contrastRecoverySpan11(:,i), normStdSpan11, contrastRecoverySpan1(:,i), normStdSpan1, contrastRecoveryStir(:,i), normStdStir, 'LineWidth', 2);
    xlabel('Contrast Recovery Coefficient');
    ylabel('Standard Deviation');
    legend('Span 11', 'Span 1', 'Stir', 'Location', 'SouthEast');
    title(sprintf('Sphere with Diameter of %.2f mm\n', 2*radioEsferas_mm(i)));  
end
fullFilename = [outputPath 'crc_vs_global_std_all_spheres'];
saveas(gca, [fullFilename], 'tif');
set(gcf,'PaperPositionMode','auto');    % Para que lo guarde en el tamaño modificado.
frame = getframe(gca);
imwrite(frame.cdata, [fullFilename '.png']);
saveas(gca, [fullFilename], 'epsc');
%%
% All in one figure:
figure;
set(gcf, 'Name', 'CRC VS Global STD');
set(gcf, 'Position', [50 50 1200 1200]);
for i = 1 : numel(radioEsferas_mm)   
    columna = ceil(i/2);
    fila = rem(i-1,2);
    columna+fila*0.95
    subplot(3,2,(columna-1)*2+1+fila*0.90);
    plot(contrastRecoverySpan11(:,i), normStdSpan11, contrastRecoverySpan1(:,i), normStdSpan1, contrastRecoveryStir(:,i), normStdStir, 'LineWidth', 2);
    %xlim([0 100]);
    %ylim([0.05 0.8]);
    xlabel('Contrast Recovery Coefficient');
    ylabel('Standard Deviation');
    legend('Span 11', 'Span 1', 'Stir', 'Location', 'SouthEast');
    title(sprintf('Sphere with Diameter of %.2f mm\n', 2*radioEsferas_mm(i))); 
    ticklabels = get(gca, 'XtickLabel');
    set(gca, 'XtickLabel', ticklabels, 'FontSize',10);
    
end
fullFilename = [outputPath 'crc_vs_global_std_all_spheres_bis'];
saveas(gca, [fullFilename], 'tif');
set(gcf,'PaperPositionMode','auto');    % Para que lo guarde en el tamaño modificado.
% frame = getframe(gca);
% imwrite(frame.cdata, [fullFilename '.png']);
saveas(gca, [fullFilename], 'epsc');