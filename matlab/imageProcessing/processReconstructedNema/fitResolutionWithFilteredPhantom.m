%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 11/03/2015
%  *********************************************************************
%  This scripts compares different reconstructed images with a degraded
%  image of a phantom.
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
%% LOAD IMAGES
% Read interfile reconstructed image:
centralSlice = 81;
% Span 11 APIRL:
span11Path = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/NEMA_IQ_20_02_2014/Reconstruction/span11/';
filename = 'Nema_Osem21_HR_final.h33';
reconVolumeSpan11 = interfileRead([span11Path filename]);
infoVolumeSpan11 = interfileinfo([span11Path filename]);

% Span 1 APIRL:
span1Path = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/NEMA_IQ_20_02_2014/Reconstruction/span1/';
filename = 'Nema_Osem21_HR_final.h33';
reconVolumeSpan1 = interfileRead([span1Path filename]);
infoVolumeSpan1 = interfileinfo([span1Path filename]);

% Stir
stirPath = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/NEMA_IQ_20_02_2014/Stir/';
filename = 'pet_im_21.hv';
reconVolumeStir = interfileRead([stirPath filename]);
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
% Show slice:
figure;
title('Slice 80 of Mask and Recon Phantom');
imshowpair(reconVolumeSpan11(:,:,80), maskPhantom(:,:,80));
%% GET NMSE RESPECT TO THE PHANTOM
meanPhantom = mean(mean(mean(attenMap_rescaled)));
meanSpan11 = mean(mean(mean(reconVolumeSpan11)));
meanStir = mean(mean(mean(reconVolumeStir)));
normalizedPhantom = attenMap_rescaled ./ meanPhantom;
normalizedSpan11 = reconVolumeSpan11 ./ meanSpan11;
normalizedStir = reconVolumeStir ./ meanStir;
nmseSpan11 = ((normalizedSpan11-normalizedPhantom).^2);
nmseStir = ((normalizedStir-normalizedPhantom).^2);
nmseSpan11_2 = ((reconVolumeSpan11-attenMap_rescaled).^2) / (meanSpan11*meanPhantom);
figure;
subplot(1,2,1);
imshow(nmseSpan11(:,:,centralSlice)/max(max(nmseSpan11(:,:,centralSlice))));
subplot(1,2,2);
imshow(nmseSpan11_2(:,:,centralSlice)/max(max(nmseSpan11_2(:,:,centralSlice))));

nmsePerSliceSpan11 = sum(sum(nmseSpan11));
nmsePerSliceSpan11 = permute(nmsePerSliceSpan11, [3 1 2]);
nmsePerSliceStir = sum(sum(nmseStir));
nmsePerSliceStir = permute(nmsePerSliceStir, [3 1 2]);
figure;
plot([nmsePerSliceSpan11 nmsePerSliceStir]);

figure;
subplot(1,2,1);
imshow(nmseSpan11(:,:,centralSlice)/max(max(nmseSpan11(:,:,centralSlice))));
subplot(1,2,2);
imshow(nmseStir(:,:,centralSlice)/max(max(nmseStir(:,:,centralSlice))));
%% MULTIRESOLUTION OF CENTRAL SLICE
sizePixel_mm = [infoVolumeSpan11.ScalingFactorMmPixel1 infoVolumeSpan11.ScalingFactorMmPixel2 infoVolumeSpan11.ScalingFactorMmPixel3];
% Std Dev of each filter:
filterStdDev_mm = sizePixel_mm(1)/3 : sizePixel_mm(1)/3 : 10;
filterStdDev_pixels = filterStdDev_mm ./ sizePixel_mm(1);
% Size of the filter (for 3 sigmas: 3*filterStdDev_pixels*2 + 1):
filterSize_pixels = round(3*filterStdDev_pixels*2 + 1);
center_mm = [2 2];
[contrastRecoveryStir, desvioBackgroundStir, desvioNormBackgroundStir, meanLungRoiStir, relativeLungErrorStir, radioEsferas_mm, centers_pixels] = procImagesQualityPhantomColdSpheres(normalizedSpan11, sizePixel_mm, center_mm, centralSlice, 0);
figure;
imshow(normalizedSpan11(:,:,centralSlice)>meanLungRoiStir*0.3)
for i = 1 : numel(filterSize_pixels)
    filter = fspecial('gaussian',[filterSize_pixels(i) filterSize_pixels(i)],filterStdDev_pixels(i));
    % Filter the phantom:
    filteredPhantom{i} = imfilter(attenMap_rescaled(:,:,centralSlice), filter);
end

centralSliceSpan11 = normalizedSpan11(:,:,centralSlice)>meanLungRoiStir*0.3;
centralSliceStir = normalizedStir(:,:,centralSlice)>meanLungRoiStir*0.3;
% Several figure with 4 filtered image per data set:
numImagesPerFigure = 4;
k = 1;
for i = 1 : numel(filterSize_pixels)
    nmseSpan11CentralSlice(i) = sum(sum(((centralSliceSpan11-filteredPhantom{i}).^2)));
    nmseStirCentralSlice(i) = sum(sum(((centralSliceStir-filteredPhantom{i}).^2)));
    
end
figure;
set(gcf, 'Position', [50 50 1600 1200]);
plot(filterStdDev_mm, nmseSpan11CentralSlice, filterStdDev_mm, nmseStirCentralSlice);
xlabel('Std Dev [mm]');
ylabel('SNR (mean/std)');
title('SNR of Central Slice for Different Filters');
