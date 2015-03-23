
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
outputPath = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/LineSources/ReconstructionEvaluation/';
mkdir(outputPath);
%setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') ':/usr/lib/x86_64-linux-gnu/']);
%% IMAGES PARAMETERS
numIterations = 60;
% Read interfile reconstructed image:
% Span 11 APIRL:
span11Path = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/LineSources/Reconstruction/span11/';
filenameSpan11 = 'Nema_Osem21_HR_withoutAttenuation_final.h33';
reconVolumeSpan11 = interfileRead([span11Path filenameSpan11]);
infoVolumeSpan11 = interfileinfo([span11Path filenameSpan11]);

% Span 1 APIRL:
span1Path = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/LineSources/Reconstruction/span1/';
filenameSpan1 = 'Nema_Osem21_HR_withoutAttenuation_final.h33';
reconVolumeSpan1 = interfileRead([span1Path filenameSpan1]);
infoVolumeSpan1 = interfileinfo([span1Path filenameSpan1]);

% Stir
stirPath = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/LineSources/Stir/';
filenameStir = 'pet_im_no_atten_21.hv';
reconVolumeStir = interfileRead([stirPath filenameStir]);
infoVolumeStir = interfileinfo([stirPath filenameStir]);

% Siemens:
siemensPath = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/LineSources/Att_maps/';
filenameSiemens = 'PET_ACQ_91_20150313115152_ima_NAC_000_000.v';
% Read the raw data because the interfile format is not the correct one:
fid = fopen([siemensPath filenameSiemens], 'r');
sizeImageSiemens = [172 172 127];
sizePixelSiemens_mm = [4.17253 4.17253 2.03125];
reconVolumeSiemens = fread(fid, sizeImageSiemens(1)*sizeImageSiemens(2)*sizeImageSiemens(3), 'single');
fclose(fid);
reconVolumeSiemens = reshape(reconVolumeSiemens, sizeImageSiemens);
% Permute dimensiones, the first dimension of the image is x, but in matlab
% are the columns:
reconVolumeSiemens = permute(reconVolumeSiemens, [2 1 3]);
% invert rows (lower y planes are first in the raw data):
% resccale to the size of the other images:
coordXsiemens = (-sizePixelSiemens_mm(2)*sizeImageSiemens(2)/2+sizePixelSiemens_mm(2)/2) : sizePixelSiemens_mm(2) : (sizePixelSiemens_mm(2)*sizeImageSiemens(2)/2);
coordYsiemens = (-sizePixelSiemens_mm(1)*sizeImageSiemens(1)/2+sizePixelSiemens_mm(1)/2) : sizePixelSiemens_mm(1) : (sizePixelSiemens_mm(1)*sizeImageSiemens(1)/2);
coordZsiemens = (-sizePixelSiemens_mm(3)*sizeImageSiemens(3)/2+sizePixelSiemens_mm(3)/2) : sizePixelSiemens_mm(3) : (sizePixelSiemens_mm(3)*sizeImageSiemens(3)/2);
[Xsiemens, Ysiemens, Zsiemens] = meshgrid(coordXsiemens, coordYsiemens, coordZsiemens);
% Idem for the dixon attenuation map:
coordXpet = (-infoVolumeSpan11.ScalingFactorMmPixel2*infoVolumeSpan11.MatrixSize2/2 + infoVolumeSpan11.ScalingFactorMmPixel2/2) : infoVolumeSpan11.ScalingFactorMmPixel2 : (infoVolumeSpan11.ScalingFactorMmPixel2*infoVolumeSpan11.MatrixSize2/2);
coordYpet = (-infoVolumeSpan11.ScalingFactorMmPixel1*infoVolumeSpan11.MatrixSize1/2 + infoVolumeSpan11.ScalingFactorMmPixel1/2) : infoVolumeSpan11.ScalingFactorMmPixel1 : (infoVolumeSpan11.ScalingFactorMmPixel1*infoVolumeSpan11.MatrixSize1/2);
coordZpet = (-infoVolumeSpan11.ScalingFactorMmPixel3*infoVolumeSpan11.MatrixSize3/2 + infoVolumeSpan11.ScalingFactorMmPixel3/2) : infoVolumeSpan11.ScalingFactorMmPixel3 : (infoVolumeSpan11.ScalingFactorMmPixel3*infoVolumeSpan11.MatrixSize3/2);
[Xpet, Ypet, Zpet] = meshgrid(coordXpet, coordYpet, coordZpet);
% Interpolate the ct image to the mr coordinates:
reconVolumeSiemens = interp3(Xsiemens,Ysiemens,Zsiemens,reconVolumeSiemens,Xpet,Ypet,Zpet); 
reconVolumeSiemens(isnan(reconVolumeSiemens)) = 0;
%% SHOW SLICES
% Get slices to show:
image = getImageFromSlices(reconVolumeSpan11,12, 1, 0);
figure;
set(gcf, 'Name', 'Slices for Span 11');
set(gcf, 'Position', [50 50 1600 1200]);
imshow(image);
colormap(hot);

image = getImageFromSlices(reconVolumeSpan1,12, 1, 0);
figure;
set(gcf, 'Name', 'Slices for Span 1');
set(gcf, 'Position', [50 50 1600 1200]);
imshow(image);
colormap(hot);

image = getImageFromSlices(reconVolumeStir,12, 1, 0);
figure;
set(gcf, 'Name', 'Slices for Stir');
set(gcf, 'Position', [50 50 1600 1200]);
imshow(image);
colormap(hot);
%% MAXIMUM INTENSITY PROJECTION
% Get MIPs in the three axes:
[mipTransverseSpan11, mipCoronalSpan11, mipSagitalSpan11] = showMaximumIntensityProjections(reconVolumeSpan11);
set(gcf, 'Name', 'Maximum Intensity Projection Span11');
[mipTransverseSpan1, mipCoronalSpan1, mipSagitalSpan1] = showMaximumIntensityProjections(reconVolumeSpan1);
set(gcf, 'Name', 'Maximum Intensity Projection Span 1');
[mipTransverseStir, mipCoronalStir, mipSagitalStir] = showMaximumIntensityProjections(reconVolumeStir);
set(gcf, 'Name', 'Maximum Intensity Projection Stir');
[mipTransverseSiemens, mipCoronalSiemens, mipSagitalSiemens] = showMaximumIntensityProjections(reconVolumeSiemens);
set(gcf, 'Name', 'Maximum Intensity Projection Siemens');

% Profile of the MIP
figure;
profileRow = round(size(mipSagitalSpan11,1)/2);
plot([mipSagitalSpan11(profileRow,:)./max(mipSagitalSpan11(profileRow,:)); mipSagitalSpan1(profileRow,:)./max(mipSagitalSpan1(profileRow,:)); mipSagitalStir(profileRow,:)./max(mipSagitalStir(profileRow,:));...
    mipSagitalSiemens(profileRow,:)./max(mipSagitalSiemens(profileRow,:))]');
set(gcf, 'Name', 'Maximum Intensity Profile');
legend('Span 11', 'Span 1', 'Stir', 'Siemens');
%% MEAN VALUE OF SLICES
meanValueSpan11 = mean(mean(mean(reconVolumeSpan11)));
meanValueSpan1 = mean(mean(mean(reconVolumeSpan1)));
meanValueStir = mean(mean(mean(reconVolumeStir)));
meanValueSiemens = mean(mean(mean(reconVolumeSiemens)));
meanValuePerSliceSpan11 = permute(mean(mean(reconVolumeSpan11)), [3 1 2]);
meanValuePerSliceSpan1 = permute(mean(mean(reconVolumeSpan1)), [3 1 2]);
meanValuePerSliceStir = permute(mean(mean(reconVolumeStir)), [3 1 2]);
meanValuePerSliceSiemens = permute(mean(mean(reconVolumeSiemens)), [3 1 2]);
slices = size(reconVolumeSpan11, 3);
figure;
set(gcf, 'Position', [50 50 1600 1200]);
plot(1:slices, meanValuePerSliceSpan11./meanValueSpan11, 1:slices, meanValuePerSliceSpan1./meanValueSpan1, 1:slices, meanValuePerSliceStir./meanValueStir,...
    1:slices, meanValuePerSliceSiemens./meanValueSiemens,'LineWidth', 2);
legend('Span 11', 'Span 1', 'Stir', 'Siemens');
%% FWHM OF EACH SLICE
for i = 1 : slices
    % Y axis (dimension 1: rows)
    fullFilename = sprintf('fwhm_y_span11_slice_%d', i);
    [fwhm_y_span11(i), fwhm_y_fitted_span11(i)] = getFwhmOfPointSourceImage(reconVolumeSpan11(:,:,i), [infoVolumeSpan1.ScalingFactorMmPixel1 infoVolumeSpan1.ScalingFactorMmPixel2], 1, 0, fullFilename);
    fullFilename = sprintf('fwhm_y_span1_slice_%d', i);
    [fwhm_y_span1(i), fwhm_y_fitted_span1(i)] = getFwhmOfPointSourceImage(reconVolumeSpan1(:,:,i), [infoVolumeSpan1.ScalingFactorMmPixel1 infoVolumeSpan1.ScalingFactorMmPixel2], 1, 0, fullFilename);
    fullFilename = sprintf('fwhm_y_stir_slice_%d', i);
    [fwhm_y_stir(i), fwhm_y_fitted_stir(i)] = getFwhmOfPointSourceImage(reconVolumeStir(:,:,i), [infoVolumeSpan1.ScalingFactorMmPixel1 infoVolumeSpan1.ScalingFactorMmPixel2], 1, 0, fullFilename);
    fullFilename = sprintf('fwhm_y_siemens_slice_%d', i);
    [fwhm_y_siemens(i), fwhm_y_fitted_siemens(i)] = getFwhmOfPointSourceImage(reconVolumeSiemens(:,:,i), [infoVolumeSpan1.ScalingFactorMmPixel1 infoVolumeSpan1.ScalingFactorMmPixel2], 1, 0, fullFilename);
    
    % X axis (dimension 2: columns):
    fullFilename = sprintf('fwhm_x_span11_slice_%d', i);
    [fwhm_x_span11(i), fwhm_x_fitted_span11(i)] = getFwhmOfPointSourceImage(reconVolumeSpan11(:,:,i), [infoVolumeSpan1.ScalingFactorMmPixel1 infoVolumeSpan1.ScalingFactorMmPixel2], 2, 0, fullFilename);
    fullFilename = sprintf('fwhm_x_span1_slice_%d', i);
    [fwhm_x_span1(i), fwhm_x_fitted_span1(i)] = getFwhmOfPointSourceImage(reconVolumeSpan1(:,:,i), [infoVolumeSpan1.ScalingFactorMmPixel1 infoVolumeSpan1.ScalingFactorMmPixel2], 2, 0, fullFilename);
    fullFilename = sprintf('fwhm_x_stir_slice_%d', i);
    [fwhm_x_stir(i), fwhm_x_fitted_stir(i)] = getFwhmOfPointSourceImage(reconVolumeStir(:,:,i), [infoVolumeSpan1.ScalingFactorMmPixel1 infoVolumeSpan1.ScalingFactorMmPixel2], 2, 0, fullFilename);
    fullFilename = sprintf('fwhm_y_siemens_slice_%d', i);
    [fwhm_x_siemens(i), fwhm_x_fitted_siemens(i)] = getFwhmOfPointSourceImage(reconVolumeSiemens(:,:,i), [infoVolumeSpan1.ScalingFactorMmPixel1 infoVolumeSpan1.ScalingFactorMmPixel2], 2, 0, fullFilename);
    %close all
end
figure;
set(gcf, 'Position', [50 50 1800 1200]);
set(gcf, 'Name', 'Transverse Resolution Analysis');
subplot(1,2,1);
plot(1:slices, fwhm_x_fitted_span11, 1:slices, fwhm_x_fitted_span1, 1:slices, fwhm_x_fitted_stir, 1:slices, fwhm_x_fitted_siemens, 'LineWidth', 2);
legend('Span 11', 'Span 1', 'Stir', 'Siemens');
xlabel('Slice');
ylabel('FWHM X [mm]');
title('Resolution in X axis');
subplot(1,2,2);
plot(1:slices, fwhm_y_fitted_span11, 1:slices, fwhm_y_fitted_span1, 1:slices, fwhm_y_fitted_stir, 1:slices, fwhm_y_fitted_siemens, 'LineWidth', 2);
legend('Span 11', 'Span 1', 'Stir', 'Siemens');
xlabel('Slice');
ylabel('FWHM Y [mm]');
title('Resolution in Y axis');
%% FWHM SAGITAL PLANES
auxImageSpan11 =  permute(reconVolumeSpan11, [1 3 2]);
auxImageSpan1 = permute(reconVolumeSpan1, [1 3 2]);
auxImageStir = permute(reconVolumeStir, [1 3 2]);
auxImageSiemens = permute(reconVolumeSiemens, [1 3 2]);
sagitalPlanesForLine = find(permute(mean(mean(auxImageSpan11)), [3 1 2]) > 1*meanValueSpan11);
fwhm_y_span11 = zeros(1, size(reconVolumeSpan11,2)); fwhm_y_fitted_span11 = zeros(1, size(reconVolumeSpan11,2)); fwhm_z_span11 = zeros(1, size(reconVolumeSpan11,2)); fwhm_z_fitted_span11 = zeros(1, size(reconVolumeSpan11,2));
fwhm_y_span1 = zeros(1, size(reconVolumeSpan11,2)); fwhm_y_fitted_span1 = zeros(1, size(reconVolumeSpan11,2)); fwhm_z_span1 = zeros(1, size(reconVolumeSpan11,2)); fwhm_z_fitted_span1 = zeros(1, size(reconVolumeSpan11,2));
fwhm_y_stir = zeros(1, size(reconVolumeSpan11,2)); fwhm_y_fitted_stir = zeros(1, size(reconVolumeSpan11,2)); fwhm_z_stir = zeros(1, size(reconVolumeSpan11,2)); fwhm_z_fitted_stir = zeros(1, size(reconVolumeSpan11,2));
fwhm_y_siemens = zeros(1, size(reconVolumeSiemens,2)); fwhm_y_fitted_siemens = zeros(1, size(reconVolumeSiemens,2)); fwhm_z_siemens = zeros(1, size(reconVolumeSiemens,2)); fwhm_z_fitted_siemens = zeros(1, size(reconVolumeSiemens,2));
for i = 1 : numel(sagitalPlanesForLine)
    % Y axis (dimension 1: rows)
    plane = sagitalPlanesForLine(i)
    fullFilename = sprintf('fwhm_y_span11_slice_%d', i);
    [fwhm_y_span11(plane), fwhm_y_fitted_span11(plane)] = getFwhmOfPointSourceImage(auxImageSpan11(:,:,plane), [infoVolumeSpan1.ScalingFactorMmPixel1 infoVolumeSpan1.ScalingFactorMmPixel3], 1, 0, fullFilename,'Y [mm]');
    
    fullFilename = sprintf('fwhm_y_span1_slice_%d', i);
    [fwhm_y_span1(plane), fwhm_y_fitted_span1(plane)] = getFwhmOfPointSourceImage(auxImageSpan1(:,:,plane), [infoVolumeSpan1.ScalingFactorMmPixel1 infoVolumeSpan1.ScalingFactorMmPixel3], 1, 0, fullFilename,'Y [mm]');
    
    fullFilename = sprintf('fwhm_y_stir_slice_%d', i);
    [fwhm_y_stir(plane), fwhm_y_fitted_stir(plane)] = getFwhmOfPointSourceImage(auxImageStir(:,:,plane), [infoVolumeSpan1.ScalingFactorMmPixel1 infoVolumeSpan1.ScalingFactorMmPixel3], 1, 0, fullFilename,'Y [mm]');
    
    fullFilename = sprintf('fwhm_y_siemens_slice_%d', i);
    [fwhm_y_siemens(plane), fwhm_y_fitted_siemens(plane)] = getFwhmOfPointSourceImage(auxImageSiemens(:,:,plane), [infoVolumeSpan1.ScalingFactorMmPixel1 infoVolumeSpan1.ScalingFactorMmPixel3], 1, 0, fullFilename,'Y [mm]');
    
    % X axis (dimension 2: columns):
    fullFilename = sprintf('fwhm_z_span11_slice_%d', i);
    [fwhm_z_span11(plane), fwhm_z_fitted_span11(plane)] = getFwhmOfPointSourceImage(auxImageSpan11(:,:,plane), [infoVolumeSpan1.ScalingFactorMmPixel1 infoVolumeSpan1.ScalingFactorMmPixel3], 2, 0, fullFilename,'Z [mm]');
    
    fullFilename = sprintf('fwhm_z_span1_slice_%d', i);
    [fwhm_z_span1(plane), fwhm_z_fitted_span1(plane)] = getFwhmOfPointSourceImage(auxImageSpan1(:,:,plane), [infoVolumeSpan1.ScalingFactorMmPixel1 infoVolumeSpan1.ScalingFactorMmPixel3], 2, 0, fullFilename,'Z [mm]');
    
    fullFilename = sprintf('fwhm_z_stir_slice_%d', i);
    [fwhm_z_stir(plane), fwhm_z_fitted_stir(plane)] = getFwhmOfPointSourceImage(auxImageStir(:,:,plane), [infoVolumeSpan1.ScalingFactorMmPixel1 infoVolumeSpan1.ScalingFactorMmPixel3], 2, 0, fullFilename,'Z [mm]');
    
    fullFilename = sprintf('fwhm_z_stir_slice_%d', i);
    [fwhm_z_siemens(plane), fwhm_z_fitted_siemens(plane)] = getFwhmOfPointSourceImage(auxImageSiemens(:,:,plane), [infoVolumeSpan1.ScalingFactorMmPixel1 infoVolumeSpan1.ScalingFactorMmPixel3], 2, 0, fullFilename,'Z [mm]');
    %close all
end
figure;
set(gcf, 'Position', [50 50 1600 1200]);
set(gcf, 'Name', 'Sagital Resolution Analysis');
planes = size(reconVolumeSpan11,2);
subplot(1,2,1);
plot(1:planes, fwhm_y_fitted_span11, 1:planes, fwhm_y_fitted_span1, 1:planes, fwhm_y_fitted_stir, 1:planes, fwhm_y_fitted_siemens, 'LineWidth', 2);
legend('Span 11', 'Span 1', 'Stir', 'Siemens');
xlabel('Slice');
ylabel('FWHM Y [mm]');
title('Resolution in Y axis');
ylim([0 8]);
subplot(1,2,2);
plot(1:planes, fwhm_z_fitted_span11, 1:planes, fwhm_z_fitted_span1, 1:planes, fwhm_z_fitted_stir, 1:planes, fwhm_z_fitted_siemens, 'LineWidth', 2);
legend('Span 11', 'Span 1', 'Stir', 'Siemens');
xlabel('Slice');
ylabel('FWHM Z [mm]');
title('Resolution in Z axis');
ylim([0 10]);