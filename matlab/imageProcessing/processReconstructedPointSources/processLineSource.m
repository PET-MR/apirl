
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
siemensPath = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/LineSources/allpass_344/';
filenameSiemens = 'PET_ACQ_91_20150313115152_PRR_1000001_20150408102558_ima_NAC_000_000.v';
% Read the raw data because the interfile format is not the correct one:
fid = fopen([siemensPath filenameSiemens], 'r');
sizeImageSiemens = [344 344 127];
sizePixelSiemens_mm = [2.08626 2.08626 2.03125];
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
% %% GENERATE LINE IMAGE
% absMaxValue = max(max(max(reconVolumeSpan11)));
% % Based on max pixel in transverse plane:
% lineMaxXY = zeros(size(reconVolumeSpan11)); 
% for i = 1 : size(reconVolumeSpan11, 3)
%     [valorMaxY, indiceMaxFila] = max(reconVolumeSpan11(:,:,i),[],1);
%     [valorMaxXY, indiceMaxCol] = max(valorMaxY,[],2);
%     %indiceLinearXY = sub2ind([285 285],indiceMaxFila(indiceMaxCol),indiceMaxCol');
%     if valorMaxXY > 0.5*absMaxValue
%         lineMaxXY(indiceMaxFila(indiceMaxCol),indiceMaxCol, i) = 1;
%     end
% end
% interfilewrite(lineMaxXY, [outputPath 'lineMaxTransverse'], [infoVolumeSpan11.ScalingFactorMmPixel1 infoVolumeSpan11.ScalingFactorMmPixel2 infoVolumeSpan11.ScalingFactorMmPixel3]);
% 
% % Based on max pixel in sagital plane:
% lineMaxYZ = zeros(size(reconVolumeSpan11)); 
% sagitalPlanes = permute(reconVolumeSpan11, [1 3 2]);
% for i = 1 : size(sagitalPlanes, 3)
%     [valorMaxY, indiceMaxFila] = max(sagitalPlanes(:,:,i),[],1);
%     [valorMaxXY, indiceMaxCol] = max(valorMaxY,[],2);
%     if valorMaxXY > 0.4*absMaxValue
%         lineMaxYZ(indiceMaxFila(indiceMaxCol),i , indiceMaxCol) = 1;
%     end
% end
% interfilewrite(lineMaxYZ, [outputPath 'lineMaxSagital'], [infoVolumeSpan11.ScalingFactorMmPixel1 infoVolumeSpan11.ScalingFactorMmPixel2 infoVolumeSpan11.ScalingFactorMmPixel3]);
% 
% % Based on max pixel in transverse plane:
% lineMaxXZ = zeros(size(reconVolumeSpan11)); 
% coronalPlanes = permute(reconVolumeSpan11, [2 3 1]);
% for i = 1 : size(coronalPlanes, 3)
%     [valorMaxY, indiceMaxFila] = max(coronalPlanes(:,:,i),[],1);
%     [valorMaxXY, indiceMaxCol] = max(valorMaxY,[],2);
%     %indiceLinearXY = sub2ind([285 285],indiceMaxFila(indiceMaxCol),indiceMaxCol');
%     if valorMaxXY > 0.4*absMaxValue
%         lineMaxXZ(i, indiceMaxFila(indiceMaxCol),indiceMaxCol) = 1;
%     end
% end
% interfilewrite(lineMaxXZ, [outputPath 'lineMaxCoronal'], [infoVolumeSpan11.ScalingFactorMmPixel1 infoVolumeSpan11.ScalingFactorMmPixel2 infoVolumeSpan11.ScalingFactorMmPixel3]);
% %lineMaxXY(indiceMaxFila, indiceMaxCol) 
% 
% % Summinng the transverse planes:
% transverseSum = sum(reconVolumeStir,3);
% interfilewrite(transverseSum, [outputPath 'lineSumTransverse'], [infoVolumeSpan11.ScalingFactorMmPixel1 infoVolumeSpan11.ScalingFactorMmPixel2 infoVolumeSpan11.ScalingFactorMmPixel3]);
% sagitalSum = sum(sagitalPlanes,3);
% interfilewrite(sagitalSum, [outputPath 'lineSumSagital'], [infoVolumeSpan11.ScalingFactorMmPixel1 infoVolumeSpan11.ScalingFactorMmPixel2 infoVolumeSpan11.ScalingFactorMmPixel3]);
% sagitalPlanes = permute(reconVolumeSpan1, [1 3 2]);
% sagitalSum = sum(sagitalPlanes,3);
% interfilewrite(sagitalSum, [outputPath 'lineSumSagitalSpan1'], [infoVolumeSpan11.ScalingFactorMmPixel1 infoVolumeSpan11.ScalingFactorMmPixel2 infoVolumeSpan11.ScalingFactorMmPixel3]);
% coronalSum = sum(coronalPlanes,3);
% interfilewrite(coronalSum, [outputPath 'lineSumCoronal'], [infoVolumeSpan11.ScalingFactorMmPixel1 infoVolumeSpan11.ScalingFactorMmPixel2 infoVolumeSpan11.ScalingFactorMmPixel3]);
% coronalPlanes = permute(reconVolumeSpan1, [2 3 1]);
% coronalSumSpan1 = sum(coronalPlanes,3);
% interfilewrite(coronalSumSpan1, [outputPath 'lineSumCoronalSpan1'], [infoVolumeSpan11.ScalingFactorMmPixel1 infoVolumeSpan11.ScalingFactorMmPixel2 infoVolumeSpan11.ScalingFactorMmPixel3]);
% % Ideal coronal sum:
% coronalPlanes = permute(lineMaxXY, [2 3 1]);
% coronalSumIdeal = sum(coronalPlanes,3);
% interfilewrite(coronalSumIdeal, [outputPath 'lineSumCoronalIdeal'], [infoVolumeSpan11.ScalingFactorMmPixel1 infoVolumeSpan11.ScalingFactorMmPixel2 infoVolumeSpan11.ScalingFactorMmPixel3]);
% % For each column convolve the ideal image
% figure;
% plot([coronalSum(100,:)./max(coronalSum(100,:)); coronalSumSpan1(100,:)./max(coronalSumSpan1(100,:)); coronalSumIdeal(100,:)./max(coronalSumIdeal(100,:))]', 'LineWidth', 2);
% legend('Span 11', 'Span 1', 'PerectLine');
% title('Coronal Sum');
% 
% % Fit Gaussian:
% %% PROJECT/BACKPROJECT LINE SOURCE
% % Execute APIRL for span 1:
% % interfilewrite(single(lineMaxXY), [outputPath 'lineSource'], [infoVolumeSpan11.ScalingFactorMmPixel1 infoVolumeSpan11.ScalingFactorMmPixel2 infoVolumeSpan11.ScalingFactorMmPixel3]);
% % projectionFilename = [span1Path 'projectSinogram3d.par'];
% % status = system(['project ' projectionFilename]);
% % backprojectionFilename = [span1Path 'backprojectSinogram3d.par'];
% % status = system(['backproject ' backprojectionFilename]);
% % Execute APIRL for span 11:
% %interfilewrite(single(lineMaxXY), [outputPath 'lineSource'], [infoVolumeSpan11.ScalingFactorMmPixel1 infoVolumeSpan11.ScalingFactorMmPixel2 infoVolumeSpan11.ScalingFactorMmPixel3]);
% projectionFilename = [span11Path 'projectSinogram3d.par'];
% status = system(['project ' projectionFilename]);
% backprojectionFilename = [span11Path 'backprojectSinogram3d.par'];
% status = system(['backproject ' backprojectionFilename]);
% backproj_linesource_span1 = interfileRead([span1Path 'BackprojectedImage.h33']); 
% backproj_linesource_span11 = interfileRead([span11Path 'BackprojectedImage.h33']); 
% for i = 1 : slices
%     [fwhm_y_ideal_span11(i), fwhm_y_fitted_ideal_span11(i)] = getFwhmOfPointSourceImage(backproj_linesource_span11(:,:,i), [infoVolumeSpan1.ScalingFactorMmPixel1 infoVolumeSpan1.ScalingFactorMmPixel2], 1, 0, '');
%     [fwhm_y_ideal_span1(i), fwhm_y_fitted_ideal_span1(i)] = getFwhmOfPointSourceImage(backproj_linesource_span1(:,:,i), [infoVolumeSpan1.ScalingFactorMmPixel1 infoVolumeSpan1.ScalingFactorMmPixel2], 1, 0, '');
%     
%     [fwhm_x_ideal_span11(i), fwhm_x_fitted_ideal_span11(i)] = getFwhmOfPointSourceImage(backproj_linesource_span11(:,:,i), [infoVolumeSpan1.ScalingFactorMmPixel1 infoVolumeSpan1.ScalingFactorMmPixel2], 2, 0, '');
%     [fwhm_x_ideal_span1(i), fwhm_x_fitted_ideal_span1(i)] = getFwhmOfPointSourceImage(backproj_linesource_span1(:,:,i), [infoVolumeSpan1.ScalingFactorMmPixel1 infoVolumeSpan1.ScalingFactorMmPixel2], 2, 0, '');
% end
% %% PROJECT/BACKPROJECT LINE SOURCE
% filter = fspecial('gaussian',[1 5],1)
% % Blurr image:
% lineMaxXY_blurred = imfilter(lineMaxXY, filter);
% % Execute APIRL for span 1:
% interfilewrite(single(reconVolumeSpan11), [outputPath 'lineSource'], [infoVolumeSpan11.ScalingFactorMmPixel1 infoVolumeSpan11.ScalingFactorMmPixel2 infoVolumeSpan11.ScalingFactorMmPixel3]);
% projectionFilename = [span1Path 'projectSinogram3d.par'];
% status = system(['project ' projectionFilename]);
% backprojectionFilename = [span1Path 'backprojectSinogram3d.par'];
% status = system(['backproject ' backprojectionFilename]);
% % Execute APIRL for span 11:
% projectionFilename = [span11Path 'projectSinogram3d.par'];
% status = system(['project ' projectionFilename]);
% backprojectionFilename = [span11Path 'backprojectSinogram3d.par'];
% status = system(['backproject ' backprojectionFilename]);
% backproj_linesource_span1 = interfileRead([span1Path 'BackprojectedImage.h33']); 
% backproj_linesource_span11 = interfileRead([span11Path 'BackprojectedImage.h33']); 
% for i = 1 : slices
%     [fwhm_y_ideal_span11(i), fwhm_y_fitted_ideal_span11(i)] = getFwhmOfPointSourceImage(backproj_linesource_span11(:,:,i), [infoVolumeSpan1.ScalingFactorMmPixel1 infoVolumeSpan1.ScalingFactorMmPixel2], 1, 0, '');
%     [fwhm_y_ideal_span1(i), fwhm_y_fitted_ideal_span1(i)] = getFwhmOfPointSourceImage(backproj_linesource_span1(:,:,i), [infoVolumeSpan1.ScalingFactorMmPixel1 infoVolumeSpan1.ScalingFactorMmPixel2], 1, 0, '');
%     
%     [fwhm_x_ideal_span11(i), fwhm_x_fitted_ideal_span11(i)] = getFwhmOfPointSourceImage(backproj_linesource_span11(:,:,i), [infoVolumeSpan1.ScalingFactorMmPixel1 infoVolumeSpan1.ScalingFactorMmPixel2], 2, 0, '');
%     [fwhm_x_ideal_span1(i), fwhm_x_fitted_ideal_span1(i)] = getFwhmOfPointSourceImage(backproj_linesource_span1(:,:,i), [infoVolumeSpan1.ScalingFactorMmPixel1 infoVolumeSpan1.ScalingFactorMmPixel2], 2, 0, '');
% end
% 
% % fwhm:
% figure;
% set(gcf, 'Position', [50 50 1800 1200]);
% set(gcf, 'Name', 'Transverse Resolution Analysis');
% subplot(1,2,1);
% plot(1:slices, fwhm_x_ideal_span11, 1:slices, fwhm_x_ideal_span1, 'LineWidth', 2);
% legend('Span 11', 'Span 1');
% xlabel('Slice');
% ylabel('FWHM X [mm]');
% title('Resolution in X axis');
% ylim([0 25]);
% subplot(1,2,2);
% plot(1:slices, fwhm_y_ideal_span11, 1:slices, fwhm_y_ideal_span1, 'LineWidth', 2);
% legend('Span 11', 'Span 1');
% xlabel('Slice');
% ylabel('FWHM Y [mm]');
% title('Resolution in Y axis');
% ylim([0 25]);
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
set(gcf, 'Name', 'Slices for STIR');
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
set(gcf, 'Name', 'Maximum Intensity Projection STIR');
[mipTransverseSiemens, mipCoronalSiemens, mipSagitalSiemens] = showMaximumIntensityProjections(reconVolumeSiemens);
set(gcf, 'Name', 'Maximum Intensity Projection Siemens');

% Profile of the MIP
figure;
profileRow = round(size(mipSagitalSpan11,1)/2);
plot([mipSagitalSpan11(profileRow,:)./max(mipSagitalSpan11(profileRow,:)); mipSagitalSpan1(profileRow,:)./max(mipSagitalSpan1(profileRow,:)); mipSagitalStir(profileRow,:)./max(mipSagitalStir(profileRow,:));...
    mipSagitalSiemens(profileRow,:)./max(mipSagitalSiemens(profileRow,:))]');
set(gcf, 'Name', 'Maximum Intensity Profile');
legend('Span 11', 'Span 1', 'STIR', 'Siemens');
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
legend('Span 11', 'Span 1', 'STIR', 'Siemens');
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
%     fullFilename = sprintf('fwhm_y_ideal_slice_%d', i);
%     [fwhm_y_ideal(i), fwhm_y_fitted_ideal(i)] = getFwhmOfPointSourceImage(lineMaxXY(:,:,i), [infoVolumeSpan1.ScalingFactorMmPixel1 infoVolumeSpan1.ScalingFactorMmPixel2], 1, 0, fullFilename);
%     
    % X axis (dimension 2: columns):
    fullFilename = sprintf('fwhm_x_span11_slice_%d', i);
    [fwhm_x_span11(i), fwhm_x_fitted_span11(i)] = getFwhmOfPointSourceImage(reconVolumeSpan11(:,:,i), [infoVolumeSpan1.ScalingFactorMmPixel1 infoVolumeSpan1.ScalingFactorMmPixel2], 2, 0, fullFilename);
    fullFilename = sprintf('fwhm_x_span1_slice_%d', i);
    [fwhm_x_span1(i), fwhm_x_fitted_span1(i)] = getFwhmOfPointSourceImage(reconVolumeSpan1(:,:,i), [infoVolumeSpan1.ScalingFactorMmPixel1 infoVolumeSpan1.ScalingFactorMmPixel2], 2, 0, fullFilename);
    fullFilename = sprintf('fwhm_x_stir_slice_%d', i);
    [fwhm_x_stir(i), fwhm_x_fitted_stir(i)] = getFwhmOfPointSourceImage(reconVolumeStir(:,:,i), [infoVolumeSpan1.ScalingFactorMmPixel1 infoVolumeSpan1.ScalingFactorMmPixel2], 2, 0, fullFilename);
    fullFilename = sprintf('fwhm_x_siemens_slice_%d', i);
    [fwhm_x_siemens(i), fwhm_x_fitted_siemens(i)] = getFwhmOfPointSourceImage(reconVolumeSiemens(:,:,i), [infoVolumeSpan1.ScalingFactorMmPixel1 infoVolumeSpan1.ScalingFactorMmPixel2], 2, 0, fullFilename);
%     fullFilename = sprintf('fwhm_x_ideal_slice_%d', i);
%     [fwhm_x_ideal(i), fwhm_x_fitted_ideal(i)] = getFwhmOfPointSourceImage(lineMaxXY(:,:,i), [infoVolumeSpan1.ScalingFactorMmPixel1 infoVolumeSpan1.ScalingFactorMmPixel2], 1, 0, fullFilename);
%     
    %close all
end

% fwhm:
figure;
set(gcf, 'Position', [50 50 1800 1200]);
set(gcf, 'Name', 'Transverse Resolution Analysis');
subplot(1,2,1);
plot(1:slices, fwhm_x_span11, 1:slices, fwhm_x_span1, 1:slices, fwhm_x_stir, 1:slices, fwhm_x_siemens, 'LineWidth', 2);
legend('Span 11', 'Span 1', 'STIR', 'Siemens');
xlabel('Slice');
ylabel('FWHM X [mm]');
title('Resolution in X axis');
subplot(1,2,2);
plot(1:slices, fwhm_y_span11, 1:slices, fwhm_y_span1, 1:slices, fwhm_y_stir, 1:slices, fwhm_y_siemens, 'LineWidth', 2);
legend('Span 11', 'Span 1', 'STIR', 'Siemens');
xlabel('Slice');
ylabel('FWHM Y [mm]');
title('Resolution in Y axis');

% fwhm with ideal:
figure;
set(gcf, 'Position', [50 50 1800 1200]);
set(gcf, 'Name', 'Transverse Resolution Analysis');
subplot(1,2,1);
plot(1:slices, fwhm_x_span11, 1:slices, fwhm_x_span1, 1:slices, fwhm_x_stir, 1:slices, fwhm_x_siemens, 'LineWidth', 2);
legend('Span 11', 'Span 1', 'STIR', 'Siemens');
xlabel('Slice');
ylabel('FWHM X [mm]');
title('Resolution in X axis');
subplot(1,2,2);
plot(1:slices, fwhm_y_span11, 1:slices, fwhm_y_span1, 1:slices, fwhm_y_stir, 1:slices, fwhm_y_siemens, 'LineWidth', 2);
legend('Span 11', 'Span 1', 'STIR', 'Siemens');
xlabel('Slice');
ylabel('FWHM Y [mm]');
title('Resolution in Y axis');

%fitted:
figure;
set(gcf, 'Position', [50 50 1800 1200]);
set(gcf, 'Name', 'Transverse Resolution Analysis');
subplot(1,2,1);
plot(1:slices, fwhm_x_fitted_span11, 1:slices, fwhm_x_fitted_span1, 1:slices, fwhm_x_fitted_stir, 1:slices, fwhm_x_fitted_siemens, 'LineWidth', 2);
legend('Span 11', 'Span 1', 'STIR', 'Siemens');
xlabel('Slice');
ylabel('FWHM of Fitted Gaussian in X [mm] ');
title('Resolution in X axis with Fitted Gaussian');
subplot(1,2,2);
plot(1:slices, fwhm_y_fitted_span11, 1:slices, fwhm_y_fitted_span1, 1:slices, fwhm_y_fitted_stir, 1:slices, fwhm_y_fitted_siemens, 'LineWidth', 2);
legend('Span 11', 'Span 1', 'STIR', 'Siemens');
xlabel('Slice');
ylabel('FWHM of Fitted Gaussian in Y [mm]');
title('Resolution in Y axis with Fitted Gaussian');
%% FWHM SAGITAL PLANES
graficar = 0;
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
    [fwhm_y_span11(plane), fwhm_y_fitted_span11(plane)] = getFwhmOfPointSourceImage(auxImageSpan11(:,:,plane), [infoVolumeSpan1.ScalingFactorMmPixel1 infoVolumeSpan1.ScalingFactorMmPixel3], 1, graficar, fullFilename,'Y [mm]');
    
    fullFilename = sprintf('fwhm_y_span1_slice_%d', i);
    [fwhm_y_span1(plane), fwhm_y_fitted_span1(plane)] = getFwhmOfPointSourceImage(auxImageSpan1(:,:,plane), [infoVolumeSpan1.ScalingFactorMmPixel1 infoVolumeSpan1.ScalingFactorMmPixel3], 1, graficar, fullFilename,'Y [mm]');
    
    fullFilename = sprintf('fwhm_y_stir_slice_%d', i);
    [fwhm_y_stir(plane), fwhm_y_fitted_stir(plane)] = getFwhmOfPointSourceImage(auxImageStir(:,:,plane), [infoVolumeSpan1.ScalingFactorMmPixel1 infoVolumeSpan1.ScalingFactorMmPixel3], 1, graficar, fullFilename,'Y [mm]');
    
    fullFilename = sprintf('fwhm_y_siemens_slice_%d', i);
    [fwhm_y_siemens(plane), fwhm_y_fitted_siemens(plane)] = getFwhmOfPointSourceImage(auxImageSiemens(:,:,plane), [infoVolumeSpan1.ScalingFactorMmPixel1 infoVolumeSpan1.ScalingFactorMmPixel3], 1, graficar, fullFilename,'Y [mm]');
    
    % X axis (dimension 2: columns):
    fullFilename = sprintf('fwhm_z_span11_slice_%d', i);
    [fwhm_z_span11(plane), fwhm_z_fitted_span11(plane)] = getFwhmOfPointSourceImage(auxImageSpan11(:,:,plane), [infoVolumeSpan1.ScalingFactorMmPixel1 infoVolumeSpan1.ScalingFactorMmPixel3], 2, graficar, fullFilename,'Z [mm]');
    
    fullFilename = sprintf('fwhm_z_span1_slice_%d', i);
    [fwhm_z_span1(plane), fwhm_z_fitted_span1(plane)] = getFwhmOfPointSourceImage(auxImageSpan1(:,:,plane), [infoVolumeSpan1.ScalingFactorMmPixel1 infoVolumeSpan1.ScalingFactorMmPixel3], 2, graficar, fullFilename,'Z [mm]');
    
    fullFilename = sprintf('fwhm_z_stir_slice_%d', i);
    [fwhm_z_stir(plane), fwhm_z_fitted_stir(plane)] = getFwhmOfPointSourceImage(auxImageStir(:,:,plane), [infoVolumeSpan1.ScalingFactorMmPixel1 infoVolumeSpan1.ScalingFactorMmPixel3], 2, graficar, fullFilename,'Z [mm]');
    
    fullFilename = sprintf('fwhm_z_stir_slice_%d', i);
    [fwhm_z_siemens(plane), fwhm_z_fitted_siemens(plane)] = getFwhmOfPointSourceImage(auxImageSiemens(:,:,plane), [infoVolumeSpan1.ScalingFactorMmPixel1 infoVolumeSpan1.ScalingFactorMmPixel3], 2, graficar, fullFilename,'Z [mm]');
    %close all
end
%fwhm:
figure;
set(gcf, 'Position', [50 50 1600 1200]);
set(gcf, 'Name', 'Sagital Resolution Analysis');
planes = size(reconVolumeSpan11,2);
subplot(1,2,1);
plot(1:planes, fwhm_y_span11, 1:planes, fwhm_y_span1, 1:planes, fwhm_y_stir, 1:planes, fwhm_y_siemens, 'LineWidth', 2);
legend('Span 11', 'Span 1', 'STIR', 'Siemens');
xlabel('Slice');
ylabel('FWHM Y [mm]');
title('Resolution in Y axis');
ylim([0 8]);
subplot(1,2,2);
plot(1:planes, fwhm_z_span11, 1:planes, fwhm_z_span1, 1:planes, fwhm_z_stir, 1:planes, fwhm_z_siemens, 'LineWidth', 2);
legend('Span 11', 'Span 1', 'STIR', 'Siemens');
xlabel('Slice');
ylabel('FWHM Z [mm]');
title('Resolution in Z axis');
ylim([0 10]);

% fitted gaussian
figure;
set(gcf, 'Position', [50 50 1600 1200]);
set(gcf, 'Name', 'Sagital Resolution Analysis');
planes = size(reconVolumeSpan11,2);
subplot(1,2,1);
plot(1:planes, fwhm_y_fitted_span11, 1:planes, fwhm_y_fitted_span1, 1:planes, fwhm_y_fitted_stir, 1:planes, fwhm_y_fitted_siemens, 'LineWidth', 2);
legend('Span 11', 'Span 1', 'STIR', 'Siemens');
xlabel('Slice');
ylabel('FWHM of Fitted Gaussian in Y [mm]');
title('Resolution in Y axis with Fitted Gaussian');
ylim([0 8]);
subplot(1,2,2);
plot(1:planes, fwhm_z_fitted_span11, 1:planes, fwhm_z_fitted_span1, 1:planes, fwhm_z_fitted_stir, 1:planes, fwhm_z_fitted_siemens, 'LineWidth', 2);
legend('Span 11', 'Span 1', 'STIR', 'Siemens');
xlabel('Slice');
ylabel('FWHM of Fitted Gaussian in Z [mm]');
title('Resolution in Z axis with Fitted Gaussian');
ylim([0 10]);
%% PLOT FOR PUBLICATION
figure;
set(gcf, 'Position', [0 0 2000 800]);
subplot(1,3,1);
plot(coordZpet, fwhm_x_span11, coordZpet, fwhm_x_span1, coordZpet, fwhm_x_stir, coordZpet, fwhm_x_siemens, 'LineWidth', 2);
legend('Span 11', 'Span 1', 'STIR', 'Siemens', 'Location', 'SouthEast');
xlabel('Z [mm]');
ylabel('FWHM in X [mm]');
title('Resolution in X Axis');
ylim([3 15]);

subplot(1,3,2);
plot(coordXpet, fwhm_y_span11, coordXpet, fwhm_y_span1, coordXpet, fwhm_y_stir, coordXpet, fwhm_y_siemens, 'LineWidth', 2);
legend('Span 11', 'Span 1', 'STIR', 'Siemens', 'Location', 'SouthEast');
xlabel('X [mm]');
ylabel('FWHM in Y [mm]');
title('Resolution in Y Axis');
ylim([3 9]);

subplot(1,3,3);
plot(coordXpet, fwhm_z_span11, coordXpet, fwhm_z_span1, coordXpet, fwhm_z_stir, coordXpet, fwhm_z_siemens, 'LineWidth', 2);
legend('Span 11', 'Span 1', 'STIR', 'Siemens', 'Location', 'SouthEast');
xlabel('X [mm]');
ylabel('FWHM in Z [mm]');
title('Resolution in Z Axis');
ylim([3 11]);

fullFilename = [outputPath 'OverallFwhm'];
saveas(gca, [fullFilename], 'tif');
set(gcf,'PaperPositionMode','auto');    % Para que lo guarde en el tamaño modificado.
frame = getframe(gca);
imwrite(frame.cdata, [fullFilename '.png']);
saveas(gca, [fullFilename], 'epsc');
