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
filename = 'Nema_Osem21_HR_iter';

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
for i = 1 : numIterations
    fullFilename = [span11Path sprintf('%s_%d.h33', filenameSpan11, i-1)];
    reconVolumeSpan11 = interfileRead(fullFilename); 
    % Mean Value of Slice / Std In Slice. For the whole volume.
    aux2dArray = reshape(reconVolumeSpan11, [size(reconVolumeSpan11,1)*size(reconVolumeSpan11,2) size(reconVolumeSpan11,3)]);
    snrSpan11(i) = mean(aux2dArray(:))./std(aux2dArray(:));
    snrSpan11Masked(i) = mean(reconVolumeSpan11(maskPhantom))./std(reconVolumeSpan11(maskPhantom));
    
    fullFilename = [stirPath sprintf('%s_%d.hv', filenameStir, i)];
    reconVolumeStir = interfileRead(fullFilename); 
    % Mean Value of Slice / Std In Slice. For the whole volume.
    aux2dArray = reshape(reconVolumeStir, [size(reconVolumeStir,1)*size(reconVolumeStir,2) size(reconVolumeStir,3)]);
    snrStir(i) = mean(aux2dArray(:))./std(aux2dArray(:));
    snrStirMasked(i) = mean(reconVolumeStir(maskPhantom))./std(reconVolumeStir(maskPhantom));
end
figure;
plot([snrSpan11; snrSpan11]')

figure;
plot([snrSpan11Masked; snrStirMasked]')
reconVolumeSpan1 = interfileRead([span1Path filename]);
infoVolumeSpan1 = interfileinfo([span1Path filename]);
reconVolumeStir = interfileRead([stirPath filename]);
%% MAXIMUM INTENSITY PROJECTION
% Get MIPs in the three axes:
[mipTransverseSpan11, mipCoronalSpan11, mipSagitalSpan11] = showMaximumIntensityProjections(reconVolumeSpan11);
set(gcf, 'Name', 'Maximum Intensity Projection Span11');
[mipTransverseSpan1, mipCoronalSpan1, mipSagitalSpan1] = showMaximumIntensityProjections(reconVolumeSpan1);
set(gcf, 'Name', 'Maximum Intensity Projection Span 1');
[mipTransverseStir, mipCoronalStir, mipSagitalStir] = showMaximumIntensityProjections(reconVolumeStir);
set(gcf, 'Name', 'Maximum Intensity Projection Stir');
%% SNR PER SLICE
% Mean Value of Slice / Std In Slice. For the whole volume.
aux2dArray = reshape(reconVolumeSpan11, [size(reconVolumeSpan11,1)*size(reconVolumeSpan11,2) size(reconVolumeSpan11,3)]);
snrSpan11 = mean(aux2dArray)./std(aux2dArray);

aux2dArray = reshape(reconVolumeSpan1, [size(reconVolumeSpan1,1)*size(reconVolumeSpan1,2) size(reconVolumeSpan1,3)]);
snrSpan1 = mean(aux2dArray)./std(aux2dArray);

aux2dArray = reshape(reconVolumeStir, [size(reconVolumeStir,1)*size(reconVolumeStir,2) size(reconVolumeStir,3)]);
snrStir = mean(aux2dArray)./std(aux2dArray);

h = figure;
plot([snrSpan11', snrSpan1', snrStir'], 'LineWidth', 2);
legend('Span 11', 'Span 1', 'Stir');
title('SNR per Slice in the Whole Volume');
xlabel('Slice');
ylabel('Mean/Std');

% Using a phantom mask:
figure;
bar([mean(reconVolumeSpan11(maskPhantom))./std(reconVolumeSpan11(maskPhantom)), mean(reconVolumeSpan1(maskPhantom))./std(reconVolumeSpan1(maskPhantom)), mean(reconVolumeStir(maskPhantom))./std(reconVolumeStir(maskPhantom))]);
legend('Span 11', 'Span 1', 'Stir');
title('SNR of Whole Masked Imaged');
ylabel('Mean/Std');
%% CONTRAST RECOVER AND NOISE
centralSlice = 81;
sizePixel_mm = [infoVolumeSpan11.ScalingFactorMmPixel1 infoVolumeSpan11.ScalingFactorMmPixel2 infoVolumeSpan11.ScalingFactorMmPixel3];
center_mm = [2 2];
[contrastRecoverySpan11, desvioBackgroundSpan11, desvioNormBackgroundSpan11, meanLungRoiSpan11, relativeLungErrorSpan11] = procImagesQualityPhantomColdSpheres(reconVolumeSpan11, sizePixel_mm, center_mm, centralSlice, 1);
[contrastRecoverySpan1, desvioBackgroundSpan1, desvioNormBackgroundSpan1, meanLungRoiSpan1, relativeLungErrorSpan1] = procImagesQualityPhantomColdSpheres(reconVolumeSpan1, sizePixel_mm, center_mm, centralSlice, 1);
[contrastRecoveryStir, desvioBackgroundStir, desvioNormBackgroundStir, meanLungRoiStir, relativeLungErrorStir, radioEsferas_mm, centers_pixels] = procImagesQualityPhantomColdSpheres(reconVolumeStir, sizePixel_mm, center_mm, centralSlice, 1);
figure;
set(gcf, 'Name', 'Recovery Contrast');
set(gcf, 'Position', [50 50 1600 1200]);
bar([contrastRecoverySpan11' contrastRecoverySpan1' contrastRecoveryStir']);
legend('Span 11', 'Span 1', 'Stir');
xlabel('Spheres [mm]');
%set(gca, 'XtickLabel', num2str((radioEsferas_mm(1:end-1))));
ylabel('Recovery Contrast [%]');
title('Cold Spheres Recovery Contrast');

figure;
set(gcf, 'Name', 'Normalized Std Dev for Different ROI sizes');
set(gcf, 'Position', [50 50 1600 1200]);
bar([desvioNormBackgroundSpan11' desvioNormBackgroundSpan1' desvioNormBackgroundStir']);
legend('Span 11', 'Span 1', 'Stir');
xlabel('ROIs Size');
%xlabel('ROIs Radius [mm]');
ylabel('Recovery Contrast [%]');
title('Background Norm Std Dev');

%% MEAN VALUE OF PHANTOM OF SLICES
slices = 19 : 111;
% Sum of Slices:
sumSlicesSpan11 = mean(reconVolumeSpan11(:,:,slices),3);
sumSlicesSpan1 = mean(reconVolumeSpan1(:,:,slices),3);
sumSlicesStir = mean(reconVolumeStir(:,:,slices),3);

figure;
set(gcf, 'Name', 'Mean Value of Phantom Slices');
set(gcf, 'Position', [50 50 1600 1200]);
subplot(1,3,1);
imshow(sumSlicesSpan11, [0 max(max(sumSlicesSpan11))]);
% Aplico el colormap
colormap(hot);
% Cambio las leyendas a las unidades que me interesan:
hcb = colorbar;
%set(hcb, 'YTickLabelMode', 'manual');
set(hcb, 'FontWeight', 'bold');
title('Span 11');

subplot(1,3,2);
imshow(sumSlicesSpan1, [0 max(max(sumSlicesSpan1))]);
% Aplico el colormap
colormap(hot);
% Cambio las leyendas a las unidades que me interesan:
hcb = colorbar;
%set(hcb, 'YTickLabelMode', 'manual');
set(hcb, 'FontWeight', 'bold');
title('Span 1');

subplot(1,3,3);
imshow(sumSlicesStir, [0 max(max(sumSlicesStir))]);
% Aplico el colormap
colormap(hot);
% Cambio las leyendas a las unidades que me interesan:
hcb = colorbar;
%set(hcb, 'YTickLabelMode', 'manual');
set(hcb, 'FontWeight', 'bold');
title('Stir');

% Profiles normalized to mean value of sumed images:
meanValueSpan11 = mean(mean(sumSlicesSpan11));
meanValueSpan1 = mean(mean(sumSlicesSpan1));
meanValueStir = mean(mean(sumSlicesStir));
profiles = [round(size(sumSlicesSpan11,1)/2) round(size(sumSlicesSpan11,2)/2); 100 95; 160 170];
h = figure;
set(gcf, 'Name', 'Profiles Normalized to Mean Value');
set(gcf, 'Position', [50 50 1600 1200]);
for i = 1 : size(profiles,1)
    subplot(size(profiles,1),size(profiles,2),(i-1)*size(profiles,2)+1);
    plot([sumSlicesSpan11(profiles(i,1),:)./meanValueSpan11; sumSlicesSpan1(profiles(i,1),:)./meanValueSpan1; sumSlicesStir(profiles(i,1),:)./meanValueStir]', 'LineWidth', 2);
    title(sprintf('Profile Y = %d', profiles(i,1))); 
    legend('Span 11', 'Span 1', 'Stir');
end
for j = 1 : size(profiles,1)
    subplot(size(profiles,1),size(profiles,2),j*size(profiles,2));
    plot([sumSlicesSpan11(:,profiles(j,2))./meanValueSpan11 sumSlicesSpan1(:,profiles(j,2))./meanValueSpan1 sumSlicesStir(:,profiles(j,2))./meanValueStir], 'LineWidth', 2);
    title(sprintf('Profile X = %d', profiles(j,2))); 
    legend('Span 11', 'Span 1', 'Stir');
end
%% PROFILES CENTERED IN EACH SPHERE
% Profiles normalized to mean value of sumed images:
h = figure;
set(gcf, 'Name', 'Profiles Normalized to Mean Value');
set(gcf, 'Position', [50 50 1600 1200]);
for i = 1 : size(centers_pixels,1)
    subplot(size(centers_pixels,1),size(centers_pixels,2),(i-1)*size(centers_pixels,2)+1);
    plot([reconVolumeSpan11(:,centers_pixels(i,1), centralSlice)./meanValueSpan11 reconVolumeSpan1(:,centers_pixels(i,1), centralSlice)./meanValueSpan1 reconVolumeStir(:,centers_pixels(i,1), centralSlice)./meanValueStir], 'LineWidth', 2);
    title(sprintf('Profile Y = %d', centers_pixels(i,1))); 
    legend('Span 11', 'Span 1', 'Stir');
end
for j = 1 : size(centers_pixels,1)
    subplot(size(centers_pixels,1),size(centers_pixels,2),j*size(centers_pixels,2));
    plot([reconVolumeSpan11(centers_pixels(j,2),:, centralSlice)./meanValueSpan11; reconVolumeSpan1(centers_pixels(j,2),:, centralSlice)./meanValueSpan1; reconVolumeStir(centers_pixels(j,2),:, centralSlice)./meanValueStir]', 'LineWidth', 2);
    title(sprintf('Profile X = %d', centers_pixels(j,2))); 
    legend('Span 11', 'Span 1', 'Stir');
end
%% MULTIRESOLUTION OF CENTRAL SLICE
centralSlice = 81;
pixelSize_mm = 2.08626;
% Std Dev of each filter:
filterStdDev_mm = pixelSize_mm : pixelSize_mm : 40;
filterStdDev_pixels = filterStdDev_mm ./ pixelSize_mm;
% Size of the filter (for 3 sigmas: 3*filterStdDev_pixels*2 + 1):
filterSize_pixels = round(3*filterStdDev_pixels*2 + 1);

for i = 1 : numel(filterSize_pixels)
    filter = fspecial('gaussian',[filterSize_pixels(i) filterSize_pixels(i)],filterStdDev_pixels(i));
    % Filter the three images:
    filteredImage{i,1} = imfilter(reconVolumeSpan11(:,:,centralSlice), filter);
    filteredImage{i,2} = imfilter(reconVolumeSpan1(:,:,centralSlice), filter);
    filteredImage{i,3} = imfilter(reconVolumeStir(:,:,centralSlice), filter);
end

% Several figure with 4 filtered image per data set:
numImagesPerFigure = 4;
k = 1;
for i = 1 : ceil(numel(filterSize_pixels)/numImagesPerFigure)
    figure;
    set(gcf, 'Name', 'Filtered Images with Gaussian Filter');
    set(gcf, 'Position', [50 50 1600 1200]);
    for j = 1 : numImagesPerFigure
        subplot(numImagesPerFigure,3,1+3*(j-1));
        imshow(filteredImage{k,1}, [0 max(max(filteredImage{k,1}))]);
        % Aplico el colormap
        colormap(hot);
        % Cambio las leyendas a las unidades que me interesan:
        hcb = colorbar;
        %set(hcb, 'YTickLabelMode', 'manual');
        set(hcb, 'FontWeight', 'bold');
        if j == 1
            title(sprintf('Span 11'));
        end
        ylabel(sprintf('Std Dev %f mm', filterStdDev_mm(k)));
        % Using a phantom mask:
        snrPerFilterSpan11(k) = mean(filteredImage{k,1}(maskPhantom(:,:,centralSlice)))./std(filteredImage{k,1}(maskPhantom(:,:,centralSlice)));
        
        subplot(numImagesPerFigure,3,2+3*(j-1));
        imshow(filteredImage{k,2}, [0 max(max(filteredImage{k,2}))]);
        % Aplico el colormap
        colormap(hot);
        % Cambio las leyendas a las unidades que me interesan:
        hcb = colorbar;
        %set(hcb, 'YTickLabelMode', 'manual');
        set(hcb, 'FontWeight', 'bold');
        if j == 1
            title(sprintf('Span 1'));
        end
        % Using a phantom mask:
        snrPerFilterSpan1(k) = mean(filteredImage{k,2}(maskPhantom(:,:,centralSlice)))./std(filteredImage{k,2}(maskPhantom(:,:,centralSlice)));
        
        subplot(numImagesPerFigure,3,3+3*(j-1));
        imshow(filteredImage{k,3}, [0 max(max(filteredImage{k,3}))]);
        % Aplico el colormap
        colormap(hot);
        % Cambio las leyendas a las unidades que me interesan:
        hcb = colorbar;
        %set(hcb, 'YTickLabelMode', 'manual');
        set(hcb, 'FontWeight', 'bold');
        if j == 1
            title(sprintf('Stir'));
        end
        % Using a phantom mask:
        snrPerFilterStir(k) = mean(filteredImage{k,3}(maskPhantom(:,:,centralSlice)))./std(filteredImage{k,3}(maskPhantom(:,:,centralSlice)));
        
        k = k+1;
        if k > numel(filterSize_pixels)
            break;
        end
    end
    
end
figure;
set(gcf, 'Position', [50 50 1600 1200]);
plot(filterStdDev_mm, snrPerFilterSpan11, filterStdDev_mm, snrPerFilterSpan1, filterStdDev_mm, snrPerFilterStir);
xlabel('Std Dev [mm]');
ylabel('SNR (mean/std)');
title('SNR of Central Slice for Different Filters');