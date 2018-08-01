function [scatter, structSizeSino, mask] = estimateScatterWithStir(emissionImage, attenuationMap, pixelSize_mm, emissionSinogram, randoms, ncf, acfs, structSizeSino3d, outputPath, scriptsPath, thresholdForTail)
% This function creates the scatter estimate.
disp('estimateScatterWithStir...')
if ~isdir(outputPath)
    mkdir(outputPath);
end
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

if nargin == 8
    thresholdForTail = 1.01; % default value for the tail.
end
% This function needs an emission image previously reconstructed
% without scatter correction and the attenuation map. 
% First apply the normalization and attenuation to the emission data:

% Project attrneuation map: it's already avaiable the numap.

% Write emission sinogram:
emissionSinogramCorrected = (emissionSinogram - randoms).*ncf;
emissionSinogramCorrected(emissionSinogramCorrected < 0) = 0;
filenameEmissionSinogram = [outputPath pathBar 'emissionSinogramCorrected'];
interfileWriteStirSino(emissionSinogramCorrected, filenameEmissionSinogram, structSizeSino3d);
filenameEmissionSinogram = [filenameEmissionSinogram '.hs'];

% Write emission image:
emissionImageFilename = [outputPath pathBar 'emissionImage'];
interfileWriteStirImage(single(emissionImage), emissionImageFilename, pixelSize_mm);
emissionImageFilename = [emissionImageFilename '.hv'];


% Write the attenuation correction factors:
filenameAcf = [outputPath pathBar 'acf'];
filenameAcfZoomed = [filenameAcf '_zoomed'];
interfileWriteStirSino(acfs, filenameAcf, structSizeSino3d);
filenameAcf = [filenameAcf '.hs'];
filenameAcfZoomed = [filenameAcf '_zoomed'];

% Write attenuation map image:
filenameAttenuationMap = [outputPath pathBar 'attenuationMap'];
interfileWriteStirImage(single(attenuationMap), filenameAttenuationMap, pixelSize_mm);
filenameAttenuationMap = [filenameAttenuationMap '.hv'];

% Then create tail maks:
% create_tail_mask_from_ACFs --ACF-filename <projdata> \
% --ACF-threshold <number (1.01)> \
% --output-filename <projdata> \
% --safety-margin <0>
filenameTailForScatterScale = [outputPath 'tail_mask.hs'];
% The total attenuation must be the same:
system(['create_tail_mask_from_ACFs --ACF-filename ' filenameAcf...
    ' --ACF-threshold ' num2str(thresholdForTail) ' --output-filename ' ...
    filenameTailForScatterScale ' --safety-margin 0 ' ]);
% Read the mask for verification:
mask = interfileReadStirSino(filenameTailForScatterScale);

% Reconstruct emission without scatter
% is already available.

% Construct sub sampled sinogram:
% Zoom:
% zoom_image zoomed.hv image_mu.hv 21 .25 0 0 5 .1626
filenameZoomedAttenuationMap = [filenameAttenuationMap(1:end-3) '_zoomed.hv'];
scaleFactorXY = 1/4;
scaleFactorZ = 1/3;
headerInfo = getInfoFromInterfile(filenameAttenuationMap);
imageSizeAtten_pixels = [headerInfo.MatrixSize1 headerInfo.MatrixSize2 headerInfo.MatrixSize3];
sizeXY = round(headerInfo.MatrixSize1 * scaleFactorXY);
sizeZ = round(headerInfo.MatrixSize3 * scaleFactorZ);
system(['zoom_image ' [filenameZoomedAttenuationMap ' '] [filenameAttenuationMap ' ']...
        ' ' num2str(sizeXY) ' ' num2str(scaleFactorXY) ' 0 0 ' num2str(sizeZ) ' ' num2str(scaleFactorZ)]);
% The total attenuation must be the same:
system(['stir_math --accumulate --including-first --times-scalar ' num2str(scaleFactorZ) ' --times-scalar ' num2str(scaleFactorXY) ' --times-scalar ' num2str(scaleFactorXY) ' '...
    ' ' filenameZoomedAttenuationMap]);

% Call scatter estimate, we need an ACTIVITY_IMAGE, a DENSITY_IMAGE, a
% LOW_RESOLUTION_DENSITY_IMAGE, SUBSAMPLED_PROJECTION_DATA_TEMPLATE and
% OUTPUT_PROJECTION_DATA.
% ACTIVITY_IMAGE : image reconstructed without scatter correction 
filenameCoarseScatter = [outputPath 'scatterCoarse.hs'];
system(['ACTIVITY_IMAGE=' emissionImageFilename ' '...
        'DENSITY_IMAGE=' [filenameAttenuationMap ' ']...
        'LOW_RESOLUTION_DENSITY_IMAGE=' [filenameZoomedAttenuationMap ' ']...
        'SUBSAMPLED_PROJECTION_DATA_TEMPLATE=' [scriptsPath pathBar 'template_span1_subsampled.hs ' ]...
        'OUTPUT_PROJECTION_DATA=' [filenameCoarseScatter ' '] ...
        'estimate_scatter ' [scriptsPath pathBar 'scatter_orig.par']]);

% THE SCALING AND FITTING I DO IT WITH MY OWN CODE
%     upsample_and_fit_single_scatter \
%         --min-scale-factor <number> \
%         --max-scale-factor <number> \
%         27--remove-interleaving <1|0> \
%         --half-filter-width <number> \
%         --output-filename <filename> \
%         --data-to-fit <filename> \
%         --data-to-scale <filename> \
%         --weights <filename>
filenameScatter = [outputPath 'scatter.hs'];
% The total attenuation must be the same:
system(['upsample_and_fit_single_scatter --min-scale-factor 0.1 --max-scale-factor 6.0 --remove-interleaving 1 --half-filter-width 2 '...
    '--output-filename ' filenameScatter ' --data-to-fit ' [filenameEmissionSinogram]...
    ' --data-to-scale ' filenameCoarseScatter ' --weights ' filenameTailForScatterScale]);
[scatter structSizeSino] = interfileReadStirSino([outputPath 'scatter.hs']);
% scatterCoarse = interfileReadStirSino([outputPath 'scatterCoarse.hs']);
% % Size of the coarse scatter:
% [X Y Z] = meshgrid(1:size(scatterCoarse,2), 1:size(scatterCoarse,1), 1:size(scatterCoarse,3));
% % Output Size:
% [X2 Y2 Z2] = meshgrid(1:(size(scatterCoarse,2)-1)/(structSizeSino3d.numTheta-1):size(scatterCoarse,2), 1:(size(scatterCoarse,1)-1)/(structSizeSino3d.numR-1):size(scatterCoarse,1), 1:(size(scatterCoarse,3)-1)/(structSizeSino3d.numZ-1):size(scatterCoarse,3));
% %Interpolate to get the direct sinograms in the desired size:
% scatterDirect = interp3(X,Y,Z,scatterCoarse,X2,Y2,Z2);
% % create an span 1 sinogram:
% structSizeSino3dSpan1 = getSizeSino3dFromSpan(structSizeSino3d.numR, structSizeSino3d.numTheta, structSizeSino3d.numZ, structSizeSino3d.rFov_mm, structSizeSino3d.zFov_mm,1, structSizeSino3d.maxAbsRingDiff);
% % Set direct sinograms:
% scatterSpan1 = zeros(structSizeSino3dSpan1.numR, structSizeSino3dSpan1.numTheta, sum(structSizeSino3dSpan1.sinogramsPerSegment));
% scatterSpan1(:,:,1:structSizeSino3d.numZ) = scatterDirect;
% % To complete the 3d sinogram I backproject and then project:
% pixelSizeAuxImage_mm = [pixelSize_mm(1) pixelSize_mm(2)  pixelSize_mm(3)*size(emissionImage,3)/structSizeSino3d.numZ];
% imageScatter = BackprojectMmr(scatterSpan1, [size(emissionImage,1) size(emissionImage,2) structSizeSino3d.numZ], pixelSizeAuxImage_mm, outputPath, 1, 0,0, 1);
% sensitivityImage = BackprojectMmr(ones(size(scatterSpan1)), [size(emissionImage,1) size(emissionImage,2) structSizeSino3d.numZ], pixelSizeAuxImage_mm, outputPath, 1, 0,0, 1);
% imageScatter = imageScatter ./ sensitivityImage;
% % And project to the desired size:
% [scatter, structSizeSinogram] = ProjectMmr(imageScatter, pixelSizeAuxImage_mm, outputPath, structSizeSino3d.span, 0,0, 1);

% Rescaling the scatter because stir is not doing it well (maybe because of
% all the zeros)
% To change the scalling, find the profile with more counts (it is possible
% to smash a few angles), filter, and scale all with that:
% aux = conv(true_2d_sino(:,126), ones(15,1)./15, 'same');
% outside2d_indices = acfs_2d_sino <= 1.05;
emissionSinogramCorrected = (emissionSinogram - randoms).*ncf;   % Recompute the corrected sinogram without forcing zeros (this allows to have a zero mean in the background).
for sinogram_to_scale = 1:size(scatter,3)       
    maskNonZeros = ncf(:,:,sinogram_to_scale) ~= 0;
    scat_2d_sino = maskNonZeros .* scatter(:,:,sinogram_to_scale);
    true_2d_sino = maskNonZeros .* emissionSinogramCorrected(:,:,sinogram_to_scale);
    acfs_2d_sino = acfs(:,:,sinogram_to_scale);
    outside2d_indices = acfs_2d_sino <= thresholdForTail; % Regions
    
    scale_factor = sum(true_2d_sino(outside2d_indices)) / sum(scat_2d_sino(outside2d_indices));
    %scan_2d_sino = scatter_with_norm_problem(:,:,sinogram_to_scale);
    %figure;plot([true_2d_sino(:,126)  scat_2d_sino(:,126)])
    scatter(:,:,sinogram_to_scale) = scatter(:,:,sinogram_to_scale) .* scale_factor;
end
