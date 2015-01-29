%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 15/01/2015
%  *********************************************************************
%  This example calls the getIntfSinogramsFromUncompressedMmr to convert an
%  uncompressed Siemenes interfile acquisition into an interfil APIRL
%  comptaible sinogram. Then read the attenuation map of the acquisition
%  and the normalization file of the scanner, applies all the corrections
%  to new span 11 and direct sinograms in order to be reconstructed with
%  apirl.
clear all 
close all
%% PATHS FOR EXTERNAL FUNCTIONS AND RESULTS
addpath('/workspaces/Martin/KCL/Biograph_mMr/mmr');
apirlPath = '/workspaces/Martin/PET/apirl-code/trunk/';
addpath(genpath([apirlPath '/matlab']));
setenv('PATH', [getenv('PATH') ':/workspaces/Martin/PET/apirl-code/trunk/build/debug/bin']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') ':/workspaces/Martin/PET/apirl-code/trunk/build/debug/bin']);
outputPath = '/workspaces/Martin/KCL/Biograph_mMr/mmr/5hr_ge68/';
%setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') ':/usr/lib/x86_64-linux-gnu/']);
%% READING THE SINOGRAMS
% Read the sinograms:
%filenameUncompressedMmr = '/workspaces/Martin/KCL/Biograph_mMr/mmr/test.s';
%outFilenameIntfSinograms = '/workspaces/Martin/KCL/Biograph_mMr/mmr/testIntf';
filenameUncompressedMmr = '/workspaces/Martin/KCL/Biograph_mMr/mmr/5hr_ge68/cylinder_5hours.s';
outFilenameIntfSinograms = '/workspaces/Martin/KCL/Biograph_mMr/mmr/5hr_ge68/cylinder_5hoursIntf';
[sinogram, delayedSinogram, structSizeSino3d] = getIntfSinogramsFromUncompressedMmr(filenameUncompressedMmr, outFilenameIntfSinograms);

% Read the normalization factors:
filenameRawData = '/workspaces/Martin/KCL/Biograph_mMr/mmr/Norm_20141008101010.n';
[componentFactors, componentLabels]  = readmMrComponentBasedNormalization(filenameRawData, 1);
%% CREATE SINOGRAMS3D SPAN 11
% Create sinogram span 11:
michelogram = generateMichelogramFromSinogram3D(sinogram, structSizeSino3d);
structSizeSino3dSpan11 = getSizeSino3dFromSpan(structSizeSino3d.numTheta, structSizeSino3d.numR, structSizeSino3d.numZ, ...
    structSizeSino3d.rFov_mm, structSizeSino3d.zFov_mm, 11, structSizeSino3d.maxAbsRingDiff);
sinogramSpan11 = reduceMichelogram(michelogram, structSizeSino3dSpan11.sinogramsPerSegment, structSizeSino3dSpan11.minRingDiff, structSizeSino3dSpan11.maxRingDiff);
clear michelogram
% Write to a file in interfile format:
outputSinogramName = [outputPath '/sinogramSpan11'];
interfileWriteSino(single(sinogramSpan11), outputSinogramName, structSizeSino3dSpan11.sinogramsPerSegment, structSizeSino3dSpan11.minRingDiff, structSizeSino3dSpan11.maxRingDiff);
%% DIRECT SINOGRAMS
numSinos2d = structSizeSino3d.sinogramsPerSegment(1);
structSizeSino2d = getSizeSino2Dstruct(structSizeSino3d.numTheta, structSizeSino3d.numR, ...
    numSinos2d, structSizeSino3d.rFov_mm, structSizeSino3d.zFov_mm);
directSinograms = single(sinogram(:,:,1:structSizeSino2d.numZ));
% Write to a file in interfile formar:
outputSinogramName = [outputPath '/directSinograms'];
interfileWriteSino(single(directSinograms), outputSinogramName);
%% GAPS
mean_sinogram = mean(sinogram, 3);
gaps_sinogram = mean_sinogram;   nongap_indices  = find(mean_sinogram > 0);
gaps_sinogram(nongap_indices) = 1.0;
%% NORMALIZATION FACTORS
% A set of 2d sinograms with normalization effects is generated. The
% geomtric normalization and crystal interference are the same for each
% sinogram.

% Geometric Normalization:
normSinoGeom = 1./ imresize(double(componentFactors{1}'), [structSizeSino3d.numTheta structSizeSino3d.numR]);
figure;
imshow((1./normSinoGeom) ./max(max((1./normSinoGeom))));
% Crystal interference, its a pattern that is repeated peridoically:
crystalInterf = double(componentFactors{2});
crystalInterf = 1./repmat(crystalInterf,structSizeSino3d.numTheta/size(crystalInterf,1),1);
% Geometric Normalization and crystal interference:
figure;
imshow(crystalInterf ./max(max(crystalInterf)));
% Generate one norm factor:
normSinoGeomInterf = normSinoGeom .* crystalInterf;
title('Sinogram Correction For Crystal Interference and Geometry');

% Generate the sinogram for crystal efficencies. To test, each detector has
% a different gain:
%sinoEfficencies = createSinogram2dFromDetectorsEfficency(componentFactors{3}(:,10), structSizeSino2d, 1);

% Axial correction factors:
axialCorrectionFactors = componentFactors{4}.*componentFactors{8};
%% ATTENUATION CORRECTION - PICK A OR B AND COMMENT THE NOT USED 
%% COMPUTE THE ACFS (OPTION A)
% Read the phantom and then generate the ACFs with apirl:
imageSize_pixels = [344 344 127];
filenameAttenMap = '/workspaces/Martin/KCL/Biograph_mMr/Mediciones/2601/interfile/PET_ACQ_16_20150116131121-0_PRR_1000001_20150126152442_umap_human_00.v';
fid = fopen(filenameAttenMap, 'r');
if fid == -1
    ferror(fid);
end
attenMap = fread(fid, imageSize_pixels(1)*imageSize_pixels(2)*imageSize_pixels(3), 'single');
attenMap = reshape(attenMap, imageSize_pixels);
fclose(fid);
% visualization
figure;
image = getImageFromSlices(attenMap, 12, 1, 0);
imshow(image, gray);
% Size of the image to cover the full fov:
sizeImage_mm = [structSizeSino3d.rFov_mm*2 structSizeSino3d.rFov_mm*2 structSizeSino3d.zFov_mm];
sizePixel_mm = sizeImage_mm ./ imageSize_pixels;

% Create ACFs of a computed phatoms with the linear attenuation
% coefficients:
acfFilename = [outputPath 'acfsSinogramSpan11'];
filenameSinogram = [outputPath 'sinogramSpan11'];
acfs3dSpan11 = createACFsFromImage(attenMap, sizePixel_mm, outputPath, acfFilename, filenameSinogram, structSizeSino3d, 1);
% Now the same for 2d sinograms:
acfFilename = [outputPath 'acfsDirectSinograms'];
filenameSinogram = [outputPath 'directSinograms'];
acfsDirectSinograms = createACFsFromImage(attenMap, sizePixel_mm, outputPath, acfFilename, filenameSinogram, structSizeSino2d, 1);
%% READ THE ACFS (OPTION B)
fid = fopen([acfFilename '.i33'], 'r');
[acfsDirectSinograms, count] = fread(fid, structSizeSino2d.numTheta*structSizeSino2d.numR*structSizeSino2d.numZ, 'single=>single');
acfsDirectSinograms = reshape(acfsDirectSinograms, [structSizeSino2d.numR structSizeSino2d.numTheta structSizeSino2d.numZ]);
% Matlab reads in a column-wise order that why angles are in the columns.
% We want to have it in the rows since APIRL and STIR and other libraries
% use row-wise order:
acfsDirectSinograms = permute(acfsDirectSinograms,[2 1 3]);
% Close the file:
fclose(fid);
%% VISUALIZATION OF DIRECT SINOGRAMS 
% SPAN1
imageOfDirectSinos = getImageFromSlices(sinogram(:,:,1: structSizeSino3d.sinogramsPerSegment(1)), 8, 1,0);
h = figure;
imshow(imageOfDirectSinos);
title('Direct Sinograms Span 1');
% SPAN11
imageOfDirectSinos = getImageFromSlices(sinogramSpan11(:,:,1: structSizeSino3d.sinogramsPerSegment(1)), 8, 1,0);
h = figure;
imshow(imageOfDirectSinos);
title('Direct Sinograms Span 11');
clear imageOfDirectSinos;
%% VISUALIZATION OF COUNTS PER SINOGRAM
% Sum of the sinograms:
countsPerSinogram = sum(sum(sinogram));
% Change vecor in 3rd dimension for 1st, to plot:
countsPerSinogram = permute(countsPerSinogram,[3 1 2]);
h = figure;
plot(countsPerSinogram);
title('Counts Per Sinogram Span 1');

countsPerSinogram = sum(sum(sinogramSpan11));
% Change vecor in 3rd dimension for 1st, to plot:
countsPerSinogram = permute(countsPerSinogram,[3 1 2]);
h = figure;
plot(countsPerSinogram);
title('Counts Per Sinogram Span 11 without Normalization');




%% CORRECTION OF SPAN 11 SINOGRAMS
if(numel(axialCorrectionFactors) ~= sum(structSizeSino3dSpan11.sinogramsPerSegment))
    perror('La cantidad de factores de correccion axial es distinto a la cantidad de sinograms');
end
% Correct sinograms 3d for normalization:
for i = 1 : sum(structSizeSino3dSpan11.sinogramsPerSegment)
    sinogramSpan11(:,:,i) = sinogramSpan11(:,:,i) .* normSinoGeomInterf .* axialCorrectionFactors(i);
end
countsPerSinogram = sum(sum(sinogramSpan11));
% Change vecor in 3rd dimension for 1st, to plot:
countsPerSinogram = permute(countsPerSinogram,[3 1 2]);
h = figure;
plot(countsPerSinogram);
title('Counts Per Sinogram Span 11 with Normalization');
% Write to a file in interfile formar:
outputSinogramName = [outputPath '/sinogramSpan11Normalized'];
interfileWriteSino(single(sinogramSpan11), outputSinogramName, structSizeSino3d.sinogramsPerSegment, structSizeSino3d.minRingDiff, structSizeSino3d.maxRingDiff);
%% CORRECTION OF DIRECT SINOGRAMS
countsPerSinogram = sum(sum(directSinograms));
% Change vecor in 3rd dimension for 1st, to plot:
countsPerSinogram = permute(countsPerSinogram,[3 1 2]);
h = figure;
plot(countsPerSinogram);
title('Counts Per Direct Sinogram without Normalization');
% Correct direct sinograms for normalization. In the axial direction, the
% first segment would be the direct sinograms, but there are 127 becuase is
% span 11. We use only the ones that cross a direct sinogram, but is not
% very precise. So we estimate them from the same GE acquisition. There is
% a pattern in the blocks in the gain of each pixel, this can be observed
% if we plot all of them normalized to the maximum:
%%
% Re arrange data for block:
numRingBlocks = 8;
numPixelsPerBlock = 8;
for ringBlock = 1 : numRingBlocks
    ringGain(:,ringBlock) = countsPerSinogram(((ringBlock-1)*numPixelsPerBlock+1):(ringBlock*numPixelsPerBlock));
    % Label for each block:
    labels{ringBlock} = sprintf('Ring Block %d', ringBlock);
end
figure;
subplot(3,2,1);
plot(ringGain);
legend(labels);
xlabel('Nº of Pixel in Block Ring');
ylabel('Counts in Pixel');
subplot(3,2,2);
for pixels = 1 : numPixelsPerBlock
    labelsPixels{pixels} = sprintf('Pixel %d', pixels);
end
plot(ringGain');
xlabel('Nº of Block');
ylabel('Counts in Pixels');
legend(labelsPixels);

subplot(3,2,3);
% Gain for each block:
for ringBlock = 1 : numRingBlocks
    ringGainNormInPosInBlock(:,ringBlock) = ringGain(:,ringBlock)./max(ringGain(:,ringBlock));
end
plot(ringGainNormInPosInBlock');
legend(labels);
xlabel('Nº of Pixel in Block Ring');
ylabel('Counts in Pixel');
subplot(3,2,4);
% Normalize to get other info:
for pixels = 1 : numPixelsPerBlock
    labelsPixels{pixels} = sprintf('Pixel %d', pixels);
    ringGainNormInBlock(pixels,:) = ringGain(pixels,:)./mean(ringGain(pixels,:));
    %ringGainNormInBlock(pixels,:) = [ringGain(pixels,1:4) ringGain(9-pixels,5:8)]./max(ringGain(pixels,:));
end
plot(ringGainNormInBlock');
xlabel('Nº of Block');
ylabel('Counts in Pixels');
legend(labelsPixels);

subplot(3,2,5);
gainPerPixel = mean(ringGain,2);
plot(1:numPixelsPerBlock,gainPerPixel);
xlabel('Nº of Pixel in Block Ring');
ylabel('Counts in Pixels');
legend(labelsPixels);

subplot(3,2,6);
% Gain for each block:
gainPerBlock = mean(ringGain,1);
plot(1:8,(ringGain(:,8)-ringGain(:,1))',1:8,ringGainNormInBlock(8,:)-ringGainNormInBlock(1,:));

xlabel('Nº of Block');
ylabel('Counts in Pixel');
% Compute block dependent gain by averaging
%%
for i = 1 : structSizeSino2d.numZ
    directSinogramsCorrected(:,:,i) = single(directSinograms(:,:,i)) .* normSinoGeomInterf .* 1./ componentFactors{8}(2*i-1);% axialCorrectionFactors(2*i);
end
% Plot
h = figure;image = getImageFromSlices(directSinogramsCorrected,10,1,1);
imshow(image);
title('Direct Sinograms with Normalization');
% Write to a file in interfile formar:
outputSinogramName = [outputPath '/directSinogramsNormalized'];
interfileWriteSino(single(directSinogramsCorrected), outputSinogramName);

countsPerSinogram = sum(sum(directSinogramsCorrected));
% Change vecor in 3rd dimension for 1st, to plot:
countsPerSinogram = permute(countsPerSinogram,[3 1 2]);
h = figure;
plot(countsPerSinogram);
title('Counts Per Direct Sinogram with Normalization');
 
% Correct for attenuation:
% Correct direct sinograms for normalization:
for i = 1 : structSizeSino2d.numZ
    directSinogramsCorrected(:,:,i) = directSinogramsCorrected(:,:,i) .* acfsDirectSinograms(:,:,i);
end

h = figure;image = getImageFromSlices(directSinogramsCorrected,10,1,1);
imshow(image);
title('Direct Sinograms with Normalization and Attenuation');
% Write to a file in interfile formar:
outputSinogramName = [outputPath '/directSinogramsNormAttenCorrected'];
interfileWriteSino(single(directSinogramsCorrected), outputSinogramName);
%% GENERATE ATTENUATION AND NORMALIZATION FACTORS FOR APIRL
atteNormFactorsDirect = zeros(size(directSinograms));
for i = 1 : structSizeSino2d.numZ
    atteNormFactorsDirect(:,:,i) = gaps_sinogram .* normSinoGeomInterf .* axialCorrectionFactors(i) .* acfsDirectSinograms(:,:,i);
end
outputSinogramName = [outputPath '/attenNormFactorsForDirectSinograms'];
interfileWriteSino(single(atteNormFactorsDirect), outputSinogramName);
h = figure;image = getImageFromSlices(atteNormFactorsDirect,10,1,1);
imshow(image);
title('Normalization and Attenuation Factors for Direct Sinograms');

%% 2D RECONSTRUCTION OF DIRECT SINOGRAMS WITH ONLY ATTENUATION CORRECTION
% Create initial estimate for reconstruction:
% Size of the image to cover the full fov:
sizeImage_mm = [structSizeSino2d.rFov_mm*2 structSizeSino2d.rFov_mm*2 structSizeSino2d.zFov_mm];
% The size in pixels based in numR and the number of rings:
sizeImage_pixels = [structSizeSino2d.numR structSizeSino2d.numR structSizeSino2d.numZ];
% Size of the pixels:
sizePixel_mm = sizeImage_mm ./ sizeImage_pixels;
% Inititial estimate:
initialEstimate = ones(sizeImage_pixels, 'single');
filenameInitialEstimate = [outputPath '/initialEstimateDirect'];
interfilewrite(initialEstimate, filenameInitialEstimate, sizePixel_mm);

% Reconstruction space dimensions
nx = floor(structSizeSino2d.numR/sqrt(2))-1;
ny = floor(structSizeSino2d.numR/sqrt(2))-1;
mm_x = 2.0445;

% Reconstruction
for slice = 1 : structSizeSino2d.numZ
    disp(sprintf('Reconstrucción de Slice %d', slice));
    em_recon(:,:,slice) = mlem_mmr( nx, ny, mm_x, structSizeSino2d.numR, directSinograms(:,:,slice)', gaps_sinogram', acfsDirectSinograms(:,:,slice)', gaps_sinogram', structSizeSino2d.thetaValues_deg, 40, 1);
end

outputVolumeName = [outputPath '/reconstructedVolDirectSinogramsAten'];
interfilewrite(single(em_recon), outputVolumeName, sizePixel_mm);

h = figure;
image = getImageFromSlices(em_recon, 12,1,0);
imshow(image);
title('Reconstructed Volume with Attenuation Correction');
h = figure;
subplot(1,2,1);
plot(em_recon(round(sizeImage_pixels(1)/2),:,round(structSizeSino2d.numZ/2)));
title('Plot of Central Columnd of Central Slice');
subplot(1,2,1);
countsPerSlice = sum(sum(em_recon));
countsPerSlice = permute(countsPerSlice, [3 1 2]);
plot(countsPerSlice);
title('Counts Per Slice');
%% 2D RECONSTRUCTION OF DIRECT SINOGRAMS WITH ATTENUATION CORRECTION AND NORMALIZATION
% Create initial estimate for reconstruction:
% Size of the image to cover the full fov:
sizeImage_mm = [structSizeSino2d.rFov_mm*2 structSizeSino2d.rFov_mm*2 structSizeSino2d.zFov_mm];
% The size in pixels based in numR and the number of rings:
sizeImage_pixels = [structSizeSino2d.numR structSizeSino2d.numR structSizeSino2d.numZ];
% Size of the pixels:
sizePixel_mm = sizeImage_mm ./ sizeImage_pixels;
% Inititial estimate:
initialEstimate = ones(sizeImage_pixels, 'single');
filenameInitialEstimate = [outputPath '/initialEstimateDirect'];
interfilewrite(initialEstimate, filenameInitialEstimate, sizePixel_mm);

% Reconstruction space dimensions
nx = floor(structSizeSino2d.numR/sqrt(2))-1;
ny = floor(structSizeSino2d.numR/sqrt(2))-1;
mm_x = 2.0445;

% Reconstruction
for slice = 1 : structSizeSino2d.numZ
    disp(sprintf('Reconstrucción de Slice %d', slice));
    em_recon(:,:,slice) = mlem_mmr( nx, ny, mm_x, structSizeSino2d.numR, directSinograms(:,:,slice)', gaps_sinogram', atteNormFactorsDirect(:,:,slice)', gaps_sinogram', structSizeSino2d.thetaValues_deg, 40, 0);
end

outputVolumeName = [outputPath '/reconstructedVolDirectSinogramsAtenNorm'];
interfilewrite(single(em_recon), outputVolumeName, sizePixel_mm);

h = figure;
image = getImageFromSlices(em_recon, 12,1,0);
imshow(image);
title('Reconstructed Volume with Attenuation Correction and Normalization');
h = figure;
subplot(1,2,1);
plot(em_recon(round(sizeImage_pixels(1)/2),:,round(structSizeSino2d.numZ/2)));
title('Plot of Central Columnd of Central Slice');
subplot(1,2,1);
countsPerSlice = sum(sum(em_recon));
countsPerSlice = permute(countsPerSlice, [3 1 2]);
plot(countsPerSlice);
title('Counts Per Slice');