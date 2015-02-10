%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 03/02/2015
%  *********************************************************************
%  This example reads an interfile span 1 sinogram, compresses to span 11,
%  generates the normalization and attenuation factors, corrects then and
%  generate the files to reconstruct them with APIRL.
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
clear sinogram
% Write to a file in interfile format:
outputSinogramName = [outputPath 'sinogramSpan11'];
interfileWriteSino(single(sinogramSpan11), outputSinogramName, structSizeSino3dSpan11.sinogramsPerSegment, structSizeSino3dSpan11.minRingDiff, structSizeSino3dSpan11.maxRingDiff);
% In int 16 also:
outputSinogramName = [outputPath 'sinogramSpan11_int16'];
interfileWriteSino(int16(sinogramSpan11), outputSinogramName, structSizeSino3dSpan11.sinogramsPerSegment, structSizeSino3dSpan11.minRingDiff, structSizeSino3dSpan11.maxRingDiff);

% The same for the delayed:
% Create sinogram span 11:
michelogram = generateMichelogramFromSinogram3D(delayedSinogram, structSizeSino3d);
delaySinogramSpan11 = reduceMichelogram(michelogram, structSizeSino3dSpan11.sinogramsPerSegment, structSizeSino3dSpan11.minRingDiff, structSizeSino3dSpan11.maxRingDiff);
clear michelogram
clear delayedSinogram
% Write to a file in interfile format:
outputSinogramName = [outputPath 'delaySinogramSpan11'];
interfileWriteSino(single(delaySinogramSpan11), outputSinogramName, structSizeSino3dSpan11.sinogramsPerSegment, structSizeSino3dSpan11.minRingDiff, structSizeSino3dSpan11.maxRingDiff);
% In int 16 also:
outputSinogramName = [outputPath 'delaySinogramSpan11_int16'];
interfileWriteSino(int16(delaySinogramSpan11), outputSinogramName, structSizeSino3dSpan11.sinogramsPerSegment, structSizeSino3dSpan11.minRingDiff, structSizeSino3dSpan11.maxRingDiff);
%% CREATE INITIAL ESTIMATE FOR RECONSTRUCTION
% Create initial estimate for reconstruction:
% Size of the image to cover the full fov:
sizeImage_mm = [structSizeSino3dSpan11.rFov_mm*2 structSizeSino3dSpan11.rFov_mm*2 structSizeSino3dSpan11.zFov_mm];
% The size in pixels based in numR and the number of rings:
sizeImage_pixels = [structSizeSino3dSpan11.numR structSizeSino3dSpan11.numR structSizeSino3dSpan11.sinogramsPerSegment(1)];
% Size of the pixels:
sizePixel_mm = sizeImage_mm ./ sizeImage_pixels;
% Inititial estimate:
initialEstimate = ones(sizeImage_pixels, 'single');
filenameInitialEstimate = [outputPath '/initialEstimate3d'];
interfilewrite(initialEstimate, filenameInitialEstimate, sizePixel_mm);
%% NORMALIZATION FACTORS
% A set of 2d sinograms with normalization effects is generated. The
% geomtric normalization and crystal interference are the same for each
% sinogram.

% Geometric Factor. The geomtric factor is one projection profile per
% plane. But it's the same for all planes, so I just use one of them.
geometricFactor = repmat(double(componentFactors{1}(:,1))', structSizeSino3d.numTheta, 1);
figure;
set(gcf, 'Position', [0 0 1600 1200]);
imshow(geometricFactor ./max(max(geometricFactor)));
title('Geometric Factor');
% Crystal interference, its a pattern that is repeated peridoically:
crystalInterfFactor = double(componentFactors{2});
crystalInterfFactor = repmat(crystalInterfFactor,structSizeSino3d.numTheta/size(crystalInterfFactor,1),1);
% Geometric Normalization and crystal interference:
figure;
imshow(crystalInterfFactor ./max(max(crystalInterfFactor)));
title('Crystal Interference Factors');
set(gcf, 'Position', [0 0 1600 1200]);

% Generate the sinograms for crystal efficencies:
sinoEfficencies = createSinogram3dFromDetectorsEfficency(componentFactors{3}, structSizeSino3dSpan11, 1);
figure;
subplot(1,2,1);
imshow(sinoEfficencies(:,:,1) ./max(max(sinoEfficencies(:,:,1))));
title('Crystal Efficencies for Sinogram 1');
subplot(1,2,2);
imshow(sinoEfficencies(:,:,200) ./max(max(sinoEfficencies(:,:,200))));
title('Crystal Efficencies for Sinogram 200');
set(gcf, 'Position', [0 0 1600 1200]);

% Axial factors:
axialFactors = 1./(componentFactors{4}.*componentFactors{8});
%% ATTENUATION CORRECTION - PICK A OR B AND COMMENT THE NOT USED 
%% COMPUTE THE ACFS (OPTION A)
% Read the phantom and then generate the ACFs with apirl:
% imageSize_pixels = [344 344 127];
% filenameAttenMap = '/workspaces/Martin/KCL/Biograph_mMr/Mediciones/2601/interfile/PET_ACQ_16_20150116131121-0_PRR_1000001_20150126152442_umap_human_00.v';
% fid = fopen(filenameAttenMap, 'r');
% if fid == -1
%     ferror(fid);
% end
% attenMap = fread(fid, imageSize_pixels(1)*imageSize_pixels(2)*imageSize_pixels(3), 'single');
% attenMap = reshape(attenMap, imageSize_pixels);
% fclose(fid);
% % visualization
% figure;
% image = getImageFromSlices(attenMap, 12, 1, 0);
% imshow(image);
% title('Attenuation Map Shifted');
% % Size of the image to cover the full fov:
% sizeImage_mm = [structSizeSino3d.rFov_mm*2 structSizeSino3d.rFov_mm*2 structSizeSino3d.zFov_mm];
% sizePixel_mm = sizeImage_mm ./ imageSize_pixels;
% 
% % The phantom was not perfectly centered, so the attenuation map is
% % shifted. I repeat the slcie 116 until the end:
% for i = 117 : size(attenMap,3)
%     attenMap(:,:,i) = attenMap(:,:,116);
% end
% figure;
% image = getImageFromSlices(attenMap, 12, 1, 0);
% imshow(image);
% title('Attenuation Map Manualy Completed');
% % Create ACFs of a computed phatoms with the linear attenuation
% % coefficients:
% acfFilename = ['acfsSinogramSpan11'];
% filenameSinogram = [outputPath 'sinogramSpan11'];
% acfs3dSpan11 = createACFsFromImage(attenMap, sizePixel_mm, outputPath, acfFilename, filenameSinogram, structSizeSino3dSpan11, 1);
%% READ THE ACFS (OPTION B)
% Span11 Sinogram:
acfFilename = [outputPath 'acfsSinogramSpan11'];
fid = fopen([acfFilename '.i33'], 'r');
numSinos = sum(structSizeSino3dSpan11.sinogramsPerSegment);
[acfsSinogramSpan11, count] = fread(fid, structSizeSino3dSpan11.numTheta*structSizeSino3dSpan11.numR*numSinos, 'single=>single');
acfsSinogramSpan11 = reshape(acfsSinogramSpan11, [structSizeSino3dSpan11.numR structSizeSino3dSpan11.numTheta numSinos]);
% Matlab reads in a column-wise order that why angles are in the columns.
% We want to have it in the rows since APIRL and STIR and other libraries
% use row-wise order:
acfsSinogramSpan11 = permute(acfsSinogramSpan11,[2 1 3]);
% Close the file:
fclose(fid);

%% VISUALIZATION OF COUNTS PER SINOGRAM
% Sum of the sinograms:
countsPerSinogram = sum(sum(sinogramSpan11));
% Change vecor in 3rd dimension for 1st, to plot:
countsPerSinogram = permute(countsPerSinogram,[3 1 2]);
h = figure;
plot(countsPerSinogram);
title('Counts Per Sinogram Span 11 without Normalization');
set(gcf, 'Position', [0 0 1600 1200]);

%% CORRECTION OF SPAN 11 SINOGRAMS
sinogramSpan11corrected = zeros(size(sinogramSpan11));
if(numel(axialFactors) ~= sum(structSizeSino3dSpan11.sinogramsPerSegment))
    perror('La cantidad de factores de correccion axial es distinto a la cantidad de sinograms');
end
% Correct sinograms 3d for normalization:
for i = 1 : sum(structSizeSino3dSpan11.sinogramsPerSegment)
    sinogramSpan11corrected(:,:,i) = sinogramSpan11(:,:,i) .* (1./geometricFactor) .* (1./crystalInterfFactor) .* (1./axialFactors(i)) .* (acfsSinogramSpan11(:,:,i));
    % Crystal efficencies, there are gaps, so avoid zero values:
    nonzero = sinoEfficencies(:,:,i) ~= 0;
    aux = sinogramSpan11corrected(:,:,i);
    crystalEff = sinoEfficencies(:,:,i);
    aux(nonzero) = aux(nonzero) .* (1./crystalEff(nonzero));
    sinogramSpan11corrected(:,:,i) = aux;
end
countsPerSinogramCorrected = sum(sum(sinogramSpan11corrected));
% Compare with the projection of span 11 of a constant image (performed with APIRL):
constProjFilename = [outputPath 'Image_Span11__projection_iter_0'];
fid = fopen([constProjFilename '.i33'], 'r');
numSinos = sum(structSizeSino3dSpan11.sinogramsPerSegment);
[constSinogramSpan11, count] = fread(fid, structSizeSino3dSpan11.numTheta*structSizeSino3dSpan11.numR*numSinos, 'single=>single');
constSinogramSpan11 = reshape(constSinogramSpan11, [structSizeSino3dSpan11.numR structSizeSino3dSpan11.numTheta numSinos]);
% Matlab reads in a column-wise order that why angles are in the columns.
% We want to have it in the rows since APIRL and STIR and other libraries
% use row-wise order:
constSinogramSpan11 = permute(constSinogramSpan11,[2 1 3]);

% Close the file:
fclose(fid);
countsPerSinogramConstImage = sum(sum(constSinogramSpan11));
countsPerSinogramConstImage = permute(countsPerSinogramConstImage,[3 1 2]);

% Change vecor in 3rd dimension for 1st, to plot:
countsPerSinogramCorrected = permute(countsPerSinogramCorrected,[3 1 2]);
h = figure;
plot([countsPerSinogram./max(countsPerSinogram) countsPerSinogramCorrected./max(countsPerSinogramCorrected) ...
    countsPerSinogramConstImage./max(countsPerSinogramConstImage)], 'LineWidth', 2);
title('Counts Per Sinogram Span 11 with Normalization');
ylabel('Counts');
xlabel('Sinogram');
legend('Uncorrected', 'Corrected', 'Projection of Constant Image', 'Location', 'SouthEast');
set(gcf, 'Position', [0 0 1600 1200]);
% Write to a file in interfile formar:
outputSinogramName = [outputPath '/sinogramSpan11Normalized'];
interfileWriteSino(single(sinogramSpan11), outputSinogramName, structSizeSino3d.sinogramsPerSegment, structSizeSino3d.minRingDiff, structSizeSino3d.maxRingDiff);
% Delete variables because there is no enough memory:
clear constSinogramSpan11
clear sinogramSpan11corrected
%% GENERATE ATTENUATION AND NORMALIZATION FACTORS AND CORECCTION FACTORS FOR SPAN11 SINOGRAMS FOR APIRL
% This factors are not for precorrect but for apply as normalizaiton in
% each iteration:
normFactorsSpan11 = zeros(size(sinogramSpan11));
for i = 1 : sum(structSizeSino3dSpan11.sinogramsPerSegment)
    % First the geomeitric, crystal interference factors:
    normFactorsSpan11(:,:,i) = geometricFactor .* crystalInterfFactor;
    % Axial factor:
    normFactorsSpan11(:,:,i) = normFactorsSpan11(:,:,i) .* axialFactors(i);
end
% Then apply the crystal efficiencies:
normFactorsSpan11 = normFactorsSpan11 .* sinoEfficencies;
% Save:
outputSinogramName = [outputPath '/NF_Span11'];
interfileWriteSino(single(normFactorsSpan11), outputSinogramName, structSizeSino3dSpan11.sinogramsPerSegment, structSizeSino3dSpan11.minRingDiff, structSizeSino3dSpan11.maxRingDiff);

% We also generate the ncf:
normCorrectionFactorsSpan11 = zeros(size(sinogramSpan11));
normCorrectionFactorsSpan11(normFactorsSpan11~=0) = 1 ./ normCorrectionFactorsSpan11(normFactorsSpan11~=0);
outputSinogramName = [outputPath '/NCF_Span11'];
interfileWriteSino(single(normCorrectionFactorsSpan11), outputSinogramName, structSizeSino3dSpan11.sinogramsPerSegment, structSizeSino3dSpan11.minRingDiff, structSizeSino3dSpan11.maxRingDiff);

% Compose with acfs:
atteNormFactorsSpan11 = normFactorsSpan11;
atteNormFactorsSpan11(acfsSinogramSpan11 ~= 0) = atteNormFactorsSpan11(acfsSinogramSpan11 ~= 0) ./acfsSinogramSpan11(acfsSinogramSpan11 ~= 0);
outputSinogramName = [outputPath '/ANF_Span11'];
interfileWriteSino(single(atteNormFactorsSpan11), outputSinogramName, structSizeSino3dSpan11.sinogramsPerSegment, structSizeSino3dSpan11.minRingDiff, structSizeSino3dSpan11.maxRingDiff);
clear atteNormFactorsSpan11;
clear normFactorsSpan11;

% The same for the correction factors:
atteNormCorrectionFactorsSpan11 = normCorrectionFactorsSpan11 .*acfsSinogramSpan11;
outputSinogramName = [outputPath '/ANCF_Span11'];
interfileWriteSino(single(atteNormCorrectionFactorsSpan11), outputSinogramName, structSizeSino3dSpan11.sinogramsPerSegment, structSizeSino3dSpan11.minRingDiff, structSizeSino3dSpan11.maxRingDiff);


