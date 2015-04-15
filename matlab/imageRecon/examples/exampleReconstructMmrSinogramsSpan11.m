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
addpath('/home/mab15/workspace/KCL/Biograph_mMr/mmr');
apirlPath = '/home/mab15/workspace/apirl-code/trunk/';
addpath(genpath([apirlPath '/matlab']));
setenv('PATH', [getenv('PATH') ':' apirlPath '/build/bin']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') ':' apirlPath '/build/bin']);
outputPath = '/home/mab15/workspace/KCL/Biograph_mMr/mmr/5hr_ge68/span11/';
mkdir(outputPath);
%setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') ':/usr/lib/x86_64-linux-gnu/']);
%% READING THE SINOGRAMS
% Read the sinograms:
%filenameUncompressedMmr = '/workspaces/Martin/KCL/Biograph_mMr/mmr/test.s';
%outFilenameIntfSinograms = '/workspaces/Martin/KCL/Biograph_mMr/mmr/testIntf';
filenameUncompressedMmr = '/home/mab15/workspace/KCL/Biograph_mMr/mmr/5hr_ge68/cylinder_5hours.s';
outFilenameIntfSinograms = '/home/mab15/workspace/KCL/Biograph_mMr/mmr/5hr_ge68/cylinder_5hoursIntf';
[sinogram, delayedSinogram, structSizeSino3d] = getIntfSinogramsFromUncompressedMmr(filenameUncompressedMmr, outFilenameIntfSinograms);

% Read the normalization factors:
filenameRawData = '/home/mab15/workspace/KCL/Biograph_mMr/mmr/Norm_20141008101010.n';
[componentFactors, componentLabels]  = readmMrComponentBasedNormalization(filenameRawData, 1);
%% CREATE SINOGRAMS3D SPAN 11
% Create sinogram span 11:
michelogram = generateMichelogramFromSinogram3D(sinogram, structSizeSino3d);
structSizeSino3dSpan11 = getSizeSino3dFromSpan(structSizeSino3d.numR, structSizeSino3d.numTheta, structSizeSino3d.numZ, ...
    structSizeSino3d.rFov_mm, structSizeSino3d.zFov_mm, 11, structSizeSino3d.maxAbsRingDiff);
sinogramSpan11 = reduceMichelogram(michelogram, structSizeSino3dSpan11);
clear michelogram
clear sinogram
% Write to a file in interfile format:
outputSinogramName = [outputPath 'sinogramSpan11'];
interfileWriteSino(single(sinogramSpan11), outputSinogramName, structSizeSino3dSpan11);
% In int 16 also:
outputSinogramName = [outputPath 'sinogramSpan11_int16'];
interfileWriteSino(int16(sinogramSpan11), outputSinogramName, structSizeSino3dSpan11);

% The same for the delayed:
% Create sinogram span 11:
michelogram = generateMichelogramFromSinogram3D(delayedSinogram, structSizeSino3d);
delaySinogramSpan11 = reduceMichelogram(michelogram, structSizeSino3dSpan11);
clear michelogram
clear delayedSinogram

% Write to a file in interfile format:
outputSinogramName = [outputPath 'delaySinogramSpan11'];
interfileWriteSino(single(delaySinogramSpan11), outputSinogramName, structSizeSino3dSpan11);
% In int 16 also:
outputSinogramName = [outputPath 'delaySinogramSpan11_int16'];
interfileWriteSino(int16(delaySinogramSpan11), outputSinogramName, structSizeSino3dSpan11);
clear delaySinogramSpan11 % Not used in this example.
%% CREATE INITIAL ESTIMATE FOR RECONSTRUCTION
% Create image from the same size than used by siemens:
% Size of the pixels:
sizePixel_mm = [4.1725 4.1725 2.0312];
% The size in pixels:
sizeImage_pixels = [172 172 127];
% Size of the image to cover the full fov:
sizeImage_mm = sizePixel_mm .* sizeImage_pixels;
% Inititial estimate:
initialEstimate = ones(sizeImage_pixels, 'single');
filenameInitialEstimate = [outputPath '/initialEstimate3d'];
interfilewrite(initialEstimate, filenameInitialEstimate, sizePixel_mm);

% Another image of high resolution:
% Size of the pixels:
factor = 2;
sizePixelHighRes_mm = [4.1725/factor 4.1725/factor 2.0312];
% The size in pixels:
sizeImageHighRes_pixels = [172*factor 172*factor 127];
% Size of the image to cover the full fov:
sizeImage_mm = sizePixelHighRes_mm .* sizeImageHighRes_pixels;
% Inititial estimate:
initialEstimateHighRes = ones(sizeImageHighRes_pixels, 'single');
filenameInitialEstimateHighRes = [outputPath '/initialEstimate3dHighRes'];
interfilewrite(initialEstimateHighRes, filenameInitialEstimateHighRes, sizePixelHighRes_mm);
%% NORMALIZATION FACTORS
% A set of 2d sinograms with normalization effects is generated. The
% geomtric normalization and crystal interference are the same for each
% sinogram.

% Geometric Factor. The geomtric factor is one projection profile per
% plane. But it's the same for all planes, so I just use one of them.
geometricFactor = repmat(double(componentFactors{1}(:,1)), 1, structSizeSino3d.numTheta);
figure;
set(gcf, 'Position', [0 0 1600 1200]);
imshow(geometricFactor' ./max(max(geometricFactor)));
title('Geometric Factor');
% Crystal interference, its a pattern that is repeated peridoically:
crystalInterfFactor = single(componentFactors{2})';
crystalInterfFactor = repmat(crystalInterfFactor, 1, structSizeSino3d.numTheta/size(crystalInterfFactor,2));
% Geometric Normalization and crystal interference:
figure;
imshow(crystalInterfFactor' ./max(max(crystalInterfFactor)));
title('Crystal Interference Factors');
set(gcf, 'Position', [0 0 1600 1200]);

% Generate the sinograms for crystal efficencies:
sinoEfficencies = createSinogram3dFromDetectorsEfficency(componentFactors{3}, structSizeSino3dSpan11, 1);
figure;
subplot(1,2,1);
imshow(sinoEfficencies(:,:,1)' ./max(max(sinoEfficencies(:,:,1))));
title('Crystal Efficencies for Sinogram 1');
subplot(1,2,2);
imshow(sinoEfficencies(:,:,200)' ./max(max(sinoEfficencies(:,:,200))));
title('Crystal Efficencies for Sinogram 200');
set(gcf, 'Position', [0 0 1600 1200]);

% Axial factors:
axialFactors = structSizeSino3dSpan11.numSinosMashed; % 1./(componentFactors{4}.*componentFactors{8});
%% ATTENUATION CORRECTION - PICK A OR B AND COMMENT THE NOT USED 
%% COMPUTE THE ACFS (OPTION A)
% Read the phantom and then generate the ACFs with apirl:
imageSizeAtten_pixels = [344 344 127];
imageSizeAtten_mm = [2.08626 2.08626 2.0312];
filenameAttenMap = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/2601/interfile/PET_ACQ_16_20150116131121-0_PRR_1000001_20150126152442_umap_human_00.v';
fid = fopen(filenameAttenMap, 'r');
if fid == -1
    ferror(fid);
end
attenMap = fread(fid, imageSizeAtten_pixels(1)*imageSizeAtten_pixels(2)*imageSizeAtten_pixels(3), 'single');
attenMap = reshape(attenMap, imageSizeAtten_pixels);
fclose(fid);
% visualization
figure;
image = getImageFromSlices(attenMap, 12, 1, 0);
imshow(image);
title('Attenuation Map Shifted');

% The phantom was not perfectly centered, so the attenuation map is
% shifted. I repeat the slcie 116 until the end:
for i = 117 : size(attenMap,3)
    attenMap(:,:,i) = attenMap(:,:,116);
end
figure;
image = getImageFromSlices(attenMap, 12, 1, 0);
imshow(image);
title('Attenuation Map Manualy Completed');
% Create ACFs of a computed phatoms with the linear attenuation
% coefficients:
acfFilename = ['acfsSinogramSpan11'];
filenameSinogram = [outputPath 'sinogramSpan11'];
acfsSinogramSpan11 = createACFsFromImage(attenMap, imageSizeAtten_mm, outputPath, acfFilename, filenameSinogram, structSizeSino3dSpan11, 1);
%% READ THE ACFS (OPTION B)
% Span11 Sinogram:
acfFilename = [outputPath 'acfsSinogramSpan11'];
fid = fopen([acfFilename '.i33'], 'r');
numSinos = sum(structSizeSino3dSpan11.sinogramsPerSegment);
[acfsSinogramSpan11, count] = fread(fid, structSizeSino3dSpan11.numTheta*structSizeSino3dSpan11.numR*numSinos, 'single=>single');
acfsSinogramSpan11 = reshape(acfsSinogramSpan11, [structSizeSino3dSpan11.numR structSizeSino3dSpan11.numTheta numSinos]);
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
    sinogramSpan11corrected(:,:,i) = sinogramSpan11(:,:,i) .* (1./geometricFactor) .* (1./crystalInterfFactor) .* (1./axialFactors(i));
end

% Crystal efficencies, there are gaps, so avoid zero values:
nonzero = sinoEfficencies ~= 0;
% Apply efficency:
sinogramSpan11corrected(nonzero) = sinogramSpan11corrected(nonzero) ./ sinoEfficencies(nonzero);
% And gaps:
sinogramSpan11corrected(~nonzero) = 0;

figure;plot(sinogramSpan11corrected(100,:,50)./max(sinogramSpan11corrected(100,:,50)));
% Write to a file in interfile formar:
outputSinogramName = [outputPath '/sinogramSpan11Normalized'];
interfileWriteSino(single(sinogramSpan11corrected), outputSinogramName, structSizeSino3dSpan11);

% Attenuation correction:
sinogramSpan11corrected = sinogramSpan11corrected .* acfsSinogramSpan11;
% Write to a file in interfile formar:
outputSinogramName = [outputPath '/sinogramSpan11NormAttenCorrected'];
interfileWriteSino(single(sinogramSpan11corrected), outputSinogramName, structSizeSino3dSpan11);

countsPerSinogramCorrected = sum(sum(sinogramSpan11corrected));


% Change vecor in 3rd dimension for 1st, to plot:
countsPerSinogramCorrected = permute(countsPerSinogramCorrected,[3 1 2]);
h = figure;
plot([countsPerSinogram./max(countsPerSinogram) countsPerSinogramCorrected./max(countsPerSinogramCorrected)], 'LineWidth', 2);
title('Counts Per Sinogram Span 11 with Normalization');
ylabel('Counts');
xlabel('Sinogram');

legend('Uncorrected', 'Corrected', 'Location', 'SouthEast');
set(gcf, 'Position', [0 0 1600 1200]);

% Delete variables because there is no enough memory:
clear constSinogramSpan11
%clear sinogramSpan11corrected
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
interfileWriteSino(single(normFactorsSpan11), outputSinogramName, structSizeSino3dSpan11);

% We also generate the ncf:
normCorrectionFactorsSpan11 = zeros(size(sinogramSpan11));
normCorrectionFactorsSpan11(normFactorsSpan11~=0) = 1 ./ normFactorsSpan11(normFactorsSpan11~=0);
outputSinogramName = [outputPath '/NCF_Span11'];
interfileWriteSino(single(normCorrectionFactorsSpan11), outputSinogramName, structSizeSino3dSpan11);

% Compose with acfs:
atteNormFactorsSpan11 = normFactorsSpan11;
atteNormFactorsSpan11(acfsSinogramSpan11 ~= 0) = atteNormFactorsSpan11(acfsSinogramSpan11 ~= 0) ./acfsSinogramSpan11(acfsSinogramSpan11 ~= 0);
outputSinogramName = [outputPath '/ANF_Span11'];
interfileWriteSino(single(atteNormFactorsSpan11), outputSinogramName, structSizeSino3dSpan11);
clear atteNormFactorsSpan11;
%clear normFactorsSpan11;

% The same for the correction factors:
atteNormCorrectionFactorsSpan11 = normCorrectionFactorsSpan11 .*acfsSinogramSpan11;
outputSinogramName = [outputPath '/ANCF_Span11'];
interfileWriteSino(single(atteNormCorrectionFactorsSpan11), outputSinogramName, structSizeSino3dSpan11);

% One for just the gaps:
gaps = sinoEfficencies ~=0;
outputSinogramName = [outputPath '/GAPS_Span11'];
interfileWriteSino(single(gaps), outputSinogramName, structSizeSino3dSpan11);

% One for just the gaps with attenuation:
gaps = single(sinoEfficencies ~=0);
gaps(acfsSinogramSpan11 ~= 0) = gaps(acfsSinogramSpan11 ~= 0) ./acfsSinogramSpan11(acfsSinogramSpan11 ~= 0);
outputSinogramName = [outputPath '/AGAPS_Span11'];
interfileWriteSino(single(gaps), outputSinogramName, structSizeSino3dSpan11);

clear gaps
%% RANDOMS FROM SINGLES IN BUCKET
sinoRandomsFromSinglesPerBucket = createRandomsFromSinglesPerBucket([filenameUncompressedMmr '.hdr']);
% Create span 11
michelogram = generateMichelogramFromSinogram3D(sinoRandomsFromSinglesPerBucket, structSizeSino3d);
randomsSinogramSpan11 = reduceMichelogram(michelogram, structSizeSino3dSpan11);
clear michelogram
% Write randoms sinogram:
outputSinogramName = [outputPath '/randomsSpan11'];
interfileWriteSino(single(randomsSinogramSpan11), outputSinogramName, structSizeSino3dSpan11);
% Correct for attenuation and normalization (to use in apirl). Since the
% projector is just geomtric, because its cancel with backprojector, the
% randoms and the scatter needs to be multiplied for the normalization
% correction factors and acfs:
randomsSinogramSpan11 = randomsSinogramSpan11 .* atteNormCorrectionFactorsSpan11;
outputSinogramName = [outputPath 'randomsSpan11_ancf'];
interfileWriteSino(single(randomsSinogramSpan11), outputSinogramName, structSizeSino3dSpan11);
%clear atteNormCorrectionFactorsSpan11;
%% READ SCATTER SINOGRAM FROM STIR
[sinogramScatter, structSizeSinoScatter] = interfileReadStirSino('/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/NEMA_IQ_20_02_2014/Stir/SCATTER1.hs');
% Correct for attenuation but normalization (to use in apirl). Since the
% projector is just geomtric, because its cancelled with backprojector, the
% randoms and the scatter needs to be multiplied for the normalization
% correction factors and acfs. However, the scatter estimated by stir is
% already computed with normalization, so just use afs.
sinogramScatter = sinogramScatter .* acfsSinogramSpan11;
outputSinogramName = [outputPath 'scatterSpan11_acf'];
interfileWriteSino(single(sinogramScatter), outputSinogramName, structSizeSino3dSpan11);
%% GENERATE OSEM AND MLEM RECONSTRUCTION FILES FOR APIRL WITH ATTENUATION AND RANDOM CORRECTION
% Low Res:
numSubsets = 21;
numIterations = 3;
saveInterval = 1;
saveIntermediate = 0;
outputFilenamePrefix = [outputPath sprintf('Nema_Osem%d_LR_withRandoms_', numSubsets)];
filenameOsemConfig_LR = [outputPath sprintf('/Osem3dSubset%d_LR_withRandoms.par', numSubsets)];
CreateOsemConfigFileForMmr(filenameOsemConfig_LR, [outputPath 'sinogramSpan11.h33'], [filenameInitialEstimate '.h33'], outputFilenamePrefix, numIterations, [], ...
    numSubsets, saveInterval, saveIntermediate, [], [outputPath 'scatterSpan11_acf.h33'], [outputPath 'randomsSpan11_ancf.h33'], [outputPath '/ANF_Span11']);

% High Res:
outputFilenamePrefix = [outputPath sprintf('Nema_Osem%d_HR_withRandoms_', numSubsets)];
filenameOsemConfig_HR = [outputPath sprintf('/Osem3dSubset%d_HR_withRandoms.par', numSubsets)];
CreateOsemConfigFileForMmr(filenameOsemConfig_HR, [outputPath 'sinogramSpan11.h33'], [filenameInitialEstimateHighRes '.h33'], outputFilenamePrefix, numIterations, [],...
    numSubsets, saveInterval, saveIntermediate, [], [outputPath 'scatterSpan11_acf.h33'], [outputPath 'randomsSpan11_ancf.h33'], [outputPath '/ANF_Span11.h33']);

%% RECONSTRUCTION OF HIGH RES IMAGE
% Execute APIRL:
status = system(['OSEM ' filenameOsemConfig_HR]) 
