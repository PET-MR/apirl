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
outputPath = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/NEMA_IQ_20_02_2014/Reconstruction/span1/';
mkdir(outputPath);
%setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') ':/usr/lib/x86_64-linux-gnu/']);
%% READING THE SINOGRAMS
% Read the sinograms:
sinogramsPath = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/NEMA_IQ_20_02_2014/';
filenameUncompressedMmr = [sinogramsPath 'PET_ACQ_194_20150220154553-0uncomp.s'];
outFilenameIntfSinograms = [sinogramsPath 'NemaIq20_02_2014_ApirlIntf.s'];
[sinogram, delayedSinogram, structSizeSino3d] = getIntfSinogramsFromUncompressedMmr(filenameUncompressedMmr, outFilenameIntfSinograms);

% Read the normalization factors:
filenameRawData = '/home/mab15/workspace/KCL/Biograph_mMr/Normalization/NormFiles/Norm_20150210112413.n';
[componentFactors, componentLabels]  = readmMrComponentBasedNormalization(filenameRawData, 1);
%% CREATE SINOGRAMS3D SPAN 11
% Create sinogram span 11:
michelogram = generateMichelogramFromSinogram3D(sinogram, structSizeSino3d);
structSizeSino3dSpan11 = getSizeSino3dFromSpan(structSizeSino3d.numR, structSizeSino3d.numTheta, structSizeSino3d.numZ, ...
    structSizeSino3d.rFov_mm, structSizeSino3d.zFov_mm, 11, structSizeSino3d.maxAbsRingDiff);
sinogramSpan11 = reduceMichelogram(michelogram, structSizeSino3dSpan11.sinogramsPerSegment, structSizeSino3dSpan11.minRingDiff, structSizeSino3dSpan11.maxRingDiff);
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
delaySinogramSpan11 = reduceMichelogram(michelogram, structSizeSino3dSpan11.sinogramsPerSegment, structSizeSino3dSpan11.minRingDiff, structSizeSino3dSpan11.maxRingDiff);
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
sizeImage_pixels = [143 143 127]; % For cover the full Fov: 596/4.1725=142.84
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
sizeImageHighRes_pixels = [sizeImage_pixels(1)*factor sizeImage_pixels(2)*factor sizeImage_pixels(3)];
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
% Read the phantom and then generate the ACFs with apirl. There are two
% attenuation maps, the one of the hardware and the one of the patient or
% human.
imageSizeAtten_pixels = [344 344 127];
imageSizeAtten_mm = [2.08626 2.08626 2.0312];
filenameAttenMap_human = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/NEMA_IQ_20_02_2014/umap/AttenMapCtManuallyRegistered.i33';
filenameAttenMap_hardware = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/NEMA_IQ_20_02_2014/umap/PET_ACQ_194_20150220154553_umap_hardware_00.v';
% Human:
fid = fopen(filenameAttenMap_human, 'r');
if fid == -1
    ferror(fid);
end
attenMap_human = fread(fid, imageSizeAtten_pixels(1)*imageSizeAtten_pixels(2)*imageSizeAtten_pixels(3), 'single');
attenMap_human = reshape(attenMap_human, imageSizeAtten_pixels);
% Then interchange rows and cols, x and y: 
attenMap_human = permute(attenMap_human, [2 1 3]);
fclose(fid);
% The mumap of the phantom it has problems in the spheres, I force all the
% pixels inside the phantom to the same value:

% Hardware:
fid = fopen(filenameAttenMap_hardware, 'r');
if fid == -1
    ferror(fid);
end
attenMap_hardware = fread(fid, imageSizeAtten_pixels(1)*imageSizeAtten_pixels(2)*imageSizeAtten_pixels(3), 'single');
attenMap_hardware = reshape(attenMap_hardware, imageSizeAtten_pixels);
% Then interchange rows and cols, x and y: 
attenMap_hardware = permute(attenMap_hardware, [2 1 3]);
fclose(fid);

% Compose both images:
attenMap = attenMap_hardware + attenMap_human;

% visualization
figure;
image = getImageFromSlices(attenMap, 12, 1, 0);
imshow(image);
title('Attenuation Map Shifted');

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
clear atteNormCorrectionFactorsSpan11;
