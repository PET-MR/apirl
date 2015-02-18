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
outputPath = '/home/mab15/workspace/KCL/Biograph_mMr/mmr/5hr_ge68/span1/';
mkdir(outputPath);
%setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') ':/usr/lib/x86_64-linux-gnu/']);
%% READING THE SINOGRAMS
% Read the sinograms:
filenameUncompressedMmr = '/home/mab15/workspace/KCL/Biograph_mMr/mmr/5hr_ge68/cylinder_5hours.s';
outFilenameIntfSinograms = [outputPath 'sinogramSpan1'];
[sinogram, delayedSinogram, structSizeSino3d] = getIntfSinogramsFromUncompressedMmr(filenameUncompressedMmr, outFilenameIntfSinograms);

% Read the normalization factors:
filenameRawData = '/home/mab15/workspace/KCL/Biograph_mMr/mmr/Norm_20141008101010.n';
[componentFactors, componentLabels]  = readmMrComponentBasedNormalization(filenameRawData, 1);

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
sinoEfficencies = createSinogram3dFromDetectorsEfficency(componentFactors{3}, structSizeSino3d, 1);
figure;
subplot(1,2,1);
imshow(sinoEfficencies(:,:,1)' ./max(max(sinoEfficencies(:,:,1))));
title('Crystal Efficencies for Sinogram 1');
subplot(1,2,2);
imshow(sinoEfficencies(:,:,200)' ./max(max(sinoEfficencies(:,:,200))));
title('Crystal Efficencies for Sinogram 200');
set(gcf, 'Position', [0 0 1600 1200]);

% Axial factors:
axialFactors = structSizeSino3d.numSinosMashed; % 1./(componentFactors{4}.*componentFactors{8});
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
acfFilename = ['acfsSinogramSpan1'];
filenameSinogram = [outputPath 'sinogramSpan1'];
acfsSinogramSpan1 = createACFsFromImage(attenMap, imageSizeAtten_mm, outputPath, acfFilename, filenameSinogram, structSizeSino3d, 1);
%% READ THE ACFS (OPTION B)
% Span11 Sinogram:
acfFilename = [outputPath 'acfsSinogramSpan1'];
fid = fopen([acfFilename '.i33'], 'r');
numSinos = sum(structSizeSino3d.sinogramsPerSegment);
[acfsSinogramSpan1, count] = fread(fid, structSizeSino3d.numTheta*structSizeSino3d.numR*numSinos, 'single=>single');
acfsSinogramSpan1 = reshape(acfsSinogramSpan1, [structSizeSino3d.numR structSizeSino3d.numTheta numSinos]);
% Close the file:
fclose(fid);

%% VISUALIZATION OF COUNTS PER SINOGRAM
% Sum of the sinograms:
countsPerSinogram = sum(sum(sinogram));
% Change vecor in 3rd dimension for 1st, to plot:
countsPerSinogram = permute(countsPerSinogram,[3 1 2]);
h = figure;
plot(countsPerSinogram);
title('Counts Per Sinogram Span 1 without Normalization');
set(gcf, 'Position', [0 0 1600 1200]);

%% CORRECTION OF SPAN 11 SINOGRAMS
sinogramCorrected = zeros(size(sinogram));
if(numel(axialFactors) ~= sum(structSizeSino3d.sinogramsPerSegment))
    perror('La cantidad de factores de correccion axial es distinto a la cantidad de sinograms');
end
% Correct sinograms 3d for normalization:
for i = 1 : sum(structSizeSino3d.sinogramsPerSegment)
    sinogramCorrected(:,:,i) = sinogram(:,:,i) .* (1./geometricFactor) .* (1./crystalInterfFactor) .* (1./axialFactors(i));
end

% Crystal efficencies, there are gaps, so avoid zero values:
nonzero = sinoEfficencies ~= 0;
% Apply efficency:
sinogramCorrected(nonzero) = sinogramCorrected(nonzero) ./ sinoEfficencies(nonzero);
% And gaps:
sinogramCorrected(~nonzero) = 0;

figure;plot(sinogramCorrected(100,:,50)./max(sinogramCorrected(100,:,50)));
% Write to a file in interfile formar:
outputSinogramName = [outputPath '/sinogramSpan1Normalized'];
interfileWriteSino(single(sinogramCorrected), outputSinogramName, structSizeSino3d);

% Attenuation correction:
sinogramCorrected = sinogramCorrected .* acfsSinogramSpan1;
% Write to a file in interfile formar:
outputSinogramName = [outputPath '/sinogramSpan1NormAttenCorrected'];
interfileWriteSino(single(sinogramCorrected), outputSinogramName, structSizeSino3d);

%% GENERATE ATTENUATION AND NORMALIZATION FACTORS AND CORECCTION FACTORS FOR SPAN11 SINOGRAMS FOR APIRL
% This factors are not for precorrect but for apply as normalizaiton in
% each iteration:
normFactorsSpan1 = zeros(size(sinogram));
for i = 1 : sum(structSizeSino3d.sinogramsPerSegment)
    % First the geomeitric, crystal interference factors:
    normFactorsSpan1(:,:,i) = geometricFactor .* crystalInterfFactor;
    % Axial factor:
    normFactorsSpan1(:,:,i) = normFactorsSpan1(:,:,i) .* axialFactors(i);
end
% Then apply the crystal efficiencies:
normFactorsSpan1 = normFactorsSpan1 .* sinoEfficencies;
% Save:
outputSinogramName = [outputPath '/NF_Span1'];
interfileWriteSino(single(normFactorsSpan1), outputSinogramName, structSizeSino3d);

% We also generate the ncf:
normCorrectionFactorsSpan1 = zeros(size(sinogram));
normCorrectionFactorsSpan1(normFactorsSpan1~=0) = 1 ./ normFactorsSpan1(normFactorsSpan1~=0);
outputSinogramName = [outputPath '/NCF_Span1'];
interfileWriteSino(single(normCorrectionFactorsSpan1), outputSinogramName, structSizeSino3d);

% Compose with acfs:
atteNormFactorsSpan1 = normFactorsSpan1;
atteNormFactorsSpan1(acfsSinogramSpan1 ~= 0) = atteNormFactorsSpan1(acfsSinogramSpan1 ~= 0) ./acfsSinogramSpan1(acfsSinogramSpan1 ~= 0);
outputSinogramName = [outputPath '/ANF_Span1'];
interfileWriteSino(single(atteNormFactorsSpan1), outputSinogramName, structSizeSino3d);
clear atteNormFactorsSpan11;
%clear normFactorsSpan11;

% The same for the correction factors:
atteNormCorrectionFactorsSpan1 = normCorrectionFactorsSpan1 .*acfsSinogramSpan1;
outputSinogramName = [outputPath '/ANCF_Span1'];
interfileWriteSino(single(atteNormCorrectionFactorsSpan1), outputSinogramName, structSizeSino3d);
clear atteNormCorrectionFactorsSpan11;

% One for just the gaps:
gaps = sinoEfficencies ~=0;
outputSinogramName = [outputPath '/GAPS_Span1'];
interfileWriteSino(single(gaps), outputSinogramName, structSizeSino3d);

% One for just the gaps with attenuation:
gaps = single(sinoEfficencies ~=0);
gaps(acfsSinogramSpan1 ~= 0) = gaps(acfsSinogramSpan1 ~= 0) ./acfsSinogramSpan1(acfsSinogramSpan1 ~= 0);
outputSinogramName = [outputPath '/AGAPS_Span1'];
interfileWriteSino(single(gaps), outputSinogramName, structSizeSino3d);

clear gaps