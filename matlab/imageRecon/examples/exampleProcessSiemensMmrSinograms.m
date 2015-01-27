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
directSinograms = sinogram(:,:,structSizeSino2d.numZ);
% Write to a file in interfile formar:
outputSinogramName = [outputPath '/directSinograms'];
interfileWriteSino(single(sinogramSpan11), outputSinogramName);
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
%% ATTENUATION CORRECTION
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
%%
% Size of the image to cover the full fov:
sizeImage_mm = [structSizeSino3d.rFov_mm*2 structSizeSino3d.rFov_mm*2 structSizeSino3d.zFov_mm];
sizePixel_mm = sizeImage_mm ./ imageSize_pixels;

% Create ACFs of a computed phatoms with the linear attenuation
% coefficients:
acfFilename = ['acfsSinogramSpan11'];
filenameSinogram = [outputPath 'directSinograms'];
acfs3dSpan11 = createACFsFromImage(attenMap, sizePixel_mm, outputPath, acfFilename, filenameSinogram, structSizeSino3d, 1);
%%
acfFilename = ['acfsDirectSinograms'];
filenameSinogram = [outputPath 'directSinograms'];
acfsDirectSinograms = createACFsFromImage(attenMap, sizePixel_mm, outputPath, acfFilename, filenameSinogram, structSizeSino2d, 1);
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
