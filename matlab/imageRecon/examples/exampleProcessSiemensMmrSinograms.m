%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 15/01/2015
%  *********************************************************************
%  This example calls the getIntfSinogramsFromUncompressedMmr to convert an
%  uncompressed Siemenes interfile acquisition into an interfil APIRL
%  comptaible sinogram
clear all 
close all
%% PATHS FOR EXTERNAL FUNCTIONS
addpath('/workspaces/Martin/KCL/Biograph_mMr/mmr');
apirlPath = '/workspaces/Martin/PET/apirl-code/trunk/';
addpath(genpath([apirlPath '/matlab']));
setenv('PATH', [getenv('PATH') ':/workspaces/Martin/PET/apirl-code/trunk/build/debug/bin']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') ':/workspaces/Martin/PET/apirl-code/trunk/build/debug/bin']);
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
%% GAPS
mean_sinogram = mean(sinogram, 3);
gaps_sinogram = mean_sinogram;   nongap_indices  = find(mean_sinogram > 0);
gaps_sinogram(nongap_indices) = 1.0;
%% NORMALIZATION SINOGRAMS
% A set of 2d sinograms with normalization effects is
% Geometric Normalization:
normSinoGeom = 1./ imresize(double(componentFactors{1}'), [structSizeSino2d.numTheta structSizeSino2d.numR]);
% Crystal interference, its a pattern that is repeated peridoically:
crystalInterf = double(componentFactors{2});
crystalInterf = 1./repmat(crystalInterf,structSizeSino2d.numTheta/size(crystalInterf,1),1);
% Geometric Normalization and crystal interference:
figure;
imshow(crystalInterf ./max(max(crystalInterf)));
% Generate one norm factor:
normSinoGeomInterf = normSinoGeom .* crystalInterf;
title('Sinogram Correction For Crystal Interference and Geometry');


% %% VISUALIZATION OF DIRECT SINOGRAMS
% imageOfDirectSinos = getImageFromSlices(sinogram(:,:,1: structSizeSino3d.sinogramsPerSegment(1)), 8, 1,0);
% h = figure;
% imshow(imageOfDirectSinos);
% 
% %%
% michelogram = generateMichelogramFromSinogram3D(sinogram, structSizeSino3d);
% structSizeSino3dSpan11 = getSizeSino3dFromSpan(structSizeSino3d.numTheta, structSizeSino3d.numR, structSizeSino3d.numZ, ...
%     structSizeSino3d.rFov_mm, structSizeSino3d.zFov_mm, 11, structSizeSino3d.maxAbsRingDiff);
% sinogramSpan11 = reduceMichelogram(michelogram, structSizeSino3dSpan11.sinogramsPerSegment, structSizeSino3dSpan11.minRingDiff, structSizeSino3dSpan11.maxRingDiff);
% %%
% imageOfDirectSinos = getImageFromSlices(sinogramSpan11(:,:,1: structSizeSino3d.sinogramsPerSegment(1)), 8, 1,0);
% h = figure;
% imshow(imageOfDirectSinos);
% %% VISUALIZATION OF COUNTS PER SINOGRAM
% % Sum of the sinograms:
% countsPerSinogram = sum(sum(sinogram));
% % Change vecor in 3rd dimension for 1st, to plot:
% countsPerSinogram = permute(countsPerSinogram,[3 1 2]);
% h = figure;
% plot(countsPerSinogram)
% 
% countsPerSinogram = sum(sum(sinogramSpan11));
% % Change vecor in 3rd dimension for 1st, to plot:
% countsPerSinogram = permute(countsPerSinogram,[3 1 2]);
% h = figure;
% plot(countsPerSinogram)


%% REBINNING SINOGRAMS
% Create a rebinned 2d sinograms from the complete 3d sinogram:
[sinograms2d, structSizeSino2d]  = ssrbFromSinogram3d(sinogram, structSizeSino3d);
interfileWriteSino(single(sinograms2d), '/workspaces/Martin/KCL/Biograph_mMr/mmr/5hr_ge68/ssrb2dSinograms');

% Create ACFs of a computed phatoms with the linear attenuation
% coefficients:
outputPath = '/workspaces/Martin/KCL/Biograph_mMr/mmr/5hr_ge68/';
acfFilename = 'acfsSinograms2dSsrb';
filenameSinogram = '/workspaces/Martin/KCL/Biograph_mMr/mmr/5hr_ge68/ssrb2dSinograms';
acfs = createNormalizationPhantomACFs(outputPath, acfFilename, filenameSinogram, structSizeSino2d, 1);

% Correct the sinograms for normalization and the write it in interfile:
corrSinoFilename = [outputPath '/correctedSinograms2dSsrb'];
for i = 1 : structSizeSino2d.numZ
    correctedSinogram = sinograms2d(:,:,i) .* normSinoGeomInterf;
end
interfileWriteSino(correctedSinogram, corrSinoFilename);

% Plot an average of each projecction to check the attenuation correction:
figure;
plot([mean(sinograms2d(:,:,10),1); mean(sinograms2d(:,:,10).*acfs(:,:,10),1)]');
legend('Mean Uncorrected Projection', 'Mean Corrected Projection');
nx = floor(structSizeSino2d.numR/sqrt(2))-1;
ny = floor(structSizeSino2d.numR/sqrt(2))-1;
mm_x        = 2.0445;
for slice = 1 : structSizeSino2d.numZ
    sino2d = double(sinograms2d(:,:,slice));
    gaps_sinogram = zeros(size(sino2d));
    gaps_sinogram(sino2d>0) = 1;
    gaps_sinogram = gaps_sinogram .* normSinoGeomInterf;
    acf_sino = double(acfs(:,:,slice));
    em_recon(:,:,slice) = mlem_mmr( nx, ny, mm_x, structSizeSino2d.numR, sino2d', gaps_sinogram', acf_sino', gaps_sinogram', structSizeSino2d.thetaValues_deg, 40, 0);
end

figure;
imageSlices = getImageFromSlices(em_recon,8);
imshow(imageSlices);
%% 2D RECONSTRUCTION OF DIRECT SINOGRAMS
% Reconstruction of direct sinograms:
% Reconstruction space dimensions
nx = floor(structSizeSino3d.numR/sqrt(2))-1;
ny = floor(structSizeSino3d.numR/sqrt(2))-1;
mm_x = 2.0445;

% Reconstruction
for slice = 1 : 64
    sino2d = double(sinogram(:,:,slice));
    gaps_sinogram = zeros(size(sino2d));
    gaps_sinogram(sino2d>0) = 1;
    gaps_sinogram = gaps_sinogram;
    acf_sino = double(acfs(:,:,slice));
    em_recon(:,:,slice) = mlem_mmr( nx, ny, mm_x, structSizeSino2d.numR, sino2d', gaps_sinogram', acf_sino', gaps_sinogram', structSizeSino2d.thetaValues_deg, 40, 0);
end

% Geometric Normalization:
normSino = imresize(double(componentFactors{1}'), [structSizeSino2d.numTheta structSizeSino2d.numR]);
normSino = 1./ (normSino);
% Reconstruction
for slice = 1 : 64
    sino2d = double(sinogram(:,:,slice)).* normSino;
    gaps_sinogram = zeros(size(sino2d));
    gaps_sinogram(sino2d>0) = 1;
    acf_sino = ones(size(sino2d));
    em_recon_geomNorm(:,:,slice) = mlem_mmr( nx, ny, mm_x, structSizeSino2d.numR, sino2d', gaps_sinogram', acf_sino', gaps_sinogram', structSizeSino2d.thetaValues_deg, 40, 0);
end

% Geometric Normalization and crystal interference:
normSino = imresize(double(componentFactors{1}'), [structSizeSino2d.numTheta structSizeSino2d.numR]);
% Crystal interference, its a pattern that is repeated peridoically:
crystalInterf = double(componentFactors{2});
crystalInterf = repmat(crystalInterf,structSizeSino2d.numTheta/size(crystalInterf,1),1);
figure;
imshow(crystalInterf ./max(max(crystalInterf)));
% Generate one norm factor:
normSino = 1./ (normSino .* crystalInterf);
% Reconstruction
for slice = 1 : 64
    sino2d = double(sinogram(:,:,slice)).* normSino;
    gaps_sinogram = zeros(size(sino2d));
    gaps_sinogram(sino2d>0) = 1;
    acf_sino = double(acfs(:,:,slice));
    em_recon_geomInterfNorm(:,:,slice) = mlem_mmr( nx, ny, mm_x, structSizeSino2d.numR, sino2d', gaps_sinogram', acf_sino', gaps_sinogram', structSizeSino2d.thetaValues_deg, 40, 1);
end
%%
figure;
slice = 10;
subplot(2,3,1);
imshow(em_recon(:,:,slice)./max(max(em_recon(:,:,slice))));
subplot(2,3,2);
imshow(em_recon_geomNorm(:,:,slice)./max(max(em_recon_geomNorm(:,:,slice))));
subplot(2,3,3);
imshow(em_recon_geomInterfNorm(:,:,slice)./max(max(em_recon_geomInterfNorm(:,:,slice))));
subplot(2,3,4:6);
plot([em_recon(121,:,slice)./mean(em_recon(121,:,slice));em_recon_geomNorm(121,:,slice)./mean(em_recon_geomNorm(121,:,slice));...
    em_recon_geomInterfNorm(121,:,slice)./mean(em_recon_geomInterfNorm(121,:,slice))]');

corrReconImage = getImageFromSlices(em_recon_geomInterfNorm, 8);
figure;
imshow(corrReconImage);

%%
crystalEff = componentFactors{3}(:,10);
figure;
plot(crystalEff)
%%
% Generate the sinogram for crystal efficencies. To test, each detector has
% a different gain:
efficenciesPerDetector = rand(1,504);
%%
efficenciesPerDetector = componentFactors{3}(:,10);
sinoEfficencies = zeros(252,344);
% Histogram of amount of times has been used each detector:
detectorIds = 1 : numel(efficenciesPerDetector);
histDetIds = zeros(1, numel(efficenciesPerDetector));
% Go for each detector and then for each combination:
for det1 = 1 : numel(efficenciesPerDetector)
    for indDet2 = 1 : 344/2
        % For each detector, the same angle for two detectors (half angles)
        indProj = det1 - indDet2;
        det2 =  mod((det1 - 2*indDet2 + (504-344)/2), 504)+1;
        if (indProj > 0) && (det2 > 0) && (indProj <= 252) && (det2 <= 504)
            histDetIds = histDetIds + hist([det1; det2], detectorIds);
            sinoEfficencies(indProj,2*indDet2-1) = efficenciesPerDetector(det1) .* efficenciesPerDetector(det2);
        end 
        det2 = mod((det1 - 2*indDet2 + 1 + (504-344)/2), 504)+1;
        if (indProj > 0) && (det2 > 0) && (indProj <= 252) && (det2 <= 504)
            histDetIds = histDetIds + hist([det1; det2], detectorIds);
            sinoEfficencies(indProj,2*indDet2) = efficenciesPerDetector(det1) .* efficenciesPerDetector(det2);
        end
    end
end
figure;
subplot(1,2,1);
imshow(sinoEfficencies./max(max(sinoEfficencies)));
subplot(1,2,2);
bar(detectorIds,histDetIds);
%%
% Geometric Normalization and crystal interference:
normSino = imresize(double(componentFactors{1}'), [structSizeSino2d.numTheta structSizeSino2d.numR]);
% Crystal interference, its a pattern that is repeated peridoically:
crystalInterf = double(componentFactors{2});
crystalInterf = repmat(crystalInterf,structSizeSino2d.numTheta/size(crystalInterf,1),1);
figure;
imshow(crystalInterf ./max(max(crystalInterf)));
% Generate one norm factor:
normSino = 1./ (normSino .* crystalInterf);
normSino = normSino .*sinoEfficencies;
% Reconstruction
for slice = 10
    sino2d = double(sinogram(:,:,slice)).* normSino;
    gaps_sinogram = zeros(size(sino2d));
    gaps_sinogram(sino2d>0) = 1;
    acf_sino = double(acfs(:,:,slice));
    [ em_recon_geomInterfNorm ] = mlem_mmr( nx, ny, mm_x, structSizeSino2d.numR, sino2d', gaps_sinogram', acf_sino', gaps_sinogram', structSizeSino2d.thetaValues_deg, 40, 1);
end
%%
figure;
imshow(sino2d);
figure;
imshow(sino2d(:,1:(size(sino2d,2)/2)));
figure;
plot(sino2d(:,3));
