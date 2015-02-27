%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 26/02/2015
%  *********************************************************************
clear all 
close all
%% PATHS FOR EXTERNAL FUNCTIONS AND RESULTS
addpath('/home/mab15/workspace/KCL/Biograph_mMr/mmr');
apirlPath = '/home/mab15/workspace/apirl-code/trunk/';
addpath(genpath([apirlPath '/matlab']));
outputPath = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/NEMA_IQ_20_02_2014/umap/';
%% READ CT IMAGES
pathCtPhantom = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/NEMA_IQ_20_02_2014/NEMA TEG/D4CTAC25MM/';
baseName = '1.2.840.113619.2.290.3.3171611701.628.1424678829.716';
% Get all the file from the path and base name.
files = dir([pathCtPhantom baseName '*']);
numSlices = numel(files);

% Read info from the images:
dicomInfo = dicominfo([pathCtPhantom files(1).name]);

% Generate the ct volume:
ctMap = zeros(dicomInfo.Rows, dicomInfo.Columns, numSlices);
% Read all the slices:
for i = 1 : numSlices
    ctMap(:,:,i) = dicomread([pathCtPhantom baseName sprintf('.%d', i)]); % Not useful because the order is not proper: [pathCtPhantom files(i).name]
end
% The padding values I force them to air:
ctMap(ctMap == dicomInfo.PixelPaddingValue) = -1000;

% Rescale to HU:
ctMap = ctMap.*dicomInfo.RescaleSlope + dicomInfo.RescaleIntercept;

% Visualization:
imageToShow = getImageFromSlices(ctMap, 12);
figure;
imshow(imageToShow);
title('CT of the Phantom');
%% CONVERT CT IN MU MAP AT 511 KEV
% We use the method with two linear conversions, one slope for HU of 0 or
% less (soft tissue) and another one for HU greater than 0 (bones).
% The correction factor for higher density tissues depends of the 140 kVp
% (Bai 2003):
switch(dicomInfo.KVP)
    case 140
        corrConvFactorHigherDensity = 0.640;
    case 130
        corrConvFactorHigherDensity = 0.605;
    case 120
        corrConvFactorHigherDensity = 0.576;
    case 110
        corrConvFactorHigherDensity = 0.509;
    otherwise
        corrConvFactorHigherDensity = 0.509;
end

% Plot the conversion curve:
valuesHU = -1000 : 2000;
conversionFactosHUtoMu511 = [(1+valuesHU(valuesHU<=0)/1000).*0.096 (1+corrConvFactorHigherDensity.*valuesHU(valuesHU>0)/1000).*0.096];
figure;
plot(valuesHU,conversionFactosHUtoMu511);
ylabel('Conversion Factor');
xlabel('Hounsfield Units');
% Apply the conversion to the ct image:
softTissue = ctMap <= 0;
boneTissue = ctMap > 0;

muMap511FromCt = zeros(size(ctMap));
muMap511FromCt(softTissue) = (1+ctMap(softTissue)/1000).*0.096;
muMap511FromCt(boneTissue) = (1+corrConvFactorHigherDensity.*ctMap(boneTissue)/1000).*0.096;

% Visualization:
imageToShow = getImageFromSlices(muMap511FromCt, 12);
figure;
imshow(imageToShow);
title('MU MAP of the Phantom From CT Scan');

%% READ DIXON ATTENUATION MAPS
% There are two attenuation maps, the one of the hardware and the one of the patient or
% human.
imageSizeAtten_pixels = [344 344 127];
pixelSizeAtten_mm = [2.08626 2.08626 2.0312];
imageSizeAtten_mm = imageSizeAtten_pixels .* pixelSizeAtten_mm;
filenameAttenMap_human = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/NEMA_IQ_20_02_2014/umap/PET_ACQ_194_20150220154553_umap_human_00.v';
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
% Visualization:
imageToShow = getImageFromSlices(attenMap_human, 12);
figure;
imshow(imageToShow);
title('MU MAP of the Phantom From Dixon Sequence');

%% REGISTER BOTH IMAGES
% Remove the bed from the scanner, using the slice without phantom:
muBedThreshold = 0.001;
bedMask = muMap511FromCt(:,:,1) > muBedThreshold;
% Dilate the mask to cover any noise:
bedMask = imdilate(bedMask, ones(3));
% Apply the mask:
for i = 1 : numSlices
    aux = muMap511FromCt(:,:,i);
    aux(bedMask) = 0;
    muMap511FromCt(:,:,i) = aux;
end
% Also remove negative values:
muMap511FromCt(muMap511FromCt < 0) = 0;

% Visualization:
imageToShow = getImageFromSlices(muMap511FromCt, 12);
figure;
imshow(imageToShow);
title('MU MAP of the Phantom From CT Scan with Bed Removed');

% First rescale to have the same pixel size. The ct to the attenuationMap.
% Coordinates of pixel of the CT:
pixelSizeCt_mm = [dicomInfo.PixelSpacing(1) dicomInfo.PixelSpacing(2) dicomInfo.SpacingBetweenSlices]; %dicomInfo.SliceThickness]; % dicomInfo.SpacingBetweenSlices
imageSizeCt_pixels = size(ctMap);
imageSizeCt_mm = imageSizeCt_pixels .* pixelSizeCt_mm;

% The MR image has some problems putting the air with higher density, I
% removed them:
meanValueWater = mean(mean(attenMap_human(imageSizeAtten_pixels(2)/2-10:imageSizeAtten_pixels(2)/2+10, imageSizeAtten_pixels(1)/2-10:imageSizeAtten_pixels(1)/2+10, round(imageSizeAtten_pixels(3)/2))))
attenMap_human_corrected = attenMap_human;
attenMap_human_corrected(attenMap_human > meanValueWater*1.1) = 0;
imageToShow = getImageFromSlices(attenMap_human_corrected, 12);
figure;
imshow(imageToShow);
title('MU MAP of the Phantom From CT Scan with Bed Removed');
%% AUTOMATIC REGISTRATION
% Parameters for registration:
[optimizer,metric] = imregconfig('multimodal');
optimizer. InitialRadius = 0.01;
Rfixed  = imref3d(size(attenMap_human),pixelSizeAtten_mm(2),pixelSizeAtten_mm(1),pixelSizeAtten_mm(3));
Rmoving = imref3d(size(muMap511FromCt),pixelSizeCt_mm(2),pixelSizeCt_mm(1),pixelSizeCt_mm(3));
% Initial Transform (aprox axial displacement and bed position):
initialTransform = affine3d([1 0 0 0; 0 1 0 0; 0 0 1 0; 2*pixelSizeAtten_mm(1) 10*pixelSizeAtten_mm(2) -2*pixelSizeAtten_mm(3) 1]);
% Step 1: Translation:
[movingRegistered Rmoving] = imregister(muMap511FromCt, Rmoving, attenMap_human_corrected, Rfixed, 'Rigid', optimizer, metric, 'InitialTransform', initialTransform);
% Show volume:
imageToShow = getImageFromSlices(movingRegistered, 12);
figure;
imshow(imageToShow);
title('MU MAP of the Phantom From CT registered With MR (1st step)');
% Show slice:
figure;
title('MU MAP of the Phantom From Dixon Sequence');
imshowpair(movingRegistered(:,:,40), attenMap_human(:,:,40));

optimizer. InitialRadius = optimizer. InitialRadius/10;
[movingRegistered Rmoving] = imregister(movingRegistered, Rmoving, attenMap_human, Rfixed, 'affine', optimizer, metric);
% Show volume:
imageToShow = getImageFromSlices(movingRegistered, 12);
figure;
imshow(imageToShow);
title('MU MAP of the Phantom From CT registered With MR (1st step)');
% Show slice:
figure;
title('MU MAP of the Phantom From Dixon Sequence');
imshowpair(movingRegistered(:,:,40), attenMap_human(:,:,40));

%% MANUAL REGISTRATION
% Resample to MR space (centered axially):
coordXct = pixelSizeCt_mm(2)/2 : pixelSizeCt_mm(2) : imageSizeCt_mm(2);
coordYct = pixelSizeCt_mm(1)/2 : pixelSizeCt_mm(1) : imageSizeCt_mm(1);
coordZct = pixelSizeCt_mm(3)/2 : pixelSizeCt_mm(3) : imageSizeCt_mm(3);
[Xct, Yct, Zct] = meshgrid(coordXct, coordYct, coordZct);
% Idem for the dixon attenuation map:
coordXpet = pixelSizeAtten_mm(2)/2 : pixelSizeAtten_mm(2) : imageSizeAtten_mm(2);
coordYpet = pixelSizeAtten_mm(1)/2 : pixelSizeAtten_mm(1) : imageSizeAtten_mm(1);
coordZpet = pixelSizeAtten_mm(3)/2 : pixelSizeAtten_mm(3) : imageSizeAtten_mm(3);
[Xpet, Ypet, Zpet] = meshgrid(coordXpet, coordYpet, coordZpet);
% Interpolate the ct image to the mr coordinates:
muMap511FromCt_rescaled = interp3(Xct,Yct,Zct,muMap511FromCt,Xpet,Ypet,Zpet); 
% I got some nans:
muMap511FromCt_rescaled(isnan(muMap511FromCt_rescaled)) = 0;

imageToShow = getImageFromSlices(muMap511FromCt_rescaled, 12);
figure;
imshow(imageToShow);
title('Manual Reg: MU MAP of the Phantom From CT Scan Resample to PET dimensions');
figure; title('Manual Reg: MU MAP of the Phantom From CT Scan Resample to PET dimensions'); imshowpair(muMap511FromCt_rescaled(:,:,40), attenMap_human(:,:,40));

% Translate:
Rmoving = imref3d(size(muMap511FromCt_rescaled),pixelSizeAtten_mm(2),pixelSizeAtten_mm(1),pixelSizeAtten_mm(3));
displacement_mm = [5 24 0];
[muMap511FromCt_rescaled_translated, Rtranslated] = imtranslate(muMap511FromCt_rescaled, Rmoving, displacement_mm,'OutputView','same');
figure;
imageToShow = getImageFromSlices(muMap511FromCt_rescaled_translated, 12);
figure;
imshow(imageToShow);
title('Manual Reg: MU MAP of the Phantom From CT Scan Resample to PET dimensions');
figure; title('Manual Reg: MU MAP of the Phantom From CT Scan Resample to PET dimensions'); imshowpair(muMap511FromCt_rescaled_translated(:,:,40), attenMap_human(:,:,40));

%% SAVE IMAGES
% Save image:
interfilewrite(single(muMap511FromCt_rescaled_translated), [outputPath 'AttenMapCtManuallyRegistered'], pixelSizeAtten_mm);
interfilewrite(single(movingRegistered), [outputPath 'AttenMapCtAutomaticRegistered'], pixelSizeAtten_mm);
