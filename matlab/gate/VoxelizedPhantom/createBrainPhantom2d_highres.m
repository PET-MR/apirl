%% EXAMPLE MLEM MARTIN PROJECTOR (ANY SPAN)
clear all 
close all
%% APIRL PATH
apirlPath = '/home/mab15/workspace/apirl-code/trunk/';
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

%% CUDA PATH
cudaPath = '/usr/local/cuda/';
setenv('PATH', [getenv('PATH') sepEnvironment cudaPath pathBar 'bin']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') sepEnvironment cudaPath pathBar 'lib64']);
%% STIR PATH
stirPath = '/usr/local/stir3.0/';
stirMatlabPath = '/home/mab15/workspace/KCL/apirl-kcl/trunk/stir/';
scriptsPath = [stirMatlabPath 'scripts/'];
%% SET ENVIRONMENT AND MATLAB PATHS
addpath(genpath([apirlPath pathBar 'matlab']));
addpath(genpath(stirMatlabPath));
setenv('PATH', [getenv('PATH') sepEnvironment apirlPath pathBar 'build' pathBar 'bin' ':' stirPath pathBar 'bin/']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') sepEnvironment apirlPath pathBar 'build' pathBar 'bin' ':' stirPath pathBar 'lib/' ]);
%% LOAD BRAIN PHANTOM
outputPath = '/home/mab15/workspace/KCL/Biograph_mMr/GateModel/svn_2d/BrainPhantomHighRes/';
if ~isdir(outputPath)
    mkdir(outputPath)
end
% Use the high res brain phantom:
pixelSize_mm = [0.5 0.5 0.5];
[phantom_rescaled, attenuationMap, refImage] = CreateBrainPhantom('/home/mab15/workspace/KCL/Phantoms/brain/brainweb/subject04_crisp_v.raws');

%% GET AN SLICE
slice = 166; % fOR THE CAUDATE.
tAct_2d = phantom_rescaled(end:-1:1,:,slice);
tMu_2d = attenuationMap(end:-1:1,:,slice);
%% SIZE OF THE SMALL ROI
% System matrix of 58.415 mm x 70.9325 mm
indicesRoiRows = 110:(110+58/0.5);
indicesRoiCols = 110:(110+70/0.5);
%% MU-MAP FOR RECONSTRUCTION
tMu_for_recon = zeros(344,344);
% The mu-map needs to have the size of the reconstruction
xLimitsAt = [-size(tMu_2d,2)/2*0.5 size(tMu_2d,2)/2*0.5];
yLimitsAt = [-size(tMu_2d,1)/2*0.5 size(tMu_2d,1)/2*0.5];
%zLimitsAt = [-size(tMu_2d,3)/2*0.5 size(tMu_2d,3)/2*0.5];
refAt  = imref2d(size(tMu_2d),xLimitsAt,yLimitsAt);
xLimits = [-size(tMu_for_recon,2)/2*2.08625 size(tMu_for_recon,2)/2*2.08625];
yLimits = [-size(tMu_for_recon,1)/2*2.08625 size(tMu_for_recon,1)/2*2.08625];
%zLimits = [-size(tMu_for_recon,3)/2*2.03125 size(tMu_for_recon,3)/2*2.03125];
refAtRecon = imref2d(size(tMu_for_recon),xLimits,yLimits);
%pixelSize_mm = [2.08625 2.08625 0.5];
%[tMu refAt] = imregister(tMu, refAt,tAct, refAct, 'affine', optimizer, metric);
[tMu_for_recon, refAtRecon] = ImageResample(tMu_2d, refAt, refAtRecon);
%% NOW I WILL CREATE TWO DATA SETS WITH DIFFERENT LESION IN THE CUADATE AND OTHER TWO WITH LESION IN THE GRAY MATTER
% Add hot and cold spots:138:165,155:188
tAct_2d_small_lesions = tAct_2d;
tAct_2d_bigger_lesions = tAct_2d;
% Hot spot:
tAct_2d_small_lesions(157:159, 154:156) = tAct_2d_small_lesions(157:159, 154:156)*2;
% Cold spot:
tAct_2d_small_lesions(151:153, 203:205) = tAct_2d_small_lesions(151:153, 203:205)*0.2;

% Hot spot:
[X, Y] = meshgrid(1:size(tAct_2d,2), 1:size(tAct_2d,1));
centerHotLesion = [158 155];
centerColdLesion = [152 204];
radiusLesions_mm = 2;
radiusLesions_pixels = radiusLesions_mm./pixelSize_mm(1);
maskHotLesion = (X-centerHotLesion(2)).^2 + (Y-centerHotLesion(1)).^2 < radiusLesions_pixels.^2;
maskColdLesion = (X-centerColdLesion(2)).^2 + (Y-centerColdLesion(1)).^2 < radiusLesions_pixels.^2;
tAct_2d_bigger_lesions(maskHotLesion) = tAct_2d_bigger_lesions(maskHotLesion)*2;
% Cold spot:
tAct_2d_bigger_lesions(maskColdLesion) = tAct_2d_bigger_lesions(maskColdLesion)*0.2;
%% CUT THE IMAGE
% This is already reduced in size.
%% CONVERSION OF IMAGES INTO GATE NEEDS
% The activity image needs to have activity cncentration values through a
% rage trnaslator. The phantom has values from 0 to 5 aprox. For typical
% patient data, a concentration of 5 KBq/cm³. A pixel value of 1 should
% have that activit concentration.
disp('IMPORTANTE: Set .mac with a linear range scale of 1 kBq.');
conversionFactor = 5./(1./(pixelSize_mm(1)/10).*(pixelSize_mm(2)/10).*(pixelSize_mm(3)/10));
tAct_2d_small_lesions = tAct_2d_small_lesions.*conversionFactor;
interfilewrite_gate(tAct_2d_small_lesions, [outputPath 'actMap_small_lesion'], [refImage.PixelExtentInWorldY refImage.PixelExtentInWorldX 0.5], 'BrainPhantomHighRes');
interfilewrite(tMu_2d, [outputPath 'muMap'], [refImage.PixelExtentInWorldY refImage.PixelExtentInWorldX]);
tAct_2d_bigger_lesions = tAct_2d_bigger_lesions.*conversionFactor;
interfilewrite_gate(tAct_2d_bigger_lesions, [outputPath 'actMap_bigger_lesion'], [refImage.PixelExtentInWorldY refImage.PixelExtentInWorldX 0.5], 'BrainPhantomHighRes');

% create a constant image.
const = zeros(size(tAct_2d_small_lesions));
const(tMu_2d>0.01) = 1;
interfilewrite(uint16(const), [outputPath 'constMap_uint16'], [refImage.PixelExtentInWorldY refImage.PixelExtentInWorldX ]);
interfilewrite(uint16(tAct_2d_small_lesions), [outputPath 'actMap_small_lesion_uint16'], [refImage.PixelExtentInWorldY refImage.PixelExtentInWorldX]);
interfilewrite(uint16(tAct_2d_bigger_lesions), [outputPath 'actMap_bigger_lesion_uint16'], [refImage.PixelExtentInWorldY refImage.PixelExtentInWorldX]);
interfilewrite(tMu_2d, [outputPath 'muMap'], [refImage.PixelExtentInWorldY refImage.PixelExtentInWorldX]);
interfilewrite(tMu_for_recon, [outputPath 'muMap'], [refAtRecon.PixelExtentInWorldY refAtRecon.PixelExtentInWorldX]);
%% MUMAP
% I need to set the materials for each range of values:
numMaterials = 3;
rangeMaterials = [-0.1 0.07;0.07 0.12; 0.12 0.18];
nameMaterials = {'Air', 'Brain', 'Skull'};
tMu_2d_uint16 = zeros(size(tMu_2d), 'uint16');
tMu_2d_uint16(tMu_2d>rangeMaterials(1,1) & tMu_2d<rangeMaterials(1,2)) = 1;
tMu_2d_uint16(tMu_2d>rangeMaterials(2,1) & tMu_2d<rangeMaterials(2,2)) = 2;
tMu_2d_uint16(tMu_2d>rangeMaterials(3,1) & tMu_2d<rangeMaterials(3,2)) = 3;

% Range atenuation:
fid = fopen(sprintf('%s/RangeAttenuation.dat',outputPath), 'w');
if(fid == -1)
    fprintf('No se pudo crear el archivo RangeAttenuation.dat');
end
% En la primera línea la cantidad de materiales:
fprintf(fid,'%d\n', numMaterials);
for i = 1 : numMaterials
    % Uso colores aleatorios:
    fprintf(fid,'%.1f\t%.1f\t%s\ttrue %.1f %.1f %.1f\n', i, i, nameMaterials{i},...
        rand(1),rand(1),rand(1));
end
fclose(fid);
interfilewrite_gate(uint16(tMu_2d_uint16), [outputPath 'muMap_uint16'], [refImage.PixelExtentInWorldY refImage.PixelExtentInWorldX 0.5], 'BrainPhantomHighRes');
%% SMALL REGION TO SIMULATE IN GATE
outputPath = '/home/mab15/workspace/KCL/Biograph_mMr/GateModel/svn_2d/CaudatePhantomHighRes/';
if ~isdir(outputPath)
    mkdir(outputPath);
end
caudateImage_small_lesions = uint16(tAct_2d_small_lesions(indicesRoiRows,indicesRoiCols));
caudateImage_bigger_lesions = uint16(tAct_2d_bigger_lesions(indicesRoiRows,indicesRoiCols));
caudateImage_aten = tMu_2d(indicesRoiRows,indicesRoiCols);
caudateImage_aten_gate = tMu_2d_uint16(indicesRoiRows,indicesRoiCols);
caudateImage_aten_gate(1,1) = 1; % Becuase it needs one pixel different.
xLimits = [-size(caudateImage_small_lesions,2)/2*0.5 size(caudateImage_small_lesions,2)/2*0.5];
yLimits = [-size(caudateImage_small_lesions,1)/2*0.5 size(caudateImage_small_lesions,1)/2*0.5];
zLimits = 0;
refCaudate = imref2d(size(caudateImage_small_lesions),xLimits,yLimits);
interfilewrite_gate(uint16(caudateImage_small_lesions), [outputPath 'actMap_small_lesions_uint16'], [refCaudate.PixelExtentInWorldY refCaudate.PixelExtentInWorldX 0.5], 'CaudatePhantomHighRes');
interfilewrite_gate(uint16(caudateImage_bigger_lesions), [outputPath 'actMap_bigger_lesions_uint16'], [refCaudate.PixelExtentInWorldY refCaudate.PixelExtentInWorldX 0.5], 'CaudatePhantomHighRes');
interfilewrite_gate(uint16(caudateImage_aten_gate), [outputPath 'muMap_uint16'], [refCaudate.PixelExtentInWorldY refCaudate.PixelExtentInWorldX 0.5], 'CaudatePhantomHighRes');
interfilewrite(caudateImage_aten, [outputPath 'muMap'], [refCaudate.PixelExtentInWorldY refCaudate.PixelExtentInWorldX]);
% create a constant image.
const = ones(size(caudateImage_small_lesions));
interfilewrite(uint16(const), [outputPath 'constMap_uint16'], [refCaudate.PixelExtentInWorldY refCaudate.PixelExtentInWorldX ]);
% Use a point source also:
point = zeros(size(caudateImage_small_lesions));
point(round(size(caudateImage_small_lesions,1)./2),1) = 1;
interfilewrite(uint16(point), [outputPath 'pointMap_uint16'], [refCaudate.PixelExtentInWorldY refCaudate.PixelExtentInWorldX ]);
%% SMALL REGION TO SIMULATE IN GATE BUT IN THE SAME SIZE
outputPath = '/home/mab15/workspace/KCL/Biograph_mMr/GateModel/svn_2d/CaudatePhantomFullImageHighRes/';
if ~isdir(outputPath)
    mkdir(outputPath);
end
caudateFullImage_small_lesions = zeros(size(tAct_2d_small_lesions));
caudateFullImage_bigger_lesions = zeros(size(tAct_2d_small_lesions));
caudateImageFullImage_aten = zeros(size(tMu_2d_uint16)); % For reconis other size interfilewrite(tMu_for_recon, [outputPath 'muMap'], [refAtRecon.PixelExtentInWorldY refAtRecon.PixelExtentInWorldX]);
caudateImageFullImage_aten_gate = zeros(size(tMu_2d_uint16));
caudateFullImage_small_lesions(indicesRoiRows,indicesRoiCols) = uint16(tAct_2d_small_lesions(indicesRoiRows,indicesRoiCols));
caudateFullImage_bigger_lesions(indicesRoiRows,indicesRoiCols) = uint16(tAct_2d_bigger_lesions(indicesRoiRows,indicesRoiCols));
caudateImageFullImage_aten(indicesRoiRows,indicesRoiCols) = tMu_2d(indicesRoiRows,indicesRoiCols);
caudateImageFullImage_aten_gate(indicesRoiRows,indicesRoiCols) = tMu_2d_uint16(indicesRoiRows,indicesRoiCols);

% Resize the mumap for recon:
tMu_for_recon = zeros(344,344);
% The mu-map needs to have the size of the reconstruction
xLimitsAt = [-size(caudateImageFullImage_aten,2)/2*0.5 size(caudateImageFullImage_aten,2)/2*0.5];
yLimitsAt = [-size(caudateImageFullImage_aten,1)/2*0.5 size(caudateImageFullImage_aten,1)/2*0.5];
%zLimitsAt = [-size(caudateImageFullImage_aten,3)/2*0.5 size(caudateImageFullImage_aten,3)/2*0.5];
refAt  = imref2d(size(caudateImageFullImage_aten),xLimitsAt,yLimitsAt);
xLimits = [-size(tMu_for_recon,2)/2*2.08625 size(tMu_for_recon,2)/2*2.08625];
yLimits = [-size(tMu_for_recon,1)/2*2.08625 size(tMu_for_recon,1)/2*2.08625];
%zLimits = [-size(tMu_for_recon,3)/2*2.03125 size(tMu_for_recon,3)/2*2.03125];
refAtRecon = imref2d(size(tMu_for_recon),xLimits,yLimits);
%[tMu refAt] = imregister(tMu, refAt,tAct, refAct, 'affine', optimizer, metric);
[tMu_for_recon, refAtRecon] = ImageResample(caudateImageFullImage_aten, refAt, refAtRecon);

xLimits = [-size(caudateFullImage_small_lesions,2)/2*0.5 size(caudateFullImage_small_lesions,2)/2*0.5];
yLimits = [-size(caudateFullImage_small_lesions,1)/2*0.5 size(caudateFullImage_small_lesions,1)/2*0.5];
zLimits = 0;
% create a constant image.
const = zeros(size(tAct_2d_small_lesions));
const(indicesRoiRows,indicesRoiCols) = 1;
interfilewrite_gate(uint16(const), [outputPath 'constMap_uint16'], [refImage.PixelExtentInWorldY refImage.PixelExtentInWorldX 0.5], 'CaudatePhantomFullImageHighRes');
interfilewrite_gate(uint16(caudateFullImage_small_lesions), [outputPath 'actMap_small_lesions_uint16'], [refImage.PixelExtentInWorldY refImage.PixelExtentInWorldX 0.5], 'CaudatePhantomFullImageHighRes');
interfilewrite_gate(uint16(caudateFullImage_bigger_lesions), [outputPath 'actMap_bigger_lesions_uint16'], [refImage.PixelExtentInWorldY refImage.PixelExtentInWorldX 0.5], 'CaudatePhantomFullImageHighRes');
interfilewrite_gate(uint16(caudateImageFullImage_aten_gate), [outputPath 'muMap_uint16'], [refImage.PixelExtentInWorldY refImage.PixelExtentInWorldX 0.5], 'CaudatePhantomFullImageHighRes');
interfilewrite(tMu_for_recon, [outputPath 'muMap'], [refAtRecon.PixelExtentInWorldY refAtRecon.PixelExtentInWorldX ]);

% Range atenuation:
fid = fopen(sprintf('%s/RangeAttenuation.dat',outputPath), 'w');
if(fid == -1)
    fprintf('No se pudo crear el archivo RangeAttenuation.dat');
end
% En la primera línea la cantidad de materiales:
fprintf(fid,'%d\n', numMaterials);
for i = 1 : numMaterials
    % Uso colores aleatorios:
    fprintf(fid,'%.1f\t%.1f\t%s\ttrue %.1f %.1f %.1f\n', i, i, nameMaterials{i},...
        rand(1),rand(1),rand(1));
end
fclose(fid);
%% CREATE MASKS
% Convert the phantom into the grid size used in reconstruction:
[tAct_2d_small_lesions_recon, refActRecon] = ImageResample(tAct_2d_small_lesions, refAt, refAtRecon);
[tAct_2d_bigger_lesions_recon, refActRecon] = ImageResample(tAct_2d_bigger_lesions, refAt, refAtRecon);


% Now the same for the striatum phantom:
imageSize_recon = [344 344];
pixelSize_mm = [2.08625 2.08625];
indicesRoiRows_recon = 147:174;
indicesRoiCols_recon = 156:189;
imageSizeRoi_recon = [numel(indicesRoiRows_recon) numel(indicesRoiCols_recon)];
% Coordinates of striatum roi:
xLimitsAt_striatum = [-size(tMu_2d,2)/2*0.5+(indicesRoiCols(1)-1)*0.5  -size(tMu_2d,2)/2*0.5+indicesRoiCols(end)*0.5];
yLimitsAt_striatum = [-size(tMu_2d,1)/2*0.5+(indicesRoiRows(1)-1)*0.5  -size(tMu_2d,1)/2*0.5+indicesRoiRows(end)*0.5];
%zLimitsAt = [-size(tMu_2d,3)/2*0.5 size(tMu_2d,3)/2*0.5];
refActRoi  = imref2d(size(tMu_2d),xLimitsAt,yLimitsAt);
xLimits = [-imageSize_recon(2)/2*pixelSize_mm(2)+(indicesRoiCols_recon(1)-1)*pixelSize_mm(2)  -imageSize_recon(2)/2*pixelSize_mm(2)+indicesRoiCols_recon(end)*pixelSize_mm(2)];
yLimits = [-imageSize_recon(1)/2*pixelSize_mm(1)+(indicesRoiRows_recon(1)-1)*pixelSize_mm(1)  -imageSize_recon(1)/2*pixelSize_mm(1)+indicesRoiRows_recon(end)*pixelSize_mm(1)];
%zLimits = [-size(tMu_for_recon,3)/2*2.03125 size(tMu_for_recon,3)/2*2.03125];
refActRecon = imref2d(imageSizeRoi_recon,xLimits,yLimits);
[caudateFullImage_small_lesions_recon, refActRecon_roi] = ImageResample(caudateFullImage_small_lesions, refActRoi, refActRecon);
[caudateFullImage_bigger_lesions_recon, refActRecon_roi] = ImageResample(caudateFullImage_bigger_lesions, refActRoi, refActRecon);

% Create a folder for the masks:
masksPath = [outputPath '/masks/'];
if ~isdir(masksPath)
    mkdir(masksPath)
end
interfilewrite(tAct_2d_small_lesions_recon, [masksPath 'actMap_small_lesions_recon_size'], [refAtRecon.PixelExtentInWorldY refAtRecon.PixelExtentInWorldX ]);
interfilewrite(tAct_2d_bigger_lesions_recon, [masksPath 'muMactMap_bigger_lesions_recon_size'], [refAtRecon.PixelExtentInWorldY refAtRecon.PixelExtentInWorldX ]);
interfilewrite(caudateFullImage_small_lesions_recon, [masksPath 'actMap_small_lesions_recon_size_roi'], [refActRecon_roi.PixelExtentInWorldY refActRecon_roi.PixelExtentInWorldX ]);
interfilewrite(caudateFullImage_bigger_lesions_recon, [masksPath 'muMactMap_bigger_lesions_recon_size_roi'], [refActRecon_roi.PixelExtentInWorldY refActRecon_roi.PixelExtentInWorldX ]);

% call Image segmenter to generate the mask:
%imageSegmenter
% I create segmentation function with it:
[masks.Caudate,maskedImage] = segmentCaudate(caudateFullImage_small_lesions_recon);
[masks.LateralVentricle,maskedImage] = segmentLateralVentricle(caudateFullImage_small_lesions_recon);
[masks.Putamen,maskedImage] = segmentPutamen(caudateFullImage_small_lesions_recon);
[masks.ColdSmallSpot,maskedImage] = segmentSmallColdSpot(caudateFullImage_small_lesions_recon);
[masks.HotSmallSpot,maskedImage] = segmentSmallHotSpot(caudateFullImage_small_lesions_recon);
[masks.CaudateWithBiggerSpots,maskedImage] = segmentCaudateWithBiggerSpots(caudateFullImage_small_lesions_recon);
[masks.ColdBigSpot,maskedImage] = segmentBigColdSpot(caudateFullImage_small_lesions_recon);
[masks.HotBigSpot,maskedImage] = segmentBigHotSpot(caudateFullImage_small_lesions_recon);
[masks.WhiteMatter,maskedImage]  = segmentWhiteMatter(caudateFullImage_small_lesions_recon);

save([masksPath 'masks_for_metrics'], 'masks');
interfilewrite(single(masks.Caudate), [masksPath 'mask_caudate_roi'], [refActRecon_roi.PixelExtentInWorldY refActRecon_roi.PixelExtentInWorldX ]);
interfilewrite(single(masks.LateralVentricle), [masksPath 'mask_lateral_ventricle_roi'], [refActRecon_roi.PixelExtentInWorldY refActRecon_roi.PixelExtentInWorldX ]);
interfilewrite(single(masks.Putamen), [masksPath 'mask_putamen_roi'], [refActRecon_roi.PixelExtentInWorldY refActRecon_roi.PixelExtentInWorldX ]);
interfilewrite(single(masks.ColdSmallSpot), [masksPath 'mask_cold_small_spot_roi'], [refActRecon_roi.PixelExtentInWorldY refActRecon_roi.PixelExtentInWorldX ]);
interfilewrite(single(masks.HotSmallSpot), [masksPath 'mask_hot_small_spot_roi'], [refActRecon_roi.PixelExtentInWorldY refActRecon_roi.PixelExtentInWorldX ]);
interfilewrite(single(masks.CaudateWithBiggerSpots), [masksPath 'mask_caudate_bigger_spot_roi'], [refActRecon_roi.PixelExtentInWorldY refActRecon_roi.PixelExtentInWorldX ]);
interfilewrite(single(masks.ColdBigSpot), [masksPath 'mask_cold_big_spot_roi'], [refActRecon_roi.PixelExtentInWorldY refActRecon_roi.PixelExtentInWorldX ]);
interfilewrite(single(masks.HotBigSpot), [masksPath 'mask_hot_big_spot_roi'], [refActRecon_roi.PixelExtentInWorldY refActRecon_roi.PixelExtentInWorldX ]);
interfilewrite(single(masks.WhiteMatter), [masksPath 'mask_white_matter_roi'], [refActRecon_roi.PixelExtentInWorldY refActRecon_roi.PixelExtentInWorldX ]);