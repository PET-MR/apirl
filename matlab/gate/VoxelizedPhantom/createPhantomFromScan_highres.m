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
%% OUTPUTPATH
realDataSet = 'FDG_Patient_06';
outputPath = ['/home/mab15/workspace/KCL/Biograph_mMr/GateModel/svn/' realDataSet '/'];
outputPath_roi = ['/home/mab15/workspace/KCL/Biograph_mMr/GateModel/svn/' realDataSet '_roi/'];
if ~isdir(outputPath)
    mkdir(outputPath)
end
%% INIT CLASS GPET
PET.scanner = 'mMR';
PET.method =  'otf_siddon_gpu';
PET.PSF.type = 'none';
PET.radialBinTrim = 0;
PET.Geom = '';
PET.method_for_randoms = 'from_ML_singles_matlab'; %'from_e7_binary_interfile';
PET.method_for_scatter = 'from_e7_binary_interfile';
% To change span:
PET.sinogram_size.span = 11; % Any span, 0 for multislice 2d, -1 for 2d.
PET.nSubsets = 1;
PET = classGpet(PET);
%% READ SCAN
patientPath = sprintf('/data/Scans/PatientData/%s/', realDataSet);
petDataPath = [patientPath 'e7/data-Converted/data-00/'];
sinogramFilename = [petDataPath 'data-00-sino-uncom_00.s.hdr'];
reconstructedFilename = [petDataPath 'data-00-OP_000_000.v.hdr'];
attenMapFilename_human = [petDataPath 'data-00-umap.v.hdr'];
attenMapFilename_hardware = [petDataPath 'data-00-umap-hardware.v.hdr'];
t1_filename = [patientPath '/MPRAGE_image/'];
[sinogram, delayedSinogram, structSizeSino3d] = interfileReadSino(sinogramFilename);
[reconstructedImage refMuMap] = interfileReadSiemensImage(reconstructedFilename);
[attenuationMapHuman refMuMap] = interfileReadSiemensImage(attenMapFilename_human);
[attenuationMapHardware refMuMap] = interfileReadSiemensImage(attenMapFilename_hardware);
attenuationMap = attenuationMapHuman + attenuationMapHardware;
pixelSize_mm = [refMuMap.PixelExtentInWorldX refMuMap.PixelExtentInWorldY refMuMap.PixelExtentInWorldZ];
% Read mr:
[MrInPet, refMr] = PET.getMrInPetImageSpace(t1_filename);
% Read Mr image:
[imageMr, refImageMr, imageMrFullFov, refImagePet] = PET.getMrInNativeImageSpace(t1_filename); % Return two images, the original Mr image (imageMr), and the Mr with its original pixel size but for the full fov of the pet scanner (imageMrFullFov)
% CT based mumap based on t1:
mumapAtlasCtDicomPath = '/data/Scans/PatientData/FDG_Patient_06/t1_nifty/LJF_25031962/_007_20161014/ct_umap_ucl_registered/mumap_ct/mumap_ct_dicom/DCM000/';
[mumapAtlasCtOriginal, refImageAtlasCt, mumapAtlasCtFullFov, refImagePet] = PET.getMrInNativeImageSpace(mumapAtlasCtDicomPath);
% For all the changes in format the mumap is upside down:
%mumapAtlasCtOriginal(:,:,1:end) = mumapAtlasCtOriginal(:,:,end:-1:1);
% For the same reason, I need to remap into pet space manually using the
% ref structure of the MR.
%[mumapAtlasCtInPet, refImageAtlasCtInPet] = PET.getMrInPetImageSpace(mumapAtlasCtDicomPath);
[mumapAtlasCtInPet, refResampledImage] = ImageResample(mumapAtlasCtOriginal, refImageAtlasCt, refMuMap);
%% SIZE OF THE SMALL ROI
% System matrix of 58.415 mm x 70.9325 mm
indicesRoiRows = 344/2-38:344/2-3;
indicesRoiCols = 344/2-22:344/2+22;
indicesSlices = 41:51;
imageRoi = zeros(size(reconstructedImage));
imageRoi(indicesRoiRows,indicesRoiCols,indicesSlices) = reconstructedImage(indicesRoiRows,indicesRoiCols,indicesSlices);
interfilewrite(single(imageRoi), [outputPath 'reconRoi'], pixelSize_mm);
% For the high res:
indicesRoiRows_highres = indicesRoiRows;
indicesRoiCols_highres = 344/2-22:344/2+22;
indicesSlices_highres = 41:51;
%% MU-MAP FOR RECONSTRUCTION
% I need one for the reconstruction and another to translate into Gate
% materials.
interfilewrite(single(attenuationMapHuman), [outputPath 'muMap_human'], pixelSize_mm);
interfilewrite(single(attenuationMapHardware), [outputPath 'muMap_hardware'], pixelSize_mm);
interfilewrite(single(attenuationMap), [outputPath 'muMap'], pixelSize_mm);

%% CUT THE IMAGE
% This is already reduced in size.
%% CONVERSION OF IMAGES INTO GATE NEEDS
% The activity image needs to have activity cncentration values through a
% rage trnaslator. The phantom has values from 0 to 5 aprox. For typical
% patient data, a concentration of 5 KBq/cm³. A pixel value of 1 should
% have that activit concentration.
disp('IMPORTANTE: Set .mac with a linear range scale of 1 kBq.');
conversionFactor = 5./(1./(pixelSize_mm(1)/10).*(pixelSize_mm(2)/10).*(pixelSize_mm(3)/10));
imageMr = imageMr.*conversionFactor;
interfilewrite_gate(single(imageMr), [outputPath 'actMap'], [refImageMr.PixelExtentInWorldY refImageMr.PixelExtentInWorldX refImageMr.PixelExtentInWorldZ], realDataSet);

% create a constant image.
const = zeros(size(attenuationMap));
const(attenuationMap>0.01) = 1;
interfilewrite(uint16(const), [outputPath 'constMap_uint16'], pixelSize_mm);
interfilewrite(uint16(imageMr), [outputPath 'actMap_uint16'], [refImageMr.PixelExtentInWorldY refImageMr.PixelExtentInWorldX refImageMr.PixelExtentInWorldZ]);
%% MUMAP
% I need to set the materials for each range of values:
numMaterials = 3;
rangeMaterials = [-0.1 0.03;0.03 0.12; 0.12 0.18];
nameMaterials = {'Air', 'Brain', 'Skull', 'Plastic', 'Aluminum'};
tMu_uint16 = zeros(size(attenuationMapHuman), 'uint16');
% First the human:
tMu_uint16(attenuationMapHuman>rangeMaterials(1,1) & attenuationMapHuman<rangeMaterials(1,2)) = 1;
tMu_uint16(attenuationMapHuman>rangeMaterials(2,1) & attenuationMapHuman<rangeMaterials(2,2)) = 2;
tMu_uint16(attenuationMapHuman>rangeMaterials(3,1) & attenuationMapHuman<rangeMaterials(3,2)) = 3;
% Then hardware:
rangeMaterials = [-0.1 0.03;0.03 0.09; 0.09 0.18];
%tMu_uint16(attenuationMapHardware>rangeMaterials(1,1) & attenuationMapHardware<rangeMaterials(1,2)) = 1;
tMu_uint16(attenuationMapHardware>rangeMaterials(2,1) & attenuationMapHardware<rangeMaterials(2,2)) = 4;
tMu_uint16(attenuationMapHardware>rangeMaterials(3,1) & attenuationMapHardware<rangeMaterials(3,2)) = 5;

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
interfilewrite_gate(uint16(tMu_uint16), [outputPath 'muMap_uint16'], pixelSize_mm, '');%realDataSet);
%% SMALL REGION TO SIMULATE IN GATE (ONLY ACTIVITY)
outputPath = outputPath_roi;
if ~isdir(outputPath)
    mkdir(outputPath);
end
indicesRoiRows
indicesRoiCols
indicesSlices
roiImage = uint16(imageMr(indicesRoiRows,indicesRoiCols, indicesSlices));

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
