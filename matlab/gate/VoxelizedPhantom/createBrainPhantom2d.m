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
outputPath = '/home/mab15/workspace/KCL/Biograph_mMr/GateModel/svn_2d/BrainPhantom/';
if ~isdir(outputPath)
    mkdir(outputPath)
end
load subject_4_tpm.mat;
load brainWeb3D.mat;
% tAct: activity image. Need to transpose and invert the y axis.
% tMu: attenuationMap. Is in different scale and size.
tAct = permute(tAct, [2 1 3]);
tAct = tAct(end:-1:1,:,:);
tMu = permute(tMu, [2 1 3]);
tMu = tMu(end:-1:1,:,:);
% Register both images:
[optimizer,metric] = imregconfig('multimodal');
optimizer.InitialRadius = 0.01;
xLimitsAt = [-size(tMu,2)/2*1.8 size(tMu,2)/2*1.8];
yLimitsAt = [-size(tMu,1)/2*1.8 size(tMu,1)/2*1.8];
zLimitsAt = [-size(tMu,3)/2*2.5 size(tMu,3)/2*2.5];
refAt  = imref3d(size(tMu),xLimitsAt,yLimitsAt,zLimitsAt);
xLimits = [-size(tAct,2)/2*2.08625 size(tAct,2)/2*2.08625];
yLimits = [-size(tAct,1)/2*2.08625 size(tAct,1)/2*2.08625];
zLimits = [-size(tAct,3)/2*2.03125 size(tAct,3)/2*2.03125];
refAct = imref3d(size(tAct),xLimits,yLimits,zLimits);
pixelSize_mm = [2.08625 2.08625 2.03125];
%[tMu refAt] = imregister(tMu, refAt,tAct, refAct, 'affine', optimizer, metric);
[tMu, refAt] = ImageResample(tMu, refAt, refAct);
%% GET AN SLICE
slice = 59; % fOR THE CAUDATE.
tAct_2d = tAct(:,:,slice);
tMu_2d = tMu(:,:,slice);
%% CUT THE IMAGE
% If we keep the full size image, we have the problem that overlaps
% withdetectors:
indexReduced = 70:size(tAct,1)-70;
tAct_2d_reduced = tAct_2d(indexReduced, indexReduced);
tMut_2d_reduced = tMu_2d(indexReduced, indexReduced);
xLimits = [-size(tAct_2d_reduced,2)/2*2.08625 size(tAct_2d_reduced,2)/2*2.08625];
yLimits = [-size(tMut_2d_reduced,1)/2*2.08625 size(tMut_2d_reduced,1)/2*2.08625];
zLimits = 0;
refAct_2d_reduced = imref2d(size(tAct_2d_reduced),xLimits,yLimits);
refAt_2d_reduced = imref2d(size(tMut_2d_reduced),xLimits,yLimits);
%% CONVERSION OF IMAGES INTO GATE NEEDS
% The activity image needs to have activity cncentration values through a
% rage trnaslator. The phantom has values from 0 to 5 aprox. For typical
% patient data, a concentration of 5 KBq/cm³. A pixel value of 1 should
% have that activit concentration.
disp('IMPORTANTE: Set .mac with a linear range scale of 1 kBq.');
conversionFactor = 5./(1./(pixelSize_mm(1)/10).*(pixelSize_mm(2)/10).*(pixelSize_mm(3)/10));
tAct_2d = tAct_2d.*conversionFactor;
tAct_2d_reduced = tAct_2d_reduced.*conversionFactor;
interfilewrite(tAct_2d, [outputPath 'actMap'], [refAct_2d_reduced.PixelExtentInWorldY refAct_2d_reduced.PixelExtentInWorldX]);
interfilewrite(tMu_2d, [outputPath 'muMap'], [refAct_2d_reduced.PixelExtentInWorldY refAct_2d_reduced.PixelExtentInWorldX]);

interfilewrite(uint16(tAct_2d), [outputPath 'actMap_uint16'], [refAct_2d_reduced.PixelExtentInWorldY refAct_2d_reduced.PixelExtentInWorldX]);
interfilewrite(uint16(tAct_2d_reduced), [outputPath 'actMap_reduced_uint16'], [refAct_2d_reduced.PixelExtentInWorldY refAct_2d_reduced.PixelExtentInWorldX]);
%% MUMAP
% I need to set the materials for each range of values:
numMaterials = 3;
rangeMaterials = [-0.1 0.07;0.07 0.12; 0.12 0.18];
nameMaterials = {'Air', 'Brain', 'Skull'};
tMu_2d_uint16 = zeros(size(tMu_2d), 'uint16');
tMu_2d_uint16(tMu_2d>rangeMaterials(1,1) & tMu_2d<rangeMaterials(1,2)) = 1;
tMu_2d_uint16(tMu_2d>rangeMaterials(2,1) & tMu_2d<rangeMaterials(2,2)) = 2;
tMu_2d_uint16(tMu_2d>rangeMaterials(3,1) & tMu_2d<rangeMaterials(3,2)) = 3;
tMu_2d_reduced_uint16 = zeros(size(tMut_2d_reduced), 'uint16');
tMu_2d_reduced_uint16(tMut_2d_reduced>rangeMaterials(1,1) & tMut_2d_reduced<rangeMaterials(1,2)) = 1;
tMu_2d_reduced_uint16(tMut_2d_reduced>rangeMaterials(2,1) & tMut_2d_reduced<rangeMaterials(2,2)) = 2;
tMu_2d_reduced_uint16(tMut_2d_reduced>rangeMaterials(3,1) & tMut_2d_reduced<rangeMaterials(3,2)) = 3;
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
interfilewrite(uint16(tMu_2d_uint16), [outputPath 'muMap_uint16'], [refAt_2d_reduced.PixelExtentInWorldY refAt_2d_reduced.PixelExtentInWorldX]);
interfilewrite(uint16(tMu_2d_reduced_uint16), [outputPath 'muMap_reduced_uint16'], [refAt_2d_reduced.PixelExtentInWorldY refAt_2d_reduced.PixelExtentInWorldX]);
%% SMALL REGION TO SIMULATE IN GATE
outputPath = '/home/mab15/workspace/KCL/Biograph_mMr/GateModel/svn_2d/CaudatePhantom/';
caudateImage = uint16(tAct_2d(138:165,155:188));
% Hot spot:
caudateImage(round(size(caudateImage,1)/2), 10) = caudateImage(round(size(caudateImage,1)/2), 10)*2;
% Cold spot:
caudateImage(round(size(caudateImage,1)/2), 25) = caudateImage(round(size(caudateImage,1)/2), 25)*0.2;
caudateImage_aten = tMu_2d_uint16(138:165,155:188);
xLimits = [-size(caudateImage,2)/2*2.08625 size(caudateImage,2)/2*2.08625];
yLimits = [-size(caudateImage,1)/2*2.08625 size(caudateImage,1)/2*2.08625];
zLimits = 0;
refCaudate = imref2d(size(caudateImage),xLimits,yLimits);
interfilewrite(uint16(caudateImage), [outputPath 'actMap_uint16'], [refCaudate.PixelExtentInWorldY refCaudate.PixelExtentInWorldX ]);
interfilewrite(uint16(caudateImage_aten), [outputPath 'muMap_uint16'], [refCaudate.PixelExtentInWorldY refCaudate.PixelExtentInWorldX ]);
% create a constant image.
const = ones(size(caudateImage));
interfilewrite(uint16(const), [outputPath 'constMap_uint16'], [refCaudate.PixelExtentInWorldY refCaudate.PixelExtentInWorldX ]);
