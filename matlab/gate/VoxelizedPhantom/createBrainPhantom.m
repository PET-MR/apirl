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
outputPath = '/home/mab15/workspace/KCL/Biograph_mMr/GateModel/svn/BrainPhantom/';
if ~isdir(outputPath)
    mkdir(outputPath)
end
load BrainMultiMaps_mMR.mat;
tAct = permute(MultiMaps_Ref.PET, [2 1 3]);
tAct = tAct(end:-1:1,:,:);
tMu = permute(MultiMaps_Ref.uMap, [2 1 3]);
tMu = tMu(end:-1:1,:,:);
pixelSize_mm = [2.08625 2.08625 2.03125];
xLimits = [-size(tAct,2)/2*pixelSize_mm(2) size(tAct,2)/2*pixelSize_mm(2)];
yLimits = [-size(tAct,1)/2*pixelSize_mm(1) size(tAct,1)/2*pixelSize_mm(1)];
zLimits = [-size(tAct,3)/2*pixelSize_mm(3) size(tAct,3)/2*pixelSize_mm(3)];
refAct = imref3d(size(tAct),xLimits,yLimits,zLimits);
refAt  = imref3d(size(tMu),xLimits,yLimits,zLimits);
%% ADD TUMOUR
centre_mm = [40 -75 15];
radius_mm = 5;
[X,Y,Z] = meshgrid([xLimits(1)+pixelSize_mm(1)/2:pixelSize_mm(1):xLimits(2)-pixelSize_mm(1)/2],[yLimits(1)+pixelSize_mm(2)/2:pixelSize_mm(2):yLimits(2)-pixelSize_mm(2)/2],[zLimits(1)+pixelSize_mm(3)/2:pixelSize_mm(3):zLimits(2)-pixelSize_mm(3)/2]);
indicesTumour = ((X-centre_mm(1)).^2 + (Y-centre_mm(2)).^2 + (Z-centre_mm(3)).^2) < radius_mm.^3;
tAct(indicesTumour) = mean(tAct(indicesTumour))*2;

centre_mm = [52 12 10];
radius_mm = 2;
[X,Y,Z] = meshgrid([xLimits(1)+pixelSize_mm(1)/2:pixelSize_mm(1):xLimits(2)-pixelSize_mm(1)/2],[yLimits(1)+pixelSize_mm(2)/2:pixelSize_mm(2):yLimits(2)-pixelSize_mm(2)/2],[zLimits(1)+pixelSize_mm(3)/2:pixelSize_mm(3):zLimits(2)-pixelSize_mm(3)/2]);
indicesTumour = ((X-centre_mm(1)).^2 + (Y-centre_mm(2)).^2 + (Z-centre_mm(3)).^2) < radius_mm.^3;
tAct(indicesTumour) = mean(tAct(indicesTumour))*2;

% Normalize to maximum of unsigned int:
tAct = tAct ./ max(max(max(tAct))) .* (2^12-1);
%% CUT THE IMAGE
% If we keep the full size image, we have the problem that overlaps
% withdetectors:
indexReduced = 70:size(tAct,1)-70;
tAct_reduced = tAct(indexReduced, indexReduced, :);
tMu_reduced = tMu(indexReduced, indexReduced, :);
xLimits = [-size(tAct_reduced,2)/2*2.08625 size(tAct_reduced,2)/2*2.08625];
yLimits = [-size(tMu_reduced,1)/2*2.08625 size(tMu_reduced,1)/2*2.08625];
zLimits = [-size(tMu_reduced,3)/2*2.03125 size(tMu_reduced,3)/2*2.03125];
refAct_reduced = imref3d(size(tAct_reduced),xLimits,yLimits,zLimits);
refAt_reduced = imref3d(size(tMu_reduced),xLimits,yLimits,zLimits);
%% CONVERSION OF IMAGES INTO GATE NEEDS
% The activity image needs to have activity cncentration values through a
% rage trnaslator. The phantom has values from 0 to 5 aprox. For typical
% patient data, a concentration of 5 KBq/cm³. A pixel value of 1 should
% have that activit concentration.
disp('IMPORTANTE: Set .mac with a linear range scale of 1 kBq.');
conversionFactor = 5./(1./(pixelSize_mm(1)/10).*(pixelSize_mm(2)/10).*(pixelSize_mm(3)/10));
tAct = tAct.*conversionFactor;
tAct_reduced = tAct_reduced.*conversionFactor;
interfilewrite(tAct, [outputPath 'actMap'], [refAct.PixelExtentInWorldY refAct.PixelExtentInWorldX refAct.PixelExtentInWorldZ]);
interfilewrite(tMu, [outputPath 'muMap'], [refAt.PixelExtentInWorldY refAt.PixelExtentInWorldX refAt.PixelExtentInWorldZ]);

interfilewrite(uint16(tAct), [outputPath 'actMap_uint16'], [refAct.PixelExtentInWorldY refAct.PixelExtentInWorldX refAct.PixelExtentInWorldZ]);
interfilewrite(uint16(tAct_reduced), [outputPath 'actMap_reduced_uint16'], [refAct.PixelExtentInWorldY refAct.PixelExtentInWorldX refAct.PixelExtentInWorldZ]);
%% MUMAP
% I need to set the materials for each range of values:
numMaterials = 3;
rangeMaterials = [-0.1 0.07;0.07 0.12; 0.12 0.18];
nameMaterials = {'Air', 'Brain', 'Skull'};
tMu_uint16 = zeros(size(tMu), 'uint16');
tMu_uint16(tMu>rangeMaterials(1,1) & tMu<rangeMaterials(1,2)) = 1;
tMu_uint16(tMu>rangeMaterials(2,1) & tMu<rangeMaterials(2,2)) = 2;
tMu_uint16(tMu>rangeMaterials(3,1) & tMu<rangeMaterials(3,2)) = 3;
tMu_reduced_uint16 = zeros(size(tMu_reduced), 'uint16');
tMu_reduced_uint16(tMu_reduced>rangeMaterials(1,1) & tMu_reduced<rangeMaterials(1,2)) = 1;
tMu_reduced_uint16(tMu_reduced>rangeMaterials(2,1) & tMu_reduced<rangeMaterials(2,2)) = 2;
tMu_reduced_uint16(tMu_reduced>rangeMaterials(3,1) & tMu_reduced<rangeMaterials(3,2)) = 3;
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
interfilewrite(uint16(tMu_uint16), [outputPath 'muMap_uint16'], [refAt.PixelExtentInWorldY refAt.PixelExtentInWorldX refAt.PixelExtentInWorldZ]);
interfilewrite(uint16(tMu_reduced_uint16), [outputPath 'muMap_reduced_uint16'], [refAt.PixelExtentInWorldY refAt.PixelExtentInWorldX refAt.PixelExtentInWorldZ]);

%% SMALL REGION TO SIMULATE IN GATE
outputPath = '/home/mab15/workspace/KCL/Biograph_mMr/GateModel/svn/CaudatePhantom/';
caudateImage = uint16(tAct(138:165,155:188,56:62));
% Hot spot:
caudateImage(round(size(caudateImage,1)/2), 10,4) = caudateImage(round(size(caudateImage,1)/2), 10,4)*2;
% Cold spot:
caudateImage(round(size(caudateImage,1)/2), 25,4) = caudateImage(round(size(caudateImage,1)/2), 25,4)*0.2;
caudateImage_aten = tMu_uint16(138:165,155:188,56:62);
xLimits = [-size(caudateImage,2)/2*2.08625 size(caudateImage,2)/2*2.08625];
yLimits = [-size(caudateImage,1)/2*2.08625 size(caudateImage,1)/2*2.08625];
zLimits = [-size(caudateImage,3)/2*2.03125 size(caudateImage,3)/2*2.03125];
refCaudate = imref3d(size(caudateImage),xLimits,yLimits,zLimits);
interfilewrite(uint16(caudateImage), [outputPath 'actMap_uint16'], [refCaudate.PixelExtentInWorldY refCaudate.PixelExtentInWorldX refCaudate.PixelExtentInWorldZ]);
interfilewrite(uint16(caudateImage_aten), [outputPath 'muMap_uint16'], [refCaudate.PixelExtentInWorldY refCaudate.PixelExtentInWorldX refCaudate.PixelExtentInWorldZ]);
