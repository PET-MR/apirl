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
outputPath = '/home/mab15/workspace/KCL/Biograph_mMr/GateModel/svn_2d/RahminPhantomHighRes/';
if ~isdir(outputPath)
    mkdir(outputPath)
end
%% READ PHANTOM
pixelSize_mm = [2.08625 2.08625 2.03125]./2;
pixelSizeRecon_mm = [2.08625 2.08625 2.03125];
matrixSize = [688 688 254];
actIm = reshape(fread(fopen('/data/Simulations/FDG-methionine_Phantom/Ground_truth/FDG/fdg_1.bin'),'single'),[688,688,254]);
attIm = 10*reshape(fread(fopen('/data/Simulations/FDG-methionine_Phantom/Ground_truth/Attenuation_maps/att_1.bin'),'single'),[688,688,254]); % /cm

%% GET AN SLICE
actIm = flip(flip(permute(actIm,[2,1,3])),3);
attIm = flip(flip(permute(attIm,[2,1,3])),3);
slice = 176;
% Get 2d phantom:
actIm_2d = actIm(:,:,slice);
attIm_2d = attIm(:,:,slice);

%% MU-MAP FOR RECONSTRUCTION
tMu_for_recon = zeros(344,344);
% The mu-map needs to have the size of the reconstruction
xLimitsAt = [-size(attIm_2d,2)/2*pixelSize_mm(2) size(attIm_2d,2)/2*pixelSize_mm(2)];
yLimitsAt = [-size(attIm_2d,1)/2*pixelSize_mm(1) size(attIm_2d,1)/2*pixelSize_mm(1)];
%zLimitsAt = [-size(tMu_2d,3)/2*0.5 size(tMu_2d,3)/2*0.5];
refAt  = imref2d(size(attIm_2d),xLimitsAt,yLimitsAt);
xLimits = [-size(tMu_for_recon,2)/2*pixelSizeRecon_mm(2) size(tMu_for_recon,2)/2*pixelSizeRecon_mm(2)];
yLimits = [-size(tMu_for_recon,1)/2*pixelSizeRecon_mm(1) size(tMu_for_recon,1)/2*pixelSizeRecon_mm(1)];
%zLimits = [-size(tMu_for_recon,3)/2*2.03125 size(tMu_for_recon,3)/2*2.03125];
refAtRecon = imref2d(size(tMu_for_recon),xLimits,yLimits);
%pixelSize_mm = [2.08625 2.08625 0.5];
%[tMu refAt] = imregister(tMu, refAt,tAct, refAct, 'affine', optimizer, metric);
[tMu_for_recon, refAtRecon] = ImageResample(attIm_2d, refAt, refAtRecon);

%% GENERATE ROI
indicesPixels = 240 : 688-240;
actIm_2d_reduced = actIm_2d(indicesPixels, indicesPixels);
attIm_2d_reduced = attIm_2d(indicesPixels, indicesPixels);
% Coordinates to include in the phantom:
coord = xLimits(1) + pixelSize_mm(1)/2 : pixelSize_mm(1) : xLimits(2) - pixelSize_mm(1)/2;
coordZ = -matrixSize(3)/2*pixelSize_mm(3) + matrixSize(3)/2: pixelSize_mm(3) : matrixSize(3)/2*pixelSize_mm(3) - pixelSize_mm(3)/2;
offsetY = coord(indicesPixels(1)) - pixelSize_mm(1)/2;
offsetX = coord(indicesPixels(1)) - pixelSize_mm(2)/2;
offsetZ = -pixelSize_mm(3)/2;
disp(sprintf('Offsets for the .mac: %f,%f,%f', offsetX, offsetY, offsetZ));

%% CONVERSION OF IMAGES INTO GATE NEEDS
% The activity image needs to have activity cncentration values through a
% rage trnaslator. The phantom has values from 0 to 5 aprox. For typical
% patient data, a concentration of 5 KBq/cm³. A pixel value of 1 should
% have that activit concentration.
disp('IMPORTANTE: Set .mac with a linear range scale of 1 kBq.');
conversionFactor = 5./(1./(pixelSize_mm(1)/10).*(pixelSize_mm(2)/10).*(pixelSize_mm(3)/10));
actIm_2d_reduced = actIm_2d_reduced.*conversionFactor;
interfilewrite(actIm_2d_reduced, [outputPath 'actMap'], pixelSize_mm(1:2)); % Now I also save it without lesions.
interfilewrite_gate(uint16(actIm_2d_reduced), [outputPath 'actMap_uint16'], pixelSize_mm, 'RahminPhantomHighRes');
interfilewrite(attIm_2d_reduced, [outputPath 'muMap'], pixelSize_mm(1:2));
interfilewrite(tMu_for_recon, [outputPath 'muMap_recon'], pixelSizeRecon_mm(1:2));
%% MUMAP
% I need to set the materials for each range of values:
numMaterials = 3;
rangeMaterials = [-0.1 0.07;0.07 0.12; 0.12 0.18];
nameMaterials = {'Air', 'Brain', 'Skull'};
attIm_2d_reduced_uint16 = zeros(size(attIm_2d_reduced), 'uint16');
attIm_2d_reduced_uint16(attIm_2d_reduced>rangeMaterials(1,1) & attIm_2d_reduced<rangeMaterials(1,2)) = 1;
attIm_2d_reduced_uint16(attIm_2d_reduced>rangeMaterials(2,1) & attIm_2d_reduced<rangeMaterials(2,2)) = 2;
attIm_2d_reduced_uint16(attIm_2d_reduced>rangeMaterials(3,1) & attIm_2d_reduced<rangeMaterials(3,2)) = 3;

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
interfilewrite_gate(uint16(attIm_2d_reduced_uint16), [outputPath 'muMap_uint16'], pixelSize_mm, 'RahminPhantomHighRes');
