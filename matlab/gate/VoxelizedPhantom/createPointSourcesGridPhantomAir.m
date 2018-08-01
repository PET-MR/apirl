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
%% LOAD POINT SOURCES GRID PHANTOM
outputPath = '/home/mab15/workspace/KCL/Biograph_mMr/GateModel/svn/PointSourcesGridPhantomAir/';
if ~isdir(outputPath)
    mkdir(outputPath)
end
phantomBinaryFile = '/home/mab15/workspace/KCL/Phantoms/PointSourcesGrid/nema.v';
imageSize_pixels = [344 344 127];
pixelSize_mm = [2.08625 2.08625 2.03125];

fid = fopen(phantomBinaryFile, 'r');
if fid == -1
    ferror(fid);
end
phantom = fread(fid, prod(imageSize_pixels), 'single');
phantom = reshape(phantom, imageSize_pixels);
phantom = permute(phantom, [2 1 3]);
fclose(fid);

% The phantom is for every slice a row of sources, so it can be reduced
% significantly, that is also needed because if not overlaps with the ring
% of the scanner.
pixelsToIncludeY = 344/2-20/2:344/2+20/2+1;
pixelsToIncludeX = 45 : 344-4;
reducedPhantom = phantom(pixelsToIncludeY,pixelsToIncludeX,:);
reducedImageSize_pixels = [numel(pixelsToIncludeY) numel(pixelsToIncludeX) 127];
% Coordinates to include in the phantom:
coord = -imageSize_pixels(1)/2*pixelSize_mm(1) + pixelSize_mm(1)/2: pixelSize_mm(1) : imageSize_pixels(1)/2*pixelSize_mm(1) - pixelSize_mm(1)/2;
coordZ = -imageSize_pixels(3)/2*pixelSize_mm(3) + pixelSize_mm(3)/2: pixelSize_mm(3) : imageSize_pixels(3)/2*pixelSize_mm(3) - pixelSize_mm(3)/2;
offsetY = coord(pixelsToIncludeY(1)) - pixelSize_mm(1)/2;
offsetX = coord(pixelsToIncludeX(1)) - pixelSize_mm(2)/2;
offsetZ = coordZ(1) - pixelSize_mm(3)/2;
disp(sprintf('Offsets for the .mac: %f,%f,%f', offsetX, offsetY, offsetZ));
%% COXVERSION OF IMAGES INTO GATE NEEDS
% The activity image needs to have activity cncentration values through a
% rage trnaslator. The phantom has values from 0 to 1. For typical
% patient data, a concentration of 5 KBq/cm³. A pixel value of 1 should
% have that activit concentration.
disp('IMPORTANTE: Set .mac with a linear range scale of 1 kBq.');
conversionFactor = 5./(1./(pixelSize_mm(1)/10).*(pixelSize_mm(2)/10).*(pixelSize_mm(3)/10));
reducedPhantom = reducedPhantom.*conversionFactor;
interfilewrite(reducedPhantom, [outputPath 'actMap'], pixelSize_mm);

% For gate:
interfilewrite_gate(uint16(reducedPhantom), [outputPath 'actMap_uint16'], pixelSize_mm, 'PointSourcesGridPhantomAir');
%% MUMAP
% I need to set the materials for each range of values:
numMaterials = 2;
rangeMaterials = [-0.1 0.07; 0.3 0.5];
nameMaterials = {'Air', 'Water'};
tMu_2d_uint16 = ones(size(reducedPhantom), 'uint16'); % A one means air
% Where the sources are, I add water:
waterIndex = imdilate(reducedPhantom, ones(3));
tMu_2d_uint16(logical(waterIndex)) = 1;
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
interfilewrite_gate(uint16(tMu_2d_uint16), [outputPath 'muMapWater_uint16'], pixelSize_mm, 'PointSourcesGridPhantomAir');

rangeMaterials = [-0.1 0.07; 0.3 0.5];
nameMaterials = {'Air', 'Water'};
tMu_2d_uint16 = ones(size(reducedPhantom), 'uint16'); % A one means air
% Where the sources are, I add water:
waterIndex = imdilate(reducedPhantom, ones(3));
tMu_2d_uint16(logical(waterIndex)) = 1;
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
interfilewrite_gate(uint16(tMu_2d_uint16), [outputPath 'muMapAir_uint16'], pixelSize_mm, 'PointSourcesGridPhantomAir');