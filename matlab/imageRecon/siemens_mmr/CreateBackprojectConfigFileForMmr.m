%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 15/04/2015
%  *********************************************************************
% Function that creates a config file to project an image with apirl.

% Backproject Parameters :=
% ; Archivo de configuración de reconstrucción MLEM para paper en Parallel Computing.
% input file := ../../inputData/Project_Nema_Siddon.h33
% input type := Sinogram3D
% ; Projectors:
% backprojector := Siddon
% output image := ../../../../images3D/constantImage_FullFov_256_256_81.h33
% output filename := Project_PointSources_
% END :=

% To backproject only a subset, add the following parameters:
% number of subsets := 21
% subset index := 5
%

function CreateBackprojectConfigFileForMmr(configfilename, inputFile, outputSample, outputFilename, numberOfSubsets, subsetIndex, useGpu)

if nargin == 6
    useGpu = 0;
end

% Handle the number of subsets:
if isempty(numberOfSubsets)
    numberOfSubsets = 0;
end
if(isempty(subsetIndex))
    subsetIndex = 0;
end

% Primero genero el archivo de encabezado.
fid = fopen(configfilename, 'w');
if(fid == -1)
    fprintf('No se pudo crear el archivo %s.', configfilename);
end
% Ahora debo ir escribiendo los campos. Algunos son fijos, y otros
% dependerán de la imagen:
fprintf(fid,'Backproject Parameters :=\n');
fprintf(fid,'input type := Sinogram3DSiemensMmr\n');
if useGpu == 0
    fprintf(fid,'backprojector := Siddon\n');
else
    fprintf(fid,'backprojector := CuSiddonProjector\n');
    fprintf(fid,'backprojector block size := {576,1,1}\n');
    fprintf(fid,'gpu id := 0\n');
end
% If we have subsets put it in the config file:
if numberOfSubsets ~= 0
    fprintf(fid,'number of subsets := %d\n', numberOfSubsets);
    fprintf(fid,'subset index := %d\n', subsetIndex-1); % Subset index is 0 base-index in apirl, while 1 in matlab.
end
fprintf(fid,'input file := %s\n', inputFile);
fprintf(fid,'output image := %s\n', outputSample);
fprintf(fid,'output filename := %s\n', outputFilename);

fclose(fid);
