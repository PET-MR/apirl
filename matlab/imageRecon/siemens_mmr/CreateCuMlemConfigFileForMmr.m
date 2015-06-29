%  *********************************************************************
%  Proyecto AR-PET. Comisión Nacional de Energía Atómica.
%  Autor: Martín Belzunce. UTN-FRBA.
%  Fecha de Creación: 03/03/2015
%  *********************************************************************
% Función que escribe un archivo de configuración de reconstrucción con
% cuMLEM para el Mmr, el radio del scanner y el tamaño del fov son fijos.
% If the gpuId, projectorThreadsPerBlock, backprojectorThreadsPerBlock,
% updateThreadsPerBlock parameters are empty ([]) it uses a default
% configuration of 0, 576, 576 and 512.
%
% The last parameter is the number of subsets, because for cuMlem command
% you can define the number of subsets and then will run the osem version.
%
% function CreateCuMlemConfigFileForMmr(configfilename, inputFile, initialEstimate, outputFilenamePrefix, numIterations, sensitivityImage,...
%    saveInterval, saveIntermediate, multiplicativeSinogram, additiveSinogram, gpuId, projectorThreadsPerBlock, backprojectorThreadsPerBlock, updateThreadsPerBlock, numSubsets)
%
%
function CreateCuMlemConfigFileForMmr(configfilename, inputFile, initialEstimate, outputFilenamePrefix, numIterations, sensitivityImage,...
    saveInterval, saveIntermediate, multiplicativeSinogram, additiveSinogram, gpuId, projectorThreadsPerBlock, backprojectorThreadsPerBlock, updateThreadsPerBlock, numSubsets)

% Primero genero el archivo de encabezado.
fid = fopen(configfilename, 'w');
if(fid == -1)
    fprintf('No se pudo crear el archivo %s.', configfilename);
end
% Ahora debo ir escribiendo los campos. Algunos son fijos, y otros
% dependerán de la imagen:
fprintf(fid,'MLEM Parameters :=\n');
fprintf(fid,'input type := Sinogram3DSiemensMmr\n');
fprintf(fid,'input file := %s\n', inputFile);
fprintf(fid,'initial estimate := %s\n', initialEstimate);
fprintf(fid,'output filename prefix := %s\n', outputFilenamePrefix);
if ~isempty(sensitivityImage)
    fprintf(fid,'sensitivity filename := %s\n', sensitivityImage);
end

fprintf(fid,'forwardprojector := CuSiddonProjector\n');
fprintf(fid,'backprojector := CuSiddonProjector\n');

if ~isempty(projectorThreadsPerBlock)
    fprintf(fid,'projector block size := {%d,1,1}\n', projectorThreadsPerBlock);
else
    fprintf(fid,'projector block size := {576,1,1}\n');
end
if ~isempty(backprojectorThreadsPerBlock)
    fprintf(fid,'backprojector block size := {%d,1,1}\n', backprojectorThreadsPerBlock);
else
    fprintf(fid,'backprojector block size := {576,1,1}\n');
end
if ~isempty(updateThreadsPerBlock)
    fprintf(fid,'pixel update block size := {%d,1,1}\n', updateThreadsPerBlock);
else
    fprintf(fid,'pixel update block size := {512,1,1}\n');
end
if ~isempty(gpuId)
    fprintf(fid,'gpu id := %d\n', gpuId);
else
    fprintf(fid,'gpu id := 0\n');
end

fprintf(fid,'number of iterations := %d\n', numIterations);
if (nargin == 15)
    if numSubsets ~= 0
        fprintf(fid,'number of subsets := %d\n', numSubsets);
    end
end
fprintf(fid,'save estimates at iteration intervals := %d\n', saveInterval);
fprintf(fid,'save estimated projections and backprojected image := %d\n', saveIntermediate);
% Por último las correcciones, sino están defnidos los escribo:
if ~strcmp(multiplicativeSinogram, '')
    fprintf(fid,'multiplicative sinogram := %s\n', multiplicativeSinogram);
end
if ~strcmp(additiveSinogram, '')
    fprintf(fid,'additive sinogram := %s\n', additiveSinogram);
end


fclose(fid);
