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
% function CreateCuMlemConfigFileForMmr(configfilename, inputFile, initialEstimate, outputFilenamePrefix, numIterations, sensitivityImage,...
%    saveInterval, saveIntermediate, acfSinogram, scatterSinogram, randomSinograms, normalizationFactors, gpuId, projectorThreadsPerBlock, backprojectorThreadsPerBlock, updateThreadsPerBlock)
%
%
function CreateCuMlemConfigFileForMmr(configfilename, inputFile, initialEstimate, outputFilenamePrefix, numIterations, sensitivityImage,...
    saveInterval, saveIntermediate, acfSinogram, scatterSinogram, randomSinograms, normalizationFactors, gpuId, projectorThreadsPerBlock, backprojectorThreadsPerBlock, updateThreadsPerBlock)

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
fprintf(fid,'save estimates at iteration intervals := %d\n', saveInterval);
fprintf(fid,'save estimated projections and backprojected image := %d\n', saveIntermediate);
% Por último las correcciones, sino están defnidos los escribo:
if ~strcmp(acfSinogram, '')
    fprintf(fid,'attenuation correction factors := %s\n', acfSinogram);
end
if ~strcmp(scatterSinogram, '')
    fprintf(fid,'scatter correction sinogram := %s\n', scatterSinogram);
end
if ~strcmp(randomSinograms, '')
    fprintf(fid,'randoms correction sinogram := %s\n', randomSinograms);
end
if ~strcmp(normalizationFactors, '')
    fprintf(fid,'normalization correction factors := %s\n', normalizationFactors);
end


fclose(fid);
