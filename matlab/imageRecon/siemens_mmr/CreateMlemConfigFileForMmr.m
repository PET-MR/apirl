%  *********************************************************************
%  Proyecto AR-PET. Comisión Nacional de Energía Atómica.
%  Autor: Martín Belzunce. UTN-FRBA.
%  Fecha de Creación: 03/03/2015
%  *********************************************************************
% Función que escribe un archivo de configuración de reconstrucción con
% MLEM para el Mmr, tiene menos parámetros porque el radio del scanner y el
% tamaño del fov son fijos.
%
% function CreateMlemConfigFileForMmr(configfilename, inputFile, initialEstimate, outputFilenamePrefix, numIterations,...
%    saveInterval, saveIntermediate, acfSinogram, scatterSinogram, randomSinograms, normalizationFactors)
%
%
function CreateMlemConfigFileForMmr(configfilename, inputFile, initialEstimate, outputFilenamePrefix, numIterations,...
    saveInterval, saveIntermediate, acfSinogram, scatterSinogram, randomSinograms, normalizationFactors)

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
fprintf(fid,'forwardprojector := Siddon\n');
fprintf(fid,'backprojector := Siddon\n');
fprintf(fid,'number of iterations := %d\n', numIterations);
fprintf(fid,'save estimates at iteration intervals := %d\n', saveInterval);
fprintf(fid,'save estimated projections and backprojected image := %d\n', saveIntermediate);
% Por último las correcciones, sino están defnidos los escribo:
if nargin > 10
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
end
fclose(fid);
