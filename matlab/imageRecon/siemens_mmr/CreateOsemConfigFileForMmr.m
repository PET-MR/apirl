%  *********************************************************************
%  Proyecto AR-PET. Comisión Nacional de Energía Atómica.
%  Autor: Martín Belzunce. UTN-FRBA.
%  Fecha de Creación: 03/03/2015
%  *********************************************************************
% Función que escribe un archivo de configuración de reconstrucción con
% OSEM para el sinograma Mmr.

function CreateOsemConfigFileForMmr(configfilename, inputFile, initialEstimate, outputFilenamePrefix, numIterations, sensitivityImage,...
    numSubsets, saveInterval, saveIntermediate, multiplicativeSinogram, additiveSinogram, varargin)

% Primero genero el archivo de encabezado.
fid = fopen(configfilename, 'w');
if(fid == -1)
    fprintf('No se pudo crear el archivo %s.', configfilename);
end
% Ahora debo ir escribiendo los campos. Algunos son fijos, y otros
% dependerán de la imagen:
fprintf(fid,'OSEM Parameters :=\n');
fprintf(fid,'input type := Sinogram3DSiemensMmr\n');
fprintf(fid,'input file := %s\n', inputFile);
fprintf(fid,'initial estimate := %s\n', initialEstimate);
fprintf(fid,'output filename prefix := %s\n', outputFilenamePrefix);
if ~isempty(sensitivityImage)
    fprintf(fid,'sensitivity filename := %s\n', sensitivityImage);
end
% El radio del scanner y del fov no son necesarios porque son fijos para el
% Sinogram3DSiemensMmr.
fprintf(fid,'forwardprojector := Siddon\n');
fprintf(fid,'backprojector := Siddon\n');
fprintf(fid,'number of iterations := %d\n', numIterations);
fprintf(fid,'number of subsets := %d\n', numSubsets);
fprintf(fid,'save estimates at iteration intervals := %d\n', saveInterval);
fprintf(fid,'save estimated projections and backprojected image := %d\n', saveIntermediate);
% Por último las correcciones, sino están defnidos los escribo:
if nargin > 9
    if ~strcmp(multiplicativeSinogram, '')
        fprintf(fid,'multiplicative sinogram := %s\n', multiplicativeSinogram);
    end
    if ~strcmp(additiveSinogram, '')
        fprintf(fid,'additive sinogram := %s\n', additiveSinogram);
    end
end

% Additional parameters:
if numel(varargin) > 0
    if strcmp(varargin{1}, 'siddon number of samples on the detector')
        fprintf(fid,'%s := %d\n', varargin{1}, varargin{2});
    else
        disp(sprintf('Not valid additional parameter: %s.', varargin{1}))
    end
end
fclose(fid);
