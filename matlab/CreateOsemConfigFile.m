%  *********************************************************************
%  Proyecto AR-PET. Comisión Nacional de Energía Atómica.
%  Autor: Martín Belzunce. UTN-FRBA.
%  Fecha de Creación: 30/08/2011
%  *********************************************************************
% Función que escribe un archivo de configuración de reconstrucción con
% OSEM:

function CreateOsemConfigFile(configfilename, inputType, inputFile, initialEstimate, outputFilenamePrefix, blindArea_mm, minDiffDetectors, numIterations,...
    numSubsets, saveInterval, saveIntermediate, multiplicativeSinogram, additiveSinogram)

% Primero genero el archivo de encabezado.
fid = fopen(configfilename, 'w');
if(fid == -1)
    fprintf('No se pudo crear el archivo %s.', configfilename);
end
% Ahora debo ir escribiendo los campos. Algunos son fijos, y otros
% dependerán de la imagen:
fprintf(fid,'OSEM Parameters :=\n');
fprintf(fid,'input type := %s\n', inputType);
fprintf(fid,'input file := %s\n', inputFile);
fprintf(fid,'initial estimate := %s\n', initialEstimate);
fprintf(fid,'output filename prefix := %s\n', outputFilenamePrefix);
% Pongo todos los parámetros necesarios para todos los tipos de sinogramas,
% ya que si están de más no importan:
cylindricalRadius_mm = 370;
fprintf(fid,'cylindrical pet radius (in mm) := %f\n', cylindricalRadius_mm);
fprintf(fid,'radius fov (in mm) := 300\n');
fprintf(fid,'axial fov (in mm) := 260\n');
fprintf(fid,'ArPet blind area (in mm) := %f\n', blindArea_mm);
fprintf(fid,'ArPet minimum difference between detectors := %d\n', minDiffDetectors);
fprintf(fid,'forwardprojector := ArPetProjector\n');
fprintf(fid,'backprojector := ArPetProjector\n');
fprintf(fid,'number of iterations := %d\n', numIterations);
fprintf(fid,'number of subsets := %d\n', numSubsets);
fprintf(fid,'save estimates at iteration intervals := %d\n', saveInterval);
fprintf(fid,'save estimated projections and backprojected image := %d\n', saveIntermediate);
% Por último las correcciones, sino están defnidos los escribo:
if nargin > 11
    if ~strcmp(multiplicativeSinogram, '')
    fprintf(fid,'multiplicative sinogram := %s\n', multiplicativeSinogram);
    end
    if ~strcmp(additiveSinogram, '')
        fprintf(fid,'additive sinogram := %s\n', additiveSinogram);
    end
end
fclose(fid);
