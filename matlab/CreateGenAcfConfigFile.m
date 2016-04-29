%  *********************************************************************
%  Proyecto AR-PET. Comisión Nacional de Energía Atómica.
%  Autor: Martín Belzunce. UTN-FRBA.
%  Fecha de Creación: 30/08/2011
%  *********************************************************************
% Función que escribe un archivo de configuración para la generación de los
% attenuation correction factors:
%
%  27/01/2015: Modificación con la que se pierde compatibilidad hacia
%  atrás, habrá que ir modificando las cosas sucesivamente a medida que
%  aparezcan los problema. Antes la función era:
% function CreateGenAcfConfigFile(configfilename, outputType, outputProjection, inputImageFile, outputFilename)
%   Ahora:
% function CreateGenAcfConfigFile(configfilename, structSizeSino, outputProjection, inputImageFile, outputFilename)
function CreateGenAcfConfigFile(configfilename, structSizeSino, outputProjection, inputImageFile, outputFilename, useGpu)

if nargin == 5
    useGpu = 0;
end

% Uso solo los datos del cilindrical, por uso esos para los acf:
cylindricalRadius_mm = 370;

% Primero genero el archivo de encabezado.
fid = fopen(configfilename, 'w');
if(fid == -1)
    fprintf('No se pudo crear el archivo %s.', configfilename);
end
% Sinogram type: 
% Type of sinogram, depending of the size structure type:
if isfield(structSizeSino,'sinogramsPerSegment')
    sinogramType = 'Sinogram3DSiemensMmr';
elseif structSizeSino.numZ > 1
    sinogramType = 'Sinograms2DinSiemensMmr';
else
    sinogramType = 'Sinogram2DinSiemensMmr';
end

% Ahora debo ir escribiendo los campos. Algunos son fijos, y otros
% dependerán de la imagen:
fprintf(fid,'generateACFs Parameters :=\n');
fprintf(fid,'output type := %s\n', sinogramType);
fprintf(fid,'input file := %s\n', inputImageFile);
fprintf(fid,'output projection := %s\n', outputProjection);
fprintf(fid,'output filename := %s\n', outputFilename);
if useGpu == 0
    fprintf(fid,'projector := Siddon\n');
else
    fprintf(fid,'projector := CuSiddonProjector\n');
    fprintf(fid,'projector block size := {128,1,1}\n');
    fprintf(fid,'gpu id := 0\n');
end
fprintf(fid,'cylindrical pet radius (in mm) := %f\n', cylindricalRadius_mm);
fprintf(fid,'radius fov (in mm) := %f\n', structSizeSino.rFov_mm);
fprintf(fid,'axial fov (in mm) := %f\n', structSizeSino.zFov_mm);
fclose(fid);