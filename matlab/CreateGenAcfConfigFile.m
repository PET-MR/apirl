%  *********************************************************************
%  Proyecto AR-PET. Comisión Nacional de Energía Atómica.
%  Autor: Martín Belzunce. UTN-FRBA.
%  Fecha de Creación: 30/08/2011
%  *********************************************************************
% Función que escribe un archivo de configuración para la generación de los
% attenuation correction factors:

function CreateGenAcfConfigFile(configfilename, outputType, outputProjection, inputImageFile, outputFilename)

% Primero genero el archivo de encabezado.
fid = fopen(configfilename, 'w');
if(fid == -1)
    fprintf('No se pudo crear el archivo %s.', configfilename);
end
% Ahora debo ir escribiendo los campos. Algunos son fijos, y otros
% dependerán de la imagen:
fprintf(fid,'generateACFs Parameters :=\n');
fprintf(fid,'output type := %s\n', outputType);
fprintf(fid,'input file := %s\n', inputImageFile);
fprintf(fid,'output projection := %s\n', outputProjection);
fprintf(fid,'output filename := %s\n', outputFilename);
% Uso solo los datos del cilindrical, por uso esos para los acf:
cylindricalRadius_mm = 370;
fprintf(fid,'cylindrical pet radius (in mm) := %f\n', cylindricalRadius_mm);
fprintf(fid,'radius fov (in mm) := 300\n');
fprintf(fid,'axial fov (in mm) := 260\n');
fclose(fid);