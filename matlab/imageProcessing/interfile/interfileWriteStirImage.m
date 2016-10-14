%  *********************************************************************
%  Proyecto AR-PET. Comisión Nacional de Energía Atómica.
%  Autor: Martín Belzunce. UTN-FRBA.
%  Fecha de Creación: 05/05/2010
%  *********************************************************************
% Función que escribe una imagen en formato interfile para stir.
% Ejemplo:
% !INTERFILE  :=
% name of data file := /home/mab15/workspace/STIR/KCL/STIR_mMR_KCL/IM/pet_im_21.v
% !GENERAL DATA :=
% !GENERAL IMAGE DATA :=
% !type of data := PET
% imagedata byte order := LITTLEENDIAN
% !PET STUDY (General) :=
% !PET data type := Image
% process status := Reconstructed
% !number format := float
% !number of bytes per pixel := 4
% number of dimensions := 3
% matrix axis label [1] := x
% !matrix size [1] := 285
% scaling factor (mm/pixel) [1] := 2.08626
% matrix axis label [2] := y
% !matrix size [2] := 285
% scaling factor (mm/pixel) [2] := 2.08626
% matrix axis label [3] := z
% !matrix size [3] := 127
% scaling factor (mm/pixel) [3] := 2.03125
% first pixel offset (mm) [1] := -296.249
% first pixel offset (mm) [2] := -296.249
% first pixel offset (mm) [3] := 0
% number of time frames := 1
% !END OF INTERFILE :=

function interfileWriteStirImage(image, filename, sizePixel)

% Debo agregar la extensión para los nombres de ambos archivos:
filenameHeader = sprintf('%s.hv', filename);
filenameImage = sprintf('%s.v', filename);
% Primero genero el archivo de encabezado.
fid = fopen(filenameHeader, 'w');
if(fid == -1)
    fprintf('No se pudo crear el archivo %s.', filenameHeader);
end
% Ahora debo ir escribiendo los campos. Algunos son fijos, y otros
% dependerán de la imagen:
fprintf(fid,'!INTERFILE :=\n');
% Debo cargar el nombre del archivo de la imagen:
% Debo cargar el nombre del archivo de la imagen. Si tengo el h33 en un
% path con directorios, para el i33 debo eliminarlo:
barras = strfind(filenameImage, '/');
if ~isempty(barras)
    filenameImageForHeader = filenameImage(barras(end)+1 : end);
else
    filenameImageForHeader = filenameImage;
end
fprintf(fid,'name of data file := %s\n', filenameImageForHeader);
fprintf(fid,'!GENERAL DATA := \n');
fprintf(fid,'!GENERAL IMAGE DATA :=\n');
fprintf(fid,'!type of data := PET\n');
fprintf(fid,'imagedata byte order := LITTLEENDIAN\n');

% Creo que hay un beta de una versión PET, pero no sabemos si en gate anda:
fprintf(fid,'!PET STUDY (General) :=\n');
fprintf(fid,'!PET data type := Image\n');

% Tipo de dato. Los formatos disponibles son:
% signed integer|unsigned integer|long float|short float|bit|ASCII
% Para esto busco el tipo de dato de la variable imagen:
structDato = whos('image');
switch(structDato.class)
    case 'uint8'
        strFormat = 'unsigned integer';
        numBytesPerPixel = 1;
    case 'uint16'
        strFormat = 'unsigned integer';
        numBytesPerPixel = 2;
    case 'uint32'
        strFormat = 'unsigned integer';
        numBytesPerPixel = 4;
    case 'int8'
        strFormat = 'signed integer';
        numBytesPerPixel = 1;
    case 'int16'
        strFormat = 'signed integer';
        numBytesPerPixel = 2;
    case 'int32'
        strFormat = 'signed integer';
        numBytesPerPixel = 4;
    case 'single'
        strFormat = 'float';
        numBytesPerPixel = 4;
    case 'double'
        strFormat = 'long float';
        numBytesPerPixel = 8;
    case 'logical'
        strFormat = 'bit';
        numBytesPerPixel = 1; % No se tiene en cuenta en este caso, le pongo 1.
end
fprintf(fid,'!number format := %s\n', strFormat);
fprintf(fid,'!number of bytes per pixel := %d\n', numBytesPerPixel);
% Cantidad de dimensiones
%fprintf(fid,'!number of dimensions := %d\n', numel(size(image)));
fprintf(fid,'!number of dimensions := %d\n', 3);
% Datos de cada dimensión:
fprintf(fid,'matrix axis label [1] := x\n');
fprintf(fid,'!matrix size [1] := %d\n', size(image,2)); % Es el ancho en realidad, o sea la coordenada x que son las columnas.
fprintf(fid,'scaling factor (mm/pixel) [1] := %f\n', sizePixel(1));

fprintf(fid,'matrix axis label [2] := y\n');
fprintf(fid,'!matrix size [2] := %d\n', size(image,1)); % Es el ancho en realidad, o sea la coordenada x que son las columnas.
fprintf(fid,'scaling factor (mm/pixel) [2] := %f\n', sizePixel(2));


if numel(sizePixel) >= 3
    fprintf(fid,'matrix axis label [3] := z\n');
    fprintf(fid,'!matrix size [3] := %d\n', size(image,3)); % Es el ancho en realidad, o sea la coordenada x que son las columnas.
    fprintf(fid,'scaling factor (mm/pixel) [3] := %f\n', sizePixel(3));
else
    fprintf(fid,'matrix axis label [3] := z\n');
    fprintf(fid,'!matrix size [3] := %d\n', 1); % Es el ancho en realidad, o sea la coordenada x que son las columnas.
    fprintf(fid,'scaling factor (mm/pixel) [3] := %f\n', 1);
end
fprintf(fid,'first pixel offset (mm) [1] := %f\n', -sizePixel(1)*size(image,2)/2);
fprintf(fid,'first pixel offset (mm) [2] := %f\n', -sizePixel(2)*size(image,1)/2);
fprintf(fid,'first pixel offset (mm) [3] := %f\n', 0);

% Nuevamente la cantidad de imágenes:
fprintf(fid,'!number of frames := 1\n');

fprintf(fid,'!END OF INTERFILE :=\n');

% Terminé con el archivo de encabezado. Cierro el archivo:
fclose(fid);

% Ahora tengo que escribir el archivo binario de la imagen:
fid = fopen(filenameImage, 'wb');
if(fid == -1)
    fprintf('No se pudo crear el archivo %s.', filenameImage);
end
% Para escribirla uso la traspuesta, porque matlab guarda en memoria
% recorriendo primero filas y después pasa a la siguiente columna. Mientras
% que las imágenes por lo general se guardan recorriendo columnas - filas.
orderedImage = permute(image,[2 1 3]);
%orderedImage = image;
fwrite(fid, orderedImage, structDato.class);
fclose(fid);
