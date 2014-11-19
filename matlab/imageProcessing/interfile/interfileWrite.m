%  *********************************************************************
%  Proyecto AR-PET. Comisión Nacional de Energía Atómica.
%  Autor: Martín Belzunce. UTN-FRBA.
%  Fecha de Creación: 05/05/2010
%  *********************************************************************
% Función que escribe una imagen en formato interfile, está pensada para
% estudios tomográficos por ahora. El objetivo es poder escribir imágenes en
% formato interfile que las pueda leer el Gate para voxelized phantom.
% Para escribir la imagen en este formato debo generar dos archivos, uno
% con el encabezado (*.h33) y otro con los datos crudos (*.i33).
% Ejemplo de una imagen con formato interfile del tipo tomographic:
% !INTERFILE :=
% !imaging modality :=nucmed 
% !originating system :=uclim
% !version of keys :=3.3
% !date of keys :=1992:01:01 
% !conversion program :=uclim
% !program author :=a. todd-pokropek
% !program version :=3.01
% !program date :=1991:07:30
% !GENERAL DATA :=
% original institution :=ucl
% contact person :=a. todd-pokropek
% data description :=tomo
% !data starting block :=0
% !name of data file :=tomo.img
% patient name :=joe doe
% !patient ID  :=12345
% patient dob :=1968:08:21
% patient sex :=M
% !study ID :=test
% exam type :=test
% data compression :=none
% data encode :=none
% !GENERAL IMAGE DATA :=
% !type of data :=Tomographic
% !imagedata byte order :=BIGENDIAN
% !number of windows :=1
% energy window [1] := Tc99m
% energy window lower level [1] := 123
% energy window upper level [1] := 155
% flood corrected :=N
% decay corrected :=N
% !SPECT STUDY (general) :=
% !number of detector heads :=1
% !number of images/window :=16
% !matrix size [1] :=64
% !matrix size [2] :=64
% !number format :=signed integer
% !number of bytes per pixel :=2
% !extent of rotation :=360
% !process status :=Reconstructed
% !maximum pixel count :=224
% !SPECT STUDY (reconstructed data) :=
% !number of slices :=16
% !slice orientation :=Transverse
% slice thickness (pixels) :=1
% centre-centre slice separation (pixels) :=1
% scatter corrected :=N
% method of scatter correction:=none
% oblique reconstruction :=N
% !END OF INTERFILE :=

% El ejemplo anterior fue sacado de:http://www.medphys.ucl.ac.uk/interfile/
% Sin embargo no podía generar imágenes que puedan ser leidas por xmedcon,
% para esto tuve que agregar el campo !total number of images :=, en la
% sección GENERAL IMAGE DATA.

% Se le agrega la opción de grabar el tamaño de píxel, para esto se le pasa
% un tercer argumento que es un vector con el tamaño de pixel en mm en x,
% y, z. Dicho argumento es opcional por lo que se la puede llamar sin
% definir el tamaño de píxel. Y en ese caso, esos campos del interfile son
% omitidos.
% interfilewrite(image, 'imageName', sizePixel)
% interfilewrite(image, 'imageName')

function interfilewrite(image, filename, sizePixel)

% Debo agregar la extensión para los nombres de ambos archivos:
filenameHeader = sprintf('%s.h33', filename);
filenameImage = sprintf('%s.i33', filename);
% Primero genero el archivo de encabezado.
fid = fopen(filenameHeader, 'w');
if(fid == -1)
    fprintf('No se pudo crear el archivo %s.', filenameHeader);
end
% Ahora debo ir escribiendo los campos. Algunos son fijos, y otros
% dependerán de la imagen:
fprintf(fid,'!INTERFILE :=\n');
fprintf(fid,'!imaging modality := nucmed \n');
fprintf(fid,'!originating system := ar-pet\n');
fprintf(fid,'!version of keys := 3.3\n');
fprintf(fid,'!date of keys := 1992:01:01\n');
fprintf(fid,'!conversion program := ar-pet\n');
fprintf(fid,'!program author := ar-pet\n');
fprintf(fid,'!program version := 1.00\n');
% Necesito escribir la fecha:
fechaHora = clock;
fprintf(fid,'!program date := %d:%d:%d\n',fechaHora(1),fechaHora(2),fechaHora(3));
fprintf(fid,'!GENERAL DATA := \n');
fprintf(fid,'original institution := cnea\n');
fprintf(fid,'contact person := m. belzunce\n');
fprintf(fid,'data description := tomo\n');
fprintf(fid,'!data starting block := 0\n');
% Debo cargar el nombre del archivo de la imagen:
% Debo cargar el nombre del archivo de la imagen. Si tengo el h33 en un
% path con directorios, para el i33 debo eliminarlo:
barras = strfind(filenameImage, '/');
if ~isempty(barras)
    filenameImageForHeader = filenameImage(barras(end)+1 : end);
else
    filenameImageForHeader = filenameImage;
end
fprintf(fid,'!name of data file := %s\n', filenameImageForHeader);
fprintf(fid,'patient name := Phantom\n');
fprintf(fid,'!patient ID  := 12345\n');
fprintf(fid,'patient dob := 1968:08:21\n');
fprintf(fid,'patient sex := M\n');
fprintf(fid,'!study ID := simulation\n');
fprintf(fid,'exam type := simulation\n');
fprintf(fid,'data compression := none\n');
fprintf(fid,'data encode := none\n');
fprintf(fid,'!GENERAL IMAGE DATA :=\n');
fprintf(fid,'!type of data := Tomographic\n');
% Cantidad de imágenes (slices):
fprintf(fid,'!total number of images := %d\n', size(image,3));
fprintf(fid,'!imagedata byte order := LITTLEENDIAN\n');
fprintf(fid,'!number of energy windows := 1\n');
fprintf(fid,'energy window [1] := F18m\n');
fprintf(fid,'energy window lower level [1] := 430\n');
fprintf(fid,'energy window upper level [1] := 620\n');
fprintf(fid,'flood corrected := N\n');
fprintf(fid,'decay corrected := N\n');
% Creo que hay un beta de una versión PET, pero no sabemos si en gate anda:
fprintf(fid,'!SPECT STUDY (general) :=\n');
% En cabezales le pongo 1, sino tira errores al leer:
fprintf(fid,'!number of detector heads := 1\n');
% Cantidad de imágenes, sería la tercera dimensión:
fprintf(fid,'!number of images/window := %d\n', size(image,3));
% Ahora cantidad de filas y columnas:
fprintf(fid,'!matrix size [1] := %d\n', size(image,2)); % Es el ancho en realidad, o sea la coordenada x que son las columnas.
fprintf(fid,'!matrix size [2] := %d\n', size(image,1)); % Filas.
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
        strFormat = 'short float';
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
% Si recibí el tamaño de píxel, lo agrego. En esta sección van los de
% (x,y):
if(nargin==3)
    if(numel(sizePixel)==3)
        fprintf(fid,'scaling factor (mm/pixel) [1] := %f\n', sizePixel(1));
        fprintf(fid,'scaling factor (mm/pixel) [2] := %f\n', sizePixel(2));
    else
        disp('Error: el tamaño del pixel debe pasarse en un vector con 3 elementos.');
    end
end
fprintf(fid,'!extent of rotation := 360\n');
fprintf(fid,'!process status := Reconstructed\n');
% Valor máximo en la iamgen:
fprintf(fid,'!maximum pixel count := %f\n', max(max(max(image))));
fprintf(fid,'!SPECT STUDY (reconstructed data) :=\n');
% Nuevamente la cantidad de imágenes:
fprintf(fid,'!number of slices := %d\n', size(image,3));
fprintf(fid,'!slice orientation := Transverse\n');
% Esto en imágenes normales no debería cambiar. En realidad es el tamaño de
% píxel en z, por lo menos así lo toma el gate en el interfile.
if(nargin==3)
    if(numel(sizePixel)==3)
        %fprintf(fid,'slice thickness (pixels) := %f\n', sizePixel(3)); %
        %Esto es solo valido para gate.
        fprintf(fid,'slice thickness (pixels) := 1\n');
    else
        fprintf(fid,'slice thickness (pixels) := 1\n');
    end
end
if(nargin==3)
    if(numel(sizePixel)==3)
        fprintf(fid,'scaling factor (mm/pixel) [3] := %f\n', sizePixel(3));
    else
        disp('Error: el tamaño del pixel debe pasarse en un vector con 3 elementos.');
    end
end
%fprintf(fid,'centre-centre slice separation (pixels) := 1\n');
fprintf(fid,'scatter corrected := N\n');
fprintf(fid,'method of scatter correction:= none\n');
fprintf(fid,'oblique reconstruction := N\n');
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
