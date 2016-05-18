%  *********************************************************************
%  Proyecto AR-PET. Comisión Nacional de Energía Atómica.
%  Autor: Martín Belzunce. UTN-FRBA.
%  Fecha de Creación: 05/05/2010
%  *********************************************************************
% Función que escribe un sinograma en formato interfile, tiene la
% posibilidad de escribir sinogramas 2d o 3d. Cuando escribe sinogramas 2d
% la matriz de entrada debe tener 2 dimensiones para el caso de 1 solo
% sinograma, y 3 dimensiones para el caso de múltiples sinogramas 2d.
% Cuando el sinograma es 3d, el sinograma tiene que tener 4 dimensiones.

% Update 16/02/2015: Dimensión 1 pasa a ser distancia al centro, y
% dimensión 2 ángulo de proyección. Esto es apra mantener compatibilidad
% con otras bibliotecas de reconstrucción, y para mantener la misma de la
% de cpp (evitando el permute que ahcía antes). Para visualizar hayq ue
% trasponer.
% Update 17/02/2015. Add number the rings in the interfile. Receives an
% struct with the size of the sino.

% Sinograma 2D:
% dimensión 1: distancia al centro del aproyección dentro del plano
% transversal
% dimensión 2: ángulo de la proyección
% dimensión 3: eje axial, o sea, número de plano transversal 
% 
% Sinograma 3D:
% dimensión 1: distancia al centro del aproyección dentro del plano
% transversal
% dimensión 2: ángulo de la proyección
% dimensión 3: vista (view)
% dimensión 4: segmento (segment)
%
% Para escribir la imagen en este formato debo generar dos archivos, uno
% con el encabezado (*.h33) y otro con los datos crudos (*.i33).

% El sinograma 2D tengo dos opciones posibles, guardarlos como spect, o
% como pet. El formato pet no está validado oficialmente así que utilizaré
% el SPECT.

% Para el sinograma 3d no me queda otra que usar el formato PET. Uso como
% ejemplo el del stir:
% !INTERFILE  :=
% name of data file :=sino3DNema.s
% originating system := Discovery STE
% !GENERAL DATA :=
% !GENERAL IMAGE DATA :=
% !type of data := PET
% imagedata byte order := LITTLEENDIAN
% !PET STUDY (General) :=
% !PET data type := Emission
% applied corrections:={arc correction}
% !number format := float
% !number of bytes per pixel := 4
% number of dimensions := 4
% matrix axis label [4] := segment
% !matrix size [4] := 23
% matrix axis label [2] := view
% !matrix size [2] := 280
% matrix axis label [3] := axial coordinate
% !matrix size [3] := { 47, 43, 43, 39, 39, 35, 35, 31, 31, 27, 27, 23, 23, 19, 19, 15, 15, 11, 11, 7, 7, 3, 3 }
% matrix axis label [1] := tangential coordinate
% !matrix size [1] := 329
% minimum ring difference per segment := { -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15, 16, -17, 18, -19, 20, -21, 22, -23 }
% maximum ring difference per segment := { 1, 3, -2, 5, -4, 7, -6, 9, -8, 11, -10, 13, -12, 15, -14, 17, -16, 19, -18, 21, -20, 23, -22  }
% number of rings:= 64
% data offset in bytes[1] := 0
% number of time frames := 1
% !END OF INTERFILE :=

% Especificación en: http://www.medphys.ucl.ac.uk/interfile/


function interfileWriteSino(sinogram, filename, structSizeSino)

% Check what OS I am running on:
if(strcmp(computer(), 'GLNXA64'))
    os = 'linux';
    pathBar = '/';
elseif(strcmp(computer(), 'PCWIN') || strcmp(computer(), 'PCWIN64'))
    os = 'windows';
    pathBar = '\';
else
    disp('OS not compatible');
    return;
end

% Debo agregar la extensión para los nombres de ambos archivos:
filenameHeader = sprintf('%s.h33', filename);
filenameSino = sprintf('%s.i33', filename);

% Evalúo que tipo de sinograma es. Si es una matriz de tres dimensiones
% puede ser un sinograma2d por slice, o un conjunto de sinogramas 3d. Para
% este último se deben cargar los parámetros de sinogramsPerSegment, min..
% y max...
if (~isfield(structSizeSino, 'numSegments'))
    tipo = 'sinogram2D';
else
    if structSizeSino.span > 0
        % Michelograma
        tipo = 'sinogram3D';
    else
        tipo = 'sinogram2D';
    end
end

if isempty(sinogram)
    % If empty its because we want to write an empty sinogram, just to read
    % the header:
    warning('Writing empty binary file.');
else
    % Check the structure with the sinogram:
    if (structSizeSino.numR ~= size(sinogram,1)) || (structSizeSino.numTheta ~= size(sinogram,2))
        error('interfilewritesino: the size of the sinogram does not match the structure with its size.');
    end
    if strcmp(tipo,'sinogram2D') && (structSizeSino.numZ ~= size(sinogram,3))
        error('interfilewritesino: the size of the sinogram does not match the structure with its size.');
    end
    if strcmp(tipo,'sinogram3D') && (sum(structSizeSino.sinogramsPerSegment) ~= size(sinogram,3))
        error('interfilewritesino: the size of the sinogram does not match the structure with its size.');
    end
end
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
% % Debo cargar el nombre del archivo de la imagen:
% % Debo cargar el nombre del archivo de la imagen. Si tengo el h33 en un
% % path con directorios, para el i33 debo eliminarlo:
% barras = strfind(filenameSino, pathBar);
% if ~isempty(barras)
%     filenameSinoForHeader = filenameSino(barras(end)+1 : end);
% else
%     filenameSinoForHeader = filenameSino;
% end
filenameSinoForHeader = filenameSino;

fprintf(fid,'!name of data file := %s\n', filenameSinoForHeader);
fprintf(fid,'patient name := Phantom\n');
fprintf(fid,'!patient ID  := 12345\n');
fprintf(fid,'patient dob := 1968:08:21\n');
fprintf(fid,'patient sex := M\n');
fprintf(fid,'!study ID := simulation\n');
fprintf(fid,'exam type := simulation\n');
fprintf(fid,'data compression := none\n');
fprintf(fid,'data encode := none\n');
if strcmp(tipo,'sinogram2D')
    fprintf(fid,'!GENERAL IMAGE DATA :=\n');
    fprintf(fid,'!type of data := Tomographic\n');
    % Cantidad de imágenes (slices):
    fprintf(fid,'!total number of images := %d\n', structSizeSino.numZ);
    fprintf(fid,'!imagedata byte order := LITTLEENDIAN\n');
    fprintf(fid,'!number of energy windows := 1\n');
    fprintf(fid,'energy window [1] := F18m\n');
    fprintf(fid,'energy window lower level [1] := 430\n');
    fprintf(fid,'energy window upper level [1] := 620\n');
    fprintf(fid,'flood corrected := N\n');
    fprintf(fid,'decay corrected := N\n');
    % Creo que hay un beta de una versión PET, pero no sabemos si en gate anda:
    fprintf(fid,'!SPECT STUDY (general) :=\n');
    % En cabezales le pongo 6, pero es al pedo:
    fprintf(fid,'!number of detector heads := 1\n');
    % Cantidad de imágenes, sería la tercera dimensión:
    fprintf(fid,'!number of images/window := 1\n');
    % Ahora cantidad de filas y columnas:
    fprintf(fid,'!matrix size [1] := %d\n', structSizeSino.numR);
    fprintf(fid,'!matrix size [2] := %d\n', structSizeSino.numTheta);
    fprintf(fid,'!matrix size [3] := %d\n', structSizeSino.numZ);
    % Tipo de dato. Los formatos disponibles son:
    % signed integer|unsigned integer|long float|short float|bit|ASCII
    % Para esto busco el tipo de dato de la variable imagen:
    structDato = whos('sinogram');
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
%     % Si recibí el tamaño de píxel, lo agrego. En esta sección van los de
%     % (x,y):
%     if(nargin==3)
%         if(numel(sizePixel)==3)
%             fprintf(fid,'scaling factor (mm/pixel) [1] := %f\n', sizePixel(1));
%             fprintf(fid,'scaling factor (mm/pixel) [2] := %f\n', sizePixel(2));
%         else
%             disp('Error: el tamaño del pixel debe pasarse en un vector con 3 elementos.');
%         end
%     end
    % Cantidad de proyecciones 2D:
    fprintf(fid, '!number of projections := %d\n', structSizeSino.numZ);
    fprintf(fid,'!extent of rotation := 360\n');
    fprintf(fid,'!process status := acquired\n');
    % Valor máximo en la iamgen:
    fprintf(fid,'!maximum pixel count := %f\n', max(max(max(sinogram))));
%     fprintf(fid,'!SPECT STUDY (acquired data) :=\n');
%     !direction of rotation := CW
% 
%                start angle := 0
%                first projection angle in data set := 0
% 
%                acquisition mode := continuous
% 
%                Centre_of_rotation := Corrected
%     fprintf(fid,'!SPECT STUDY (reconstructed data) :=\n');
%     % Nuevamente la cantidad de imágenes:
%     fprintf(fid,'!number of slices := %d\n', size(sinogram,3));
%     fprintf(fid,'!slice orientation := Transverse\n');
%     % Esto en imágenes normales no debería cambiar:
%     fprintf(fid,'slice thickness (pixels) := 1\n');
%     if(nargin==3)
%         if(numel(sizePixel)==3)
%             fprintf(fid,'scaling factor (mm/pixel) [3] := %f\n', sizePixel(3));
%         else
%             disp('Error: el tamaño del pixel debe pasarse en un vector con 3 elementos.');
%         end
%     end
%     fprintf(fid,'centre-centre slice separation (pixels) := 1\n');
%     fprintf(fid,'scatter corrected := N\n');
%     fprintf(fid,'method of scatter correction:= none\n');
%     fprintf(fid,'oblique reconstruction := N\n');
    fprintf(fid,'!END OF INTERFILE :=\n');
elseif strcmp(tipo,'sinogram3D')
    % Para escribir el sinogram 3D:
    fprintf(fid,'!GENERAL IMAGE DATA :=\n');
    fprintf(fid,'!type of data := PET\n');
    fprintf(fid,'imagedata byte order := LITTLEENDIAN\n');
    fprintf(fid,'!PET STUDY (General) :=\n');
    fprintf(fid,'!PET data type := Emission\n');
   % fprintf(fid,'applied corrections:={arc correction}\n');
    % Tipo de dato. Los formatos disponibles son:
    % signed integer|unsigned integer|long float|short float|bit|ASCII
    % Para esto busco el tipo de dato de la variable imagen:
    structDato = whos('sinogram');
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
    fprintf(fid,'number of dimensions := 4\n');
    fprintf(fid,'matrix axis label [4] := segment\n');
    fprintf(fid,'!matrix size [4] := %d\n', numel(structSizeSino.sinogramsPerSegment));
    fprintf(fid,'matrix axis label [2] := view\n');
    fprintf(fid,'!matrix size [2] := %d\n', structSizeSino.numTheta);
    fprintf(fid,'matrix axis label [3] := axial coordinate\n');
    fprintf(fid,'!matrix size [3] := {');
    fprintf(fid,' %d', structSizeSino.sinogramsPerSegment(1));
    for i = 2 : numel(structSizeSino.sinogramsPerSegment)
         fprintf(fid,', %d', structSizeSino.sinogramsPerSegment(i));
    end
    fprintf(fid,' }\n');
    fprintf(fid,'matrix axis label [1] := tangential coordinate\n');
    fprintf(fid,'!matrix size [1] := %d\n', structSizeSino.numR);
    fprintf(fid,'minimum ring difference per segment := { ');
    fprintf(fid,' %d', structSizeSino.minRingDiff(1));
    for i = 2 : numel(structSizeSino.minRingDiff)
         fprintf(fid,', %d', structSizeSino.minRingDiff(i));
    end
    fprintf(fid,' }\n');
    fprintf(fid,'maximum ring difference per segment := { ');
    fprintf(fid,' %d', structSizeSino.maxRingDiff(1));
    for i = 2 : numel(structSizeSino.maxRingDiff)
         fprintf(fid,', %d', structSizeSino.maxRingDiff(i));
    end
    fprintf(fid,' }\n');
    fprintf(fid,'number of rings := %d\n', structSizeSino.numZ);
    fprintf(fid,'data offset in bytes[1] := 0\n');
    fprintf(fid,'number of time frames := 1\n');
    fprintf(fid,'!END OF INTERFILE :=\n');
end

% Terminé con el archivo de encabezado. Cierro el archivo:
fclose(fid);

% Ahora tengo que escribir el archivo binario de la imagen:
fid = fopen(filenameSino, 'wb');
if(fid == -1)
    fprintf('No se pudo crear el archivo %s.', filenameSino);
end
fwrite(fid, sinogram, structDato.class);
fclose(fid);
