%  *********************************************************************
%  Proyecto AR-TGS. Comisión Nacional de Energía Atómica.
%  Autor: Martín Belzunce. UTN-FRBA.
%  Fecha de Creación: 20/09/2013
%  *********************************************************************
% Función que realiza la reconstrucción de un sinograma 2D y calibra la
% imagen en actividad. Para esto recibe un sinograma de entrada, el tamaño
% de la imagen de la salida, la matriz del sistema para realizar la
% reconstrucción y un sinograma de calibración. Este último es un sinograma
% adqurido de un cilindro con actividad uniforme y conocida (actInCalSino_mCi) en un tiempo conocido.
% Tiene un séptimo parámetro opcional para determinar si la reconstrucción es
% penalizada, en caso de serlo es un penalización de mediana con un peso
% fijo. En el futuro se puede cambiar ese peso para que sea configurable.

% 07/05/2014: Hago que calibrationSinogram pueda ser el nombre del archivo
% como era hasta ahora o la matriz con el sinograma directamente. Para eso
% verifico si es un strin o no.
function reconstructedImage_uCi_pixel = Mlem2dCalActivity(inputSinogram, effRelToCs, tiempoAdquirido_sec, outputSize, filenameSystemMatrix, calibrationSinogram, ...
    tiempoAdqCalSino, actInCalSino_mCi, filenameSystemMatrixCal, isPenalized)
% Verifico si me pasaron el sexto parámetro:
if(nargin < 10)
    isPenalized = false;
else
    penalizationWeight = 0.2;
end
reconstructedImage_uCi_pixel = zeros(outputSize);
numIter = 80;
numPixels = outputSize(1)*outputSize(2);
numBins = size(inputSinogram,1)*size(inputSinogram,2);
%% LECTURA DE SINOGRAMA DE CALIBRACIÓN Y MATRIZ
% Matriz del sistema para reconstrucción de calibración:.
fid = fopen(filenameSystemMatrixCal, 'r');
if fid ~= -1
    % La generé en orden por fila:
    systemMatrixCal = fread(fid, [numPixels numBins], 'single');
    % Peroi matlab las lee en roden de columna, por lo que tengo que hacer la
    % traspuesta:
    systemMatrixCal = systemMatrixCal';
    fclose(fid);
else
    error('Error al leer la matriz del sistema.');
end
% Matriz del sistema para reconstrucción:
fid = fopen(filenameSystemMatrix, 'r');
if fid ~= -1
    % La generé en orden por fila:
    systemMatrix = fread(fid, [numPixels numBins], 'single');
    % Peroi matlab las lee en roden de columna, por lo que tengo que hacer la
    % traspuesta:
    systemMatrix = systemMatrix';
    fclose(fid);
else
    error('Error al leer la matriz del sistema.');
end
% Sinograma de calibración:
% Me fijo si es un string para leer el archivo o si es la matriz
% directamente:
if ischar(calibrationSinogram)
    calibrationSinogram = interfileread(filenameCalibrationSinogram);
elseif size(calibrationSinogram) ~= size(inputSinogram)
        error('El tamaño del sinograma de calibración es distinto del de entrada.');
end
%% MATRIZ DE CALIBRACIÓN
% Reconstruyo el sinograma de calibración.
if  isPenalized
    calibrationImages = Mlem2Dpenalized(calibrationSinogram, systemMatrixCal, systemMatrix, ...
        outputSize, numIter, false, 'mediana', penalizationWeight);
else
    calibrationImages = Mlem2D(calibrationSinogram, systemMatrixCal, systemMatrix, ...
        outputSize, numIter, false);
    % Filtro de mediana para eliminarle el ruido:
    calibrationImages = medfilt2(calibrationImages, [3 3]);
end
% Genero imagen de calibración a cuentas.
% Para esta actividad la imagen de salida reconstruida da aprox 1.
% La actividad total en el fantoma es de 10 mCi. La conectración por cm3 es
% de 10 mCi/(pi*300mm^2*100mm) = 0.35 uCi/cm³. Si lo considero solo como una
% feta la conecntración es de 10 mCi/(pi*300mm^2)=11/uCi/cm². 
% Voy a dividr la imagen de actividad teórica con la imagen reconstruida 
% con normalización. Esta va a ser la imagen que pase a calibración.
% Para el paper trabajo con el segmento fino. La altura del segmento es de
% solo  4mm.
timeSimulated_sec = tiempoAdqCalSino;
totalActivity_uCi = actInCalSino_mCi .* 1000;
radius_cm = 30;
heightSegment_cm = 4;
totalArea_cm2 = pi * radius_cm.^2;
sizePixel_cm = 2 * radius_cm / outputSize(1);
areaPixel_cm2 = sizePixel_cm ^2;
volPixel_cm2 = areaPixel_cm2 * heightSegment_cm;
% mCi por Pixel para esta simulación:
imageConcentration_uCi_pixel = zeros(outputSize);
% Fov circular:
x = 1 : size(imageConcentration_uCi_pixel,2);
y = 1 : size(imageConcentration_uCi_pixel,1);
[X,Y] = meshgrid(x,y);
indicesCirculo = ((X-size(imageConcentration_uCi_pixel,2)/2-0.5).^2 + (Y-size(imageConcentration_uCi_pixel,1)/2-0.5).^2) <= (size(imageConcentration_uCi_pixel,1)/2).^2;
imageConcentration_uCi_pixel(indicesCirculo) = totalActivity_uCi/totalArea_cm2 * areaPixel_cm2;
% La imagen Concentracion por uCi de referencia la uso de calibración con
% la reconstruída en cps. O sea la iamgen reconstruida la divido por el
% tiempo de simulación.
% Ahora para la imagen reconstruida del fantoma de calibracion, genera
% una imagen de conversión de cps a uCi:
% Convierto a tasa la medición:
calibrationImagesInCps = calibrationImages ./ timeSimulated_sec;
% Ahora la imagen de conversión:
imageConverter_CountRate_to_uCi_pixel = imageConcentration_uCi_pixel ./ calibrationImagesInCps;
% Los píxels que estaban en cero, dan infinito, los vuelvo a cero:
imageConverter_CountRate_to_uCi_pixel(isinf(imageConverter_CountRate_to_uCi_pixel)|isnan(imageConverter_CountRate_to_uCi_pixel)) = 0;
% h = figure;
% imshow(imageConverter_CountRate_to_uCi_pixel./max(max(imageConverter_CountRate_to_uCi_pixel)));
% title('Imagen de conversión a actividad');
%% RECONSTRUCCIÓN DE LA IMAGEN
diamCilindro_mm = 600;
sizePixel_cm = [diamCilindro_mm/outputSize(1) diamCilindro_mm/outputSize(2) 5];   % Las imágenes las reconstruí usando una simulación con 0.4 de altura de píxel, pero luego de la conversión
                            % me queda en uCi/pixel. Para pasarla a
                            % concentración la paso a alto de segmento de 5
                            % cm.
% Filtro post procesamiento:
h_filter = ones(3);
h_filter = h_filter./(sum(sum(h_filter)));
if  isPenalized
    [reconstructedImages likelihoodVector] = Mlem2Dpenalized(inputSinogram, systemMatrix, systemMatrix, ...
        outputSize, numIter, false, 'mediana', penalizationWeight);

else
    [reconstructedImages likelihoodVector] = Mlem2D(inputSinogram, systemMatrix, systemMatrix, ...
        outputSize, numIter, false);
    reconstructedImages = imfilter(reconstructedImages, h_filter);
end
% Una vez que la reconstrui, le aplico la conversión a actividad:
reconstructedImage_uCi_pixel =(reconstructedImages./tiempoAdquirido_sec).*imageConverter_CountRate_to_uCi_pixel*effRelToCs;
% La activityImages están en uCi/pixel, también las genero en uCi/cc:
activityImage_uCi_cc = reconstructedImage_uCi_pixel ./ (sizePixel_cm(1)*sizePixel_cm(2)*sizePixel_cm(3)); 