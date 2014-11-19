%  *********************************************************************
%  Proyecto AR-PET. Comisión Nacional de Energía Atómica.
%  Autor: Martín Belzunce. UTN-FRBA.
%  Fecha de Creación: 15/02/2012
%  *********************************************************************

% Función que aplica una imagen de calibración a una imagen ya recosntruida.
% Es para ecualizar la ganancia de cada píxel.
function outputImage = calibrarImagen(interfileImageName, interfileCalibrationName, totalActivity, outputFilename)
% Imagen reconstruida:
inputImage = interfileread(interfileImageName);

% Imagen de calibración:
calibrationImage = interfileread(interfileCalibrationName);

% Aplico normalización:
calibrationImage = calibrationImage .* totalActivity ./sum(sum(calibrationImage));
outputImage = inputImage .* calibrationImage;

% Escribo en interfile:
interfilewrite(outputImage, outputFilename);

