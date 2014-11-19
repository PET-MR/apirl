%  *********************************************************************
%  Proyecto AR-PET. Comisión Nacional de Energía Atómica.
%  Autor: Martín Belzunce. UTN-FRBA. contacto: martin.a.belzunce@gmail.com
%  Fecha de Creación: 18/04/2011
%  *********************************************************************

% Este script crea un fantoma de Image Quality de NEMA NU 2-2001  
% para Gate utilizando voxelized phantom.
% Para generar el voxelized phantom necesito dos imágenes interfiles, una
% para el mapa de atenuación y otra para la distribución de actividades.
% Las imágenes deben ser unsigned short int, y los valores de cada voxel
% son índices a una tabla donde se contendrá los valores reales de
% atenuación y actividad.

% El fantoma de IQ tiene en el plano XY dimensiones de 300x230mmx200mm. Los
% espesores del plástico exterior del fantoma serán de 3mm, mientras que
% las tapas serán de 10mm. El espesor del plástico de las esferas es de
% 1mm. Por lo que utilizaremos tamaño del píxel 1mm.

clear all 
close all
% Necesito la función interfilewrite, si no está en path la agrego:
addpath('/sources/MATLAB/WorkingCopy/ImageProcessing');
% Directorio de salida del fantoma:
outputPath = '..';
%% INICIALIZACIÓN IMÁGENES
sizePixel_mm = 2;
sizeImage_mm = [300 300 200];
sizeImage_pixels = sizeImage_mm ./ sizePixel_mm;
imageAtenuation = uint16(zeros(sizeImage_pixels));
imageActivity = uint16(zeros(sizeImage_pixels));

% El x vanza como los índices, osea a la izquierda es menor, a la derecha
% mayor.
coordX = -((sizeImage_mm(2)/2)-sizePixel_mm/2):sizePixel_mm:((sizeImage_mm(2)/2)-sizePixel_mm/2);
% El y y el z van al revés que los índices, o sea el valor geométrico va a
% contramano con los índices de las matrices.
coordY = -((sizeImage_mm(1)/2)-sizePixel_mm/2):sizePixel_mm:((sizeImage_mm(1)/2)-sizePixel_mm/2);
coordZ = -((sizeImage_mm(3)/2)-sizePixel_mm/2):sizePixel_mm:((sizeImage_mm(3)/2)-sizePixel_mm/2);
[X,Y,Z] = meshgrid(coordX, coordY, coordZ);
origin = [0 0 0];
%% IMAGEN INTERFILE
% Se crea la imagen de atenuación y activdad, que está formada por agua en su
% interior, el aire exterior, el inserto de atenuación y el Plástico de los
% bordes. Esta imagen contiene indices del material al que corresponde cada
% píxel, luego eso
indexAir = 0;
indexWater = 1;
indexLung = 2;
indexPlastic = 3;
indexColdSpheres = 4;
indexHotSpheres= 5;
% Genero vectores con esta información para luego generar el archivo de
% rangos de atenuación:
numMaterials = 2;
indexMaterials = [0 1];
nameMaterials = {'Air', 'Water'};
% Genero el interior de agua que está formado por medio cilindro centrado
% en (0;-35).
radiusCylinder_mm = 100;
indexCylinder = (sqrt((X-0).^2+(Y-0).^2) < radiusCylinder_mm)  & (Z>-90)&(Z<90);
imageAtenuation(indexCylinder) = indexWater;

% Por último todos los bordes de vidrio:

%% IMAGEN DE ACTIVIDAD
% Para pasar los voxeles a actividad hay dos opciones: un conversor lineal
% o otro por rangos de actividad. Utilizaremos el que es por rango de
% actividad.
% Según la norma nema las conectracciones de actividad deben ser las
% siguientes:
% Concentración de actividad 1.0:
backgroundActivityConcentration_kBq_cc = 1.0;
% Considero que el linear range se setea con 1 = 1 Bq
disp('IMPORTANTE: Setear el linear range del .mac con escala de 1 Bq.');
% O sea que en la imagen de actividad, el valor 1 representa 1 Bq. Ahora
% calculo los valores de actividad de los píxeles:
backgroundActivityConcentration_Bq = backgroundActivityConcentration_kBq_cc * (sizePixel_mm/10)^3 *1000;
% Seteo la actividad de fondo:
imageActivity(indexCylinder) = backgroundActivityConcentration_Bq;


%% VISUALIZACIÓN
% Veo la imagen 3D, mostrando los slices secuencialmente-
% Primero atenuación:
h = figure;
maxValueAtenuation = double(max(max(max(imageAtenuation))));
for i = 1 : size(imageAtenuation,3)
    imshow(double(imageAtenuation(:,:,i))./maxValueAtenuation);
    colormap(jet);
    title(sprintf('Imagen de Atenuación. Slice %d de %d.', i, size(imageAtenuation,3)));
    pause(0.2);
end
% Luego actividad:
h = figure;
maxValueActivity = double(max(max(max(imageActivity))));
for i = 1 : size(imageActivity,3)
    imshow(double(imageActivity(:,:,i))./maxValueActivity);
    colormap(jet);
    title(sprintf('Imagen de Actividad. Slice %d de %d.', i, size(imageActivity,3)));
    pause(0.2);
end
%% ESCRITURA DE IMÁGENES EN INTERFILE
% Por ahora los píxeles son del mismo tamaño en los 3 ejes:
sizePixel = [sizePixel_mm sizePixel_mm sizePixel_mm];
% Escribo los archivos de salida:
interfilewrite(imageActivity, sprintf('%s/CilindricalVoxelizedActivity',outputPath), sizePixel);
interfilewrite(imageAtenuation, sprintf('%s/CilindricalVoxelizedAttenuation',outputPath), sizePixel);

%% ESRCITURA DE ARCHIVO DE RANGOS DE ATENUACIÓN
% Primero genero el archivo de encabezado.
fid = fopen(sprintf('%s/RangeAttenuation.dat',outputPath), 'w');
if(fid == -1)
    fprintf('No se pudo crear el archivo RangeAttenuation.dat');
end
% Ejemplo
%   4
%   0     0     Air         false 0.0  0.0  0.0  0.2
%   4     4     Water       true 1.0  0.0  0.0  0.2
%   5     5     Water       true 0.0  1.0  0.0  0.2
%   14    15    Water       true 0.0  0.0  1.0  0.2
% En la primera línea la cantidad de materiales:
fprintf(fid,'%d\n', numMaterials);
for i = 1 : numMaterials
    % Uso colores aleatorios:
    fprintf(fid,'%d\t%d\t%s\ttrue %.1f %.1f %.1f\n', indexMaterials(i), indexMaterials(i), nameMaterials{i},...
        rand(1),rand(1),rand(1));
end
fclose(fid);
