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
%% INICIALIZACIÓN IMÁGENES
sizePixel_mm = 2;
sizeImage_mm = [230 300 200];
sizeImage_pixels = sizeImage_mm ./ sizePixel_mm;
imageAtenuation = uint16(zeros(sizeImage_pixels));
imageActivity = uint16(zeros(sizeImage_pixels));

% El x vanza como los índices, osea a la izquierda es menor, a la derecha
% mayor.
coordX = -((sizeImage_mm(2)/2)-sizePixel_mm/2):sizePixel_mm:((sizeImage_mm(2)/2)-sizePixel_mm/2);
% El y y el z van al revés que los índices, o sea el valor geométrico va a
% contramano con los índices de las matrices.
coordY = ((sizeImage_mm(1)/2)-sizePixel_mm/2):-sizePixel_mm:-((sizeImage_mm(1)/2)-sizePixel_mm/2);
coordZ = ((sizeImage_mm(3)/2)-sizePixel_mm/2):-sizePixel_mm:-((sizeImage_mm(3)/2)-sizePixel_mm/2);
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
numMaterials = 3;
indexMaterials = [0 1 2];
nameMaterials = {'Air', 'Water', 'Lung'};
% Genero el interior de agua que está formado por medio cilindro centrado
% en (0;-35).
indexHalfCylinder = (sqrt((X-0).^2+(Y+35).^2) < 147) & (Y>-35) & (Z>-90)&(Z<90);
% Genero los cuartos cilindros de las esquinas centrados en (+70,-35) y
% (-70,+35):
indexQuarterCylinder1 = (sqrt((X-70).^2+(Y+35).^2) < 77) & (X>70) & (Y<0) & (Z>-90)&(Z<90);
indexQuarterCylinder2 = (sqrt((X+70).^2+(Y+35).^2) < 77) & (X<-70) & (Y<0) & (Z>-90)&(Z<90);
% Por último el cuadrado inferior de 140 x 77 centrado en (0;38.5):
indexBox = (X<70) & (X>-70) & (Y>-(35+77)) & (Y<-35) & (Z>-90)&(Z<90);
imageAtenuation(indexHalfCylinder|indexQuarterCylinder1|indexQuarterCylinder2|indexBox) = indexWater;

% El inserto que emula los pulmones, atravieza los 180mm de largo del fantom
% y tiene 50mm de diametro exterior, hacemos 23 mm de radio y 2 mm de
% plástico exterior. El mismo está centrado en (0,0):
indexInsert = (sqrt((X-0).^2+(Y-0).^2) < 23) & (Z>-90)&(Z<90);
imageAtenuation(indexInsert) = indexLung;

% Por último todos los bordes de vidrio:

%% IMAGEN DE ACTIVIDAD
% Para pasar los voxeles a actividad hay dos opciones: un conversor lineal
% o otro por rangos de actividad. Utilizaremos el que es por rango de
% actividad.
% Según la norma nema las conectracciones de actividad deben ser las
% siguientes:
backgroundActivityConcentration_kBq_cc = 2.0;
hotSpheresActivityConcentration_kBq_cc = backgroundActivityConcentration_kBq_cc*4;
% Considero que el linear range se setea con 1 = 1 Bq
disp('IMPORTANTE: Setear el linear range del .mac con escala de 1 Bq.');
% O sea que en la imagen de actividad, el valor 1 representa 1 Bq. Ahora
% calculo los valores de actividad de los píxeles:
backgroundActivityConcentration_Bq = backgroundActivityConcentration_kBq_cc * (sizePixel_mm/10)^3 *1000;
hotSpheresActivityConcentration_Bq = backgroundActivityConcentration_Bq * 4;
% Seteo la actividad de fondo:
imageActivity(indexHalfCylinder|indexQuarterCylinder1|indexQuarterCylinder2|indexBox) = backgroundActivityConcentration_Bq;
% Tengo 6 esferas de 10, 13, 17, 22, 28 y 37mm de diámetro. Siendo las 4
% primeras esferas calientes, y las dos últimas esferas frías. Todas las
% esferas tienen el mismo centro en el eje z, o sea que están centradas en
% el mismo plano transversal. Están ubicados a 68 mm del tope del fantoma.
% Teniendo en cuenta que el largo itnerior del fantoma es de 180 mm. O sea
% va entre -90 y +90.
radioEsferas_mm = [10 13 17 22 28 37] ./2 ;
centroZ_esferas = 90 - 68;
% Dentro del plano transversal las esferas están ubicadas con sus centros
% equiespaciados angularmente, y a una distancia de 57,2mm del centro del
% fantoma. Estando además la esfera de 17 sobre el eje horizontal, del lado
% izquierdo. A partir de esto obtengo los centros en x e y de cada esfera:
anguloEsferas_deg = 60 : 60 : 360;
distanciaCentroEsferas = 57.2;
centroX_esferas = distanciaCentroEsferas * cosd(anguloEsferas_deg);
centroY_esferas = distanciaCentroEsferas * sind(anguloEsferas_deg);
% Ahora con los centros, los radios y la ecuación de la esferas las formo.
% Recordando que algunas son calientes y otras frías.
actividadEsferas = [repmat(hotSpheresActivityConcentration_Bq,1,4) 0 0];
for i = 1 : numel(radioEsferas_mm)
    indexEsfera = (sqrt((X-centroX_esferas(i)).^2+(Y-centroY_esferas(i)).^2 ...
    +(Z-centroZ_esferas).^2) < radioEsferas_mm(i));
    imageActivity(indexEsfera) = actividadEsferas(i);
end
% Por último el inserto de baja densidad no tiene nada, así que le asigno
% una actividad de cero:
imageActivity(indexInsert) = 0;

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
interfilewrite(imageActivity, 'IQnemaVoxelizedActivity', sizePixel);
interfilewrite(imageAtenuation, 'IQnemaVoxelizedAttenuation', sizePixel);
%% ESCRITURA DE IMÁGENES PARA PRUEBAS
% Por ahora los píxeles son del mismo tamaño en los 3 ejes:
sizePixel = [sizePixel_mm sizePixel_mm sizePixel_mm];
% Escribo los archivos de salida:
interfilewrite(single(imageActivity), 'IQnemaVoxelizedActivity_float', sizePixel);
interfilewrite(single(imageAtenuation), 'IQnemaVoxelizedAttenuation_float', sizePixel);
%% ESRCITURA DE ARCHIVO DE RANGOS DE ATENUACIÓN
% Primero genero el archivo de encabezado.
fid = fopen('RangeAttenuation.dat', 'w');
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

%% VISUALIZACIÓN PARA INFORME O TESIS
slice = 40;
imageAtenuation = single(imageAtenuation);
imageActivity = single(imageActivity);
imagenPresentacion = [imageAtenuation(:,:,slice)./max(max(imageAtenuation(:,:,slice))) imageActivity(:,:,slice)./max(max(imageActivity(:,:,slice)))];
h = figure;
imshow(imagenPresentacion);
line([size(imageAtenuation(:,:,slice),2)+1 size(imageAtenuation(:,:,slice),2)+1], [0 2*size(imageAtenuation(:,:,slice),1)],'Color','w','LineWidth', 2);
set(gcf, 'Position', [50 50 1200 800]);
outputGraphsPath = '/workspaces/Martin/Doctorado/Tesis/Tesis Martín Belzunce/docusTesis/Figuras/Capitulo4/';
graphicFilename = sprintf('FantomaNemaAtenActiv');
set(gcf,'PaperPositionMode','auto');    % Para que lo guarde en el tamaño modificado.
set(gcf,'InvertHardcopy', 'off');
saveas(gcf, [outputGraphsPath graphicFilename], 'fig');
frame = getframe(gca);
imwrite(frame.cdata, [outputGraphsPath graphicFilename '.png']);
saveas(gca, [outputGraphsPath graphicFilename], 'epsc');
