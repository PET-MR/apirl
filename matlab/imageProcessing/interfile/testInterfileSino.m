% Script de prueba de interfileWriteSinogram

%% Un sinograma 2D:
p = phantom;
sino2D = single(radon(p)');
interfileWriteSino(sino2D, '/home/martin/testSino2D');

%% Múltiples sinogramas 2D
% Utilizo la definición del fantoma de IQ en createIQvoxelizedPhantom:

sizePixel_mm = 2;
sizeImage_mm = [230 300 200];
sizeImage_pixels = sizeImage_mm ./ sizePixel_mm;
imageAtenuation = uint8(zeros(sizeImage_pixels));
imageActivity = single(zeros(sizeImage_pixels));

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

% Genero las proyecciones:
for i = 1 : size(imageAtenuation,3)
    sino2D_multislice(:,:,i) = radon(imageAtenuation(:,:,i));
end
interfileWriteSino(sino2D_multislice, '/home/martin/testSino2D_multislice');

%% Ahora sinograma 3d
% Leo un sino 3D binario y lo guardo como interfile para ver como queda:
pathSino3D = '/datos/Sinogramas/Sinograma GE/sino3DNema.s';
fid = fopen(pathSino3D, 'r');
sino3D = fread(fid, inf, 'single');
fclose(fid);
sino3D = reshape(sino3D, 329,280, 553);
numSinos = [47, 43, 43, 39, 39, 35, 35, 31, 31, 27, 27, 23, 23, 19, 19, 15, 15, 11, 11, 7, 7, 3, 3];
minRingDiff = [-1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15, 16, -17, 18, -19, 20, -21, 22, -23];
maxRingDiff = [1, 3, -2, 5, -4, 7, -6, 9, -8, 11, -10, 13, -12, 15, -14, 17, -16, 19, -18, 21, -20, 23, -22];

interfileWriteSino(sino3D, '/home/martin/testSino3D');