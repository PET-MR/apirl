%**************************************************************************
%
%  Curso An�lisis de Im�genes M�dicas
%  TP FINAL
%
%  Autor: Mart�n Belzunce
%
%  Fecha: 03/11/2010
%
%  Descripcion:
%  Script que lee una imagen y le aplica los algoritmos de Difusi�n tanto
%  Homog�nea e Inhomg�nea llamando a las funciones
%  SuavizadoDifusonHomogenea y SuavizadoDifusionInhomogenea.
%
%**************************************************************************
%% INICALIZACI�N DE VARIABLES
clear all 
close all
pathImage = '../../aneurysm.raw';

% Dimensiones de la Imagen:
sizeI = [120 120];

%% LECTURA DE LA IMAGEN
% Lectura de la imagen en formato crudo:
fid = fopen (pathImage);
if(fid==-1)
    disp(['No se encontr� la imagen en' pathImage]);
    return;
end
vector = fread (fid,'ushort');
fclose(fid);
% Tengo un vector con todos los datos que lo debo convertir a una matriz:
imageData = uint16(vec2mat(vector, sizeI(2)));

%% VISUALIZACI�N DE LA IMAGEN
h = figure;
set(h, 'Position', [100 0 800 800]);
subplot(2,1,1);
title('Imagen Original');
imshow(imageData);
subplot(2,1,2);
imhist(imageData);
title('Histograma de la Imagen Original');

% Como la imagen aprovecha solo la mitad superior del rango din�mico,
% escalo los valores linealmente para abarcar los 16 bits de intensidades:
imageData = (imageData - 2^15) * 2;
h = figure;
set(h, 'Position', [100 0 800 800]);
subplot(2,1,1);
title('Imagen Ajustada');
imshow(imageData);
subplot(2,1,2);
imhist(imageData);
title('Histograma de la Imagen Ajustada');

%% DIFUSI�N HOMOG�NEA
% Para la difusi�n homogenea le debemos pasar una constante de difusi�n, en
% este caso vamos a usar 1/4:
D = 1;
[outputImage, k] = SuavizadoDifusionHomogenea(imageData, D, 'ErrorRelativo', 1e-5, 'ForwardEuler', 'Orden2');
h = figure;
imshow(outputImage);
title(sprintf('Imagen Suavizada por Difusi�n Homog�nea orden 2 luego de %d Iteraciones', k)); 
set(h, 'Position', [100 0 800 800]);

[outputImage, k] = SuavizadoDifusionHomogenea(imageData, D, 'Iteraciones', 50, 'ForwardEuler', 'Orden2');
h = figure;
imshow(outputImage);
title(sprintf('Imagen Suavizada por Difusi�n Homog�nea orden 2 luego de %d Iteraciones', k)); 
set(h, 'Position', [100 0 800 800]);

[outputImage, k] = SuavizadoDifusionHomogenea(imageData, D, 'Iteraciones', 500, 'ForwardEuler', 'Orden2');
h = figure;
imshow(outputImage);
title(sprintf('Imagen Suavizada por Difusi�n Homog�nea orden 2 luego de %d Iteraciones', k)); 
set(h, 'Position', [100 0 800 800]);

[outputImage, k] = SuavizadoDifusionHomogenea(imageData, D, 'Iteraciones', 5000, 'ForwardEuler', 'Orden2');
h = figure;
imshow(outputImage);
title(sprintf('Imagen Suavizada por Difusi�n Homog�nea orden 2 luego de %d Iteraciones', k)); 
set(h, 'Position', [100 0 800 800]);

[outputImage, k] = SuavizadoDifusionHomogenea(imageData, D, 'ErrorRelativo', 1e-5, 'ForwardEuler', 'Orden4');
h = figure;
imshow(outputImage);
title(sprintf('Imagen Suavizada por Difusi�n Homog�nea orden 4 luego de %d Iteraciones', k)); 
set(h, 'Position', [100 0 800 800]);

[outputImage, k] = SuavizadoDifusionHomogenea(imageData, D, 'Iteraciones', 50, 'ForwardEuler', 'Orden4');
h = figure;
imshow(outputImage);
title(sprintf('Imagen Suavizada por Difusi�n Homog�nea orden 4 luego de %d Iteraciones', k)); 
set(h, 'Position', [100 0 800 800]);

[outputImage, k] = SuavizadoDifusionHomogenea(imageData, D, 'Iteraciones', 500, 'ForwardEuler', 'Orden4');
h = figure;
imshow(outputImage);
title(sprintf('Imagen Suavizada por Difusi�n Homog�nea orden 4 luego de %d Iteraciones', k));
set(h, 'Position', [100 0 800 800]);

[outputImage, k] = SuavizadoDifusionHomogenea(imageData, D, 'Iteraciones', 5000, 'ForwardEuler', 'Orden4');
h = figure;
imshow(outputImage);
title(sprintf('Imagen Suavizada por Difusi�n Homog�nea orden 4 luego de %d Iteraciones', k));
set(h, 'Position', [100 0 800 800]);


% Comparo con un filtro de mediana:
h = fspecial('gaussian', [3 3], 0.5);
outputGausiana = imfilter(imageData, h);
h = figure;
imshow(outputGausiana);
title('Imagen filtrada con Kernel Gaussiano');
set(h, 'Position', [100 0 800 800]);

% Para realizar la comparaci�n voy calculando el error relativo entre la
% imagen con filtrado por difusi�n homog�nea contra la imagen filtrada con
% un kernel gaussiano. Grafico dicho error, y observo cuando ese error se
% hace m�nimo.
errores = [];
outputGausiana = double(outputGausiana);
for k = 1 : 1 : 100
    [outputImage, t] = SuavizadoDifusionHomogenea(imageData, D, 'Iteraciones', k, 'ForwardEuler', 'Orden2');
    outputImage = double(outputImage);
    pixelsCero = outputGausiana ~= 0;
    error = sqrt(sum(((outputImage(pixelsCero)-outputGausiana(pixelsCero))./outputGausiana(pixelsCero)).^2))./numel(outputGausiana(pixelsCero));
    errores = [ errores; k error ];
end
% Grafico los errores:
h = figure;
set(h, 'Position', [100 0 800 800]);
plot(errores(:,1), errores(:,2)*100)
title('Error entre Suavizado por Difusi�n Homog�nea vs Suavizado Gausseano');
ylabel('error(%)');
xlabel('Iteraciones');
% Obtengo el m�nimo de los errores:
[MinimoError index] = min(errores(:,2));
disp(sprintf('El suavizado por Difusi�n homog�nea equivalente al suavizado gaussiano se da en %d iteraciones', errores(index,1)));
% vuelvo a generar esa imagen(en errores(index,1) tengo la cantidad de iteraciones:
[outputImage, t] = SuavizadoDifusionHomogenea(imageData, D, 'Iteraciones', errores(index,1), 'ForwardEuler', 'Orden2');
h = figure;
subplot(1,2,1);
imshow(uint16(outputGausiana));
title('Imagen Suavizada con Filtro Gaussiano de 3x3 y Desv�o 0.5');
subplot(1,2,2);
imshow(outputImage);
title(sprintf('Imagen Suavizada por Difusi�n Homog�nea en %d iteraciones', errores(index,1)));
set(h, 'Position', [100 0 1100 800]);
%% DIFUSI�N INHOMOG�NEA
% Primero grafico la funciones que voy a utilizar como modelos para el
% suavizado con preservaci�n de bordes:
c = [1000 10000];
for j = 1 : numel(c)
    intValues = 1 : 2^16;
    coefValues = coefDifusion(intValues, 3.3, 4, c(j));
    h = figure;
    set(h, 'Position', [100 0 800 800]);
    plot(intValues, coefValues);
    title(sprintf('Funci�n de Coeficientes de Difusi�n para c = %d', c(j)));
end

% Primero busco un cierto nivel de tolerancia:
for j = 1 : numel(c)
    [outputImage, k] = SuavizadoDifusionInhomogenea(imageData, 3.3, 4, c(j),...
            'ErrorRelativo', 1e-6, 'ForwardEuler', 'Orden2');
    h = figure;
    imshow(outputImage);
    title(sprintf('Imagen Suavizada por Difusi�n Inhomog�nea con c=%d luego de %d Iteraciones', c(j), k));
    set(h, 'Position', [100 0 800 800]);
end

% Luego trabajo por cantidad de iteraciones.
% Ejecuto el suavizado por difusi�n inhomog�nea para distinta cantidad de
% iteraciones (con c=1000):
c = [1000 10000];
iteraciones = [50 500 5000];
for j = 1 : numel(c)
    for i = 1 : numel(iteraciones)
        [outputImage, k] = SuavizadoDifusionInhomogenea(imageData, 3.3, 4, c(j),...
            'Iteraciones', iteraciones(i), 'ForwardEuler', 'Orden2');
        h = figure;
        imshow(outputImage);
        title(sprintf('Imagen Suavizada por Difusi�n Inhomog�nea con c=%d luego de %d Iteraciones', c(j), k));
        set(h, 'Position', [100 0 800 800]);
    end
end
