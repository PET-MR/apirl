%% GENERACIÓN DE SINOGRAMAS
close all 
clear all
addpath('/workspaces/Martin/TGS/AnalisisResEspacial/trunk/matlab');
outputPath = '/workspaces/Martin/Doctorado/Tesis/Tesis Martín Belzunce/docusTesis/Figuras/Capitulo3';
p = phantom;
p = createDaponteImage([128 128]);
h = figure;
imshow(p);
% Tengo que visualizar el fantoma, luego visualizar el espectro y
% finalmente graficar dos filas de cada uno de ellos.
frame = getframe(gca);
outputFilename = sprintf('%s/fantoma',outputPath);
imwrite(frame.cdata, [outputFilename '.png']);
saveas(gca, outputFilename, 'epsc');
saveas(gca, outputFilename, 'tif');

% Sinogramas
sino = radon(p)';
h = figure;
imshow(sino./max(max(sino)))  
outputFilename = sprintf('%s/sino',outputPath);
imwrite(frame.cdata, [outputFilename '.png']);
saveas(gca, outputFilename, 'epsc');
saveas(gca, outputFilename, 'tif');

% Transformada de Fourier de la imagen:
imageFourier = fft2(p);
imageFourierCentrada = abs(fftshift(imageFourier));
h = figure;
imshow(imageFourierCentrada./max(max(imageFourierCentrada)))  
outputFilename = sprintf('%s/fftPhantom',outputPath);
imwrite(frame.cdata, [outputFilename '.png']);
saveas(gca, outputFilename, 'epsc');
saveas(gca, outputFilename, 'tif');
% Miro una proyección dada:
indiceProyeccion = 40;
imageFourierCentradaRotada = imrotate(imageFourierCentrada,indiceProyeccion);
h = figure;
imshow(imageFourierCentradaRotada./max(max(imageFourierCentradaRotada)))  
fftProyeccion1 = imageFourierCentradaRotada(91,:);
h = figure;
plot(fftProyeccion1);

% Grafico la proyección a 40º:
h = figure;
plot(sino(40,:));

% Transformada de Fourier de la mima:
fftProyeccion = fft(sino(40,:));
fftProyeccion = fftshift(fftProyeccion);
h = figure;
plot(abs(fftProyeccion));