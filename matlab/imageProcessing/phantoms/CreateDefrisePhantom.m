%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 30/10/2015
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
%% PARAMETERS
pixelSize_mm = [2.08625 2.08625 2.03125];
sizeImage_pixels = [286 286 127];
sizeImage_mm = sizeImage_pixels.*pixelSize_mm;
phantom = zeros(sizeImage_pixels);
% Number of Discs:
numDiscs = 10;
widthDiscs_pixels = 6;
widthDiscs_mm = pixelSize_mm(3).*widthDiscs_pixels;
sepDiscs_pixels = 4;
sepDiscs_mm = pixelSize_mm(3).*sepDiscs_pixels;
radioDiscs_mm = 200;
%% COORD SYSTEM
% El x vanza como los índices, osea a la izquierda es menor, a la derecha
% mayor.
coordX = -((sizeImage_mm(2)/2)-pixelSize_mm/2):pixelSize_mm:((sizeImage_mm(2)/2)-pixelSize_mm/2);
% El y y el z van al revés que los índices, o sea el valor geométrico va a
% contramano con los índices de las matrices.
coordY = -((sizeImage_mm(1)/2)-pixelSize_mm/2):pixelSize_mm:((sizeImage_mm(1)/2)-pixelSize_mm/2);
coordZ = -((sizeImage_mm(3)/2)-pixelSize_mm/2):pixelSize_mm:((sizeImage_mm(3)/2)-pixelSize_mm/2);
[X,Y,Z] = meshgrid(coordX, coordY, coordZ);
origin = [0 0 0];
%% CREATE IMAGE
zFirstDisc_mm = -(numDiscs/2)*(widthDiscs_mm + sepDiscs_mm)+sepDiscs_mm/2;
for i = 1 : numDiscs
    indexCylinder = (sqrt((X-0).^2+(Y-0).^2) < radioDiscs_mm)  & (Z>(zFirstDisc_mm+(i-1)*(widthDiscs_mm + sepDiscs_mm))) & (Z<(zFirstDisc_mm+widthDiscs_mm+(i-1)*(widthDiscs_mm + sepDiscs_mm)));
    phantom(indexCylinder) = 1;
end

%% VISUALIZACIÓN
% Veo la imagen 3D, mostrando los slices secuencialmente-
% Primero atenuación:
h = figure;
for i = 1 : size(phantom,3)
    imshow(phantom(:,:,i),[]);
    pause(0.2);
end
interfilewrite(phantom, 'phantom', pixelSize_mm);