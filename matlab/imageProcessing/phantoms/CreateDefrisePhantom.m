%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 30/10/2015
%  *********************************************************************

% Function that creates a defrise phantom.
function [phantom refImage] = CreateDefrisePhantom(sizeImage_pixels, pixelSize_mm)
%% PARAMETERS
sizeImage_mm = sizeImage_pixels.*pixelSize_mm;
phantom = zeros(sizeImage_pixels);
% Number of Discs:
numDiscs = 10;
widthDiscs_pixels = 5;
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

refImage = imref3d(sizeImage_pixels, pixelSize_mm(2), pixelSize_mm(1), pixelSize_mm(3));