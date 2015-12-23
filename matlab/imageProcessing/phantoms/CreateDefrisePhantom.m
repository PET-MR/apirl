%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 22/12/2015
%  *********************************************************************

% Function that read the binary file from the brain web phantom.
function [phantom refImage] = CreateDefrisePhantom(sizeImage_pixels, pixelSize_mm)
%% PARAMETERS
sizeImage_mm = sizeImage_pixels.*pixelSize_mm;
phantom = zeros(sizeImage_pixels);
% Number of Discs:
numDiscs = 10;
widthDiscs_pixels = 5;
widthDiscs_mm = pixelSize_mm(3).*widthDiscs_pixels;
sepDiscs_pixels = 5;
sepDiscs_mm = pixelSize_mm(3).*sepDiscs_pixels;
radioDiscs_mm = 150;
%% COORD SYSTEM
% El x vanza como los índices, osea a la izquierda es menor, a la derecha
% mayor.
coordX = -((sizeImage_mm(2)/2)-pixelSize_mm(2)/2):pixelSize_mm(2):((sizeImage_mm(2)/2)-pixelSize_mm(2)/2);
% El y y el z van al revés que los índices, o sea el valor geométrico va a
% contramano con los índices de las matrices.
coordY = -((sizeImage_mm(1)/2)-pixelSize_mm(1)/2):pixelSize_mm(1):((sizeImage_mm(1)/2)-pixelSize_mm(1)/2);
coordZ = -((sizeImage_mm(3)/2)-pixelSize_mm(3)/2):pixelSize_mm(3):((sizeImage_mm(3)/2)-pixelSize_mm(3)/2);
[X,Y,Z] = meshgrid(coordX, coordY, coordZ);

[X_2D,Y_2D] = meshgrid(coordX, coordY);
origin = [0 0 0];
%% BACKGROUND CYLINDER
zFirstDisc_pixels = round((sizeImage_pixels(3)-(widthDiscs_pixels+sepDiscs_pixels)*numDiscs)/2);
zFirstDisc_mm = Z(1,1,zFirstDisc_pixels);
indexBackground = (sqrt((X-0).^2+(Y-0).^2) < radioDiscs_mm*1.25)  & (Z>=zFirstDisc_mm-sepDiscs_mm) & (Z<(zFirstDisc_mm+numDiscs*(widthDiscs_mm + sepDiscs_mm)));
phantom(indexBackground) = 0.5;
%% CREATE IMAGE
indexCylinder2d = (sqrt((X_2D-0).^2+(Y_2D-0).^2) < radioDiscs_mm);
for i = 1 : numDiscs
    z_pixels = zFirstDisc_pixels+(i-1)*(widthDiscs_pixels+sepDiscs_pixels):zFirstDisc_pixels+(i-1)*(widthDiscs_pixels+sepDiscs_pixels)+widthDiscs_pixels-1;
    indexCylinder = logical(zeros(size(phantom)));
    for j = 1 : numel(z_pixels)
        indexCylinder(:,:,z_pixels(j))=indexCylinder2d;
    end
    phantom(indexCylinder) = 1;
end

refImage = imref3d(sizeImage_pixels, pixelSize_mm(2), pixelSize_mm(1), pixelSize_mm(3));