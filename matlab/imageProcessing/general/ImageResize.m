%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 13/07/2015
%  *********************************************************************
%  This function resizes an image, scaling it in the three dimensiones.
%  Received as a parameter the image and a three elements vector with the
%  output size for the new image.
%
%  Example:
%   [resizedImage] = ImageResize(image, outputSize)
function [resizedImage] = ImageResize(image, outputSize)

% Grid:
[X,Y,Z] = meshgrid(1:size(image,2), 1:size(image,1), 1:size(image,3));

scalingFactor = outputSize./size(image);
% Output grid:
[X2, Y2, Z2] = meshgrid(1:1/scalingFactor(2):size(image,2), 1:1/scalingFactor(1):size(image,1), 1:1/scalingFactor(3):size(image,3));

% Interpolate the image to the new coordinate system:
resizedImage = interp3(X,Y,Z,double(image),X2,Y2,Z2, 'linear',0);
resizedImage(isnan(resizedImage)) = 0;


