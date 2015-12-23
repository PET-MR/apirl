%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 22/12/2015
%  *********************************************************************

% Function that read the binary file from the brain web phantom.
% It uses the high resolution dimension:
% imageSizePhantom_pixels = [362 434 362];
% pixelSizePhantom_mm = [0.5 0.5 0.5];
% It returns the image resized to the standard PET mMR image size, and
% an attenuation map.

function [phantom_rescaled, attenuationMap, refImage] = CreateBrainPhantom(binaryFilename, imageSize_pixels, pixelSize_mm)
%% PARAMETERS
imageSizePhantom_pixels = [362 434 362];
pixelSizePhantom_mm = [0.5 0.5 0.5];
if nargin == 1
    % set the default pixel value and matrix size:
    % Size of the pixels:
    pixelSize_mm = [2.08625 2.08625 2.03125];
    % The size in pixels:
    imageSize_pixels = [286 286 127];
end
%% READ BINARY IMAGE
% Read image:
fid = fopen(binaryFilename, 'r');
if fid == -1
    ferror(fid);
end
phantom = fread(fid, imageSizePhantom_pixels(1)*imageSizePhantom_pixels(2)*imageSizePhantom_pixels(3), 'uint8');
phantom = reshape(phantom, imageSizePhantom_pixels);
% Then interchange rows and cols, x and y: 
phantom = permute(phantom, [2 1 3]);
fclose(fid);
imageSizePhantom_pixels = size(phantom);
%% CONVERT TO DEFAULT SIZE
% Size of the image to cover the full fov:
sizeImage_mm = pixelSize_mm .* imageSize_pixels;
% Resample to the emission images space (centered axially):
coordXphantom = (-pixelSizePhantom_mm(2)*imageSizePhantom_pixels(2)/2+pixelSizePhantom_mm(2)/2) : pixelSizePhantom_mm(2) : (pixelSizePhantom_mm(2)*imageSizePhantom_pixels(2)/2);
coordYphantom = (-pixelSizePhantom_mm(1)*imageSizePhantom_pixels(1)/2+pixelSizePhantom_mm(1)/2) : pixelSizePhantom_mm(1) : (pixelSizePhantom_mm(1)*imageSizePhantom_pixels(1)/2);
coordZphantom = (-pixelSizePhantom_mm(3)*imageSizePhantom_pixels(3)/2+pixelSizePhantom_mm(3)/2) : pixelSizePhantom_mm(3) : (pixelSizePhantom_mm(3)*imageSizePhantom_pixels(3)/2);
[Xphantom, Yphantom, Zphantom] = meshgrid(coordXphantom, coordYphantom, coordZphantom);
% Idem for the dixon attenuation map:
coordXpet = (-pixelSize_mm(2)*imageSize_pixels(2)/2 + pixelSize_mm(2)/2) : pixelSize_mm(2) : (pixelSize_mm(2)*imageSize_pixels(2)/2);
coordYpet = (-pixelSize_mm(1)*imageSize_pixels(1)/2 + pixelSize_mm(1)/2) : pixelSize_mm(1) : (pixelSize_mm(1)*imageSize_pixels(1)/2);
coordZpet = (-pixelSize_mm(3)*imageSize_pixels(3)/2 + pixelSize_mm(3)/2) : pixelSize_mm(3) : (pixelSize_mm(3)*imageSize_pixels(3)/2);
[Xpet, Ypet, Zpet] = meshgrid(coordXpet,coordYpet, coordZpet);
% Interpolate the ct image to the mr coordinates:
phantom_rescaled = interp3(Xphantom,Yphantom,Zphantom,phantom,Xpet,Ypet,Zpet); 
phantom_rescaled(isnan(phantom_rescaled)) = 0;
%% CREATE ATTENUATION MAP
% brainweb material id:
% 0=Background, 1=CSF, 2=Gray Matter, 3=White Matter, 4=Fat, 5=Muscle, 6=Muscle/Skin, 7=Skull, 8=vessels, 9=around fat, 10 =dura matter, 11=bone marrow
attenuationMap = phantom_rescaled;
% Set the attenuation of bones:
mu_bone_1_cm = 0.1732;
mu_tissue_1_cm = 0.1007;
attenuationMap(phantom_rescaled >0) = mu_tissue_1_cm;

refImage = imref3d(imageSize_pixels, pixelSize_mm(2), pixelSize_mm(1), pixelSize_mm(3));