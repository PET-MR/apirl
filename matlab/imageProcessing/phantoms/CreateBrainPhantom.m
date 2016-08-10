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
    pixelSize_mm = [0.5 0.5 0.5];
    % The size in pixels:
    imageSize_pixels = [362 434 362];
end
%% READ BINARY IMAGE
% Read image:
fid = fopen(binaryFilename, 'r');
if fid == -1
    ferror(fid);
end
phantom = fread(fid, imageSizePhantom_pixels(1)*imageSizePhantom_pixels(2)*imageSizePhantom_pixels(3), 'uint16');
phantom = reshape(phantom, imageSizePhantom_pixels);
% Then interchange rows and cols, x and y: 
phantom = permute(phantom, [2 1 3]);
fclose(fid);
imageSizePhantom_pixels = size(phantom);
%% CONVERT TO DEFAULT SIZE
% Size of the image to cover the full fov:
sizeImage_mm = pixelSize_mm .* imageSize_pixels;
if pixelSize_mm ~= pixelSizePhantom_mm | imageSize_pixels ~= imageSizePhantom_pixels
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
    % I am having problems with the first slices, patch them:
    phantom_rescaled(:,:,1:5) = 0;
else
    phantom_rescaled = phantom;
end
%% 
whiteMatterAct = 32;
grayMatterAct = 96;
skinAct = 16;
indicesWhiteMatter = phantom_rescaled == 48;
indicesGrayMatter = phantom_rescaled == 32;
indicesSkin = phantom_rescaled == 96;
indicesSkull = phantom_rescaled == 112;
indicesMarrow = phantom_rescaled == 177;
indicesDura = phantom_rescaled == 161;
indicesBone = indicesSkull | indicesMarrow | indicesDura;
%% CREATE ATTENUATION MAP
% brainweb material id:
% 0=Background, 1=CSF, 2=Gray Matter, 3=White Matter, 4=Fat, 5=Muscle, 6=Muscle/Skin, 7=Skull, 8=vessels, 9=around fat, 10 =dura matter, 11=bone marrow
attenuationMap = phantom_rescaled;
% Set the attenuation of bones:
mu_bone_1_cm = 0.13;
mu_bone_1_cm = 0.13;
mu_tissue_1_cm = 0.0975;
attenuationMap(phantom_rescaled >0) = mu_tissue_1_cm;
attenuationMap(indicesBone) = mu_bone_1_cm;

%% TRANSFORM THE ATANOMY INTO PET SIGNALS
phantom_rescaled(indicesWhiteMatter) = whiteMatterAct;
phantom_rescaled(indicesGrayMatter) = grayMatterAct;
phantom_rescaled(indicesSkin) = skinAct;
phantom_rescaled(~indicesWhiteMatter & ~indicesGrayMatter & ~indicesSkin) = 0;


refImage = imref3d(imageSize_pixels, pixelSize_mm(2), pixelSize_mm(1), pixelSize_mm(3));