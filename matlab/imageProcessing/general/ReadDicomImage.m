%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 01/07/2015
%  *********************************************************************
%  This function reads a set of dicom images stored in a path. Receives the
%  path as a parameter and additionaly a baseFilename to filter the files
%  inside the folder. This might be useful if there are several data sets
%  in the same folder. If the folder contains only one data set it can be
%  an empty string.
%  It return a 3d image and the (x,y,z) coordinates of each pixel stored in
%  three diferent matrices. It return the coordinate map instead of origin
%  and pixel size, because the image can be rotated in the coordinate
%  system.
%
%  Example:
%   [image, affineMatrix, xMap_mm, yMap_mm, zMap_mm] = ReadDicomImage('/data/BRAIN_PETMR/T1_fl2D_TRA/', '')
%   [image, affineMatrix, xMap_mm, yMap_mm, zMap_mm] = ReadDicomImage('/data/BRAIN_PETMR/T1_fl2D_TRA/', 'BRAIN_PETMR.MR.CDT_PLANT_SINGLE_BED_PETMR.0030')
function [image, affineMatrix, xMap_mm, yMap_mm, zMap_mm, dicomInfo] = ReadDicomImage(path, baseFilename)

% Read info from the images:
files = dir([path baseFilename '*']);
% Remove . and .. :
filesToRemove = {'.','..'};
cellStruct = struct2cell(files);
index = zeros(1,size(cellStruct,2));
for i = 1 : numel(filesToRemove)
    index = index | strcmp(cellStruct(1,:), filesToRemove{i});
end
cellStruct(index) = [];
files(index) = [];

% Map of pixel indexes:
dicomInfo = dicominfo([path files(1).name]);
if isfield(dicomInfo, 'ImagesInAcquisition')
    numSlices = dicomInfo.ImagesInAcquisition;
else
    numSlices = numel(files); % It would be better to use dicomInfo.ImagesInAcquisition.
end
% Poistion of the top-left pixel of the first slice:
posTopLeftPixel_1 = dicomInfo.ImagePositionPatient;
% Pixel spacing:
pixelSpacing_mm = [dicomInfo.PixelSpacing(1) dicomInfo.PixelSpacing(2) dicomInfo.SliceThickness];
% We suppose that each slice has the same ImageOrientationPatient:
dircosX = dicomInfo.ImageOrientationPatient(1:3);
dircosY = dicomInfo.ImageOrientationPatient(4:6);
% Check orthogonality:
dotProd = dot(dircosX, dircosY);
if(dotProd > 1e-5)
    warning('The axes are not orthogonal in the dicom image.');
end

[indexCol, indexRow] = meshgrid(double(0:0:dicomInfo.Width-1),double(dicomInfo.Height-1));
for i = 1 : numSlices
    dicomInfo = dicominfo([path files(i).name]);
    sliceCoordinates(i) = dicomInfo.SliceLocation;
    image(:,:,i) = dicomread([path files(i).name]);
    
    % This is the full code to get the coordinates of each pixel. But it
    % won't work without meshgrid because it needs perfectly uniform
    % sampling. Anyway dicomInfo.ImagePositionPatient(2) and
    % dicomInfo.ImagePositionPatient(4) is aprox 0.
    xMap_mm(:,:,i) = dicomInfo.ImagePositionPatient(1) + dicomInfo.ImageOrientationPatient(1) * dicomInfo.PixelSpacing(1) .* indexCol + dicomInfo.ImageOrientationPatient(4) * dicomInfo.PixelSpacing(2) .* indexRow; 
    yMap_mm(:,:,i) = dicomInfo.ImagePositionPatient(2) + dicomInfo.ImageOrientationPatient(2) * dicomInfo.PixelSpacing(1) .* indexCol + dicomInfo.ImageOrientationPatient(5) * dicomInfo.PixelSpacing(2) .* indexRow; 
    zMap_mm(:,:,i) = dicomInfo.ImagePositionPatient(3) + dicomInfo.ImageOrientationPatient(3) * dicomInfo.PixelSpacing(1) .* indexCol + dicomInfo.ImageOrientationPatient(6) * dicomInfo.PixelSpacing(2) .* indexRow; 
end
% Poistion of the top-left pixel of the last slice:
posTopLeftPixel_N = dicomInfo.ImagePositionPatient;
% Dir z:
dirZ = (posTopLeftPixel_N - posTopLeftPixel_1) / (numSlices-1);
% Affine transformation matrix to got from image space to patient space:
affineMatrix = [dircosX.*pixelSpacing_mm(1) dircosY.*pixelSpacing_mm(2) dirZ posTopLeftPixel_1; 0 0 0 1];
%affineMatrix = [dircosX dircosY dirZ./(sliceCoordinates(2)-sliceCoordinates(1)) [0; 0; 0]; 0 0 0 1];