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
%   [image, xMap_mm, yMap_mm, zMap_mm] = ReadDicomImage('/data/BRAIN_PETMR/T1_fl2D_TRA/', '')
%   [image, xMap_mm, yMap_mm, zMap_mm] = ReadDicomImage('/data/BRAIN_PETMR/T1_fl2D_TRA/', 'BRAIN_PETMR.MR.CDT_PLANT_SINGLE_BED_PETMR.0030')
function [image, xMap_mm, yMap_mm, zMap_mm] = ReadDicomImage(path, baseFilename)

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
numSlices = numel(files); % It would be better to use dicomInfo.ImagesInAcquisition.

% Map of pixel indexes:
dicomInfo = dicominfo([path files(1).name]);
[indexCol, indexRow] = meshgrid(double(0:dicomInfo.Height-1),double(0:dicomInfo.Width-1));
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
