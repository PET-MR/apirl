%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 10/02/2015
%  *********************************************************************
%  function [image, pixelSize_mm, origin_mm, bedPosition_mm, info] = interfileReadSiemensImage(headerFilename)
%
%  This function reads the an image in the siemens interfile format. It
%  returns the image,  a 3-D spatial referencing object (imref3d) that
%  contains the spatial reference of each dimension in the patient
%  coordinate, the bed position and an struct with the info in the header. 

function [image, refImage, bedPosition_mm, info]  = interfileReadSiemensImage(headerFilename)
 
% Check what OS I am running on:
if(strcmp(computer(), 'GLNXA64'))
    os = 'linux';
    pathBar = '/';
elseif(strcmp(computer(), 'PCWIN') || strcmp(computer(), 'PCWIN64'))
    os = 'windows';
    pathBar = '\';
else
    disp('OS not compatible');
    return;
end

% Read header:
[info] = getInfoFromSiemensIntf(headerFilename);

% Get the pixel size:
pixelSize_mm = [info.ScaleFactorMmPixel2 info.ScaleFactorMmPixel1 info.ScaleFactorMmPixel3];    % ScaleFactorMmPixel1 is for the x axes: columns.
% Get the size of the image:
imageSize_pixels = [info.MatrixSize2 info.MatrixSize1 info.MatrixSize3];
% Origin:
fieldsImage_str = strsplit(info.ImageInfo1(2:end-1), ',');
fieldsImage = str2double(fieldsImage_str);
origin_mm = [fieldsImage(4) fieldsImage(3) fieldsImage(2)];
% Initial bed position:
bedPosition_mm = info.StartHorizontalBedPositionMm;

           
% read image:
%image = zeros(imageSize_pixels);
fid = fopen(info.NameOfDataFile, 'r');
if(fid == -1)
    % Try adding the path:
    [pathstr,name,ext] = fileparts(headerFilename);
    fid = fopen([pathstr pathBar info.NameOfDataFile], 'r');
    if(fid == -1)
        error(sprintf('The binary data file: %s couldn''t be opened.', info.NameOfDataFile));
    end
end
if ~strcmp(info.NumberFormat, 'float')
    error('Image Number Format is not float.');
end
image = fread(fid, imageSize_pixels(1)*imageSize_pixels(2)*imageSize_pixels(3), 'single');
fclose(fid);
image = reshape(image, [imageSize_pixels(1) imageSize_pixels(2) imageSize_pixels(3)]);
image = permute(image, [2 1 3]);

% Creathe the imref object:
refImage = imref3d(size(image), [origin_mm(2)-pixelSize_mm(2) origin_mm(2)-pixelSize_mm(2)+pixelSize_mm(2)*size(image,2)], ...
    [origin_mm(1)-pixelSize_mm(1) origin_mm(1)-pixelSize_mm(1)+pixelSize_mm(1)*size(image,1)], [origin_mm(3)-pixelSize_mm(3) origin_mm(3)-pixelSize_mm(3)+pixelSize_mm(3)*size(image,3)]); % x coordinate (cols), y coordinates(rows), z(coordinates(3rd).

% If I want the same reference than the one used by siemens coordiante
% system I overwrite it:
% The dicom image for pet images contains this limits for a 344x344x127
% image:
% XWorldLimits= [-356.8832 361.1168];
% YWorldLimits= [-359.8493 358.1507];
% ZWorldLimits= [-133.2885 124.7115];
XWorldLimits= [origin_mm(2)-pixelSize_mm(2)/2 origin_mm(2)-pixelSize_mm(2)/2+pixelSize_mm(2)*size(image,2)];
YWorldLimits= [origin_mm(1)-pixelSize_mm(1)/2 origin_mm(1)-pixelSize_mm(1)/2+pixelSize_mm(1)*size(image,1)];
XWorldLimits= [-359 359];
YWorldLimits= [-359-pixelSize_mm(1) 359-pixelSize_mm(1)];

ZWorldLimits= [-258/2+pixelSize_mm(3) 258/2+pixelSize_mm(3)];
image = flip(image,3);
refImage = imref3d(size(image), XWorldLimits, YWorldLimits, ZWorldLimits);

