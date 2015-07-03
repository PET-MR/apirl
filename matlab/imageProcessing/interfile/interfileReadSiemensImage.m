%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 10/02/2015
%  *********************************************************************
%  function [image, pixelSize_mm, origin_mm, bedPosition_mm, info] = interfileReadSiemensImage(headerFilename)
%
%  This function reads the an image in the siemens interfile format. It
%  returns the image, the pixel size (pixelSize_mm), the coordinates of the
%  first pixel (origin_mm) in the top left of the first slice and the info
%  of the header. 

function [image, pixelSize_mm, origin_mm, bedPosition_mm, info]  = interfileReadSiemensImage(headerFilename)
 
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
pixelSize_mm = [info.ScaleFactorMmPixel1 info.ScaleFactorMmPixel2 info.ScaleFactorMmPixel3];
% Get the size of the image:
imageSize_pixels = [info.MatrixSize1 info.MatrixSize2 info.MatrixSize3];
% Origin:
fieldsImage_str = strsplit(info.ImageInfo1(2:end-1), ',');
fieldsImage = str2double(fieldsImage_str);
origin_mm = [fieldsImage(3) fieldsImage(4) fieldsImage(2)];
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




