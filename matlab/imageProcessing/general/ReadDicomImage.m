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
%  It receives a third parameter applyAffineTransform, that is a flag that
%  enables to apply the affine transform of the affine matrix stored in the
%  dicom header.
%  It returns the image (transformed or not, depending on
%  applyAffineTransform), a 3-D spatial referencing object (imref3d) that
%  contains the spatial reference of each dimension in the patient
%  coordinate (if applyAffineTransform is not enabled it return the
%  reference to image coordinates), the affine matrix created from the
%  dicom header and the struct dicomInfo with the information of all the
%  field in the dicom header.
%
%  Example:
%   [image, imageRef3d, affineMatrix, dicomInfo] = ReadDicomImage('/data/BRAIN_PETMR/T1_fl2D_TRA/', '', 0)
%   [image, imageRef3d, affineMatrix, dicomInfo] = ReadDicomImage('/data/BRAIN_PETMR/T1_fl2D_TRA/', 'BRAIN_PETMR.MR.CDT_PLANT_SINGLE_BED_PETMR.0030',1)
function [image, imageRef3d, affineMatrix, dicomInfo] = ReadDicomImage(path, baseFilename, applyAffineTransform)

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
    slice = dicomread([path files(i).name]);
    if isfield(dicomInfo, 'InstanceNumber') % If we have the number of slice use it.
        sliceIndex = dicomInfo.InstanceNumber;
    else
        sliceIndex = i;
    end
    image(:,:,sliceIndex) = slice;
    sliceCoordinates(sliceIndex) = dicomInfo.SliceLocation;
    % If the last or first slice I need the coordinates:
    if sliceIndex == numSlices
        % Poistion of the top-left pixel of the last slice:
        posTopLeftPixel_N = dicomInfo.ImagePositionPatient;
    elseif sliceIndex == 1
        % Poistion of the top-left pixel of the first slice:
        posTopLeftPixel_1 = dicomInfo.ImagePositionPatient;
    end
    % This is the full code to get the coordinates of each pixel. Now we
    % apply it in the affine matrix instead of getting each coordinate.
%     xMap_mm(:,:,i) = dicomInfo.ImagePositionPatient(1) + dicomInfo.ImageOrientationPatient(1) * dicomInfo.PixelSpacing(1) .* indexCol + dicomInfo.ImageOrientationPatient(4) * dicomInfo.PixelSpacing(2) .* indexRow; 
%     yMap_mm(:,:,i) = dicomInfo.ImagePositionPatient(2) + dicomInfo.ImageOrientationPatient(2) * dicomInfo.PixelSpacing(1) .* indexCol + dicomInfo.ImageOrientationPatient(5) * dicomInfo.PixelSpacing(2) .* indexRow; 
%     zMap_mm(:,:,i) = dicomInfo.ImagePositionPatient(3) + dicomInfo.ImageOrientationPatient(3) * dicomInfo.PixelSpacing(1) .* indexCol + dicomInfo.ImageOrientationPatient(6) * dicomInfo.PixelSpacing(2) .* indexRow; 
end

sliceThickness = abs(sliceCoordinates(2)-sliceCoordinates(1));
if ~issorted(sliceCoordinates)
    warning('The slices are not sorted or there are some slices repeated.')
end

% % Code added because there are sequences with slices not ordered in the
% % z coordinate.
% if sliceThickness > 0
%     % Thr slice increases in z coordinate:
%     [sortedSliceCoordinates sortedIndexes] = sort(sliceCoordinates, 'ascend');
% else
%     [sortedSliceCoordinates sortedIndexes] = sort(sliceCoordinates, 'ascend');
% end
% % Get the Poistion of the top-left pixel of the last slice:
% dicomInfo = dicominfo([path files(i).name]);
% % Re-arrange the image in incresing z coordinate.
%     image = image(:,:,sortedIndexes);


% Dir z:
dirZ = (posTopLeftPixel_N - posTopLeftPixel_1) / (numSlices-1);

    
affineMatrix = [dircosX dircosY dirZ./sliceThickness posTopLeftPixel_1; 0 0 0 1];
if applyAffineTransform
    % We can apply directly the affine transform, the problem is that
    % scales the current image to have each pixel of 1mm. This makes the
    % images too big and is slower. We can use an alternative transform, where
    % only the rotation is applied. The scaling and offset is implemented
    % through the imageref3d (see 2nd method), but it's not exactly the same.
    
    % 1ST OPTION
    % Affine transformation matrix to go from image space to patient space:
%     affineMatrix = [dircosX.*pixelSpacing_mm(1) dircosY.*pixelSpacing_mm(2) dirZ posTopLeftPixel_1; 0 0 0 1];
%     matlabAffine = affine3d(affineMatrix'); % Create an affine object for matlab (that affine matrix is transposed as expected by matlab).
%     [image imageRef3d] = imwarp(image, matlabAffine); % Apply transform.

%   2ND OPTION
    affineMatrix = [dircosX dircosY dirZ./sliceThickness [0;0;0]; 0 0 0 1];   % Affine matrix without scaling and translation.
    matlabAffine = affine3d(affineMatrix'); % Create an affine object for matlab.
    
    xLim = [posTopLeftPixel_1(1)-pixelSpacing_mm(1)/2 posTopLeftPixel_1(1)-pixelSpacing_mm(1)/2+pixelSpacing_mm(1)*size(image,2)];
    yLim = [posTopLeftPixel_1(2)-pixelSpacing_mm(2)/2 posTopLeftPixel_1(2)-pixelSpacing_mm(2)/2+pixelSpacing_mm(2)*size(image,1)];
    % The z axis is different, it can go from negative to positive or in
    % the opposite way:
    incZ = sign(posTopLeftPixel_N(3)-posTopLeftPixel_1(3));
    zLim = [posTopLeftPixel_1(3)-incZ.*sliceThickness/2 posTopLeftPixel_N(3)+incZ.*sliceThickness/2];
    inImageRef3d = imref3d(size(image), pixelSpacing_mm(1), pixelSpacing_mm(2), sliceThickness);  % The limits needs to be ascending values, thats why we use min and max.
    % This is a more practical implementation
    [image imageRef3d] = imwarp(image, inImageRef3d, matlabAffine);
    % Add the displacemente to the reference:
    imageRef3d.XWorldLimits = imageRef3d.XWorldLimits + posTopLeftPixel_1(1);
    imageRef3d.YWorldLimits = imageRef3d.YWorldLimits + posTopLeftPixel_1(2);
    imageRef3d.ZWorldLimits = imageRef3d.ZWorldLimits + posTopLeftPixel_1(3);
    
else
    imageRef3d = imref3d(size(image), 1, 1, 1);
end
% Affine transformation matrix to go from image space to patient space. I
% overwrite the affineMatrix used before because this is the correct
% affineMatrix:
affineMatrix = [dircosX.*pixelSpacing_mm(1) dircosY.*pixelSpacing_mm(2) dirZ posTopLeftPixel_1; 0 0 0 1];

% Convert to single the image:
image = single(image);