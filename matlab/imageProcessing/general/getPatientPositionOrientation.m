function out = getPatientPositionOrientation(path)
% Read info from the images:
% path = '\\Bioeng202-pc\pet-m\FDG_Patient_02\MPRAGE_image\';
baseFilename =[];
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

    
affineMatrix = [dircosX dircosY dirZ./sliceThickness [0;0;0]; 0 0 0 1];


out.affineMatrix = affineMatrix;
out.posTopLeftPixel_N = posTopLeftPixel_N;
out.posTopLeftPixel_1 = posTopLeftPixel_1;
out.sliceThickness = sliceThickness;
out.ImageSize = size(image);
out.pixelSpacing_mm = pixelSpacing_mm;
out.dirZ =dirZ;
out.dircosX = dircosX;
out.dircosY = dircosY;
