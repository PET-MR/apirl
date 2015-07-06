function [mrRescaled, imagePet, pixelSize_mm, origin_mm] = RemapMrIntoPetImageSpace(interfileHeaderPet, pathMrDicom, baseDicomFilename, vis)


% Read interfile image:
[imagePet, pixelSize_mm, origin_mm, bedPosition_mm, info]  = interfileReadSiemensImage(interfileHeaderPet);
% Coordinates of the image:
coordX = origin_mm(1) : pixelSize_mm(1) : origin_mm(1)+pixelSize_mm(1)*(size(imagePet,1)-1);
coordY = origin_mm(2) : pixelSize_mm(2) : origin_mm(2)+pixelSize_mm(2)*(size(imagePet,2)-1);
% Based on bed position:
% coordZ = origin_mm(3) : pixelSize_mm(3) : origin_mm(3)+pixelSize_mm(3)*(size(imagePet,3)-1);
% Based on dicom reconstructed image:
coordZ = (floor(size(imagePet,3)/2)-1)*pixelSize_mm(3) : -pixelSize_mm(3) : -(floor(size(imagePet,3)/2+1))*pixelSize_mm(3);
[xPet_mm, yPet_mm, zPet_mm] = meshgrid(coordX, coordY, coordZ);

% Read dicom image:
[imageMr, affineMatrix, xMap_mm, yMap_mm, zMap_mm, dicomInfo] = ReadDicomImage(pathMrDicom, baseDicomFilename);

%% INTERPOLATION
% If the image rows and columns are in the same direction than y and x
% axis, I can use interp3 that is much easier to interpolate. But if the
% image is rotated from the axis, interp3 doesn't work and we need to use
% griddata that is much slower. For that reason in that case we use the
% affineMatrix of the dicom image to rotate and translate the original
% image into the patient space.

%mrRescaled = griddata(xMap_mm,yMap_mm,zMap_mm,double(imageMr),xPet_mm,yPet_mm,zPet_mm, 'nearest');%, 'linear'); 
% mrRescaled = interp3(xMap_mm,yMap_mm,zMap_mm,double(imageMr),xPet_mm,yPet_mm,zPet_mm, 'linear');
% mrRescaled(isnan(mrRescaled)) = 0;
refPetImage = imref3d(size(imagePet), [min(coordX) max(coordX)], [min(coordY) max(coordY)], [min(coordZ) max(coordZ)]);
% Get coordinate maps:
[X, Y, Z] = meshgrid(0:size(imageMr,2)-1,0:size(imageMr,1)-1, 0:size(imageMr,3)-1); % The affine transform is for zero-based indexes.
xMr_mm = affineMatrix(1,1) .* X + affineMatrix(1,2) .* Y + affineMatrix(1,3) .*Z + affineMatrix(1,4);
yMr_mm = affineMatrix(2,1) .* X + affineMatrix(2,2) .* Y + affineMatrix(2,3) .*Z + affineMatrix(2,4);
zMr_mm = affineMatrix(3,1) .* X + affineMatrix(3,2) .* Y + affineMatrix(3,3) .*Z + affineMatrix(3,4);
% Rotate the image and coordiantes to get proper grid parallel to x and y
% axes:
matlabAffine = affine3d(affineMatrix'); % Creat an affine object for matlab (that affine matrix is transposed as expected by matlab).
%invertRotationAffine = invert(matlabAffine)

% Apply transform:
refMrImage = imref3d(size(imageMr), dicomInfo.PixelSpacing(2), dicomInfo.PixelSpacing(1), 5.2);
[mrImageTransformed refTransMrImage] = imwarp(imageMr, matlabAffine);
xCoordMr_mm = refTransMrImage.XWorldLimits(1) + refTransMrImage.PixelExtentInWorldX/2: refTransMrImage.PixelExtentInWorldX : refTransMrImage.XWorldLimits(end);
yCoordMr_mm = refTransMrImage.YWorldLimits(1) + refTransMrImage.PixelExtentInWorldY/2 : refTransMrImage.PixelExtentInWorldY : refTransMrImage.YWorldLimits(end);
zCoordMr_mm = refTransMrImage.ZWorldLimits(1) + refTransMrImage.PixelExtentInWorldZ/2: refTransMrImage.PixelExtentInWorldZ : refTransMrImage.ZWorldLimits(end);
[xMr_mm, yMr_mm, zMr_mm] = meshgrid(xCoordMr_mm, yCoordMr_mm, zCoordMr_mm);
mrRescaled = interp3(xMr_mm,yMr_mm,zMr_mm,double(mrImageTransformed),xPet_mm,yPet_mm,zPet_mm, 'linear');
mrRescaled(isnan(mrRescaled)) = 0;
% c = permute(imageMr, [3 1 2]);
% for i = 1 : size(c,3)
%     c(:,:,i) = imrotate(c(:,:,i), -5.6723, 'bilinear','crop');
%     imshow(c(:,:,i), [0 max(max(c(:,:,i)))]);
% end
% B = permute(c, [2 3 1]);
if(vis)
    figure; 
    for i = 1 : size(mrImageTransformed,3)
         imshow(mrImageTransformed(:,:,i),[0 max(max(mrImageTransformed(:,:,i)))]);
        pause(0.2);
    end
end
%% VISUALIZATION
% If visualization enabled, show slice by slice:
if(vis)
    figure; 
    for i = 1 : size(imagePet,3)
         fusedImage = imfuse(imagePet(:,:,i),mrRescaled(:,:,i));%,'falsecolor','Scaling','joint','ColorChannels',[1 2 0]);
         imshow(fusedImage);
        pause(0.2);
    end
end