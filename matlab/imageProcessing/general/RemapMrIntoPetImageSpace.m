function [mrRescaled, imagePet, pixelSize_mm, origin_mm] = RemapMrIntoPetImageSpace(interfileHeaderPet, pathMrDicom, baseDicomFilename, vis)


% Read interfile image:
[imagePet, pixelSize_mm, origin_mm, info]  = interfileReadSiemensImage(interfileHeaderPet);
% Coordinates of the image:
coordX = origin_mm(1) : pixelSize_mm(1) : origin_mm(1)+pixelSize_mm(1)*(size(imagePet,1)-1);
coordY = origin_mm(2) : pixelSize_mm(2) : origin_mm(2)+pixelSize_mm(2)*(size(imagePet,2)-1);
coordZ = origin_mm(3) : pixelSize_mm(3) : origin_mm(3)+pixelSize_mm(3)*(size(imagePet,3)-1);
[xPet_mm, yPet_mm, zPet_mm] = meshgrid(coordX, coordY, coordZ);

% Read dicom image:
[imageMr, xMap_mm, yMap_mm, zMap_mm] = ReadDicomImage(pathMrDicom, baseDicomFilename);

%% INTERPOLATION
% Use griddata instead of itnerp3 because the dicom can return a grid that
% is rotated respecto to x and y coordinates.
mrRescaled = griddata(xMap_mm,yMap_mm,zMap_mm,double(imageMr),xPet_mm,yPet_mm,zPet_mm, 'linear'); 
mrRescaled(isnan(mrRescaled)) = 0;
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