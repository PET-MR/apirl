function [resampledMrImage, imagePet, refResampledImage] = RemapMrIntoPetImageSpace(interfileHeaderPet, pathMrDicom, baseDicomFilename, vis)
%% READ INPUT IMAGE
% Read interfile image:
[imagePet, refPetImage, bedPosition_mm, info]  = interfileReadSiemensImage(interfileHeaderPet);

% Read dicom image:
applyAffineTransform = 1;
[imageMr, refMrImage, affineMatrix, dicomInfo] = ReadDicomImage(pathMrDicom, baseDicomFilename, applyAffineTransform);

%% INTERPOLATE TO GET THE NEW IMAGE
[resampledMrImage, refResampledImage] = ImageResample(imageMr, refMrImage, refPetImage);
%% VISUALIZATION
% If visualization enabled, show slice by slice:
if(vis)
    figure; 
    for i = 1 : size(imagePet,3)
        fusedImage = imfuse(imagePet(:,:,i),resampledMrImage(:,:,i));%,'falsecolor','Scaling','joint','ColorChannels',[1 2 0]);
        imshow(fusedImage);
        pause(0.2);
    end
end