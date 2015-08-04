%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 01/07/2015
%  *********************************************************************
%  This function resamples an MR image in dicom format into a PET image in
%  siemens interfile format. The first parameter is the full filename of
%  the siemens interfile image.
%  Receives the dicom path as a parameter and additionaly a baseFilename to filter the files
%  inside the folder. This might be useful if there are several data sets
%  in the same folder. If the folder contains only one data set it can be
%  an empty string.
%  It returns the resampled mr image, the pet image, and a 3-D
%  spatial referencing object (imref3d) that contains the spatial reference
%  of the image.

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