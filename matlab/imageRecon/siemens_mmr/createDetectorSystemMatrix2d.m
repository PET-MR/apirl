%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 19/02/2015
%  *********************************************************************
%  Function that creates the system matrix for the mlem algorithm for crystal
%  efficiencies. Goes from detectors to sinograms. This one is for he 3d
%  case.
function [detector1SystemMatrix, detector2SystemMatrix] = createDetectorSystemMatrix2d(normalize, removeGaps)
%% SYSTEM MATRIZ FOR SPAN1 SINOGRAMS
if nargin == 1
    removeGaps = 1; % Remove gaps by default.
end
% Size of mMr Sinogram's
numTheta = 252; numR = 344; numRings = 1; maxAbsRingDiff = 60; rFov_mm = 594/2; zFov_mm = 258;
structSizeSino2d = getSizeSino2dStruct(numR, numTheta, numRings, rFov_mm, zFov_mm);
numDetectorsPerRing = 504;
numDetectors = numDetectorsPerRing*numRings;
if removeGaps == 1
    maskGaps = 1;
else
    maskGaps = 0;
end
[mapaDet1Ids, mapaDet2Ids] = createMmrDetectorsIdInSinogram();

numBins = numel(mapaDet1Ids);
% Image with as many rows as bins in the sinogram, and as many cols as
% detectors. I create a saprse matrix for 3d because the size is to big:
maxNumNonZeros = numBins; % One detector per bin.
% This takes too much time:
% detector1SystemMatrix = spalloc(numBins,numDetectors,maxNumNonZeros);
% detector2SystemMatrix = spalloc(numBins,numDetectors,maxNumNonZeros);
% for i = 1 : numDetectors
%     detector1SystemMatrix(:,i) = mapaDet1Ids(:) == i;
%     detector2SystemMatrix(:,i) = mapaDet2Ids(:) == i;
% end
indicesBins = 1:numBins;

% This is more fficient to compute it:
detector1SystemMatrix = sparse(indicesBins(mapaDet1Ids~=0&mapaDet2Ids~=0), double(mapaDet1Ids(mapaDet1Ids~=0&mapaDet2Ids~=0)), true, numBins,numDetectors,maxNumNonZeros);
detector2SystemMatrix = sparse(indicesBins(mapaDet1Ids~=0&mapaDet2Ids~=0), double(mapaDet2Ids(mapaDet1Ids~=0&mapaDet2Ids~=0)), true, numBins,numDetectors,maxNumNonZeros);


combinedNormalization =  sum(detector1SystemMatrix,1) + sum(detector2SystemMatrix,1);

% Check if the matrix is required to be normalized (by default is normalized):
if nargin == 1 || ((nargin==2)&&(normalize==1))
    for i = 1 : size(detector1SystemMatrix,2)
        if(combinedNormalization(i)~=0) % Because of the gaps.
            detector1SystemMatrix(:,i) = detector1SystemMatrix(:,i)./combinedNormalization(i)';
            detector2SystemMatrix(:,i) = detector2SystemMatrix(:,i)./combinedNormalization(i)';
        end
    end
end

