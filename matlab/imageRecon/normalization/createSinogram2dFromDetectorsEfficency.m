%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 23/01/2015
%  *********************************************************************
%  function sinoEfficencies = createSinogram2dFromDetectorsEfficency(efficenciesPerDetector, structSizeSino2d, visualization)
%
%  This functions create a 2d sinogram for the correction of detector
%  efficencies from the individual detector efficencies reveived in the
%  parameter efficenciesPerDetector. This is for the siemens biograph mMr.

function sinoEfficencies = createSinogram2dFromDetectorsEfficency(efficenciesPerDetector, structSizeSino2d, visualization)

sinoEfficencies = zeros(structSizeSino2d.numTheta, structSizeSino2d.numR);
mapaDet1Ids = zeros(structSizeSino2d.numTheta, structSizeSino2d.numR, 'uint16');
mapaDet2Ids = zeros(structSizeSino2d.numTheta, structSizeSino2d.numR, 'uint16');
numDetectors = numel(efficenciesPerDetector);
% minDiff between detectors:
minDiffDetectors = (numDetectors - structSizeSino2d.numR) / 2;
% This would be to include the gaps:
%efficenciesPerDetector(1:9:end) = 0;
% Offset. La proyeccion empieza en
% Histogram of amount of times has been used each detector:
detectorIds = 1 : numDetectors;
histDetIds = zeros(1, numDetectors);
% The first projection (0º) is centered in the scanner. Using a base id of
% 1, it cover blocks from 45:54 1:10. The central bin+1 is the first pixel
% of block 1, so it would be detector id 1. That is for the detector 1. For
% each projection a pixel is used twice (half angles).
detector1IdsForFirstProj = round([504*2-171:504*2, 1:172]./2);
% For the detector 2 is the opposite:
detector2IdsForFirstProj = round([338.5:-0.5:(338-344/2)+1]); % There are half angles, thus there are oblique with 1 pixel od diff lors for 0º projections.
detector2IdsForFirstProj(detector2IdsForFirstProj>numDetectors) = detector2IdsForFirstProj(detector2IdsForFirstProj>numDetectors) - numDetectors;
% I use this as a base for the next projections, for each new projection
% the detector1 id is shifted left and the detector 2 id is shifted right:
for idProj = 1 :  size(sinoEfficencies,1)
    % Shift left is the same tha sum 1
    idDet1 = detector1IdsForFirstProj + idProj - 1;
    idDet1(idDet1>numDetectors) = idDet1(idDet1>numDetectors) - numDetectors;
    % The opposite for det 2:
    idDet2 = detector2IdsForFirstProj + idProj - 1;
    idDet2(idDet2>numDetectors) = idDet2(idDet2>numDetectors) - numDetectors;
    sinoEfficencies(idProj, :) = efficenciesPerDetector(idDet1).*efficenciesPerDetector(idDet2);
    mapaDet1Ids(idProj, :) = idDet1;
    mapaDet2Ids(idProj, :) = idDet2;
    histDetIds = histDetIds + hist([idDet1 idDet2], detectorIds);
end

if visualization
    figure;
    subplot(2,2,1);
    imshow(sinoEfficencies./max(max(sinoEfficencies)));
    title('Sinogram With Detector Efficencies');
    subplot(2,2,2);
    bar(detectorIds,histDetIds);
    title('Histogram of Detectors Used');
    subplot(2,2,3);
    imshow(mapaDet1Ids);
    title('Id Detectors 1');
    subplot(2,2,4);
    imshow(mapaDet2Ids);
    title('Id Detectors 2');
end