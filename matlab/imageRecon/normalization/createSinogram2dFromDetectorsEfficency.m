%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 23/01/2015
%  *********************************************************************
%  function sinoEfficencies = createSinogram2dFromDetectorsEfficency(efficenciesPerDetector, structSizeSino2d, visualization)
%
%  This functions create a 2d sinogram for the correction of detector
%  efficencies from the individual detector efficencies reveived in the
%  parameter efficenciesPerDetector.

function sinoEfficencies = createSinogram2dFromDetectorsEfficency(efficenciesPerDetector, structSizeSino2d, visualization)

sinoEfficencies = zeros(structSizeSino2d.numTheta, structSizeSino2d.numR);
numDetectors = numel(efficenciesPerDetector);
% minDiff between detectors:
minDiffDetectors = (numDetectors - structSizeSino2d.numR) / 2;

% Histogram of amount of times has been used each detector:
detectorIds = 1 : numDetectors;
histDetIds = zeros(1, numDetectors);
% Go for each detector and then for each combination:
for det1 = 1 : numel(efficenciesPerDetector)
    for indDet2 = 1 : structSizeSino2d.numR/2
        % For each detector, the same angle for two detectors (half angles)
        indProj = det1 - indDet2;
        det2 =  mod(det1 - 2*indDet2 + numDetectors-minDiffDetectors, numDetectors)+1;
        if (indProj > 0) && (det2 > 0) && (indProj <= structSizeSino2d.numTheta) && (det2 <= numDetectors)
            histDetIds = histDetIds + hist([det1; det2], detectorIds);
            sinoEfficencies(indProj,2*indDet2-1) = efficenciesPerDetector(det1) .* efficenciesPerDetector(det2);
        end 
        det2 = mod((det1 - 2*indDet2 + 1 + numDetectors-minDiffDetectors), numDetectors)+1;
        if (indProj > 0) && (det2 > 0) && (indProj <= structSizeSino2d.numTheta) && (det2 <= numDetectors)
            histDetIds = histDetIds + hist([det1; det2], detectorIds);
            sinoEfficencies(indProj,2*indDet2) = efficenciesPerDetector(det1) .* efficenciesPerDetector(det2);
        end
    end
end
sum(histDetIds)
if visualization
    figure;
    subplot(1,2,1);
    imshow(sinoEfficencies./max(max(sinoEfficencies)));
    title('Sinogram With Detector Efficencies');
    subplot(1,2,2);
    bar(detectorIds,histDetIds);
    title('Histogram of Detectors Used');
end