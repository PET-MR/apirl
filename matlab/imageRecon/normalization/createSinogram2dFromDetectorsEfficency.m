%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 23/01/2015
%  *********************************************************************
%  function sinoEfficencies = createSinogram2dFromDetectorsEfficency(efficenciesPerDetector, structSizeSino2d, method, visualization)
%
%  This functions create a 2d sinogram for the correction of detector
%  efficencies from the individual detector efficencies reveived in the
%  parameter efficenciesPerDetector. This is for the siemens biograph mMr.
%  Parameters:
%   -efficenciesPerDetector: efficencies of each of the 508 crystal elements in a ring.
%   -structSizeSino2d: struct with the size of the sinos 2d. This should be
%   the addecuate for the mMr. Because the algorithm takes into account
%   that there is no polar mashing.
%   -method: there are two methods, one developed by me (Martin Belzunce)
%   taking some information from the sinogram viewer of siemens tools and
%   other one taken from stir (that they have the correct information).
%   Both should have the same result. Method 1: mine. Method2: stir
%   formual.
%   -visualization: to visualiza the results.

%  The method 2 usis this formula from stir implemented in a proper and
%  quic method for matlab:
%  From stir:
% %   /*
%          adapted from CTI code
%          Note for implementation: avoid using % with negative numbers
%          so add num_detectors before doing modulo num_detectors)
%         */
%       uncompressed_view_tangpos_to_det1det2[v_num][tp_num].det1_num = 
%         (v_num + (tp_num >> 1) + num_detectors) % num_detectors;
%       uncompressed_view_tangpos_to_det1det2[v_num][tp_num].det2_num = 
%         (v_num - ( (tp_num + 1) >> 1 ) + num_detectors/2) % num_detectors;
%
%
function sinoEfficencies = createSinogram2dFromDetectorsEfficency(efficenciesPerDetector, structSizeSino2d, method, visualization)

sinoEfficencies = zeros(structSizeSino2d.numR, structSizeSino2d.numTheta);
mapaDet1Ids = zeros(structSizeSino2d.numR, structSizeSino2d.numTheta, 'uint16');
mapaDet2Ids = zeros(structSizeSino2d.numR, structSizeSino2d.numTheta, 'uint16');
numDetectors = numel(efficenciesPerDetector);
% minDiff between detectors:
minDiffDetectors = (numDetectors - structSizeSino2d.numR) / 2;
% This would be to include the gaps:
efficenciesPerDetector(1:9:end) = 0;
% Offset. La proyeccion empieza en
% Histogram of amount of times has been used each detector:
detectorIds = 1 : numDetectors;
histDetIds = zeros(1, numDetectors);

% See which method to use:
if(method == 1)  
    % The first projection (0º) is centered in the scanner. Using a base id of
    % 1, it cover blocks from 45:54 1:10. The central bin+1 is the first pixel
    % of block 1, so it would be detector id 1. That is for the detector 1. For
    % each projection a pixel is used twice (half angles).
    detector1IdsForFirstProj = round([numDetectors*2-171:numDetectors*2, 1:(structSizeSino2d.numR/2)]./2);
    % For the detector 2 is the opposite:
    detector2IdsForFirstProj = round([338.5:-0.5:(338-structSizeSino2d.numR/2)+1]); % There are half angles, thus there are oblique with 1 pixel od diff lors for 0º projections.
    detector2IdsForFirstProj(detector2IdsForFirstProj>numDetectors) = detector2IdsForFirstProj(detector2IdsForFirstProj>numDetectors) - numDetectors;
    % I use this as a base for the next projections, for each new projection
    % the detector1 id is shifted left and the detector 2 id is shifted right:
    for idProj = 1 :  size(sinoEfficencies,2)
        % Shift left is the same tha sum 1
        idDet1 = detector1IdsForFirstProj + idProj - 1;
        idDet1(idDet1>numDetectors) = idDet1(idDet1>numDetectors) - numDetectors;
        % The opposite for det 2:
        idDet2 = detector2IdsForFirstProj + idProj - 1;
        idDet2(idDet2>numDetectors) = idDet2(idDet2>numDetectors) - numDetectors;
        sinoEfficencies(:, idProj) = efficenciesPerDetector(idDet1).*efficenciesPerDetector(idDet2);
        mapaDet1Ids(:, idProj) = idDet1;
        mapaDet2Ids(:, idProj) = idDet2;
        histDetIds = histDetIds + hist([idDet1 idDet2], detectorIds);
    end
elseif method == 2
    % Method 2:
    theta = [0:structSizeSino2d.numTheta-1]'; % The index of thetas goes from 0 to numTheta-1 (in stir)
    r = (-structSizeSino2d.numR/2):(-structSizeSino2d.numR/2+structSizeSino2d.numR-1);
    [THETA, R] = meshgrid(theta, r);
    mapaDet1Ids = rem((THETA + floor(R/2) + numDetectors), numDetectors) + 1;   % The +1 is added in matlab version respect than c version, because here we have 1-base indexes.
    mapaDet2Ids = rem((THETA - floor((R+1)/2) + numDetectors/2), numDetectors) + 1; % The +1 is added in matlab version respect than c version, because here we have 1-base indexes.
    sinoEfficencies = efficenciesPerDetector(mapaDet1Ids).*efficenciesPerDetector(mapaDet2Ids);
    histDetIds = hist([mapaDet1Ids(:); mapaDet2Ids(:)], detectorIds);
else
    error('The method fiel has only two valid values: 1 (loop method) and 2 (matrix method)');
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
    imshow(int16(mapaDet1Ids));
    title('Id Detectors 1');
    subplot(2,2,4);
    imshow(int16(mapaDet2Ids));
    title('Id Detectors 2');
end