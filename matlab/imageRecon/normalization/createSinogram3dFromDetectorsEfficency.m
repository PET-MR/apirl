%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 23/01/2015
%  *********************************************************************
%  function sinoEfficencies = createSinogram3dFromDetectorsEfficency(efficenciesPerDetector, structSizeSino3d, visualization)
%
%  This functions create a 3d sinogram for the correction of detector
%  efficencies from the individual detector efficencies received in the
%  parameter efficenciesPerDetector. This is for the siemens biograph mMr.
%  Parameters:
%   -efficenciesPerDetector: efficencies of each of the 508 crystal
%   elements per ring. The matrix should be crystalElementsInRingxNumRings
%   -structSizeSino3d: struct with the size of the sinos 3d. This should be
%   the addecuate for the mMr. Because the algorithm takes into account
%   that there is no polar mashing.
%   -visualization: to visualiza the results.

function sinoEfficencies = createSinogram3dFromDetectorsEfficency(efficenciesPerDetector, structSizeSino3d, visualization)

% Size of the sinogram:
sinoEfficencies = zeros(structSizeSino3d.numR, structSizeSino3d.numTheta, sum(structSizeSino3d.sinogramsPerSegment), 'single');

% First we create a map with the indexes fo each crystal element in each
% transverse 2d sinogram.
mapaDet1Ids = zeros(structSizeSino3d.numR, structSizeSino3d.numTheta, 'uint16');
mapaDet2Ids = zeros(structSizeSino3d.numR, structSizeSino3d.numTheta, 'uint16');
numDetectors = size(efficenciesPerDetector,1);
numRings = size(efficenciesPerDetector,2);
if numRings ~= structSizeSino3d.numZ
    error('The amount of rings in the sinogram is different than the rings in the crystal efficencies.');
end
% This would be to include the gaps. It starts in the first position and
% repeats for each block. [gap 8 crystal elements gap 8 crystal
% elements...]. The same order is repeated in each ring.
efficenciesPerDetector(9:9:end,:) = 0;
% Offset. La proyeccion empieza en
% Histogram of amount of times has been used each detector:
detectorIds = 1 : numDetectors;
rings = 1 : numRings;
histDetIds = zeros(1, numDetectors);
histRingsUsed = zeros(1, numRings);

% Create the map with the id of the crystal element for detector 1 and
% detector 2:
theta = [0:structSizeSino3d.numTheta-1]'; % The index of thetas goes from 0 to numTheta-1 (in stir)
r = (-structSizeSino3d.numR/2):(-structSizeSino3d.numR/2+structSizeSino3d.numR-1);
[THETA, R] = meshgrid(theta,r);
mapaDet1Ids = rem((THETA + floor((R)/2) + numDetectors-1), numDetectors) + 1;   % The +1 is added in matlab version respect than c version, because here we have 1-base indexes.
mapaDet2Ids = rem((THETA - floor((R+1)/2) + numDetectors/2 -1), numDetectors) + 1; % The +1 is added in matlab version respect than c version, because here we have 1-base indexes.
histDetIds = hist([mapaDet1Ids(:); mapaDet2Ids(:)], detectorIds);

% Now we start going through each possible sinogram, then get the rings of
% each sinogram and get the crystal efficency for that ring in det1 and the
% crystal effiencies for the sencdo ring with det2. When there is axial
% compression the efficency is computed as an average of each of them. For
% example an sinogram compressed from two different axial position:
% det1(ring1_a)*det2(ring2_a)+det1(ring1_b)*det2(ring2_b)
indiceSino = 1; % indice del sinogram 3D.
for segment = 1 : structSizeSino3d.numSegments
    % Por cada segmento, voy generando los sinogramas correspondientes y
    % contándolos, debería coincidir con los sinogramas para ese segmento: 
    numSinosThisSegment = 0;
    % Recorro todos los z1 para ir rellenando
    for z1 = 1 : (structSizeSino3d.numZ*2)
        numSinosZ1inSegment = 0;   % Cantidad de sinogramas para z1 en este segmento
        % Recorro completamente z2 desde y me quedo con los que están entre
        % minRingDiff y maxRingDiff. Se podría hacer sin recorrer todo el
        % sinograma pero se complica un poco.
        z1_aux = z1;    % z1_aux la uso para recorrer.
        for z2 = 1 : structSizeSino3d.numZ
            % Ahora voy avanzando en los sinogramas correspondientes,
            % disminuyendo z1 y aumentnado z2 hasta que la diferencia entre
            % anillos llegue a maxRingDiff.
            if ((z1_aux-z2)<=structSizeSino3d.maxRingDiff(segment))&&((z1_aux-z2)>=structSizeSino3d.minRingDiff(segment))
                % Me asguro que esté dentro del tamaño del michelograma:
                if(z1_aux>0)&&(z2>0)&&(z1_aux<=structSizeSino3d.numZ)&&(z2<=structSizeSino3d.numZ)
                    numSinosZ1inSegment = numSinosZ1inSegment + 1;
                    % Get the efficencies for each ring and then multiply
                    % both factor and add (the addition is ebcause we have
                    % to average all the crystal combinations for a
                    % determined axial compression).
                    efficenciesZ1 = efficenciesPerDetector(:,z1_aux);
                    efficenciesZ2 = efficenciesPerDetector(:,z2);
                    sinoEfficencies(:,:,indiceSino) = sinoEfficencies(:,:,indiceSino) + efficenciesZ1(mapaDet1Ids) .* efficenciesZ2(mapaDet2Ids);
                    histRingsUsed = histRingsUsed + hist([z1_aux z2], rings);
                end
            end
            % Pase esta combinación de (z1,z2), paso a la próxima:
            z1_aux = z1_aux - 1;
        end
        if(numSinosZ1inSegment>0)
            % I average the efficencies dividing by the number of axial
            % combinations used for this sino:
            sinoEfficencies(:,:,indiceSino) = sinoEfficencies(:,:,indiceSino) / numSinosZ1inSegment;
            numSinosThisSegment = numSinosThisSegment + 1;
            indiceSino = indiceSino + 1;
        end
    end    
end



if visualization
    subplot(2,2,1);
    bar(detectorIds,histDetIds);
    title('Histogram of Detectors Used');
    subplot(2,2,2);
    bar(rings,histRingsUsed);
    title('Histogram of Rings Used');
    subplot(2,2,3);
    imshow(int16(mapaDet1Ids));
    title('Id Detectors 1');
    subplot(2,2,4);
    imshow(int16(mapaDet2Ids));
    title('Id Detectors 2');
end