%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 19/02/2015
%  *********************************************************************
%  Function that creates the system matrix for the mlem algorithm for crystal
%  efficiencies. Goes from detectors to sinograms. This one is for he 3d
%  case.
function [detector1SystemMatrix, detector2SystemMatrix] = createDetectorSystemMatrix3d(span, normalize)
%% SYSTEM MATRIZ FOR SPAN1 SINOGRAMS
% Size of mMr Sinogram's
numTheta = 252; numR = 344; numRings = 64; maxAbsRingDiff = 60; rFov_mm = 594/2; zFov_mm = 258;
structSizeSino3d = getSizeSino3dFromSpan(numR, numTheta, numRings, rFov_mm, zFov_mm, span, maxAbsRingDiff);
numDetectorsPerRing = 504;
numDetectors = numDetectorsPerRing*numRings;

[mapaDet1Ids, mapaDet2Ids] = createMmrDetectorsIdInSinogram3d();
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
if span == 1
    % This is more fficient to compute it:
    detector1SystemMatrix = sparse(1:numBins, double(mapaDet1Ids(:)), true, numBins,numDetectors,maxNumNonZeros);
    detector2SystemMatrix = sparse(1:numBins, double(mapaDet2Ids(:)), true, numBins,numDetectors,maxNumNonZeros);
else
    numBins = numR*numTheta*sum(structSizeSino3d.sinogramsPerSegment);
    accIndicesBin = [];
    accDetector1Ids = [];
    accDetector2Ids = [];
    % Transform the det Ids map to michelogram (4d matrix):
    structSizeSino3d_span1 = getSizeSino3dFromSpan(numR, numTheta, numRings, rFov_mm, zFov_mm, 1, maxAbsRingDiff);
    michelogramDet1Ids = generateMichelogramFromSinogram3D(mapaDet1Ids, structSizeSino3d_span1);
    michelogramDet2Ids = generateMichelogramFromSinogram3D(mapaDet2Ids, structSizeSino3d_span1);
    clear mapaDet1Ids;
    clear mapaDet2Ids;
    % Now we start going through each possible sinogram, then get the rings of
    % each sinogram and get the crystal efficency for that ring in det1 and the
    % crystal effiencies for the sencdo ring with det2. When there is axial
    % compression the efficency is computed as an average of each of them. For
    % example an sinogram compressed from two different axial position:
    % det1(ring1_a)*det2(ring2_a)+det1(ring1_b)*det2(ring2_b)
    indiceSino = 1; % indice del sinogram 3D.
    numCrystalsMashed = zeros(1,numBins);
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
                        % Generate an sparse matrix for this sinogram. Need
                        % the linear index in the sinogram (depends on
                        % index sino:
                        indicesBin = ((indiceSino-1)*numR*numTheta + 1) : indiceSino*numR*numTheta; 
                        % Detector ids for this bin:               
                        detector1Ids = michelogramDet1Ids(:,:,z1_aux,z2);
                        detector2Ids = michelogramDet2Ids(:,:,z1_aux,z2);
                        % Acumulate indexes:
                        accIndicesBin = [accIndicesBin indicesBin];
                        accDetector1Ids = [accDetector1Ids detector1Ids(:)'];
                        accDetector2Ids = [accDetector2Ids detector2Ids(:)'];
                        % Normalization factor for number of crystal used:
                        numCrystalsMashed(indicesBin) = numCrystalsMashed(indicesBin) + 1;
                    end
                end
                % Pase esta combinación de (z1,z2), paso a la próxima:
                z1_aux = z1_aux - 1;
            end
            if(numSinosZ1inSegment>0)
                % I average the efficencies dividing by the number of axial
                % combinations used for this sino:
                numSinosThisSegment = numSinosThisSegment + 1;
                indiceSino = indiceSino + 1;
            end
        end    
    end
    % Create the sparse matrix:
    detector1SystemMatrix = sparse(accIndicesBin, double(accDetector1Ids(:)), 1./numCrystalsMashed(accIndicesBin), numBins,numDetectors,maxNumNonZeros);
    detector2SystemMatrix = sparse(accIndicesBin, double(accDetector2Ids(:)), 1./numCrystalsMashed(accIndicesBin), numBins,numDetectors,maxNumNonZeros);
end

combinedNormalization =  sum(detector1SystemMatrix,1) + sum(detector2SystemMatrix,1);

% Check if the matrix is required to be normalized (by default is normalized):
if nargin == 1 || ((nargin==2)&&(normalize==1))
    for i = 1 : size(detector1SystemMatrix,2)
        detector1SystemMatrix(:,i) = detector1SystemMatrix(:,i)./combinedNormalization(i)';
        detector2SystemMatrix(:,i) = detector2SystemMatrix(:,i)./combinedNormalization(i)';
    end
end

