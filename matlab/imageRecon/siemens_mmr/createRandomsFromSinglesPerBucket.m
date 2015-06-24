%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 09/03/2015
%  *********************************************************************
%  sinoRandoms = createRandomsFromSinglesPerBucket(headerFilename)
%
%  Creates a sinogram3d random estimate from the singles per bucket. It
%  receives as a parameter the interfile header of siemens mMr, where the
%  singles per bucket are defined. Then the tau*si*sj is used.

function [sinoRandoms, structSizeSino] = createRandomsFromSinglesPerBucket(headerFilename)

% Read interfile header:
[structInterfile, structSizeSino] = getInfoFromInterfile(headerFilename);
% Size of the sinogram:
sinoRandoms = zeros(structSizeSino.numR, structSizeSino.numTheta, sum(structSizeSino.sinogramsPerSegment), 'single');

% First we create a map with the indexes fo each crystal element in each
% transverse 2d sinogram.
mapaDet1Ids = zeros(structSizeSino.numR, structSizeSino.numTheta, 'uint16');
mapaDet2Ids = zeros(structSizeSino.numR, structSizeSino.numTheta, 'uint16');
numDetectors = 504;
singles_rates_per_bucket = structInterfile.SinglesPerBucket;

% Offset. La proyeccion empieza en
% Histogram of amount of times has been used each detector:
detectorIds = 1 : numDetectors;
% Parameters of the buckets:
numberOfTransverseCrystalsPerBlock = 8;
numberOfAxialCrystalsPerBlock = 8;
numberOfTransverseBlocksPerBucket = 2;
numberOfAxialBlocksPerBucket = 1;
numberOfBuckets = 224;
numberofAxialBlocks = 8;
numberofTransverseBlocksPerRing = 8;
numberOfBucketsInRing = numberOfBuckets / (numberofTransverseBlocksPerRing);
numberOfBlocksInBucketRing = numberOfBuckets / (numberofTransverseBlocksPerRing*numberOfAxialBlocksPerBucket);
numberOfTransverseCrystalsPerBlock = 9; % includes the gap
numberOfTransverseCrystalsPerBlock_withoutGaps = 8; % includes the gap
numberOfAxialCrystalsPerBlock = 8;

% Create the map with the id of the crystal element for detector 1 and
% detector 2:
[mapDet1Ids, mapDet2Ids] = createMmrDetectorsIdInSinogram();
% Convert to BucketId:
mapBucket1Ids = ceil(mapDet1Ids/(numberOfTransverseBlocksPerBucket*numberOfTransverseCrystalsPerBlock));
mapBucket2Ids = ceil(mapDet2Ids/(numberOfTransverseBlocksPerBucket*numberOfTransverseCrystalsPerBlock));

% histBuckets = zeros(size(1:numberOfBuckets));

% Now we start going through each possible sinogram, then get the rings of
% each sinogram and get the crystal efficency for that ring in det1 and the
% crystal effiencies for the sencdo ring with det2. When there is axial
% compression the efficency is computed as an average of each of them. For
% example an sinogram compressed from two different axial position:
% det1(ring1_a)*det2(ring2_a)+det1(ring1_b)*det2(ring2_b)
indiceSino = 1; % indice del sinogram 3D.
for segment = 1 : structSizeSino.numSegments
    % Por cada segmento, voy generando los sinogramas correspondientes y
    % contándolos, debería coincidir con los sinogramas para ese segmento: 
    numSinosThisSegment = 0;
    % Recorro todos los z1 para ir rellenando
    for z1 = 1 : (structSizeSino.numZ*2)
        numSinosZ1inSegment = 0;   % Cantidad de sinogramas para z1 en este segmento
        % Recorro completamente z2 desde y me quedo con los que están entre
        % minRingDiff y maxRingDiff. Se podría hacer sin recorrer todo el
        % sinograma pero se complica un poco.
        z1_aux = z1;    % z1_aux la uso para recorrer.
        for z2 = 1 : structSizeSino.numZ
            % Ahora voy avanzando en los sinogramas correspondientes,
            % disminuyendo z1 y aumentnado z2 hasta que la diferencia entre
            % anillos llegue a maxRingDiff.
            if ((z1_aux-z2)<=structSizeSino.maxRingDiff(segment))&&((z1_aux-z2)>=structSizeSino.minRingDiff(segment))
                % Me asguro que esté dentro del tamaño del michelograma:
                if(z1_aux>0)&&(z2>0)&&(z1_aux<=structSizeSino.numZ)&&(z2<=structSizeSino.numZ)
                    numSinosZ1inSegment = numSinosZ1inSegment + 1;
                    % Get the singles rate for each ring:
                    axialBucket1 = ceil(z1_aux / (numberOfAxialBlocksPerBucket*numberOfAxialCrystalsPerBlock));
                    axialBucket2 = ceil(z2 / (numberOfAxialBlocksPerBucket*numberOfAxialCrystalsPerBlock));
                    bucketsId1 = mapBucket1Ids + (axialBucket1-1)*numberOfBucketsInRing;
                    bucketsId2 = mapBucket2Ids + (axialBucket2-1)*numberOfBucketsInRing;
%                     histBuckets = histBuckets + hist([bucketsId1(:); bucketsId2(:)], 1:numberOfBuckets);
                    sinoRandoms(:,:,indiceSino) = structInterfile.CoincidenceWindowWidthNs*1e-9* singles_rates_per_bucket(bucketsId1) ./ (numberOfTransverseCrystalsPerBlock_withoutGaps*numberOfAxialCrystalsPerBlock*numberOfTransverseBlocksPerBucket*numberOfAxialBlocksPerBucket) .* singles_rates_per_bucket(bucketsId2) ./ (numberOfTransverseCrystalsPerBlock_withoutGaps*numberOfAxialCrystalsPerBlock*numberOfTransverseBlocksPerBucket*numberOfAxialBlocksPerBucket); % I need to normalize to singles rate per bin
                end
            end
            % Pase esta combinación de (z1,z2), paso a la próxima:
            z1_aux = z1_aux - 1;
        end
        if(numSinosZ1inSegment>0)
            % I average the efficencies dividing by the number of axial
            % combinations used for this sino:
            %acquisition_dependant_ncf_3d(:,:,indiceSino) = acquisition_dependant_ncf_3d(:,:,indiceSino) / numSinosZ1inSegment;
            numSinosThisSegment = numSinosThisSegment + 1;
            indiceSino = indiceSino + 1;
        end
    end    
end
% bar(histBuckets);
% The image is in cps, convert to total counts, in order to be ready to
% apply to the randoms correction:
sinoRandoms = sinoRandoms .* structInterfile.ImageDurationSec;

