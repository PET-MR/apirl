%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 23/02/2016
%  *********************************************************************
%  Function that makes the fan sum from
function [crystalCounts] = FansumMmr(sinogram, structSizeSino3d, method)

% By default method 1:
if nargin == 2 
    method = 1;
end
% Crystals in the mmr scanner:
numRings = 64;
numDetectorsPerRing = 504;

% Vector with the counts in each crystal:
crystalCounts = zeros(numDetectorsPerRing, numRings);
crystalCountsVector = zeros(numDetectorsPerRing* numRings,1);

if method == 1 
    % Maps qith the id of each crystal in the sinogram:
    [mapaDet1Ids, mapaDet2Ids] = createMmrDetectorsIdInSinogram3d();

    % For each crystal, get all the sinogram bins where its involved and sum
    % the counts:
    fprintf('Fansum Crystal ');
    for i = 1 : numel(crystalCountsVector)
        fprintf('%d,',i);
        crystalAsDet1 = mapaDet1Ids == i;
        crystalAsDet2 = mapaDet2Ids == i;
        crystalCountsVector(i) = sum(sinogram(crystalAsDet1|crystalAsDet2));
    end
    crystalCounts = reshape(crystalCountsVector, [numDetectorsPerRing numRings]);
elseif method == 2
    % Create the map with the id of the crystal element for detector 1 and
    % detector 2:
    theta = [0:structSizeSino3d.numTheta-1]'; % The index of thetas goes from 0 to numTheta-1 (in stir)
    r = (-structSizeSino3d.numR/2):(-structSizeSino3d.numR/2+structSizeSino3d.numR-1);
    [THETA, R] = meshgrid(theta,r);
    mapaDet1Ids_2d = rem((THETA + floor(R/2) + numDetectorsPerRing -1), numDetectorsPerRing) + 1;   % The +1 is added in matlab version respect than c version, because here we have 1-base indexes.
    mapaDet2Ids_2d = rem((THETA - floor((R+1)/2) + numDetectorsPerRing/2 -1), numDetectorsPerRing) + 1; % The +1 is added in matlab version respect than c version, because here we have 1-base indexes.

    fprintf('Fansum Ring ');
    for j = 1 : size(crystalCounts,2)
        fprintf('%d,',j);
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
                            if(j == z1_aux)||(j == z2)
                                sinogramPlane = sinogram(:,:,indiceSino);
                                parfor i = 1 : size(crystalCounts,1)
                                    crystalAsDet1 = mapaDet1Ids_2d == i;
                                    crystalAsDet2 = mapaDet2Ids_2d == i;
                                    crystalCounts(i,j) = crystalCounts(i,j) + sum(sinogramPlane(crystalAsDet1|crystalAsDet2));
                                end
                            end
                        end
                    end

                    % Pase esta combinación de (z1,z2), paso a la próxima:
                    z1_aux = z1_aux - 1;
                end
                if(numSinosZ1inSegment>0)
                    numSinosThisSegment = numSinosThisSegment + 1;
                    indiceSino = indiceSino + 1;
                end
            end
        end    
    end
elseif method == 3
    % I go for each bin and sum in the respective crystal the counts:
    % Maps qith the id of each crystal in the sinogram:
    [mapaDet1Ids, mapaDet2Ids] = createMmrDetectorsIdInSinogram3d();

    for i = 1 : size(sinogram,1)
        for j = 1 : size(sinogram,2)
            for k = 1 : size(sinogram,3)
                crystalCountsVector(mapaDet1Ids(i,j,k)) = crystalCountsVector(mapaDet1Ids(i,j,k)) + sinogram(i,j,k);
                crystalCountsVector(mapaDet2Ids(i,j,k)) = crystalCountsVector(mapaDet2Ids(i,j,k)) + sinogram(i,j,k);
            end
        end
    end
    crystalCounts = reshape(crystalCountsVector, [numDetectorsPerRing numRings]);
elseif method == 4
    [detector1SystemMatrix, detector2SystemMatrix] = createDetectorSystemMatrix3d(1, 0);
    combinedNormalization =  sum(detector1SystemMatrix',2) + sum(detector2SystemMatrix',2);
    combSystemMatrix = (detector1SystemMatrix+detector2SystemMatrix);
    crystalCountsVector = double(sinogram(:))'*combSystemMatrix;
    crystalCounts = reshape(crystalCountsVector, [numDetectorsPerRing numRings]);
end

