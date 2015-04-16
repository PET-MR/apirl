%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 19/02/2015
%  *********************************************************************
%  function [mapaDet1Ids mapaDet2Ids] = createMmrDetectorsIdInSinogram()
%
%  This functions create two sinograms with the id of each detector in a
%  bin LOR. It works for a sinogram 3d for span 1, so one bin one detector. 

function [mapaDet1Ids, mapaDet2Ids] = createMmrDetectorsIdInSinogram3d()

% Size of mMr Sinogram's
numTheta = 252; numR = 344; numRings = 64; maxAbsRingDiff = 60; rFov_mm = 594/2; zFov_mm = 258; span = 1;
structSizeSino3d = getSizeSino3dFromSpan(numR, numTheta, numRings, rFov_mm, zFov_mm, span, maxAbsRingDiff);

% First we create a map with the indexes fo each crystal element in each
% transverse 2d sinogram.
mapaDet1Ids = zeros(structSizeSino3d.numR, structSizeSino3d.numTheta, sum(structSizeSino3d.sinogramsPerSegment), 'uint16');
mapaDet2Ids = zeros(structSizeSino3d.numR, structSizeSino3d.numTheta, sum(structSizeSino3d.sinogramsPerSegment), 'uint16');
numDetectorsPerRing = 504 ;
numDetectors = numDetectorsPerRing * numRings;

% Create the map with the id of the crystal element for detector 1 and
% detector 2:
theta = [0:structSizeSino3d.numTheta-1]'; % The index of thetas goes from 0 to numTheta-1 (in stir)
r = (-structSizeSino3d.numR/2):(-structSizeSino3d.numR/2+structSizeSino3d.numR-1);
[THETA, R] = meshgrid(theta,r);
mapaDet1Ids_2d = rem((THETA + floor(R/2) + numDetectorsPerRing), numDetectorsPerRing) + 1;   % The +1 is added in matlab version respect than c version, because here we have 1-base indexes.
mapaDet2Ids_2d = rem((THETA - floor((R+1)/2) + numDetectorsPerRing/2), numDetectorsPerRing) + 1; % The +1 is added in matlab version respect than c version, because here we have 1-base indexes.

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
                    % Get the detector id for the sinograms. The 2d id I
                    % need to add the offset of the ring:
                    mapaDet1Ids(:,:,indiceSino) = mapaDet1Ids_2d + numDetectorsPerRing*(z1_aux-1);
                    mapaDet2Ids(:,:,indiceSino) = mapaDet2Ids_2d + numDetectorsPerRing*(z2-1);
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