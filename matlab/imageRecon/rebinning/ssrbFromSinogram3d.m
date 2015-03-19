%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 20/01/2015
%  *********************************************************************
%  function [sinograms2d, structSizeSino2d]  = ssrbFromSinogram3d(sinogram3d, structSinzeSino3d)
% 
%  This functions receives a sinogram 3d and its size and rebin its in a
%  set of sinograms 2d. Returns the sinogram and its size.
%  It's considered that the first segment are direct sinograms. The
%  amount of the 2d sinograms is the amoun of sinograms of the first
%  segment.

function [sinograms2d, structSizeSino2d]  = ssrbFromSinogram3d(sinogram3d, structSizeSino3d)

% Size of sinogram 2d. We use 2*NumRings-1, in order to have an
% intermediate coordinate:
numSinos2d = structSizeSino3d.numZ * 2 - 1;
structSizeSino2d = getSizeSino2Dstruct(structSizeSino3d.numTheta, structSizeSino3d.numR, ...
    numSinos2d, structSizeSino3d.rFov_mm, structSizeSino3d.zFov_mm);
% We create an axial coordinate of each direct sinogram in rings. 
stepZ_rings = structSizeSino3d.numZ / (numSinos2d + 1);
zValues_rings = 1 : stepZ_rings : structSizeSino3d.numZ;


% Initialize sinograms 2d:
sinograms2d = zeros(structSizeSino2d.numTheta, structSizeSino2d.numR, structSizeSino2d.numZ);

% Now convert each sinogram of each segment in direct sinograms:
indiceSino = 1; % index of sinogram3d.
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
                    newZ = (z1_aux + z2)/2;
                    [hist,index] = histc(newZ, zValues_rings);
                    sinograms2d(:,:,index) = single(sinograms2d(:,:,index)) + single(sinogram3d(:,:,indiceSino));
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
    % Check if the sinograms in the segment are equal than expected:
    if(numSinosThisSegment ~= structSizeSino3d.sinogramsPerSegment(segment))
        perror(sprintf('The amount of sinograms in segment %d is different than expected.\n',segment));
    end
end
    