%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 15/01/2015
%  *********************************************************************
%  michelogram = generateMichelogramFromSinogram3D(sinogram3d, structSizeSinogram3d)
% 
%  Michelogram is a sinogram 3d without span. The difference in a sinogra3d
%  with span 1, is that in the michelogram a 4d matrix is used with
%  (r, theta,z1,z2).

function michelogram = generateMichelogramFromSinogram3D(sinogram3d, structSizeSino3d)

michelogram = zeros( structSizeSino3d.numR, structSizeSino3d.numTheta, structSizeSino3d.numZ, structSizeSino3d.numZ);

% Recorro todos los sinogramas 3d y los asigno a donde corresponde:
% Recorro todos los segmentos, y voy analizando la diferencia entre
% anillos.
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
            if ((z2-z1_aux)<=structSizeSino3d.maxRingDiff(segment))&&((z2-z1_aux)>=structSizeSino3d.minRingDiff(segment))
                % Me asguro que esté dentro del tamaño del michelograma:
                if(z1_aux>0)&&(z2>0)&&(z1_aux<=structSizeSino3d.numZ)&&(z2<=structSizeSino3d.numZ)
                    numSinosZ1inSegment = numSinosZ1inSegment + 1;
                    michelogram(:,:,z1_aux,z2) = sinogram3d(:,:,indiceSino);
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

% Cuando terminé indiceSino debería ser igual a la totalidad de sinogramas
% + 1:
if( indiceSino ~= (sum(structSizeSino3d.sinogramsPerSegment)+1))
    fprintf('Error: La cantidad de sinogramas escritos es distinta a la suma de los sinogramas por segmento.\n');
end