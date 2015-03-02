%  *********************************************************************
%  Proyecto AR-PET. Comisión Nacional de Energía Atómica.
%  Autor: Martín Belzunce. UTN-FRBA.
%  Fecha de Creación: 30/05/2012
%  *********************************************************************

% Función que reduce un michelograma completo, a un sinograma 3d reducido
% por segmentos para mejorar la estadística del mismo. Recibe el
% michelograma, que es una matriz de 4 dimensiones con tamaño (NR, Nproy,
% NZ, NZ), y tres vectores con los sinogramas por segmento, y la máxima y
% mínima diferencia entre anillos de ellos.

function sinogram3D = reduceMichelogram(michelogram, structSizeSino)

sinogram3D = [];

if(size(structSizeSino.sinogramsPerSegment)~=size(structSizeSino.minRingDiff))|(size(structSizeSino.sinogramsPerSegment)~=size(structSizeSino.maxRingDiff))
    printf('Error: Los vectores sinogramsPerSegment, minRingDiff, maxRingDiff deben ser del mismo tamaño\n');
end

% Verifico que el michelograma sea correcto:
if(size(michelogram,3) ~= size(michelogram,4))
    printf('Error: El michelograma no tiene la misma cantidad de anillos para z1 que para z2');
end

% Creo el sinogram 3D, que va a tener 3 dimensiones, (Nr, Nproy, 
% SinogramIndex), si lo quisiera hacer de 4 dimensiones indexado por
% segmento debería usar un cell array porque cada segmento tiene distinta
% cantidad de sinogramas.

% Recorro todos los segmentos, y voy analizando la diferencia entre
% anillos.
indiceSino = 1; % indice del sinogram 3D.
sinogram3D = single(zeros(structSizeSino.numR, structSizeSino.numTheta, sum(structSizeSino.sinogramsPerSegment)));
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
                    sinogram3D(:,:,indiceSino) = sinogram3D(:,:,indiceSino) + michelogram(:,:,z1_aux,z2);
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

% Cuando terminé indiceSino debería ser igual a la totalidad de sinogramas
% + 1:
if( indiceSino ~= (sum(structSizeSino.sinogramsPerSegment)+1))
    fprintf('Error: La cantidad de sinogramas escritos es distinta a la suma de los sinogramas por segmento.\n');
end