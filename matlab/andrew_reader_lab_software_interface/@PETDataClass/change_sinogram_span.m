function [sinogram_out, sinogram_size_out] = change_sinogram_span(objPETRawData, sinogram_in, sinogram_size_in)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    if sinogram_size_in.span ~= 1
        % The output span is different than span 1, so we need to expand to
        % span 1 and then compressing:
        michelogram = zeros( sinogram_size_in.nRadialBins, sinogram_size_in.nAnglesBins, sinogram_size_in.nRings, sinogram_size_in.nRings);
        tic;
        % Recorro todos los sinogramas 3d y los asigno a donde corresponde:
        % Recorro todos los segmentos, y voy analizando la diferencia entre
        % anillos.
        indiceSino = 1; % indice del sinogram 3D.
        for segment = 1 : sinogram_size_in.nSeg
            % Por cada segmento, voy generando los sinogramas correspondientes y
            % contándolos, debería coincidir con los sinogramas para ese segmento: 
            numSinosThisSegment = 0;
            % Recorro todos los z1 para ir rellenando
            for z1 = 1 : (sinogram_size_in.nRings*2)
                numSinosZ1inSegment = 0;   % Cantidad de sinogramas para z1 en este segmento
                % Recorro completamente z2 desde y me quedo con los que están entre
                % minRingDiffss y maxRingDiffss. Se podría hacer sin recorrer todo el
                % sinograma pero se complica un poco.
                z1_aux = z1;    % z1_aux la uso para recorrer.
                for z2 = 1 : sinogram_size_in.nRings
                    % Ahora voy avanzando en los sinogramas correspondientes,
                    % disminuyendo z1 y aumentnado z2 hasta que la diferencia entre
                    % anillos llegue a maxRingDiffss.
                    if ((z1_aux-z2)<=sinogram_size_in.maxRingDiffss(segment))&&((z1_aux-z2)>=sinogram_size_in.minRingDiffss(segment))
                        % Me asguro que esté dentro del tamaño del michelograma:
                        if(z1_aux>0)&&(z2>0)&&(z1_aux<=sinogram_size_in.nRings)&&(z2<=sinogram_size_in.nRings)
                            numSinosZ1inSegment = numSinosZ1inSegment + 1;
                            michelogram(:,:,z1_aux,z2) = sinogram_in(:,:,indiceSino);
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
            if(numSinosThisSegment ~= sinogram_size_in.nPlanesPerSeg(segment))
                perror(sprintf('The amount of sinograms in segment %d is different than expected.\n',segment));
            end
        end
        % Reduce the michelogram with axial compression:
    end
    
    % Outpus sinogram:
    sinogram_size_out = init_sinogram_size(objPETRawData, span, sinogram_size_in.nRings, sinogram_size_in.nRings.maxRingDiffserence);
    sinogram_out = single(sinogram_size_out.matrixSize);
    % Recorro todos los segmentos, y voy analizando la diferencia entre
    % anillos.
    indiceSino = 1; % indice del sinogram 3D.
    
    tic;
    for segment = 1 : sinogram_size.nSeg
        % Por cada segmento, voy generando los sinogramas correspondientes y
        % contándolos, debería coincidir con los sinogramas para ese segmento: 
        numSinosThisSegment = 0;
        % Recorro todos los z1 para ir rellenando
        for z1 = 1 : (sinogram_size_out.nRings*2)
            numSinosZ1inSegment = 0;   % Cantidad de sinogramas para z1 en este segmento
            % Recorro completamente z2 desde y me quedo con los que están entre
            % minRingDiffs y maxRingDiffs. Se podría hacer sin recorrer todo el
            % sinograma pero se complica un poco.
            z1_aux = z1;    % z1_aux la uso para recorrer.
            for z2 = 1 : sinogram_size_out.nRings
                % Ahora voy avanzando en los sinogramas correspondientes,
                % disminuyendo z1 y aumentnado z2 hasta que la diferencia entre
                % anillos llegue a maxRingDiffs.
                if ((z1_aux-z2)<=sinogram_size_out.maxRingDiffs(segment))&&((z1_aux-z2)>=sinogram_size_out.minRingDiffs(segment))
                    % Me asguro que esté dentro del tamaño del michelograma:
                    if(z1_aux>0)&&(z2>0)&&(z1_aux<=sinogram_size_out.nRings)&&(z2<=sinogram_size_out.nRings)
                        numSinosZ1inSegment = numSinosZ1inSegment + 1;
                        sinogram_out(:,:,indiceSino) = sinogram_out(:,:,indiceSino) + michelogram(:,:,z1_aux,z2);
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
    if( indiceSino ~= (sum(sinogram_size_out.nPlanesPerSeg)+1))
        fprintf('Error: La cantidad de sinogramas escritos es distinta a la suma de los sinogramas por segmento.\n');
    end
end

