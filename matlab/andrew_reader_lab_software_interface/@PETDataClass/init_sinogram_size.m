% *********************************************************************
% Reconstruction Framework for Siemens Biograph mMR.  
% class: Gpet
% Authors: Martin Belzunce, Abolfazl Mehranian. Kings College London.
% Date: 08/02/2016
% *********************************************************************

function sinogram_size_out = init_sinogram_size(objPETRawData, inSpan, numRings, maxRingDifference)
    objPETRawData.sinogram_size.nRadialBins = 344;
    objPETRawData.sinogram_size.nAnglesBins = 252;
    objPETRawData.sinogram_size.span = inSpan;
    objPETRawData.sinogram_size.nRings = numRings;
    objPETRawData.sinogram_size.maxRingDifference = maxRingDifference;
    % Number of planes mashed in each plane of the sinogram:
    objPETRawData.sinogram_size.numPlanesMashed = [];

    % Number of planes in odd and even segments:
    numPlanesOdd = floor(objPETRawData.sinogram_size.span/2);
    numPlanesEven = ceil(objPETRawData.sinogram_size.span/2);
    % Ahora tengo que determinar la cantidad de segmentos y las diferencias
    % minimas y maximas en cada uno de ellos. Se que las diferencias maximas y
    % minimas del segmento cero es +-span/2:
    objPETRawData.sinogram_size.minRingDiffs = -floor(objPETRawData.sinogram_size.span/2);
    objPETRawData.sinogram_size.maxRingDiffs = floor(objPETRawData.sinogram_size.span/2);
    objPETRawData.sinogram_size.nSeg = 1;
    % Empiezo a ir agregando los segmentos hasta llegar numRings o a la máxima
    % diferencia entra anillos:
    while(abs(objPETRawData.sinogram_size.minRingDiffs(objPETRawData.sinogram_size.nSeg)) < objPETRawData.sinogram_size.maxRingDifference)  % El abs es porque voy a estar comparando el segmento con diferencias negativas.
        % Si no llegue a esa condición, tengo un segmento más hacia cada lado
        % primero hacia el positivo y luego negativo:
        objPETRawData.sinogram_size.nSeg = objPETRawData.sinogram_size.nSeg+1;
        % Si estoy en el primer segmento a agregar es -1, sino es -2 para ir
        % con el positivo:
        if objPETRawData.sinogram_size.nSeg == 2
            objPETRawData.sinogram_size.minRingDiffs(objPETRawData.sinogram_size.nSeg) = objPETRawData.sinogram_size.minRingDiffs(objPETRawData.sinogram_size.nSeg-1) + objPETRawData.sinogram_size.span;
            objPETRawData.sinogram_size.maxRingDiffs(objPETRawData.sinogram_size.nSeg) = objPETRawData.sinogram_size.maxRingDiffs(objPETRawData.sinogram_size.nSeg-1) + objPETRawData.sinogram_size.span;
        else
            % Si me paso de la máxima difrencia de anillos la trunco:
            if (objPETRawData.sinogram_size.maxRingDiffs(objPETRawData.sinogram_size.nSeg-2) + objPETRawData.span) <= maxRingDifference
                objPETRawData.sinogram_size.minRingDiffs(objPETRawData.sinogram_size.nSeg) = objPETRawData.sinogram_size.minRingDiffs(objPETRawData.sinogram_size.nSeg-2) + objPETRawData.sinogram_size.span;
                objPETRawData.sinogram_size.maxRingDiffs(objPETRawData.sinogram_size.nSeg) = objPETRawData.sinogram_size.maxRingDiffs(objPETRawData.sinogram_size.nSeg-2) + objPETRawData.sinogram_size.span;
            else
                objPETRawData.sinogram_size.minRingDiffs(objPETRawData.sinogram_size.nSeg) = objPETRawData.sinogram_size.minRingDiffs(objPETRawData.sinogram_size.nSeg-2) + objPETRawData.sinogram_size.span;
                objPETRawData.sinogram_size.maxRingDiffs(objPETRawData.sinogram_size.nSeg) = objPETRawData.sinogram_size.maxRingDifference;
            end
        end
        % Ahora hacia el lado de las diferencias negativas:
        objPETRawData.sinogram_size.nSeg = objPETRawData.sinogram_size.nSeg+1;
        if (abs(objPETRawData.sinogram_size.minRingDiffs(objPETRawData.sinogram_size.nSeg-2) - objPETRawData.sinogram_size.span)) <= objPETRawData.sinogram_size.maxRingDifference
            objPETRawData.sinogram_size.minRingDiffs(objPETRawData.sinogram_size.nSeg) = objPETRawData.sinogram_size.minRingDiffs(objPETRawData.sinogram_size.nSeg-2) - objPETRawData.sinogram_size.span;  % Acá siempre debo ir -2 no tengo problema con el primero.
            objPETRawData.sinogram_size.maxRingDiffs(objPETRawData.sinogram_size.nSeg) = objPETRawData.sinogram_size.maxRingDiffs(objPETRawData.sinogram_size.nSeg-2) - objPETRawData.sinogram_size.span;
        else
            objPETRawData.sinogram_size.minRingDiffs(objPETRawData.sinogram_size.nSeg) = -objPETRawData.sinogram_size.maxRingDifference;  % Acá siempre debo ir -2 no tengo problema con el primero.
            objPETRawData.sinogram_size.maxRingDiffs(objPETRawData.sinogram_size.nSeg) = objPETRawData.sinogram_size.maxRingDiffs(objPETRawData.sinogram_size.nSeg-2) - objPETRawData.sinogram_size.span;
        end
    end

    % Ahora determino la cantidad de sinogramas por segmentos, recorriendo cada
    % segmento:
    objPETRawData.sinogram_size.nPlanesPerSeg = zeros(1,objPETRawData.sinogram_size.nSeg);

    for segment = 1 : objPETRawData.sinogram_size.nSeg
        % Por cada segmento, voy generando los sinogramas correspondientes y
        % contándolos, debería coincidir con los sinogramas para ese segmento:
        numSinosThisSegment = 0;
        % Recorro todos los z1 para ir rellenando
        for z1 = 1 : (numRings*2)
            numSinosZ1inSegment = 0;   % Cantidad de sinogramas para z1 en este segmento
            % Recorro completamente z2 desde y me quedo con los que están entre
            % minRingDiff y maxRingDiff. Se podría hacer sin recorrer todo el
            % sinograma pero se complica un poco.
            z1_aux = z1;    % z1_aux la uso para recorrer.
            for z2 = 1 : numRings
                % Ahora voy avanzando en los sinogramas correspondientes,
                % disminuyendo z1 y aumentnado z2 hasta que la diferencia entre
                % anillos llegue a maxRingDiff.
                if ((z1_aux-z2)<=objPETRawData.sinogram_size.maxRingDiffs(segment))&&((z1_aux-z2)>=objPETRawData.sinogram_size.minRingDiffs(segment))
                    % Me asguro que esté dentro del tamaño del michelograma:
                    if(z1_aux>0)&&(z2>0)&&(z1_aux<=numRings)&&(z2<=numRings)
                        numSinosZ1inSegment = numSinosZ1inSegment + 1;
                    end
                end
                % Pase esta combinación de (z1,z2), paso a la próxima:
                z1_aux = z1_aux - 1;
            end
            if(numSinosZ1inSegment>0)
                objPETRawData.sinogram_size.numPlanesMashed = [objPETRawData.sinogram_size.numPlanesMashed numSinosZ1inSegment];
                numSinosThisSegment = numSinosThisSegment + 1;
            end
        end
        % Guardo la cantidad de segmentos:
        objPETRawData.sinogram_size.nPlanesPerSeg(segment) = numSinosThisSegment;
    end
    objPETRawData.sinogram_size.nSinogramPlanes = sum(objPETRawData.sinogram_size.nPlanesPerSeg);
    objPETRawData.sinogram_size.matrixSize = [objPETRawData.sinogram_size.nRadialBins objPETRawData.sinogram_size.nAnglesBins objPETRawData.sinogram_size.nSinogramPlanes];
    sinogram_size_out = objPETRawData.sinogram_size;
end