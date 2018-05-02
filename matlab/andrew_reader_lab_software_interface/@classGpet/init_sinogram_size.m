% *********************************************************************
% Reconstruction Framework for Siemens Biograph mMR.  
% class: Gpet
% Authors: Martin Belzunce, Abolfazl Mehranian. Kings College London.
% Date: 08/02/2016
% *********************************************************************

function init_sinogram_size(objGpet, inSpan, numRings, maxRingDifference)
    objGpet.sinogram_size.span = inSpan;
    if objGpet.sinogram_size.span == -1 % span =-1, means a 2d sinogram:
        if(nargin>2)
            if numRings ~= 1
                warning('A 2d sinogram size (span=-1) was selected but numRings greater than 1 was selected. numRings overrided to 1.');
                numRings = 1;
            end
        else
            numRings = 1;
        end
        aux = objGpet.sinogram_size;
        % Clear sinogram size to remove 3d parameters:
        objGpet.sinogram_size = rmfield(objGpet.sinogram_size, fieldnames(objGpet.sinogram_size));
        objGpet.sinogram_size.nRadialBins = aux.nRadialBins;
        objGpet.sinogram_size.nAnglesBins = aux.nAnglesBins;
        objGpet.sinogram_size.span = aux.span;
        objGpet.sinogram_size.nRings = numRings;
        objGpet.sinogram_size.maxRingDifference = 0;
        objGpet.sinogram_size.matrixSize = [objGpet.sinogram_size.nRadialBins objGpet.sinogram_size.nAnglesBins 1];
        objGpet.sinogram_size.nSinogramPlanes = numRings;
    elseif objGpet.sinogram_size.span == 0
        % multi slice 2d:
        aux = objGpet.sinogram_size;
        % Clear sinogram size to remove 3d parameters:
        objGpet.sinogram_size = rmfield(objGpet.sinogram_size, fieldnames(objGpet.sinogram_size));
        objGpet.sinogram_size.nRadialBins = aux.nRadialBins;
        objGpet.sinogram_size.nAnglesBins = aux.nAnglesBins;
        objGpet.sinogram_size.span = aux.span;
        objGpet.sinogram_size.nRings = numRings;
        objGpet.sinogram_size.maxRingDifference = 0;
        objGpet.sinogram_size.nSinogramPlanes = numRings;
        objGpet.sinogram_size.matrixSize = [objGpet.sinogram_size.nRadialBins objGpet.sinogram_size.nAnglesBins objGpet.sinogram_size.nRings];
    else
        % 3d sinogram:
        objGpet.sinogram_size.nRings = numRings;
        objGpet.sinogram_size.maxRingDifference = maxRingDifference;
        % Number of planes mashed in each plane of the sinogram:
        objGpet.sinogram_size.numPlanesMashed = [];
        objGpet.sinogram_size.minRingDiffs = [];
        objGpet.sinogram_size.maxRingDiffs = [];
        objGpet.sinogram_size.nPlanesPerSeg = [];
        % Number of planes in odd and even segments:
        numPlanesOdd = floor(objGpet.sinogram_size.span/2);
        numPlanesEven = ceil(objGpet.sinogram_size.span/2);
        % Ahora tengo que determinar la cantidad de segmentos y las diferencias
        % minimas y maximas en cada uno de ellos. Se que las diferencias maximas y
        % minimas del segmento cero es +-span/2:
        objGpet.sinogram_size.minRingDiffs = -floor(objGpet.sinogram_size.span/2);
        objGpet.sinogram_size.maxRingDiffs = floor(objGpet.sinogram_size.span/2);
        objGpet.sinogram_size.nSeg = 1;
        % Empiezo a ir agregando los segmentos hasta llegar numRings o a la máxima
        % diferencia entra anillos:
        while(abs(objGpet.sinogram_size.minRingDiffs(objGpet.sinogram_size.nSeg)) < objGpet.sinogram_size.maxRingDifference)  % El abs es porque voy a estar comparando el segmento con diferencias negativas.
            % Si no llegue a esa condición, tengo un segmento más hacia cada lado
            % primero hacia el positivo y luego negativo:
            objGpet.sinogram_size.nSeg = objGpet.sinogram_size.nSeg+1;
            % Si estoy en el primer segmento a agregar es -1, sino es -2 para ir
            % con el positivo:
            if objGpet.sinogram_size.nSeg == 2
                % Si me paso de la máxima difrencia de anillos la trunco:
                if (objGpet.sinogram_size.maxRingDiffs(objGpet.sinogram_size.nSeg-1) + objGpet.sinogram_size.span ) <= objGpet.sinogram_size.maxRingDifference
                    objGpet.sinogram_size.minRingDiffs(objGpet.sinogram_size.nSeg) = objGpet.sinogram_size.minRingDiffs(objGpet.sinogram_size.nSeg-1) + objGpet.sinogram_size.span;
                    objGpet.sinogram_size.maxRingDiffs(objGpet.sinogram_size.nSeg) = objGpet.sinogram_size.maxRingDiffs(objGpet.sinogram_size.nSeg-1) + objGpet.sinogram_size.span;
                else
                    objGpet.sinogram_size.minRingDiffs(objGpet.sinogram_size.nSeg) = objGpet.sinogram_size.minRingDiffs(objGpet.sinogram_size.nSeg-1) + objGpet.sinogram_size.span;
                    objGpet.sinogram_size.maxRingDiffs(objGpet.sinogram_size.nSeg) = objGpet.sinogram_size.maxRingDifference;
                end
            else
                % Si me paso de la máxima difrencia de anillos la trunco:
                if (objGpet.sinogram_size.maxRingDiffs(objGpet.sinogram_size.nSeg-2) + objGpet.sinogram_size.span ) <= objGpet.sinogram_size.maxRingDifference
                    objGpet.sinogram_size.minRingDiffs(objGpet.sinogram_size.nSeg) = objGpet.sinogram_size.minRingDiffs(objGpet.sinogram_size.nSeg-2) + objGpet.sinogram_size.span;
                    objGpet.sinogram_size.maxRingDiffs(objGpet.sinogram_size.nSeg) = objGpet.sinogram_size.maxRingDiffs(objGpet.sinogram_size.nSeg-2) + objGpet.sinogram_size.span;
                else
                    objGpet.sinogram_size.minRingDiffs(objGpet.sinogram_size.nSeg) = objGpet.sinogram_size.minRingDiffs(objGpet.sinogram_size.nSeg-2) + objGpet.sinogram_size.span;
                    objGpet.sinogram_size.maxRingDiffs(objGpet.sinogram_size.nSeg) = objGpet.sinogram_size.maxRingDifference;
                end
            end
            % Ahora hacia el lado de las diferencias negativas:
            objGpet.sinogram_size.nSeg = objGpet.sinogram_size.nSeg+1;
            if (abs(objGpet.sinogram_size.minRingDiffs(objGpet.sinogram_size.nSeg-2) - objGpet.sinogram_size.span)) <= objGpet.sinogram_size.maxRingDifference
                objGpet.sinogram_size.minRingDiffs(objGpet.sinogram_size.nSeg) = objGpet.sinogram_size.minRingDiffs(objGpet.sinogram_size.nSeg-2) - objGpet.sinogram_size.span;  % Acá siempre debo ir -2 no tengo problema con el primero.
                objGpet.sinogram_size.maxRingDiffs(objGpet.sinogram_size.nSeg) = objGpet.sinogram_size.maxRingDiffs(objGpet.sinogram_size.nSeg-2) - objGpet.sinogram_size.span;
            else
                objGpet.sinogram_size.minRingDiffs(objGpet.sinogram_size.nSeg) = -objGpet.sinogram_size.maxRingDifference;  % Acá siempre debo ir -2 no tengo problema con el primero.
                objGpet.sinogram_size.maxRingDiffs(objGpet.sinogram_size.nSeg) = objGpet.sinogram_size.maxRingDiffs(objGpet.sinogram_size.nSeg-2) - objGpet.sinogram_size.span;
            end
        end

        % Ahora determino la cantidad de sinogramas por segmentos, recorriendo cada
        % segmento:
        objGpet.sinogram_size.nPlanesPerSeg = zeros(1,objGpet.sinogram_size.nSeg);

        for segment = 1 : objGpet.sinogram_size.nSeg
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
                meanZ1 = 0; % mean z1 values in rings, averages all the rings combinations for this plane.
                meanZ2 = 0; % mean z2 values in rings, averages all the rings combinations for this plane.
                for z2 = 1 : numRings
                    % Ahora voy avanzando en los sinogramas correspondientes,
                    % disminuyendo z1 y aumentnado z2 hasta que la diferencia entre
                    % anillos llegue a maxRingDiff.
                    if ((z1_aux-z2)<=objGpet.sinogram_size.maxRingDiffs(segment))&&((z1_aux-z2)>=objGpet.sinogram_size.minRingDiffs(segment))
                        % Me asguro que esté dentro del tamaño del michelograma:
                        if(z1_aux>0)&&(z2>0)&&(z1_aux<=numRings)&&(z2<=numRings)
                            numSinosZ1inSegment = numSinosZ1inSegment + 1;
                            meanZ1 = meanZ1 + z1_aux;
                            meanZ2 = meanZ2 + z2;
                        end
                    end
                    % Pase esta combinación de (z1,z2), paso a la próxima:
                    z1_aux = z1_aux - 1;
                end
                if(numSinosZ1inSegment>0)
                    meanZ1 = meanZ1./numSinosZ1inSegment;
                    meanZ2 = meanZ2./numSinosZ1inSegment;
                    objGpet.sinogram_size.numPlanesMashed = [objGpet.sinogram_size.numPlanesMashed numSinosZ1inSegment];
                    numSinosThisSegment = numSinosThisSegment + 1;
                    objGpet.sinogram_size.segments(numSinosThisSegment).z1(numSinosThisSegment) = meanZ1;
                    objGpet.sinogram_size.segments(numSinosThisSegment).z2(numSinosThisSegment) = meanZ2;
                end
            end
            % Guardo la cantidad de segmentos:
            objGpet.sinogram_size.nPlanesPerSeg(segment) = numSinosThisSegment;
        end
        objGpet.sinogram_size.nSinogramPlanes = sum(objGpet.sinogram_size.nPlanesPerSeg);
        % Cast to int32 to be used in the mex file:
        objGpet.sinogram_size.maxRingDiffs = int32(objGpet.sinogram_size.maxRingDiffs);
        objGpet.sinogram_size.minRingDiffs = int32(objGpet.sinogram_size.minRingDiffs);
        objGpet.sinogram_size.nPlanesPerSeg = int32(objGpet.sinogram_size.nPlanesPerSeg);
    end
    % subsets:
    objGpet.osem_subsets(objGpet.nSubsets, objGpet.sinogram_size.nAnglesBins);
    % matrix size:
    objGpet.sinogram_size.matrixSize = [objGpet.sinogram_size.nRadialBins objGpet.sinogram_size.nAnglesBins objGpet.sinogram_size.nSinogramPlanes];

end