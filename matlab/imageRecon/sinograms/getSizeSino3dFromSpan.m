%  *********************************************************************
%  Proyecto AR-PET. Comisión Nacional de Energía Atómica.
%  Autor: Martín Belzunce. UTN-FRBA.
%  Fecha de Creación: 23/08/2011
%  *********************************************************************
%  function structSizeSino3D = getSizeSino3Dstruct(numTheta, numR, numZ, rFov, zFov, sinogramsPerSegment, minRingDiff, maxRingDiff)
% 
%  Función que genera una estructura con la información del tamaño de un
%  sinograma de 3 dimensiones, tanto el tamaño en bines como su
%  representación dentro del campo de visión. El sinograma3d es un
%  michelograma reducido con polar mashing, por lo que necesita la cantidad
%  de segmentos (dada por el tamaño del vector sinogramsPerSegment), los
%  sinogramas por segmento sinogramsPerSegment, y las mínimas y máximas
%  diferencias entre anillos para ellos.
%  Esta estructura se utiliza como parámetro en las funciones que
%  manejan sinogramas, y contiene los siguientes campos:
%  
%  structSizeSino3D.numTheta : cantidad de ángulos de las proyecciones.
% 
%  structSizeSino3D.numR: cantidad de desplazamientos de las proyecciones.
%
%  structSizeSino3D.numZ: número de slices o anillos. Será la cantidad de
%  sinogramas 2d.
% 
%  structSizeSino3D.numSegments: número de slices o anillos. Será la cantidad de
%  sinogramas 2d.
% 
%  structSizeSino3D.maxRingDiffs: número de slices o anillos. Será la cantidad de
%  sinogramas 2d.
% 
%  structSizeSino3D.rFov: radio del Field of View 
%
%  structSizeSino3D.zFov: largo del FoV.
%
%  structSizeSino3D.rValues_mm: valores de la variable r para cada
%  proyección. Es un vector con numR elementos.
%
%  structSizeSino3D.thetaValues_deg: valores de la variable theta para cada
%  proyección. Es un vector con numTheta elementos.
%
%  structSizeSino3D.zValues_mm: valores de la variable z para cada
%  proyección. Es un vector con numZ elementos.
%
%  structSizeSino3D.sinogramsPerSegment
%
%  structSizeSino3D.minRingDiff
%
%  structSizeSino3D.maxRingDiff
%
%  Recibe los valores de cada uno de esos campos como parámetro, y devuelve
%  la estructura en si.
%
%  Ejemplo:
%   structSizeSino3D = getSizeSino3Dstruct(192, 192, 24, 300, 200, sinogramsPerSegment, minRingDiff, maxRingDiff)


function structSizeSino3D = getSizeSino3dFromSpan(numTheta, numR, numZ, rFov, zFov, span, maxAbsRingDiff)

% Genero un vector con los valores de r:
deltaR = (2*rFov) / numR;
rValues_mm = -(rFov - deltaR/2) : deltaR : (rFov - deltaR/2);
% Otro con los de Theta:
deltaTheta = 180 / numTheta;
thetaValues_deg = (deltaTheta/2) : deltaTheta : (180 - deltaTheta/2);
% Por último, uno con los de Z:
deltaZ = zFov / numZ;
zValues_mm = -(zFov/2 - deltaZ/2) : deltaZ : (zFov/2 - deltaZ/2);

% Ahora determino toda la estructura del sinograma 3D. El span me indica,
% la cantidad de sinogramas que tengo en las combinaciones pares e impares
% del axial mashing.
numSinosImpar = floor(span/2);
numSinosPar = ceil(span/2);
% Ahora tengo que determinar la cantidad de segmentos y las diferencias
% minimas y maximas en cada uno de ellos. Se que las diferencias maximas y
% minimas del segmento cero es +-span/2:
minRingDiffs = -floor(span/2);
maxRingDiffs = floor(span/2);
numSegments = 1;
% Empiezo a ir agregando los segmentos hasta llegar numRings o a la máxima
% diferencia entra anillos:
while(abs(minRingDiffs(numSegments)) < maxAbsRingDiff)  % El abs es porque voy a estar comparando el segmento con diferencias negativas.
    % Si no llegue a esa condición, tengo un segmento más hacia cada lado
    % primero hacia el positivo y luego negativo:
    numSegments = numSegments+1;
    % Si estoy en el primer segmento a agregar es -1, sino es -2 para ir
    % con el positivo:
    if numSegments == 2
        minRingDiffs(numSegments) = minRingDiffs(numSegments-1) + span;
        maxRingDiffs(numSegments) = maxRingDiffs(numSegments-1) + span;
    else
        % Si me paso de la máxima difrencia de anillos la trunco:
        if (maxRingDiffs(numSegments-2) + span) <= maxAbsRingDiff
            minRingDiffs(numSegments) = minRingDiffs(numSegments-2) + span;
            maxRingDiffs(numSegments) = maxRingDiffs(numSegments-2) + span;
        else
            minRingDiffs(numSegments) = minRingDiffs(numSegments-2) + span;
            maxRingDiffs(numSegments) = maxAbsRingDiff;
        end
    end
    % Ahora hacia el lado de las diferencias negativas:
    numSegments = numSegments+1;
    if (abs(minRingDiffs(numSegments-2) - span)) <= maxAbsRingDiff
        minRingDiffs(numSegments) = minRingDiffs(numSegments-2) - span;  % Acá siempre debo ir -2 no tengo problema con el primero.
        maxRingDiffs(numSegments) = maxRingDiffs(numSegments-2) - span;  
    else
        minRingDiffs(numSegments) = -maxAbsRingDiff;  % Acá siempre debo ir -2 no tengo problema con el primero.
        maxRingDiffs(numSegments) = maxRingDiffs(numSegments-2) - span;  
    end
end

% Ahora determino la cantidad de sinogramas por segmentos, recorriendo cada
% segmento:
sinogramsPerSegment = zeros(1,numSegments);
numRings = numZ;
for segment = 1 : numSegments
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
            if ((z1_aux-z2)<=maxRingDiffs(segment))&&((z1_aux-z2)>=minRingDiffs(segment))
                % Me asguro que esté dentro del tamaño del michelograma:
                if(z1_aux>0)&&(z2>0)&&(z1_aux<=numRings)&&(z2<=numRings)
                    numSinosZ1inSegment = numSinosZ1inSegment + 1;
                end
            end
            % Pase esta combinación de (z1,z2), paso a la próxima:
            z1_aux = z1_aux - 1;
        end
        if(numSinosZ1inSegment>0)
            numSinosThisSegment = numSinosThisSegment + 1;
        end
    end 
    % Guardo la cantidad de segmentos:
    sinogramsPerSegment(segment) = numSinosThisSegment;
end
structSizeSino3D = getSizeSino3Dstruct(numTheta, numR, numZ, rFov, zFov, sinogramsPerSegment, minRingDiffs, maxRingDiffs);
