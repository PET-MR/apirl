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
%  structSizeSino3D.maxAbsRingDiff
%
%  Recibe los valores de cada uno de esos campos como parámetro, y devuelve
%  la estructura en si. Si no se recibe maxAbsRingDiff, se considera numZ
%
%  Ejemplo:
%   structSizeSino3D = getSizeSino3Dstruct(192, 192, 24, 300, 200, sinogramsPerSegment, minRingDiff, maxRingDiff)


function structSizeSino3D = getSizeSino3Dstruct(numTheta, numR, numZ, rFov, zFov, sinogramsPerSegment, minRingDiff, maxRingDiff, maxAbsRingDiff)

if nargin < 9
    maxAbsRingDiff = numZ;
end

% Genero un vector con los valores de r:
deltaR = (2*rFov) / numR;
rValues_mm = -(rFov - deltaR/2) : deltaR : (rFov - deltaR/2);
% Otro con los de Theta:
deltaTheta = 180 / numTheta;
thetaValues_deg = (deltaTheta/2) : deltaTheta : (180 - deltaTheta/2);
% Por último, uno con los de Z:
deltaZ = zFov / numZ;
zValues_mm = -(zFov/2 - deltaZ/2) : deltaZ : (zFov/2 - deltaZ/2);
% Cantidad de segmentos:
numSegments = numel(sinogramsPerSegment);

structSizeSino3D = struct('numTheta', numTheta, 'numR', numR, 'numZ', numZ, 'numSegments', numSegments, 'rFov_mm', rFov,...
    'zFov_mm', zFov, 'rValues_mm', rValues_mm, 'thetaValues_deg', thetaValues_deg, ...
    'zValues_mm', zValues_mm, 'sinogramsPerSegment', sinogramsPerSegment,'minRingDiff', minRingDiff, 'maxRingDiff', maxRingDiff, 'maxAbsRingDiff', maxAbsRingDiff);
