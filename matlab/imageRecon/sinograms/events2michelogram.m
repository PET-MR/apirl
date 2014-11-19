%  *********************************************************************
%  Proyecto AR-PET. Comisión Nacional de Energía Atómica.
%  Autor: Martín Belzunce. UTN-FRBA.
%  Fecha de Creación: 21/05/2012
%  *********************************************************************

% La función recibe como parámetros todos los eventos en coincidencia
% detectados a través de una matriz que en cada fila contiene la posición
% de los dos eventos en coincidencia: [X1 Y1 Z1 X2 Y2 Z2]; y el tamaño del
% Michelograma en una matriz
function michelograma = Events2Michelogram(MatrizCoincidencias, structSizeMichelogram)

% Creo el michelograma.
michelograma = single(zeros(structSizeMichelogram.numTheta, structSizeMichelogram.numR, structSizeMichelogram.numZ, structSizeMichelogram.numZ));

%  structSizeMichelogram.numTheta : cantidad de ángulos de las proyecciones.
% 
%  structSizeMichelogram.numR: cantidad de desplazamientos de las proyecciones.
%
%  structSizeMichelogram.numZ: número de slices o anillos.
%
%  structSizeMichelogram.maxRingDiffs: máxima diferencia entre slices o

% Necesito generar los theta y luego los r, los z se mantienen los de los
% eventos:
% Ahora lleno los sinogramas. Para esto convierto el par de
% eventos (X1,Y1,Z1) y (X2,Y2,Z2) en lors del tipo (Thita,r,z).
% El ángulo thita está determinado por atan((Y1-Y2)/(X1-X2))+90
theta = atand((MatrizCoincidencias(:,2)-MatrizCoincidencias(:,5))./(MatrizCoincidencias(:,1)-MatrizCoincidencias(:,4))) + 90;
% El offset r lo puedo obtener reemplazando (x,y) con alguno de
% los dos puntos en la ecuación: r=x*cos(thita)+y*sin(thita)-
r_sino = cosd(theta).*MatrizCoincidencias(:,1) + sind(theta).*MatrizCoincidencias(:,2);
indicesFueraFov = (abs(r_sino) > structSizeMichelogram.rFov_mm);
r_sino(indicesFueraFov) = [];
theta(indicesFueraFov) = [];
MatrizCoincidencias(indicesFueraFov,:) = [];
% Ahora cumulo todo en el michelograma:
michelograma = hist5([theta r_sino MatrizCoincidencias(:,3) MatrizCoincidencias(:,6)], {structSizeMichelogram.thetaValues_deg, structSizeMichelogram.rValues_mm, ...
    structSizeMichelogram.zValues_mm, structSizeMichelogram.zValues_mm});
