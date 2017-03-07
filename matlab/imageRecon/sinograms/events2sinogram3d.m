%  *********************************************************************
%  Proyecto AR-PET. Comisión Nacional de Energía Atómica.
%  Autor: Martín Belzunce. UTN-FRBA.
%  Fecha de Creación: 21/05/2012
%  *********************************************************************

% La función recibe como parámetros todos los eventos en coincidencia
% detectados a través de una matriz que en cada fila contiene la posición
% de los dos eventos en coincidencia: [X1 Y1 Z1 X2 Y2 Z2]; y el tamaño del
% Sinograma3D en la estructura que se genera con getSizeSino3Dstruct. Esta
% función además utiliza la función ReduceMichelgram.
function sinogram3D = Events2Sinogram3D(MatrizCoincidencias, structSizeSino3D)
addpath('/sources/MATLAB/WorkingCopy/ImageRecon');
addpath('/sources/MATLAB/WorkingCopy/utils');
% Creo el sinograma 3d.
sinogram3D = single(zeros(structSizeSino3D.numR, structSizeSino3D.numTheta, sum(structSizeSino3D.sinogramsPerSegment)));


% Necesito generar los theta y luego los r, los z se mantienen los de los
% eventos:
% Ahora lleno los sinogramas. Para esto convierto el par de
% eventos (X1,Y1,Z1) y (X2,Y2,Z2) en lors del tipo (Thita,r,z).
% El ángulo thita está determinado por atan((Y1-Y2)/(X1-X2))+90
theta = atan2d((MatrizCoincidencias(:,2)-MatrizCoincidencias(:,5)),(MatrizCoincidencias(:,1)-MatrizCoincidencias(:,4)));
theta0_180 = (theta >= 0) && (theta < 180);
z1 = MatrizCoincidencias(:,3);
z2 = MatrizCoincidencias(:,6);
% Swap depending on phi value:
z1(~theta0_180) = MatrizCoincidencias(~theta0_180,6);
z2(~theta0_180) = MatrizCoincidencias(~theta0_180,3);
theta(~theta0_180) = 180 - theta(~theta0_180);
theta = theta + 90;
% El offset r lo puedo obtener reemplazando (x,y) con alguno de
% los dos puntos en la ecuación: r=x*cos(thita)+y*sin(thita)-
r_sino = cosd(theta).*MatrizCoincidencias(:,1) + sind(theta).*MatrizCoincidencias(:,2);
% Si esta fuera del FOV lo elimino, primero transversal:
indicesFueraFov = (abs(r_sino) > structSizeSino3D.rFov_mm);
r_sino(indicesFueraFov) = [];
theta(indicesFueraFov) = [];
MatrizCoincidencias(indicesFueraFov,:) = [];
% También hago lo mismo con el zfov:
indicesFueraZfov = (MatrizCoincidencias(:,3) < -structSizeSino3D.zFov_mm/2) | (MatrizCoincidencias(:,3) > structSizeSino3D.zFov_mm/2) | ...
    (MatrizCoincidencias(:,6) < -structSizeSino3D.zFov_mm/2) | (MatrizCoincidencias(:,6) > structSizeSino3D.zFov_mm/2);
MatrizCoincidencias(indicesFueraZfov,:) = [];
r_sino(indicesFueraZfov) = [];
theta(indicesFueraZfov) = [];
% Primero genero un michelograma y luego reduzco a un sinograma:
michelograma = hist5([r_sino theta z1 z2], {structSizeSino3D.rValues_mm, structSizeSino3D.thetaValues_deg,  ...
    structSizeSino3D.zValues_mm, structSizeSino3D.zValues_mm});
sinogram3D = reduceMichelogram(michelograma, structSizeSino3D.sinogramsPerSegment, structSizeSino3D.minRingDiff, structSizeSino3D.maxRingDiff);