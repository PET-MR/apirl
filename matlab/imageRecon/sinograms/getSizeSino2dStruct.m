%  *********************************************************************
%  Proyecto AR-PET. Comisión Nacional de Energía Atómica.
%  Autor: Martín Belzunce. UTN-FRBA.
%  Fecha de Creación: 23/08/2011
%  *********************************************************************
%  function structSimu = getSimuStruct(path, nameSimu, outputName,
%  numSplits, filesPerSplit)
% 
%  Función que genera una estructura con la información del tamaño de un
%  sinograma de 2 dimensiones, tanto el tamaño en bines como su
%  representación dentro del campo de visión.
%  Esta estructura se utiliza como parámetro en las funciones que
%  manejan sinogramas, y contiene los siguientes campos:
% 
%  structSizeSino2D.numR: cantidad de desplazamientos de las proyecciones.
%  
%  structSizeSino2D.numTheta : cantidad de ángulos de las proyecciones.
%
%  structSizeSino2D.numZ: número de slices o anillos. Será la cantidad de
%  sinogramas 2d.
% 
% 
%  structSizeSino2D.rFov: radio del Field of View 
%
%  structSizeSino2D.zFov: largo del FoV.
%
%  structSizeSino2D.rValues_mm: valores de la variable r para cada
%  proyección. Es un vector con numR elementos.
%
%  structSizeSino2D.thetaValues_deg: valores de la variable theta para cada
%  proyección. Es un vector con numTheta elementos.
%
%  structSizeSino2D.zValues_mm: valores de la variable z para cada
%  proyección. Es un vector con numZ elementos.
%
%  Recibe los valores de cada uno de esos campos como parámetro, y devuelve
%  la estructura en si.
%
%  Ejemplo:
%   structSizeSino3D = getSizeSino2dStruct(192, 192, 24, 300, 200)


function structSizeSino2D = getSizeSino2dStruct(numR, numTheta, numZ, rFov, zFov)

% Genero un vector con los valores de r:
deltaR = (2*rFov) / numR;
rValues_mm = -(rFov - deltaR/2) : deltaR : (rFov - deltaR/2);
% Otro con los de Theta:
deltaTheta = 180 / numTheta;
thetaValues_deg = (deltaTheta/2) : deltaTheta : (180 - deltaTheta/2);
% Por último, uno con los de Z:
deltaZ = zFov / numZ;
zValues_mm = -(zFov/2 - deltaZ/2) : deltaZ : (zFov/2 - deltaZ/2);

structSizeSino2D = struct('numTheta', numTheta, 'numR', numR, 'numZ', numZ, 'rFov_mm', rFov,...
    'zFov_mm', zFov, 'rValues_mm', rValues_mm, 'thetaValues_deg', thetaValues_deg, ...
    'zValues_mm', zValues_mm);
