%  *********************************************************************
%  Proyecto AR-PET. Comisión Nacional de Energía Atómica.
%  Autor: Martín Belzunce. UTN-FRBA.
%  Fecha de Creación: 27/12/2011
%  *********************************************************************
%

% Agrego paths para simulación de fantoma y escritura interfile:
addpath('/sources/MATLAB/WorkingCopy/ImageRecon','/sources/MATLAB/WorkingCopy/ImageProcessing');

imagenFantoma = phantom;
sinograma2D = SimulacionSino(imagenFantoma,192,192,1000000,'CantR');

interfilewrite(single(imagenFantoma), '../../samples/reconSinogram2d/image2dSheppLogan');
interfileWriteSino(single(sinograma2D), '../../samples/reconSinogram2d/sino2dSheppLogan');
