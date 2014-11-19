%  *********************************************************************
%  Proyecto AR-PET. Comisión Nacional de Energía Atómica.
%  Autor: Martín Belzunce. 
%  Fecha de Creación: 15/02/2013
%  *********************************************************************
%	       PROCESAMIENTO DE LA MATRIZ DE COINCIDENCIAS PARA LA OBTENCIÓN DE
%	       UN SINOGRAMA 3D
%
% La matriz de coincidencias tiene x1, y1, z1 e1 t1 x2, y2, z2 e2 t2 y se
% guarda un evento por coluna, así se puede ir leyendo eventos
% secuencialmente. Recordar que en matlab la matriz se guarda en memoria
% recorriendo filas y luego columna.
% También recibe como parámetro energyWindow y el widthScatterWindow. El
% primero es un vector con los límites inferior y superior de la ventana de
% energía y el segundo es el ancho de las ventanas de energía de corrección
% de scatter. La mismas están centradas en los límites inferior y superior
% de
function sinogram3D = getSinogram3DFromCoincidenceMatrix(filenameCoincidenceMatrix, outputFilename, structSizeSino3D, energyWindow, widthScatterWindows_TripleWindow)


addpath('/workspaces/Martin/PET/Coincidencia/trunk/matlab/LibreriaCoincidencia')
addpath('/sources/MATLAB/WorkingCopy/ImageRecon');
addpath('/sources/MATLAB/WorkingCopy/ImageProcessing');
addpath('/sources/MATLAB/WorkingCopy/utils');

%% CONSTANTES DEL PROCESAMIENTO
% Cantidad de Eventos Por Lectura:
eventsPerRead = 10^6;
% Filas que ocupa cada evento:
numRowsPerEvent = 10;
% Inicializo el sinograma 2d a partir del tamaño definido en la estructura:
sinogram3D = single(zeros(structSizeSino3D.numTheta,structSizeSino3D.numR, sum(structSizeSino3D.sinogramsPerSegment)));
sinogram3Dscatter = single(zeros(structSizeSino3D.numTheta,structSizeSino3D.numR, sum(structSizeSino3D.sinogramsPerSegment)));
% Indice de filas útiles:
rowEnergy1 = 4;
rowEnergy2 = 9;
%% APERTURA DEL ARCHIVO
fid = fopen(filenameCoincidenceMatrix);
if fid == -1
    disp(sprintf('No se pudo abrir el archivo: %s', filenameCoincidenceMatrix));
    return;
end
%% LECTURA SECUENCIAL DEL ARCHIVO
disp(sprintf('########### Iniciando procesamiento de %s #############', filenameCoincidenceMatrix));
eventosTotales = 0;
while(~feof(fid))
    coincidenceMatrix = fread(fid, [numRowsPerEvent eventsPerRead],'single');
    eventosTotales = eventosTotales + size(coincidenceMatrix,2);
    disp(sprintf('Lectura de %d eventos. En total se han leído: %d eventos.', size(coincidenceMatrix,2),eventosTotales));
    % Hago el filtrado por energía de ventana principal y ventanas de
    % scatter. Para que no haya scatter, ambos eventos deben estar en la
    % ventana principal, y se considera uqe hubo scatter cuando al menos
    % uno de los dos eventos está fuera de la ventana de energía:
    indicesVentanaPrincipal = ((coincidenceMatrix(rowEnergy1,:)>energyWindow(1)) & (coincidenceMatrix(rowEnergy1,:)<energyWindow(2))) & ...
        ((coincidenceMatrix(rowEnergy2,:)>energyWindow(1)) & (coincidenceMatrix(rowEnergy2,:)<energyWindow(2)));
    % Ahora las ventanas de scatter:
    indicesSW1 = ((coincidenceMatrix(rowEnergy1,:)<(energyWindow(1)+widthScatterWindows_TripleWindow/2)) & (coincidenceMatrix(rowEnergy1,:) > (energyWindow(1) - widthScatterWindows_TripleWindow/2))) ...
        | ((coincidenceMatrix(rowEnergy2,:)<(energyWindow(1)+widthScatterWindows_TripleWindow/2)) & (coincidenceMatrix(rowEnergy2,:) > (energyWindow(1) - widthScatterWindows_TripleWindow/2)));
    indicesSW2 = ((coincidenceMatrix(rowEnergy1,:)<(energyWindow(2)+widthScatterWindows_TripleWindow/2)) & (coincidenceMatrix(rowEnergy1,:) > (energyWindow(2) - widthScatterWindows_TripleWindow/2))) ...
        | ((coincidenceMatrix(rowEnergy2,:)<(energyWindow(2)+widthScatterWindows_TripleWindow/2)) & (coincidenceMatrix(rowEnergy2,:) > (energyWindow(2) - widthScatterWindows_TripleWindow/2)));
    %% SINOGRAMA 3D
    matrizEventos = [coincidenceMatrix(1:3,indicesVentanaPrincipal)' coincidenceMatrix(6:8,indicesVentanaPrincipal)'];
    matrizEventosScatter = [coincidenceMatrix(1:3,indicesSW1|indicesSW2)' coincidenceMatrix(6:8,indicesSW1|indicesSW2)'];
    % Genero el sinograma3D
    sinogram3D = sinogram3D + Events2Sinogram3D(matrizEventos, structSizeSino3D);
    % Sinograma de scatter
    sinogram3Dscatter = sinogram3Dscatter + Events2Sinogram3D(matrizEventosScatter, structSizeSino3D);
end
disp(sprintf('Porcentaje de Eventos que se eliminaron por estar fuera del Fov: %d', 100*sum(sum(sum(sinogram3D)))/eventosTotales));
% Sinograma corregido:
sinogram3Dcorregido = sinogram3D - sinogram3Dscatter./(2*widthScatterWindows_TripleWindow).*(energyWindow(2)-energyWindow(1));
% Si alguno queda negativo lo fuerzo a cero:
sinogram3Dcorregido(sinogram3Dcorregido<0) = 0;
%% CIERRO ARCHIVO
fclose(fid);
%% GUARDO EL SINOGRAMA EN INTERFILE
% Sinograma principal:
interfileWriteSino(sinogram3D, outputFilename, structSizeSino3D.sinogramsPerSegment, structSizeSino3D.minRingDiff, structSizeSino3D.maxRingDiff);
% Sinograma de scatter:
interfileWriteSino(sinogram3Dscatter, [outputFilename '_scatter'], structSizeSino3D.sinogramsPerSegment, structSizeSino3D.minRingDiff, structSizeSino3D.maxRingDiff);
% Sinograma corregido:
interfileWriteSino(sinogram3Dcorregido, [outputFilename '_scatterCorrected'], structSizeSino3D.sinogramsPerSegment, structSizeSino3D.minRingDiff, structSizeSino3D.maxRingDiff);