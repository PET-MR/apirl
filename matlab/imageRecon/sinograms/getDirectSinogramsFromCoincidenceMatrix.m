%  *********************************************************************
%  Proyecto AR-PET. Comisión Nacional de Energía Atómica.
%  Autor: Martín Belzunce. 
%  Fecha de Creación: 15/02/2013
%  *********************************************************************
%	       PROCESAMIENTO DE LA MATRIZ DE COINCIDENCIAS PARA LA OBTENCIÓN DE
%	       SINOGRAMAS DIRECTOS 2D
%
% La matriz de coincidencias tiene x1, y1, z1 e1 t1 x2, y2, z2 e2 t2 y se
% guarda un evento por coluna, así se puede ir leyendo eventos
% secuencialmente. Recordar que en matlab la matriz se guarda en memoria
% recorriendo filas y luego columna.
% Se descartan todos los eventos con una diferencia entre anillos mayor a
% 1.
function sinograms2D = getDirectSinogramsFromCoincidenceMatrix(filenameCoincidenceMatrix, outputFilename, structSizeSino2D, energyWindow, widthScatterWindows_TripleWindow)


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
sinograms2D = single(zeros(structSizeSino2D.numR, structSizeSino2D.numTheta, ...
    structSizeSino2D.numZ));
sinograms2Dscatter = single(zeros(structSizeSino2D.numR, structSizeSino2D.numTheta, ...
    structSizeSino2D.numZ));
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
eventosDirectosTotales = 0;
while(~feof(fid))
    coincidenceMatrix = fread(fid, [numRowsPerEvent eventsPerRead],'single');
    eventosTotales = eventosTotales + size(coincidenceMatrix,2);
    disp(sprintf('Lectura de %d eventos. En total se han leído: %d eventos.', size(coincidenceMatrix,2),eventosTotales));
    % Filtro los que no son directos:
    indicesNoDirectos = abs(coincidenceMatrix(3,:) - coincidenceMatrix(8,:)) > (structSizeSino2D.zValues_mm(2)-structSizeSino2D.zValues_mm(1));
    coincidenceMatrix(:,indicesNoDirectos) = [];
    eventosDirectosTotales = eventosDirectosTotales + size(coincidenceMatrix,2);
    disp(sprintf('De los cuales quedan: %d eventos. Eventos directos totales: %d', size(coincidenceMatrix,2),eventosDirectosTotales));
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
    matrizEventos = [coincidenceMatrix(1:3,indicesVentanaPrincipal)' coincidenceMatrix(6:8,indicesVentanaPrincipal)'];
    matrizEventosScatter = [coincidenceMatrix(1:3,indicesSW1|indicesSW2)' coincidenceMatrix(6:8,indicesSW1|indicesSW2)'];
    %% SINOGRAMAS 2D
    % Ahora lleno los sinogramas. Para esto convierto el par de
    % eventos (X1,Y1,Z1) y (X2,Y2,Z2) en lors del tipo (Thita,r,z).
    % El ángulo thita está determinado por atan((Y1-Y2)/(X1-X2))+90
    theta = atand((matrizEventos(:,2)-matrizEventos(:,5))./(matrizEventos(:,1)-matrizEventos(:,4))) + 90;
    % El offset r lo puedo obtener reemplazando (x,y) con alguno de
    % los dos puntos en la ecuación: r=x*cos(thita)+y*sin(thita)-
    r_sino = cosd(theta).*matrizEventos(:,1) + sind(theta).*matrizEventos(:,2);
    % Si quedan fuera del fov, los elimino:
    indicesFueraFov = abs(r_sino)>structSizeSino2D.rFov_mm;
    r_sino(indicesFueraFov) = [];
    theta(indicesFueraFov) = [];
    matrizEventos(indicesFueraFov,:) = [];
    % Acumulo todo en el array de sinogramas 2D utilizando la
    % función creada hist4.
    sinograms2D = sinograms2D + hist4([r_sino theta matrizEventos(:,3)], ...
        {structSizeSino2D.rValues_mm structSizeSino2D.thetaValues_deg structSizeSino2D.zValues_mm});

    % Ahora lleno los sinogramas. Para esto convierto el par de
    % eventos (X1,Y1,Z1) y (X2,Y2,Z2) en lors del tipo (Thita,r,z).
    % El ángulo thita está determinado por atan((Y1-Y2)/(X1-X2))+90
    theta = atand((matrizEventosScatter(:,2)-matrizEventosScatter(:,5))./(matrizEventosScatter(:,1)-matrizEventosScatter(:,4))) + 90;
    % El offset r lo puedo obtener reemplazando (x,y) con alguno de
    % los dos puntos en la ecuación: r=x*cos(thita)+y*sin(thita)-
    r_sino = cosd(theta).*matrizEventosScatter(:,1) + sind(theta).*matrizEventosScatter(:,2);
    % Si quedan fuera del fov, los elimino:
    indicesFueraFov = abs(r_sino)>structSizeSino2D.rFov_mm;
    r_sino(indicesFueraFov) = [];
    theta(indicesFueraFov) = [];
    matrizEventosScatter(indicesFueraFov,:) = [];
    % Acumulo todo en el array de sinogramas 2D utilizando la
    % función creada hist4.
    sinograms2Dscatter = sinograms2Dscatter + hist4([r_sino theta matrizEventosScatter(:,3)], ...
        {structSizeSino2D.rValues_mm structSizeSino2D.thetaValues_deg structSizeSino2D.zValues_mm});
    
end

disp(sprintf('Porcentaje de Eventos que se eliminaron por estar fuera del Fov: %d', 100*sum(sum(sinograms2D))/eventosDirectosTotales));
% Sinograma corregido:
sinograms2Dcorregido = sinograms2D - sinograms2Dscatter./(2*widthScatterWindows_TripleWindow).*(energyWindow(2)-energyWindow(1));
% Si alguno queda negativo lo fuerzo a cero:
sinograms2Dcorregido(sinograms2Dcorregido<0) = 0;
%% CIERRO ARCHIVO
fclose(fid);
%% GUARDO EL SINOGRAMA EN INTERFILE
% Guardo todos los sinogramas:
interfileWriteSino(sinograms2D, outputFilename);
interfileWriteSino(sinograms2D, [outputFilename '_scatter']);
interfileWriteSino(sinograms2D, [outputFilename '_scatterCorrected']);