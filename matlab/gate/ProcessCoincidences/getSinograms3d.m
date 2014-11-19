%  *********************************************************************
%  Proyecto AR-PET. Comisión Nacional de Energía Atómica.
%  Autor: Martín Belzunce. UTN-FRBA.
%  Fecha de Creación: 23/08/2011
%  *********************************************************************
%	            GENERACIÓN DE SINOGRAMA 2D A PARTIR DE SIMULACIÓN GATE
%  function sinogram3D =  getSinograms3Dideal(outputPath, structSimu, structSizeSino3D, energyWindow, widthScatterWindows_TripleWindow, graficarOnline)
%
%  Genera sinogramas 3d sin degradar resolución ni incorporar zonas
%  muertas.

function [sinogram3D, TiempoSimulacion] = getSinograms3Dideal(outputFilename, structSimu, structSizeSino3D, energyWindow, widthScatterWindows_TripleWindow, graficarOnline)

%% OTRAS FUNCIONES QUE SE UTILIZAN
% Agrego el path donde tengo algunas funciones de uso general. En este caso
% hist4.
addpath('/sources/MATLAB/FuncionesGenerales');
addpath('/sources/MATLAB/VersionesFinales');
addpath('/sources/MATLAB/WorkingCopy/ImageProcessing');
addpath('/sources/MATLAB/WorkingCopy/ImageRecon');

%% CONSTANTES GEOMÉTRICAS DEL CRISTAL
cantCabezales = 6;
cristalSizeY_mm = 304.8;   % Ancho del cristal 304.8mm.
cristalSizeX_mm = 406.4;   % Ancho del cristal 406.4mm.
espesorCristal_mm = 25.4;
ladoHexagonoScanner_mm = 415.69;
distCentroCabezal_mm = 360;
%%  VARIABLES PARA GENERACIÓN DE SINOGRAMAS 3D
sinogram3D = single(zeros(structSizeSino3D.numTheta,structSizeSino3D.numR, sum(structSizeSino3D.sinogramsPerSegment)));
sinogram3Dscatter = single(zeros(structSizeSino3D.numTheta,structSizeSino3D.numR, sum(structSizeSino3D.sinogramsPerSegment)));
%% VARIABLES AUXILIARES Y PARA VISUALIZACIÓN DE RESULTADOS PARCIALES
% Valores de cada coordenada dentro de la simulación:
valoresX = -400:2:400;                   % Valores posibles de emisión de positrones en la variable Z (O sea, subre todo el FOV).
valoresY = -500:2:500;                   % Valores posibles de emisión de positrones en la variable Z (O sea, subre todo el FOV).
valoresZ = -158:2:158;                   % Valores posibles de emisión de positrones en la variable Z (O sea, subre todo el FOV).
% Conjunto de valores utilziados apra generar histogramas XY:
valoresYX = {valoresY valoresX};           % Cell Array con los valores posibles de las variables x e y dentro del sistema.
% Matricez para imágenes de detección en el plano XY de todo el scanner:
planoDetXY = cell(1, cantCabezales);
planoDetProyectadoDetXY = zeros(numel(valoresYX{1}), numel(valoresYX{2}));
% Imágenes para posiciones de detección en cada cabezal:
valoresYXcabezal = {-(cristalSizeY_mm/2):1:(cristalSizeY_mm/2) -(cristalSizeX_mm/2):1:(cristalSizeX_mm/2)};
planoXYcabezal = cell(1, cantCabezales);
for i = 1 : cantCabezales
    planoDetXY{i} = zeros(numel(valoresYX{1}), numel(valoresYX{2}));
    planoXYcabezal{i} = zeros(numel(valoresYXcabezal{1}), numel(valoresYXcabezal{2}));
end

%% OTRAS VARIABLES A INICIALIZAR
% Variables relacionadas con el conteo de eventos:
eventosTotales = 0;
eventosTrues = 0;
eventosRandoms = 0;
eventosConScatter = 0;
eventosSinCompton = 0;
eventosTotalesEnVentana = 0;
eventosTruesConScatter = 0;
eventosTotalesCabezal = zeros(cantCabezales,1);
CombinacionCabezales = zeros(cantCabezales);
TiempoSplitsAnteriores = 0; % Variable que acumula los tiempos de adquisición de los aplits ya procesados.
TiempoSimulacion = 0;            % Variable que indica el tiempo total de la simulación.

% Variables para histogramas y filtrado de energía:
%  Variables que definen las ventanas de energía, siempre existe una
%  ventana de energía sobre el pico que denominamos A, y es opcional una
%  ventana B en la zona de energía que hay poco compton (aprox 350-350keV).
% Ahora la ventana de energía la saco del structSimu:
% VentanaAsup = 0.580; % Límite superior de la ventana de energía del pico.
% VentanaAinf = 0.430;    % Límite inferior de la ventana de energía en pico.
VentanaAinf = energyWindow(1);
VentanaAsup = energyWindow(2);
%  secundaria B.
CanalesEnergia =  0:0.001:2.0;  % Canales de Energía para el histograma.
HistEnergias = zeros(1,numel(CanalesEnergia));  % Inicialización en cero del Histograma de Energías.
HistEnergiasFilt = zeros(1,numel(CanalesEnergia));  % Inicialización en cero del Histograma de Energías Filtrado.
HistEnergiasSinCompton = zeros(1,numel(CanalesEnergia));  % Inicialización en cero del Histograma de Energías de Eventos Cin Comptons.


%% INICIALIZACIÓN DE VARIABLES A UTILIZAR EN EL PROCESAMIENTO DEL COINCIDENCE
eventosTotales = 0;

TiempoSplitsAnteriores = 0; % Variable que acumula los tiempos de adquisición de los aplits ya procesados.
TiempoSimulacion = 0;            % Variable que indica el tiempo total de la simulación.

%% PARÁMETROS DEL ARCHIVO DE COINCIDENCIAS DEL GATE
% Índices de columnas en el archivo de salida:
numColumnas = 46;
numColumnasPorSingle = numColumnas / 2;
colEventId1 = 2;
colEventId2 = 2 + numColumnasPorSingle;
colTimeStamp1 = 7;
colTimeStamp2 = 7 + numColumnasPorSingle;
colEnergy1 = 8;
colEnergy2 = 8 + numColumnasPorSingle;
colEmisionX1 = 4;
colEmisionX2 = 4 + numColumnasPorSingle;
colEmisionY1 = 5;
colEmisionY2 = 5 + numColumnasPorSingle;
colEmisionZ1 = 6;
colEmisionZ2 = 6 + numColumnasPorSingle;
colDetectionX1 = 9;
colDetectionX2 = 9 + numColumnasPorSingle;
colDetectionY1 = 10;
colDetectionY2 = 10 + numColumnasPorSingle;
colDetectionZ1 = 11;
colDetectionZ2 = 11 + numColumnasPorSingle;
colVolIdBase1 = 12;
colVolIdBase2 = 12 + numColumnasPorSingle;
colVolIdRsector1 = 13;
colVolIdRsector2 = 13 + numColumnasPorSingle;
colVolIdModule1 = 14;
colVolIdModule2 = 14 + numColumnasPorSingle;
colCompton1 = 18;
colCompton2 = 18 + numColumnasPorSingle;

%% PROCESAMIENTO DE COINCIDENCES
for i = 1 : structSimu.numSplits
    fprintf('################ PROCESANDO SPLIT %d ###################\n',i);
    % Nombre Base del Archivo de Salida para el Split i
    if structSimu.numSplits > 1
        % Si se ha hecho el split de la simulación en varias simulaciones
        % tengo que usar el formato de nombre que le da el splitter.
        NombreBase = sprintf('%s%d%s', structSimu.name,i,structSimu.digiName); 
    else
        % El nombre por default cuando no se usa el spliter es
        % gateSingles.dat
        NombreBase = sprintf('%s%s',structSimu.name,structSimu.digiName); 
    end
    % Nombre del Archivo a abrir para el split i, teniendo en cuenta que es
    % el primero que se ha generado, por lo tanto no tiene adicionales al
    % final.
    NombreArch = sprintf('%s/%s.dat',structSimu.path,NombreBase);
    PrimeraLectura = 1; % Indica que es la primera lectura del split. De esta forma se pueden guardar los tiempos inciales por cada split.
    for j = 1 : structSimu.numFilesPerSplit 
        fprintf('################ PROCESANDO ARCHIVO %d DEL SPLIT %d ###################\n',j,i);
        FID = fopen(NombreArch, 'r');
        if (FID == -1)
            disp('Error al abrir el archivo');
            TiempoSplit = 0;
        else
            fprintf('Iniciando el procesamiento del archivo %s.\n', NombreArch);
            while feof(FID) == 0
                datos = textscan(FID, '%f', 100000 * 46);
                if mod(numel(datos{1}), 46) ~= 0
                    % El archivo no esta entero, puede que se haya cortado
                    % por una finalizacion abrupta de la simulacion.
                    % Elimino la coincidencia incompleta
                    datos{1}(end-(mod(numel(datos{1}),46)-1) : end) =[];
                end
                % Le doy forma de matriz a todos los datos leídos.
                coincidenceMatrix = reshape(datos{1}, 46, numel(datos{1})/46)';
                eventosLeidos = size(coincidenceMatrix,1);

                % Guardo el tiempo inicial del archivo.
                if PrimeraLectura
                    TiempoInicialSplit = coincidenceMatrix(1,colTimeStamp1);
                    PrimeraLectura = 0;
                end
                % Obtengo el tiempo de duración de la simulación hasta el
                % evento procesado en el ciclo actual.
                TiempoSplit = coincidenceMatrix(end,colTimeStamp1) - TiempoInicialSplit;
                TiempoSimulacion = TiempoSplit + TiempoSplitsAnteriores;
                disp('');
                fprintf('Tiempo Inicial Split: %d\n', TiempoInicialSplit);
                fprintf('Tiempo Medición Split: %d\n', TiempoSplit);
                fprintf('Tiempo Total Simulación: %d\n', TiempoSimulacion);
                fprintf('\n\n');
                disp('%%%%%%%%%%% VENTANA DE ENERGÍA %%%%%%%%%%%');
                eventosTotales = eventosTotales + eventosLeidos;
                fprintf('Coincidencias leídas: %d\n', eventosTotales);

                % Si no tengo datos, salgo:
                if isempty(coincidenceMatrix)
                    break;
                end

                % En la cadena de procesamiento lo primero que hago es el
                % filtrado en energía.
                HistEnergias = HistEnergias + hist([coincidenceMatrix(:,colEnergy1); coincidenceMatrix(:,colEnergy2)], CanalesEnergia);
                if(graficarOnline)
                    figure(1);
                    bar(CanalesEnergia, HistEnergias);
                    title('Histograma de Energías');
                end

                % Simplemente necesitamos las coincidencias en ventana de
                % energía. Por lo que aplico el filtro de ventana de energías:
                indicesFuera = (coincidenceMatrix(:,colEnergy1)<(VentanaAinf)) | (coincidenceMatrix(:,colEnergy1)>(VentanaAsup)) | ...
                        (coincidenceMatrix(:,colEnergy2)<(VentanaAinf)) |(coincidenceMatrix(:,colEnergy2)>(VentanaAsup));
                % Me quedo con solo los eventos en ventana
                coincidenceMatrix(indicesFuera,:) = [];
                eventosTotalesEnVentana = eventosTotalesEnVentana + numel(coincidenceMatrix(:,1));
                fprintf('Coincidencias en ventana de energía: %d\n', eventosTotalesEnVentana);
                
                % Hago el filtrado por energía de ventana principal y ventanas de
                % scatter. Para que no haya scatter, ambos eventos deben estar en la
                % ventana principal, y se considera uqe hubo scatter cuando al menos
                % uno de los dos eventos está fuera de la ventana de energía:
                indicesVentanaPrincipal = ((coincidenceMatrix(:,colEnergy1)>energyWindow(1)) & (coincidenceMatrix(:,colEnergy1)<energyWindow(2))) & ...
                    ((coincidenceMatrix(:,colEnergy2)>energyWindow(1)) & (coincidenceMatrix(:,colEnergy2)<energyWindow(2)));
                % Ahora las ventanas de scatter:
                indicesSW1 = ((coincidenceMatrix(:,colEnergy1)<(energyWindow(1)+widthScatterWindows_TripleWindow/2)) & (coincidenceMatrix(:,colEnergy1) > (energyWindow(1) - widthScatterWindows_TripleWindow/2))) ...
                    | ((coincidenceMatrix(:,colEnergy2)<(energyWindow(1)+widthScatterWindows_TripleWindow/2)) & (coincidenceMatrix(:,colEnergy2) > (energyWindow(1) - widthScatterWindows_TripleWindow/2)));
                indicesSW2 = ((coincidenceMatrix(:,colEnergy1)<(energyWindow(2)+widthScatterWindows_TripleWindow/2)) & (coincidenceMatrix(:,colEnergy1) > (energyWindow(2) - widthScatterWindows_TripleWindow/2))) ...
                    | ((coincidenceMatrix(:,colEnergy2)<(energyWindow(2)+widthScatterWindows_TripleWindow/2)) & (coincidenceMatrix(:,colEnergy2) > (energyWindow(2) - widthScatterWindows_TripleWindow/2)));

                % Si no tengo datos, salgo:
                if isempty(coincidenceMatrix(indicesVentanaPrincipal,:))
                    break;
                end

                % Grafico La energía filtrada
                HistEnergiasFilt = HistEnergiasFilt + hist([coincidenceMatrix(indicesVentanaPrincipal,colEnergy1); coincidenceMatrix(indicesVentanaPrincipal,colEnergy2)], CanalesEnergia);
                if(graficarOnline)
                    figure(2);
                    bar(CanalesEnergia, HistEnergiasFilt);
                    title('Histograma de Energías luego de Filtro de Ventana');
                end

                %% SINOGRAMA 3D
                % Genero el sinograma3D con l ventana principal:
                % Proceso la posición de los eventos para generar sinogramas:
                posDet = [coincidenceMatrix(indicesVentanaPrincipal,colDetectionX1) coincidenceMatrix(indicesVentanaPrincipal,colDetectionY1) coincidenceMatrix(indicesVentanaPrincipal,colDetectionZ1) ...
                    coincidenceMatrix(indicesVentanaPrincipal,colDetectionX2) coincidenceMatrix(indicesVentanaPrincipal,colDetectionY2) coincidenceMatrix(indicesVentanaPrincipal,colDetectionZ2)];
                sinogram3D = sinogram3D + Events2Sinogram3D(posDet, structSizeSino3D);
                % Sinograma de scatter, con las ventanas de scatter:
                % Proceso la posición de los eventos para generar sinogramas:
                posDet = [coincidenceMatrix(indicesSW1|indicesSW2,colDetectionX1) coincidenceMatrix(indicesSW1|indicesSW2,colDetectionY1) coincidenceMatrix(indicesSW1|indicesSW2,colDetectionZ1) ...
                    coincidenceMatrix(indicesSW1|indicesSW2,colDetectionX2) coincidenceMatrix(indicesSW1|indicesSW2,colDetectionY2) coincidenceMatrix(indicesSW1|indicesSW2,colDetectionZ2)];
                sinogram3Dscatter = sinogram3Dscatter + Events2Sinogram3D(posDet, structSizeSino3D);
                %% LLENADO DE SINOGRAMAS 2D
                if(graficarOnline)
                    figure(6);
                    imshow(sinogram3D(:,:,round(sum(structSizeSino3D.sinogramsPerSegment)/2))/max(max(sinogram3D(:,:,round(sum(structSizeSino3D.sinogramsPerSegment)/2)))));
                end
                        %% FIN DEL LOOP
            end
            fclose(FID);
        end
        % Nombre del próximo archivo dentro del split (Esto pasa cuando el
        % archivo de salida por split es de más de 1.9GBytes).
        NombreArch = sprintf('%s/%s_%d.dat',structSimu.path,NombreBase,j);
    end
 %% CAMBIO DE SPLIT
    % Fin de los archvos correspondientes a un split, ahora se seguirá con
    % el próximo split si es que hay uno. Actualiza la variable que lleva
    % el tiempo acumulado.
    TiempoSplitsAnteriores = TiempoSplitsAnteriores + TiempoSplit;
end    
TiempoSimulacion = TiempoSplitsAnteriores;
%% CORRECCION DE SCATTER
% Sinograma corregido:
sinogram3Dcorregido = sinogram3D - sinogram3Dscatter./(2*widthScatterWindows_TripleWindow).*(energyWindow(2)-energyWindow(1));
% Si alguno queda negativo lo fuerzo a cero:
sinogram3Dcorregido(sinogram3Dcorregido<0) = 0;

%% GUARDO EL SINOGRAMA EN INTERFILE
% Sinograma principal:
interfileWriteSino(sinogram3D, outputFilename, structSizeSino3D.sinogramsPerSegment, structSizeSino3D.minRingDiff, structSizeSino3D.maxRingDiff);
% Sinograma de scatter:
interfileWriteSino(sinogram3Dscatter, [outputFilename '_scatter'], structSizeSino3D.sinogramsPerSegment, structSizeSino3D.minRingDiff, structSizeSino3D.maxRingDiff);
% Sinograma corregido:
interfileWriteSino(sinogram3Dcorregido, [outputFilename '_scatterCorrected'], structSizeSino3D.sinogramsPerSegment, structSizeSino3D.minRingDiff, structSizeSino3D.maxRingDiff);