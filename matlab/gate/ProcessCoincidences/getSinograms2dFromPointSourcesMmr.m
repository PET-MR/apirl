%  *********************************************************************
%  Project: EPSRC - Simultaneous PET-MR Modelling and Reconstruction for Imaging Brain Disorders
%  Author: Martin Belzunce. Kings College London.
%  Created: 20/04/2016
%  *********************************************************************
%  Generates a 2d sinogram for each point source simulated with a 2d gate simulation of the mmr.

function [sinograms, sinograms_without_randoms, sinograms_without_scatter_randoms, timeSimulations, emissionMap] = getSinograms2dFromPointSourcesMmr(outputPath, structSimu, structSizeSino2D, numSources, graficarOnline)


%%  VARIABLES PARA GENERACIÓN DE SINOGRAMAS 3D
sinograms = cell(numSources,1);
for i = 1 : numSources
    sinograms{i} = single(zeros(structSizeSino2D.numR,structSizeSino2D.numTheta, sum(structSizeSino2D.numZ)));
    sinograms_without_randoms{i} = single(zeros(structSizeSino2D.numR,structSizeSino2D.numTheta, sum(structSizeSino2D.numZ)));
    sinograms_without_scatter_randoms{i} = single(zeros(structSizeSino2D.numR,structSizeSino2D.numTheta, sum(structSizeSino2D.numZ)));
end
%% VARIABLES AUXILIARES Y PARA VISUALIZACIÓN DE RESULTADOS PARCIALES
% Valores de cada coordenada dentro de la simulación:
valoresX = -400:2:400;                   % Valores posibles de emisión de positrones en la variable Z (O sea, subre todo el FOV).
valoresY = -500:2:500;                   % Valores posibles de emisión de positrones en la variable Z (O sea, subre todo el FOV).
valoresZ = -158:2:158;                   % Valores posibles de emisión de positrones en la variable Z (O sea, subre todo el FOV).
% Conjunto de valores utilziados apra generar histogramas XY:
valoresYX = {valoresY valoresX};           % Cell Array con los valores posibles de las variables x e y dentro del sistema.
% Matricez para imágenes de detección en el plano XY de todo el scanner:
histDetectionXY = zeros(numel(valoresYX{1}), numel(valoresYX{2}));
%% EMISSION MAP
imageSize_pixels = [344 344];
pixelSize_mm = [2.08625 2.08625];
coordX = -pixelSize_mm(2)*imageSize_pixels(2)/2+pixelSize_mm(2)/2:pixelSize_mm(2):pixelSize_mm(2)*imageSize_pixels(2)/2-pixelSize_mm(2)/2;
emissionMap = zeros(imageSize_pixels);

%% SCANNER PARAMETERS
numberofAxialBlocks = 1;
numberofTransverseBlocksPerRing = 56;
numberOfBlocks = numberofTransverseBlocksPerRing*numberofAxialBlocks;
numberOfTransverseCrystalsPerBlock = 9; % includes the gap
coordY = -pixelSize_mm(1)*imageSize_pixels(1)/2+pixelSize_mm(1)/2:pixelSize_mm(1):pixelSize_mm(1)*imageSize_pixels(1)/2-pixelSize_mm(1)/2;
numberOfAxialCrystalsPerBlock = 1;
numberOfBlocksPerRing = 56;
numberOfRings = 1;
numberOfCrystalRings = numberofAxialBlocks*numberOfAxialCrystalsPerBlock;
numberOfTransverseCrystalsPerRing = numberOfTransverseCrystalsPerBlock*numberOfBlocksPerRing; % includes the gap
numberOfCrystals = numberOfTransverseCrystalsPerRing*numberOfRings;
% DetectorIds
[mapaDet1Ids, mapaDet2Ids] = createMmrDetectorsIdInSinogram();

histCrystalsComb = zeros(numberOfCrystals, numberOfCrystals);
histCrystalsCombScatter = zeros(numberOfCrystals, numberOfCrystals);
%% OTRAS VARIABLES A INICIALIZAR
% Variables relacionadas con el conteo de eventos:
eventosTotales = 0;
eventosTrues = 0;
eventosRandoms = 0;
eventosConScatter = 0;
eventosSinCompton = 0;
eventosTotalesEnVentana = 0;
eventosTruesConScatter = 0;
TiempoSplitsAnteriores = 0; % Variable que acumula los tiempos de adquisición de los aplits ya procesados.
timeSimulations = 0;            % Variable que indica el tiempo total de la simulación.

% Variables para histogramas y filtrado de energía:
CanalesEnergia =  0:0.001:2.0;  % Canales de Energía para el histograma.
HistEnergias = zeros(1,numel(CanalesEnergia));  % Inicialización en cero del Histograma de Energías.
HistEnergiasFilt = zeros(1,numel(CanalesEnergia));  % Inicialización en cero del Histograma de Energías Filtrado.
HistEnergiasSinCompton = zeros(1,numel(CanalesEnergia));  % Inicialización en cero del Histograma de Energías de Eventos Cin Comptons.


%% INICIALIZACIÓN DE VARIABLES A UTILIZAR EN EL PROCESAMIENTO DEL COINCIDENCE
eventosTotales = 0;

TiempoSplitsAnteriores = 0; % Variable que acumula los tiempos de adquisición de los aplits ya procesados.

%% PARÁMETROS DEL ARCHIVO DE COINCIDENCIAS DEL GATE
% Índices de columnas en el archivo de salida:
numColumnas = 20*2;
numColumnasPorSingle = numColumnas / 2;
colEventId1 = 2;
colEventId2 = 2 + numColumnasPorSingle;
colTimeStamp1 = 7;
colTimeStamp2 = 7 + numColumnasPorSingle;
colEnergy1 = 8;
colEnergy2 = 8 + numColumnasPorSingle;
colSourceId1 = 3;
colSourceId2 = 3 + numColumnasPorSingle;
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
colVolIdScanner = 12;
colVolIdScanner2 = 12 + numColumnasPorSingle;
colVolIdBlock = 13;
colVolIdBlock2 = 13 + numColumnasPorSingle;
colVolIdCrystal = 14;
colVolIdCrystal2 = 14 + numColumnasPorSingle;

colCompton1 = 15;
colCompton2 = 15 + numColumnasPorSingle;

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
                datos = textscan(FID, '%f', 100000 * numColumnas);
                if mod(numel(datos{1}), numColumnas) ~= 0
                    % El archivo no esta entero, puede que se haya cortado
                    % por una finalizacion abrupta de la simulacion.
                    % Elimino la coincidencia incompleta
                    datos{1}(end-(mod(numel(datos{1}),numColumnas)-1) : end) =[];
                end
                % Le doy forma de matriz a todos los datos leídos.
                coincidenceMatrix = reshape(datos{1}, numColumnas, numel(datos{1})/numColumnas)';
                eventosLeidos = size(coincidenceMatrix,1);

                % Guardo el tiempo inicial del archivo.
                if PrimeraLectura
                    TiempoInicialSplit = coincidenceMatrix(1,colTimeStamp1);
                    PrimeraLectura = 0;
                end
                % Obtengo el tiempo de duración de la simulación hasta el
                % evento procesado en el ciclo actual.
                TiempoSplit = coincidenceMatrix(end,colTimeStamp1) - TiempoInicialSplit;
                timeSimulations = TiempoSplit + TiempoSplitsAnteriores;
                disp('');
                fprintf('Tiempo Inicial Split: %d\n', TiempoInicialSplit);
                fprintf('Tiempo Medición Split: %d\n', TiempoSplit);
                fprintf('Tiempo Total Simulación: %d\n', timeSimulations);
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
                    title('Energy Spectrum');
                end

                % Histogram of detection position:
                histDetectionXY = histDetectionXY + hist3([coincidenceMatrix(:,colDetectionY1),coincidenceMatrix(:,colDetectionX1);...
                    coincidenceMatrix(:,colDetectionY2),coincidenceMatrix(:,colDetectionX2)], valoresYX);
                if(graficarOnline)
                    figure(2);
                    imshow(histDetectionXY, []);
                    title('Histogram of Detections in Plane XY');
                end
                
                % Emission map:
                emissionMap = emissionMap + hist3([coincidenceMatrix(:,colEmisionY1), coincidenceMatrix(:,colEmisionX1)],...
                    {coordY, coordX});
                if(graficarOnline)
                    figure(3);
                    imshow(emissionMap,[]);
                    title(sprintf('Slice %d of the Emission Map', slice));
                end
                %% PROCESS EACH POINT SOURCE
                for i = 1 : numSources
                    indiceEventsForSource = (coincidenceMatrix(:,colSourceId1) == (i-1)) & (coincidenceMatrix(:,colSourceId2) == (i-1));
                    indiceRandoms = coincidenceMatrix(:,colEventId1) ~= coincidenceMatrix(:,colEventId2);
                    indiceScatter = (coincidenceMatrix(:,colCompton1)>0) | (coincidenceMatrix(:,colCompton2)>0);
                    %% SINOGRAM 2D
                    % Need to convert the indexes in the simulation into the
                    % crystal indexes used by mmr. In the simulation, the
                    % crystal Id is the crystal within a block [0:8;9:18;..]
                    % Then there are 56*8 blocks, again startiing axially:
                    globalCrystalId1 = rem(coincidenceMatrix(indiceEventsForSource,colVolIdCrystal),numberOfTransverseCrystalsPerBlock) + floor(coincidenceMatrix(indiceEventsForSource,colVolIdCrystal)/numberOfTransverseCrystalsPerBlock)*numberOfTransverseCrystalsPerRing+...
                        rem(coincidenceMatrix(indiceEventsForSource,colVolIdBlock),numberofTransverseBlocksPerRing)*numberOfTransverseCrystalsPerBlock + floor(coincidenceMatrix(indiceEventsForSource,colVolIdBlock)/numberofTransverseBlocksPerRing)*numberOfTransverseCrystalsPerRing*numberofAxialBlocks;
                    globalCrystalId2 = rem(coincidenceMatrix(indiceEventsForSource,colVolIdCrystal2),numberOfTransverseCrystalsPerBlock) + floor(coincidenceMatrix(indiceEventsForSource,colVolIdCrystal2)/numberOfTransverseCrystalsPerBlock)*numberOfTransverseCrystalsPerRing+...
                        rem(coincidenceMatrix(indiceEventsForSource,colVolIdBlock2),numberofTransverseBlocksPerRing)*numberOfTransverseCrystalsPerBlock + floor(coincidenceMatrix(indiceEventsForSource,colVolIdBlock2)/numberofTransverseBlocksPerRing)*numberOfTransverseCrystalsPerRing*numberofAxialBlocks;
                    % To generate the sinogram, I use the detector id:
                    %crystalIdInRing1 = rem(globalCrystalId1,numberOfTransverseCrystalsPerRing)+1;
                    %ringId1 = floor(globalCrystalId1/numberOfTransverseCrystalsPerRing)+1;
                    % Based 1 index:
                    globalCrystalId1 = globalCrystalId1 + 1;
                    globalCrystalId2 = globalCrystalId2 + 1;
                    % Convert from Gate to mmr:
                    globalCrystalId1 = 130-globalCrystalId1;
                    globalCrystalId1(globalCrystalId1<=0) = globalCrystalId1(globalCrystalId1<=0) + 504; 
                    globalCrystalId2 = 130-globalCrystalId2;
                    globalCrystalId2(globalCrystalId2<=0) = globalCrystalId2(globalCrystalId2<=0) + 504; 

                    % Histogram with a combination of crystals:
                    histCrystalsComb = hist3([globalCrystalId1 globalCrystalId2], {1:numberOfCrystals 1:numberOfCrystals});
                    % Gaps:
                    histCrystalsComb(9:9:end,:) = 0;
                    histCrystalsComb(:,9:9:end) = 0;
                    sinograms{i}(:) = sinograms{i}(:) + histCrystalsComb(sub2ind(size(histCrystalsComb),mapaDet1Ids(:), mapaDet2Ids(:)));
                    sinograms{i}(:) =  sinograms{i}(:) + histCrystalsComb(sub2ind(size(histCrystalsComb),mapaDet2Ids(:), mapaDet1Ids(:)));
                    
                    % Histogram with a combination of crystals:
                    histCrystalsComb = hist3([globalCrystalId1(~indiceRandoms) globalCrystalId2(~indiceRandoms)], {1:numberOfCrystals 1:numberOfCrystals});
                    % Gaps:
                    histCrystalsComb(9:9:end,:) = 0;
                    histCrystalsComb(:,9:9:end) = 0;
                    sinograms_without_randoms{i}(:) = sinograms_without_randoms{i}(:) + histCrystalsComb(sub2ind(size(histCrystalsComb),mapaDet1Ids(:), mapaDet2Ids(:)));
                    sinograms_without_randoms{i}(:) =  sinograms_without_randoms{i}(:) + histCrystalsComb(sub2ind(size(histCrystalsComb),mapaDet2Ids(:), mapaDet1Ids(:)));
                    
                    % Histogram with a combination of crystals:
                    histCrystalsComb = hist3([globalCrystalId1(~indiceRandoms & ~indiceScatter) globalCrystalId2(~indiceRandoms & ~indiceScatter)], {1:numberOfCrystals 1:numberOfCrystals});
                    % Gaps:
                    histCrystalsComb(9:9:end,:) = 0;
                    histCrystalsComb(:,9:9:end) = 0;
                    sinograms_without_scatter_randoms{i}(:) = sinograms_without_scatter_randoms{i}(:) + histCrystalsComb(sub2ind(size(histCrystalsComb),mapaDet1Ids(:), mapaDet2Ids(:)));
                    sinograms_without_scatter_randoms{i}(:) =  sinograms_without_scatter_randoms{i}(:) + histCrystalsComb(sub2ind(size(histCrystalsComb),mapaDet2Ids(:), mapaDet1Ids(:)));
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
%% SAVE SINOGRAMS
for i = 1 : numSources
    interfileWriteSino(sinograms{i}, [outputPath sprintf('sinogram_%d',i)], structSizeSino2D);
end

interfilewrite(emissionMap, [outputPath 'emissionMap'], pixelSize_mm);


