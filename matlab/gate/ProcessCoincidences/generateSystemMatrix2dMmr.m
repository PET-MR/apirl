%  *********************************************************************
%  Project: EPSRC - Simultaneous PET-MR Modelling and Reconstruction for Imaging Brain Disorders
%  Author: Martin Belzunce. Kings College London.
%  Created: 20/04/2016
%  *********************************************************************
%  Generates a system matrix from 2d mmr, it receives the limits of the
%  pixels in x and y. Example, pixel(1,1) it has the coordinates
%  [coordX(1):coordX(2),coordY(1):coordY(2)]. 

function [systemMatrix, TiempoSimulacion, emissionMap] = generateSystemMatrix2dMmr(outputPath, structSimu, structSizeSino2D, coordX, coordY, graficarOnline)

%%  VARIABLES PARA GENERACIÓN DE SINOGRAMAS 3D
sinogram = single(zeros(structSizeSino2D.numR,structSizeSino2D.numTheta, sum(structSizeSino2D.numZ)));
sinogram_scatter = single(zeros(structSizeSino2D.numR,structSizeSino2D.numTheta, sum(structSizeSino2D.numZ)));
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
imageSize_pixels = [numel(coordY)-1 numel(coordX)-1];
pixelSize_mm = [coordY(2)-coordY(1) coordX(2)-coordX(1)];
coordX_center_mm = coordX(1) + pixelSize_mm(2)/2 : pixelSize_mm(2) : coordX(end)- pixelSize_mm(2)/2 + 0.1;% - pixelSize_mm(2)/2; % Coordinates centred in the pixel (coordX are the edges).
coordY_center_mm = coordY(1) + pixelSize_mm(1)/2 : pixelSize_mm(1) : coordY(end)- pixelSize_mm(1)/2 + 0.1;% - pixelSize_mm(1)/2; % Coordinates centred in the pixel (coordY are the edges).
emissionMap = zeros(imageSize_pixels);
%% SYSTEM MATRIX
% One sinogram for pxiel:
systemMatrix = cell(imageSize_pixels);
systemMatrix_scatter = cell(imageSize_pixels);
for i = 1 : numel(systemMatrix)
    systemMatrix{i} = zeros(size(sinogram));
    systemMatrix_scatter{i} = zeros(size(sinogram));
end
%% SCANNER PARAMETERS
numberofAxialBlocks = 1;
numberofTransverseBlocksPerRing = 56;
numberOfBlocks = numberofTransverseBlocksPerRing*numberofAxialBlocks;
numberOfTransverseCrystalsPerBlock = 9; % includes the gap
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
TiempoSimulacion = 0;            % Variable que indica el tiempo total de la simulación.

% Variables para histogramas y filtrado de energía:
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
numColumnas = 20*2;
numColumnasPorSingle = numColumnas / 2;
colEventId1 = 2;
colEventId2 = 2 + numColumnasPorSingle;
colSourceId1 = 3;
colSourceId2 = 3 + numColumnasPorSingle;
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
                % Remove randoms because they are not part of the system
                % matrix:
                indiceRandomsEvents = (coincidenceMatrix(:,colEventId1) ~= coincidenceMatrix(:,colEventId2));
                coincidenceMatrix(indiceRandomsEvents,:) = [];
                
                % Emission map:
                emissionMap = emissionMap + hist3([coincidenceMatrix(:,colEmisionY1), coincidenceMatrix(:,colEmisionX1)],...
                    {coordY_center_mm, coordX_center_mm});
                if(graficarOnline)
                    figure(3);
                    imshow(emissionMap,[]);
                    title(sprintf('Slice %d of the Emission Map', slice));
                end
                %% GENERATE SYSTEM MATRIX
                for y = 1 : size(systemMatrix,1)
                    for x = 1 : size(systemMatrix,2)
                        indexThisPixel = (coincidenceMatrix(:,colEmisionY1) >= coordY(y)) & (coincidenceMatrix(:,colEmisionY1) < coordY(y+1)) & ...
                            (coincidenceMatrix(:,colEmisionX1) >= coordX(x)) & (coincidenceMatrix(:,colEmisionX1) < coordX(x+1));
                                        % Need to convert the indexes in the simulation into the
                        % crystal indexes used by mmr. In the simulation, the
                        % crystal Id is the crystal within a block [0:8;9:18;..]
                        % Then there are 56*8 blocks, again startiing axially:
                        globalCrystalId1 = rem(coincidenceMatrix(indexThisPixel,colVolIdCrystal),numberOfTransverseCrystalsPerBlock) + floor(coincidenceMatrix(indexThisPixel,colVolIdCrystal)/numberOfTransverseCrystalsPerBlock)*numberOfTransverseCrystalsPerRing+...
                            rem(coincidenceMatrix(indexThisPixel,colVolIdBlock),numberofTransverseBlocksPerRing)*numberOfTransverseCrystalsPerBlock + floor(coincidenceMatrix(indexThisPixel,colVolIdBlock)/numberofTransverseBlocksPerRing)*numberOfTransverseCrystalsPerRing*numberofAxialBlocks;
                        globalCrystalId2 = rem(coincidenceMatrix(indexThisPixel,colVolIdCrystal2),numberOfTransverseCrystalsPerBlock) + floor(coincidenceMatrix(indexThisPixel,colVolIdCrystal2)/numberOfTransverseCrystalsPerBlock)*numberOfTransverseCrystalsPerRing+...
                            rem(coincidenceMatrix(indexThisPixel,colVolIdBlock2),numberofTransverseBlocksPerRing)*numberOfTransverseCrystalsPerBlock + floor(coincidenceMatrix(indexThisPixel,colVolIdBlock2)/numberofTransverseBlocksPerRing)*numberOfTransverseCrystalsPerRing*numberofAxialBlocks;
                        % To generate the sinogram, I use the detector id:
                        %crystalIdInRing1 = rem(globalCrystalId1,numberOfTransverseCrystalsPerRing)+1;
                        %ringId1 = floor(globalCrystalId1/numberOfTransverseCrystalsPerRing)+1;
                        % Based 1 index:
                        globalCrystalId1 = globalCrystalId1 + 1;
                        globalCrystalId2 = globalCrystalId2 + 1;
                        % Convert from Gate to mmr:
                        globalCrystalId1 = globalCrystalId1+121;
                        globalCrystalId1(globalCrystalId1>504) = globalCrystalId1(globalCrystalId1>504) - 504; 
                        globalCrystalId2 = globalCrystalId2+121;
                        globalCrystalId2(globalCrystalId2>504) = globalCrystalId2(globalCrystalId2>504) - 504;

                        % Histogram with a combination of crystals:
                        histCrystalsComb = hist3([globalCrystalId1 globalCrystalId2], {1:numberOfCrystals 1:numberOfCrystals});
                        % Gaps:
                        histCrystalsComb(9:9:end,:) = 0;
                        histCrystalsComb(:,9:9:end) = 0;
                        % From the histogram of crystals:
                        systemMatrix{y,x}(:) = systemMatrix{y,x}(:) + histCrystalsComb(sub2ind(size(histCrystalsComb),mapaDet1Ids(:), mapaDet2Ids(:)));
                        systemMatrix{y,x}(:) =  systemMatrix{y,x}(:) + histCrystalsComb(sub2ind(size(histCrystalsComb),mapaDet2Ids(:), mapaDet1Ids(:)));
                        
                        % To debug the scatter
%                         % The same for scattered events:
%                         indicesScatter = coincidenceMatrix(indexThisPixel,colCompton1)>0 | coincidenceMatrix(indexThisPixel,colCompton2)>0;
%                         histCrystalsCombScatter = hist3([globalCrystalId1(indicesScatter) globalCrystalId2(indicesScatter)], {1:numberOfCrystals 1:numberOfCrystals});
%                         histCrystalsCombScatter(9:9:end,:) = 0;
%                         systemMatrix_scatter{y,x}(:) = systemMatrix_scatter{y,x}(:) + histCrystalsCombScatter(sub2ind(size(histCrystalsCombScatter),mapaDet1Ids(:), mapaDet2Ids(:)));
%                         systemMatrix_scatter{y,x}(:) =  systemMatrix_scatter{y,x}(:) + histCrystalsCombScatter(sub2ind(size(histCrystalsCombScatter),mapaDet2Ids(:), mapaDet1Ids(:)));
                        
                    end
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
%% COMPLETE SINOGRAMS
for y = 1 : size(systemMatrix,1)
    for x = 1 : size(systemMatrix,2)
        interfileWriteSino(single(systemMatrix{y,x}), [outputPath sprintf('sinogram_y%d_x%d',y,x)], structSizeSino2D);
    end
end
%% WRITE EMISSION
interfilewrite(emissionMap, [outputPath 'emissionMap'], pixelSize_mm);


