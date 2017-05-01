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

function [sinogram, sinogram_scatter, sinogram_randoms, TiempoSimulacion, emissionMap, histDetectionXY] = getSinograms3dMmr(outputPath, structSimu, structSizeSino3D, graficarOnline, pixelSize_mm, removeRandoms, simulatedTransverseCrystalsPerBlock)
fovSize_mm = [2.08625 2.08625 2.03125].*[344 344 127];
if nargin < 5
    pixelSize_mm = [2.08625 2.08625 2.03125];
    imageSize_pixels = [344 344 127];
else
    imageSize_pixels = fovSize_mm./ pixelSize_mm;
end
%%  VARIABLES PARA GENERACIÓN DE SINOGRAMAS 3D
sinogram = single(zeros(structSizeSino3D.numR,structSizeSino3D.numTheta, sum(structSizeSino3D.sinogramsPerSegment)));
sinogram_scatter = single(zeros(structSizeSino3D.numR,structSizeSino3D.numTheta, sum(structSizeSino3D.sinogramsPerSegment)));
sinogram_randoms = single(zeros(structSizeSino3D.numR,structSizeSino3D.numTheta, sum(structSizeSino3D.sinogramsPerSegment)));
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
coordX = -pixelSize_mm(2)*imageSize_pixels(2)/2+pixelSize_mm(2)/2:pixelSize_mm(2):pixelSize_mm(2)*imageSize_pixels(2)/2-pixelSize_mm(2)/2;
emissionMap = zeros(imageSize_pixels);
%% SCANNER PARAMETERS
numberofAxialBlocks = 8;
numberofTransverseBlocksPerRing = 56;
numberOfBlocks = numberofTransverseBlocksPerRing*numberofAxialBlocks;
numberOfTransverseCrystalsPerBlock = 9; % includes the gap
if nargin < 7
    simulatedTransverseCrystalsPerBlock = numberOfTransverseCrystalsPerBlock; % this is because you can simulate 9 vrystals or 8 crystal an the gaps, for the mmr crystal index we still need to use the 9 pixels, but not for the crystals in gate.
end
coordY = -pixelSize_mm(1)*imageSize_pixels(1)/2+pixelSize_mm(1)/2:pixelSize_mm(1):pixelSize_mm(1)*imageSize_pixels(1)/2-pixelSize_mm(1)/2;
coordZ = -pixelSize_mm(3)*imageSize_pixels(3)/2+pixelSize_mm(3)/2:pixelSize_mm(3):pixelSize_mm(3)*imageSize_pixels(3)/2-+pixelSize_mm(3)/2;
numberOfAxialCrystalsPerBlock = 8;
numberOfBlocksPerRing = 56;
numberOfRings = 64;
numberOfCrystalRings = numberofAxialBlocks*numberOfAxialCrystalsPerBlock;
numberOfTransverseCrystalsPerRing = numberOfTransverseCrystalsPerBlock*numberOfBlocksPerRing; % includes the gap
numberOfCrystals = numberOfTransverseCrystalsPerRing*numberOfRings;
% DetectorIds
[mapaDet1Ids, mapaDet2Ids] = createMmrDetectorsIdInSinogram3d();

histCrystalsComb = zeros(numberOfCrystals, numberOfCrystals);
histCrystalsCombScatter = zeros(numberOfCrystals, numberOfCrystals);
histCrystalsCombRandoms = zeros(numberOfCrystals, numberOfCrystals);
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
                if isempty(datos{1})
                    break;
                end
                if mod(numel(datos{1}), numColumnas) ~= 0
                    % El archivo no esta entero, puede que se haya cortado
                    % por una finalizacion abrupta de la simulacion.
                    % Elimino la coincidencia incompleta
                    datos{1}(end-(mod(numel(datos{1}),numColumnas)-1) : end) =[];
                    % Remove an addional row:
                    datos{1}(end-numColumnas+1 : end) =[];
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
                
                % Emission map:
                emissionMap = emissionMap + hist4([coincidenceMatrix(:,colEmisionY1), coincidenceMatrix(:,colEmisionX1), coincidenceMatrix(:,colEmisionZ1)],...
                    {coordY, coordX, coordZ});
                if(graficarOnline)
                    figure(3);
                    slice = 64;
                    imshow(emissionMap(:,:,slice),[]);
                    title(sprintf('Slice %d of the Emission Map', slice));
                end
                %% REMOVE RANDOMS
                % Remove randoms because they are not part of the system
                % matrix:
                % If e want to remove the scatter, do it now:
                if removeRandoms
                    indiceRandomsEvents = (coincidenceMatrix(:,colEventId1) ~= coincidenceMatrix(:,colEventId2));
                    coincidenceMatrix(indiceRandomsEvents,:) = [];
                end
                %% SINOGRAM 3D
                % Need to convert the indexes in the simulation into the
                % crystal indexes used by mmr. In the simulation, the
                % crystal Id is the crystal within a block [0:8;9:18;..]
                % Then there are 56*8 blocks, again startiing axially:
                globalCrystalId1 = rem(coincidenceMatrix(:,colVolIdCrystal),simulatedTransverseCrystalsPerBlock) + floor(coincidenceMatrix(:,colVolIdCrystal)/simulatedTransverseCrystalsPerBlock)*numberOfTransverseCrystalsPerRing+...
                    rem(coincidenceMatrix(:,colVolIdBlock),numberofTransverseBlocksPerRing)*numberOfTransverseCrystalsPerBlock + floor(coincidenceMatrix(:,colVolIdBlock)/numberofTransverseBlocksPerRing)*numberOfTransverseCrystalsPerRing*numberofAxialBlocks;
                globalCrystalId2 = rem(coincidenceMatrix(:,colVolIdCrystal2),simulatedTransverseCrystalsPerBlock) + floor(coincidenceMatrix(:,colVolIdCrystal2)/simulatedTransverseCrystalsPerBlock)*numberOfTransverseCrystalsPerRing+...
                    rem(coincidenceMatrix(:,colVolIdBlock2),numberofTransverseBlocksPerRing)*numberOfTransverseCrystalsPerBlock + floor(coincidenceMatrix(:,colVolIdBlock2)/numberofTransverseBlocksPerRing)*numberOfTransverseCrystalsPerRing*numberofAxialBlocks;
                % To generate the sinogram, I use the detector id:
                %crystalIdInRing1 = rem(globalCrystalId1,numberOfTransverseCrystalsPerRing)+1;
                %ringId1 = floor(globalCrystalId1/numberOfTransverseCrystalsPerRing)+1;
                % Based 1 index:
                % globalCrystalId1 = globalCrystalId1 + 1;
                % globalCrystalId2 = globalCrystalId2 + 1;
                % Convert from Gate to mmr:
                globalCrystalId1_ring = floor((globalCrystalId1)/504)+1;
                globalCrystalId1_Inring = rem((globalCrystalId1),504)+1;
                globalCrystalId2_ring = floor((globalCrystalId2)/504)+1;
                globalCrystalId2_Inring = rem((globalCrystalId2),504)+1;
                % Convert from Gate to mmr:
                if simulatedTransverseCrystalsPerBlock == numberOfTransverseCrystalsPerBlock
                    globalCrystalId1_Inring = globalCrystalId1_Inring+121;
                else
                    globalCrystalId1_Inring = globalCrystalId1_Inring+126;
                end
                globalCrystalId1_Inring(globalCrystalId1_Inring>504) = globalCrystalId1_Inring(globalCrystalId1_Inring>504) - 504; 
                if simulatedTransverseCrystalsPerBlock == numberOfTransverseCrystalsPerBlock
                    globalCrystalId2_Inring = globalCrystalId2_Inring+121;
                else
                    globalCrystalId2_Inring = globalCrystalId2_Inring+126;
                end
                globalCrystalId2_Inring(globalCrystalId2_Inring>504) = globalCrystalId2_Inring(globalCrystalId2_Inring>504) - 504; 
                % Go back to linear indices:
                globalCrystalId1 = sub2ind([504,64],globalCrystalId1_Inring,globalCrystalId1_ring);
                globalCrystalId2 = sub2ind([504,64],globalCrystalId2_Inring,globalCrystalId2_ring);
                % To remove the gaps using the crystal index if they were not included in the simulation:
                if simulatedTransverseCrystalsPerBlock == numberOfTransverseCrystalsPerBlock
                    indicesGaps = find((rem(globalCrystalId1,9)~=0)&(rem(globalCrystalId2,9)~=0));
                else
                    indicesGaps = globalCrystalId1 >= 0; % all the events.
                end
                % Histogram with a combination of crystals:
                histCrystalsComb = histCrystalsComb + hist3([globalCrystalId1(indicesGaps) globalCrystalId2(indicesGaps)], {1:numberOfCrystals 1:numberOfCrystals});
                % Gaps:
                if simulatedTransverseCrystalsPerBlock == numberOfTransverseCrystalsPerBlock
                    histCrystalsComb(9:9:end,:) = 0;
                    histCrystalsComb(:,9:9:end) = 0;
                end
                
                % The same for scattered events:
                indicesScatter = coincidenceMatrix(indicesGaps,colCompton1)>0 | coincidenceMatrix(indicesGaps,colCompton2)>0;
                histCrystalsCombScatter = histCrystalsCombScatter + hist3([globalCrystalId1(indicesGaps(indicesScatter)) globalCrystalId2(indicesGaps(indicesScatter))], {1:numberOfCrystals 1:numberOfCrystals});
                % Gaps:
                if simulatedTransverseCrystalsPerBlock == numberOfTransverseCrystalsPerBlock
                    histCrystalsCombScatter(9:9:end,:) = 0;
                    histCrystalsCombScatter(:,9:9:end) = 0;
                end
                
%                 % The same for randoms events:
%                 indicesRandoms = coincidenceMatrix(indicesGaps,colEventId1) ~= coincidenceMatrix(indicesGaps,colEventId2);
%                 histCrystalsCombRandoms = histCrystalsCombRandoms + hist3([globalCrystalId1(indicesGaps(indicesRandoms)) globalCrystalId2(indicesGaps(indicesRandoms))], {1:numberOfCrystals 1:numberOfCrystals});
%                 % Gaps:
%                 if simulatedTransverseCrystalsPerBlock == numberOfTransverseCrystalsPerBlock
%                     histCrystalsCombRandoms(9:9:end,:) = 0;
%                     histCrystalsCombRandoms(:,9:9:end) = 0;
%                 end
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
% From the histogram of crystals:
sinogram(:) = histCrystalsComb(sub2ind(size(histCrystalsComb),mapaDet1Ids(:), mapaDet2Ids(:)));
sinogram(:) =  sinogram(:) + histCrystalsComb(sub2ind(size(histCrystalsComb),mapaDet2Ids(:), mapaDet1Ids(:)));
sinogram_scatter(:) = histCrystalsCombScatter(sub2ind(size(histCrystalsComb),mapaDet1Ids(:), mapaDet2Ids(:)));
sinogram_scatter(:) =  sinogram_scatter(:) + histCrystalsCombScatter(sub2ind(size(histCrystalsComb),mapaDet2Ids(:), mapaDet1Ids(:)));
sinogram_randoms(:) = histCrystalsCombRandoms(sub2ind(size(histCrystalsComb),mapaDet1Ids(:), mapaDet2Ids(:)));
sinogram_randoms(:) =  sinogram_randoms(:) + histCrystalsCombRandoms(sub2ind(size(histCrystalsComb),mapaDet2Ids(:), mapaDet1Ids(:)));
interfileWriteSino(sinogram, [outputPath 'sinogram'], structSizeSino3D);
interfileWriteSino(sinogram_scatter, [outputPath 'sinogram_scatter'], structSizeSino3D);
interfileWriteSino(sinogram_randoms, [outputPath 'sinogram_randoms'], structSizeSino3D);
%% WRITE EMISSION
interfilewrite(emissionMap, [outputPath 'emissionMap'], pixelSize_mm);
