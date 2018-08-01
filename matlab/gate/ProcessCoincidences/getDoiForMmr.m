%  *********************************************************************
%  Proyecto AR-PET. Comisión Nacional de Energía Atómica.
%  Autor: Martín Belzunce. UTN-FRBA.
%  Fecha de Creación: 16/05/2017
%  *********************************************************************
%  This function get the mean interaction position of the scanner for each
%  singoram bin.
% It return the doi in x,y,z for each sinogram bin.

function [sinoMeanDoiCrystalX1, sinoMeanDoiCrystalX2, sinoMeanDoiCrystalY1, sinoMeanDoiCrystalY2, sinoMeanDoiCrystalZ1, sinoMeanDoiCrystalZ2, sinogram] = getDoiForMmr(outputPath, structSimu, structSizeSino3D)

%%  VARIABLES PARA GENERACIÓN DE SINOGRAMAS 3D
sinogram = single(zeros(structSizeSino3D.numR,structSizeSino3D.numTheta, sum(structSizeSino3D.sinogramsPerSegment)));
%% VARIABLES AUXILIARES Y PARA VISUALIZACIÓN DE RESULTADOS PARCIALES
% Valores de cada coordenada dentro de la simulación:
valoresX = -400:2:400;                   % Valores posibles de emisión de positrones en la variable Z (O sea, subre todo el FOV).
valoresY = -500:2:500;                   % Valores posibles de emisión de positrones en la variable Z (O sea, subre todo el FOV).
valoresZ = -158:2:158;                   % Valores posibles de emisión de positrones en la variable Z (O sea, subre todo el FOV).
% Conjunto de valores utilziados apra generar histogramas XY:
valoresYX = {valoresY valoresX};           % Cell Array con los valores posibles de las variables x e y dentro del sistema.
% Matricez para imágenes de detección en el plano XY de todo el scanner:
histDetectionXY = zeros(numel(valoresYX{1}), numel(valoresYX{2}));

%% SCANNER PARAMETERS
numberofAxialBlocks = 8;
numberofTransverseBlocksPerRing = 56;
numberOfBlocks = numberofTransverseBlocksPerRing*numberofAxialBlocks;
numberOfTransverseCrystalsPerBlock = 9; % this is the number used in the sinograms (includes the gaps)
simulatedTransverseCrystalsPerBlock = 8; % this is what is real in the scanner and what is modelled in gate.
numberOfAxialCrystalsPerBlock = 8;
numberOfBlocksPerRing = 56;
numberOfRings = 64;
numberOfCrystalRings = numberofAxialBlocks*numberOfAxialCrystalsPerBlock;
numberOfTransverseCrystalsPerRing = numberOfTransverseCrystalsPerBlock*numberOfBlocksPerRing; % includes the gap
numberOfCrystals = numberOfTransverseCrystalsPerRing*numberOfRings;
% DetectorIds
[mapaDet1Ids, mapaDet2Ids] = createMmrDetectorsIdInSinogram3d();
% Hist with combination of cristals
histCrystalsComb = zeros(numberOfCrystals, numberOfCrystals, 'single');
meanDoiCrystalX1 = zeros(numberOfCrystals, numberOfCrystals, 'single');
meanDoiCrystalX2 = zeros(numberOfCrystals, numberOfCrystals, 'single');
meanDoiCrystalY1 = zeros(numberOfCrystals, numberOfCrystals, 'single');
meanDoiCrystalY2 = zeros(numberOfCrystals, numberOfCrystals, 'single');
meanDoiCrystalZ1 = zeros(numberOfCrystals, numberOfCrystals, 'single');
meanDoiCrystalZ2 = zeros(numberOfCrystals, numberOfCrystals, 'single');
%% OTRAS VARIABLES A INICIALIZAR
% Variables relacionadas con el conteo de eventos:
eventosTrues = 0;
eventosRandoms = 0;
eventosConScatter = 0;
eventosSinCompton = 0;
eventosTotalesEnVentana = 0;
eventosTruesConScatter = 0;


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
                datos = textscan(FID, '%f', 500000 * numColumnas);
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

                %% REMOVE RANDOMS AND SCATTER
                indiceRandomsEvents = (coincidenceMatrix(:,colEventId1) ~= coincidenceMatrix(:,colEventId2));
                indiceScattersEvents = coincidenceMatrix(:,colCompton1)>0 | coincidenceMatrix(:,colCompton2)>0;
                coincidenceMatrix(indiceRandomsEvents|indiceScattersEvents,:) = [];
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
                globalCrystalId1_Inring = globalCrystalId1_Inring+126;
                globalCrystalId1_Inring(globalCrystalId1_Inring>504) = globalCrystalId1_Inring(globalCrystalId1_Inring>504) - 504; 
                globalCrystalId2_Inring = globalCrystalId2_Inring+126;
                globalCrystalId2_Inring(globalCrystalId2_Inring>504) = globalCrystalId2_Inring(globalCrystalId2_Inring>504) - 504; 
                % Go back to linear indices:
                globalCrystalId1 = sub2ind([504,64],globalCrystalId1_Inring,globalCrystalId1_ring);
                globalCrystalId2 = sub2ind([504,64],globalCrystalId2_Inring,globalCrystalId2_ring);
                % The gaps are already simulated, there is no need to
                % include them artificially.
                % Histogram with a combination of crystals:
                histCrystalsComb = histCrystalsComb + hist3([globalCrystalId1 globalCrystalId2], {1:numberOfCrystals 1:numberOfCrystals});
                % linear index:
                ind = sub2ind(size(histCrystalsComb),globalCrystalId1,globalCrystalId2);
                % Doi
%                 meanDoiCrystalX1(ind) = meanDoiCrystalX1(ind) + coincidenceMatrix(:,colDetectionX1);
%                 meanDoiCrystalX2(ind) = meanDoiCrystalX2(ind) + coincidenceMatrix(:,colDetectionX2);
%                 meanDoiCrystalY1(ind) = meanDoiCrystalY1(ind) + coincidenceMatrix(:,colDetectionY1);
%                 meanDoiCrystalY2(ind) = meanDoiCrystalY2(ind) + coincidenceMatrix(:,colDetectionY2);
%                 meanDoiCrystalZ1(ind) = meanDoiCrystalZ1(ind) + coincidenceMatrix(:,colDetectionZ1);
%                 meanDoiCrystalZ2(ind) = meanDoiCrystalZ2(ind) + coincidenceMatrix(:,colDetectionZ2);
                meanDoiCrystalX1(:) = meanDoiCrystalX1(:) + accumarray(ind, coincidenceMatrix(:,colDetectionX1),[numel(histCrystalsComb),1]);
                meanDoiCrystalX2(:) = meanDoiCrystalX2(:) + accumarray(ind, coincidenceMatrix(:,colDetectionX2),[numel(histCrystalsComb),1]);
                meanDoiCrystalY1(:) = meanDoiCrystalY1(:) + accumarray(ind, coincidenceMatrix(:,colDetectionY1),[numel(histCrystalsComb),1]);
                meanDoiCrystalY2(:) = meanDoiCrystalY2(:) + accumarray(ind, coincidenceMatrix(:,colDetectionY2),[numel(histCrystalsComb),1]);
                meanDoiCrystalZ1(:) = meanDoiCrystalZ1(:) + accumarray(ind, coincidenceMatrix(:,colDetectionZ1),[numel(histCrystalsComb),1]);
                meanDoiCrystalZ2(:) = meanDoiCrystalZ2(:) + accumarray(ind, coincidenceMatrix(:,colDetectionZ2),[numel(histCrystalsComb),1]);
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
%% SAVE RESULTS
save([outputPath 'meanDoiPerCrystal'], 'meanDoiCrystalX1', 'meanDoiCrystalX2', 'meanDoiCrystalY1', 'meanDoiCrystalY2', 'meanDoiCrystalZ1', 'meanDoiCrystalZ2', 'histCrystalsComb', '-v7.3');
%% CONVERT INTO SINOGRAMS
% Save the mean doi per crystal in a span 1 sinogram because needs less
% memory, create sinogram, store data and clear the histogram.
indInHist = sub2ind(size(histCrystalsComb),mapaDet1Ids(:), mapaDet2Ids(:));
sinogram = single(zeros(structSizeSino3D.numR,structSizeSino3D.numTheta, sum(structSizeSino3D.sinogramsPerSegment)));
sinogram(:) = histCrystalsComb(indInHist);

sinoMeanDoiCrystalX1 = single(zeros(structSizeSino3D.numR,structSizeSino3D.numTheta, sum(structSizeSino3D.sinogramsPerSegment)));
sinoMeanDoiCrystalX1(:) = meanDoiCrystalX1(indInHist);
sinoMeanDoiCrystalX1(sinogram~=0) = sinoMeanDoiCrystalX1(sinogram~=0)./sinogram(sinogram~=0);
clear meanDoiCrystalX1;

sinoMeanDoiCrystalX2 = single(zeros(structSizeSino3D.numR,structSizeSino3D.numTheta, sum(structSizeSino3D.sinogramsPerSegment)));
sinoMeanDoiCrystalX2(:) = meanDoiCrystalX2(indInHist);
sinoMeanDoiCrystalX2(sinogram~=0) = sinoMeanDoiCrystalX2(sinogram~=0)./sinogram(sinogram~=0);
clear meanDoiCrystalX2;

sinoMeanDoiCrystalY1 = single(zeros(structSizeSino3D.numR,structSizeSino3D.numTheta, sum(structSizeSino3D.sinogramsPerSegment)));
sinoMeanDoiCrystalY1(:) = meanDoiCrystalY1(indInHist);
sinoMeanDoiCrystalY1(sinogram~=0) = sinoMeanDoiCrystalY1(sinogram~=0)./sinogram(sinogram~=0);
clear meanDoiCrystalY1;

sinoMeanDoiCrystalY2 = single(zeros(structSizeSino3D.numR,structSizeSino3D.numTheta, sum(structSizeSino3D.sinogramsPerSegment)));
sinoMeanDoiCrystalY2(:) = meanDoiCrystalY2(indInHist);
sinoMeanDoiCrystalY2(sinogram~=0) = sinoMeanDoiCrystalY2(sinogram~=0)./sinogram(sinogram~=0);
clear meanDoiCrystalY2;

sinoMeanDoiCrystalZ1 = single(zeros(structSizeSino3D.numR,structSizeSino3D.numTheta, sum(structSizeSino3D.sinogramsPerSegment)));
sinoMeanDoiCrystalZ1(:) = meanDoiCrystalZ1(indInHist);
sinoMeanDoiCrystalZ1(sinogram~=0) = sinoMeanDoiCrystalZ1(sinogram~=0)./sinogram(sinogram~=0);
clear meanDoiCrystalZ1;

sinoMeanDoiCrystalZ2 = single(zeros(structSizeSino3D.numR,structSizeSino3D.numTheta, sum(structSizeSino3D.sinogramsPerSegment)));
sinoMeanDoiCrystalZ2(:) = meanDoiCrystalZ2(indInHist);
sinoMeanDoiCrystalZ2(sinogram~=0) = sinoMeanDoiCrystalZ2(sinogram~=0)./sinogram(sinogram~=0);
clear meanDoiCrystalZ2;

% % Vreify coordinates:
% mapaDet1Ids_2d = mapaDet1Ids(:,:,1);
% mapaDet2Ids_2d = mapaDet2Ids(:,:,1);
% % Get mean values:
% num_crystals_per_ring = 504;
% % Internal radius:
% radius_mm = 328;
% % entry coordinates per crystal:
% crystal_sep_mm = 4.0890;
% crystal_size_mm = 4;
% offsetBlockAngle_deg = 3.2109; % This was the rotation angle for the first block.
% blockSize_mm = 32.7120;
% stepAngle_deg = radtodeg(crystal_sep_mm./radius_mm); % Step for each crystal
% offsetBlockAngle_deg = stepAngle_deg.*1; % The offset is one crystal (half the gap+half the first pixel).
% rotationAngle_deg = offsetBlockAngle_deg + stepAngle_deg.* [0 : num_crystals_per_ring-1];
% % Coordinates per crystal, first crystal:
% x1_crystals = radius_mm .* sind(rotationAngle_deg);
% y1_crystals = -radius_mm .* cosd(rotationAngle_deg);
% figure; plot(x1_crystals,y1_crystals, 'o'); title('Crystal coordinates'); axis equal; hold on;
% plot(sinoMeanDoiCrystalX1(1:8*344*252),sinoMeanDoiCrystalY1(1:8*344*252), 'x'); 
%% SAVE RESULTS
save([outputPath 'sinoMeanDoiPerCrystal'], 'sinoMeanDoiCrystalX1', 'sinoMeanDoiCrystalX2', 'sinoMeanDoiCrystalY1', 'sinoMeanDoiCrystalY2', 'sinoMeanDoiCrystalZ1', 'sinoMeanDoiCrystalZ2', 'sinogram', '-v7.3');
