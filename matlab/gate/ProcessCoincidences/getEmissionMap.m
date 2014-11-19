%  *********************************************************************
%  Proyecto AR-PET. Comisión Nacional de Energía Atómica.
%  Autor: Martín Belzunce. UTN-FRBA.
%  Fecha de Creación: 23/08/2011
%  *********************************************************************
%	            MAPA DE EMISIÓN DE SIMULACIÓN GATE
% function emissionMap = getEmissionMap(PathSimu, NombreMedicion,
% NombreSalida, CantSplits, ArchivosPorSplit)
%
%  Función que obtiene un mapa de emisión de una simulación del AR-PET. No
%  tiene en cuenta ningún filtro de energía. Simplemente filtra los eventos
%  trues para tener un mapa de emisión correcto. La idea de esta función es
%  chequear que las fuentes de la simulación se han seteado adecuadamente.
%
%  Los parámetros que recibe son:
%
%   structSimu: estructura con los datos de la simulación.
%
%   GraficarOnLine: realiza el gráfico de los histogramas de emisión en
%   cada coordenada mientras procesa.
%
%  Ejemplo de llamada:
%   emissionMap = getEmissionMap( '/datos/Simulaciones/PET/AR-PET/SimulacionCylinder080711', 'Cylinder', 'Coincidences', 54, 1)

function emissionMap = getEmissionMap(structSimu, GraficarOnLine)

%% OTRAS FUNCIONES QUE SE UTILIZAN
% Agrego el path donde tengo algunas funciones de uso general. En este caso
% hist4.
addpath('/sources/MATLAB/FuncionesGenerales');
addpath('/sources/MATLAB/VersionesFinales');
addpath('/sources/MATLAB/WorkingCopy/ImageProcessing')

%%  VARIABLES PARA MAPA DE EMISIÓN
% PAra el mapa de emisión se necesita definir un radio y largo del campo de
% visión, que en el caso del AR-PET lo vamos a considerar de 300 mm para
% ambos casos.
zFov = 300;
rFov = 300;
% Valores de cada coordenada, a partir del fov definido:
valoresX = -rFov : 2 : rFov;                   % Valores posibles de emisión de positrones en la variable Z (O sea, subre todo el FOV).
valoresY = -rFov : 2 : rFov;                   % Valores posibles de emisión de positrones en la variable Z (O sea, subre todo el FOV).
valoresZ = -zFov/2 : 5 : zFov/2;                   % Valores posibles de emisión de positrones en la variable Z (O sea, subre todo el FOV).

% Histograma de emisiones en cada coordenada:
histEmisionesX = zeros(1,numel(valoresX));         % Inicialización de histograma de emisión en X.
histEmisionesY = zeros(1,numel(valoresY));         % Inicialización de histograma de emisión en Y.
histEmisionesZ = zeros(1,numel(valoresZ));         % Inicialización de histograma de emisión en Z.

% Mapa tridimensional del Fov completo:
emissionMap = zeros(numel(valoresY), numel(valoresX), numel(valoresZ));
%% CONSTANTES GEOMÉTRICAS DEL CRISTAL
CristalSizeY = 304.8;   % Ancho del cristal 304.8mm.
CristalSizeX = 406.4;   % Ancho del cristal 406.4mm.

%% INICIALIZACIÓN DE VARIABLES EXCLUSIVAS DEL COINCIDENCE
eventosTotales = 0;

TiempoSplitsAnteriores = 0; % Variable que acumula los tiempos de adquisición de los aplits ya procesados.
TiempoTotal = 0;            % Variable que indica el tiempo total de la simulación.

% Índices de columnas en el archivo de salida:
colEventId1 = 2;
colEventId2 = 2 + 23;
colTimeStamp1 = 7;
colTimeStamp2 = 7 + 23;
colEnergy1 = 8;
colEnergy2 = 8 + 23;
colEmisionX1 = 4;
colEmisionX2 = 4 + 23;
colEmisionY1 = 5;
colEmisionY2 = 5 + 23;
colEmisionZ1 = 6;
colEmisionZ2 = 6 + 23;
colDetectionX1 = 9;
colDetectionX2 = 9 + 23;
colDetectionY1 = 10;
colDetectionY2 = 10 + 23;
colDetectionZ1 = 11;
colDetectionZ2 = 11 + 23;
colVolIdBase1 = 12;
colVolIdBase2 = 12 + 23;
colVolIdRsector1 = 13;
colVolIdRsector2 = 13 + 23;
colVolIdModule1 = 14;
colVolIdModule2 = 14 + 23;
colCompton1 = 18;
colCompton2 = 18 + 23;

%% PROCESAMIENTO DE COINCIDENCES
for i = 1 : structSimu.numSplits
    disp(sprintf('################ PROCESANDO SPLIT %d ###################',i));
    % Nombre Base del Archivo de Salida para el Split i
    if structSimu.numSplits>1
        % Si se ha hecho el split de la simulación en varias simulaciones
        % tengo que usar el formato de nombre que le da el splitter.
        NombreBase = sprintf('%s%d%s',structSimu.name,i,structSimu.digiName); 
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
        disp(sprintf('################ PROCESANDO ARCHIVO %d DEL SPLIT %d ###################',j,i));
        FID = fopen(NombreArch, 'r');
        if (FID == -1)
            disp('Error al abrir el archivo');
            break;
        end
        disp(sprintf('Iniciando el procesamiento del archivo %s.', NombreArch));
        while feof(FID) == 0
            datos = textscan(FID, '%f', 400000 * 46);
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
            TiempoTotal = TiempoSplit + TiempoSplitsAnteriores;
            disp('');
            disp(sprintf('Tiempo Inicial Split: %d', TiempoInicialSplit));
            disp(sprintf('Tiempo Medición Split: %d', TiempoSplit));
            disp(sprintf('Tiempo Total Simulación: %d', TiempoTotal));
            disp(sprintf('\n\n'));
            disp('%%%%%%%%%%% VENTANA DE ENERGÍA %%%%%%%%%%%');
            eventosTotales = eventosTotales + eventosLeidos;
            disp(sprintf('Coincidencias leídas: %d', eventosTotales));
            
            % Como quiero hacer un mapa de emisión para verificar fantomas,
            % solo necesito los eventos trues:
            indicesTrues = (coincidenceMatrix(:,colEmisionX1) == coincidenceMatrix(:,colEmisionX2)) & (coincidenceMatrix(:,colEmisionY1) == coincidenceMatrix(:,colEmisionY2)) ...
                & (coincidenceMatrix(:,colEmisionZ1) == coincidenceMatrix(:,colEmisionZ2));
            
            % Hago el mapa de emisión en cada una de las coordenadas:
            histEmisionesX = histEmisionesX + hist(coincidenceMatrix(indicesTrues,colEmisionX1), valoresX);
            histEmisionesY = histEmisionesY + hist(coincidenceMatrix(indicesTrues,colEmisionY1), valoresY);
            histEmisionesZ = histEmisionesZ + hist(coincidenceMatrix(indicesTrues,colEmisionZ1), valoresZ);
            
            if(GraficarOnLine)
                % Los grafico:
                figure(1);
                bar(valoresX, histEmisionesX);
                title('Histograma de Emisiones en X');

                figure(2);
                bar(valoresY, histEmisionesY);
                title('Histograma de Emisiones en Y');

                figure(3);
                bar(valoresZ, histEmisionesZ);
                title('Histograma de Emisiones en Z');
            end
            % Por último, mapa de emisión 3d:
            emissionMap = emissionMap + hist4([coincidenceMatrix(indicesTrues,colEmisionY1), coincidenceMatrix(indicesTrues,colEmisionX1), coincidenceMatrix(indicesTrues,colEmisionZ1)],...
                {valoresY, valoresX, valoresZ});
            
             if(GraficarOnLine)
                % Los grafico:
                image = getImageFromSlices(emissionMap, 8);
                figure(4);
                set(gcf, 'Position', [100 100 1600 800]);
                imshow(image./max(max(image)));
             end
            
            %% FIN DEL LOOP
        end
        fclose(FID);
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


%% VISUALIZACIÓN DE MAPA  DE EMISIÓN EN FOV
% Visualizo los distintos cortes del mapa de emisión. Debo tener en cuenta
% que la coordenada Y (las filas) y la coordenada Z están invertida, ya que
% el histograma solo permite utilizar valores ascendentes pero en la
% realidad la variación del índice de filas es opuesto al de Y.

% Ahora si lo visualizo:
figure;
maxEmission = max(max(max(emissionMap)));
for i = 1 : size(emissionMap,3)
    imshow(emissionMap(:,:,i)./ maxEmission);
    title(sprintf('Mapa de emisión para z = %f', valoresZ(i)));
    pause(0.3);
end

% También genero una única imagen con los slices para la visualización:
image = getImageFromSlices(emissionMap, 8);
h = figure;
set(gcf, 'Position', [100 100 1600 800]);
imshow(image./max(max(image)));
