%  *********************************************************************
%  Proyecto AR-PET. Comisión Nacional de Energía Atómica.
%  Autor: Martín Belzunce. UTN-FRBA.
%  Fecha de Creación: 23/08/2011
%  *********************************************************************
%	            GENERACIÓN DE SINOGRAMA 2D A PARTIR DE SIMULACIÓN GATE
%  function sinograms2D = getSinograms2Dideal(outputPath, structSimu, rebinningMethod, graficarOnline)
%
%  Función genera sinogramas 2D a partir de una simulación del AR-PET. Los
%  datos de la simulación se brindan a través de la estructura structSimu,
%  que contiene el path, el nombre de la simulación y la salida, la
%  cantidad de splits, y los arhcivos por split de la simulación. Para
%  generar dicha estructura utilizar la función getSimuStruct() disponible
%  en esta librería de funciones de procesamiento de simulaciones. Se
%  genera un sinograma 2D por cada slice del sistema, utilizando un método
%  de rebinning (SSRB o MSRB) y se los guarda en formato interfile. El
%  tamaño de los sinogramas se pasa con la estrctura structSizeSino2D, la
%  misma se genera con la función getSizeSino2Dstruct() que se encuentra en
%  MATLAB/ImageRecon. Además para la generación de los sinogramas se 
%  le debe pasar la resolución espacial del sistema, y las zonas muertas en
%  los cabezales.
%  Es igual que getSinograms2D pero solo tiene en cuenta los eventos como
%  ideales, es decís no degarada espacialmente, no incluye zonas muertas ni
%  hace ningún procesamiento adicional.
%
%  Detalle de los parámetros de entrada:
%   - outputPath: directorio de salida donde se guardarán los sinogramas de
%   salida.
%   - structSimu: estructura con los datos de la simulación. Esta
%   estructura se genera con la función getSimuStruct().
%   - energyWindow: Agrego la ventana de energía en un vector con [LimInf
%   LimSup].
%   - structSizeSino2D: estructura con el tamaño de sinogramas 2d.
%   - rebinningMethod: string con el método para el rebinning, los valores
%   válidos son 'ssrb' y 'msrb'.
%   - graficarOnline: flag que indica si se desea graficar los resultados
%   durante el procesamiento.
%
%  Ejemplo de llamada:
%   sinograms2D = getSinograms2D(outputPath, structSimu, resEspacial, rebinningMethod, zonasMuertas_mm, graficarOnLine)

function [sinograms2D, TiempoSimulacion] = getSinograms2Dideal(outputPath, structSimu, energyWindow, structSizeSino2D, rebinningMethod, graficarOnline)

%% OTRAS FUNCIONES QUE SE UTILIZAN
% Agrego el path donde tengo algunas funciones de uso general. En este caso
% hist4.
addpath('/sources/MATLAB/FuncionesGenerales');
addpath('/sources/MATLAB/VersionesFinales');
addpath('/sources/MATLAB/WorkingCopy/ImageProcessing')
addpath('/sources/MATLAB/WorkingCopy/ImageRecon')

%% CONSTANTES GEOMÉTRICAS DEL CRISTAL
cantCabezales = 6;
cristalSizeY_mm = 304.8;   % Ancho del cristal 304.8mm.
cristalSizeX_mm = 406.4;   % Ancho del cristal 406.4mm.
espesorCristal_mm = 25.4;
ladoHexagonoScanner_mm = 415.69;
distCentroCabezal_mm = 360;
%%  VARIABLES PARA GENERACIÓN DE SINOGRAMAS 2D
sinograms2D = zeros(structSizeSino2D.numTheta, structSizeSino2D.numR, ...
            structSizeSino2D.numZ);
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

                % Si no tengo datos, salgo:
                if isempty(coincidenceMatrix)
                    break;
                end

                % Grafico La energía filtrada
                HistEnergiasFilt = HistEnergiasFilt + hist([coincidenceMatrix(:,colEnergy1); coincidenceMatrix(:,colEnergy2)], CanalesEnergia);
                if(graficarOnline)
                    figure(2);
                    bar(CanalesEnergia, HistEnergiasFilt);
                    title('Histograma de Energías luego de Filtro de Ventana');
                end

                % Proceso la posición de los eventos para generar sinogramas:
                posDet = [coincidenceMatrix(:,colDetectionX1) coincidenceMatrix(:,colDetectionY1) coincidenceMatrix(:,colDetectionZ1) ...
                    coincidenceMatrix(:,colDetectionX2) coincidenceMatrix(:,colDetectionY2) coincidenceMatrix(:,colDetectionZ2)];
                %% REBINNING
                % Ya tengo todos los datos filtrados. Ahora debo aplicar el
                % rebinning. Hice una modificación y todos los métodos
                % de rebinning reciben los eventos enteros, esto es por
                % una cuestión de compatibilidad.
                switch(rebinningMethod)
                    case 'ssrb'
                        % Rebinning ssrb.
                        posDet = ssrb(posDet);
                        %% LLENADO DE SINOGRAMAS 2D
                        % Ahora lleno los sinogramas. Para esto convierto el par de
                        % eventos (X1,Y1,Z1) y (X2,Y2,Z2) en lors del tipo (Thita,r,z).
                        % El ángulo thita está determinado por atan((Y1-Y2)/(X1-X2))+90
                        theta = atand((posDet(:,2)-posDet(:,5))./(posDet(:,1)-posDet(:,4))) + 90;
                        % El offset r lo puedo obtener reemplazando (x,y) con alguno de
                        % los dos puntos en la ecuación: r=x*cos(thita)+y*sin(thita)-
                        r_sino = cosd(theta).*posDet(:,1) + sind(theta).*posDet(:,2);
                        % Acumulo todo en el array de sinogramas 2D utilizando la
                        % función creada hist4.
                        sinograms2D = sinograms2D + hist4([theta r_sino posDet(:,3)], ...
                            {structSizeSino2D.thetaValues_deg structSizeSino2D.rValues_mm structSizeSino2D.zValues_mm});
                    case 'msrb' 
                        % Rebinning msrb.
                        sinograms2D = msrb(posDet, structSizeSino2D);
                end
                %% LLENADO DE SINOGRAMAS 2D
                if(graficarOnline)
                    figure(6);
                    imshow(sinograms2D(:,:,round(structSizeSino2D.numZ/2))/max(max(sinograms2D(:,:,round(structSizeSino2D.numZ/2)))));
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
%% GUARDO SINOGRAMAS A DISCO
% Se guardan los sinogramas en formato interfile. Para esto utilizo la
% función interfileWriteSino.
filename = sprintf('%s/sinograms2dIdeal_%s', outputPath, rebinningMethod);
interfileWriteSino(single(sinograms2D), filename);


%% VISUALIZACIÓN DE ALGUNAS IMÁGENES
figure(1);
bar(CanalesEnergia, HistEnergias);
title('Histograma de Energías');

figure(2);
bar(CanalesEnergia, HistEnergiasFilt);
title('Histograma de Energías Filtrado');
%% VISUALIZACIÓN DE SINOGRAMAS 2D
% Visualizo los sinogramas 2D para cada slice.

% Ahora si lo visualizo:
image = getImageFromSlices(sinograms2D, 8);
h = figure;
set(gcf, 'Position', [100 100 1600 800]);
imshow(image./max(max(image)));

