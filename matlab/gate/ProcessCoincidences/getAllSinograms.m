%  *********************************************************************
%  Proyecto AR-PET. Comisión Nacional de Energía Atómica.
%  Autor: Martín Belzunce. UTN-FRBA.
%  Fecha de Creación: 23/08/2011
%  *********************************************************************
%	            GENERACIÓN DE SINOGRAMA 2D A PARTIR DE SIMULACIÓN GATE
%  function getAllSinograms(outputPath, structSimu, resEspacial, structSizeSino2D, resEspacialFwhm_mm, rebinningMethod, zonasMuertas_mm, graficarOnline)
%
%  Función genera sinogramas 2D, Michelogramas y datos en Modo Lista a 
%  partir de una simulación del AR-PET. Los
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
%  Para el michelograma, se tiene en cuenta lo mismo que para los
%  sinogramas 2D, la cantidad de "anillos" z1 y z2 es el mismo que la
%  cantidad de slices de los sinogramas 2D.
%  Los archivos en modo lista está en ASCII, y tiene los siguientes campos:
%  [X1 Y1 Z1 E1 T1 X2 Y2 Z2 E2 T2]
%  Donde x,y,z son las coordenadas en el world en mm, la energía de cada
%  evento en kev y el timestamp.
%  Detalle de los parámetros de entrada:
%   - outputPath: directorio de salida donde se guardarán los sinogramas de
%   salida.
%   - structSimu: estructura con los datos de la simulación. Esta
%   estructura se genera con la función getSimuStruct().
%   - structSizeSino2D: estructura con el tamaño de sinogramas 2d.
%   - resEspacial_mm: resolución espacial en FWHM en mm.
%   - rebinningMethod: string con el método para el rebinning, los valores
%   válidos son 'ssrb' y 'msrb'.
%   - zonaMuerta_mm: vector con zona muerta en los bordes del cristal en x
%   e y. Siendo x el lado largo del cabezal (406.4 mm) e y el lado corto
%   (304.8 mm).
%   - graficarOnline: flag que indica si se desea graficar los resultados
%   durante el procesamiento.
%
%  Ejemplo de llamada:
%   getAllSinograms(outputPath, structSimu, resEspacial, structSizeSino2D,
%    structSizeSino3D, rebinningMethod, zonasMuertas_mm, graficarOnline)
function getAllSinograms(outputPath, structSimu, structSizeSino2D, structSizeSino3D, resEspacialFwhm_mm, rebinningMethod, zonaMuerta_mm, graficarOnline)


%% OTRAS FUNCIONES QUE SE UTILIZAN
% Agrego el path donde tengo algunas funciones de uso general. En este caso
% hist4.
addpath('/sources/MATLAB/FuncionesGenerales');
addpath('/sources/MATLAB/VersionesFinales');
addpath('/sources/MATLAB/WorkingCopy/ImageProcessing')
addpath('/sources/MATLAB/WorkingCopy/ImageRecon')
addpath('/sources/MATLAB/WorkingCopy/utils')
%% CONSTANTES GEOMÉTRICAS DEL CRISTAL
cantCabezales = 6;
cristalSizeY_mm = 304.8;   % Ancho del cristal 304.8mm.
cristalSizeX_mm = 406.4;   % Ancho del cristal 406.4mm.
espesorCristal_mm = 25.4;
ladoHexagonoScanner_mm = 415.69;
distCentroCabezal_mm = 360;
%%  VARIABLES PARA GENERACIÓN DE SINOGRAMAS 2D
indicesZonaMuertaAcumulado = cell(numel(resEspacialFwhm_mm), size(zonaMuerta_mm,1));
% Matriz de 3 dimensionez donde se guardarán los sinogramas 2d:
sinograms2D = cell(numel(resEspacialFwhm_mm), size(zonaMuerta_mm,1));
for r = 1 : numel(resEspacialFwhm_mm)
    for zm = 1 : size(zonaMuerta_mm,1)
        sinograms2D{r,zm} = single(zeros(structSizeSino2D.numTheta, structSizeSino2D.numR, ...
            structSizeSino2D.numZ));
    end
end
%%  VARIABLES PARA GENERACIÓN DE MICHELOGRAMA
% Cell array con todos los michelgoramas:
michelograms = cell(numel(resEspacialFwhm_mm), size(zonaMuerta_mm,1));
for r = 1 : numel(resEspacialFwhm_mm)
    for zm = 1 : size(zonaMuerta_mm,1)
        % Matriz de 4 dimensionez para cada michelograma:
        michelograms{r,zm} = single(zeros(structSizeSino3D.numTheta, structSizeSino3D.numR, ...
            structSizeSino3D.numZ, structSizeSino3D.numZ));
    end
end
%%  VARIABLES PARA GENERACIÓN DE DATOS EN MODO LISTA
% Los mismos se van escribiendo a medida que se generan los datos. Pero
% primero debo abrir los archivos y generar los fid:
for r = 1 : numel(resEspacialFwhm_mm)
    for zm = 1 : size(zonaMuerta_mm,1)
        fileNameModoLista{r,zm} = sprintf('%s/listMode_res%.2fmm_zm%.2fmm_%.2fmm.dat', outputPath, resEspacialFwhm_mm(r), zonaMuerta_mm(zm,:));
        fidsModoLista{r,zm} = fopen(fileNameModoLista{r,zm}, 'w');
        numEventoModoLista{r, zm} = 0;
    end
end

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
planoXYcabezalZonaMuerta = cell(numel(resEspacialFwhm_mm), size(zonaMuerta_mm,1), cantCabezales);
for i = 1 : cantCabezales
    planoDetXY{i} = zeros(numel(valoresYX{1}), numel(valoresYX{2}));
    planoXYcabezal{i} = zeros(numel(valoresYXcabezal{1}), numel(valoresYXcabezal{2}));
    for r = 1 : numel(resEspacialFwhm_mm)
        for zm = 1 : size(zonaMuerta_mm,1)
            planoXYcabezalZonaMuerta{r,zm,i} = zeros(numel(valoresYXcabezal{1}), numel(valoresYXcabezal{2}));
        end
    end
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
TiempoTotal = 0;            % Variable que indica el tiempo total de la simulación.

% Variables para histogramas y filtrado de energía:
%  Variables que definen las ventanas de energía, siempre existe una
%  ventana de energía sobre el pico que denominamos A, y es opcional una
%  ventana B en la zona de energía que hay poco compton (aprox 350-350keV).
VentanaAsup = 0.580; % Límite superior de la ventana de energía del pico.
VentanaAinf = 0.430;    % Límite inferior de la ventana de energía en pico.
HabilitaVentanaB = 0;   % Habilita la utilización de la ventana de energía
%  secundaria B.
CanalesEnergia =  0:0.001:2.0;  % Canales de Energía para el histograma.
HistEnergias = zeros(1,numel(CanalesEnergia));  % Inicialización en cero del Histograma de Energías.
HistEnergiasFilt = zeros(1,numel(CanalesEnergia));  % Inicialización en cero del Histograma de Energías Filtrado.
HistEnergiasSinCompton = zeros(1,numel(CanalesEnergia));  % Inicialización en cero del Histograma de Energías de Eventos Cin Comptons.


%% INICIALIZACIÓN DE VARIABLES A UTILIZAR EN EL PROCESAMIENTO DEL COINCIDENCE
eventosTotales = 0;

TiempoSplitsAnteriores = 0; % Variable que acumula los tiempos de adquisición de los aplits ya procesados.
TiempoTotal = 0;            % Variable que indica el tiempo total de la simulación.

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
    fprintf('################ PROCESANDO SPLIT %d ###################',i);
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
        fprintf('################ PROCESANDO ARCHIVO %d DEL SPLIT %d ###################',j,i);
        FID = fopen(NombreArch, 'r');
        if (FID == -1)
            disp('Error al abrir el archivo');
        end
        fprintf('Iniciando el procesamiento del archivo %s.', NombreArch);
        while feof(FID) == 0
            datos = textscan(FID, '%f', 20000 * 46);
            if mod(numel(datos{1}), 46) ~= 0
                % El archivo no esta entero, puede que se haya cortado
                % por una finalizacion abrupta de la simulacion.
                % Elimino la coincidencia incompleta
                datos{1}(end-(mod(numel(datos{1}),46)-1) : end) =[];
            end
            % Le doy forma de matriz a todos los datos leídos.
            coincidenceMatrix = single(reshape(datos{1}, 46, numel(datos{1})/46)');
            % Borro datos para liberar memoria:
            clear datos
            
            
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
            fprintf('Tiempo Inicial Split: %d', TiempoInicialSplit);
            fprintf('Tiempo Medición Split: %d', TiempoSplit);
            fprintf('Tiempo Total Simulación: %d', TiempoTotal);
            fprintf('\n\n');
            disp('%%%%%%%%%%% VENTANA DE ENERGÍA %%%%%%%%%%%');
            eventosTotales = eventosTotales + eventosLeidos;
            fprintf('Coincidencias leídas: %d', eventosTotales);
            
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
            fprintf('Coincidencias en ventana de energía: %d', eventosTotalesEnVentana);
            
            % Grafico La energía filtrada
            HistEnergiasFilt = HistEnergiasFilt + hist([coincidenceMatrix(:,colEnergy1); coincidenceMatrix(:,colEnergy2)], CanalesEnergia);
            if(graficarOnline)
                figure(2);
                bar(CanalesEnergia, HistEnergiasFilt);
                title('Histograma de Energías luego de Filtro de Ventana');
            end
            
            % Proceso la posición de los eventos para generar sinogramas:
            %% PROCESAMIENTO DE POSICIONES DE LOS EVENTOS 
            % Para el procesamiento de las coincidencias, primero proyecto
            % todas las posiciones sobre la mitad del cristal, suponiendo
            % que esa es una de las zonas más probables de interacción, y
            % es la que en principio se consideraría en una medición.
            % Luego con las posiciones proyectadas, hago un cambio de
            % coordenadas para cada cristal de manera de obtener las
            % posiciones (X,Y) dentro de cada cabezal.
            % Tercero, con las coordenadas de los eventos sobre el cristal,
            % aplico el filtro de zona muerta, eliminando los elementos que
            % caen en el borde del cristal con un ancho definido con uno de
            % los parámetros de entrada.
            % Cuarto, con los datos que han pasado el filtro de zona
            % muerta, aplico el método de rebinning pasado como parámetro
            % para transofrmar eventos oblicuos en eventos directos.
            % Por último, acumulo dichos eventos en sinogramas.
            
            % El procesamiento lo separo por cabezal, ya que las distintas
            % proyecciones y transformaciones de coordenadas dependen del
            % sistema.
            % Proceso primero todos los primeros singles dentro de las
            % coincidencias, y luego el segundo.
            % Al hacer este procesamiento secuencial debo generar las
            % variables antes. Voy a hacer una copia de las posiciones de
            % detección para cada uno de los casos de resolución espacial y
            % zona muerta, y los guardo en un cell array. Ya que primero lo
            % degrado espacialmente y luego debo filtrar las posiciones de
            % la zona muerta. Por lo que las distintas matrices son de
            % distinto tamaño (por esto utilizo un cell array).
            % Creo el cell array:
            posDet = cell(numel(resEspacialFwhm_mm), size(zonaMuerta_mm,1));
            % Copio en cada elemento las posiciones en columnas
            % [X1,Y1,Z1,X2,Y2,Z2]:
            for r = 1 : numel(resEspacialFwhm_mm)
                for zm = 1 : size(zonaMuerta_mm,1)
                    posDet{r, zm} = coincidenceMatrix(:,[colDetectionX1 colDetectionY1 colDetectionZ1 ...
                        colDetectionX2 colDetectionY2 colDetectionZ2]);
                end
            end
            % Por un lado obtengo las posiciones degradadas, y por otro los
            % índices de los elementos a eliminar. Ya que no los puedo ir
            % eliminando secuencialmente, porque pierdo la referencia delos
            % índices. Creo el cell array de índices:
            indicesZonaMuertaAcumulado = cell(numel(resEspacialFwhm_mm), size(zonaMuerta_mm,1));
            for r = 1 : numel(resEspacialFwhm_mm)
                for zm = 1 : size(zonaMuerta_mm,1)
                    indicesZonaMuertaAcumulado{r, zm} = false(size(coincidenceMatrix,1),1);
                end
            end
            % Ahora si hago el procesamiento.
            for indEvento = 0 : 1
                % Recorro cada cabezal.
                for cabezal = 0 : cantCabezales-1  
                    % Filtro los eventos por ID y por cabezal.
                    indicesCabezal = coincidenceMatrix(:,colVolIdRsector1+numColumnasPorSingle*indEvento) == cabezal;
                    
                    % Acumulo las posicione en una imagen de proyección que
                    % usamos para verificar:
                    planoDetXY{cabezal+1} = planoDetXY{cabezal+1} + hist3([coincidenceMatrix(indicesCabezal,colDetectionY1+numColumnasPorSingle*indEvento) ...
                        coincidenceMatrix(indicesCabezal,colDetectionX1+numColumnasPorSingle*indEvento)], valoresYX);
                    if(graficarOnline)
                        figure(3);
                        imshow(planoDetXY{cabezal+1}/max(max(planoDetXY{cabezal+1})));
                        title(sprintf('Corte en el plano XY de los eventos recibidos'));
                    end
                    %% PROYECCIÓN SOBRE EL PUNTO MEDIO DEL CRISTAL
                    % Todos los eventos los proyectos sobre la profundidad
                    % media del cabezal. Como no tenemos profundidad de
                    % interacción le asignamos el punto medio de la
                    % profundidad. Luego habría que ajustar a la posición más
                    % probable de interacción, que no sabemos realmente si es
                    % el punto medio.
                    % Transformo el Xworl, Yworld, Zworld en el Xevent,Yevent,
                    % Zevent que va a ser el que yo consideraría si estuviera
                    % haciendo una medición. 
                    % La profundidad de interacción afecta al plano xy de la
                    % simulación, y la proyección sobre la profundidad media
                    % dependerá de la posición del cabezal.
                    % Para la proyección se utiliza el parámetro Yop. Para
                    % los cabezales rectos, no es necesario Yop ya que es
                    % directo.
                    % El z se mantiene igual: Zevento = Zworld
                    
                    % Por otro lado, se calcula offsetX y offsetY, que son
                    % los valores de la coordenada central del cabezal para
                    % proyectar a un único plano (x,y) sobre el cabezal.
                    % cabezal 1:    
                    %   Xevento = 37,27
                    %   Yevento = Yworld

                    % cabezal2:
                    % Xevento = (Yworld+Xworld/tan(30)+37,46)*sen(30(*cos(30)
                    % Yevento = tan(30)*Xevento + 37,46
                    % Cada cabezal es de 406,4mm en el lado que forma el
                    % hexágono, pero están levemente separados siendo el lado
                    % del hexágono 415,692mm.
                    Zevento = coincidenceMatrix(indicesCabezal,colDetectionZ1+numColumnasPorSingle*indEvento);
                    Proyectar = 0;        % Variable que me dice si es necesario hacer una proyección
                                                    % de los eventos sobre el valor medio de la
                                                    % profundidad de interacción.
                    switch cabezal
                        case 0
                            Xeventos = ones(size(coincidenceMatrix(indicesCabezal,colDetectionX1+numColumnasPorSingle*indEvento))).*...
                                (distCentroCabezal_mm+espesorCristal_mm/2);
                            Yeventos = coincidenceMatrix(indicesCabezal,colDetectionY1+numColumnasPorSingle*indEvento);
                            Xcabezal = Yeventos;
                        case 1
                            AnguloCabezal = 30;
                            Yop = -(ladoHexagonoScanner_mm + (espesorCristal_mm/2) / cosd(30));
                            OffsetY =  -( (espesorCristal_mm/2) * cosd(30) + (cristalSizeX_mm/2) + ...
                                (ladoHexagonoScanner_mm-(cristalSizeX_mm/2)) *sind(30));
                            OffsetX = (espesorCristal_mm/2) * sind(30) + (cristalSizeX_mm * cosd(30)./2);
                            Proyectar = 1;
                        case 2
                            AnguloCabezal = 150;
                            Yop = -(ladoHexagonoScanner_mm + (espesorCristal_mm/2) / cosd(30));
                            OffsetY =  -( (espesorCristal_mm/2) * cosd(30) + (cristalSizeX_mm/2) + (ladoHexagonoScanner_mm-(cristalSizeX_mm/2)) *sind(30));
                            OffsetX = -( (espesorCristal_mm/2) * sind(30) + (cristalSizeX_mm * cosd(30)./2));
                            Proyectar = 1;
                        case 3
                            Xeventos = -ones(size(coincidenceMatrix(indicesCabezal,colDetectionX1+numColumnasPorSingle*indEvento))).*...
                                (distCentroCabezal_mm+espesorCristal_mm/2);
                            Yeventos = coincidenceMatrix(indicesCabezal,colDetectionY1+numColumnasPorSingle*indEvento);
                            Xcabezal = Yeventos;
                        case 4
                            AnguloCabezal = 30;
                            Yop = ladoHexagonoScanner_mm + (espesorCristal_mm/2) / cosd(30);
                            OffsetY =  ((espesorCristal_mm/2) * cosd(30) + (cristalSizeX_mm/2) + ...
                                (ladoHexagonoScanner_mm-(cristalSizeX_mm/2)) *sind(30));
                            OffsetX =  -((espesorCristal_mm/2) * sind(30) + (cristalSizeX_mm * cosd(30)./2));
                            Proyectar = 1;
                        case 5
                            %Yop = 360 + 1.27 * cosd(30);
                            % Xevento = (MatrizSinglesCabezal(:,8) + MatrizSinglesCabezal(:,7)./tand(150) - 414.6).*tand(150)./(1+tand(150)*tand(150));
                            % Yevento = Xevento.*tand(150) + 414.6;
                            % Proyecto las posiciones absolutas, a las
                            % posiciones intra cabezal.
                            % Traslado al origen y luego Roto 30 grados.
                            % Xcabezal = (Yevento - 400 - 12.7*cosd(30))*cos(30) + (Xevento - 12.7*sind(30))*sin(30);
                            % Xcabezal = (Yevento - 400 - 12.7*cosd(30))*sind(-30) + (Xevento - 12.7*sind(30))*cosd(-30)
                            % Ycabezal = Zevento;
                            AnguloCabezal = 150;
                            Yop = ladoHexagonoScanner_mm + (espesorCristal_mm/2) / cosd(30);
                            OffsetY = ((espesorCristal_mm/2) * cosd(30) + (cristalSizeX_mm/2) + ...
                                (ladoHexagonoScanner_mm-(cristalSizeX_mm/2)) *sind(30));
                            OffsetX = (espesorCristal_mm/2) * sind(30) + (cristalSizeX_mm * cosd(30)./2);
                            Proyectar = 1;
                            % PlanoCabezal = hist3([Xcabezal Ycabezal], {0:1:400 -150:1:150});
                    end
                    % Primero hago la proyección del evento a la profundidad de
                    % interacción media.
                    if Proyectar
                        eventos = [coincidenceMatrix(indicesCabezal,colDetectionX1+23*indEvento) coincidenceMatrix(indicesCabezal,colDetectionY1+23*indEvento) Yop*ones(size(coincidenceMatrix(indicesCabezal,colDetectionY1+23*indEvento)))] * [cosd(AnguloCabezal).^2  sind(AnguloCabezal)*cosd(AnguloCabezal);
                            sind(AnguloCabezal)*cosd(AnguloCabezal) sind(AnguloCabezal).^2; -sind(AnguloCabezal)*cosd(AnguloCabezal)  1-sind(AnguloCabezal)*sind(AnguloCabezal)];
                        Xeventos = eventos(:,1);
                        Yeventos = eventos(:,2);
                    end
                    % Voy guardando la proyección para luego graficarla.
                    planoDetProyectadoDetXY = planoDetProyectadoDetXY + hist3([Yeventos Xeventos], valoresYX);
                    if(graficarOnline)
                        figure(4);
                        imshow(planoDetProyectadoDetXY/max(max(planoDetProyectadoDetXY)));
                        title(sprintf('Corte en el plano XY de los eventos proyectados a la distancia media'));
                    end
                    % Reemplazo con los valores proyectados las posiciones
                    % de la matriz de coincidences orginal.
                    coincidenceMatrix(indicesCabezal,colDetectionX1+23*indEvento) = Xeventos;
                    coincidenceMatrix(indicesCabezal,colDetectionY1+23*indEvento) = Yeventos;
                    
                    % Ahora hago la proyección para calcular las posiciones
                    % directamente sobre el cabezal. Recordando que a nivel de
                    % cabezal el eje largo queda representado en el eje X y el
                    % lado corto en el eje Y. El origen del sistema de
                    % coordenadas XY está en el centro del cristal.
                    % La coordenada Y del cabezal es la Z del evento, no es
                    % necesario realizar ninguna operación adicional. Mientras
                    % que para obtener el eje X debo realizar una rotación y
                    % desplazamiento de los ejes X e Y y quedarme con la
                    % posición X rotada.
                    % Recordamos que la matriz de rotación en el plano está
                    % dada por [x2; y2] = [x1; y1]*[cos(Phi) sin(Phi); -sin(Phi) cos(Phi)]
                    Ycabezal = Zevento;
                    if Proyectar
                        Xcabezal = -(Yeventos-OffsetY) .*sind(-AnguloCabezal) + (Xeventos-OffsetX) .*cosd(-AnguloCabezal);
                        Zcabezal = (Xeventos-OffsetX) .*sind(-AnguloCabezal) + (Yeventos-OffsetY) .*cosd(-AnguloCabezal);
                    end
                    planoXYcabezal{cabezal+1} = planoXYcabezal{cabezal+1} + hist3([Ycabezal Xcabezal], valoresYXcabezal);
                    if(graficarOnline)
                        figure(5);
                        subplot(2,3, cabezal+1);
                        imshow(planoXYcabezal{cabezal+1}./max(max(planoXYcabezal{cabezal+1})));
                        title(sprintf('Eventos sobre Cabezal %d', cabezal));      
                    end
                    
                    % Puedo tener múltiples degradaciones espaciales, y
                    % zonas muertas. Entonces obtengo para todos los casos
                    % las coordenadas resultantes para cada cabezal:
                    for r = 1 : numel(resEspacialFwhm_mm)   % Por ahora considero que la resolución espacial es la misma en los dos sentidos.
                        for zm = 1 : size(zonaMuerta_mm, 1)
                                % Tengo que quedarme con los datos
                                % degradados espacialmente y filtrados,
                                % para luego vovler a proyectar a las
                                % coordenadas originales para utilizarlos
                                % en el calculo del singograma. 
                                % Primero aplico las zonas muertas, porque
                                % lo que importa es la posición de llegada,
                                % no la medida.
                                %% APLICACIÓN DE ZONAS MUERTAS
                                % Cada cabezal tiene 406,4 x 304,8mm y están cubiertos por 8x6
                                % pmts cada uno de 52x52mm. O sea que los pmts cubren un área
                                % de 416x312. Al estar centrado significa que los pmts
                                % sobresalen 4.8mm y 3.6mm en cada lado del rectángulo.
                                % En cada cabezal los eventos que ciagan en los pmts de los bordes no pueden ser posicionados
                                % de esta forma se desperdica dichas zona cuyos límites son:
                                % En X: [(-208+52),(208-52)]
                                % En Y: [(-156+52), (156-52)]                
                                % Con métodos no lineales no se sabe exactamente cual
                                % es la dimensión de la zona muerta, por eso se recibe
                                % como parámetro. En zonaMuerta_mm(1) tengo la zona
                                % muerta al borde del cristal para la coordenada x, y
                                % en zonaMuerta_mm(2) para la coordenada y.
                                indicesZonaMuerta = (Xcabezal > (cristalSizeX_mm/2-zonaMuerta_mm(zm,1))) | (Xcabezal< (-(cristalSizeX_mm/2-zonaMuerta_mm(zm,1)))) ...
                                    | (Ycabezal > (cristalSizeY_mm/2-zonaMuerta_mm(zm,2))) | (Ycabezal < (-(cristalSizeY_mm/2-zonaMuerta_mm(zm,2))));
                                % Elimino los eventos de la zona muerta.
                                XcabezalZonaMuerta = Xcabezal(~indicesZonaMuerta);
                                YcabezalZonaMuerta = Ycabezal(~indicesZonaMuerta);
                                planoXYcabezalZonaMuerta{r,zm,cabezal+1} = planoXYcabezalZonaMuerta{r,zm,cabezal+1} + hist3([YcabezalZonaMuerta XcabezalZonaMuerta], valoresYXcabezal);
                                if(graficarOnline)
                                    figure(5 + r * size(zonaMuerta_mm, 1) + zm);
                                    set(gcf, 'Name', sprintf('Resolución Espacial %f, Zona Muerta: %f mm', resEspacialFwhm_mm(r), zonaMuerta_mm(zm,:)));
                                    subplot(2,3, cabezal+1);
                                    imshow(planoXYcabezalZonaMuerta{r,zm,cabezal+1}./max(max(planoXYcabezalZonaMuerta{r,zm,cabezal+1})));
                                    title(sprintf('Eventos Finales sobre Cabezal %d', cabezal));      
                                end
                                indicesZonaMuertaAcumulado{r,zm}(indicesCabezal) = indicesZonaMuertaAcumulado{r,zm}(indicesCabezal) | indicesZonaMuerta;
                                %% DEGRADACIÓN ESPACIAL
                                % Aplico una degradación espacial para simular la
                                % resolución espacial del sistema. El fwhm de la
                                % resolución es pasado como parámetro. Recordamos que
                                % fwhm = desvio * 2.34. La degradación espacial la hago
                                % sobre las coordenadas del cristal, ya que si las
                                % hiciera sobre las originales no sería perpendicular
                                % al cristal y la variación no sería uniforme para cada
                                % cristal.
                                % Para la degradación, para cada valor de coordenada
                                % obtengo un valor con distribución normal con media el
                                % valor original, y el desvío resultante de la
                                % resolución del sistema.
                                XcabezalSp = normrnd(Xcabezal, resEspacialFwhm_mm(r)/2.34);
                                YcabezalSp = normrnd(Ycabezal, resEspacialFwhm_mm(r)/2.34);
                                
                                % Una vez que tengo el valor degradado
                                % sobre el plano del cabezal debo volver al
                                % sistema de coordenadas global:
                                % Recordamos que la proyección a un plano
                                % que hicimos fue:
                                % [xcabezal; y2] = [(xevento-centro); (yevento-centro)]*[cos(Phi) sin(Phi); -sin(Phi) cos(Phi)]
                                % Si obtenemos la inversa, queda: 
                                % [(xevento-centro); (yevento-centro)] = [xcabezal; zcabezal]*[cos(Phi) -sin(Phi); sin(Phi) cos(Phi)]
                                Zevento = Ycabezal;
                                % Guardo los valores en posDet{r, zm},
                                % siendo las columnas X1 Y1 Z1 X2 Y2 Z2.
                                if Proyectar
                                    posDet{r, zm}(indicesCabezal,1+indEvento*3) = XcabezalSp .* cosd(-AnguloCabezal) + Zcabezal .* sind(-AnguloCabezal) + OffsetX;
                                    posDet{r, zm}(indicesCabezal,2+indEvento*3) = -XcabezalSp .* sind(-AnguloCabezal) + Zcabezal .* cosd(-AnguloCabezal) + OffsetY;
                                else
                                    % Si no hay que proyectar, el Y evento es igual al X del cabezal: 
                                    posDet{r, zm}(indicesCabezal,2+indEvento*3) = XcabezalSp;
                                    % El Xeventos se mantiene, no se
                                    % modifica:
                                    posDet{r, zm}(indicesCabezal,1+indEvento*3) = Xeventos;
                                end
                                % El Z del evento es el y del cabezal:
                                posDet{r, zm}(indicesCabezal,3+indEvento*3) = YcabezalSp;
                 
                                
                        end
                    end
                end
            end
            % Tengo las posiciones espaciales degradadas me falta aplicar
            % las zonas muertas uqe ya tengo los índices:
            for r = 1 : numel(resEspacialFwhm_mm)   % Por ahora considero que la resolución espacial es la misma en los dos sentidos.
                for zm = 1 : size(zonaMuerta_mm, 1)
                    posDet{r, zm}(indicesZonaMuertaAcumulado{r,zm},:) = [];
                end
            end

            for r = 1 : numel(resEspacialFwhm_mm)   % Por ahora considero que la resolución espacial es la misma en los dos sentidos.
                for zm = 1 : size(zonaMuerta_mm, 1)
                    %% REBINNING
                    % Ya tengo todos los datos filtrados. Ahora debo aplicar el
                    % rebinning:
                    switch(rebinningMethod)
                        case 'ssrb'
                            % Rebinning ssrb.
                            posDetSsrb{r,zm} = ssrb(posDet{r, zm});
                        case 'msrb' 
                            % Rebinning msrb.
                    end
                    %% LLENADO DE SINOGRAMAS 2D
                    % Ahora lleno los sinogramas. Para esto convierto el par de
                    % eventos (X1,Y1,Z1) y (X2,Y2,Z2) en lors del tipo (Thita,r,z).
                    % El ángulo thita está determinado por atan((Y1-Y2)/(X1-X2))+90
                    theta = atand((posDet{r, zm}(:,2)-posDet{r, zm}(:,5))./(posDet{r, zm}(:,1)-posDet{r, zm}(:,4))) + 90;
                    % El offset r lo puedo obtener reemplazando (x,y) con alguno de
                    % los dos puntos en la ecuación: r=x*cos(thita)+y*sin(thita)-
                    r_sino = cosd(theta).*posDet{r, zm}(:,1) + sind(theta).*posDet{r, zm}(:,2);
                    % Acumulo todo en el array de sinogramas 2D utilizando la
                    % función creada hist4.
                    sinograms2D{r,zm} = sinograms2D{r,zm} + hist4([theta r_sino posDetSsrb{r,zm}(:,3)], ...
                        {structSizeSino2D.thetaValues_deg structSizeSino2D.rValues_mm structSizeSino2D.zValues_mm});
                    if(graficarOnline)
                        figure(5 + numel(resEspacialFwhm_mm) * size(zonaMuerta_mm, 1) + r * size(zonaMuerta_mm, 1) + zm);
                        imshow(sinograms2D{r,zm}(:,:,round(structSizeSino2D.numZ/2))/max(max(sinograms2D{r,zm}(:,:,round(structSizeSino2D.numZ/2)))));
                        title(sprintf('Sinograma del slice Cental para res=%f y zona meurta [%f]', resEspacialFwhm_mm(r), zonaMuerta_mm(zm,:)));
                    end
                    %% LLENADO DE MICHELOGRAMA
                    michelograms{r,zm} = michelograms{r,zm} + hist5([theta r_sino posDet{r, zm}(:,3) posDet{r, zm}(:,6)], ...
                        {structSizeSino3D.thetaValues_deg structSizeSino3D.rValues_mm structSizeSino3D.zValues_mm structSizeSino3D.zValues_mm});
                    %% ESCRIBO LOS ARCHIVOS EN MODO LISTA
                    % Como se escribe por columna, la matriz de modo lista,
                    % debe tener cada evento en una columna:
                    numEventoModoLista{r, zm} = (numEventoModoLista{r, zm}(end)+1) : 1 : (numEventoModoLista{r, zm}(end)+size(posDet{r, zm},1));
                    fwrite(fidsModoLista{r,zm}, [numEventoModoLista{r, zm}' posDet{r, zm}(:,1:3) coincidenceMatrix(~indicesZonaMuertaAcumulado{r,zm},colEnergy1) coincidenceMatrix(~indicesZonaMuertaAcumulado{r,zm},colTimeStamp1)...
                        posDet{r, zm}(:,4:6) coincidenceMatrix(~indicesZonaMuertaAcumulado{r,zm},colEnergy2) coincidenceMatrix(~indicesZonaMuertaAcumulado{r,zm},colTimeStamp2)]', 'single');
                    
                    %fprintf(fidModoLista{r,zm}, "%f,%f,%f(indicesZonaMuertaAcumulado{r,zm},:)
                    %% FIN DEL LOOP
                end
            end
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

%% CIERRO LOS ARCHIVOS DE MODO LISTA
for r = 1 : numel(resEspacialFwhm_mm)
    for zm = 1 : size(zonaMuerta_mm,1)
        fclose(fidsModoLista{r,zm});
    end
end

%% REDUCCIÓN DE LOS MICHELOGRAMAS
% Llevo los michelogramas a sinogramas 3D. Para esto debo utilizar los
% vectores sinogramsPerSegment, minRingDiff y maxRingDiff.
sinograms3d = cell(size(michelograms));
for r = 1 : numel(resEspacialFwhm_mm)   % Por ahora considero que la resolución espacial es la misma en los dos sentidos.
    for zm = 1 : size(zonaMuerta_mm, 1)
        % Voy generando el sinograma y eliminando el michelograma para no
        % consumir tanta ram, que ya de por si es demasiada.
        sinograms3d{r,zm} = reduceMichelogram(michelograms{r,zm}, structSizeSino3D.sinogramsPerSegment, structSizeSino3D.minRingDiff, structSizeSino3D.maxRingDiff);
        % Borro el contenido de michelgorams:
        michelograms{r,zm} = [];
    end
end
%% GUARDO SINOGRAMAS A DISCO
% Se guardan los sinogramas en formato interfile. Para esto utilizo la
% función interfileWriteSino.
for r = 1 : numel(resEspacialFwhm_mm)   % Por ahora considero que la resolución espacial es la misma en los dos sentidos.
    for zm = 1 : size(zonaMuerta_mm, 1)
        % Escribo los sinogramas 2d:
        filename = sprintf('%s/sinograms2d_res%.2f_zm%.2f_%.2f', outputPath, resEspacialFwhm_mm(r), zonaMuerta_mm(zm,:));
        interfileWriteSino(single(sinograms2D{r,zm}), filename);
        % Escribo el sinograma 3d:
        filename = sprintf('%s/sinogram3d_res%.2f_zm%.2f_%.2f', outputPath, resEspacialFwhm_mm(r), zonaMuerta_mm(zm,:));
        interfileWriteSino(single(sinograms3d{r,zm}), filename);
    end
end
%% VISUALIZACIÓN DE SINOGRAMAS 2D
% Visualizo los sinogramas 2D para cada slice.

% Ahora si lo visualizo:
for r = 1 : numel(resEspacialFwhm_mm)   % Por ahora considero que la resolución espacial es la misma en los dos sentidos.
    for zm = 1 : size(zonaMuerta_mm, 1)
%         figure;
%         set(gcf, 'Name', sprintf('Sinogramas con ResEsp(FWHM):%.2fmm ZonaMuerta(%.2f,%.2f)mm', resEspacialFwhm_mm(r), zonaMuerta_mm(zm,:)));
%         set(gcf, 'Position', [100 100 1600 800]);
%         maxCounts = max(max(max(sinograms2D{r,zm})));
%         for i = 1 : size(sinograms2D{r,zm},3)
%             imshow(sinograms2D{r,zm}(:,:,i)./ maxCounts);
%             title(sprintf('Mapa de emisión para z = %f', structSizeSino2D.zValues_mm(i)));
%             pause(0.3);
%         end
        
        % También genero una única imagen con los slices para la visualización:
        image = getImageFromSlices(sinograms2D{r,zm}, 8);
        h = figure;
        set(gcf, 'Name', sprintf('Sinogramas con ResEsp(FWHM):%.2fmm ZonaMuerta(%.2f,%.2f)mm', resEspacialFwhm_mm(r), zonaMuerta_mm(zm,:)));
        set(gcf, 'Position', [100 100 1600 800]);
        imshow(image./max(max(image)));
    end
end
