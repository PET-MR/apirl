%  *********************************************************************
%  Proyecto AR-PET. Comisión Nacional de Energía Atómica.
%  Autor: Martín Belzunce. UTN-FRBA.
%  Fecha de Creación: 25/07/2011
%  *********************************************************************
%	            PROCESAMIENTO DE COINCIDENCIAS
%
%  Script de procesamiento de simulaciones del AR-PET.
%  La distancia entre las caras internas de los cristales opuestos es de
%  720mm (diámetro del hexágono). Cada cabezal tiene 406,4mm x 304,6 mm.
%  El fantoma utilizado es la representación en Gate del 
%  NEMA 2-2001 Scatter Phantom.  Las dimensiones del
%  mismo han sido extraídas de "NEMA NU 2, Section 4, 2001".
%  Este fantoma debe ser utilizado para la simulación del 
%  ensayo NEMA "Scatter fraction, count losses, Randoms Measurements".	  
%
%  Este script realiza una verificación del archivo de coincidences de la
%  simulación, para esto calcula las tasas de adquisición, y las posiciones
%  de detección. Luego aplica una degradación espacial sobre el cabezal
%  para simula la resolución espacial del sistema, y genera sinogramas para
%  cada una de las condiciones mencionadas.
%
%  PathSimu : Path del directorio donde se encuentra la simulación ;
%
%  NombreMedicion = Nombre de la Medición ASCII seteada en el gate (Primera
%  parte del nombre de los archivos de salida).
%
%  NombreSalida = Nombre de la salida del digitizer de
%  coincidencias(Segunda parte del nombre de los archivos coincidence de salida).
%
%  NombreSalidaDelayed = Nombre de la salida del digitizer de
%  coincidencias demoradas(Segunda parte del nombre de los archivos 
%  coincidence demorada de salida).
%
%  CantSplits = Cantidad de Splits si se corrió la simulación con el
%  jobsplitter, setearlo en 1 en caso contrario.
%
%  ArchivosPorSplit = Cantidad de Archivos de Salida por Split (o de la
%  única salida si no hay split). Cuando se llega al límite de 1.9 GB el
%  gate genera un nuevo archivo agregando un _n al final.
%
%  HabilitarGraficos = Habilita la visualización de gráficos en el
%  procesamiento de las coincidencias.
%
%  Las mismas variables pero para las coincidencias demoradas:
%
%  CantSplitsDelayed
%
%  ArchivosPorSplitDelayed
%
%  HabilitarGraficosDelayed
%
%  Variables que definen las ventanas de energía, siempre existe una
%  ventana de energía sobre el pico que denominamos A, y es opcional una
%  ventana B en la zona de energía que hay poco compton (aprox 350-350keV).
%
%  VentanaAsup : Límite superior de la ventana de energía del pico.
%
%  VentanaAinf : Límite inferior de la ventana de energía en pico.
%
%  HabilitaVentanaB : Habilita la utilización de la ventana de energía
%  secundaria B.
%
%  VentanaBinf : Límite inferior de la ventana de energía secundaria B
%
%  VentanaBsup : Límite superior de la ventana de energía secundaria B.
%
%  Una vez que ha sido llamado, recordar que si se quiere procesar de
%  vuelta otra medición debe realizar un clear all y close all desde el
%  script llamante, ya que este script no realiza dichas operaciones.

clear all
close all

%% INICIALIZACIÓN DE VARIABLES 
PathSimu = '/datos/Simulaciones/PET/AR-PET/SimulacionCylinder080711';   % Path del directorio donde se encuentra la simulación ;
NombreMedicion = 'Cylinder';    % Nombre de la Medición ASCII seteada en el gate (Primera
%  parte del nombre de los archivos de salida).
NombreSalida = 'Coincidences';  % Nombre de la salida del digitizer de
%  coincidencias(Segunda parte del nombre de los archivos coincidence de
%  salida).
%
CantSplits = 54;    % Cantidad de Splits si se corrió la simulación con el
%  jobsplitter, setearlo en 1 en caso contrario.
%
ArchivosPorSplit = 1;   % Cantidad de Archivos de Salida por Split (o de la
%  única salida si no hay split). Cuando se llega al límite de 1.9 GB el
%  gate genera un nuevo archivo agregando un _n al final.
%
HabilitarGraficos = 1;  % Habilita la visualización de gráficos en el
%  procesamiento de las coincidencias.
%
%  Variables que definen las ventanas de energía, siempre existe una
%  ventana de energía sobre el pico que denominamos A, y es opcional una
%  ventana B en la zona de energía que hay poco compton (aprox 350-350keV).
%
VentanaAsup = 0.580; % Límite superior de la ventana de energía del pico.
%
VentanaAinf = 0.430;    % Límite inferior de la ventana de energía en pico.
%
HabilitaVentanaB = 0;   % Habilita la utilización de la ventana de energía
%  secundaria B.
%
%  VentanaBinf : Límite inferior de la ventana de energía secundaria B
%
%  VentanaBsup : Límite superior de la ventana de energía secundaria B.
%
%  Una vez que ha sido llamado, recordar que si se quiere procesar de
%  vuelta otra medición debe realizar un clear all y close all desde el
%  script llamante, ya que este script no realiza dichas operaciones.

% Agrego el path donde tengo algunas funciones de uso general. En este caso
% hist4.
addpath('/sources/MATLAB/FuncionesGenerales');
addpath('/sources/MATLAB/VersionesFinales');
MatricesCoincidencias = cell(1,20); % Un cell arrays con 20 matrices de coincidencias cada una con una ventana temporal distinta
TruesM = 0;
MatrizSingles = [];
CanalesEnergia =  0:0.001:3.0;  % Canales de Energía para el histograma.
HistEnergias = zeros(1,numel(CanalesEnergia));  % Inicialización en cero del Histograma de Energías.
HistEnergiasFilt = zeros(1,numel(CanalesEnergia));  % Inicialización en cero del Histograma de Energías Filtrado.
HistEnergiasSinCompton = zeros(1,numel(CanalesEnergia));  % Inicialización en cero del Histograma de Energías de Eventos Cin Comptons.
ValoresX = -400:1:400;                   % Valores posibles de emisión de positrones en la variable Z (O sea, subre todo el FOV).
ValoresY = -500:1:500;                   % Valores posibles de emisión de positrones en la variable Z (O sea, subre todo el FOV).
ValoresZ = -158:1:158;                   % Valores posibles de emisión de positrones en la variable Z (O sea, subre todo el FOV).
EmisionesX = zeros(1,numel(ValoresX));         % Inicialización de histograma de emisión en X.
EmisionesY = zeros(1,numel(ValoresY));         % Inicialización de histograma de emisión en Y.
EmisionesZ = zeros(1,numel(ValoresZ));         % Inicialización de histograma de emisión en Z.
ValoresX = -400:2:400;                   % Valores posibles de emisión de positrones en la variable Z (O sea, subre todo el FOV).
ValoresY = -500:2:500;                   % Valores posibles de emisión de positrones en la variable Z (O sea, subre todo el FOV).
ValoresZ = -158:2:158;                   % Valores posibles de emisión de positrones en la variable Z (O sea, subre todo el FOV).
ValoresXY = {ValoresX ValoresY};           % Cell Array con los valores posibles de las variables x e y dentro del sistema.
ValoresYX = {ValoresY ValoresX};           % Cell Array con los valores posibles de las variables x e y dentro del sistema.
PlanoDetXY = zeros(numel(ValoresYX{1}), numel(ValoresYX{2}));  % Inicialización de Plano XY de Detección.
ValoresYZ = {ValoresY ValoresZ };           % Cell Array con los valores posibles de las variables y e z dentro del sistema.
PlanoDetYZ = zeros(numel(ValoresYZ{1}), numel(ValoresYZ{2}));  % Inicialización de Plano XY de Detección.    
ValoresYXcabezal = {-150:1:150 -200:1:200};
CanalesTiempo = -20e-9:0.5e-9:20e-9;
HistDifTiempos = zeros(1, numel(CanalesTiempo));

indiceBaseFigures = 100;
CantFiguresPorCabezal = 10;
CantCabezales = 6;

%% CONSTANTES GEOMÉTRICAS DEL CRISTAL
CristalSizeY = 304.8;   % Ancho del cristal 304.8mm.
CristalSizeX = 406.4;   % Ancho del cristal 406.4mm.
%% DEFINICIÓN DE LOS SINOGRAMAS
% Para el ensayo del scatter fraction se utilizan sinogramas 2D. Los
% eventos con ángulo oblicuo se llevan al respectivo plano 2D utilizando
% SSRB. A continuación se definen todos los parámetros que definen los
% sinogramas 2D de nuestro equipo.
% Tamaño de los sinogramas:
Nrings = 20;    % 30 anillos, o sea 30 sinos 2D.
Nthita = 192;   % Cada sino 2D tiene 300 bins en ángulo.
Nr = 192;       % Cada sino 2D tiene 300 bins en la coordenada espacial.
% Tamaño del FOV:
DFOV = 500;     % Diámetro del FOV: 500mm.
RFOV = DFOV /2;
% Define los valores medio de cada bin de las 3 coordenadas:
% Los cristales tienen 406,4mmx304,8mm
AnchoBinRing = 304.8 / Nrings;
AnchoBinThita = 180 / Nthita;
AnchoBinR = DFOV / Nr;
ValoresRing = -CristalSizeY/2 + AnchoBinRing/2 : AnchoBinRing : CristalSizeY/2 - AnchoBinRing/2;
ValoresThita = AnchoBinThita/2 : AnchoBinThita : 180 - AnchoBinThita /2;
ValoresR = -RFOV + AnchoBinR/2 : AnchoBinR : RFOV - AnchoBinR /2;
%% INICIALIZACIÓN DE VARIABLES EXCLUSIVAS DEL COINCIDENCE
EventosTotales = 0;
CoincidenciasTotales = 0;
EventosTrues = 0;
EventosRandoms = 0;
EventosSinCompton = 0;
EventosTotalesEnVentana = 0;
TotalTruesReales = 0;
EventosTotalesCabezal = zeros(CantCabezales,1);
for i = 1 : CantCabezales
    PlanoXYcabezal{i} = zeros(numel(ValoresYXcabezal{1}), numel(ValoresYXcabezal{2}));
    PlanoXYcabezalZonaMuerta{i} = zeros(numel(ValoresYXcabezal{1}), numel(ValoresYXcabezal{2}));
    PlanoDetXYcabezal{i} = zeros(numel(ValoresYX{1}), numel(ValoresYX{2}));
    PlanoDetProyectadoXYcabezal{i} = zeros(numel(ValoresYX{1}), numel(ValoresYX{2}));
    DesviosX{i} = [];
    DesviosX{i} = [];
    DesviosY{i} = [];
end
CombinacionCabezales = zeros(CantCabezales);

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
%% GEOMETRÍA DEL CABEZAL:
sizePmt_mm = [52 52];   % Tamaño de un pmt.
sizeCristal_mm = [406.4 304.8]; % Tamaño del cristal.
sizePmtCristal_mm = sizePmt_mm .* [8 6];    % Tamaño cubierto por todos los pmts de un cristal.
%% ZONAS MUERTAS
% Cada cabezal tiene 406,4 x 304,8mm y están cubiertos por 8x6
% pmts cada uno de 52x52mm. O sea que los pmts cubren un área
% de 416x312. Al estar centrado significa que los pmts
% sobresalen 4.8mm y 3.6mm en cada lado del rectángulo.
% En cada cabezal los eventos que ciagan en los pmts de los bordes no pueden ser posicionados
% de esta forma se desperdica dichas zona cuyos límites son:
% En X: [(-208+52),(208-52)]
% En Y: [(-156+52), (156-52)]     
zonasMuertas_pmt = [1 0.5]; % Zonas muertas en pmts, una y media.
zonasMuertas_mm = [(sizeCristal_mm - (sizePmtCristal_mm - zonasMuertas_pmt(1) .* sizePmt_mm)); ...
    (sizeCristal_mm - (sizePmtCristal_mm - zonasMuertas_pmt(2) .* sizePmt_mm))];
zonasUtilesCristal_mm = repmat(sizeCristal_mm, size(zonasMuertas_mm,1), 1) - zonasMuertas_mm;
%% ARCHIVOS DE SALIDA
% Este script genera uno o múltiples archivos de salida en modo lista de
% los eventos. Los mismos serán archivos binarios con formato numérico
% float (single) y que contienen los siguientes campos por cada entrada:
% x1 y1 z1 x2 y2 z2 energia compton(si o no)
pathListFile = '/datos/Sinogramas/SimulacionesGate/Cylinder080711/Cylinder080711_ListMode_full.dat';
pathListFile_zonaMuerta1pmt = '/datos/Sinogramas/SimulacionesGate/Cylinder080711/Cylinder080711_ListMode_zonaMuerta1pmt.dat';
pathListFile_zonaMuerta1pmt = '/datos/Sinogramas/SimulacionesGate/Cylinder080711/Cylinder080711_ListMode_zonaMuerta1pmt.dat';
%% PROCESAMIENTO DE COINCIDENCES
for i = 1 : CantSplits
    disp(sprintf('################ PROCESANDO SPLIT %d ###################',i));
    % Nombre Base del Archivo de Salida para el Split i
    if CantSplits>1
        % Si se ha hecho el split de la simulación en varias simulaciones
        % tengo que usar el formato de nombre que le da el splitter.
        NombreBase = sprintf('%s%d%s',NombreMedicion,i,NombreSalida); 
    else
        % El nombre por default cuando no se usa el spliter es
        % gateSingles.dat
        NombreBase = sprintf('%s%s',NombreMedicion,NombreSalida); 
    end
    % Nombre del Archivo a abrir para el split i, teniendo en cuenta que es
    % el primero que se ha generado, por lo tanto no tiene adicionales al
    % final.
    NombreArch = sprintf('%s/%s.dat',PathSimu,NombreBase);
    PrimeraLectura = 1; % Indica que es la primera lectura del split. De esta forma se pueden guardar los tiempos inciales por cada split.
    for j = 1 : ArchivosPorSplit 
        disp(sprintf('################ PROCESANDO ARCHIVO %d DEL SPLIT %d ###################',j,i));
        FID = fopen(NombreArch, 'r');
        if (FID == -1)
            disp('Error al abrir el archivo');
        end
        disp(sprintf('Iniciando el procesamiento del archivo %s.', NombreArch));
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
            EventosLeidos = size(coincidenceMatrix,1);
            
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
            EventosTotales = EventosTotales + EventosLeidos;
            disp(sprintf('Coincidencias leídas: %d', EventosTotales));
    %% ESPECTROS DE ENERGIAS Y FILTRO DE VENTANA
            % En la cadena de procesamiento lo primero que hago es el
            % filtrado en energía.
            HistEnergias = HistEnergias + hist([coincidenceMatrix(:,colEnergy1); coincidenceMatrix(:,colEnergy2)], CanalesEnergia);
            if(HabilitarGraficos)
                figure(1);
                bar(CanalesEnergia, HistEnergias);
                title('Histograma de Energías');
            end
            % Grafico el histograma de Energía sin Comptons en el fantoma.
            indicesSinCompton = (coincidenceMatrix(:,colCompton1)==0)&(coincidenceMatrix(:,colCompton2)==0);
            EventosSinCompton = EventosSinCompton + sum(indicesSinCompton);
            HistEnergiasSinCompton = HistEnergiasSinCompton + hist([coincidenceMatrix(indicesSinCompton,colEnergy1); coincidenceMatrix(indicesSinCompton,colEnergy2)], CanalesEnergia);
            if(HabilitarGraficos)
                figure(11);
                bar(CanalesEnergia, HistEnergiasSinCompton);
                title('Histograma de Energías Sin Los Eventos que hicieron Compton en el Fantoma');
            end
            % Hago un Filtrado de Energia sobre todos los eventos antes de
            % calcular la coincidencia.
            % Para el cáluclo del FWHM y la ventana elimino los canales de
            % menos de 400Kev
            HistEnergias2 = HistEnergias;
            HistEnergias2(CanalesEnergia<0.4) = 0;
            Pico = max(HistEnergias2);       % Calculo el pico de energía
            P = find(HistEnergias2>Pico/2);  % Me quedo con aquellos canales que tienen la mitad o más cuentas que el canal del pico.
                                            % Además             disp(sprintf('\n\n%%%%%%%%  TASAS DE SINGLES EN VENTANA DE ENERGÍA  %%%%%%%%%%%'));agrego como condición
                                            % que sea mayor que 300 para
                                            % evitar las cuentas por
                                            % scatter.
            P1 = P(1);                      % El primer canal de estos es donde se inicia la ventana del FWHM
            P2 = P(end);                    % El último canal de estos es donde termina la ventana
            % El filtro de energía lo hago sobre la coincidencia, esto
            % significa que si uno de los dos eventos no entra en ventana
            % se elimina toda la coincidencia.
            % Para filtrar energía en FWHM:
            if HabilitaVentanaB == 0
                indicesFuera = (coincidenceMatrix(:,colEnergy1)<(VentanaAinf)) | (coincidenceMatrix(:,colEnergy1)>(VentanaAsup)) | ...
                    (coincidenceMatrix(:,colEnergy2)<(VentanaAinf)) |(coincidenceMatrix(:,colEnergy2)>(VentanaAsup));
                % Me quedo con solo los eventos en ventana
                coincidenceMatrix(indicesFuera,:) = [];
            else
                indicesEnVentana1 = ((coincidenceMatrix(:,colEnergy1)<VentanaBsup)&(coincidenceMatrix(:,colEnergy1)>VentanaBinf)) |...
                    ((coincidenceMatrix(:,colEnergy1)>VentanaAinf)&(coincidenceMatrix(:,colEnergy1)<VentanaAsup));
                indicesEnVentana2 = ((coincidenceMatrix(:,colEnergy2)<VentanaBsup)&(coincidenceMatrix(:,colEnergy2)>VentanaBinf)) |...
                    ((coincidenceMatrix(:,colEnergy2)>VentanaAinf)&(coincidenceMatrix(:,colEnergy2)<VentanaAsup));
                coincidenceMatrix = coincidenceMatrix( indicesEnVentana1& indicesEnVentana2,:);
            end
            
            % Grafico La energía filtrada
            HistEnergiasFilt = HistEnergiasFilt + hist([coincidenceMatrix(:,colEnergy1); coincidenceMatrix(:,colEnergy2)], CanalesEnergia);
            if(HabilitarGraficos)
                figure(2);
                bar(CanalesEnergia, HistEnergiasFilt);
                title('Histograma de Energías luego de Filtro de Ventana');
            end
            EventosTotalesEnVentana = EventosTotalesEnVentana + numel(coincidenceMatrix(:,1));
            disp(sprintf('Resolución en energía: %f% FWHM', (CanalesEnergia(P2)-CanalesEnergia(P1))/0.511));
            disp(sprintf('Ventana de energía: %0.f-%.0f keV', VentanaAinf*1000, VentanaAsup*1000));
            disp(sprintf('Coincidencias en ventana de energía: %d', EventosTotalesEnVentana));
            disp(sprintf('Porcentaje de coincidencias en ventana de energía: %f', EventosTotalesEnVentana/EventosTotales*100));
            disp(sprintf('Coincidencias sin Compton en fantoma: %d', EventosSinCompton));
            disp(sprintf('Porcentaje de coincidencias sin compton(obtenido del GATE): %f', EventosSinCompton/EventosTotales*100));
    %% TASAS DE COINCIDENCES
            disp(sprintf('\n\n'));
            disp('%%%%%%%%  TASAS DE COINCIDENCIAS  %%%%%%%%%%%');
            % Tasa de Prompts
            disp(sprintf('%d eventos totales en ventana. Tiempo de Adquisición: %f',EventosTotalesEnVentana, TiempoTotal)); 
            TasaTotal = EventosTotalesEnVentana / TiempoTotal;
            disp(sprintf('Tasa de PROMPTS en todo el scanner: %f', TasaTotal));
            % Tasa de Trues y Randoms Real
            % La calculo usando el event ID
            indicesIguales = (coincidenceMatrix(:,colEventId1) == coincidenceMatrix(:,colEventId2));
            EventosTrues = EventosTrues + sum(indicesIguales);
            EventosRandoms = EventosTotalesEnVentana - EventosTrues;
            TasaRandoms = EventosRandoms / TiempoTotal;
            TasaTrues = EventosTrues / TiempoTotal;
            disp('%% TRUES Y RANDOMS REALES (OBTENIDOS DE LA SIMULACIÓN) %%');
            disp(sprintf('Tasa de TRUES: %f' , TasaTrues));
            disp(sprintf('Tasa de RANDOMS: %f' , TasaRandoms));
            disp(sprintf('Eventos Trues: %.0f. Eventos Randoms: %.0f. Porcentaje de Randoms: %.2f.' , EventosTrues, EventosRandoms, EventosRandoms/EventosTotalesEnVentana*100));
           
            % Calculo la tasa por cabezal.
            for cabezal = 0 : CantCabezales-1
                disp(sprintf('%%%%%%%%  Cabezal %d  %%%%%%%%%%%', cabezal));
                % Primero debo filtrar los eventos por ID:
                indicesCabezal = (coincidenceMatrix(:,colVolIdRsector1) == cabezal) | (coincidenceMatrix(:,colVolIdRsector2) == cabezal);
                EventosTotalesCabezal(cabezal+1) = EventosTotalesCabezal(cabezal+1) + sum(indicesCabezal,1);
                %% TASA
                Tasa =  EventosTotalesCabezal(cabezal+1) /TiempoTotal;
                disp(sprintf('Tasa de Adquisición en cabezal %d por cuentas totales sobre tiempo: %f cps', cabezal, Tasa));
                Tasa = 1./mean(diff([coincidenceMatrix(indicesCabezal,colTimeStamp1); coincidenceMatrix(indicesCabezal,colTimeStamp2)]));
                disp(sprintf('Tasa de Adquisición en cabezal %d por promedio de tiempo entre eventos: %f cps', cabezal, Tasa));
                HistDifTiempos = HistDifTiempos + hist((coincidenceMatrix(indicesCabezal,colTimeStamp1)-coincidenceMatrix(indicesCabezal,colTimeStamp2)),CanalesTiempo);
                if(HabilitarGraficos)
                    figure(indiceBaseFigures+CantFiguresPorCabezal*cabezal+1);
                    bar(CanalesTiempo, HistDifTiempos);
                    title(sprintf('Histograma de Diferencia de Tiempos entre Eventos en Coincidencia para el Caebzal %d', cabezal));
                    xlabel('Tiempo en segundos');
                    ylabel('Cuentas');
                end
            end
            % Hago un histograma de la combinación de cabezales para las
            % coincidencias.
            CombinacionCabezales = CombinacionCabezales + hist3([coincidenceMatrix(:,colVolIdRsector1) coincidenceMatrix(:,colVolIdRsector2)], {0:5 0:5});
            if(HabilitarGraficos)
                figure(3);
                bar3(CombinacionCabezales);
                title('Cuentas Por Combinación de Cabezales');
            end
    %% PROCESAMIENTO DE POSICIONES DE LOS EVENTOS    
            % El Procesamiento se hace de a grupos de 100000 datos leídos, y luego se suman
            % los resultados. Ya que si se levantan todos los datos en RAM
            % es probable que nos quedemos sin memoria, además de que se
            % empieza a hacer más lento por el volcado de memoria a disco.
            % El procesamiento lo separo por cabezal, y por evento de la
            % coincidencia. Estos es, primero agarro todos los primeros
            % eventos de los dos de la coincidencia, y convierto la
            % posición primero en la profundidad de interacción media y
            % luego en coordenadas intra cabezal.
            indicesZonaMuertaAcumulado = logical(zeros(size(coincidenceMatrix(:,1))));  % Inicializo variable en la que se van guardando los índices de eventos que corresponden a una zona muerta.
            for indEvento = 0 : 1
                for cabezal = 0 : CantCabezales-1  
                    disp(sprintf('%%%%%%%%  Cabezal %d  %%%%%%%%%%%', cabezal));
                    % Primero debo filtrar los eventos por ID:
                    indicesCabezal = coincidenceMatrix(:,colVolIdRsector1+23*indEvento) == cabezal;
                    %% CORTE TRANSAXIAL
                    % Grafico un corte transaxial de las posiciones de los singles, o sea de
                    % las detecciones en el cristal. Lo hago tomando solo las emisiones
                    % detectadas en coincidencia con ventana de 50 nseg.
                    %figure(indiceBaseFigures+CantFiguresPorCabezal*cabezal+4);
                    PlanoDetXYcabezal{cabezal+1} = PlanoDetXYcabezal{cabezal+1} + hist3([coincidenceMatrix(indicesCabezal,colDetectionY1+23*indEvento) coincidenceMatrix(indicesCabezal,colDetectionX1+23*indEvento)], ValoresYX);
                    if(HabilitarGraficos)
                        figure(indiceBaseFigures+CantFiguresPorCabezal*cabezal+2);
                        imshow(PlanoDetXYcabezal{cabezal+1}/max(max(PlanoDetXYcabezal{cabezal+1})));
                        title(sprintf('Corte en el plano XY de los eventos recibidos en el cabezal %d', cabezal));
                    end
                    %% CORTE LONGITUDINAL
                    % Grafico un corte longitudinal de las posiciones de detección. O sea una
                    % proyección sobre el eje X de los Singles.
                    %  figure(indiceBaseFigures+CantFiguresPorCabezal*cabezal+5);
%                     PlanoDetYZ = PlanoDetYZ + hist3([MatrizCoincidences(indicesCabezal,8+12*indEvento) MatrizCoincidences(indicesCabezal,9+12*indEvento)], ValoresYZ);
%                     if(HabilitarGraficos)
%                        figure(indiceBaseFigures+CantFiguresPorCabezal*cabezal+3);
%                        imshow(PlanoDetYZ/max(max(PlanoDetYZ)));
%                     end

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
                    % Zevento = Zworld

                    % cabezal 1:    
                    %   Xevento = 37,27
                    %   Yevento = Yworld

                    % cabezal2:
                    % Xevento = (Yworld+Xworld/tan(30)+37,46)*sen(30(*cos(30)
                    % Yevento = tan(30)*Xevento + 37,46

                    % Cada cabezal es de 406,4mm en el lado que forma el
                    % hexágono, pero están levemente separados siendo el lado
                    % del hexágono 415,692mm.
                    Zevento = coincidenceMatrix(indicesCabezal,colDetectionZ1+23*indEvento);
                    Proyectar = 0;        % Variable que me dice si es necesario hacer una proyección
                                                    % de los eventos sobre el valor medio de la
                                                    % profundidad de interacción.
                    switch cabezal
                        case 0
                            Xeventos = ones(size(coincidenceMatrix(indicesCabezal,colDetectionX1+23*indEvento))).*374.6;
                            Yeventos = coincidenceMatrix(indicesCabezal,colDetectionY1+23*indEvento);
                            Xcabezal = Yeventos;
                        case 1
                            AnguloCabezal = 30;
                            Yop = -(415.69 + 12.7 / cosd(30));
                            OffsetY =  -( 12.7 * cosd(30) + 203.2 + (415.69-203.2) *sind(30));
                            OffsetX = 12.7 * sind(30) + (400 * cosd(30)./2);
                            Proyectar = 1;
                        case 2
                            AnguloCabezal = 150;
                            Yop = -(415.69 + 12.7 / cosd(30));
                            OffsetY =  -( 12.7 * cosd(30) + 203.2 + (415.69-203.2) *sind(30));
                            OffsetX = -( 12.7 * sind(30) + (406.4 * cosd(30)./2));
                            Proyectar = 1;
                        case 3
                            Xeventos = -ones(size(coincidenceMatrix(indicesCabezal,colDetectionX1+23*indEvento))).*374.6;
                            Yeventos = coincidenceMatrix(indicesCabezal,colDetectionY1+23*indEvento);
                            Xcabezal = Yeventos;
                        case 4
                            AnguloCabezal = 30;
                            Yop = 415.69 + 12.7 / cosd(30);
                            OffsetY =  (12.7 * cosd(30) + 203.2 + (415.69-203.2) *sind(30));
                            OffsetX =  -( 12.7 * sind(30) + (406.4 * cosd(30)./2));
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
                            Yop = 415.69 + 12.7 / cosd(30);
                            OffsetY =  (12.7 * cosd(30) + 203.2 + (415.69-203.2) *sind(30));
                            OffsetX = 12.7 * sind(30) + (406.4 * cosd(30)./2);
                            Proyectar = 1;
                            % PlanoCabezal = hist3([Xcabezal Ycabezal], {0:1:400 -150:1:150});
                    end
                    % Primero hago la proyección del evento a la profundidad de
                    % interacción media.
                    if Proyectar
                        Eventos = [coincidenceMatrix(indicesCabezal,colDetectionX1+23*indEvento) coincidenceMatrix(indicesCabezal,colDetectionY1+23*indEvento) Yop*ones(size(coincidenceMatrix(indicesCabezal,colDetectionY1+23*indEvento)))] * [cosd(AnguloCabezal).^2  sind(AnguloCabezal)*cosd(AnguloCabezal);
                            sind(AnguloCabezal)*cosd(AnguloCabezal) sind(AnguloCabezal).^2; -sind(AnguloCabezal)*cosd(AnguloCabezal)  1-sind(AnguloCabezal)*sind(AnguloCabezal)];
                        Xeventos = Eventos(:,1);
                        Yeventos = Eventos(:,2);
                    end
                    % Voy guardando la proyección para luego graficarla.
                    PlanoDetProyectadoXYcabezal{cabezal+1} = PlanoDetProyectadoXYcabezal{cabezal+1} + hist3([Yeventos Xeventos], ValoresYX);
                    if(HabilitarGraficos)
                        figure(indiceBaseFigures+CantFiguresPorCabezal*cabezal+4);
                        imshow(PlanoDetProyectadoXYcabezal{cabezal+1}/max(max(PlanoDetProyectadoXYcabezal{cabezal+1})));
                        title(sprintf('Corte en el plano XY de los eventos recibidos en el cabezal %d', cabezal));
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
                    end
                    % Calculo de los desvíos en cada eje:
                    DesviosX{cabezal+1} = [DesviosX{cabezal+1} std(Xcabezal)];
                    DesviosY{cabezal+1} = [DesviosY{cabezal+1} std(Ycabezal)];
                    PlanoXYcabezal{cabezal+1} = PlanoXYcabezal{cabezal+1} + hist3([Ycabezal Xcabezal], ValoresYXcabezal);
                    if(HabilitarGraficos)
                        figure(indiceBaseFigures+CantFiguresPorCabezal*cabezal+5);
                        imshow(PlanoXYcabezal{cabezal+1}./max(max(PlanoXYcabezal{cabezal+1})));
                        title(sprintf('Eventos sobre Cabezal %d en el Sistema de Coordenadas propio', cabezal));      
                    end
       %% Filtrado de zonas muertas del cabezal.
                % Cada cabezal tiene 406,4 x 304,8mm y están cubiertos por 8x6
                % pmts cada uno de 52x52mm. O sea que los pmts cubren un área
                % de 416x312. Al estar centrado significa que los pmts
                % sobresalen 4.8mm y 3.6mm en cada lado del rectángulo.
                % En cada cabezal los eventos que ciagan en los pmts de los bordes no pueden ser posicionados
                % de esta forma se desperdica dichas zona cuyos límites son:
                % En X: [(-208+52),(208-52)]
                % En Y: [(-156+52), (156-52)]                
                    indicesZonaMuerta = (Xcabezal > (208-52)) | (Xcabezal< (-208+52)) | (Ycabezal > (156-52)) | (Ycabezal< (-156+52));
                    % Elimino los eventos de la zona muerta.
                    Xcabezal(indicesZonaMuerta) = [];
                    Ycabezal(indicesZonaMuerta) = [];
                    PlanoXYcabezalZonaMuerta{cabezal+1} = PlanoXYcabezalZonaMuerta{cabezal+1} + hist3([Ycabezal Xcabezal], ValoresYXcabezal);
                    if(HabilitarGraficos)
                        figure(indiceBaseFigures+CantFiguresPorCabezal*cabezal+6);
                        imshow(PlanoXYcabezalZonaMuerta{cabezal+1}./max(max(PlanoXYcabezalZonaMuerta{cabezal+1})));
                        title(sprintf('Eventos sobre Cabezal %d con Zona Muerta en el Sistema de Coordenadas propio', cabezal));      
                    end
                    indicesZonaMuertaAcumulado(indicesCabezal) = indicesZonaMuertaAcumulado(indicesCabezal) | indicesZonaMuerta; 
                end
            end
            % Ahora lleno los sinogramas. Para esto convierto el par de
            % eventos (X1,Y1,Z1) y (X2,Y2,Z2) en lors del tipo (Thita,r,z).
            % El ángulo thita está determinado por atan((Y1-Y2)/(X1-X2))+90
            Thita = atand((coincidenceMatrix(:,colDetectionY1)-coincidenceMatrix(:,colDetectionY2))./(coincidenceMatrix(:,colDetectionX1)-coincidenceMatrix(:,colDetectionX2))) + 90;
            % El offset r lo puedo obtener reemplazando (x,y) con alguno de
            % los dos puntos en la ecuación: r=x*cos(thita)+y*sin(thita)-
            r = cosd(Thita).*coincidenceMatrix(:,colDetectionX1) + sind(Thita).*coincidenceMatrix(:,colDetectionY1);
            % El valor de ring (z o profundidad) lo calculo haciendo ssrb, o sea
            % obteniendo el punto medio entre z1 y z2: Ring = (z1+z2)/2
            Ring = (coincidenceMatrix(:,colDetectionZ1)+coincidenceMatrix(:,colDetectionZ2))/2;
            % Acumulo todo en el array de sinogramas 2D utilizando la
            % función creada hist4.
            Sinogramas2D = Sinogramas2D + hist4([Thita r Ring], {ValoresThita ValoresR ValoresRing}); 
            
            % Genero los sinogramas para el caso en que tenemos en cuenta
            % una zona muerta por la imposibilidad de posicionar en los
            % bordes del crsital.
            Thita(indicesZonaMuertaAcumulado) = [];
            r(indicesZonaMuertaAcumulado) = [];
            Ring(indicesZonaMuertaAcumulado) = [];
            Sinogramas2DconZonaMuerta = Sinogramas2DconZonaMuerta + hist4([Thita r Ring], {ValoresThita ValoresR ValoresRing}); 
            
            %% FIN DEL LOOP
        end
        fclose(FID);
        % Nombre del próximo archivo dentro del split (Esto pasa cuando el
        % archivo de salida por split es de más de 1.9GBytes).
        NombreArch = sprintf('%s/%s_%d.dat',PathSimu,NombreBase,j);
    end
 %% CAMBIO DE SPLIT
    % Fin de los archvos correspondientes a un split, ahora se seguirá con
    % el próximo split si es que hay uno. Actualiza la variable que lleva
    % el tiempo acumulado.
    TiempoSplitsAnteriores = TiempoSplitsAnteriores + TiempoSplit;
end    


%% VISUALIZACIÓN DE RESULTADOS
% Observo los resultados más importantes:

% Histogramas de Energías:
figure(1);
bar(CanalesEnergia, HistEnergias);
title('Histograma de Energías');

% Visualizo los distintos sinogramas:


%% ESCRITURA DE LOS SINOGRAMAS
% Agrego directorio donde se encuentra la función para escribir sinogramas
% en interfile:
addpath('/sources/MATLAB/WorkingCopy/ImageProcessing');
interfileWriteSino(Sinogramas2D, '/datos/Sinogramas/SimulacionesGate/Cylinder080711/sino2Dmultislice_full');