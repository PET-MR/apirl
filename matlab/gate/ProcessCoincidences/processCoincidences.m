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
%
%  Este script realiza una verificación del archivo de coincidences de la
%  simulación, para esto calcula las tasas de adquisición, y las posiciones
%  de detección.
%
%  Los parámetros que recibe son:
%
%  structSimu: estructura con los datos de la simulación.
%
%   - energyWindow: Agrego la ventana de energía en un vector con [LimInf
%   LimSup].
%
%   graficarOnline: realiza el gráfico de los histogramas de emisión en
%   cada coordenada mientras procesa.
%  Devuelvo una tabla con los resultados:
%   resultadoEventos = [eventosTotales, eventosTotalesEnVentana, eventosTrues, eventosRandoms, eventosConScatter, eventosTruesConScatter];

function resultadoEventos = processCoincidences(structSimu, graficarOnline, energyWindow)

% Agrego el path donde tengo algunas funciones de uso general. En este caso
% hist4.
addpath('/sources/MATLAB/FuncionesGenerales');
addpath('/sources/MATLAB/VersionesFinales');
addpath('/sources/MATLAB/WorkingCopy/ImageProcessing')

%% CONSTANTES GEOMÉTRICAS DEL SISTEMA
CristalSizeY = 304.8;   % Ancho del cristal 304.8mm.
CristalSizeX = 406.4;   % Ancho del cristal 406.4mm.

CantCabezales = 6;
%% INICIALIZACIÓN DE VARIABLES 
% Variables relacionadas con el conteo de eventos:
eventosTotales = 0;
eventosTrues = 0;
eventosRandoms = 0;
eventosConScatter = 0;
eventosSinCompton = 0;
eventosTotalesEnVentana = 0;
eventosTruesConScatter = 0;
eventosTotalesCabezal = zeros(CantCabezales,1);
CombinacionCabezales = zeros(CantCabezales);
TiempoSplitsAnteriores = 0; % Variable que acumula los tiempos de adquisición de los aplits ya procesados.
TiempoTotal = 0;            % Variable que indica el tiempo total de la simulación.

% Variables para histogramas y filtrado de energía:
%  Variables que definen las ventanas de energía, siempre existe una
%  ventana de energía sobre el pico que denominamos A, y es opcional una
%  ventana B en la zona de energía que hay poco compton (aprox 350-350keV).
% Ahora la ventana de energía la saco del structSimu:
% VentanaAsup = 0.580; % Límite superior de la ventana de energía del pico.
% VentanaAinf = 0.430;    % Límite inferior de la ventana de energía en pico.
VentanaAinf = energyWindow(1);
VentanaAsup = energyWindow(2);

HabilitaVentanaB = 0;   % Habilita la utilización de la ventana de energía
%  secundaria B.
CanalesEnergia =  0:0.001:2.0;  % Canales de Energía para el histograma.
HistEnergias = zeros(1,numel(CanalesEnergia));  % Inicialización en cero del Histograma de Energías.
HistEnergiasFilt = zeros(1,numel(CanalesEnergia));  % Inicialización en cero del Histograma de Energías Filtrado.
HistEnergiasSinCompton = zeros(1,numel(CanalesEnergia));  % Inicialización en cero del Histograma de Energías de Eventos Cin Comptons.

% Coordenadas geométricas del volumen de interés del world:
ValoresX = -400:1:400;                   % Valores posibles de emisión de positrones en la variable Z (O sea, subre todo el FOV).
ValoresY = -500:1:500;                   % Valores posibles de emisión de positrones en la variable Z (O sea, subre todo el FOV).
ValoresZ = -158:1:158;                   % Valores posibles de emisión de positrones en la variable Z (O sea, subre todo el FOV).
ValoresYX = {ValoresY ValoresX};           % Cell Array con los valores posibles de las variables x e y dentro del sistema.
ValoresYZ = {ValoresY ValoresZ };           % Cell Array con los valores posibles de las variables y e z dentro del sistema.
% Planos de detección:
planoDetXY = zeros(numel(ValoresY), numel(ValoresX));
planoDetEmiXY = zeros(numel(ValoresY), numel(ValoresX));
planoDetYZ = zeros(numel(ValoresY), numel(ValoresZ));

% Mapa tridimensional del Fov completo:
ValoresXemision = -400:20:400;                   % Valores posibles de emisión de positrones en la variable Z (O sea, subre todo el FOV).
ValoresYemision = -400:20:400;                   % Valores posibles de emisión de positrones en la variable Z (O sea, subre todo el FOV).
ValoresZemision = -150:20:150;                   % Valores posibles de emisión de positrones en la variable Z (O sea, subre todo el FOV).
emissionMap = zeros(numel(ValoresYemision), numel(ValoresXemision), numel(ValoresZemision));

% Constantes de sinogramas
numTheta = 96; numR = 124; numZ = 41; rFov_mm = 400; zFov_mm = 300; % eL ZfOV_MM DEBE VARIAR CON LA ZONA MUERTA EN EL DETECTOR.
structSizeSino2D = getSizeSino2Dstruct(numTheta, numR, numZ, rFov_mm, zFov_mm);
sinograms2D_ssrb = single(zeros(structSizeSino2D.numTheta, structSizeSino2D.numR, ...
    structSizeSino2D.numZ));
valoresRhistograma = -360:360;
histR = zeros(1,numel(valoresRhistograma));

% Tiempo:
CanalesTiempo = -20e-9:0.5e-9:20e-9;
HistDifTiempos = zeros(1, numel(CanalesTiempo));
%% INICIALIZACIÓN DE VARIABLES EXCLUSIVAS DEL COINCIDENCE
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
        else
            disp(sprintf('Iniciando el procesamiento del archivo %s.', NombreArch));
            while feof(FID) == 0
                datos = textscan(FID, '%f', 300000 * 46);
                if mod(numel(datos{1}), 46) ~= 0
                    % El archivo no esta entero, puede que se haya cortado
                    % por una finalizacion abrupta de la simulacion.
                    % Elimino la coincidencia incompleta
                    datos{1}(end-(mod(numel(datos{1}),46)-1) : end) =[];
                end
                % Le doy forma de matriz a todos los datos leídos.
                coincidenceMatrix = reshape(datos{1}, 46, numel(datos{1})/46)';
                if ~isempty(coincidenceMatrix)
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
        %             %% GRAFICO TRANSVERSAL DE LA DETECCION Y EMISION FUSIONADAS
        %             % Sirve para verificar que los tamaos estan ben:
        %             if(graficarOnline)
        % %                 planoDetEmiXY = planoDetEmiXY + hist3([coincidenceMatrix(:,colDetectionY1) coincidenceMatrix(:,colDetectionX1); ...
        % %                 coincidenceMatrix(:,colDetectionY2) coincidenceMatrix(:,colDetectionX2); ...
        % %                 coincidenceMatrix(:,colEmisionY1) coincidenceMatrix(:,colEmisionX1);...
        % %                 coincidenceMatrix(:,colEmisionY2) coincidenceMatrix(:,colEmisionX2);], ValoresYX);
        % 
        %                 title(sprintf('Corte en el plano YZ de los eventos detectados'));
        %                 for k = 1 : numel(coincidenceMatrix(:,colDetectionY2))
        %                     x = ValoresX;
        %                     y=(coincidenceMatrix(k,colDetectionY1)-coincidenceMatrix(k,colDetectionY2))./...
        %                         (coincidenceMatrix(k,colDetectionX1)-coincidenceMatrix(k,colDetectionX2)).*...
        %                         (x-coincidenceMatrix(k,colDetectionX1))+coincidenceMatrix(k,colDetectionY1);
        %                     planoDetEmiXY = planoDetEmiXY + hist3([y',x'],ValoresYX);
        %                 end
        %                 
        %                 figure(8);
        %                 imshow(planoDetEmiXY/max(max(planoDetEmiXY)));
        %             end
            %% ESPECTROS DE ENERGIAS Y FILTRO DE VENTANA
                    % En la cadena de procesamiento lo primero que hago es el
                    % filtrado en energía.
                    HistEnergias = HistEnergias + hist([coincidenceMatrix(:,colEnergy1); coincidenceMatrix(:,colEnergy2)], CanalesEnergia);
                    if(graficarOnline)
                        figure(1);
                        bar(CanalesEnergia, HistEnergias);
                        title('Histograma de Energías');
                    end
                    % Grafico el histograma de Energía sin Comptons en el fantoma.
                    indicesSinCompton = (coincidenceMatrix(:,colCompton1)==0)&(coincidenceMatrix(:,colCompton2)==0);
                    eventosSinCompton = eventosSinCompton + sum(indicesSinCompton);
                    HistEnergiasSinCompton = HistEnergiasSinCompton + hist([coincidenceMatrix(indicesSinCompton,colEnergy1); coincidenceMatrix(indicesSinCompton,colEnergy2)], CanalesEnergia);
                    if(graficarOnline)
                        figure(2);
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
                    if(graficarOnline)
                        figure(3);
                        bar(CanalesEnergia, HistEnergiasFilt);
                        title('Histograma de Energías luego de Filtro de Ventana');
                    end
                    eventosTotalesEnVentana = eventosTotalesEnVentana + numel(coincidenceMatrix(:,1));
                    disp(sprintf('Resolución en energía: %f% FWHM', (CanalesEnergia(P2)-CanalesEnergia(P1))/0.511));
                    disp(sprintf('Ventana de energía: %0.f-%.0f keV', VentanaAinf*1000, VentanaAsup*1000));
                    disp(sprintf('Coincidencias en ventana de energía: %d', eventosTotalesEnVentana));
                    disp(sprintf('Porcentaje de coincidencias en ventana de energía: %f', eventosTotalesEnVentana/eventosTotales*100));
                    disp(sprintf('Coincidencias sin Compton en fantoma: %d', eventosSinCompton));
                    disp(sprintf('Porcentaje de coincidencias sin compton(obtenido del GATE): %f', eventosSinCompton/eventosTotales*100));
            %% TASAS DE COINCIDENCES
                    disp(sprintf('\n\n'));
                    disp('%%%%%%%%  TASAS DE COINCIDENCIAS  %%%%%%%%%%%');
                    % Tasa de Prompts
                    disp(sprintf('%d eventos totales en ventana. Tiempo de Adquisición: %f',eventosTotalesEnVentana, TiempoTotal)); 
                    TasaTotal = eventosTotalesEnVentana / TiempoTotal;
                    disp(sprintf('Tasa de PROMPTS en todo el scanner: %f', TasaTotal));
                    % Tasa de Trues y Randoms Real
                    % La calculo usando el event ID
                    indicesIguales = (coincidenceMatrix(:,colEventId1) == coincidenceMatrix(:,colEventId2));
                    indicesIguales = (coincidenceMatrix(:,colEmisionX1) == coincidenceMatrix(:,colEmisionX2)) & (coincidenceMatrix(:,colEmisionY1) == coincidenceMatrix(:,colEmisionY2)) ...
                    & (coincidenceMatrix(:,colEmisionZ1) == coincidenceMatrix(:,colEmisionZ2));
                    eventosTrues = eventosTrues + sum(indicesIguales);
                    eventosRandoms = eventosTotalesEnVentana - eventosTrues;
                    TasaRandoms = eventosRandoms / TiempoTotal;
                    TasaTrues = eventosTrues / TiempoTotal;
                    disp('%% TRUES Y RANDOMS REALES (OBTENIDOS DE LA SIMULACIÓN) %%');
                    disp(sprintf('Tasa de TRUES: %f' , TasaTrues));
                    disp(sprintf('Tasa de RANDOMS: %f' , TasaRandoms));
                    disp(sprintf('Eventos Trues: %.0f. Eventos Randoms: %.0f. Porcentaje de Randoms: %.2f.' , eventosTrues, eventosRandoms, eventosRandoms/eventosTotalesEnVentana*100));
                    % De los eventos trues me interesa saber cuantos fueron
                    % scattereados:
                    indicesConScatter = (coincidenceMatrix(:,colCompton1)~=0)|(coincidenceMatrix(:,colCompton2)~=0);
                    eventosConScatter = eventosConScatter + sum(indicesConScatter);
                    tasaScatter = eventosConScatter / TiempoTotal;
                    fprintf('Tasa de Scatter: %f\n', tasaScatter);
                    % Idem pero solo con eventos trues:
                    indicesTruesConScatter = (coincidenceMatrix(indicesIguales,colCompton1)~=0)|(coincidenceMatrix(indicesIguales,colCompton2)~=0);
                    eventosTruesConScatter = eventosTruesConScatter + sum(indicesTruesConScatter);
                    tasaTruesScatter = eventosTruesConScatter / TiempoTotal;
                    fprintf('Tasa de Scatter: %f\n', tasaTruesScatter);

                    % Calculo la tasa por cabezal.
                    for cabezal = 0 : CantCabezales-1
                        disp(sprintf('%%%%%%%%  Cabezal %d  %%%%%%%%%%%', cabezal));
                        % Primero debo filtrar los eventos por ID:
                        indicesCabezal = (coincidenceMatrix(:,colVolIdRsector1) == cabezal) | (coincidenceMatrix(:,colVolIdRsector2) == cabezal);
                        eventosTotalesCabezal(cabezal+1) = eventosTotalesCabezal(cabezal+1) + sum(indicesCabezal,1);
                        %% TASA
                        Tasa =  eventosTotalesCabezal(cabezal+1) /TiempoTotal;
                        disp(sprintf('Tasa de Adquisición en cabezal %d por cuentas totales sobre tiempo: %f cps', cabezal, Tasa));
                        Tasa = 1./mean(diff([coincidenceMatrix(indicesCabezal,colTimeStamp1); coincidenceMatrix(indicesCabezal,colTimeStamp2)]));
                        disp(sprintf('Tasa de Adquisición en cabezal %d por promedio de tiempo entre eventos: %f cps', cabezal, Tasa));
                        HistDifTiempos = HistDifTiempos + hist((coincidenceMatrix(indicesCabezal,colTimeStamp1)-coincidenceMatrix(indicesCabezal,colTimeStamp2)),CanalesTiempo);
                        if(graficarOnline)
                            figure(4);
                            subplot(2,3, cabezal+1);
                            bar(CanalesTiempo, HistDifTiempos);
                            title(sprintf('Histograma de Diferencia de Tiempos entre Eventos en Coincidencia para el Caebzal %d', cabezal+1));
                            xlabel('Tiempo en segundos');
                            ylabel('Cuentas');
                        end
                    end
                    % Elimino randoms eliminar luego:
                    coincidenceMatrix = coincidenceMatrix(indicesIguales,:);

                    % Hago un histograma de la combinación de cabezales para las
                    % coincidencias.
                    CombinacionCabezales = CombinacionCabezales + hist3([coincidenceMatrix(:,colVolIdRsector1) coincidenceMatrix(:,colVolIdRsector2)], {0:5 0:5});
                    if(graficarOnline)
                        figure(5);
                        bar3(CombinacionCabezales);
                        title('Cuentas Por Combinación de Cabezales');
                    end
                    %% CORTE TRANSAXIAL
                    % Grafico un corte transaxial de las posiciones de los singles, o sea de
                    % las detecciones en el cristal. 
                    planoDetXY = planoDetXY + hist3([coincidenceMatrix(:,colDetectionY1) coincidenceMatrix(:,colDetectionX1); ...
                        coincidenceMatrix(:,colDetectionY2) coincidenceMatrix(:,colDetectionX2)], ValoresYX);
                    if(graficarOnline)
                        figure(6);
                        imshow(planoDetXY/max(max(planoDetXY)));
                        title(sprintf('Corte en el plano XY de los eventos detectados'));
                    end
                    %% CORTE LONGITUDINAL
                    % Grafico un corte longitudinal de las posiciones de detección. O sea una
                    % proyección sobre el eje X de los Singles.
                    %  figure(indiceBaseFigures+CantFiguresPorCabezal*cabezal+5);
                    planoDetYZ = planoDetYZ + hist3([coincidenceMatrix(:,colDetectionY1) coincidenceMatrix(:,colDetectionZ1); ...
                        coincidenceMatrix(:,colDetectionY2) coincidenceMatrix(:,colDetectionZ2)], ValoresYZ);
                    if(graficarOnline)
                       figure(7);
                       imshow(planoDetYZ/max(max(planoDetYZ)));
                       title(sprintf('Corte en el plano YZ de los eventos detectados'));
                    end

                    % Por último, mapa de emisión 3d:
                    emissionMap = emissionMap + hist4([coincidenceMatrix(:,colEmisionY1), coincidenceMatrix(:,colEmisionX1), coincidenceMatrix(:,colEmisionZ1)],...
                        {ValoresYemision, ValoresXemision, ValoresZemision});

                     if(graficarOnline)
                        % Los grafico:
                        image = getImageFromSlices(emissionMap, 8);
                        figure(8);
                        set(gcf, 'Position', [100 100 1600 800]);
                        imshow(image./max(max(image)));
                     end


                    % Uso el sinograma para una roi de 80 cm radio:
                    %indiceRoi = sqrt((coincidenceMatrix(:,colEmisionX1)+57.2).^2 + coincidenceMatrix(:,colEmisionY1).^2) < 10;
                    indiceRoi = sqrt((coincidenceMatrix(:,colEmisionX1)).^2 + coincidenceMatrix(:,colEmisionY1).^2) < 75;
                    % Rebinning ssrb.
                    %posDet = ssrb([coincidenceMatrix(:,colDetectionX1) coincidenceMatrix(:,colDetectionY1) coincidenceMatrix(:,colDetectionZ1) ...
                    %    coincidenceMatrix(:,colDetectionX2) coincidenceMatrix(:,colDetectionY2) coincidenceMatrix(:,colDetectionZ2)]);
                    % En vez de hacer el rebinning, obtengo solo sinogramas
                    % directos:
        %                 posDet = [coincidenceMatrix(indiceRoi,colDetectionX1) coincidenceMatrix(indiceRoi,colDetectionY1) coincidenceMatrix(indiceRoi,colDetectionZ1) ...
        %                    coincidenceMatrix(indiceRoi,colDetectionX2) coincidenceMatrix(indiceRoi,colDetectionY2) coincidenceMatrix(indiceRoi,colDetectionZ2)];
        %                 indiceDirectos = abs(posDet(:,3)-posDet(:,6)) < 10;
        %                 posDet = posDet(indiceDirectos,:);
                    posDet = ssrb([coincidenceMatrix(indiceRoi,colDetectionX1) coincidenceMatrix(indiceRoi,colDetectionY1) coincidenceMatrix(indiceRoi,colDetectionZ1) ...
                        coincidenceMatrix(indiceRoi,colDetectionX2) coincidenceMatrix(indiceRoi,colDetectionY2) coincidenceMatrix(indiceRoi,colDetectionZ2)]);
                    % LLENADO DE SINOGRAMAS 2D
                    % Ahora lleno los sinogramas. Para esto convierto el par de
                    % eventos (X1,Y1,Z1) y (X2,Y2,Z2) en lors del tipo (Thita,r,z).
                    % El ángulo thita está determinado por atan((Y1-Y2)/(X1-X2))+90
                    theta = atand((posDet(:,2)-posDet(:,5))./(posDet(:,1)-posDet(:,4)))+90;
                    % El offset r lo puedo obtener reemplazando (x,y) con alguno de
                    % los dos puntos en la ecuación: r=x*cos(thita)+y*sin(thita)-
                    r_sino = cosd(theta).*posDet(:,1) + sind(theta).*posDet(:,2);
                    % Acumulo todo en el array de sinogramas 2D utilizando la
                    % función creada hist4.
                    sinograms2D_ssrb = sinograms2D_ssrb + hist4([theta r_sino posDet(:,3)], ...
                        {structSizeSino2D.thetaValues_deg structSizeSino2D.rValues_mm structSizeSino2D.zValues_mm});
                        %% GRAFICO UN SINOGRAMA A MODO DE VERFICACION
                    if(graficarOnline)
                        % Grafico el sinograma central:
                        figure(9);
                        imshow(sinograms2D_ssrb(:,:,round(numZ/2))./max(max(sinograms2D_ssrb(:,:,round(numZ/2)))));
                        figure(10);
                        histR = histR + hist(r_sino, valoresRhistograma);
                        plot(valoresRhistograma, histR);
                        figure(11);
                        imshow(sinograms2D_ssrb(:,:,round(numZ/2)+7)./max(max(sinograms2D_ssrb(:,:,round(numZ/2)+7))));
                    end
                    %% FIN DEL LOOP
                end
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
figure(100);
image = getImageFromSlices(sinograms2D_ssrb, 8);
imshow(image./max(max(image)));
%% RESULTADOS DE EVENTOS
disp('\n\n\n');
disp('########################################################');
disp('###################### RESULTADOS ########################');
fprintf('Tiempo Total simulado: %f seg', TiempoTotal);
fprintf('Resolución en energía: %f% FWHM\n', (CanalesEnergia(P2)-CanalesEnergia(P1))/0.511);
fprintf('Ventana de energía utilizada: %0.f-%.0f keV\n', VentanaAinf*1000, VentanaAsup*1000);
disp('##### CUENTAS TOTALES #####');
fprintf('Eventos en Coincidencia Totales: %d\n', eventosTotales);
fprintf('Coincidencias en Ventana (Cuentas que tendrá el sinograma): %d. En porcentaje de las totales: %f.\n', eventosTotalesEnVentana, eventosTotalesEnVentana/eventosTotales*100);
fprintf('Coincidencias Trues: %d. En porcentaje de las Coincidencias: %f.\n', eventosTrues, eventosTrues ./ eventosTotalesEnVentana * 100);
fprintf('Coincidencias Random: %d. En porcentaje de las Coincidencias: %f.\n', eventosRandoms, eventosRandoms./ eventosTotalesEnVentana * 100);
fprintf('Coincidencias Scatter (Incluye trues y randoms): %d. En porcentaje de las Coincidencias: %f.\n', eventosConScatter, eventosConScatter./ eventosTotalesEnVentana * 100);
fprintf('Coincidencias Trues con Scatter: %d. En porcentaje de las Coincidencias: %f.\n', eventosTruesConScatter, eventosTruesConScatter./ eventosTotalesEnVentana * 100);

% Devuelvo una tabla con los resultados:
resultadoEventos = [eventosTotales, eventosTotalesEnVentana, eventosTrues, eventosRandoms, eventosConScatter, eventosTruesConScatter];
%% GRÁFICOS DE RESULTADOS
% Observo los resultados más importantes:

% Histogramas de Energías, total, de eventos sin compton para evaluar peak
% to total del detector, y luego de filtro de ventana.
figure(1);
bar(CanalesEnergia, HistEnergias);
title('Histograma de Energías');
xlabel('Energía [MeV]');
ylabel('Cuentas');
set(gcf, 'Position', [10 10 1400 700]);
set(gcf,'PaperPositionMode','auto');    % Para que lo guarde en el tamaño modificado.

figure(2);
bar(CanalesEnergia, HistEnergiasSinCompton);
title('Histograma de Energías Sin Los Eventos que hicieron Compton en el Fantoma');
xlabel('Energía [MeV]');
ylabel('Cuentas');
set(gcf, 'Position', [10 10 1400 700]);
set(gcf,'PaperPositionMode','auto');    % Para que lo guarde en el tamaño modificado.

figure(3);
bar(CanalesEnergia, HistEnergiasFilt);
title('Histograma de Energías luego de Filtro de Ventana');
xlabel('Energía [MeV]');
ylabel('Cuentas');
set(gcf, 'Position', [10 10 1400 700]);
set(gcf,'PaperPositionMode','auto');    % Para que lo guarde en el tamaño modificado.

% Histograma de diferencia de tiempo entre eventos en coincidencia:
figure(4);
for cabezal = 0 : CantCabezales-1
        subplot(2,3, cabezal+1);
        bar(CanalesTiempo, HistDifTiempos);
        title(sprintf('Histograma de Diferencia de Tiempos entre Eventos en Coincidencia para el Caebzal %d', cabezal + 1));
        xlabel('Tiempo en segundos');
        ylabel('Cuentas');
end
set(gcf, 'Position', [10 10 1400 700]);
set(gcf,'PaperPositionMode','auto');    % Para que lo guarde en el tamaño modificado.

% Combinación de cabezales:
figure(5);
bar3(CombinacionCabezales);
title('Cuentas Por Combinación de Cabezales');
xlabel('Cabezal Evento 1 de Coincidencia');
ylabel('Cabezal Evento 2 de Coincidencia');
set(gcf, 'Position', [10 10 1400 700]);
set(gcf,'PaperPositionMode','auto');    % Para que lo guarde en el tamaño modificado.

% Plano de detección XY
figure(6);
imshow(planoDetXY/max(max(planoDetXY)));
title(sprintf('Corte en el plano XY de los eventos detectados'));
set(gcf, 'Position', [10 10 1400 700]);
set(gcf,'PaperPositionMode','auto');    % Para que lo guarde en el tamaño modificado.

% Plano de detección YZ.
figure(7);
imshow(planoDetYZ/max(max(planoDetYZ)));
title(sprintf('Corte en el plano YZ de los eventos detectados'));
set(gcf, 'Position', [10 10 1400 700]);
set(gcf,'PaperPositionMode','auto');    % Para que lo guarde en el tamaño modificado.

% Mapa de emisión:
image = getImageFromSlices(emissionMap, 8);
figure(8);
set(gcf, 'Position', [100 100 1600 800]);
imshow(image./max(max(image)));

% Grafico un corte del slice central:
image = emissionMap(:,:,round(size(emissionMap,3)/2));
figure(9);
set(gcf, 'Position', [100 100 1600 800]);
plot(image(round(size(image,1)/2),:));