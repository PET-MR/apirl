%% FUNCIÓN SIMULACIONSINO
% Autor: Martín Belzunce.
% 
% Institución: CNEA y UTN.
% 
% Última Fecha de Modificación: 19/11/2008.
% 
% Prototipo: function Sinograma=SimulacionSino(Imagen,CantAngulos,ResolEsp,CantEventos)
% 
% Esta función genera un Sinograma Simulado estadísticamente a partir de
% una imagen. Para esto se recorre la imagen píxel a píxel, y en cada uno
% de ellos se generan emisiones aleatorias proporcionales en cantidad a la
% intensidad del píxel en la imagen. Dichas emsiones salen con un ángulo
% aleatorio entre 0 y 179. A partir de los ángulos de cada evento, y las
% posiciones de los píxeles, se calcula el valor de la distancia al origen
% R. Luego la totalidad de los eventos es agregada al Sinograma que se está
% generando.
%
% Prototiopo General de la Función: Sinograma=SimulacionSino(Imagen,CantAngulos,R,CantEventos,Modo)
% Esta función consta con tres modos de funcionamiento seleccionable
% mediante el parámetro Modo que puede tomar los siguiente valores:
% 'ResEsp', 'CantR', 'Valores'.
%
% Modo 'ResEsp'
% Sinograma=SimulacionSino(Imagen,CantAngulos,ResEsp,CantEventos,Modo)
% Se debe pasar como parámetros la Cantidad de ángulos que formarán el
% sinograma, se consideran a los ángulos equiespaciados entre 0 y 179. Por
% otro lado, de debe pasar la Resolución Espacial que se desee tomar como
% parámetro del sistema. A partir de la Resolución Espacial, y de las
% dimensiones del Field of View (FOV), se determina la cantidad de valores
% de R que formaron el Sinograma. Se considera un FOV cuadrado de 50 cm de
% lado. La Resolución Espacial debe ser pasada en mm.
% Se le debe pasar un valor deseado de Cantidad de Eventos
% Generados. Dicho valor es solo aproximado debido a la características
% aleatorias del proceso de generación de eventos.
% Por último se le debe pasar el parámetro Modo con el valor 'CantR'.
% 
% Modo 'CantR'
% Sinograma=SimulacionSino(Imagen,CantAngulos,CantR,CantEventos,Modo)
% Se debe pasar como parámetros la Cantidad de ángulos que formarán el
% sinograma, se consideran a los ángulos equiespaciados entre 0 y 179. Por
% otro lado, de debe pasar la Cantidad de Valores de R que se desean obtener. 
% A partir de dicho valor y de las dimensiones del PET modelado se
% obtendrán los distintos valores posibles de R.
% Se le debe pasar un valor deseado de Cantidad de Eventos
% Generados. Dicho valor es solo aproximado debido a la caracter�sticas
% aleatorias del proceso de generación de eventos.
% Por último se le debe pasar el parámetro Modo con el valor 'CantR'.
%
% Modo 'Valores'
% Sinograma=SimulacionSino(Imagen,ValoresAngulos,ValoresR,CantEventos,Modo)
% Se debe pasar como parámetros todos los valores que puede tomar el ángulo
% Phi y la variable R. (Ejemplo: ValoresAngulos = 1:180). Además tenga en
% cuenta que los valores de R deberán tener en cuenta la geometríaHexagonal del PET:
% RMAX = 400;                           % Radio Mayor del Hexagono regular, o sea la distancia desde el centro a cualquiera de sus vertices.
% RMIN = sqrt(3)*RMAX/2;        % Radio Menor del Hexagono regular, o sea la distancia desde el centro a cualquiera de sus lados.
% RFOV = 250;                            % Radio del Field of View
% DFOV = 2*RFOV;                     % Diametro del Field of View. Lo considero cuadrado.
% Se le debe pasar un valor deseado de Cantidad de Eventos
% Generados. Dicho valor es solo aproximado debido a la caracter�sticas
% aleatorias del proceso de generación de eventos.
% Por �ltimo se le debe pasar el parámetro Modo con el valor 'Valores'.
%
% El sinograma es una matriz cuyas filas están representadas por la variable R, 
% mientras que las columnas por el ángulo Phi, y que representa las
% proyecciones realizadas de los píxeles de una imagen. Cada proyección
% está paramterizada por el ángulo Phi y la variable R. Esta última es la
% distancia de la proyección al origen, y el ángulo Phi el ángulo que de
% dicha distancia perpendicular a la proyección.

function Sinograma=SimulacionSino(Imagen,CantAngulos,R,CantEventos,Modo)
%% INICIALIZACIÓN
% Primero defino los parametros del PET Hexagonal. Todas las distancias
% estan en mm. Se consider a un FOV de 50cmx50cm
NProy = CantAngulos;        % Numero de Proyecciones. O sea me determina el muestro angular. Por lo tanto me dice cuantas columnas tendra el sinograma.
RMAX = 400;                        % Radio Mayor del Hexagono regular, o sea la distancia desde el centro a cualquiera de sus vertices.
RMIN = sqrt(3)*RMAX/2;     % Radio Menor del Hexagono regular, o sea la distancia desde el centro a cualquiera de sus lados.
RFOV = 250;                         % Radio del Field of View
DFOV = 2*RFOV;                  % Diametro del Field of View. Lo considero cuadrado.
% Luego analizo el modo de trabajo:
switch Modo
    case 'CantR'
        NR = R;             % Cantidad de valores diferentes de R que van a haber en el sinograma. O sea la cantidad de filas del mismo.
        ValoresR = -(RFOV*sqrt(2)):2*RFOV*sqrt(2)/(NR-1):(RFOV*sqrt(2));     % Valores del Offset R.
        stepPhi = 180/NProy;
        ValoresPhi = stepPhi/2:stepPhi:180;       % Valores del angulo Phi determinado por el numero de Proyecciones
    case 'ResEsp'
        ResEsp = R;
        MuestEsp = ResEsp/3;        % Muestreo Espacial. Debe ser como minimo la frecuencia de Nyquist, o sea ResEsp/2;
        ValoresR = -(RFOV*sqrt(2)):MuestEsp:(RFOV*sqrt(2));     % Valores del Offset R.
        NR = numel(ValoresR);
        stepPhi = 180/NProy;
        ValoresPhi = stepPhi/2:stepPhi:180;       % Valores del angulo Phi determinado por el número de Proyecciones
    case 'Valores'
        % Se han pasado como parámetros los ángulos y los valores de R de
        % las proyecciones a generar
        ValoresR = R;
        ValoresPhi = CantAngulos;
        NR = numel(ValoresR);
        NProy = numel(ValoresPhi);
    otherwise
        disp('Opción de ejecución no válida. Ejecute ''help SimulacionSino'' para obtener más información');
end
  
LORSTOTALES = NProy*NR;     % Cantidad totales de LORs, o sea el numero de elementos que tendra el sinograma.
Origen = [0 0];             % Centro del FOV
SizeIm = size(Imagen);      % Tamaño de la Imagen
Sinograma = zeros(NR,NProy);% Inicializo el Sinograma en cero. Las filas representan la variable R, mientras que las columnas la variable Phi
% La Imagen debe mapear la misma región que el Sinograma. Los
% valores de coordenadas (x,y) de cada píxel irán desde -RFOV a
% RFOV. El desplazamiento en x e y para cada incremento de píxel 
% dependerá de la cantidad de píxeles que tenga la imagen.
% Como x es el eje horizontal representa a las columnas en la imagen, mientras
% que el eje y representa las filas. 
Xmax = RFOV;                                % Valor máximo en X
Ymax = RFOV;                                % Valor máximo en Y
AnchoPixX = 2 * Xmax / (SizeIm(2));         % Ancho de Píxel. Es igual en X que en Y para imágenes cuadradas.
AnchoPixY = 2 * Ymax / (SizeIm(1));         % Ancho de Píxel. 
xax = -(Xmax-AnchoPixX/2) : AnchoPixX : Xmax-AnchoPixX/2;    % Valores de la variable x que puede tomar un píxel
yay = (Ymax-AnchoPixY/2 : -AnchoPixY : -(Ymax-AnchoPixY/2))';% Valores de la variable y que puede tomar un píxel


%% GENERACIÓN DEL SINOGRAMA
% El valor medio de eventos por pixel se obtiene del valor del pixel sobre
% el total de la imagen por la cantidad de eventos. Luego la cantidad de
% eventos generados en cada píxel tiene una distribución poisson.
eventosMedioPorPixel = CantEventos.* Imagen ./sum(sum(Imagen)); % Estimación del valor mediod e eventos de cada pixel
Sinograma = hist3([0 0],{ValoresPhi ValoresR});                             % Inicializo el Sinograma
% h=waitbar(0,'Generando Sinograma Simulado');
for i=1:SizeIm(1)
   for j=1:SizeIm(2)
       if Imagen(i,j)> 0
           Emisiones = rand(1,random('Poisson',eventosMedioPorPixel(i,j)))*180;         % Genero los eventos correspondientes al píxel (i,j) con ángulos aleatorios
           x = xax(j);                                                  % Posición en x del píxel (i,j)
           y = yay(i);                                                  % Posición en y del píxel (i,j)
           AuxR = x.*cos(Emisiones*pi/180)+y.*sin(Emisiones*pi/180);    % Valores de R para los eventos generados
           Sinograma = Sinograma + hist3([Emisiones' AuxR'],{ValoresPhi ValoresR});
       end
       % waitbar(((i-1)*SizeIm(2)+j)/(SizeIm(1)*SizeIm(2)),h);
   end
end
%% FIN
% Cierro el Waitbar
% close(h);
% Se devuelve Sinograma