%% FUNCTION PROJECTION
% Función que realiza la proyección de una imagen en distintos ángulos. O
% sea, realiza una transformada discreta de Radón. 
function Sinograma = Projection(Imagen,CantAngs,CantR)

%% INICIALIZACION DE VARIABLES
% Inicializo todas las variables necesarias para ejecutar el algoritmo.
[SizeY, SizeX] = size(Imagen);  % Tamaño de la imagen a proyectar.
Sinograma = zeros(CantAngs, CantR);     % Genero el Sinograma en ceros.
Angs = 0:180/CantAngs:179;                  % Valores de los ángulos de la proyección.
Rmax = floor(max(SizeX,SizeY)/2);  % Valor máximo de R.
PasoR = 2*Rmax/CantR;
Rs = -(Rmax-PasoR/2):PasoR:(Rmax-PasoR/2);    % Valores de los desplazamientos R según CantR.
                                    % Depende de las dimensiones de la imagen.
% El sistema de coordenadas utilizado tiene el origen (0,0) en el centro de
% la imagen. El eje vertical lo denominamos y, mientras que al horizontal
% x. El desplazamiento R es una distancia constante para todos los ángulos,
% compuesta de un movimiento en x y otro en y cuyas cantidades depende de
% cada ángulo.
offset = ceil((Rs)/2);
% La Imagen Reconstruída debe mapear la misma región que el Sinograma, por
% lo tanto los valores de coordenadas (x,y) de cada píxel irán desde -Rmax a
% Rmax, estos valores extremos solo estarán en la diagonal. El desplazamiento 
% en x e y para cada incremento de píxel dependerá de la cantidad de
% píxeles que tenga la imagen.
% Como x es el eje horizontal representa a las columnas en la imagen, mientras
% que el eje y representa las filas. Rmax se dará en la diagonal central,
% la cual tiene un �ngulo de atan(Filas/Columnas)
AngDiag = atan(SizeY/SizeX);        % Ángulo de la diagonal de la Imagen
Xmax = Rmax * cos(AngDiag);                 % Valor máximo en X
Ymax = Rmax * sin(AngDiag);                 % Valor máximo en Y

% Tamaño de píxel:

%%[SizePixelX, SizePixelY] = [2 * Xmax / (SizeX) 2 * Xmax / (SizeY)];         % Ancho de P�xel. Deber�a ser igual en X que en Y.
                                           
% Por lo tanto calculo uno solo
% Voy a mapear cada p�xel con una coordenada (x,y), la coordenada de cada
% p�xel estar� dada por la posici�n (x,y) del punto central del p�xel seg�n
% el sistema de coordenadas explicado arriba

% Matriz con las coordenadas x de todos los píxeles. Las mismas están
% calculadas respecto del centro de los píxeles:
xax = -(floor(SizeX/2)-0.5) : 1 : floor(SizeX/2)-0.5;    % Valores de la variable x que puede tomar un pixel
yay = ((floor(SizeY/2)-0.5) : -1 : -(floor(SizeY/2)-0.5))';% Valores de la variable y que puede tomar un p�xel
y = repmat(yay, 1,SizeY);               % Coordenada y para cada p�xel de la Imagen
x = repmat(xax, SizeX, 1);              % Coordenada x para cada p�xel de la Imagen
costheta = cos(Angs*pi/180);                % Coseno de cada �ngulo de proyecci�n
sintheta = sin(Angs*pi/180);                % Seno de cada �ngulo de proyecci�n
for i = 1 : numel(Angs)
%     r = x .* costheta(i) - y.*sintheta(i);
%     Sinograma(i,:) = hist(r(:), Rs);
    imRot = imrotate(Imagen,Angs(i), 'nearest', 'crop');
    Sinograma(i,:) = resample(sum(imRot,1), CantR, SizeY);
end
