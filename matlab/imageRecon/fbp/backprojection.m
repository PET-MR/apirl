%% FUNCION BACKPROJECTION
% Autor: Martín Belzunce.
% 
% Institución: UTN.
% 
% Última Fecha de Modificación: 19/11/2008.
% 
% Adaptación de la función iradon de MATLAB
%
% Esta funci�n realiza el algoritmo de Backprojection Filtrada, para
% reconstruir una imagen a partir de un Sinograma. Se le debe pasar como
% parametros el Sinograma, el tama�o de la imagen a reconstruir, el
% filtro a aplicar y su frecuencia de corte.
%
% Esta funci�n presupone que todas las proyecciones se han hecho en
% intervalos equiespaseados de �ngulos entre 0 y 179.
%
% La cantidad de posiciones sobre las que se desplazan las proyecciones
% de un mismo �ngulo deben ser impares. De esta forma, los valore de R
% pueden estar centrados en cero e ir de -Rmax:0:Rmax. Durante la operaci�n
% de BackProjection se debe llenar para los distintos valores continuos de R
% calculados, una matriz con valores de R fijos. Por lo que se debe
% realizar una interpolaci�n, la misma es seleccionada en el quinto
% par�metro de entrada de la misma.
% 
% Devuelve la Imagen reconstru�da sin normalizar cuando se le pide solo un
% valor de salida. Si se le piden dos valores de salida, devulve la Imagen,
% y la respuesta del filtro aplicado.
%
% Vale aclarar que el sinograma contiene los �ngulos en las Filas, y los
% offsets en las columnas. Por esta raz�n, se debe tener en cuenta que
% MATLAB realiza la transformada de rad�n, ordenando los datos al rev�s.
% Por esa raz�n se recomienda utilizar la funci�n Projection.
%
% Prototipos: Imagen = BackProjection(Sinograma, SizeIm, Filtro, Fc)
% [Imagen,Hfiltro] = BackProjection(Sinograma, SizeIm, Filtro, Fc)
% [Imagen,Hfiltro] = BackProjection(Sinograma, SizeIm, Filtro, Fc, MetodoInterp)
%
% Filtros Disponibles: 'Rampa', 'Shepp-Logan', 'Cosine', 'Hamming', 'Hann',
% 'Ninguno'
% Metodos de Interpolaci�n Disponibles:
% 'vecino','lineal','spline','pchip','cubic','v5cubic'.

function varargout = BackProjection(varargin)
if (nargout ~= 1) & (nargout ~= 2)
    disp 'Cantidad de argumentos de salida incorrecta. Llamar a "help BackProjection" para m�s informaci�n.'
    return
end
if nargin == 1
    Sinograma = varargin{1};
    SizeIm = [256 256];
    Filtro = 'Shepp-Logan';
    Fc = 0.5;
    MetodoInterp = 'vecino';
elseif nargin == 4
    Sinograma = varargin{1};
    SizeIm = varargin{2};
    Filtro = varargin{3};
    Fc = varargin{4};
    MetodoInterp = 'vecino';
elseif nargin == 5
    Sinograma = varargin{1};
    SizeIm = varargin{2};
    Filtro = varargin{3};
    Fc = varargin{4};
    MetodoInterp = varargin{5};
else
    disp 'Cantidad de argumentos de entrada incorrecta. Llamar a "help BackProjection" para m�s informaci�n.'
    return
end
if(size(SizeIm) ~= [1 2])
    disp 'El campo SizeIm debe ser una vector de 2 Filas con los campos [Filas Columnas] de la imagen'
    return
end
%% INICIALIZACION DE VARIABLES
% Inicializo todas las variables necesarias para ejecutar el algoritmo
[CantAngs,CantR] = size(Sinograma);         % Cantidad de �ngulos y Rs del Sinograma
Proyeccion = zeros(CantR,1);                % Tama�o de cada proyecci�n en un �ngulo phi cualquiera
stepAng = 180/CantAngs;
Angs = 0+stepAng/2:stepAng:180-stepAng/2;                  % Valores de los �ngulos de la proyecci�n
Rmax = floor(CantR/2);                      % Valor m�ximo de R         
PasoR = 2*Rmax/CantR;
Rs = -(Rmax-PasoR/2):PasoR:(Rmax-PasoR/2);    % Valores de los desplazamientos R según CantR. D
% El sistema de coordenadas utilizado tiene el origen (0,0) en el centro de
% la imagen. El eje vertical lo denominamos y, mientras que al horizontal
% x. El desplazamiento R es una distancia constante para todos los �ngulos,
% compuesta de un movimiento en x y otro en y cuyas cantidades depende de
% cada �ngulo.
SinogramaFiltrado = zeros(CantAngs,CantR);
SinogramaAux = Sinograma;
offset = ceil((Rs)/2);
% La Imagen Reconstru�da debe mapear la misma regi�n que el Sinograma, por
% lo tanto los valores de coordenadas (x,y) de cada p�xel ir�n desde -Rmax a
% Rmax, estos valores extremos solo estar�n en la diagonal. El desplazamiento 
% en x e y para cada incremento de p�xel depender� de la cantidad de
% p�xeles que tenga la imagen.
% Como x es el eje horizontal representa a las columnas en la imagen, mientras
% que el eje y representa las filas. Rmax se dar� en la diagonal central,
% la cual tiene un �ngulo de atan(Filas/Columnas)
AngDiag = atan(SizeIm(1)/SizeIm(2));        % �ngulo de la diagonal de la Imagen
Xmax = Rmax * cos(AngDiag);                 % Valor m�ximo en X
Ymax = Rmax * sin(AngDiag);                 % Valor m�ximo en Y
AnchoPixX = 2 * Xmax / (SizeIm(2));         % Ancho de P�xel. Deber�a ser igual en X que en Y.
                                            % Por lo tanto calculo uno solo
AnchoPixY = 2 * Ymax / (SizeIm(1));         % Ancho de P�xel. Deber�a ser igual en X que en Y.
                                            % Por lo tanto calculo uno solo
% Voy a mapear cada p�xel con una coordenada (x,y), la coordenada de cada
% p�xel estar� dada por la posici�n (x,y) del punto central del p�xel seg�n
% el sistema de coordenadas explicado arriba
xax = -(Xmax-AnchoPixX/2) : AnchoPixX : Xmax-AnchoPixX/2;    % Valores de la variable x que puede tomar un pixel
yay = (Ymax-AnchoPixY/2 : -AnchoPixY : -(Ymax-AnchoPixY/2))';% Valores de la variable y que puede tomar un p�xel
y = repmat(yay, 1,SizeIm(2));               % Coordenada y para cada p�xel de la Imagen
x = repmat(xax, SizeIm(1), 1);              % Coordenada x para cada p�xel de la Imagen
costheta = cos(Angs*pi/180);                % Coseno de cada �ngulo de proyecci�n
sintheta = sin(Angs*pi/180);                % Seno de cada �ngulo de proyecci�n
Imagen = zeros(SizeIm);                     % Inicializaci�n de la Imagen con ceros
%% INICIALIZACION DE FILTRO
% Inicializo los valores del filtro a partir del filtro seleccionado y la
% frecuncia de corte pasada

order = 2^nextpow2(2*CantR);           % El orden del vector filtro lo hago potencia de dos para que le pueda aplicar la fft
H = 2*( 0:(order/2) )./order;       % Inicializo los valores de la Respueta del Filtro como un Filtro Rampa, ya que los dem�s
                                    % son una modulaci�n de la rampa
w = 2*pi*(0:size(H,2)-1)/order;     % Valores de frecuencia

switch Filtro
    case 'Rampa'
        % Filtro Rampa
        % No tengo que modificar la respuesta del filtro, simplemente pongo
        % en cero los valores que excedan la Freucnecia de corte
        H(w>pi*Fc) = 0;
    case 'Shepp-Logan'
        % Filtro Shepp-Logan
        H(2:end) = H(2:end).* (sin(w(2:end)/(2*Fc))./(w(2:end)/(2*Fc))); %H = H .* 2 .* Fc ./ pi .* sin(pi.*H./(2*Fc)); 
    case 'Cosine'
        % Filtro Coseno
        H(2:end) = H(2:end) .* cos(w(2:end)/(2*Fc)); %H = H .* cos(w/(2*Fc));
    case 'Hamming'  
        % Filtro Hamming
        H(2:end) = H(2:end) .* (.54 + .46 * cos(w(2:end)/Fc)); %H = H .* (.54 + .46 * cos(w./Fc));
    case 'Hann'
        % Filtro Hann
        H(2:end) = H(2:end) .*(1+cos(w(2:end)./Fc))./ 2; %H = 0.5*H.*(1+cos(w./Fc));
    case 'Ninguno'
        % No Aplico Filtro
        H = ones(order);              % Si no uso filtro, pongo el filtro todo en uno
    otherwise
        disp 'No se reconoce el Filtro requerido, llame a help Backprohection para ver los filtros disponibles';
        return;
end
if strcmp(Filtro,'Ninguno') == 0
    H(w>pi*Fc) = 0;
    H = [H' ; H(end-1:-1:2)'];          % Le doy la simetria a la respuesta del filtro
end
%% BACKPROJECTION FILTRADA
% Realiza la Backprojection Filtrada del sinograma, utilizando el filtro
% inicializado en la secci�n anterior
SinogramaAux(1,length(H))=0;        % Zero padding en los puntos que agregu� para la FFT 
ImFFT=fft(SinogramaAux,[],2);       % Calculo la FFT de las Proyecciones. 
                                    % Se calcula la FFT para cada fila del sinograma
                                    % O sea que como resultado tengo en
                                    % cada fila la FFT de la proyecci�n
                                    % para cada �ngulo
for i=1:CantAngs
    ImFFT(i,:)=ImFFT(i,:).*H';      % Realizo el filtrado en frecuencia de 
                                    % cada proyecci�n
end
SinogramaFiltrado = real(ifft(ImFFT,[],2)); % Calculo la Transformada Inversa de Fourier
                                            % de cada proyecci�n. De esta forma
                                            % obtengo las proyecciones ya
                                            % filtradas.
SinogramaFiltrado(:,CantR+1:end) = [];      % Vuelvo las proyecciones a su tama�o original
                                            % eliminando los puntos agregados para
                                            % la FFT.
% Con las proyecciones ya filtradas ahora realizo la Backprojection
% para cada proyecci�n
% La forma de realizarla depende del m�todo de inerpolaci�n seleccionado.
% El mismo se encuentra en la variable MetodoInterp
switch MetodoInterp
    case 'vecino'
        % Interpolaci�n de Vecino M�s Pr�ximo
        for k = 1 : CantAngs
            Proyeccion = SinogramaFiltrado(k,:);            % Separo la proyecci�n para el �ngulo K
            R = x*costheta(k)+y*sintheta(k);                % C�lculo todos los valores de R para cada 
           % Imagen=Imagen+Proyeccion(round(R+Rmax));        % Relleno la Imagen Final sumando las cuentas
           Imagen=Imagen+Proyeccion(ceil(R+Rmax));
                                                            % de cada valor de R de la
                                                            % proyecci�n, para su
                                                            % correspondiente p�xel (x,y).
                                                            % Sumo Rmax para poder indexar
                                                            % dentro de la matriz
        end
    case 'lineal'
        % Interpolaci�n
        for k = 1 : CantAngs
            Proyeccion = SinogramaFiltrado(k,:);            % Separo la proyecci�n para el �ngulo K
            R = x*costheta(k)+y*sintheta(k);                % C�lculo todos los valores de R para cada 
            a = floor(R);                                   % Se interpola entre a y a+1
            Imagen=Imagen+(R-a).*Proyeccion(round(R+Rmax))+(a+1-R).*Proyeccion(round(R+Rmax));  % Relleno la Imagen Final sumando una proporci�n de las cuentas
                                                                                                % en el p�xel a y otra parte en el p�xel a+1. (Inerpolaci�n Lineal)
                                                                                                % Sumo Rmax para poder indexar
                                                                                                % dentro de la matriz
        end
    
    case {'spline','pchip','cubic','v5cubic'}
       for k = 1 : CantAngs
            Proyeccion = SinogramaFiltrado(k,:);                % Separo la proyecci�n para el �ngulo K
            R = x*costheta(k)+y*sintheta(k);                    % C�lculo todos los valores de R para cada 
            Rinterp = interp1(Rs,Proyeccion,R(:),MetodoInterp); % Interpolaci�n de todos los valores de R
                                                                % en los posibles Rs. Me devulve
                                                                % un vector de SizeIm(1)xSizeIm(2).
            Imagen=Imagen+reshape(Rinterp,SizeIm(1),SizeIm(2)); % Relleno la Imagen Final sumando las cuentas
                                                                % de cada valor de R de la
                                                                % proyecci�n. Para esto redimensiono el
                                                                % vector en una imagen.
       end
    otherwise
        disp('M�todo de interpolaci�n incorrecto.'); 
        
end

%% ARGUMENTOS DE SALIDA
% Cargo los argumnetos de salida
varargout{1} = Imagen;  % Siempre se devulve la imagen
if nargout == 2
    varargout{2} = H    % Respuesta del Filtro
end