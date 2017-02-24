% Función que procesa las imágenes para evaluar la performance de la
% reconstrucción y armar gráficos comaprativos:
function [maskSphereROIs, maskBackgroundROIs] = generateRoisForNemaPhantom(volumen, sizePixel_mm, centrePhantom_pixels, offsetsSpheres_mm, anguloEsferas_deg, mostrarResultadosParciales)
if nargin == 3
    [value,centralSlice] = max(mean(mean(recon)));
    mostrarResultadosParciales = 0;
    anguloEsferas_deg = 60 : 60 : 360;
    offsetsSpheres_mm = zeros(3,6);
else
    centralSlice = centrePhantom_pixels(3);
end

%% INICIALIZACIÓN DE VARIABLES
% Recibo como parámetro la imagen que debe ser todo el volumen y el tamaño
% de píxel (vector con 3 elementos).
% Coordenadas necesarias para generar las distintas ROIs:
radioEsferas_mm = 0.9 .* [10 13 17 22 28 37] ./2 ;
% Indice que indice si es hot sphere (1) o cold sphere (0):
indicesHotSpheres = logical([1 1 1 1 0 0]);
centroZ_esferas_mm = (centralSlice-1) .* sizePixel_mm(3) + sizePixel_mm(3)/2; % Debe ser el centro del slice.
sizeImage_pixels = size(volumen);
sizeImage_mm = sizePixel_mm .* sizeImage_pixels;
% El x vanza como los índices, osea a la izquierda es menor, a la derecha
% mayor.
coordX = -((sizeImage_mm(2)/2)-sizePixel_mm(2)/2):sizePixel_mm(2):((sizeImage_mm(2)/2)-sizePixel_mm(2)/2);
% El y y el z van al revés que los índices, o sea el valor geométrico va a
% contramano con los índices de las matrices.
coordY = ((sizeImage_mm(1)/2)-sizePixel_mm(1)/2):-sizePixel_mm(1):-((sizeImage_mm(1)/2)-sizePixel_mm(1)/2);
coordZ = -((sizeImage_mm(3)/2)-sizePixel_mm(3)/2):sizePixel_mm(3):((sizeImage_mm(3)/2)-sizePixel_mm(3)/2);
% Centre in mm:
centrePhantom_mm = [coordY(centrePhantom_pixels(1)) coordX(centrePhantom_pixels(2)) coordZ(centrePhantom_pixels(3))];
[X,Y,Z] = meshgrid(coordX, coordY, coordZ);
origin = [0 0 0];
% Dentro del plano transversal las esferas están ubicadas con sus centros
% equiespaciados angularmente, y a una distancia de 57,2mm del centro del
% fantoma. Estando además la esfera de 17 sobre el eje horizontal, del lado
% izquierdo. A partir de esto obtengo los centros en x e y de cada esfera:
distanciaCentroEsferas_mm = 57;
centroX_esferas_mm = distanciaCentroEsferas_mm * cosd(anguloEsferas_deg) + centrePhantom_mm(2) + offsetsSpheres_mm(2,:);
centroY_esferas_mm = distanciaCentroEsferas_mm * sind(anguloEsferas_deg) + centrePhantom_mm(1) + offsetsSpheres_mm(1,:);
% Agrego una esfera más que es para analizar el inserto deLung:
radioEsferas_mm = [radioEsferas_mm 30/2];
% La misma se ubica en (0,0):
centroX_esferas_mm = [centroX_esferas_mm centrePhantom_mm(2)];
centroY_esferas_mm = [centroY_esferas_mm centrePhantom_mm(1)];

% Centros esferas ROIs, tienen que estar como mínimo a 15 mm del borde y no
% deben llegar a estar a 15 mm de la esfera. En la media circunferencia del
% fantoma coloco 9 (luego elimino 2 que no sirven):
anguloROIs_deg = 0 : 180/8 : 180;
distanciaCentroROIs_mm = 130; % Tiene que ser menor que 147-15 y mayor que 37+15.
centroX_ROIsFondo_mm = distanciaCentroROIs_mm * cosd(anguloROIs_deg) + centrePhantom_mm(2);
centroY_ROIsFondo_mm = distanciaCentroROIs_mm * sind(anguloROIs_deg) + centroY_esferas_mm(3); % Centred in the same y position as the second and third spheres.
% La 4 y 6 no sirven:
centroX_ROIsFondo_mm([4 6]) = [];
centroY_ROIsFondo_mm([4 6]) = [];
% Correct the spheres in the x direction to fit them in the phantom:
offsetsXbackgroundROIs_mm = [-25 -20 -10 0 10 20 25];
offsetsYbackgroundROIs_mm = [10 0 0 5 0 0 10];
centroX_ROIsFondo_mm = centroX_ROIsFondo_mm + offsetsXbackgroundROIs_mm;
centroY_ROIsFondo_mm = centroY_ROIsFondo_mm + offsetsYbackgroundROIs_mm;
% Esas dos las agrego en diagonal hacia abajo (me baso en la primera y
% ultima que genere antes:
centroX_ROIsFondo_mm = [centroX_ROIsFondo_mm centroX_ROIsFondo_mm(1)-18 centroX_ROIsFondo_mm(end)+18];
centroY_ROIsFondo_mm = [centroY_ROIsFondo_mm centroY_ROIsFondo_mm(1)-35 centroY_ROIsFondo_mm(end)-35];

% Ahora agrego las 3 ROIs que faltan en la parte recta de abajo (y va entre
% -35 y -35-77y recorren de 70 a -70:
stepX_mm = 45; % Doy 10 mm de margen a cada lado para que quede lejos del borde.
centroX_ROIsFondo_mm = [centroX_ROIsFondo_mm -45:stepX_mm:45];
centroY_ROIsFondo_mm = [centroY_ROIsFondo_mm -75 -75 -75];



% Acontinuación debo procesar cada una de las esferas, ya que el
% procedimiento indica que incluso para el fondo se debe utilizar el mismo
% tamaño de esfera:
if mostrarResultadosParciales
    h1 =figure;
    set(gcf, 'Position', [100 100 900 900]);
    imshow(volumen(:,:,centralSlice)./max(max(volumen(:,:,centralSlice))));
    hold on;
end
for i = 1 : numel(radioEsferas_mm)
    % Genero los puntos que forman parte de la circunferencia:
    if mostrarResultadosParciales
        xCirc_mm = (centroX_esferas_mm(i)-radioEsferas_mm(i)) : 0.1 : (centroX_esferas_mm(i)+radioEsferas_mm(i));
        xCirc_pixels = xCirc_mm ./ sizePixel_mm(2) + sizeImage_pixels(2)/2; % Las coordenadas en mm se centran en cero y en pixeles es en el tamalo de la imagen sobre 2.
        yCirc_pos_pixels = -(sqrt(radioEsferas_mm(i).^2 - (xCirc_mm-centroX_esferas_mm(i)).^2) + centroY_esferas_mm(i)) ./ sizePixel_mm(1) + sizeImage_pixels(1)/2; % El eje y va al revés.
        yCirc_neg_pixels = -(-sqrt(radioEsferas_mm(i).^2 - (xCirc_mm-centroX_esferas_mm(i)).^2) + centroY_esferas_mm(i)) ./ sizePixel_mm(1) + sizeImage_pixels(1)/2;
        plot(xCirc_pixels,yCirc_pos_pixels,'LineWidth',2);
        plot(xCirc_pixels,yCirc_neg_pixels,'LineWidth',2);  
    end
    % Mascara para esfera caliente:
    maskSphereROIs{i} = ((X(:,:,centralSlice) - centroX_esferas_mm(i)).^2 + (Y(:,:,centralSlice) - centroY_esferas_mm(i)).^2) < radioEsferas_mm(i).^2; % Now, only for an slice, before:((X - centroX_esferas_mm(i)).^2 + (Y - centroY_esferas_mm(i)).^2) < radioEsferas_mm(i).^2

    % Inicializo la máscara de fondo:
    for j = 1 : numel(centroX_ROIsFondo_mm)
        maskBackgroundROIs{i,j} = logical(zeros(size(maskSphereROIs{i})));
        if mostrarResultadosParciales
            xCirc_mm = (centroX_ROIsFondo_mm(j)-radioEsferas_mm(i)) : 0.1 : (centroX_ROIsFondo_mm(j)+radioEsferas_mm(i));
            xCirc_pixels = xCirc_mm ./ sizePixel_mm(2) + sizeImage_pixels(2)/2; % Las coordenadas en mm se centran en cero y en pixeles es en el tamalo de la imagen sobre 2.
            yCirc_pos_pixels = -(sqrt(radioEsferas_mm(i).^2 - (xCirc_mm-centroX_ROIsFondo_mm(j)).^2) + centroY_ROIsFondo_mm(j)) ./ sizePixel_mm(1) + sizeImage_pixels(1)/2; % El eje y va al revés.
            yCirc_neg_pixels = -(-sqrt(radioEsferas_mm(i).^2 - (xCirc_mm-centroX_ROIsFondo_mm(j)).^2) + centroY_ROIsFondo_mm(j)) ./ sizePixel_mm(1) + sizeImage_pixels(1)/2;
            plot(xCirc_pixels,yCirc_pos_pixels,'LineWidth',2);
            plot(xCirc_pixels,yCirc_neg_pixels,'LineWidth',2); 
        end
        % Máscara para esta ROI:
        maskBackgroundROIs{i,j} = (((X(:,:,centralSlice) - centroX_ROIsFondo_mm(j)).^2 + (Y(:,:,centralSlice) - centroY_ROIsFondo_mm(j)).^2) < radioEsferas_mm(i).^2);
    end
end

