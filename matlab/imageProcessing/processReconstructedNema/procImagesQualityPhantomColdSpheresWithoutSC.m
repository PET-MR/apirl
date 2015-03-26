%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 25/03/2015
%  *********************************************************************
%  This function computes the contrast recovery for each spheres but it
%  uses just a neighbour ROI to compute the mean value of the background,
%  instead of the mean of all the rois. This is because is intended to use
%  in images without scatter correction, where a bias in the background is
%  found.
function [contrastRecovery, desvioBackground, desvioNormBackground, radioEsferas_mm, centro_pixels] = procImagesQualityPhantomColdSpheresWithoutSC(volumen, sizePixel_mm, center_mm, sliceCentral, mostrarResultadosParciales)
% Necesito algunas funciones generales:
if nargin == 3
    sliceCentral = round(size(volumen,3)/2);
    mostrarResultadosParciales = 0;
end
% Redimensiono la imagen, para que el tamaño de píxel sea 1.0 mm (Con 0.5 mm se hace muy grande la memoria). Esto es
% para tener precisión en el cálculo, ya que la norma pide precisión como
% mínimo de 1 mm. Los slices los dejo del mismo tamaño.
% sizePixelDeseado_mm = [1.0 1.0];
% if((sizePixelDeseado_mm(1) < sizePixel_mm(1))||(sizePixelDeseado_mm(2) < sizePixel_mm(1)))
%     factorEscala = max(sizePixel_mm(1)/sizePixelDeseado_mm(1), sizePixel_mm(2)/sizePixelDeseado_mm(2));
%     sizeEscalado = round(size(volumen) .* [factorEscala factorEscala 1]);
%     volumenEscalado = zeros(sizeEscalado);
%     for z = 1 : size(volumen,3)
%         volumenEscalado(:,:,z) = imresize(volumen(:,:,z),[sizeEscalado(1) sizeEscalado(2)], 'nearest');
%     end
%     % Una vez reescalado reemplazo los nombres de variables:
%     sizePixel_mm =  sizePixel_mm .* size(volumen) ./ size(volumenEscalado);
%     volumen = volumenEscalado;
% end
%% INICIALIZACIÓN DE VARIABLES
% Recibo como parámetro la imagen que debe ser todo el volumen y el tamaño
% de píxel (vector con 3 elementos).

% Coordenadas necesarias para generar las distintas ROIs:
radioEsferas_mm = 1.0 .* [10 13 17 22 28 37] ./2 ;
% Indice que indice si es hot sphere (1) o cold sphere (0). Este fantoma es
% todo frío.
indicesHotSpheres = logical([0 0 0 0 0 0]);
centroZ_esferas_mm = (sliceCentral-1) .* sizePixel_mm(3) + sizePixel_mm(3)/2; % Debe ser el centro del slice.
sizeImage_pixels = size(volumen);
sizeImage_mm = sizePixel_mm .* sizeImage_pixels;
% El x vanza como los índices, osea a la izquierda es menor, a la derecha
% mayor.
coordX = -((sizeImage_mm(2)/2)-sizePixel_mm/2):sizePixel_mm:((sizeImage_mm(2)/2)-sizePixel_mm/2);
% El y y el z van al revés que los índices, o sea el valor geométrico va a
% contramano con los índices de las matrices.
coordY = ((sizeImage_mm(1)/2)-sizePixel_mm/2):-sizePixel_mm:-((sizeImage_mm(1)/2)-sizePixel_mm/2);
coordZ = -((sizeImage_mm(3)/2)-sizePixel_mm/2):sizePixel_mm:((sizeImage_mm(3)/2)-sizePixel_mm/2);
[X,Y,Z] = meshgrid(coordX, coordY, coordZ);
origin = [0 0 0];
% Dentro del plano transversal las esferas están ubicadas con sus centros
% equiespaciados angularmente, y a una distancia de 57,2mm del centro del
% fantoma. Estando además la esfera de 17 sobre el eje horizontal, del lado
% izquierdo. A partir de esto obtengo los centros en x e y de cada esfera:
anguloEsferas_deg = 360-60 : -60 : 0;
distanciaCentroEsferas_mm = 58;
centroX_esferas_mm = (distanciaCentroEsferas_mm-center_mm(1)) * cosd(anguloEsferas_deg);
centroY_esferas_mm = (distanciaCentroEsferas_mm-center_mm(2)) * sind(anguloEsferas_deg);
% Sin lung insert. No lo agrego.
centroY_esferas_mm(2) = centroY_esferas_mm(2) + 2;

centro_pixels = round([(centroX_esferas_mm'+sizeImage_mm(2)/2)./sizePixel_mm(2) (centroY_esferas_mm'+sizeImage_mm(1)/2)./sizePixel_mm(1)]);
% Centros esferas ROIs, tienen que estar como mínimo a 15 mm del borde y no
% deben llegar a estar a 15 mm de la esfera. En la media circunferencia del
% fantoma coloco 9 (luego elimino 2 que no sirven):
anguloROIs_deg = 0 : 180/8 : 180;
offsetY_mm = -35; 
distanciaCentroROIs_mm = 75 - offsetY_mm; % Tiene que ser menor que 147-15 y mayor que 37+15.
centroX_ROIsFondo_mm = distanciaCentroROIs_mm * cosd(anguloROIs_deg);
centroY_ROIsFondo_mm = distanciaCentroROIs_mm * sind(anguloROIs_deg) + offsetY_mm;
% La 4 y 6 no sirven:
centroX_ROIsFondo_mm([4 6]) = [];
centroY_ROIsFondo_mm([4 6]) = [];
% Esas dos las agrego en diagonal hacia abajo (me baso en la primera y
% ultima que genere antes:
centroX_ROIsFondo_mm = [centroX_ROIsFondo_mm centroX_ROIsFondo_mm(1)-20 centroX_ROIsFondo_mm(end)+20];
centroY_ROIsFondo_mm = [centroY_ROIsFondo_mm centroY_ROIsFondo_mm(1)-35 centroY_ROIsFondo_mm(end)-35];
% Ahora agrego las 3 ROIs que faltan en la parte recta de abajo (y va entre
% -35 y -35-77y recorren de 70 a -70:
stepX_mm = 120/2; % Doy 10 mm de margen a cada lado para que quede lejos del borde.
centroX_ROIsFondo_mm = [centroX_ROIsFondo_mm -60:stepX_mm:60];
centroY_ROIsFondo_mm = [centroY_ROIsFondo_mm -75 -75 -75];

% In this processing, each sphere is contrasted with the mean value of a
% nieghbour ROIs.
indexNeighbourghsRois = [11 12; 10 11; 6 5; 4 5; 3 4; 3 2];

if mostrarResultadosParciales
    h4 =figure;
    imshow(volumen(:,:,sliceCentral)./max(max(volumen(:,:,sliceCentral))));
    colormap(hot);
    hold on;
end
        
% Acontinuación debo procesar cada una de las esferas, en NEMA el
% procedimiento indica que incluso para el fondo se debe utilizar el mismo
% tamaño de esfera. Acá uso una esfera más grande:
for i = 1 : numel(radioEsferas_mm)
    if mostrarResultadosParciales
        h1 =figure;
        imshow(volumen(:,:,sliceCentral)./max(max(volumen(:,:,sliceCentral))));
        colormap(hot);
        hold on;
    end
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
    maskEsferaCaliente = ((X - centroX_esferas_mm(i)).^2 + (Y - centroY_esferas_mm(i)).^2) < radioEsferas_mm(i).^2;
    if mostrarResultadosParciales
        h2 = figure;
        imshow(maskEsferaCaliente(:,:,sliceCentral));
    end
    % Slice a procesar:
    sliceEnProceso = volumen(:,:,sliceCentral);
    % Con la esfera principal obtengo las cuentas de la esfera caliente:
    meanHotSpheres(i) = mean(mean(sliceEnProceso(maskEsferaCaliente(:,:,sliceCentral))));
    % Ahora grafico y proceso las esferas de fondo:
    if mostrarResultadosParciales
        figure(h1);
    end
    % Inicializo la máscara de fondo:
    maskROIsFondo = logical(zeros(size(maskEsferaCaliente)));
    % Cahnge to indexRadio=i to use roi of the same size of the sphere:
    indexRadio = numel(radioEsferas_mm);
    % Not all the spheres, just two for neighbourghs for each roi: 
    for j = 1 : size(indexNeighbourghsRois,2)
        if mostrarResultadosParciales
            xCirc_mm = (centroX_ROIsFondo_mm(indexNeighbourghsRois(i,j))-radioEsferas_mm(indexRadio)) : 0.1 : (centroX_ROIsFondo_mm(indexNeighbourghsRois(i,j))+radioEsferas_mm(indexRadio));
            xCirc_pixels = xCirc_mm ./ sizePixel_mm(2) + sizeImage_pixels(2)/2; % Las coordenadas en mm se centran en cero y en pixeles es en el tamalo de la imagen sobre 2.
            yCirc_pos_pixels = -(sqrt(radioEsferas_mm(indexRadio).^2 - (xCirc_mm-centroX_ROIsFondo_mm(indexNeighbourghsRois(i,j))).^2) + centroY_ROIsFondo_mm(indexNeighbourghsRois(i,j))) ./ sizePixel_mm(1) + sizeImage_pixels(1)/2; % El eje y va al revés.
            yCirc_neg_pixels = -(-sqrt(radioEsferas_mm(indexRadio).^2 - (xCirc_mm-centroX_ROIsFondo_mm(indexNeighbourghsRois(i,j))).^2) + centroY_ROIsFondo_mm(indexNeighbourghsRois(i,j))) ./ sizePixel_mm(1) + sizeImage_pixels(1)/2;
            plot(xCirc_pixels,yCirc_pos_pixels,'LineWidth',2);
            plot(xCirc_pixels,yCirc_neg_pixels,'LineWidth',2); 
        end
        % Máscara para esta ROI:
        maskRoiFondo = (((X - centroX_ROIsFondo_mm(indexNeighbourghsRois(i,j))).^2 + (Y - centroY_ROIsFondo_mm(indexNeighbourghsRois(i,j))).^2) < radioEsferas_mm(i).^2);
        % Obtengo el valor medio de cada ROI y la guardo en una matriz de 3
        % dimensiones, la primera indica la esfera que se está analizando,
        % la segunda la ROI dentro del slice y la tercera el slice:
        for k = 1:5   %Voy desde clice central-2 a slice central +2:
            sliceEnProceso = volumen(:,:,sliceCentral-3+k);
            meanBackgroundRoi(i,j,k) = mean(sliceEnProceso(maskRoiFondo(:,:,sliceCentral-3+k)));
        end
        % Máscara de Fondo con todas las ROIs (Para visualización):
        maskROIsFondo = maskROIsFondo | maskRoiFondo;
    end
    if mostrarResultadosParciales
        h3 = figure;
        imshow(maskROIsFondo(:,:,sliceCentral));
    end
    
    % Recuperación de contraste para esa ROI:
    meanFondo(i) = mean(meanBackgroundRoi(:));
    % Calculo recuperacion de contraste para todas las esferas:
    if(indicesHotSpheres(i))
        contrastRecovery(i) = (meanHotSpheres(i)/meanFondo(i)-1)/(relacionHotBackground-1)*100;
    else
        contrastRecovery(i) = (1-(meanHotSpheres(i) ./ meanFondo(i))) *100;
    end
    
    % For the std dev in the backgorund take all the rois:
    if mostrarResultadosParciales
        figure(h4);
        hold on;
    end
    for j = 1 : numel(centroX_ROIsFondo_mm)
        if mostrarResultadosParciales
            xCirc_mm = (centroX_ROIsFondo_mm(j)-radioEsferas_mm(i)) : 0.1 : (centroX_ROIsFondo_mm(j)+radioEsferas_mm(i));
            xCirc_pixels = xCirc_mm ./ sizePixel_mm(2) + sizeImage_pixels(2)/2; % Las coordenadas en mm se centran en cero y en pixeles es en el tamalo de la imagen sobre 2.
            yCirc_pos_pixels = -(sqrt(radioEsferas_mm(i).^2 - (xCirc_mm-centroX_ROIsFondo_mm(j)).^2) + centroY_ROIsFondo_mm(j)) ./ sizePixel_mm(1) + sizeImage_pixels(1)/2; % El eje y va al revés.
            yCirc_neg_pixels = -(-sqrt(radioEsferas_mm(i).^2 - (xCirc_mm-centroX_ROIsFondo_mm(j)).^2) + centroY_ROIsFondo_mm(j)) ./ sizePixel_mm(1) + sizeImage_pixels(1)/2;
            plot(xCirc_pixels,yCirc_pos_pixels,'LineWidth',2);
            plot(xCirc_pixels,yCirc_neg_pixels,'LineWidth',2); 
        end
        % Máscara para esta ROI:
        maskRoiFondo = (((X - centroX_ROIsFondo_mm(j)).^2 + (Y - centroY_ROIsFondo_mm(j)).^2) < radioEsferas_mm(i).^2);
        % Obtengo el valor medio de cada ROI y la guardo en una matriz de 3
        % dimensiones, la primera indica la esfera que se está analizando,
        % la segunda la ROI dentro del slice y la tercera el slice:
        for k = 1:5   %Voy desde clice central-2 a slice central +2:
            sliceEnProceso = volumen(:,:,sliceCentral-3+k);
            meanBackgroundRoi(i,j,k) = mean(sliceEnProceso(maskRoiFondo(:,:,sliceCentral-3+k)));
            squareErrorBackgroundRoi(i,j,k) = sum((sliceEnProceso(maskRoiFondo(:,:,sliceCentral-3+k))-meanBackgroundRoi(i,j,k)).^2);
        end
        % Máscara de Fondo con todas las ROIs (Para visualización):
        maskROIsFondo = maskROIsFondo | maskRoiFondo;
        
    end
    %  Calculo el desvío en zona uniforme, que se obtiene calculando el desvio de las cuentas promedio de las 60 esferas
    % el error cuadrático dentro de cada esfera ya lo calcule previamente.
    meanFondo(i) = mean(meanBackgroundRoi(i,:));
    % ahora sumo y divido por n-1:
    %desvioBackground(i) = sqrt(sum(squareErrorBackgroundRoi(i,:)) / (numel(squareErrorBackgroundRoi(i,:))-1));
    desvioBackground(i) = std(meanBackgroundRoi(i,:));
    % El desvío normalizado:
    desvioNormBackground(i) = desvioBackground(i) ./ meanFondo(i);

    % Cierro ventanas de visualización para que no se vayan acumulando:
    if mostrarResultadosParciales
        close(h2);
        close(h3);
    end
end

% Si genere los gráficos guardo la imagen:
if mostrarResultadosParciales
    figure(h1);
    %set(gcf, 'Position', [100 100 1600 1000]);
    set(gcf,'PaperPositionMode','auto');    % Para que lo guarde en el tamaño modificado.
    saveas(gcf, 'ROIsIqPhantom', 'fig');
%    export_fig(['ROIsIqPhantom_exp_fig.png'])
    saveas(gca, 'ROIsIqPhantom', 'epsc');
    
    figure(h4);
    %set(gcf, 'Position', [100 100 1600 1000]);
    set(gcf,'PaperPositionMode','auto');    % Para que lo guarde en el tamaño modificado.
    saveas(gcf, 'BackgroundROIsIqPhantom', 'fig');
%    export_fig(['ROIsIqPhantom_exp_fig.png'])
    saveas(gca, 'BackgroundROIsIqPhantom', 'epsc');
end

