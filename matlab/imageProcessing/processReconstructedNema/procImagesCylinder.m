% Función que procesa las imágenes para evaluar la performance de la
% reconstrucción y armar gráficos comaprativos:
function [meanFondo, desvioBackground, desvioNormBackground] = procImagesCylinder(volumen, sizePixel_mm, slicesAprocesar, mostrarResultadosParciales)
% Necesito algunas funciones generales:
addpath('/sources/MATLAB/WorkingCopy/ImageProcessing');
addpath('/sources/MATLAB/WorkingCopy/export_fig');
if nargin == 3
    sliceCentral = round(size(volumen,3)/2);
    mostrarResultadosParciales = 0;
end
%% INICIALIZACIÓN DE VARIABLES
sizeImage_pixels = size(volumen);
sizeImage_mm = sizePixel_mm .* sizeImage_pixels;
% El x vanza como los índices, osea a la izquierda es menor, a la derecha
% mayor.
coordX = -((sizeImage_mm(2)/2)-sizePixel_mm/2):sizePixel_mm:((sizeImage_mm(2)/2)-sizePixel_mm/2);
% El y y el z van al revés que los índices, o sea el valor geométrico va a
% contramano con los índices de las matrices.
coordY = ((sizeImage_mm(1)/2)-sizePixel_mm/2):-sizePixel_mm:-((sizeImage_mm(1)/2)-sizePixel_mm/2);
coordZ = -((sizeImage_mm(3)/2)-sizePixel_mm/2):sizePixel_mm:((sizeImage_mm(3)/2)-sizePixel_mm/2);
% El z no lo uso para las ROIs:
[X,Y] = meshgrid(coordX, coordY);
origin = [0 0 0];

% Genero 12 ROIs:
radioROIs_mm = 15;
anguloROIs_deg = 45 : 45 : 360;
distanciaCentroROIs_mm = 60; % Tiene que ser menor que 147-15 y mayor que 37+15.
centroX_ROIsFondo_mm = distanciaCentroROIs_mm * cosd(anguloROIs_deg);
centroY_ROIsFondo_mm = distanciaCentroROIs_mm * sind(anguloROIs_deg);
% Agrego una ROI en el centro en (0,0):
centroX_ROIsFondo_mm = [centroX_ROIsFondo_mm 0];
centroY_ROIsFondo_mm = [centroY_ROIsFondo_mm 0];


% Para ver la iamgen con zoom:
indicesFilasCol = round(sizeImage_pixels(1)/4):(sizeImage_pixels(1)-round(sizeImage_pixels(1)/4));
sizeZoomedImage_pixels = [numel(indicesFilasCol) numel(indicesFilasCol)];
% Proceso todos los slices
for i = 1 : numel(slicesAprocesar)
    % Slice a procesar:
    sliceEnProceso = volumen(:,:,slicesAprocesar(i));
    % Inicializo la máscara de fondo:
    maskRoiFondo = logical(zeros(size(sliceEnProceso)));
    maskROIsFondo = logical(zeros(size(sliceEnProceso)));
    if mostrarResultadosParciales
        h2 =figure;
        imshow(volumen(indicesFilasCol,indicesFilasCol,slicesAprocesar(i))./max(max(volumen(indicesFilasCol,indicesFilasCol,slicesAprocesar(i)))));
        hold on;
    end
    for j = 1 : numel(centroX_ROIsFondo_mm)
        if mostrarResultadosParciales
            xCirc_mm = (centroX_ROIsFondo_mm(j)-radioROIs_mm) : 0.1 : (centroX_ROIsFondo_mm(j)+radioROIs_mm);
            xCirc_pixels = xCirc_mm ./ sizePixel_mm(2) + sizeZoomedImage_pixels(2)/2; % Las coordenadas en mm se centran en cero y en pixeles es en el tamalo de la imagen sobre 2.
            yCirc_pos_pixels = -(sqrt(radioROIs_mm.^2 - (xCirc_mm-centroX_ROIsFondo_mm(j)).^2) + centroY_ROIsFondo_mm(j)) ./ sizePixel_mm(1) + sizeZoomedImage_pixels(1)/2; % El eje y va al revés.
            yCirc_neg_pixels = -(-sqrt(radioROIs_mm.^2 - (xCirc_mm-centroX_ROIsFondo_mm(j)).^2) + centroY_ROIsFondo_mm(j)) ./ sizePixel_mm(1) + sizeZoomedImage_pixels(1)/2;
            plot(xCirc_pixels,yCirc_pos_pixels,'LineWidth',2);
            plot(xCirc_pixels,yCirc_neg_pixels,'LineWidth',2); 
        end
        % Máscara para esta ROI:
        maskRoiFondo = (((X - centroX_ROIsFondo_mm(j)).^2 + (Y - centroY_ROIsFondo_mm(j)).^2) < radioROIs_mm.^2);
        % Obtengo los valores para la ROI:
        meanBackgroundRoi(i,j) = mean(sliceEnProceso(maskRoiFondo));
        stdBackgroundRoi(i,j) = std(sliceEnProceso(maskRoiFondo));
        % Máscara de Fondo con todas las ROIs (Para visualización):
        maskROIsFondo = maskROIsFondo | maskRoiFondo;
    end
    if mostrarResultadosParciales
        h3 = figure;
        imshow(maskROIsFondo);
    end
    % Media de todas las ROIs en el slice:
    meanFondo(i) = mean(meanBackgroundRoi(i,:));
    % Obtengo el desvío en el slice:
    %  Calculo el desvío en zona uniforme, que se obtiene calculando el desvio de las cuentas promedio de las 60 esferas:
    desvioBackground(i) = std(meanBackgroundRoi(i,:));
    % El desvío normalizado:
    desvioNormBackground(i) = desvioBackground(i) ./ meanFondo(i);

    % Si genere los gráficos guardo la imagen:
%     if mostrarResultadosParciales && (i == round(numel(slicesAprocesar)/2))
%         figure(h2);
%         colormap(hot);
%         set(gcf, 'Position', [100 100 1600 1000]);
%         set(gcf,'PaperPositionMode','auto');    % Para que lo guarde en el tamaño modificado.
%         saveas(gcf, 'ROIsCylinderPhantom', 'fig');
%         export_fig(['ROIsCylinderPhantom_exp_fig.png'])
%         saveas(gca, 'ROIsCylinderPhantom', 'epsc');
%     end
    % Cierro ventanas de visualización para que no se vayan acumulando:
    if mostrarResultadosParciales
        close(h2);
        close(h3);
    end
    
end



