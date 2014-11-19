%  *********************************************************************
%  Proyecto AR-PET. Comisión Nacional de Energía Atómica.
%  Autor: Martín Belzunce. UTN-FRBA.
%  Fecha de Creación: 29/06/2012
%  *********************************************************************
%

% La SystemMatrix tiene en las filas las lors, y en las columnas los
% píxels:

function [imagen, likelihood] = Mlem2Dpenalized(sinogram2D, systemMatrixProjector, systemMatrixBackrojector, sizeImage, numIterations, enablePlot, strFilter, param)

sizeProjection = size(sinogram2D);
% La imagen empieza siendo una constante:
imagen = ones(sizeImage);
vectorImage = reshape(imagen, sizeImage(1)*sizeImage(2), 1);
% Genero un vector con el sinograma. Uso la traspuesta del sinograma,
% porque es como está ordenado en la systemmatrix. (orden fila en vez del
% roden colimna que usa matlab):
vectorSinogram = reshape(sinogram2D', numel(sinogram2D), 1);
% La sensibility Image es la suma de las filas:
vectorSensitivityImage = sum(systemMatrixBackrojector, 1)';
% Vector de likelihood:
likelihood = zeros(numIterations,1);
if enablePlot
    % Hago un reshape para visualizarla:
    sensitivityImage = reshape(vectorSensitivityImage, sizeImage);
    figure(200);
    subplot(1,2,1);
    imshow(sensitivityImage./max(max(sensitivityImage)));
    title('Sensitivity Image');
    
    imageSinogram2D = reshape(vectorSinogram, size(sinogram2D));
    subplot(1,2,2);
    imshow(imageSinogram2D./max(max(imageSinogram2D)));
    title('Sinograma de Entrada');
end

% Seteo un umbral de actualización:
updateThreshold = min(vectorSensitivityImage) + (max(vectorSensitivityImage)-min(vectorSensitivityImage)) * 0.0001;
% Itero:
for k = 1 : numIterations
    % Genero una imagenOld, que es la imagen de la iteración anterior para
    % el penalized likelihood:
    imagenOld = reshape(vectorImage, sizeImage);
    % Primero hago la proyección de la imagen actual:
    vectorEstimatedProjection = systemMatrixProjector*vectorImage;
    
    if enablePlot
        % Hago un reshape para visualizarla:
        estimatedProjection = reshape(vectorEstimatedProjection, sizeProjection);
        figure(2001);
        set(gcf, 'Name', sprintf('Procesamiento de Iteración %d', k));
        subplot(1,4,1);
        imshow(estimatedProjection./max(max(estimatedProjection)));
        title(sprintf('Proyección estimada', k));
    end
    likelihood(k) = sum(vectorSinogram.*log(vectorEstimatedProjection)-vectorEstimatedProjection);
    % Genero sinograma de corrección:
    vectorCorrectionProjection= vectorSinogram./vectorEstimatedProjection;
    vectorCorrectionProjection((vectorSinogram==0)&(vectorEstimatedProjection==0)) = 0;
    if enablePlot
        % Hago un reshape para visualizarla:
        correctionProjection = reshape(vectorCorrectionProjection, sizeProjection);
        figure(2001);
        subplot(1,4,2);
        imshow(correctionProjection./max(max(correctionProjection)));
        title(sprintf('Proyección de Corrección', k));
    end
    
    % Ahora genero imagen de corrección haciendo la backprojection:
    vectorCorrectionImage = systemMatrixBackrojector'*vectorCorrectionProjection;
    if enablePlot
        % Hago un reshape para visualizarla:
        correctionImage = reshape(vectorCorrectionImage, sizeImage);
        figure(2001);
        subplot(1,4,3);
        imshow(correctionImage./max(max(correctionImage)));
        title(sprintf('Imagen de Corrección', k));
    end
    
    % Actualizo la imagen haciendo la multiplicación de la imagen por la de
    % corrección y dividiendo por la sensitivity:
    vectorImage(vectorSensitivityImage>=updateThreshold) = vectorImage(vectorSensitivityImage>=updateThreshold) .* vectorCorrectionImage(vectorSensitivityImage>=updateThreshold)./vectorSensitivityImage(vectorSensitivityImage>=updateThreshold);
    vectorImage(vectorSensitivityImage<updateThreshold) = 0;
    % Hago un reshape para la penalización:
    imagen = reshape(vectorImage, sizeImage);
    if enablePlot        
        figure(2001);
        subplot(1,4,4);
        imshow(imagen./max(max(imagen)));
        title(sprintf('Imagen Iteración %d', k));
    end
    

    % Le aplico el filtro:
    if strcmp(strFilter, 'mediana')
        % PAra el prior de mediana, tengo que hacer
        % 1/(1+Beta*(ImagenOld-Mediana)/Mediana:
        beta = param;
        imagenMediana = medfilt2(imagenOld, [3 3]);
        % Hago la penalización:
        imagen(imagenMediana~=0) = imagen(imagenMediana~=0) ./ (1 + beta*(imagenOld(imagenMediana~=0)-imagenMediana(imagenMediana~=0))./imagenMediana(imagenMediana~=0));
    elseif strcmp(strFilter, 'quadratic')
        % La función cuadrática:
        plot(8/(3*sqrt(3))*r.^2./(1+(r./100).^2).^2)
plot(8/(3*sqrt(3))*(r./100)^2./(1+(r./100).^2))
        % Para los bordes le agrego una fila y columna a cada costado y le
        % replico filas y columnas:
        imagenPadded = padarray(imagen, [1 1], 'replicate','both');
        kernelWeightNeighbors = 1/81.*[1/sqrt(2) 1 1/sqrt(2); 1 0 1; 1/sqrt(2) 1 1/sqrt(2)];
        for i = 1 : size(imagen,1)
            for j = 1 : size(imagen,2)
                clusterNeighbors = imagenPadded(i:i+2, j:j+2);
                cluster_r = imagen(i,j) - clusterNeighbors;
                difV_r = zeros(3);
                difV_r(cluster_r~=0) = cluster_r(cluster_r~=0).^2;
                imagen(i,j) = imagen(i,j) ./ (1-beta*sum(sum(difV_r.*kernelWeightNeighbors)));
            end
        end
    elseif strcmp(strFilter, 'geman')
        % Para los bordes le agrego una fila y columna a cada costado y le
        % replico filas y columnas:
        delta = param(2);
        beta = param(1);
        imagenPadded = padarray(imagen, [1 1], 'replicate','both');
        kernelWeightNeighbors = [1/sqrt(2) 1 1/sqrt(2); 1 0 1; 1/sqrt(2) 1 1/sqrt(2)];
        for i = 1 : size(imagen,1)
            for j = 1 : size(imagen,2)
                clusterNeighbors = imagenPadded(i:i+2, j:j+2);
                cluster_r = imagen(i,j) - clusterNeighbors;
                difV_r = zeros(3); 
%                 difV_r = 8/(3*sqrt(3))*(cluster_r./1)^2./(1+(cluster_r./1).^2); 
                difV_r = 16/(3*sqrt(3))*(cluster_r.*delta^2)./(delta.^2+cluster_r.^2).^2; 
                imagen(i,j) = imagen(i,j) ./ (1+beta*sum(sum(difV_r.*kernelWeightNeighbors)));
            end
        end
    elseif strcmp(strFilter, 'gibbs')
        kernelWeightNeighbors = [1/sqrt(2) 1 1/sqrt(2); 1 0 1; 1/sqrt(2) 1 1/sqrt(2)];
        for i = 2 : size(imagen,1)-1
            for j = 2 : size(imagen,2)-1
                clusterNeighbors = imagen(i-1:i+1, j-1:j+1);
                cluster_r = imagen(i,j) - clusterNeighbors;
                difV_r = zeros(3);
                difV_r(cluster_r~=0) = cluster_r(cluster_r~=0)./abs(cluster_r(cluster_r~=0)) .* (1 + alpha.^2 .* (cluster_r(cluster_r~=0)./delta - delta./cluster_r(cluster_r~=0)).^2).^(-0.5);
                imagen(i,j) = imagen(i,j) ./ (1+beta*sum(sum(difV_r.*kernelWeightNeighbors)));
            end
        end
    end
    if enablePlot        
        figure(2001);
        subplot(1,4,4);
        imshow(imagen./max(max(imagen)));
        title(sprintf('Imagen penalizada Iteración %d', k));
    end
    % Vuelvo al vector:
    vectorImage = reshape(imagen, sizeImage(1)*sizeImage(2), 1);
end
imagen = reshape(vectorImage, sizeImage)';