%  *********************************************************************
%  Proyecto AR-PET. Comisión Nacional de Energía Atómica.
%  Autor: Martín Belzunce. UTN-FRBA.
%  Fecha de Creación: 29/06/2012
%  *********************************************************************
%

% La SystemMatrix tiene en las filas las lors, y en las columnas los
% píxels:

function [imagen, likelihood] = Mlem2D(sinogram2D, systemMatrixProjector, systemMatrixBackrojector, sizeImage, numIterations, enablePlot)

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
    %vectorImage(vectorSensitivityImage<updateThreshold) = 0;
    if enablePlot
        % Hago un reshape para visualizarla:
        imagen = reshape(vectorImage, sizeImage);
        figure(2001);
        subplot(1,4,4);
        imshow(imagen./max(max(imagen)));
        title(sprintf('Imagen Iteración %d', k));
    end
    
end
imagen = reshape(vectorImage, sizeImage)';