% Función que procesa las imágenes para evaluar la performance de la
% reconstrucción y armar gráficos comaprativos:
function [contrastRecovery, stdBackground, stdNormBackground, meanLungRoi, relativeLungError] = procNemaPhantomWithMasks(volume, centralSlice, maskSpheres, maskBackground, indicesHotSpheres, relacionHotBackground)
% maskBackground is a 2d cell array with the mask for the background, first
% dimension sizew, second position
for i = 1 : numel(maskSpheres)
    % Slice to process
    sliceToProcess = volume(:,:,centralSlice);
    % Get the mean value in the sphere:
    meanHotSpheres(i) = mean(mean(sliceToProcess(maskSpheres{i})));    
    
    % Get the mean background for this sphere size, that is compute for
    % ROIs of the same size in 5 slices centred in the one used for the hot
    % spheres:
    for j = 1 : size(maskBackground,2)
        for k = 1:5   %Voy desde clice central-2 a slice central +2:
            sliceToProcess = volume(:,:,centralSlice-3+k);
            meanBackgroundRoi(i,j,k) = mean(sliceToProcess(maskBackground{i,j}));
            stdBackgroundRoi(i,j,k) = std(sliceToProcess(maskBackground{i,j}));
        end
    end
    meanBackground(i) = mean(meanBackgroundRoi(:));
    % The last sphere is the lung insert:
    if i <= numel(indicesHotSpheres)
        if(indicesHotSpheres(i))
            contrastRecovery(i) = (meanHotSpheres(i)/meanBackground(i)-1)/(relacionHotBackground-1)*100;
        else
            contrastRecovery(i) = (1-(meanHotSpheres(i) ./ meanBackground(i))) *100;
        end
        %  Calculo el desvío en zona uniforme, que se obtiene calculando el desvio de las cuentas promedio de las 60 esferas:
        stdBackground(i) = std(meanBackgroundRoi(i,:));
        % El desvío normalizado:
        stdNormBackground(i) = stdBackground(i) ./ meanBackground(i);
    else
        % For the 5 slices I need to do this:
        for k = 1:5
            sliceToProcess = volume(:,:,centralSlice-3+k);
            meanLungRoi(k) = mean(mean(sliceToProcess(maskSpheres{i})));
            % Se hace el cociente con las 12ROIs de fondo de el slice:
            relativeLungError(k) = meanLungRoi(k) ./ mean(meanBackgroundRoi(i,:,k));
        end
    end

end


