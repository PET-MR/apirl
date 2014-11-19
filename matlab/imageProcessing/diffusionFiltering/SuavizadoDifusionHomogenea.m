%**************************************************************************
%
%  Curso An�lisis de Im�genes M�dicas
%
%  Autor: Mart�n Belzunce
%
%  Fecha: 21/12/2010
%
%  Descripcion:
%  Implementaci�n del filtrado por Difusi�n Homog�nea
%
%**************************************************************************

function [outputImage, k] = SuavizadoDifusionHomogenea(inputImage, D, stopMethod, stopNumber, integrationMethod, ordenDifFinitas)

% Casteo la imagen a oduble para poder manipular y procesar la imagen sin
% limitaciones por el tipo de dato.
inputImage = double(inputImage);

outputImage = zeros(size(inputImage));
inputImage_1 = zeros(size(inputImage));

dt = 0.001;
h = 1;
h2 = h.^2;

% Seg�n el orden de las diferencias centrales para el discretizado
% espacial, ser� el ancho del borde de la imagen en la que no se puede
% aplicar el suavizado. Para orden 2 un p�xel desde el borde no puede
% aplizarse el suavizado, para orden 4 son dos los p�xeles del borde que no
% deben ser tenidos en cuenta:
switch(ordenDifFinitas)
    case 'Orden2'
        I0 = 1;
        J0 = 1;
    case 'Orden4'
        I0 = 2;
        J0 = 2;
end
switch integrationMethod
    case 'ForwardEuler'
        % Voy iterando hasta que el error cuadr�tico entre im�genes
        % consecutivas sea menor a la tolerancia con que fue llamada la
        % funci�n:
        error = stopNumber + 1; % A la variable error le asigno un valor iniciaal que me asegure que entre al loop.
        k = 0;
        while((strcmp(stopMethod, 'ErrorRelativo')&&(error>stopNumber))||(strcmp(stopMethod, 'Iteraciones')&&(k<stopNumber)))
            % Recorro los p�xeles de la imagen:
            for i = (1 + I0) : (size(inputImage,1) - I0)
                for j = (1 + J0) : (size(inputImage,2) - J0)
                    % En inputImage se encuentra la imagen en el ciclo k,
                    % calculamos la imagen para k + 1. Para eso aplico la 
                    % ecuaci�n utilizando el m�todo de forward euler: 
                    switch(ordenDifFinitas)
                        case 'Orden2'
                            inputImage_1(i,j) = inputImage(i,j) + (D * dt /h2) * (inputImage(i+1,j) + inputImage(i-1,j) +...
                                inputImage(i,j+1) + inputImage(i,j-1) - 4 * inputImage(i,j));
                        case 'Orden4'
                            inputImage_1(i,j) = inputImage(i,j) + (D * dt / (12*h2)) * (-inputImage(i+2,j) + 16*inputImage(i+1,j) +...
                                16 * inputImage(i-1,j) - inputImage(i-2,j) - inputImage(i,j+2) + 16*inputImage(i,j+1) + ...
                                16*inputImage(i,j-1) - inputImage(i,j-2) - 60 * inputImage(i,j));
                    end
                end
            end
            % Ahora le aplico condiciones de contorno de Neumann, o sea que
            % la derivada en el contorno debe ser igual a cero:
            switch(ordenDifFinitas)
                case 'Orden2'
                    inputImage_1(1,:) = inputImage_1(2,:);
                    inputImage_1(end,:) = inputImage_1(end-1,:);
                    inputImage_1(:,1) = inputImage_1(:,2);
                    inputImage_1(:,end) = inputImage_1(:, end -1);
            
                case 'Orden4'
                    % Si es orden 4 el contorno es doble:
                    inputImage_1([1 2],:) = inputImage_1([3 3],:);
                    inputImage_1([end end-1],:) = inputImage_1([end-2 end-2],:);
                    inputImage_1(:,[1 2]) = inputImage_1(:,[3 3]);
                    inputImage_1(:,[end end-1]) = inputImage_1(:, [end-2 end-2]);
            end
            % Calculo el error entre las dos im�genes, como el promedio de
            % los errores cuadr�ticos entre los p�xeles de la iteraci�n k+1
            % y los de la iteraci�n k.         
            % El error que utilizo es el relativo, por lo que los valores
            % que son igual a cero (que me generar�a inf o NaN) no los
            % tengo en cuenta.
            pixelsCero = (inputImage ~= 0);
            error = sqrt(sum(((inputImage_1(pixelsCero)-inputImage(pixelsCero))./inputImage(pixelsCero)).^2))./numel(inputImage(pixelsCero));
            % Ahora la imagen k+1 pasa a ser la imagen k:
            inputImage = inputImage_1;
            k = k+1;
        end
    case 'Runge-Kutta'
        
end
outputImage = uint16(inputImage_1);