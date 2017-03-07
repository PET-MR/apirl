%**************************************************************************
%
%  Curso An�lisis de Im�genes M�dicas
%
%  Autor: Mart�n Belzunce
%
%  Fecha: 21/12/2010
%
%  Descripcion:
%  Implementaci�n del filtrado por Difusi�n Inhomog�nea
%
%**************************************************************************

function [outputImage, k] = SuavizadoDifusionInhomogenea(inputImage, lambda, alpha, c,...
    stopMethod, stopNumber, integrationMethod, ordenDifFinitas)

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
            disp(sprintf('Iteracion %d', k));
            % Recorro los p�xeles de la imagen:
            for i = (1 + I0) : (size(inputImage,1) - I0)
                for j = (1 + J0) : (size(inputImage,2) - J0)
                    % En inputImage se encuentra la imagen en el ciclo k,
                    % calculamos la imagen para k + 1. Para eso aplico la 
                    % ecuaci�n utilizando el m�todo de forward euler.
                    % 1) Necesito calcular 4 coeficientes D para cada
                    % p�xel: D(i+1/2,j); D(i-1/2,j); D(i, j+1/2);
                    % D(i,j-1/2). Para calcular dichos coeficientes
                    % necesito calcular por interpolaci�n bilineal inputImage(i+1/2,j+1/2);
                    % inputImage(i-1/2,j+1/2); inputImage(i-1/2, j+1/2);
                    % inputImage(i+1/2,j-1/2).
                    % 1.a)
                    % inputImage(i+1/2,j+1/2):
                    i_M_M = 1/4*(inputImage(i,j)+inputImage(i+1,j)+inputImage(i,j+1)+inputImage(i+1,j+1));
                    % inputImage(i+1/2,j-1/2):
                    i_M_m = 1/4*(inputImage(i,j)+inputImage(i+1,j)+inputImage(i,j-1)+inputImage(i+1,j-1));
                    % inputImage(i-1/2,j+1/2):
                    i_m_M = 1/4*(inputImage(i,j)+inputImage(i-1,j)+inputImage(i,j+1)+inputImage(i-1,j+1));
                    % inputImage(i-1/2,j-1/2):
                    i_m_m = 1/4*(inputImage(i,j)+inputImage(i-1,j)+inputImage(i,j-1)+inputImage(i-1,j-1));
                    
                    % 1.b) Calculo las derivadas parciales en cada sentido
                    % para cada punto:
                    % (dinputImage/dx)(i+1/2,j):
                    dI_dx_M_0 = (inputImage(i+1,j) - inputImage(i,j)) ./h;
                    % (dinputImage/dy)(i+1/2,j):
                    dI_dy_M_0 = (i_M_M - i_M_m) ./h;
                    
                    % (dinputImage/dx)(i-1/2,j):
                    dI_dx_m_0 = (inputImage(i,j) - inputImage(i-1,j)) ./h;
                    % (dinputImage/dy)(i-1/2,j):
                    dI_dy_m_0 = (i_m_M - i_m_m) ./h;
                    
                    % (dinputImage/dx)(i,j+1/2):
                    dI_dx_0_M = (i_M_M - i_m_M) ./h;
                    % (dinputImage/dy)(i,j+1/2):
                    dI_dy_0_M = (inputImage(i,j+1) - inputImage(i,j)) ./h;
                    
                    % (dinputImage/dx)(i,j-1/2):
                    dI_dx_0_m = (i_M_m - i_m_m) ./h;
                    % (dinputImage/dy)(i,j-1/2):
                    dI_dy_0_m = (inputImage(i,j) - inputImage(i,j-1)) ./h;
                    
                    %1.c) Finalmente calculo los 4 coeficientes D
                    %calculando el m�dulo de los gradiente en los 4 puntos
                    %de inter�s:
                    ModGrad_M_0 = sqrt(dI_dx_M_0.^2 + dI_dy_M_0.^2);
                    ModGrad_m_0 = sqrt(dI_dx_m_0.^2 + dI_dy_m_0.^2);
                    ModGrad_0_M = sqrt(dI_dx_0_M.^2 + dI_dy_0_M.^2);
                    ModGrad_0_m = sqrt(dI_dx_0_m.^2 + dI_dy_0_m.^2);
                    
                    % 2) Calculo los coeficientes a partir de los m�dulos
                    % de los gradientes y los par�metros para funci�n de
                    % variaci�n del coeficiente de difusi�n:
                    D_M_0 = coefDifusion(ModGrad_M_0, lambda, alpha, c);
                    D_m_0 = coefDifusion(ModGrad_m_0, lambda, alpha, c);
                    D_0_M = coefDifusion(ModGrad_0_M, lambda, alpha, c);
                    D_0_m = coefDifusion(ModGrad_0_m, lambda, alpha, c);

                    % 3) Calculo el valor del p�xel en k+1 a partir de la
                    % imagen k actual y de los 4 coeficientes:
                    switch(ordenDifFinitas)
                        % Aplicamos solo la ecuaci�n de orden 2:
                        case 'Orden2'
                            inputImage_1(i,j) = inputImage(i,j) + (dt /h2) * (D_M_0*(inputImage(i+1,j)-inputImage(i,j))...
                                - D_m_0*(inputImage(i,j)-inputImage(i-1,j)) + D_0_M*(inputImage(i,j+1)-inputImage(i,j))...
                                - D_0_m*(inputImage(i,j)-inputImage(i,j-1)));
                            
                    end
                end
            end
            % Ahora le aplico condiciones de contorno de Neumann, o sea que
            % la derivada en el contorno debe ser igual a cero:
            inputImage_1(1,:) = inputImage_1(2,:);
            inputImage_1(end,:) = inputImage_1(end-1,:);
            inputImage_1(:,1) = inputImage_1(:,2);
            inputImage_1(:,end) = inputImage_1(:, end -1);
            
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
% outputImage = uint16(inputImage_1);
outputImage = inputImage_1;