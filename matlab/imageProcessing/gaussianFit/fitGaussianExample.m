%**************************************************************************
%
%  Autor: Mart�n Belzunce
%
%  Fecha: 03/11/2010
%
%  Descripcion:
%  Se realiza el ajuste de la curva del histograma de dos im�genes de T1 de
%  cerebro como la suma de 3 gaussianas que representan las intensidades de
%  T1 para cada uno de los 3 tipos de tejidos (Materia blanca, Materia
%  gris y l�quido cefalo raqu�deo).
%
%**************************************************************************

clc
clear all;
close all;

%% PAR�METROS DE ENTRADA
% Nombre de las dos im�genes Dicom a utilizar:
filenameDicoms = {'T1_corr.dcm', 'T1_ncorr.dcm'};    % Path relativo y nombre de los dos archivos.
% Bines para el histograma:
nBines = 100;
% Canales del Histograma:
canalesHist = 0 : 5000/(nBines) : 5000;
%% LECTURA DE IMAGEN
% Obtenemos informaci�n del header del archivo Dicom de la imagen con
% inhomogeinedad corregida, la misma es guardada en una estrctura:
structInfo_Corr = dicominfo(filenameDicoms{1});
% Leemos la imagen:
I_Corr = dicomread(filenameDicoms{1});
% Hacemos lo mismo con la no crregida:
structInfo_nCorr = dicominfo(filenameDicoms{2});
I_nCorr = dicomread(filenameDicoms{2});

%% C�LCULO DE LOS HISTOGRAMAS
% Calculamos los histogramas de las dos im�genes:
% [cuentas_Corr, canales_Corr] = hist(double(I_Corr(:)), nBines);
% [cuentas_nCorr, canales_nCorr] = hist(double(I_nCorr(:)), nBines);
[cuentas_Corr, canales_Corr] = hist(double(I_Corr(:)), canalesHist);
[cuentas_nCorr, canales_nCorr] = hist(double(I_nCorr(:)), canalesHist);
% Elimino los canales del fondo(el primero):
cuentas_Corr(1) = [];
cuentas_nCorr(1) = [];
canales_Corr(1) = [];
canales_nCorr(1) = [];
%% VISUALIZACI�N DE LAS IM�GENES
% Para la visualizaci�n de las im�genes las convierto en double, y las
% normaliza para mejorar la visualizaci�n.
figure;
set(gcf, 'Position', [50 50 1300 750]);
subplot(2,1,1);
imshow(double(I_Corr)./double(max(max(I_Corr))));
title('Imagen T1 con Inhomogeneidad Corregida');
subplot(2,1,2);
bar(canales_Corr, cuentas_Corr);
title('Histograma con los valores reales de la imagen T1');

figure;
set(gcf, 'Position', [50 50 1300 750]);
subplot(2,1,1);
imshow(double(I_nCorr)./double(max(max(I_nCorr))));
title('Imagen T1 con Inhomogeneidad no Corregida');
subplot(2,1,2);
bar(canales_nCorr, cuentas_nCorr);
title('Histograma con los valores reales de la imagen T1');

%% AJUSTE DE HISTOGRAMAS A 3 GAUSSIANAS
% Valores iniciales de las 9 variables a estimar, que recordamos son la
% med�a, el desv�o y el factor de ganancia de cada campana de gauss:
X0_Corr = [7500 1100 100 2600 1400 200 800 1900 200];
X0_nCorr = [7500 900 100 2800 1300 250 500 1800 250];

% Opciones para LSQNONLIN:
options = optimset('Largescale','off');

% Calculamos los coeficientes del ajuste para cada uno de los histogramas.
% Lo hacemos mediante dos funciones distintas pero el proceso es
% exactamente el mismo, en un caso usamos lsqnonlin que permite resolver
% cualquier problema por lsq no lineal, a esta funci�n se le pasa el
% puntero a una funci�n que calcula el error entre el modelo propuesto y
% los datos reales. En el otro caso usamos lsqcurvefit que es una funci�n
% que realiza lo mismo que lsqnonlin pero orientada a fiteo de curvas, la
% ventaja de esta es que la funci�n que hay que pasarle como par�metro es
% simplemente la funci�n modelo propuesta (y no el error).
%coef_Corr = lsqnonlin(@diff_3Gaussian,X0_Corr,[],[],options,canales_Corr,cuentas_Corr);
coef_Corr = lsqcurvefit(@Gaussian3,X0_Corr,canales_Corr,cuentas_Corr);
coef_nCorr = lsqnonlin(@diff_3Gaussian,X0_nCorr,[],[],options,canales_nCorr,cuentas_nCorr);

% Ahora genero la curva formada por la suma de las 3 gaussianas, a partir
% de los coeficientes estimados:
Yfit1_Corr = coef_Corr(1).*exp(-0.5.*(canales_Corr-coef_Corr(2)).^2/coef_Corr(3)^2);
Yfit2_Corr = coef_Corr(4).*exp(-0.5.*(canales_Corr-coef_Corr(5)).^2/coef_Corr(6)^2);
Yfit3_Corr = coef_Corr(7).*exp(-0.5.*(canales_Corr-coef_Corr(8)).^2/coef_Corr(9)^2);
Yfit_Corr = Yfit1_Corr + Yfit2_Corr + Yfit3_Corr;

Yfit1_nCorr = coef_nCorr(1).*exp(-0.5.*(canales_nCorr-coef_nCorr(2)).^2/coef_nCorr(3)^2);
Yfit2_nCorr = coef_nCorr(4).*exp(-0.5.*(canales_nCorr-coef_nCorr(5)).^2/coef_nCorr(6)^2);
Yfit3_nCorr = coef_nCorr(7).*exp(-0.5.*(canales_nCorr-coef_nCorr(8)).^2/coef_nCorr(9)^2);
Yfit_nCorr = Yfit1_nCorr + Yfit2_nCorr + Yfit3_nCorr;

%% VISUALIZACI�N DE LOS RESULTADOS
% Visualizamos el resultado del ajuste graficando las 3 gaussianas por
% seprado, la suma de ellas, y el histograma real.
% Graficamos el primer ajuste:
figure;
set(gcf, 'Position', [50 50 1300 750]);
plot(canales_Corr, Yfit1_Corr, 'r', canales_Corr, Yfit2_Corr, 'g', canales_Corr, Yfit3_Corr, 'b',...
    canales_Corr, Yfit_Corr, 'ok-',canales_Corr, cuentas_Corr, '*m-');
title('Ajuste de histograma de imagen T1 con inhomogeinedad corregida a 3 gaussianas')
legend('Gaussiana MB', 'Gaussiana MG', 'Gaussiana LCR', 'Suma de las 3 Gaussianas', 'Histograma Real');

figure;
set(gcf, 'Position', [50 50 1300 750]);
plot(canales_nCorr, Yfit1_nCorr, 'r', canales_nCorr, Yfit2_nCorr, 'g', canales_nCorr, Yfit3_nCorr, 'b',...
    canales_nCorr, Yfit_nCorr, 'ok-',canales_nCorr, cuentas_nCorr, '*m-');
title('Ajuste de histograma de imagen T1 con inhomogeinedad no corregida a 3 gaussianas')
legend('Gaussiana MB', 'Gaussiana MG', 'Gaussiana LCR', 'Suma de las 3 Gaussianas', 'Histograma Real');

% Imprimo en consola los resultados que me interesan, en este caso la media
% de la campana para cada tejido:
disp('##################### RESULTADOS PARA IMAGEN CON INHOMOGEINEDAD CORREGIDA #####################');
disp(sprintf('Valor medio de T1 para materia blanca en imagen con inhomogeinedad corregida: %f mseg', coef_Corr(2)));
disp(sprintf('Valor medio de T1 para materia gris en imagen con inhomogeinedad corregida: %f mseg', coef_Corr(5)));
disp(sprintf('Valor medio de T1 para l�quido c�falo raqu�deo en imagen con inhomogeinedad corregida: %f mseg', coef_Corr(8)));
disp('###############################################################################################');
disp('\n');
disp('##################### RESULTADOS PARA IMAGEN CON INHOMOGEINEDAD NO CORREGIDA #####################');
disp(sprintf('Valor medio de T1 para materia blanca en imagen con inhomogeinedad no corregida: %f mseg', coef_nCorr(2)));
disp(sprintf('Valor medio de T1 para materia gris en imagen con inhomogeinedad no corregida: %f mseg', coef_nCorr(5)));
disp(sprintf('Valor medio de T1 para l�quido c�falo raqu�deo en imagen con inhomogeinedad no corregida: %f mseg', coef_nCorr(8)));
disp('###############################################################################################');