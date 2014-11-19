%% TESTBENCH
% Autor: Mart�n Belzunce
% �ltima Fecha de Modificaci�n: 26/11/08
%
% Este Testbench genera una serie de sinogramas modelados como un proceso
% poisson puro, y con cantidad de eventos variables. Luego reconstruye cada
% sinograma con distintas variantes del algoritmo Filtred Backprojection.
% Cambiando el tipo de filtro y su frecuencia de corte, y el m�todo de
% inerpolaci�n.
%% GENERACI�N DE SINOGRAMAS
% Genero una serie de sinogramas con distinta cantidad de eventos, para
% poder analizar el efecto de la estad�stica en la reconstrucci�n de
% im�genes.
% Los sinogramas se generan a partir de un fantoma Shepp-Logan.
% Se genera un cell array, con un sinograma en cada elemento del mismo.
p = phantom;    % Fantoma Shepp-Logan
i = 1;          % Indices para las im�genes
for j = 4 : 0.5 : 8
    Sinogramas{i} = SimulacionSino(p ,180 ,5 ,10^j ,'ResEsp');
    figure(i);
    imshow(Sinogramas{i}/max(max(Sinogramas{i})));
    title(sprintf('Sinograma de Phantoma Shepp-Logan con %d eventos', sum(sum(Sinogramas{i}))));
    i = i+1;
end
%% RECONSTRUCCI�N CON BACKPROJECTION CON INTERPOLACI�N DE VECINO M�S PR�XIMO
% Se reconstruye la imagen para todos los sinogramas creados, utilizando el
% algoritmo de Filtered BackProjection, y utilizando interpolaci�n de
% vecino m�s pr�ximo. Se aplican todos los filtros disponibles, con
% distintas frecuencias de corte. La imagen utilizada es de 256x256.
% Se compara el algoritmo realizado con el iradon de MATLAB para vreificar
% que la implementaci�n es correcta. A la funci�n iradon no se le puede
% elegir el tama�o de la imagen de salida, por lo que las im�genes a
% comparar ser�n de distinto tama�o.
close all
tic
for Fc = 0.5 : 0.1 : 1
    for j = 1 : 9
        Imagen = BackProjection(Sinogramas{j}, [256 256], 'Rampa', Fc);
        figure(i);
        imshow(Imagen/max(max(Imagen)));
        title(sprintf('Imagen reconstru�da con BackProjection propia. Filtro Rampa. Vecino M�s Pr�ximo. Frecuencia de Corte: %d. Cantidad de Eventos: %d', Fc, sum(sum(Sinogramas{j}))));
        i = i + 1;
        Imagen = BackProjection(Sinogramas{j}, [256 256], 'Shepp-Logan', Fc);
        figure(i);
        imshow(Imagen/max(max(Imagen)));
        title(sprintf('Imagen reconstru�da con BackProjection propia. Filtro Shepp-Logan. Vecino M�s Pr�ximo. Frecuencia de Corte: %d. Cantidad de Eventos: %d', Fc, sum(sum(Sinogramas{j}))));
        i = i + 1;
        Imagen = BackProjection(Sinogramas{j}, [256 256], 'Cosine', Fc);
        figure(i);
        imshow(Imagen/max(max(Imagen)));
        title(sprintf('Imagen reconstru�da con BackProjection propia. Filtro Cosine. Vecino M�s Pr�ximo. Frecuencia de Corte: %d. Cantidad de Eventos: %d', Fc, sum(sum(Sinogramas{j}))));
        i = i + 1;
        Imagen = BackProjection(Sinogramas{j}, [256 256], 'Hamming', Fc);
        figure(i);
        imshow(Imagen/max(max(Imagen)));
        title(sprintf('Imagen reconstru�da con BackProjection propia. Filtro Hamming. Vecino M�s Pr�ximo. Frecuencia de Corte: %d. Cantidad de Eventos: %d', Fc, sum(sum(Sinogramas{j}))));
        i = i + 1;
        Imagen = BackProjection(Sinogramas{j}, [256 256], 'Hann', Fc);
        figure(i);
        imshow(Imagen/max(max(Imagen)));
        title(sprintf('Imagen reconstru�da con BackProjection propia. Filtro Hann. Vecino M�s Pr�ximo. Frecuencia de Corte: %d. Cantidad de Eventos: %d', Fc, sum(sum(Sinogramas{j}))));
    end
end
toc
Tiempo = toc;
disp(sprintf('Tiempo de ejecuci�n de esta secci�n: %f', Tiempo));
%% RECONSTRUCCI�N CON BACKPROJECTION CON INTERPOLACI�N LINEAL
% Se reconstruye la imagen para todos los sinogramas creados, utilizando el
% algoritmo de Filtered BackProjection, y utilizando interpolaci�n lineal.
% Se aplican todos los filtros disponibles, con distintas frecuencias de 
% corte. La imagen utilizada es de 256x256.
% Se compara el algoritmo realizado con el iradon de MATLAB para vreificar
% que la implementaci�n es correcta. A la funci�n iradon no se le puede
% elegir el tama�o de la imagen de salida, por lo que las im�genes a
% comparar ser�n de distinto tama�o.
close all
tic
for Fc = 0.5 : 0.1 : 1
    for j = 1 : 8
        Imagen = BackProjection(Sinogramas{j}, [256 256], 'Rampa', Fc, 'lineal');
        figure(i);
        imshow(Imagen/max(max(Imagen)));
        title(sprintf('Imagen reconstru�da con BackProjection propia. Filtro Rampa. Interpolaci�n lineal. Frecuencia de Corte: %d. Cantidad de Eventos: %d', Fc, sum(sum(Sinogramas{j}))));
        i = i + 1;
        Imagen = BackProjection(Sinogramas{j}, [256 256], 'Shepp-Logan', Fc, 'lineal');
        figure(i);
        imshow(Imagen/max(max(Imagen)));
        title(sprintf('Imagen reconstru�da con BackProjection propia. Filtro Shepp-Logan.  Interpolaci�n lineal. Frecuencia de Corte: %d. Cantidad de Eventos: %d', Fc, sum(sum(Sinogramas{j}))));
        i = i + 1;
        Imagen = BackProjection(Sinogramas{j}, [256 256], 'Cosine', Fc, 'lineal');
        figure(i);
        imshow(Imagen/max(max(Imagen)));
        title(sprintf('Imagen reconstru�da con BackProjection propia. Filtro Cosine.  Interpolaci�n lineal. Frecuencia de Corte: %d. Cantidad de Eventos: %d', Fc, sum(sum(Sinogramas{j}))));
        i = i + 1;
        Imagen = BackProjection(Sinogramas{j}, [256 256], 'Hamming', Fc, 'lineal');
        figure(i);
        imshow(Imagen/max(max(Imagen)));
        title(sprintf('Imagen reconstru�da con BackProjection propia. Filtro Hamming.  Interpolaci�n lineal. Frecuencia de Corte: %d. Cantidad de Eventos: %d', Fc, sum(sum(Sinogramas{j}))));
        i = i + 1;
        Imagen = BackProjection(Sinogramas{j}, [256 256], 'Hann', Fc, 'lineal');
        figure(i);
        imshow(Imagen/max(max(Imagen)));
        title(sprintf('Imagen reconstru�da con BackProjection propia. Filtro Hann.  Interpolaci�n lineal. Frecuencia de Corte: %d. Cantidad de Eventos: %d', Fc, sum(sum(Sinogramas{j}))));
    end
end
toc
Tiempo = toc;
disp(sprintf('Tiempo de ejecuci�n de esta secci�n: %f', Tiempo));
%% RECONSTRUCCI�N CON BACKPROJECTION CON INTERPOLACI�N LINEAL
% Se reconstruye la imagen para todos los sinogramas creados, utilizando el
% algoritmo de Filtered BackProjection, y utilizando interpolaci�n lineal.
% Se aplican todos los filtros disponibles, con distintas frecuencias de 
% corte. La imagen utilizada es de 256x256.
% Se compara el algoritmo realizado con el iradon de MATLAB para vreificar
% que la implementaci�n es correcta. A la funci�n iradon no se le puede
% elegir el tama�o de la imagen de salida, por lo que las im�genes a
% comparar ser�n de distinto tama�o.
close all
tic
for Fc = 0.5 : 0.1 : 1
    for j = 1 : 8
        Imagen = BackProjection(Sinogramas{j}, [256 256], 'Rampa', Fc, 'cubic');
        figure(i);
        imshow(Imagen/max(max(Imagen)));
        title(sprintf('Imagen reconstru�da con BackProjection propia. Filtro Rampa.  Interpolaci�n c�bica. Frecuencia de Corte: %d. Cantidad de Eventos: %d', Fc, sum(sum(Sinogramas{j}))));
        i = i + 1;
        Imagen = BackProjection(Sinogramas{j}, [256 256], 'Shepp-Logan', Fc, 'cubic');
        figure(i);
        imshow(Imagen/max(max(Imagen)));
        title(sprintf('Imagen reconstru�da con BackProjection propia. Filtro Shepp-Logan.  Interpolaci�n c�bica. Frecuencia de Corte: %d. Cantidad de Eventos: %d', Fc, sum(sum(Sinogramas{j}))));
        i = i + 1;
        Imagen = BackProjection(Sinogramas{j}, [256 256], 'Cosine', Fc, 'cubic');
        figure(i);
        imshow(Imagen/max(max(Imagen)));
        title(sprintf('Imagen reconstru�da con BackProjection propia. Filtro Cosine.  Interpolaci�n c�bica. Frecuencia de Corte: %d. Cantidad de Eventos: %d', Fc, sum(sum(Sinogramas{j}))));
        i = i + 1;
        Imagen = BackProjection(Sinogramas{j}, [256 256], 'Hamming', Fc, 'cubic');
        figure(i);
        imshow(Imagen/max(max(Imagen)));
        title(sprintf('Imagen reconstru�da con BackProjection propia. Filtro Hamming.  Interpolaci�n c�bica. Frecuencia de Corte: %d. Cantidad de Eventos: %d', Fc, sum(sum(Sinogramas{j}))));
        i = i + 1;
        Imagen = BackProjection(Sinogramas{j}, [256 256], 'Hann', Fc, 'cubic');
        figure(i);
        imshow(Imagen/max(max(Imagen)));
        title(sprintf('Imagen reconstru�da con BackProjection propia. Filtro Hann.  Interpolaci�n c�bica. Frecuencia de Corte: %d. Cantidad de Eventos: %d', Fc, sum(sum(Sinogramas{j}))));
    end
end
toc
Tiempo = toc;
disp(sprintf('Tiempo de ejecuci�n de esta secci�n: %f', Tiempo));