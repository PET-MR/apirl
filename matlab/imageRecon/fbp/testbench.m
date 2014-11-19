%% TESTBENCH
% Autor: Martín Belzunce
% Última Fecha de Modificación: 26/11/08
%
% Este Testbench genera una serie de sinogramas modelados como un proceso
% poisson puro, y con cantidad de eventos variables. Luego reconstruye cada
% sinograma con distintas variantes del algoritmo Filtred Backprojection.
% Cambiando el tipo de filtro y su frecuencia de corte, y el método de
% inerpolación.
%% GENERACIÓN DE SINOGRAMAS
% Genero una serie de sinogramas con distinta cantidad de eventos, para
% poder analizar el efecto de la estadística en la reconstrucción de
% imágenes.
% Los sinogramas se generan a partir de un fantoma Shepp-Logan.
% Se genera un cell array, con un sinograma en cada elemento del mismo.
p = phantom;    % Fantoma Shepp-Logan
i = 1;          % Indices para las imágenes
for j = 4 : 0.5 : 8
    Sinogramas{i} = SimulacionSino(p ,180 ,5 ,10^j ,'ResEsp');
    figure(i);
    imshow(Sinogramas{i}/max(max(Sinogramas{i})));
    title(sprintf('Sinograma de Phantoma Shepp-Logan con %d eventos', sum(sum(Sinogramas{i}))));
    i = i+1;
end
%% RECONSTRUCCIÓN CON BACKPROJECTION CON INTERPOLACIÓN DE VECINO MÁS PRÓXIMO
% Se reconstruye la imagen para todos los sinogramas creados, utilizando el
% algoritmo de Filtered BackProjection, y utilizando interpolación de
% vecino más próximo. Se aplican todos los filtros disponibles, con
% distintas frecuencias de corte. La imagen utilizada es de 256x256.
% Se compara el algoritmo realizado con el iradon de MATLAB para vreificar
% que la implementación es correcta. A la función iradon no se le puede
% elegir el tamaño de la imagen de salida, por lo que las imágenes a
% comparar serán de distinto tamaño.
close all
tic
for Fc = 0.5 : 0.1 : 1
    for j = 1 : 9
        Imagen = BackProjection(Sinogramas{j}, [256 256], 'Rampa', Fc);
        figure(i);
        imshow(Imagen/max(max(Imagen)));
        title(sprintf('Imagen reconstruída con BackProjection propia. Filtro Rampa. Vecino Más Próximo. Frecuencia de Corte: %d. Cantidad de Eventos: %d', Fc, sum(sum(Sinogramas{j}))));
        i = i + 1;
        Imagen = BackProjection(Sinogramas{j}, [256 256], 'Shepp-Logan', Fc);
        figure(i);
        imshow(Imagen/max(max(Imagen)));
        title(sprintf('Imagen reconstruída con BackProjection propia. Filtro Shepp-Logan. Vecino Más Próximo. Frecuencia de Corte: %d. Cantidad de Eventos: %d', Fc, sum(sum(Sinogramas{j}))));
        i = i + 1;
        Imagen = BackProjection(Sinogramas{j}, [256 256], 'Cosine', Fc);
        figure(i);
        imshow(Imagen/max(max(Imagen)));
        title(sprintf('Imagen reconstruída con BackProjection propia. Filtro Cosine. Vecino Más Próximo. Frecuencia de Corte: %d. Cantidad de Eventos: %d', Fc, sum(sum(Sinogramas{j}))));
        i = i + 1;
        Imagen = BackProjection(Sinogramas{j}, [256 256], 'Hamming', Fc);
        figure(i);
        imshow(Imagen/max(max(Imagen)));
        title(sprintf('Imagen reconstruída con BackProjection propia. Filtro Hamming. Vecino Más Próximo. Frecuencia de Corte: %d. Cantidad de Eventos: %d', Fc, sum(sum(Sinogramas{j}))));
        i = i + 1;
        Imagen = BackProjection(Sinogramas{j}, [256 256], 'Hann', Fc);
        figure(i);
        imshow(Imagen/max(max(Imagen)));
        title(sprintf('Imagen reconstruída con BackProjection propia. Filtro Hann. Vecino Más Próximo. Frecuencia de Corte: %d. Cantidad de Eventos: %d', Fc, sum(sum(Sinogramas{j}))));
    end
end
toc
Tiempo = toc;
disp(sprintf('Tiempo de ejecución de esta sección: %f', Tiempo));
%% RECONSTRUCCIÓN CON BACKPROJECTION CON INTERPOLACIÓN LINEAL
% Se reconstruye la imagen para todos los sinogramas creados, utilizando el
% algoritmo de Filtered BackProjection, y utilizando interpolación lineal.
% Se aplican todos los filtros disponibles, con distintas frecuencias de 
% corte. La imagen utilizada es de 256x256.
% Se compara el algoritmo realizado con el iradon de MATLAB para vreificar
% que la implementación es correcta. A la función iradon no se le puede
% elegir el tamaño de la imagen de salida, por lo que las imágenes a
% comparar serán de distinto tamaño.
close all
tic
for Fc = 0.5 : 0.1 : 1
    for j = 1 : 8
        Imagen = BackProjection(Sinogramas{j}, [256 256], 'Rampa', Fc, 'lineal');
        figure(i);
        imshow(Imagen/max(max(Imagen)));
        title(sprintf('Imagen reconstruída con BackProjection propia. Filtro Rampa. Interpolación lineal. Frecuencia de Corte: %d. Cantidad de Eventos: %d', Fc, sum(sum(Sinogramas{j}))));
        i = i + 1;
        Imagen = BackProjection(Sinogramas{j}, [256 256], 'Shepp-Logan', Fc, 'lineal');
        figure(i);
        imshow(Imagen/max(max(Imagen)));
        title(sprintf('Imagen reconstruída con BackProjection propia. Filtro Shepp-Logan.  Interpolación lineal. Frecuencia de Corte: %d. Cantidad de Eventos: %d', Fc, sum(sum(Sinogramas{j}))));
        i = i + 1;
        Imagen = BackProjection(Sinogramas{j}, [256 256], 'Cosine', Fc, 'lineal');
        figure(i);
        imshow(Imagen/max(max(Imagen)));
        title(sprintf('Imagen reconstruída con BackProjection propia. Filtro Cosine.  Interpolación lineal. Frecuencia de Corte: %d. Cantidad de Eventos: %d', Fc, sum(sum(Sinogramas{j}))));
        i = i + 1;
        Imagen = BackProjection(Sinogramas{j}, [256 256], 'Hamming', Fc, 'lineal');
        figure(i);
        imshow(Imagen/max(max(Imagen)));
        title(sprintf('Imagen reconstruída con BackProjection propia. Filtro Hamming.  Interpolación lineal. Frecuencia de Corte: %d. Cantidad de Eventos: %d', Fc, sum(sum(Sinogramas{j}))));
        i = i + 1;
        Imagen = BackProjection(Sinogramas{j}, [256 256], 'Hann', Fc, 'lineal');
        figure(i);
        imshow(Imagen/max(max(Imagen)));
        title(sprintf('Imagen reconstruída con BackProjection propia. Filtro Hann.  Interpolación lineal. Frecuencia de Corte: %d. Cantidad de Eventos: %d', Fc, sum(sum(Sinogramas{j}))));
    end
end
toc
Tiempo = toc;
disp(sprintf('Tiempo de ejecución de esta sección: %f', Tiempo));
%% RECONSTRUCCIÓN CON BACKPROJECTION CON INTERPOLACIÓN LINEAL
% Se reconstruye la imagen para todos los sinogramas creados, utilizando el
% algoritmo de Filtered BackProjection, y utilizando interpolación lineal.
% Se aplican todos los filtros disponibles, con distintas frecuencias de 
% corte. La imagen utilizada es de 256x256.
% Se compara el algoritmo realizado con el iradon de MATLAB para vreificar
% que la implementación es correcta. A la función iradon no se le puede
% elegir el tamaño de la imagen de salida, por lo que las imágenes a
% comparar serán de distinto tamaño.
close all
tic
for Fc = 0.5 : 0.1 : 1
    for j = 1 : 8
        Imagen = BackProjection(Sinogramas{j}, [256 256], 'Rampa', Fc, 'cubic');
        figure(i);
        imshow(Imagen/max(max(Imagen)));
        title(sprintf('Imagen reconstruída con BackProjection propia. Filtro Rampa.  Interpolación cúbica. Frecuencia de Corte: %d. Cantidad de Eventos: %d', Fc, sum(sum(Sinogramas{j}))));
        i = i + 1;
        Imagen = BackProjection(Sinogramas{j}, [256 256], 'Shepp-Logan', Fc, 'cubic');
        figure(i);
        imshow(Imagen/max(max(Imagen)));
        title(sprintf('Imagen reconstruída con BackProjection propia. Filtro Shepp-Logan.  Interpolación cúbica. Frecuencia de Corte: %d. Cantidad de Eventos: %d', Fc, sum(sum(Sinogramas{j}))));
        i = i + 1;
        Imagen = BackProjection(Sinogramas{j}, [256 256], 'Cosine', Fc, 'cubic');
        figure(i);
        imshow(Imagen/max(max(Imagen)));
        title(sprintf('Imagen reconstruída con BackProjection propia. Filtro Cosine.  Interpolación cúbica. Frecuencia de Corte: %d. Cantidad de Eventos: %d', Fc, sum(sum(Sinogramas{j}))));
        i = i + 1;
        Imagen = BackProjection(Sinogramas{j}, [256 256], 'Hamming', Fc, 'cubic');
        figure(i);
        imshow(Imagen/max(max(Imagen)));
        title(sprintf('Imagen reconstruída con BackProjection propia. Filtro Hamming.  Interpolación cúbica. Frecuencia de Corte: %d. Cantidad de Eventos: %d', Fc, sum(sum(Sinogramas{j}))));
        i = i + 1;
        Imagen = BackProjection(Sinogramas{j}, [256 256], 'Hann', Fc, 'cubic');
        figure(i);
        imshow(Imagen/max(max(Imagen)));
        title(sprintf('Imagen reconstruída con BackProjection propia. Filtro Hann.  Interpolación cúbica. Frecuencia de Corte: %d. Cantidad de Eventos: %d', Fc, sum(sum(Sinogramas{j}))));
    end
end
toc
Tiempo = toc;
disp(sprintf('Tiempo de ejecución de esta sección: %f', Tiempo));