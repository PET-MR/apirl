%% TESTBENCH
% Autor: Martín Belzunce
% Última Fecha de Modificación: 26/11/08
%
%
% Este Testbench genera una serie de sinogramas modelados como un proceso
% poisson puro, y con cantidad de eventos variables. Luego reconstruye cada
% sinograma con el algoritmo ML-EM.
% Se muestran todas las imágenes generadas en cada iteración de la
% reconstrucción, y una curva de evolución del ML.
%% GENERACIÓN DE SINOGRAMAS
% Genero una serie de sinogramas con distinta cantidad de eventos, para
% poder analizar el efecto de la estadística en la reconstrucción de
% imágenes.
% Los sinogramas se generan a partir de un fantoma Shepp-Logan.
% Se genera un cell array, con un sinograma en cada elemento del mismo.
p = phantom;    % Fantoma Shepp-Logan
i = 1;          % Indices para las imágenes
NProy = 180;
NR = 213;
for j = 4 : 0.5 : 8
    Sinogramas{i} = SimulacionSino(p ,NProy ,NR ,10^j ,'CantR');
    figure(i);
    imshow(Sinogramas{i}/max(max(Sinogramas{i})));
    title(sprintf('Sinograma de Phantoma Shepp-Logan con %d eventos', sum(sum(Sinogramas{i}))));
    i = i+1;
end
NombreSinos = sprintf('Sinogramas%dx%d', NProy, NR);    % Guardo la matriz ya normalizada.
save(NombreSinos, 'Sinogramas');
CantSinos = i-1;
%%
load('Aij180x213_128x128.mat');
for k = 1 : size(Sinogramas,2)
    disp('######################################################');
    disp(sprintf('Reconstrucción de Sinograma con %d eventos.', sum(sum(Sinogramas{k}))));
    [Imagen, L, Xi] = MLEM(Sinogramas{k}, Aij, 20);
    for j = 1: size(Xi,1)
        disp(sprintf('Imagen reconstruída con MLEM. Nº de Iteración %d :', j-1)); 
        figure(i);
        imshow(Xi{j}/max(max(Xi{j})));
        title(sprintf('Imagen reconstruída con MLEM. Nº de Iteración: %d', j-1));
        i = i + 1;
        figure(i);
    end
    disp(sprintf('Curva de evolución de ML:'));
    plot(L);
    title(sprintf('Curva de evolución de ML'));
    i = i + 1;
end
