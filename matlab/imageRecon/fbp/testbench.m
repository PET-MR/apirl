%% TESTBENCH
% Autor: Mart�n Belzunce
% �ltima Fecha de Modificaci�n: 26/11/08
%
% Este Testbench genera una serie de sinogramas modelados como un proceso
% poisson puro, y con cantidad de eventos variables. Luego reconstruye cada
% sinograma con distintas variantes del algoritmo Filtred Backprojection.
% Cambiando el tipo de filtro y su frecuencia de corte, y el m�todo de
% inerpolaci�n.
clear all
close all
addpath('/sources/MATLAB/WorkingCopy/ImageRecon')
addpath('/sources/MATLAB/WorkingCopy/ImageProcessing')
%% GENERACIÓN DE SINOGRAMAS CON DISTINTA CANTIDAD DE PROYECCIONES
p = phantom;
h = figure;
imshow(p);
outputGraphsPath = '/workspaces/Martin/Doctorado/Tesis/Tesis Martín Belzunce/docusTesis/Figuras/Capitulo3/backprojection/';
graphicFilename = sprintf('fantomaSheppLogan');
set(gcf,'PaperPositionMode','auto');    % Para que lo guarde en el tamaño modificado.
frame = getframe(gca);
imwrite(frame.cdata, [outputGraphsPath graphicFilename '.png']);
saveas(gca, [outputGraphsPath graphicFilename], 'epsc');
%% PROYECCIÓN Y RETROPROYECCIÓN CON FILTRO Y SIN FILTRO
sinograma = radon(p,0.5:1:179.5);
imagenSinRampa = iradon(sinograma, 0.5:1:179.5, 'linear', 'none', 1, 256);
imagenConRampa = iradon(sinograma, 0.5:1:179.5, 'linear', 'Ram-Lak',1, 256);
figure;
imagen = [imagenSinRampa./max(max(imagenSinRampa)) imagenConRampa]./max(max(imagenConRampa));
imshow(imagen)
line([size(imagenSinRampa,2) size(imagenSinRampa,2)],[0 size(imagen,1)],'color','w','LineWidth', 2)
set(gcf, 'Position', [50 50 1600 1200]);
set(gcf, 'InvertHardcopy', 'off')
% Agrego una leyenda en cada imagen para identificarlas:
text(15 , 15, 'Sin Filtro','Color','w','FontSize',16,'FontWeight','bold')
text( 15 + size(imagenSinRampa,2), 15, 'Filtro Rampa','Color','w','FontSize',16,'FontWeight','bold')
outputGraphsPath = '/workspaces/Martin/Doctorado/Tesis/Tesis Martín Belzunce/docusTesis/Figuras/Capitulo3/backprojection/';
graphicFilename = sprintf('backprojectionConSinRampa');
set(gcf,'PaperPositionMode','auto');    % Para que lo guarde en el tamaño modificado.
saveas(gcf, [outputGraphsPath graphicFilename], 'fig');
frame = getframe(gca);
imwrite(frame.cdata, [outputGraphsPath graphicFilename '.png']);
saveas(gca, [outputGraphsPath graphicFilename], 'epsc');
%% PROYECCIÓN Y RETROPROYECCIÓN CON ANGULOS LIMITIDOS
close all
angulos = [180 90 60 30 15 8];
imagen = [];
for i = 1 : numel(angulos)
    % sinogramasConAngulos{i} = Projection(p,angulos(i),angulos(i));
    sinogramasConAngulos{i} = radon(p,180/angulos(i):180/angulos(i):(180-180/angulos(i)));
    h = figure
    %imagen{i} = BackProjection(sinogramasConAngulos{i}, [size(sinogramasConAngulos{i},2) size(sinogramasConAngulos{i},2)], 'Rampa', 1, 'lineal')
    imagenes{i} = iradon(sinogramasConAngulos{i}, 180/angulos(i):180/angulos(i):(180-180/angulos(i)), 'linear', 'Ram-Lak', 1, 256);
    imshow(imagenes{i} ./max(max(imagenes{i})));
end

i = 1; fila = 1; col = 1;
for i = 1 : numel(imagenes)
    col = rem(i,2);
    if col == 0
        col = 2;
    end
    fila = ceil(i / 2);
    imagen(((fila-1)*size(imagenes{i},1)+1) : (fila*size(imagenes{i},1)), ((col-1)*size(imagenes{i},2)+1) : (col*size(imagenes{i},2))) = (imagenes{i}./max(max(imagenes{i})));    
end
h = figure
imshow(imagen ./max(max(imagen)));

line([size(imagenes{1},2) size(imagenes{1},2)],[0 size(imagenes{1},1)*3],'color','w','LineWidth', 2)
line([0 size(imagenes{1},2)*2 ],[size(imagenes{1},1) size(imagenes{1},1)],'color','w','LineWidth', 2)
line([0 size(imagenes{1},2)*2 ],[size(imagenes{1},1)*2 size(imagenes{1},1)*2],'color','w','LineWidth', 2)
set(gcf, 'Position', [50 50 1600 1200]);
set(gcf, 'InvertHardcopy', 'off')
% Agrego una leyenda en cada imagen para identificarlas:
for i = 1 : numel(imagenes)
    col = rem(i,2);
    if col == 0
        col = 2;
    end
    fila = ceil(i / 2);
    text(15 + size(imagenes{1},2) * (col-1), 15 + size(imagenes{1},1) * (fila-1) , sprintf('%d Proy',angulos(i)) ,'Color','w','FontSize',9.5,'FontWeight','bold')
end
outputGraphsPath = '/workspaces/Martin/Doctorado/Tesis/Tesis Martín Belzunce/docusTesis/Figuras/Capitulo3/backprojection/';
graphicFilename = sprintf('backprojectionCantProy');
set(gcf,'PaperPositionMode','auto');    % Para que lo guarde en el tamaño modificado.
saveas(gcf, [outputGraphsPath graphicFilename], 'fig');
frame = getframe(gca);
imwrite(frame.cdata, [outputGraphsPath graphicFilename '.png']);
saveas(gca, [outputGraphsPath graphicFilename], 'epsc');
%% GENERACI�N DE SINOGRAMAS CON RUIDO ESTADÍSTICO
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
% Visualización de Sinogramas para tesis
h = figure;
imageSinograms = zeros(size(Sinogramas{1}).*[4 2]);
i = 1; fila = 1; col = 1;
for i = 1 : numel(Sinogramas)-1
    %Sinogramas{i} = Sinogramas{i}';
    col = rem(i,2);
    if col == 0
        col = 2;
    end
    fila = ceil(i / 2);
    imageSinograms(((fila-1)*size(Sinogramas{i},1)+1) : (fila*size(Sinogramas{i},1)), ((col-1)*size(Sinogramas{i},2)+1) : (col*size(Sinogramas{i},2))) = (Sinogramas{i}./max(max(Sinogramas{i})));    
end
imshow(imageSinograms)
line([size(Sinogramas{1},2) size(Sinogramas{1},2)],[0 size(Sinogramas{1},1)*4],'color','w','LineWidth', 2)
% line([size(Sinogramas{1},2)*2 size(Sinogramas{1},2)*2],[0 size(Sinogramas{1},1)*2],'color','w','LineWidth', 2)
% line([size(Sinogramas{1},2)*3 size(Sinogramas{1},2)*3],[0 size(Sinogramas{1},1)*2],'color','w','LineWidth', 2)
% line([size(Sinogramas{1},2)*4 size(Sinogramas{1},2)*4],[0 size(Sinogramas{1},1)*2],'color','w','LineWidth', 2)
line([0 size(Sinogramas{1},2)*5],[size(Sinogramas{1},1)*1 size(Sinogramas{1},1)*1],'color','w','LineWidth', 2)
line([0 size(Sinogramas{1},2)*5],[size(Sinogramas{1},1)*2 size(Sinogramas{1},1)*2],'color','w','LineWidth', 2)
line([0 size(Sinogramas{1},2)*5],[size(Sinogramas{1},1)*3 size(Sinogramas{1},1)*3],'color','w','LineWidth', 2)
% line([0 size(Sinogramas{1},2)*5],[size(Sinogramas{1},1)*2 size(Sinogramas{1},1)*2],'color','w','LineWidth', 2)
% line([0 size(Sinogramas{1},2)*5],[size(Sinogramas{1},1)*3 size(Sinogramas{1},1)*3],'color','w','LineWidth', 2)
% line([0 size(Sinogramas{1},2)*5],[size(Sinogramas{1},1)*4 size(Sinogramas{1},1)*4],'color','w','LineWidth', 2)
set(gcf, 'Position', [50 50 1600 1200]);
set(gcf, 'InvertHardcopy', 'off')
% Agrego una leyenda en cada imagen para identificarlas:
for i = 1 : numel(Sinogramas)-1
    col = rem(i,2);
    if col == 0
        col = 2;
    end
    fila = ceil(i / 2);
    text(15 + size(Sinogramas{1},2) * (col-1), 15 + size(Sinogramas{1},1) * (fila-1) , sprintf('%d Eventos',sum(sum(Sinogramas{i}))) ,'Color','w','FontSize',9.5,'FontWeight','bold')
end
outputGraphsPath = '/workspaces/Martin/Doctorado/Tesis/Tesis Martín Belzunce/docusTesis/Figuras/Capitulo3/backprojection/';
graphicFilename = sprintf('sinogramas');
set(gcf,'PaperPositionMode','auto');    % Para que lo guarde en el tamaño modificado.
saveas(gcf, [outputGraphsPath graphicFilename], 'fig');
frame = getframe(gca);
imwrite(frame.cdata, [outputGraphsPath graphicFilename '.png']);
saveas(gca, [outputGraphsPath graphicFilename], 'epsc');

% Guardo todos los sinogramas como interfile:
outputResultsPath = '/workspaces/Martin/Doctorado/Tesis/Tesis Martín Belzunce/docusTesis/Resultados/Capitulo3';
for i = 1 : numel(Sinogramas)-1
    % Estos sinogramas son para mlem y ahí el fov lo tengo circular, por lo
    % que debo recortar los valores de r que se pasan:
    numR = size(Sinogramas{i},2);
    binsEliminarPorLado = round((numR-numR / sqrt(2)) / 2);
    SinogramaMlem = Sinogramas{i}(:, binsEliminarPorLado: (end-binsEliminarPorLado));
    interfileWriteSino(single(SinogramaMlem), sprintf('%s/Sino_%dEventos', outputResultsPath,sum(sum(Sinogramas{i}))));
end
%% RECONTRUCCIÓN DE SINOGRAMAS CON BACKPROJECTION Y FILTRO RAMPA
close all
for i = 1 : numel(Sinogramas) -1
    % Agrando un poco el sinograma porque sino no entra todo
    h = figure
    imagenesRampa{i} = iradon(Sinogramas{i}', 180/size(Sinogramas{i},1)/2:180/size(Sinogramas{i},1):(180-180/size(Sinogramas{i},1)/2), 'linear', 'Ram-Lak', 1, 300);
    imshow(imagenesRampa{i} ./max(max(imagenesRampa{i})));
end

i = 1; fila = 1; col = 1;
imagen = [];
for i = 1 : numel(imagenesRampa)
    col = rem(i,2);
    if col == 0
        col = 2;
    end
    fila = ceil(i / 2);
    imagen(((fila-1)*size(imagenesRampa{i},1)+1) : (fila*size(imagenesRampa{i},1)), ((col-1)*size(imagenesRampa{i},2)+1) : (col*size(imagenesRampa{i},2))) = (imagenesRampa{i}./max(max(imagenesRampa{i})));    
end
h = figure
imshow(imagen ./max(max(imagen)));

line([size(imagenesRampa{1},2) size(imagenesRampa{1},2)],[0 size(imagenesRampa{1},1)*4],'color','w','LineWidth', 2)
line([0 size(imagenesRampa{1},2)*4 ],[size(imagenesRampa{1},1) size(imagenesRampa{1},1)],'color','w','LineWidth', 2)
line([0 size(imagenesRampa{1},2)*4 ],[size(imagenesRampa{1},1)*2 size(imagenesRampa{1},1)*2],'color','w','LineWidth', 2)
line([0 size(imagenesRampa{1},2)*4 ],[size(imagenesRampa{1},1)*3 size(imagenesRampa{1},1)*3],'color','w','LineWidth', 2)
set(gcf, 'Position', [50 50 1600 1200]);
set(gcf, 'InvertHardcopy', 'off')
% Agrego una leyenda en cada imagen para identificarlas:
for i = 1 : numel(imagenesRampa)
    col = rem(i,2);
    if col == 0
        col = 2;
    end
    fila = ceil(i / 2);
    text(15 + size(imagenesRampa{1},2) * (col-1), 15 + size(imagenesRampa{1},1) * (fila-1) , sprintf('%d Eventos',sum(sum(Sinogramas{i})))  ,'Color','w','FontSize',9.5,'FontWeight','bold')
end
outputGraphsPath = '/workspaces/Martin/Doctorado/Tesis/Tesis Martín Belzunce/docusTesis/Figuras/Capitulo3/backprojection/';
graphicFilename = sprintf('backprojectionRampaRuidoPois');
set(gcf,'PaperPositionMode','auto');    % Para que lo guarde en el tamaño modificado.
saveas(gcf, [outputGraphsPath graphicFilename], 'fig');
frame = getframe(gca);
imwrite(frame.cdata, [outputGraphsPath graphicFilename '.png']);
saveas(gca, [outputGraphsPath graphicFilename], 'epsc');
%% RECONSTRUCCIÓN DE SINOGRAMA 6 CON DISTINTOS FILTROS
i = 6; % Selección del sinograma 
% Agrando un poco el sinograma porque sino no entra todo
h = figure
imagenesFiltros{1} = iradon(Sinogramas{i}', 180/size(Sinogramas{i},1)/2:180/size(Sinogramas{i},1):(180-180/size(Sinogramas{i},1)/2), 'linear', 'Shepp-Logan', 0.7, 300);
imshow(imagenesFiltros{1} ./max(max(imagenesFiltros{1})));
h = figure
imagenesFiltros{2} = iradon(Sinogramas{i}', 180/size(Sinogramas{i},1)/2:180/size(Sinogramas{i},1):(180-180/size(Sinogramas{i},1)/2), 'linear', 'Cosine', 0.7, 300);
imshow(imagenesFiltros{2} ./max(max(imagenesFiltros{2})));
h = figure
imagenesFiltros{3} = iradon(Sinogramas{i}', 180/size(Sinogramas{i},1)/2:180/size(Sinogramas{i},1):(180-180/size(Sinogramas{i},1)/2), 'linear', 'Hamming', 0.7, 300);
imshow(imagenesFiltros{3} ./max(max(imagenesFiltros{3})));
h = figure
imagenesFiltros{4} = iradon(Sinogramas{i}', 180/size(Sinogramas{i},1)/2:180/size(Sinogramas{i},1):(180-180/size(Sinogramas{i},1)/2), 'linear', 'Hann', 0.7, 300);
imshow(imagenesFiltros{4} ./max(max(imagenesFiltros{4})));

i = 1; fila = 1; col = 1;
imagen = [];
for i = 1 : numel(imagenesFiltros)
    col = rem(i,2);
    if col == 0
        col = 2;
    end
    fila = ceil(i / 2);
    imagen(((fila-1)*size(imagenesFiltros{i},1)+1) : (fila*size(imagenesFiltros{i},1)), ((col-1)*size(imagenesFiltros{i},2)+1) : (col*size(imagenesFiltros{i},2))) = (imagenesFiltros{i}./max(max(imagenesFiltros{i})));    
end
h = figure
imshow(imagen ./max(max(imagen)));

line([size(imagenesFiltros{1},2) size(imagenesFiltros{1},2)],[0 size(imagenesFiltros{1},1)*4],'color','w','LineWidth', 2)
line([0 size(imagenesFiltros{1},2)*4 ],[size(imagenesFiltros{1},1) size(imagenesFiltros{1},1)],'color','w','LineWidth', 2)
line([0 size(imagenesFiltros{1},2)*4 ],[size(imagenesFiltros{1},1)*2 size(imagenesFiltros{1},1)*2],'color','w','LineWidth', 2)
line([0 size(imagenesFiltros{1},2)*4 ],[size(imagenesFiltros{1},1)*3 size(imagenesFiltros{1},1)*3],'color','w','LineWidth', 2)
set(gcf, 'Position', [50 50 1200 800]);
set(gcf, 'InvertHardcopy', 'off')
% Agrego una leyenda en cada imagen para identificarlas:
leyendas = {'Shepp-Logan', 'Coseno', 'Hamming', 'Hann'}
for i = 1 : numel(imagenesFiltros)
    col = rem(i,2);
    if col == 0
        col = 2;
    end
    fila = ceil(i / 2);
    text(15 + size(imagenesFiltros{1},2) * (col-1), 15 + size(imagenesFiltros{1},1) * (fila-1) , leyendas{i}  ,'Color','w','FontSize',9.5,'FontWeight','bold')
end
outputGraphsPath = '/workspaces/Martin/Doctorado/Tesis/Tesis Martín Belzunce/docusTesis/Figuras/Capitulo3/backprojection/';
graphicFilename = sprintf('backprojectionRuidoPoisVariosFilt');
set(gcf,'PaperPositionMode','auto');    % Para que lo guarde en el tamaño modificado.
saveas(gcf, [outputGraphsPath graphicFilename], 'fig');
frame = getframe(gca);
imwrite(frame.cdata, [outputGraphsPath graphicFilename '.png']);
saveas(gca, [outputGraphsPath graphicFilename], 'epsc');

%% RECONSTRUCCIÓN DE SINOGRAMA 6 CON DISTINTOS FILTROS HANN
close all
i = 7; % Selección del sinograma 
fc = [0.4 0.6 0.8 1.0];
% Agrando un poco el sinograma porque sino no entra todo
h = figure
imagenesFiltros{1} = iradon(Sinogramas{i}', 180/size(Sinogramas{i},1)/2:180/size(Sinogramas{i},1):(180-180/size(Sinogramas{i},1)/2), 'linear', 'Hann', fc(1), 300);
imshow(imagenesFiltros{1} ./max(max(imagenesFiltros{1})));
h = figure
imagenesFiltros{2} = iradon(Sinogramas{i}', 180/size(Sinogramas{i},1)/2:180/size(Sinogramas{i},1):(180-180/size(Sinogramas{i},1)/2), 'linear', 'Hann', fc(2), 300);
imshow(imagenesFiltros{2} ./max(max(imagenesFiltros{2})));
h = figure
imagenesFiltros{3} = iradon(Sinogramas{i}', 180/size(Sinogramas{i},1)/2:180/size(Sinogramas{i},1):(180-180/size(Sinogramas{i},1)/2), 'linear', 'Hann', fc(3), 300);
imshow(imagenesFiltros{3} ./max(max(imagenesFiltros{3})));
h = figure
imagenesFiltros{4} = iradon(Sinogramas{i}', 180/size(Sinogramas{i},1)/2:180/size(Sinogramas{i},1):(180-180/size(Sinogramas{i},1)/2), 'linear', 'Hann', fc(4), 300);
imshow(imagenesFiltros{4} ./max(max(imagenesFiltros{4})));

i = 1; fila = 1; col = 1;
imagen = [];
for i = 1 : numel(imagenesFiltros)
    col = rem(i,2);
    if col == 0
        col = 2;
    end
    fila = ceil(i / 2);
    imagen(((fila-1)*size(imagenesFiltros{i},1)+1) : (fila*size(imagenesFiltros{i},1)), ((col-1)*size(imagenesFiltros{i},2)+1) : (col*size(imagenesFiltros{i},2))) = (imagenesFiltros{i}./max(max(imagenesFiltros{i})));    
end
h = figure
imshow(imagen ./max(max(imagen)));

line([size(imagenesFiltros{1},2) size(imagenesFiltros{1},2)],[0 size(imagenesFiltros{1},1)*4],'color','w','LineWidth', 2)
line([0 size(imagenesFiltros{1},2)*4 ],[size(imagenesFiltros{1},1) size(imagenesFiltros{1},1)],'color','w','LineWidth', 2)
line([0 size(imagenesFiltros{1},2)*4 ],[size(imagenesFiltros{1},1)*2 size(imagenesFiltros{1},1)*2],'color','w','LineWidth', 2)
line([0 size(imagenesFiltros{1},2)*4 ],[size(imagenesFiltros{1},1)*3 size(imagenesFiltros{1},1)*3],'color','w','LineWidth', 2)
set(gcf, 'Position', [50 50 800 600]);
set(gcf, 'InvertHardcopy', 'off')

for i = 1 : numel(imagenesFiltros)
    col = rem(i,2);
    if col == 0
        col = 2;
    end
    fila = ceil(i / 2);
    text(15 + size(imagenesFiltros{1},2) * (col-1), 15 + size(imagenesFiltros{1},1) * (fila-1) , sprintf('Fc = %.1f', fc(i)),'Color','w','FontSize',9.5,'FontWeight','bold')
end
outputGraphsPath = '/workspaces/Martin/Doctorado/Tesis/Tesis Martín Belzunce/docusTesis/Figuras/Capitulo3/backprojection/';
graphicFilename = sprintf('backprojectionRuidoPoisFiltHann_Fc');
set(gcf,'PaperPositionMode','auto');    % Para que lo guarde en el tamaño modificado.
saveas(gcf, [outputGraphsPath graphicFilename], 'fig');
frame = getframe(gca);
imwrite(frame.cdata, [outputGraphsPath graphicFilename '.png']);
saveas(gca, [outputGraphsPath graphicFilename], 'epsc');


%% RECONSTRUCCI�N CON BACKPROJECTION CON DISTINTSO METODOS DE INTERPOLACION
close all
clear imagenesFiltros
i = 7; % Selección del sinograma 
fc =0.6
% Agrando un poco el sinograma porque sino no entra todo
h = figure
imagenesFiltros{1} = iradon(Sinogramas{i}', 180/size(Sinogramas{i},1)/2:180/size(Sinogramas{i},1):(180-180/size(Sinogramas{i},1)/2), 'nearest', 'Hann', fc, 300);
imshow(imagenesFiltros{1} ./max(max(imagenesFiltros{1})));
h = figure
imagenesFiltros{2} = iradon(Sinogramas{i}', 180/size(Sinogramas{i},1)/2:180/size(Sinogramas{i},1):(180-180/size(Sinogramas{i},1)/2), 'linear', 'Hann', fc, 300);
imshow(imagenesFiltros{2} ./max(max(imagenesFiltros{2})));
h = figure
imagenesFiltros{3} = iradon(Sinogramas{i}', 180/size(Sinogramas{i},1)/2:180/size(Sinogramas{i},1):(180-180/size(Sinogramas{i},1)/2), 'spline', 'Hann', fc, 300);
imshow(imagenesFiltros{3} ./max(max(imagenesFiltros{3})));

i = 1; fila = 1; col = 1;
imagen = [];
for i = 1 : numel(imagenesFiltros)
    col = rem(i,3);
    if col == 0
        col = 3;
    end
    fila = ceil(i / 3);
    imagen(((fila-1)*size(imagenesFiltros{i},1)+1) : (fila*size(imagenesFiltros{i},1)), ((col-1)*size(imagenesFiltros{i},2)+1) : (col*size(imagenesFiltros{i},2))) = (imagenesFiltros{i}./max(max(imagenesFiltros{i})));    
end
h = figure
imshow(imagen ./max(max(imagen)));

line([size(imagenesFiltros{1},2) size(imagenesFiltros{1},2)],[0 size(imagenesFiltros{1},1)*4],'color','w','LineWidth', 2)
line([size(imagenesFiltros{1},2)*2 size(imagenesFiltros{1},2)*2],[0 size(imagenesFiltros{1},1)*4],'color','w','LineWidth', 2)
line([0 size(imagenesFiltros{1},2)*4 ],[size(imagenesFiltros{1},1) size(imagenesFiltros{1},1)],'color','w','LineWidth', 2)
line([0 size(imagenesFiltros{1},2)*4 ],[size(imagenesFiltros{1},1)*2 size(imagenesFiltros{1},1)*2],'color','w','LineWidth', 2)
line([0 size(imagenesFiltros{1},2)*4 ],[size(imagenesFiltros{1},1)*3 size(imagenesFiltros{1},1)*3],'color','w','LineWidth', 2)
set(gcf, 'Position', [50 50 800 600]);
set(gcf, 'InvertHardcopy', 'off')
leyendas = {'Vecino', 'bilineal', 'spline'}
for i = 1 : numel(imagenesFiltros)
    col = rem(i,3);
    if col == 0
        col = 3;
    end
    fila = ceil(i / 3);
    text(15 + size(imagenesFiltros{1},2) * (col-1), 15 + size(imagenesFiltros{1},1) * (fila-1) , leyendas{i},'Color','w','FontSize',9.5,'FontWeight','bold')
end
outputGraphsPath = '/workspaces/Martin/Doctorado/Tesis/Tesis Martín Belzunce/docusTesis/Figuras/Capitulo3/backprojection/';
graphicFilename = sprintf('backprojectionRuidoPoisFiltHann_Fc0.6_Interp');
set(gcf,'PaperPositionMode','auto');    % Para que lo guarde en el tamaño modificado.
saveas(gcf, [outputGraphsPath graphicFilename], 'fig');
frame = getframe(gca);
imwrite(frame.cdata, [outputGraphsPath graphicFilename '.png']);
saveas(gca, [outputGraphsPath graphicFilename], 'epsc');