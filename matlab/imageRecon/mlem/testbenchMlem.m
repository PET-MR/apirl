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
clear all
close all
addpath('/sources/MATLAB/WorkingCopy/ImageRecon')
addpath('/sources/MATLAB/WorkingCopy/ImageProcessing')
% %% ADAPTO SINOGRAMAS A MLEM
% % Esto lo debo hacer previo a la reconstrucción si use sinogramas apra FOV
% % cuadrado.
% cuentasPorSino = [10029 31425 99826 315967 1001233 3163069 10002754 31622653];
% sinosPath = '/workspaces/Martin/Doctorado/Tesis/Tesis Martín Belzunce/docusTesis/Resultados/Capitulo3';
% for i = 1 : numel(cuentasPorSino)
%     % Leo el sinograma:
%     sino = interfileread(sprintf('%s/Sino_%dEventos.h33',sinosPath,cuentasPorSino(i)));
%     % Estos sinogramas son para mlem y ahí el fov lo tengo circular, por lo
%     % que debo recortar los valores de r que se pasan:
%     numR = size(sino,2);
%     binsEliminarPorLado = round((numR-numR / sqrt(2)) / 2);
%     sino = sino(:, binsEliminarPorLado: (end-binsEliminarPorLado));
%     interfileWriteSino(single(sino), sprintf('%s/Sino_%dEventos', sinosPath,cuentasPorSino(i)));
% end
%% LEVANTO LAS IMÁGENES RECONSTRUIDAS
% Las imágenes las levanto porque ya las reconstruí por afuera. Hay para
% distinta cantidad de cuentas:
cuentasPorSino = [10029 31425 99826 315967 1001233 3163069 10002754 31622653];
close all
imagesPath = '/workspaces/Martin/Doctorado/Tesis/Tesis Martín Belzunce/docusTesis/Resultados/Capitulo3/mlem';
for i = 1 : numel(cuentasPorSino)
    % Agrando un poco el sinograma porque sino no entra todo
    h = figure
    imagenesMlem{i} = interfileread(sprintf('%s/MLEM_Siddon_Sinogram2D_%d__final.h33',imagesPath,cuentasPorSino(i)));
    % La imagen esta invertida assí que tengo que cambiar el eje y (filas):
    imagenesMlem{i} = imagenesMlem{i}(end:-1:1,:);
    imshow(imagenesMlem{i} ./max(max(imagenesMlem{i})));
end

i = 1; fila = 1; col = 1;
imagen = [];
for i = 1 : numel(imagenesMlem)
    col = rem(i,2);
    if col == 0
        col = 2;
    end
    fila = ceil(i / 2);
    imagen(((fila-1)*size(imagenesMlem{i},1)+1) : (fila*size(imagenesMlem{i},1)), ((col-1)*size(imagenesMlem{i},2)+1) : (col*size(imagenesMlem{i},2))) = (imagenesMlem{i}./max(max(imagenesMlem{i})));    
end
h = figure
imshow(imagen ./max(max(imagen)));

line([size(imagenesMlem{1},2) size(imagenesMlem{1},2)],[0 size(imagenesMlem{1},1)*4],'color','w','LineWidth', 2)
line([0 size(imagenesMlem{1},2)*4 ],[size(imagenesMlem{1},1) size(imagenesMlem{1},1)],'color','w','LineWidth', 2)
line([0 size(imagenesMlem{1},2)*4 ],[size(imagenesMlem{1},1)*2 size(imagenesMlem{1},1)*2],'color','w','LineWidth', 2)
line([0 size(imagenesMlem{1},2)*4 ],[size(imagenesMlem{1},1)*3 size(imagenesMlem{1},1)*3],'color','w','LineWidth', 2)
set(gcf, 'Position', [50 50 1600 1200]);
set(gcf, 'InvertHardcopy', 'off')
% Agrego una leyenda en cada imagen para identificarlas:
for i = 1 : numel(imagenesMlem)
    col = rem(i,2);
    if col == 0
        col = 2;
    end
    fila = ceil(i / 2);
    text(15 + size(imagenesMlem{1},2) * (col-1), 15 + size(imagenesMlem{1},1) * (fila-1) , sprintf('%d Eventos',cuentasPorSino(i))  ,'Color','w','FontSize',9.5,'FontWeight','bold')
end
outputGraphsPath = '/workspaces/Martin/Doctorado/Tesis/Tesis Martín Belzunce/docusTesis/Figuras/Capitulo3/mlem/';
graphicFilename = sprintf('mlemRuidoPois_50iter');
set(gcf,'PaperPositionMode','auto');    % Para que lo guarde en el tamaño modificado.
saveas(gcf, [outputGraphsPath graphicFilename], 'fig');
frame = getframe(gca);
imwrite(frame.cdata, [outputGraphsPath graphicFilename '.png']);
saveas(gca, [outputGraphsPath graphicFilename], 'epsc');
%% LEVANTO LAS IMÁGENES RECONSTRUIDAS CON MENOS ITERACIONES
cuentasPorSino = [10029 31425 99826 315967 1001233 3163069 10002754 31622653];
close all
imagesPath = '/workspaces/Martin/Doctorado/Tesis/Tesis Martín Belzunce/docusTesis/Resultados/Capitulo3/mlem';
% Repito todas las operaciones para distinta cantidad de iteraciones:
iteraciones = [0 10 20 30 40];
for iteracion = 1 : numel(iteraciones)
    for i = 1 : numel(cuentasPorSino)
        % Agrando un poco el sinograma porque sino no entra todo
        h = figure
        imagenesMlem{i} = interfileread(sprintf('%s/MLEM_Siddon_Sinogram2D_%d__iter_%d.h33',imagesPath,cuentasPorSino(i), iteraciones(iteracion)));
        % La imagen esta invertida assí que tengo que cambiar el eje y (filas):
        imagenesMlem{i} = imagenesMlem{i}(end:-1:1,:);
        imshow(imagenesMlem{i} ./max(max(imagenesMlem{i})));
    end

    i = 1; fila = 1; col = 1;
    imagen = [];
    for i = 1 : numel(imagenesMlem)
        col = rem(i,2);
        if col == 0
            col = 2;
        end
        fila = ceil(i / 2);
        imagen(((fila-1)*size(imagenesMlem{i},1)+1) : (fila*size(imagenesMlem{i},1)), ((col-1)*size(imagenesMlem{i},2)+1) : (col*size(imagenesMlem{i},2))) = (imagenesMlem{i}./max(max(imagenesMlem{i})));    
    end
    h = figure
    imshow(imagen ./max(max(imagen)));

    line([size(imagenesMlem{1},2) size(imagenesMlem{1},2)],[0 size(imagenesMlem{1},1)*4],'color','w','LineWidth', 2)
    line([0 size(imagenesMlem{1},2)*4 ],[size(imagenesMlem{1},1) size(imagenesMlem{1},1)],'color','w','LineWidth', 2)
    line([0 size(imagenesMlem{1},2)*4 ],[size(imagenesMlem{1},1)*2 size(imagenesMlem{1},1)*2],'color','w','LineWidth', 2)
    line([0 size(imagenesMlem{1},2)*4 ],[size(imagenesMlem{1},1)*3 size(imagenesMlem{1},1)*3],'color','w','LineWidth', 2)
    set(gcf, 'Position', [50 50 1600 1200]);
    set(gcf, 'InvertHardcopy', 'off')
    % Agrego una leyenda en cada imagen para identificarlas:
    for i = 1 : numel(imagenesMlem)
        col = rem(i,2);
        if col == 0
            col = 2;
        end
        fila = ceil(i / 2);
        text(15 + size(imagenesMlem{1},2) * (col-1), 15 + size(imagenesMlem{1},1) * (fila-1) , sprintf('%d Eventos',cuentasPorSino(i))  ,'Color','w','FontSize',9.5,'FontWeight','bold')
    end
    outputGraphsPath = '/workspaces/Martin/Doctorado/Tesis/Tesis Martín Belzunce/docusTesis/Figuras/Capitulo3/mlem/';
    graphicFilename = sprintf('mlemRuidoPois_%diter', iteraciones(iteracion));
    set(gcf,'PaperPositionMode','auto');    % Para que lo guarde en el tamaño modificado.
    saveas(gcf, [outputGraphsPath graphicFilename], 'fig');
    frame = getframe(gca);
    imwrite(frame.cdata, [outputGraphsPath graphicFilename '.png']);
    saveas(gca, [outputGraphsPath graphicFilename], 'epsc');
end
%% RECONSTRUCCIÓN DE UN SINOGRAMA PERO POR ITERACIONES
cuentasPorSino = 1001233;
cuentasPorSino = 3163069;
close all
imagesPath = '/workspaces/Martin/Doctorado/Tesis/Tesis Martín Belzunce/docusTesis/Resultados/Capitulo3/mlem';
% Repito todas las operaciones para distinta cantidad de iteraciones:
iteraciones = [0 10 20 30 40 50];
imagenesMlem = [];
for i = 1 : numel(iteraciones)-1
    % Agrando un poco el sinograma porque sino no entra todo
    h = figure
    imagenesMlem{i} = interfileread(sprintf('%s/MLEM_Siddon_Sinogram2D_%d__iter_%d.h33',imagesPath,cuentasPorSino, iteraciones(i)));
    % La imagen esta invertida assí que tengo que cambiar el eje y (filas):
    imagenesMlem{i} = imagenesMlem{i}(end:-1:1,:);
    imshow(imagenesMlem{i} ./max(max(imagenesMlem{i})));
end
i = i + 1;
% Leo una más que es la final con 50 iteraciones:
imagenesMlem{i} = interfileread(sprintf('%s/MLEM_Siddon_Sinogram2D_%d__final.h33',imagesPath,cuentasPorSino));
imagenesMlem{i} = imagenesMlem{i}(end:-1:1,:);

i = 1; fila = 1; col = 1;
imagen = [];
for i = 1 : numel(imagenesMlem)
    col = rem(i,3);
    if col == 0
        col = 3;
    end
    fila = ceil(i / 3);
    imagen(((fila-1)*size(imagenesMlem{i},1)+1) : (fila*size(imagenesMlem{i},1)), ((col-1)*size(imagenesMlem{i},2)+1) : (col*size(imagenesMlem{i},2))) = (imagenesMlem{i}./max(max(imagenesMlem{i})));    
end
h = figure
imshow(imagen ./max(max(imagen)));

line([size(imagenesMlem{1},2) size(imagenesMlem{1},2)],[0 size(imagenesMlem{1},1)*3],'color','w','LineWidth', 2)
line([size(imagenesMlem{1},2)*2 size(imagenesMlem{1},2)*2],[0 size(imagenesMlem{1},1)*3],'color','w','LineWidth', 2)
line([0 size(imagenesMlem{1},2)*3 ],[size(imagenesMlem{1},1) size(imagenesMlem{1},1)],'color','w','LineWidth', 2)
%line([0 size(imagenesMlem{1},2)*2 ],[size(imagenesMlem{1},1)*2 size(imagenesMlem{1},1)*2],'color','w','LineWidth', 2)
%line([0 size(imagenesMlem{1},2)*4 ],[size(imagenesMlem{1},1)*3 size(imagenesMlem{1},1)*3],'color','w','LineWidth', 2)
set(gcf, 'Position', [50 50 1600 1200]);
set(gcf, 'InvertHardcopy', 'off')
% Agrego una leyenda en cada imagen para identificarlas:
iteraciones(1) = 1;
for i = 1 : numel(imagenesMlem)
    col = rem(i,3);
    if col == 0
        col = 3;
    end
    fila = ceil(i / 3);
    text(15 + size(imagenesMlem{1},2) * (col-1), 15 + size(imagenesMlem{1},1) * (fila-1) , sprintf('%d Iteraciones',iteraciones(i))  ,'Color','w','FontSize',9.5,'FontWeight','bold')
end
outputGraphsPath = '/workspaces/Martin/Doctorado/Tesis/Tesis Martín Belzunce/docusTesis/Figuras/Capitulo3/mlem/';
graphicFilename = sprintf('mlemRuidoPois_%d_varias_iter', cuentasPorSino);
set(gcf,'PaperPositionMode','auto');    % Para que lo guarde en el tamaño modificado.
saveas(gcf, [outputGraphsPath graphicFilename], 'fig');
frame = getframe(gca);
imwrite(frame.cdata, [outputGraphsPath graphicFilename '.png']);
saveas(gca, [outputGraphsPath graphicFilename], 'epsc');
