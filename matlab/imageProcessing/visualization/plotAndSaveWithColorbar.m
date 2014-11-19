%  *********************************************************************
%  Proyecto AR-PET. Comisión Nacional de Energía Atómica.
%  Autor: Martín Belzunce. UTN-FRBA.
%  Fecha de Creación: 31/08/2011
%  *********************************************************************
%  Función que muestra una imagen con colorbar a partir de un colormap
%  recibido como parámetro y lo guarda en distintos formatos (eps, png y
%  jpg) en el directorio y con el nombre de salida recibido en los
%  parámetros 3 y 4. El quinto parámetro es opcional e indica el tamaño del
%  figure, sino se pasa se utiliza uno por defecto.
function image = PlotAndSaveWithColorbar(image, cmap, outputPath, outputName, sizeFigure)
if nargin == 4
    sizeFigure = [50 50 1600 1200];
end
% Primero obtengo el máximo de la imagen:
maximo = max(max(image));
% Visualizo:
h = figure;
set(h,'Color',[1 1 1]);
imshow(image./maximo);
set(gcf, 'Position', sizeFigure);
% Aplico el colormap
colormap(cmap);
% Cambio las leyendas a las unidades que me interesan:
hcb = colorbar;
%set(hcb, 'title', 'Atenuación Lineal');
set(hcb, 'YTickLabelMode', 'manual');
set(hcb, 'FontWeight', 'bold');
ticks = str2num(get(hcb, 'YTickLabel'));
% Por alguna razón no me devuelve el cero y el 1:
ticks = [0; ticks; 1];   
labels = num2str(ticks*maximo,'%.2f');
set(hcb, 'YTickLabel', labels);
%set(hcb, 'YTickLabel', labels, 'FontSize', 16);
set(hcb);

set(gcf,'PaperPositionMode','auto');    % Para que lo guarde en el tamaño modificado.
set(gcf,'InvertHardcopy','on');
saveas(gcf, [outputPath '/' outputName], 'fig');
saveas(gca, [outputPath '/' outputName], 'png');
saveas(gca, [outputPath '/' outputName], 'epsc');