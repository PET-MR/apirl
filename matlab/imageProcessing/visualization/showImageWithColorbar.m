%  *********************************************************************
%  Proyecto AR-PET. Comisión Nacional de Energía Atómica.
%  Autor: Martín Belzunce. UTN-FRBA.
%  Fecha de Creación: 31/08/2011
%  *********************************************************************
%  Función que muestra una imagen con colorbar a partir de un colormap
%  recibido como parámetro y escala la escala de valores al valor adecuado.
%  Recibe un tercer parámetro opcional que es el tamaño de la imagen.
function image = showImageWithColorbar(image, cmap, sizeFigure)
if nargin == 2
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

