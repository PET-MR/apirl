%  *********************************************************************
%  Proyecto AR-PET. Comisión Nacional de Energía Atómica.
%  Autor: Martín Belzunce. UTN-FRBA.
%  Fecha de Creación: 31/08/2011
%  *********************************************************************
%  Función que genera una única imagen en la que se visualizan los
%  distintos slices de un volumen. Se le debe pasar como parámetros el
%  volumen en una matriz de 3 dimensiones. Y luego la cantidad de columnas,
%  en que quiero dividir los slices:
% El parámetro enableSepLines con true
function image = getImageFromSlices(volume, columns, enableSepLines, sliceIndependientes)

if nargin == 2
    enableSepLines = false;
    sliceIndependientes = 0;
end

% A partir de la cantidad de slices y de columnas, obtengo las filas
% necesarias:
rows = ceil(size(volume,3) / columns);
% Creo la imagen de salida:
image = zeros(size(volume,1)*rows, size(volume,2)*columns);

% Para visualización normalizo a 1, y las líneas de separación las hago con
% intensidad = 1:
if ~sliceIndependientes
    volume = volume ./ max(max(max(volume)));
else
    for i = 1 : size(volume,3)
        volume(:,:,i) = volume(:,:,i) ./ max(max(volume(:,:,i)));
    end
end

% Cargo cada slice:
for i = 1 : rows
    for j = 1 : columns
        % Verifico que no haya pasado de la última imagen:
        if ((i-1)*columns+j) <= size(volume,3)
            image((size(volume,1)*(i-1)+1):(size(volume,1)*i),...
                (size(volume,2)*(j-1)+1):(size(volume,2)*j)) = volume(:,:,(i-1)*columns+j);
        end
    end
end

% Agrego las líneas:
if enableSepLines
    for i = 1 : rows-1
        image(((size(volume,1)*i):(size(volume,1)*i+1)),:) = 1;
    end
    for j = 1 : columns-1
        image(:,(size(volume,2)*j : (size(volume,2)*j+1))) = 1;
    end
end
