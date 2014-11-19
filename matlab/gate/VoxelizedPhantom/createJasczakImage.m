%  *********************************************************************
%  Proyecto AR-PET. Comisión Nacional de Energía Atómica.
%  Autor: Martín Belzunce. UTN-FRBA.
%  Fecha de Creación: 19/01/2012
%  *********************************************************************
%   imagePhantom = createDaponteImage(sizeImage)
%  Función que crea una imagen del fantoma Daponte según el tamaño elegido.
function imagePhantom = createJasczakImage(sizeImage)
% Creo la matriz que contendra la imagen del fantoma:
imagePhantom = zeros(sizeImage, 'single');
actividadFondo = 2.5;
% 10 uCi por pixel, para pixel de 1cm x 1cm x0,4cm (Daponte con espesor de
% 4mm)
actividadRods = 15;

% La idea es obtener las coordenadas de cada rod, a partir de la cantidad
% de rods t el diámetro de cada una de ella.
diamInicialesRods_mm = [6.4, 7.9, 9.5, 11.1, 12.7, 15.9];
numFilasRods = [8 6 5 4 4 3];
numFilasRods = [7 5 5 4 4 3];
% Centros Iniciales:
offset_mm = 5;
centrosInicialesRodsX_mm = [0 3*offset_mm+diamInicialesRods_mm(2) 1*offset_mm+diamInicialesRods_mm(3) 0 -1.0*offset_mm-diamInicialesRods_mm(5) -1*offset_mm-diamInicialesRods_mm(6)];    
centrosInicialesRodsY_mm = [3*offset_mm 2.5*offset_mm -0.5*offset_mm -3.5*offset_mm -0.0*offset_mm 1.5*offset_mm] + [diamInicialesRods_mm(1:2)./2 -diamInicialesRods_mm(3:5)/2 diamInicialesRods_mm(6)/2];

% Primera porción de pizza:
indiceRods = 1;
i = 1;
offset = 0;
for j = 1 : numFilasRods(i)
    for k = 1: (j)
        if ~((j == numFilasRods(i)) && (k==1 || k==j))
            centrosRodsX_mm(indiceRods) = centrosInicialesRodsX_mm(i) - offset +2*(k-1)*diamInicialesRods_mm(i);
            centrosRodsY_mm(indiceRods) = centrosInicialesRodsY_mm(i) +2*(j-1)*diamInicialesRods_mm(i);

            radioRods_mm(indiceRods) = diamInicialesRods_mm(i)/2;
            indiceRods = indiceRods + 1;
        end
    end
    offset = offset + diamInicialesRods_mm(i);
end


% Segunda porción de pizza:
i = 2;
offset = 0;
for j = 1 : numFilasRods(i)
    for k = 1: (numFilasRods(i)-j+1)
        centrosRodsX_mm(indiceRods) = centrosInicialesRodsX_mm(i) + offset +2*(k-1)*diamInicialesRods_mm(i);
        centrosRodsY_mm(indiceRods) = centrosInicialesRodsY_mm(i) +2*(j-1)*diamInicialesRods_mm(i);
        radioRods_mm(indiceRods) = diamInicialesRods_mm(i)/2;
        indiceRods = indiceRods + 1;
    end
    offset = offset + diamInicialesRods_mm(i);
end

% Tercera porción de pizza:
i = 3;
offset = 0;
for j = 1 : numFilasRods(i)
    for k = 1: (numFilasRods(i)-j+1)
        centrosRodsX_mm(indiceRods) = centrosInicialesRodsX_mm(i) + offset +2*(k-1)*diamInicialesRods_mm(i);
        centrosRodsY_mm(indiceRods) = centrosInicialesRodsY_mm(i) -2*(j-1)*diamInicialesRods_mm(i);
        radioRods_mm(indiceRods) = diamInicialesRods_mm(i)/2;
        indiceRods = indiceRods + 1;
    end
    offset = offset + diamInicialesRods_mm(i);
end

% Cuarta porción de pizza:
i = 4;
offset = 0;
for j = 1 : numFilasRods(i)
    for k = 1: (j)
        centrosRodsX_mm(indiceRods) = centrosInicialesRodsX_mm(i) - offset +2*(k-1)*diamInicialesRods_mm(i);
        centrosRodsY_mm(indiceRods) = centrosInicialesRodsY_mm(i) -2*(j-1)*diamInicialesRods_mm(i);
        radioRods_mm(indiceRods) = diamInicialesRods_mm(i)/2;
        indiceRods = indiceRods + 1;
    end
    offset = offset + diamInicialesRods_mm(i);
end

% Quinta porción de pizza:
i = 5;
offset = 0;
for j = 1 : numFilasRods(i)
    for k = 1: (numFilasRods(i)-j+1)
        centrosRodsX_mm(indiceRods) = centrosInicialesRodsX_mm(i) - offset -2*(k-1)*diamInicialesRods_mm(i);
        centrosRodsY_mm(indiceRods) = centrosInicialesRodsY_mm(i) -2*(j-1)*diamInicialesRods_mm(i);
        radioRods_mm(indiceRods) = diamInicialesRods_mm(i)/2;
        indiceRods = indiceRods + 1;
    end
    offset = offset + diamInicialesRods_mm(i);
end

% Sexta porción de pizza:
i = 6;
offset = 0;
for j = 1 : numFilasRods(i)
    for k = 1: (numFilasRods(i)-j+1)
        centrosRodsX_mm(indiceRods) = centrosInicialesRodsX_mm(i) - offset -2*(k-1)*diamInicialesRods_mm(i);
        centrosRodsY_mm(indiceRods) = centrosInicialesRodsY_mm(i) +2*(j-1)*diamInicialesRods_mm(i);
        radioRods_mm(indiceRods) = diamInicialesRods_mm(i)/2;
        indiceRods = indiceRods + 1;
    end
    offset = offset + diamInicialesRods_mm(i);
end
    
% Consideramos que el FOV tiene un radio de 300 mm. A partir de ahi
% se calculan la cantidad de píxeles de cada radio. La imagen debe
% ser cuadrada, pero si las dimensiones apsadas no lo son, se
% considera el lado de menor tamaño como los 300mm.
radio_fov_mm = 108;
lado_fov_pixels = min(sizeImage);
% Coordenadas de píxeles (X columnas e Y filas):
[coordenadasX, coordenadasY] = meshgrid(1:sizeImage(2), 1:sizeImage(1));        

% Fondo:
indicesFondo = ((coordenadasX - sizeImage(2)/2).^2 + (coordenadasY - sizeImage(1)/2).^2) < ((sizeImage(2)/2).^2);
imagePhantom(indicesFondo) = actividadRods/4;

% Este fantoma sería para la ventana de cesio, así que simplemente asigno
% los rods, el fondo se considera cero. Recorro cada rod, paso sus
% coordenadas y dimensiones a píxeles y lo grafico:
for i=1 : numel(radioRods_mm)
    radioRod_pixels = (lado_fov_pixels * radioRods_mm(i) / radio_fov_mm) / 2;
    centroRodX_pixels = (lado_fov_pixels * (centrosRodsX_mm(i)+radio_fov_mm) / radio_fov_mm) / 2;
    centroRodY_pixels = (lado_fov_pixels * (-centrosRodsY_mm(i)+radio_fov_mm) / radio_fov_mm) / 2;
    indicesRods = ((coordenadasX - centroRodX_pixels).^2 + (coordenadasY - centroRodY_pixels).^2) < (radioRod_pixels.^2);
    imagePhantom(indicesRods) = actividadRods;
    disp(sprintf('Rod: %d. Centro (%f,%f). Radio: %f.', i, centrosRodsX_mm(i), centrosRodsY_mm(i), radioRods_mm(i)));
end

% Listo. Lo visualizo:
figure;
imshow(imagePhantom./max(max(imagePhantom)));

% escribo un txt con la definición de distintas fuentes en gate:
% /gate/Jaszcazk/daughters/name rod6mm_22
% /gate/Jaszcazk/daughters/insert cylinder
% /gate/rod6mm_22/placement/setTranslation -12.8 69.4 -46 mm
% /gate/rod6mm_22/geometry/setRmax 3.2 mm
% /gate/rod6mm_22/geometry/setRmin 0.0 mm
% /gate/rod6mm_22/geometry/setHeight 88 mm
% /gate/rod6mm_22/setMaterial Air
% #/gate/rod6mm_22/vis/forceWireframe
% /gate/rod6mm_22/vis/setColor gray
fid = fopen('rodsJasczak.mac', 'w');
for i=1 : numel(radioRods_mm)
    fprintf(fid, '/gate/agua/daughters/name rod%dmm_%d\n', round(2*radioRods_mm(i)), i);
    fprintf(fid, '/gate/agua/daughters/insert cylinder\n');
    fprintf(fid, '/gate/rod%dmm_%d/placement/setTranslation %.2f %.2f -49 mm\n', round(2*radioRods_mm(i)), i, centrosRodsX_mm(i), centrosRodsY_mm(i));
    fprintf(fid, '/gate/rod%dmm_%d/geometry/setRmax %.2f mm\n', round(2*radioRods_mm(i)), i, radioRods_mm(i));
    fprintf(fid, '/gate/rod%dmm_%d/geometry/setRmin 0.0 mm\n', round(2*radioRods_mm(i)), i);
    fprintf(fid, '/gate/rod%dmm_%d/geometry/setHeight 88 mm\n', round(2*radioRods_mm(i)), i);
    fprintf(fid, '/gate/rod%dmm_%d/setMaterial Air\n', round(2*radioRods_mm(i)), i);
    fprintf(fid, '#/gate/rod%dmm_%d/vis/forceWireframe\n', round(2*radioRods_mm(i)), i);
    fprintf(fid, '/gate/rod%dmm_%d/vis/setColor gray\n', round(2*radioRods_mm(i)), i);
    fprintf(fid, '\n');
end
for i=1 : numel(radioRods_mm)
    fprintf(fid,'/gate/rod%dmm_%d/attachPhantomSD\n', round(2*radioRods_mm(i)), i);
end
fclose(fid);

fid = fopen('forbidRodsJasczak.mac', 'w');
for i=1 : numel(radioRods_mm)
    fprintf(fid, '/gate/source/F18CylinderSource/gps/Forbid rod%dmm_%d\n', round(2*radioRods_mm(i)), i);
end
fclose(fid);