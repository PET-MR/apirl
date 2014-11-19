%  *********************************************************************
%  Proyecto AR-PET. Comisión Nacional de Energía Atómica.
%  Autor: Martín Belzunce. UTN-FRBA.
%  Fecha de Creación: 19/02/2013
%  *********************************************************************
%	            OBTENCIÓN DE FWHM DE UNA IMAGEN
%  function sinograms2D = getFwhmOfPointSourceImage(inputImage,sizePixel_mm,dim, graficar, fullFilename)
%
%  Función que obtiene el FWHM de una imagen con una fuente puntual. Recibe
%  como parámetro la imagen y el eje en el que se quiere obtener el FWHM.
%  El mismo se obtiene a partir de un corte paralelo al eje de análisis que
%  pasa por el pico de intensidad de la imagen. Se hace a nivel de píxel y
%  fiteando una gaussiana y sacando su desvío. La misma puede ser tando de
%  dos como 3 dimensiones. Opcionalmente recibe un flag para graficar los
%  resultados y el directorio de salida donde guardarlos.
%
%  Detalle de los parámetros de entrada:
%   - inputImage: imagen de dos o tres dimensiones.
%   - sizePixel_mm: tamaño de píxel en mm, debe tener tanto elementos como
%   dimensiones inputImage.
%   - dim: dimensión en la que se hará el cálculo, siendo 1:x 2:y 3:z
%   - graficar: o no se grafican los resultados, 1 si. Si se pasa 1 se debe
%   pasar también como parámetro el outputPath.
%   - fullFilename: nnombre del archivo de salida completo (incluyendo path) sin extensión.
%
%  Parámetros de Salida:
%   - fwhm: calculado a partir de la iamgen.
%   -fwhm_ fiteado: calculado a partir del desvío de una guassiana fiteada
%   en el corte.
%  Ejemplo de llamada:
%   [fwhm, fwhm_fiteado] = getFwhmOfPointSourceImage(image,[10 10],1, 1,
%   './result')

function [fwhm, fwhm_fiteado] = getFwhmOfPointSourceImage(inputImage, sizePixel_mm, dim, graficar, fullFilename)

% Si no recibo el parámetro de graficar, no se grafica.
if nargin == 2
    graficar = 0;
end

if nargin == 3
    error('Si se desea graficar los resultados se debe indicar en el cuarto parámetro. Ej: [fwhm, fwhm_fiteado] = getFwhmOfPointSourceImage(inputImage,dim, graficar, outputPath)');
end

% Verifico que la dimensión que me piden no exceda el tamaño de la imagen:
if ndims(inputImage) < dim
    error('La imagen no tiene la dimensión requerida');
end

% Verifico que tengo el tamaño del píxel en cada coordenada
if ndims(inputImage) ~= numel(sizePixel_mm)
    error('El vector sizePixel_mm debe tener tanto elementos comodimensiones la imagen a procesar.');
end

% Verifico que sea una imagen de dos o tres dimensiones:
if (ndims(inputImage) ~= 3) && (ndims(inputImage) ~= 2)
    error('La matriz Imagen tiene que ser de dos o tres dimensiones.');
end

% Tamaño del grafico:
AGRANDARfIGURE=[5 109 1432 712];  

% Genero variables necesaris:
sizeImage = size(inputImage);
% Calculo las coordenadas en mm de cada píxel. Considero fov cilíndrico por
% lo que x e y van de .rfov a rfov, y el eje z de 0 a zfov.
coordPixels_mm{1} = -(sizePixel_mm(1)*sizeImage(1)/2-sizePixel_mm(1)/2):sizePixel_mm(1):(sizePixel_mm(1)*sizeImage(1)/2-sizePixel_mm(1)/2);
coordPixels_mm{2} = -(sizePixel_mm(2)*sizeImage(2)/2-sizePixel_mm(2)/2):sizePixel_mm(2):(sizePixel_mm(2)*sizeImage(2)/2-sizePixel_mm(2)/2);
% Si es un volumen calculo la tercera:
if (ndims(inputImage) == 3)
    coordPixels_mm{3} = sizePixel_mm(3)/2:sizePixel_mm(3):(sizePixel_mm(3)*sizeImage(3)-sizePixel_mm(3)/2);
end

coordPixelsPasoFino_mm{1} = -(sizePixel_mm(1)*sizeImage(1)/2-sizePixel_mm(1)/8):sizePixel_mm(1)/4:(sizePixel_mm(1)*sizeImage(1)/2-sizePixel_mm(1)/8);
coordPixelsPasoFino_mm{2} = -(sizePixel_mm(2)*sizeImage(2)/2-sizePixel_mm(2)/8):sizePixel_mm(2)/4:(sizePixel_mm(2)*sizeImage(2)/2-sizePixel_mm(2)/8);
% Si es un volumen calculo la tercera:
if (ndims(inputImage) == 3)
    coordPixelsPasoFino_mm{3} = sizePixel_mm(3)/8:sizePixel_mm(3)/4:(sizePixel_mm(3)*sizeImage(3)-sizePixel_mm(3)/8);
end

% Si quiero el corte enx o y, busco el slice donde está el pico, si me
% piden en el eje z busco la coordenada y del pico, para esto permuto entre
% las coordenadas 2 y 3:
if (dim == 3)
    inputImage = permute(inputImage,[1 3 2]);
end
[valor,slicePico] = max(max(max(inputImage)));
% Ahora proceso en el plano transversal (plano XY para dim 1 y 2, e YZ para dim 3, al estar permutado es la misma operación para ambos casos):
imagenPlanar = inputImage(:,:,slicePico);
% Obtención de resolución en FWHM fiteando una gaussiana:
pico = max(max(imagenPlanar));
[fila, columna] = find(imagenPlanar==pico);
% Ahora guardo un vector con el corte sobre el eje que me pidieron:
if dim == 1
    % Si me piden el eje x son en realidad las columnas
    vector = imagenPlanar(fila(1),:)./pico;
    % También almacena la coordenada del pico en este eje, en este caso en
    % las columnas:
    indicePicoEje = columna(1);
    % Texto para leyenda de grafico:
    textLabel = 'X [mm]';
elseif dim == 2
    % Si me piden el eje y son en realidad las filas.
    vector = imagenPlanar(:,columna(1))./pico;
    % En este caso el pico está en las columnas:
    indicePicoEje = fila(1);
    % Para el fiteo debo trasponer el vector:
    vector = vector';
    textLabel = 'Y [mm]';
elseif dim == 3
    % Si me piden el eje z serían las columnas ya que permute las
    % dimensiones 2 y 3 de la imagen.
    vector = imagenPlanar(fila(1),:)./pico;
    indicePicoEje = columna(1);
    textLabel = 'Z [mm]';
end

% Calculo el FWHM:
indicesMayoresMitad = find(vector>=0.5);
fwhm = (indicesMayoresMitad(end)-indicesMayoresMitad(1))*sizePixel_mm(dim);
% Si la hago fiteando una guassiana:
ParamsGaussiana = lsqcurvefit(@Gaussian,[max(vector),coordPixels_mm{dim}(indicePicoEje), fwhm/2.35],coordPixels_mm{dim},vector);
fwhm_fiteado = ParamsGaussiana(3)*2.32;

% Si hay que graficar, lo hago y guardo el gráfico:
if graficar
    
    % Grafico la original y la fiteada y la guardo:
    h1 = figure;
    plot(coordPixels_mm{dim}, vector, coordPixelsPasoFino_mm{dim}, Gaussian(ParamsGaussiana, coordPixelsPasoFino_mm{dim}), 'LineWidth',3);
    h2=legend('Corte de Imagen', 'Gaussian Ajustada','Location','NorthEast');
    set(h2, 'FontSize',16)
    set(gcf, 'Position', AGRANDARfIGURE);
    %     title('Log-Likelihood p', 'FontSize',20,'FontWeight','Bold');
    ylabel('Intensidad Normalizada','FontSize',18,'FontWeight','Bold');
    xlabel('Z [mm]','FontSize',18,'FontWeight','Bold');
    h3 = text(50,0.5, sprintf('FWHM: %.2f mm', fwhm));
    set(h3, 'FontSize',20)
    h3 = text(50,0.25, sprintf('FWHM fiteado: %.2f mm', fwhm_fiteado));
    set(h3, 'FontSize',20)
    saveas(gca, [fullFilename], 'tif');
    set(gcf,'PaperPositionMode','auto');    % Para que lo guarde en el tamaño modificado.
    frame = getframe(gca);
    imwrite(frame.cdata, [fullFilename '.png']);
    saveas(gca, [fullFilename], 'epsc');
end