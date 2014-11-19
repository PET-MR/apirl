%  *********************************************************************
%  Proyecto AR-PET. Comisión Nacional de Energía Atómica.
%  Autor: Martín Belzunce. UTN-FRBA.
%  Fecha de Creación: 29/08/2011
%  *********************************************************************
%
%  functionEventosRebinned = msrb(Evento1,Evento2)
%  Función que realiza el rebinning msrb. Referencia en:
%  Three-dimensional image reconstruction for PET by multi-slice rebinning and axial image
%  filtering (Lewitt 1994)
%  El parámetros eventosEntrada es una matriz de nx6, donde n es la
%  cantidad de eventos y las primeras columas son las coordenadas del
%  primer evento y las segundas 3 las del segundo. Las coordenadas de las
%  coincidencias están con las coordanas x,y,z del sistema. Devuelve un
%  vector con todos los eventos rebinneados, como en MSRB se tiene
%  múltiples eventos por cada evento, la matriz de salida va a ser más
%  grande que las de entrada. Se tendrá una lista de todos los eventos
%  rebinneado. Al final, fuera de esta función hay que normalizar la
%  cantidad total de eventos.

%  posición o anillo medio a partir de dos posiciones axiales. Recibe como
%  parámetros un vector con las coordenadas completas de cada evento, ya
%  que a diferencia del ssrb, en este caso si se necesitan. Los x e y de
%  cada evento, los considero como los globales del GATE o del AR-PET y no
%  los internos del cabezal.
% La operación necesita varios parámetros, entre ellos la distancia total
% entre los evento, la distancia de intersección en el plano XY del FOV.

% Me devuelve múltiples líneas por cada evento, ya que el mismo se proyecta
% en distintos z. La cantidad de z en los que se proyecta depende de los
% eventos en si.
% Para un eje z con nRings el incremento con que se procesa z es:
%   incr = nRings/(zRing1-zRing2+1)

% Lo modifiqué porque era poco óptimo comutacionalmente:
% Ahora devuelve el sinograma directamente:
function [sinograms2D] = msrb(eventosEntrada, structSizeSino2D)

% indices:
indiceX1 = 1;
indiceY1 = 2;
indiceZ1 = 3;
indiceX2 = 4;
indiceY2 = 5;
indiceZ2 = 6;

% Valores centrales de los anillos:
deltaZ = structSizeSino2D.zFov_mm / structSizeSino2D.numZ;

% Ahora lleno los sinogramas. Para esto convierto el par de
% eventos (X1,Y1,Z1) y (X2,Y2,Z2) en lors del tipo (Thita,r,z).
% El ángulo thita está determinado por atan((Y1-Y2)/(X1-X2))+90
theta = atand((eventosEntrada(:,2)-eventosEntrada(:,5))./(eventosEntrada(:,1)-eventosEntrada(:,4))) + 90;
% El offset r lo puedo obtener reemplazando (x,y) con alguno de
% los dos puntos en la ecuación: r=x*cos(thita)+y*sin(thita)-
r_sino = cosd(theta).*eventosEntrada(:,1) + sind(theta).*eventosEntrada(:,2);
% Si quedan fuera del fov, los elimino:
indicesFueraFov = abs(r_sino)>structSizeSino2D.rFov_mm;
r_sino(indicesFueraFov) = [];
theta(indicesFueraFov) = [];
eventosEntrada(indicesFueraFov,:) = [];

% Parámetros que necesito:
z_medio = (eventosEntrada(:,indiceZ1)+eventosEntrada(:,indiceZ2))/2;
% Largo de la LOR en el plano XY(t_1_2 en los papers):
deltaX = eventosEntrada(:,indiceX1) - eventosEntrada(:,indiceX2);
deltaY = eventosEntrada(:,indiceY1) - eventosEntrada(:,indiceY2);
largoLor = sqrt(deltaX.^2 + deltaY.^2);

% Calculo la intersección de cada punto con el FOV, para esto obtengo la
% pendiente y uno de los puntos para la ecuación paramétrica:
lorVx = deltaX;
lorVy = deltaY;
lorVz = eventosEntrada(:,indiceZ1) - eventosEntrada(:,indiceZ2);
% La forma de calcularlo lo saco de siddon:
%   float segundoTermino = sqrt(4*(LOR.Vx*LOR.Vx*(rFov_mm*rFov_mm-LOR.P0.Y*LOR.P0.Y)
% 	  +LOR.Vy*LOR.Vy*(rFov_mm*rFov_mm-LOR.P0.X*LOR.P0.X)) + 8*LOR.Vx*LOR.P0.X*LOR.Vy*LOR.P0.Y);
% // Obtengo los valores de alpha donde se intersecciona la recta con la circunferencia.
%   // Como la debería cruzar en dos puntos hay dos soluciones.
%   alpha_xy_1 = (-2*(LOR.Vx*LOR.P0.X+LOR.Vy*LOR.P0.Y) + segundoTermino)/(2*(LOR.Vx*LOR.Vx+LOR.Vy*LOR.Vy));
%   alpha_xy_2 = (-2*(LOR.Vx*LOR.P0.X+LOR.Vy*LOR.P0.Y) - segundoTermino)/(2*(LOR.Vx*LOR.Vx+LOR.Vy*LOR.Vy));
%   
%   // Valores de alpha de entrada y de salida. El de entrada es el menor, porque la lor
%   // se recorre desde P0 a P1.
%   alpha_min = min(alpha_xy_1, alpha_xy_2);
%   alpha_max = max(alpha_xy_1, alpha_xy_2);
%   
%   // Coordenadas dentro de la imagen de los dos puntos de entrada:
%   x_1_mm = LOR.P0.X + LOR.Vx * alpha_min;
%   y_1_mm = LOR.P0.Y + LOR.Vy * alpha_min;
%   z_1_mm = LOR.P0.Z + LOR.Vz * alpha_min;
%   
%   x_2_mm = LOR.P0.X + LOR.Vx * alpha_max;
%   y_2_mm = LOR.P0.Y + LOR.Vy * alpha_max;
%   z_2_mm = LOR.P0.Z + LOR.Vz * alpha_max;
discriminanteEcuacion = sqrt(4.*(lorVx.^2.*(structSizeSino2D.rFov_mm.^2 - eventosEntrada(:,indiceY1).^2) + lorVy.^2.*(structSizeSino2D.rFov_mm.^2 - eventosEntrada(:,indiceX1).^2)) + ...
    8.*lorVx.*eventosEntrada(:,indiceX1).*lorVy.*eventosEntrada(:,indiceY1));
alpha1 = (-2*(lorVx.*eventosEntrada(:,indiceX1)+lorVy.*eventosEntrada(:,indiceY1)) + discriminanteEcuacion) ./ (2*(lorVx.^2+lorVy.^2));
alpha2 = (-2*(lorVx.*eventosEntrada(:,indiceX1)+lorVy.*eventosEntrada(:,indiceY1)) - discriminanteEcuacion) ./ (2*(lorVx.^2+lorVy.^2));

% alphaMin = min([alpha1, alpha2],[],2);
% alphaMax = max([alpha1, alpha2],[],2);

% Ya puedo sacar los puntos:
x1_fov = eventosEntrada(:,indiceX1) + lorVx .* alpha1;
y1_fov = eventosEntrada(:,indiceY1) + lorVy .* alpha1;
z1_fov = eventosEntrada(:,indiceZ1) + lorVz .* alpha1;
x2_fov = eventosEntrada(:,indiceX1) + lorVx .* alpha2;
y2_fov = eventosEntrada(:,indiceY1) + lorVy .* alpha2;
z2_fov = eventosEntrada(:,indiceZ1) + lorVz .* alpha2;

% Ahora si ya calculo el largo en el Fov:
t_1_2 = sqrt((x1_fov-x2_fov).^2+(y1_fov-y2_fov).^2);

% Ahora si empiezo con el rebinning, necesito los z máximos y mínimos del
% Fov:
% z_lo = z_medio - radiusFov.*abs(Evento1(indiceX1) - Evento1(indiceX2));
% Pero supuestamente ya los tengo, son los z1_fov y z2_fov.
% El incremento depende de los z:
% zInf = min(z2_fov,z1_fov); % Increible pero esto me da malos resultados
% en algunos número, por esolo hago de otra forma.
% zSup = max(z1_fov, z2_fov);
indiceMenor = z1_fov <= z2_fov;
zInf = zeros(size(z1_fov));
zSup = zeros(size(z1_fov));
zInf(indiceMenor) = z1_fov(indiceMenor);
zInf(~indiceMenor) = z2_fov(~indiceMenor);
zSup(indiceMenor) = z2_fov(indiceMenor);
zSup(~indiceMenor) = z1_fov(~indiceMenor);
% Si algún z es complejo (pasa, calculo de porque es un random o scatter
% que no cruza el FOV:
indicesComplex = (imag(zInf)~=0)|(imag(zSup)~=0);
zInf(indicesComplex) = [];
zSup(indicesComplex) = [];
r_sino(indicesComplex) = [];
theta(indicesComplex) = [];
eventosEntrada(indicesComplex,:) = [];
stepZ = structSizeSino2D.zFov_mm/structSizeSino2D.numZ;
% Tengo que obtener los zInf y zSup en índices de anillos:
zInf_rings = round((zInf+(structSizeSino2D.zFov_mm/2 - deltaZ/2)) ./ deltaZ)+1;
zInf_rings(zInf_rings<1) = 1;
zSup_rings = round((zSup+(structSizeSino2D.zFov_mm/2 - deltaZ/2)) ./ deltaZ)+1;
zSup_rings(zSup_rings>structSizeSino2D.numZ) = structSizeSino2D.numZ;
% Cada sinograma correspondiente a todos los anillos entre zInf y zSup
% deben ser incrementados en una cantidad incr. Dicha cantidad es la misma
% para cierta coincidencia analizada, pero cambia con cada coincidencia
% para mantener la cantidad de cuentas por LORs. Para ello el incremento es
% inversamente proporcional al máximo incremento. El máximo incremento es
% el número de anillos. Entonces en los casos extremos si la
% coincidencia está entre mismos anillos, se incrmenta numZ, por otro lado,
% si zInf y zSup cubren todos los anillos, se incrementa en uno cada
% posicion.
incr = (structSizeSino2D.numZ./(zSup_rings-zInf_rings + 1));
numRingsPorLor = (zSup_rings-zInf_rings + 1);
% Ahora para cada z, tengo múltiples valores de z lamentablemente solo lo
% % puedo hacer con un for:
% eventosSalida = zeros(sum(numRingsPorLor.*incr), 6);
% indiceSalida = 1;   % Con esto recorro el vector de salida que es distinto de i.
% for i = 1 : size(eventosEntrada, 1)
%     % Para cada LOR genero las coordenadas z de cada anillo intersectado y
%     % la repito incr veces:
%     for j = zInf_rings(i) : zSup_rings(i)
%         for k = 1 : round(incr(i))
%             eventosSalida(indiceSalida + k - 1,:) = eventosEntrada(i,:);
%             eventosSalida(indiceSalida + k - 1,3) = (zInf_rings(i)-1)*deltaZ -(zFov_mm/2-deltaZ/2);
%             eventosSalida(indiceSalida + k - 1,6) = (zInf_rings(i)-1)*deltaZ -(zFov_mm/2-deltaZ/2);
%         end
%         % Actualizo el indice de salida:
%         indiceSalida = indiceSalida + round(incr(i));
%     end
% end

% Inicializo el sinograma 2d a partir del tamaño definido en la estructura:
sinograms2D = single(zeros(structSizeSino2D.numTheta, structSizeSino2D.numR, ...
    structSizeSino2D.numZ));

% Acumulo todo en el array de sinogramas 2D porque cada evento lo dbeo
% distribuir en distintos anillos e incrementarlo en distintas cantidades. % Obtengo los índices en r y en theta de los bin a procesar:
r_sino(r_sino < structSizeSino2D.rValues_mm(1)) = structSizeSino2D.rValues_mm(1);
r_sino(r_sino > structSizeSino2D.rValues_mm(end)) = structSizeSino2D.rValues_mm(end);
theta(theta < structSizeSino2D.thetaValues_deg(1)) = structSizeSino2D.thetaValues_deg(1);
theta(theta > structSizeSino2D.thetaValues_deg(end)) = structSizeSino2D.thetaValues_deg(end);
indicesR = interp1(structSizeSino2D.rValues_mm,1:numel(structSizeSino2D.rValues_mm),r_sino,'nearest');
r_sino = structSizeSino2D.rValues_mm(indicesR)';
indicesTheta = interp1(structSizeSino2D.thetaValues_deg,1:numel(structSizeSino2D.thetaValues_deg),theta,'nearest');
theta = structSizeSino2D.thetaValues_deg(indicesTheta)';
for i = 1 : size(eventosEntrada, 1)
    for z = zInf_rings(i) : zSup_rings(i)
        sinograms2D(indicesTheta(i), indicesR(i), z) = sinograms2D(indicesTheta(i), indicesR(i), z) + (incr(i)/structSizeSino2D.numZ); % El incremento lo normalizo sobre numZ para mantener la cantidad de cuentas totales.
    end
end