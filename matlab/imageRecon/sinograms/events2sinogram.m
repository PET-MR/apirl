% Obtención del ángulo Phi: (Y2-Y1)/(X2-X1) = -1/ tan(Phi). Por lo tanto,
% Phi = atan(-(X2-X1)/(Y2-Y1))
% Obtención del desplazamiento espacial r: Teniendo en cuenta que 
% r = cos(Phi).x + sen(Phi).y. Entonces r = cos(Phi).X1 + sen(Phi).Y1   
function Sinograma = Events2Sinogram (X1, Y1, X2, Y2)

ValoresPhi = 
Valoresr = 
% Cálculo de coordenadas Phi y R.
Phi = atan(-(X2-X1)./(Y2-Y1));
r = cos(Phi).*X1 + sin(Phi).*Y1;

% Se realiza un histograma de los conjuntos (Phi,r) para relenar el
% Sinograma.

