%  *********************************************************************
%  Proyecto AR-PET. Comisión Nacional de Energía Atómica.
%  Autor: Martín Belzunce. UTN-FRBA.
%  Fecha de Creación: 29/08/2011
%  *********************************************************************
%
%  function z = ssrb(z1,z2)
%  Función que realiza el rebinning ssrb, que simplemente calcula la
%  posición o anillo medio a partir de dos posiciones axiales. Recibe como
%  parámetros un vector con coordenadas z1 y z2, de la posición axial de
%  los dos eventos en coincidencia. Devuelve un vector del mismo tamaño con
%  la posición media. Puede pasarsele como coordenadas en mm, como en
%  anillos.
%  19/11/12: Cambio para que reciba todo el evento y lo devuelva completo.
%  Esto es para hacerlo compatible con el msrb. Los eventos son matrices de
%  nx3, donde la tercer componente es el z.
function [eventosSalida]  = ssrb(eventosEntrada)

eventosSalida = eventosEntrada;
eventosSalida(:,3) = (eventosEntrada(:,3)+eventosEntrada(:,6))/2;
eventosSalida(:,6) = (eventosEntrada(:,3)+eventosEntrada(:,6))/2;