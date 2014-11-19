%**************************************************************************
%
%  Curso An�lisis de Im�genes M�dicas
%
%  Autor: Mart�n Belzunce
%
%  Fecha: 21/12/2010
%
%  Descripcion:
%  Funci�n modelo para el suavizado de im�genes por difusi�n inhomog�nea.
%
%**************************************************************************

function D = coefDifusion(ModGrad, lambda, alpha, c)

    D = 1 - exp(-lambda*((c./ModGrad).^alpha));