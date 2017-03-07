%**************************************************************************
%
%  Curso Análisis de Imágenes Médicas
%
%  Autor: Martín Belzunce
%
%  Fecha: 21/12/2010
%
%  Descripcion:
%  Función modelo para el suavizado de imágenes por difusión inhomogénea.
%
%**************************************************************************

function D = coefDifusion(ModGrad, lambda, alpha, c)

    D = 1 - exp(-lambda*((c./ModGrad).^alpha));