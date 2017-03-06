/**
\mainpage Página principal de Biblioteca de Reconstrucción de Imágenes para PET APIRL (AR-PET Image Reconstruction Library)
\image html encabezado.jpg


\section intro Introducción
A continuación se describen algunos ejemplos del uso de la herramienta doxygen para documentar código fuente.
La cual nos permitira dejar indicado las capacidades generales del código asi como también sus
limitaciones, errores y resultados obtenidos. En la sección 
\ref herramientas0 se encuentran los links a las herramientas necesarias para generar una documentación similar a la presente.


\section ejemplos Ejemplos
\par Encabezado de unidad.
\verbatim
	\file scriptBase.m
	\brief Ejemplo de utilización comandos de matlab.
	\author Jerónimo F. Atencio (jerome5416@gmail.com)
	\date 2010.02.18
	\version 1.0.0	
	\attention Este comentario es opcional.
	\bug Agrega este comentario a la lista de bugs.
	\warnings Agrega este comentario a la lista de warnings.
	\todo Agregar clase logger, para los logs de reconstrucción.
	\notes: Ejemplo de notas.
\endverbatim

\par Encabezado de función
\verbatim

 \fn prototipoDeLafuncion
 \brief breve descripción

 Descripción completa de la funcion, debe dejarse un renglón entre el brief y esta descripción
 \param parametros
 \return Valores de retorno.

 \warning Escribir los warnings.
 \bug Bugs, sin palabras...
 \todo Lo que hay por hacer.
 \note Notas.
\endverbatim

\par Formulas.

<b>Formula en linea</b>
\verbatim

	\f$ z = \sqrt{(x_0)^2+(y_0)^2}\f$
\endverbatim
\f$ z = \sqrt{(x_0)^2+(y_0)^2}\f$

<b>Formula en bloque</b>
\verbatim
	\f[
	\frac{1}{T}. 
	\sum_{k=1}^N f[k]h[w-k]
	\f]
\endverbatim
\f[
	\frac{1}{T}. 
	\sum_{k=1}^N f[k]h[w-k]
\f]

<b>Integral</b>
\verbatim
	\f[ 
	\int_{a}^b t dt 
	\f] 
\endverbatim
	\f[ 
	\int_{a}^b t dt 
	\f]

<b>Ejemplo del manual de doxygen</b>
\verbatim
 \f[
    |I_2|=\left| \int_{0}^T \psi(t) 
             \left\{ 
                u(a,t)-
                \int_{\gamma(t)}^a 
                \frac{d\theta}{k(\theta,t)}
                \int_{a}^\theta c(\xi)u_t(\xi,t)\,d\xi
             \right\} dt
          \right|
  \f]
\endverbatim
 \f[
    |I_2|=\left| \int_{0}^T \psi(t) 
             \left\{ 
                u(a,t)-
                \int_{\gamma(t)}^a 
                \frac{d\theta}{k(\theta,t)}
                \int_{a}^\theta c(\xi)u_t(\xi,t)\,d\xi
             \right\} dt
          \right|
  \f]


<b>Letras griegas</b>
Se muestran ejemplos de letras griegas en mayúscula y minúscula.
\verbatim
	\f[	
	\pi
	\Pi		
	\f]
\endverbatim
	\f[	
	\pi
	\Pi	
	\f]

\section linksDoc Ejemplo de link dentro de la documentación

Visite la sección <a href="modules.html"><b>Modulos</b></a> 

 \section herramientas0 Herramientas.
 Las herramientas necesarias para generar una documentación similar 
 a esta a partir de código deberá usar el siguiente software que permite
 extraer comentarios de su código.	

	- Necesarios
		- Doxygen  http://www.doxygen.org/
		- Graphviz http://www.graphviz.org/	
	- Para generar pdf y formulas
		- Ghostscript http://www.ghostscript.com/
		- MikTex http://miktex.org/2.8/setup
	- Encaso de usar Matlab
		- Filter pattern para Matlab http://www.mathworks.com/matlabcentral/fileexchange/25925-using-doxygen-with-matlab
		- Perl http://www.activestate.com/activeperl/

\section variablesEntorno0 Variables de entorno
Setee la variable de entorno PATH con los siguiente valores
\verbatim
	- C:\Archivos de programa\gs\gs8.71\bin
	- C:\Archivos de programa\MiKTeX 2.8\miktex\bin
	- C:\Archivos de programa\Graphviz2.26.3\bin
\endverbatim

*/