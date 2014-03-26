/**
	\file Sinogram2Dtgs.cpp
	\brief Archivo que contiene la implementación de la clase Sinogram2Dtgs.

	Este archivo define la clase Sinogram2tgs. 
	Al ser un sinograma de SPECT tiene proyecciones de 0 a 360º. Además le agrego propiedades
	que hacen a este tipo de sinograma, por ejemplo el largo y ancho del colimador, para poder
	obtener el Cone of Response para cada lor. Ahora también se incluye el ancho total del colimador
	para poder modelar la penetración a través del plomo.
	\todo Extenderlo de manera genérico a distintas geometrías.
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.10.01
	\version 1.0.0
*/

#include <Sinogram2Dtgs.h>


Sinogram2Dtgs::Sinogram2Dtgs()
{
  /* Para un sinograma genérico los ángulos de proyecciones van de 0 a 180º. */
  minAng_deg = 0;
  maxAng_deg = 360;
}

Sinogram2Dtgs::Sinogram2Dtgs(unsigned int nProj, unsigned int nR, float rFov_mm, float dCrystalToCenterFov, 
				float lColimator_mm, float wCollimator_mm, float wHoleCollimator_mm)
{
  Sinogram2Dtgs();
  radioFov_mm = rFov_mm;
  numR = nR;
  numProj = nProj;
  distCrystalToCenterFov = dCrystalToCenterFov;
  lengthColimator_mm = lColimator_mm;
  widthCollimator_mm = wCollimator_mm;
  widthHoleCollimator_mm = wHoleCollimator_mm;
  // Allocates Memory for th Sinogram
  ptrSinogram = (float*) malloc(numProj*numR*sizeof(float));
  // Allocates Memory for the value's vectors
  ptrAngValues_deg = (float*) malloc(numProj*sizeof(float));
  ptrRvalues_mm = (float*) malloc(numR*sizeof(float));
  // Initialization
  float RIncrement = (2 * radioFov_mm) / numR;
  float PhiIncrement = (float)maxAng_deg / numProj;
  for(unsigned int i = 0; i < numProj; i ++)
  {
	  // Initialization of Phi Values
	  ptrAngValues_deg[i] = PhiIncrement/2 + i * PhiIncrement;
	  for(unsigned int j = 0; j < numR; j++)
	  {
		  if(i == 0)
		  {
			  // ptrRvalues initialization is necesary just one time
			  
			  ptrRvalues_mm[j] = RIncrement/2 + j * RIncrement - radioFov_mm;
		  }
		  ptrSinogram[i * numR + j] = 0;
	  }
  }
}

/// Constructor de Copia
Sinogram2Dtgs::Sinogram2Dtgs(const Sinogram2Dtgs* srcSinogram2D) : Sinogram2D((Sinogram2D*) srcSinogram2D)
{
  // Sigo con la copia de los parámetros propios de Sinogram2Dtgs:
  this->lengthColimator_mm = srcSinogram2D->lengthColimator_mm;
  this->widthHoleCollimator_mm = srcSinogram2D->widthHoleCollimator_mm;
  this->widthCollimator_mm = srcSinogram2D->widthCollimator_mm;
  this->distCrystalToCenterFov = srcSinogram2D->distCrystalToCenterFov;
}

void Sinogram2Dtgs::setGeometricParameters(float rFov_mm, float dCrystalToCenterFov, float lColimator_mm, float wCollimator_mm, float wHoleCollimator_mm)
{
  this->radioFov_mm = rFov_mm;
  this->distCrystalToCenterFov = dCrystalToCenterFov;
  this->lengthColimator_mm = lColimator_mm;
  this->widthCollimator_mm = wCollimator_mm;
  this->widthHoleCollimator_mm = wHoleCollimator_mm;
  float rStep = (2 * radioFov_mm) / numR;
  for(unsigned int j = 0; j < numR; j++)
  {	  
	ptrRvalues_mm[j] = rStep/2 + j * rStep - radioFov_mm;
  }
}

bool Sinogram2Dtgs::getFovLimits(Line2D lor, Point2D* limitPoint1, Point2D* limitPoint2)
{
  // El FoV es cuadrado para el sinogram2dTgs. Después se le podrían agregar las dos opciones.
  // Las rectas que delimitan el Fov son x=-radioFov_mm x=radioFov_mm y=-radioFov_mm y=radioFov_mm.
  Point2D intersectionPoints[4]; // Voy a evaluar 4 puntos de intersección para MinValueX, pointMaxValueX, pointMinValueY, pointMaxValueY;
  int numGoodPoints = 0;
  int indexLimitPoints[2];	// Índice de los puntos límites dentro de los 4 puntos totales.
  
  intersectionPoints[0].X = -this->radioFov_mm;	// pointMinValueX
  intersectionPoints[1].X = this->radioFov_mm;	// pointMaxValueX
  intersectionPoints[2].Y = -this->radioFov_mm;	// pointMinValueY
  intersectionPoints[3].Y = this->radioFov_mm;	// pointMaxValueY
  
  // La recta de la lor si cruza el fov lo debería hacer en dos puntos, o sea cortando a dos
  // de las 4 rectas que delimitan el fov. Tengo casos particulares, en que no corta ninguna recta dentro de los límites del 
  // fov, o sea no atravieza el FoV y también podría pasar que sea paralela a la misma recta. En ese caso
  // la considero fuera del FoV. (Otra opción sería considerar como los puntos los dos extremos de cada lado.)
  
  // Calculo los alpha de intersección para cada una de las cuatro:
  float alpha_minX = (intersectionPoints[0].X - lor.P0.X) / lor.Vx;
  float alpha_maxX = (intersectionPoints[1].X - lor.P0.X) / lor.Vx;
  float alpha_minY = (intersectionPoints[2].Y - lor.P0.Y) / lor.Vy;
  float alpha_maxY = (intersectionPoints[3].Y - lor.P0.Y) / lor.Vy;
  
  // Ahora obtengo la otra coordenada para cada punto:
  intersectionPoints[0].Y = lor.P0.Y + alpha_minX * lor.Vy;
  intersectionPoints[1].Y = lor.P0.Y + alpha_maxX * lor.Vy;
  intersectionPoints[2].X = lor.P0.X + alpha_minY * lor.Vx;
  intersectionPoints[3].X = lor.P0.X + alpha_maxY * lor.Vx;
  
  // Me fijo si está entre los límites cada variable, y voy contando los puntos válidos.
  for(int i = 0; i < 4; i++)
  {
	// En realidad una de las coordenadas ya se que está bien, pero chequeo las dos, por que cambia para cada punto.
	if((intersectionPoints[i].Y >= -this->radioFov_mm) && (intersectionPoints[i].Y <= this->radioFov_mm))
	{
	  if((intersectionPoints[i].X >= -this->radioFov_mm) && (intersectionPoints[i].X <= this->radioFov_mm))
	  {
		numGoodPoints++;
	  }
	  if(numGoodPoints > 2)
	  {
		// No puede haber más de dos puntos.
		return false;
	  }
	  indexLimitPoints[numGoodPoints-1] = i;
	}
  }
  if(numGoodPoints == 2)
  {
	// Obtuve el resultado correcto:
	(*limitPoint1) = intersectionPoints[indexLimitPoints[0]];
	(*limitPoint2) = intersectionPoints[indexLimitPoints[1]];
	return true;
  }
  else
	return false;
}

float Sinogram2Dtgs::getSegmentLengthInCollimator(float offsetDetector_mm, float offsetCollimatorSurface_mm)
{
  float lengthInMaterial_mm = 0;
  Point2D pointOnDetector, pointOnCollimatorSurf;
  Point2D intersection1, intersection2; // Puntos de posible inersección entre aire y plomo.
  // En principio el largo del segmento es independiente del bin que se procese.
  // Ya que todos los detectores tienen la misma disposición. Considero el (0,0) al
  // centro del detector.
  // Obtengo el punto sobre el detector relativo al centro:
  pointOnDetector.X = offsetDetector_mm;	// Coordenada en x inicial.
  pointOnDetector.Y = 0;	// En Y considero el 0 como la superficie del colimador.
  // El punto sobre el colimador:
  pointOnCollimatorSurf.X = offsetCollimatorSurface_mm;	// Coordenada en x inicial (para thita 0º).
  pointOnCollimatorSurf.Y = this->lengthColimator_mm;	// El y de la superficie del coliamdor es la distancia al cero, o sea al detector.
  
  // Ya tengo definida la recta, por los dos puntos. Tengo que obtener el largo del segmento en
  // la zona del colimador. Eso es sencillo, es la distancia entre los dos puntos. Pero luego debo
  // determinar que parte de ese segmento cruzó por material del colimador y que otra parte por aire.
  // La zona de aire está delimitada por dos rectas paralelas al eje y:
  float rectaX1 = -(this->widthHoleCollimator_mm/2);
  float rectaX2 = (this->widthHoleCollimator_mm/2);
  
  // Tengo 4 casos, la lor cruza solo plomo, plomo y aire, plomo-aire-plomo o solo aire. Cada uno de esos 3 casos
  // se da para distintas configuraciones:
  if (pointOnDetector.X>=rectaX2)
  {
	// La lor toca el detector sobre el plomo de la derecha, puede ser plomo-plomo, plomo-aire, plomo-aire-plomo
	if(pointOnCollimatorSurf.X>=rectaX2)
	{
	  // El punto sobre el colimador está del mismo lado del plomo, o sea estoy en caso plomo-plomo:
	  lengthInMaterial_mm = distBetweenPoints(pointOnCollimatorSurf, pointOnDetector);
	}
	else if(pointOnCollimatorSurf.X>=rectaX1)
	{
	  // Punto en colimador está entre X1 y X2, caso plomo-aire.
	  // Calculo el punto de intersección (conX2) y después la distancia:
	  intersection1.X = rectaX2;
	  intersection1.Y = (pointOnCollimatorSurf.Y - pointOnDetector.Y)/(pointOnCollimatorSurf.X - pointOnDetector.X) *
		(intersection1.X - pointOnDetector.X);
	  // Ahora distancia entre el detector y el punto de intersección.
	  lengthInMaterial_mm = distBetweenPoints(pointOnDetector, intersection1);
	}
	else
	{
	  // Caso plomo-aire-plomo. Tengo dos intersecciones con X1 y X2 respectivamente. La distancia
	  // total será la suma entre distancias detector-X2 y colimador X1.
	  intersection1.X = rectaX1;
	  intersection1.Y = (pointOnCollimatorSurf.Y - pointOnDetector.Y)/(pointOnCollimatorSurf.X - pointOnDetector.X) *
		(intersection1.X - pointOnDetector.X);
	  intersection2.X = rectaX2;
	  intersection2.Y = (pointOnCollimatorSurf.Y - pointOnDetector.Y)/(pointOnCollimatorSurf.X - pointOnDetector.X) *
		(intersection2.X - pointOnDetector.X);
	  lengthInMaterial_mm = distBetweenPoints(pointOnCollimatorSurf, intersection1) + distBetweenPoints(pointOnDetector, intersection2);
	}
  }
  else if (pointOnDetector.X<=rectaX1)
  {
	// La lor toca el detector sobre el plomo de la izquierda, puede ser plomo-plomo, plomo-aire, plomo-aire-plomo.
	if(pointOnCollimatorSurf.X<=rectaX1)
	{
	  // El punto sobre el colimador está del mismo lado del plomo, o sea estoy en caso plomo-plomo:
	  lengthInMaterial_mm = distBetweenPoints(pointOnCollimatorSurf, pointOnDetector);
	}
	else if(pointOnCollimatorSurf.X<=rectaX2)
	{
	  // Punto en colimador está entre X1 y X2, caso plomo-aire.
	  // Calculo el punto de intersección (conX1) y después la distancia:
	  intersection1.X = rectaX1;
	  intersection1.Y = (pointOnCollimatorSurf.Y - pointOnDetector.Y)/(pointOnCollimatorSurf.X - pointOnDetector.X) *
		(intersection1.X - pointOnDetector.X);
	  // Ahora distancia entre el detector y el punto de intersección.
	  lengthInMaterial_mm = distBetweenPoints(pointOnDetector, intersection1);
	}
	else
	{
	  // Caso plomo-aire-plomo. Tengo dos intersecciones con X1 y X2 respectivamente. La distancia
	  // total será la suma entre distancias detector-X2 y colimador X1.
	  intersection1.X = rectaX1;
	  intersection1.Y = (pointOnCollimatorSurf.Y - pointOnDetector.Y)/(pointOnCollimatorSurf.X - pointOnDetector.X) *
		(intersection1.X - pointOnDetector.X);
	  intersection2.X = rectaX2;
	  intersection2.Y = (pointOnCollimatorSurf.Y - pointOnDetector.Y)/(pointOnCollimatorSurf.X - pointOnDetector.X) *
		(intersection2.X - pointOnDetector.X);
	  lengthInMaterial_mm = distBetweenPoints(pointOnDetector, intersection1) + distBetweenPoints(pointOnCollimatorSurf, intersection2);
	}
  }
  //else if((pointOnDetector.X>rectaX1)&&(pointOnDetector.X<rectaX2))
  else
  {
	// La lor toca el detector en aire, por lo que tengo tres caso: aire-aire, aire-plomo(a derecha), aire-plomo(a izquierda)
	if(pointOnCollimatorSurf.X<=rectaX1)
	{
	  // Cruza el plomo a izquierda, obtengo el punto de intersección:
	  intersection1.X = rectaX1;
	  intersection1.Y = (pointOnCollimatorSurf.Y - pointOnDetector.Y)/(pointOnCollimatorSurf.X - pointOnDetector.X) *
		(intersection1.X - pointOnDetector.X);
	  // Ahora distancia entre el plomo y el punto de intersección.
	  lengthInMaterial_mm = distBetweenPoints(pointOnCollimatorSurf, intersection1);
	}
	else if(pointOnCollimatorSurf.X>=rectaX2)
	{
	  // Cruza el plomo a derecha, obtengo el punto de intersección:
	  intersection1.X = rectaX2;
	  intersection1.Y = (pointOnCollimatorSurf.Y - pointOnDetector.Y)/(pointOnCollimatorSurf.X - pointOnDetector.X) *
		(intersection1.X - pointOnDetector.X);
	  // Ahora distancia entre el plomo y el punto de intersección.
	  lengthInMaterial_mm = distBetweenPoints(pointOnCollimatorSurf, intersection1);
	}
	else
	{
	  // Caso de solo aire:
	  lengthInMaterial_mm = 0;
	}
  }
}

Point2D Sinogram2Dtgs::getPointOnDetector(int indexAng, int indexR)
{
  float thita_rad = getAngValue(indexAng) * DEG_TO_RAD;
  float r = getRValue(indexR);
  float x0 = r;	// Coordenada en x inicial (para thita 0º).
  float y0 = -this->distCrystalToCenterFov;	// Coordenada en y inicial, el y donde se encuentra la superficie del detector.
  Point2D pointOnDetectorSurface;
  // Roto el punto para el ángulo de proyección deseado:
  pointOnDetectorSurface.X = x0 * cos(thita_rad) + y0 * sin(thita_rad);
  pointOnDetectorSurface.Y = -x0 * sin(thita_rad) + y0 * cos(thita_rad);
  return pointOnDetectorSurface;
}

Point2D Sinogram2Dtgs::getPointOnDetector(int indexAng, int indexR, float offsetDetector_mm)
{
  float thita_rad = getAngValue(indexAng) * DEG_TO_RAD;
  float r = getRValue(indexR);
  float x0 = r + offsetDetector_mm;	// Coordenada en x inicial (para thita 0º).
  float y0 = -this->distCrystalToCenterFov;	// Coordenada en y inicial, el y donde se encuentra la superficie del detector.
  Point2D pointOnDetectorSurface;
  // Roto el punto para el ángulo de proyección deseado:
  pointOnDetectorSurface.X = x0 * cos(thita_rad) + y0 * sin(thita_rad);
  pointOnDetectorSurface.Y = -x0 * sin(thita_rad) + y0 * cos(thita_rad);
  return pointOnDetectorSurface;
}

Point2D Sinogram2Dtgs::getPointOnCollimatorSurface(int indexAng, int indexR, float offsetCollimator_mm)
{
  float thita_rad = getAngValue(indexAng) * DEG_TO_RAD;
  float r = getRValue(indexR);
  float x0 = r + offsetCollimator_mm;	// Coordenada en x inicial (para thita 0º).
  float y0 = -(this->distCrystalToCenterFov - this->lengthColimator_mm);	// Coordenada en y inicial, el y donde se encuentra la superficie del detector.
  Point2D pointOnCollimatorSurface;
  // Roto el punto para el ángulo de proyección deseado:
  pointOnCollimatorSurface.X = x0 * cos(thita_rad) + y0 * sin(thita_rad);
  pointOnCollimatorSurface.Y = -x0 * sin(thita_rad) + y0 * cos(thita_rad);
  return pointOnCollimatorSurface;
}

/// Coordenadas de una lor del TGS, que tiene en cuenta que para cada colimador puede haber LORs
/// oblicuas. Debe pasarsele como dato la distancia del centro del detector al punto sobre la superficie
/// del detector que toca la lor; y la misma distancia pero sobre la cara exterior del agujero del colimador.
/// Esas dos distancias permiten obtener la inclinación de la LOR.
void Sinogram2Dtgs::getPointsFromTgsLor (int indexAng, int indexR, float offsetDetector_mm, float offsetCollimator_mm, Point2D* p1, Point2D* p2)
{
  float thita_rad = getAngValue(indexAng) * DEG_TO_RAD;
  float r = getRValue(indexR);
  /// Para lor lor debo obtener un punto sobre el detector, y un punto en el extremo opuesto.
  /// Punto sobre el detector:
  double x0 = r + offsetDetector_mm;	/// Posición en X del punto sobre el detector: r + offsetDetector.
  // En el sistema final es y0 = -this->distCrystalToCenterFov, tenerlo en cuenta porque algunas simulaciones no lo tienen así, y por eso
  // por ahora lo estoy poniendo en +this->distCrystalToCenterFov
  double y0 = this->distCrystalToCenterFov;	/// Coordenada Y sobre el detector.
  /// Punto en cara opuesta. La coordenada Y es la misma pero con signo opuesto, mientras que para la X
  /// la debo proyectar en base a la recta que forma entre (r+OffsetDetector) y (r+OffsetCaraColimador):
  double y1 = -this->distCrystalToCenterFov;
  double x1 = x0 + (offsetCollimator_mm-offsetDetector_mm) / this->lengthColimator_mm * this->distCrystalToCenterFov * 2;
  
  p1->X = x0 * cos(thita_rad) + y0 * sin(thita_rad);
  p1->Y = -x0 * sin(thita_rad) + y0 * cos(thita_rad);
  p2->X = x1 * cos(thita_rad) + y1 * sin(thita_rad);
  p2->Y = -x1 * sin(thita_rad) + y1 * cos(thita_rad);
}

