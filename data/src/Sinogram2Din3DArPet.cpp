/**
	\file Sinogram2Din3DArPet.cpp
	\brief Archivo que contiene la implementación de la clase Sinogram2Din3DArPet.

	Este archivo define la clase Sinogram2Din3DArPet. La misma define un sinograma de dos dimensiones
	que se encuentra en el scanner ArPet que realiza adquisiciones 3D, por lo que tiene información sobre
	la combinación de anillos a la que pertenece y si son más de una porque hay polar mashing.
	\todo Implementar proyectores.
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.10.01
	\version 1.0.0
*/

#include <Sinogram2Din3DArPet.h>
#include <Geometry.h>

const int Sinogram2Din3DArPet::numCabezales = 6;

/** Largo del detector en y. */
const float Sinogram2Din3DArPet::lengthDetectorY_mm = 304.6f;

/** Largo del detector en x.*/
const float Sinogram2Din3DArPet::lengthDetectorX_mm = 406.8f;

/** Espesor del detector, */
const float Sinogram2Din3DArPet::depthDetector_mm = 25.4f;

/** Linear attenuation coef for collimator. */
const float Sinogram2Din3DArPet::linearAttenuationCoef_1_cm = 0.34f;

/** Distancia del centro del FOV a la superficie de cada detector. */
const float Sinogram2Din3DArPet::distCenterToDetector_mm = 360.0f;	 
	

Sinogram2Din3DArPet::Sinogram2Din3DArPet(char* fileHeaderPath, float rFov_mm)
{
  radioFov_mm = rFov_mm;
  this->readFromInterfile(fileHeaderPath);
  // Por defecto sin zona ciega y que se admitan todas las coincidencias.
  lengthFromBorderBlindArea_mm = 0;
  minDiffDetectors = 1;
}

Sinogram2Din3DArPet::Sinogram2Din3DArPet(unsigned int nProj, unsigned int nR, float rFov_mm):Sinogram2DinCylindrical3Dpet(nProj, nR, rFov_mm, Sinogram2Din3DArPet::distCenterToDetector_mm)
{
  // Por defecto sin zona ciega y que se admitan todas las coincidencias.
  lengthFromBorderBlindArea_mm = 0;
  minDiffDetectors = 1;
}

Sinogram2Din3DArPet::~Sinogram2Din3DArPet()
{
  /// Limpio la memoria
  
}

/// Constructor de Copia
Sinogram2Din3DArPet::Sinogram2Din3DArPet(const Sinogram2Din3DArPet* srcSinogram2D):Sinogram2DinCylindrical3Dpet((Sinogram2DinCylindrical3Dpet*) srcSinogram2D)
{
  // La configuración del sinograma 3d la hace el constructor de Sinogram2DinCylindrical3Dpet
  lengthFromBorderBlindArea_mm = srcSinogram2D->lengthFromBorderBlindArea_mm;
  minDiffDetectors = srcSinogram2D->minDiffDetectors;
}

/// Constructor de Copia desde Sinogram2DinCylindrical3Dpet
Sinogram2Din3DArPet::Sinogram2Din3DArPet(const Sinogram2DinCylindrical3Dpet* srcSinogram2D):Sinogram2DinCylindrical3Dpet((Sinogram2DinCylindrical3Dpet*) srcSinogram2D)
{
  // La configuración del sinograma 3d la hace el constructor de Sinogram2DinCylindrical3Dpet
  lengthFromBorderBlindArea_mm = 0;
  minDiffDetectors = 1;
}

Sinogram2Din3DArPet::Sinogram2Din3DArPet(const Sinogram2Din3DArPet* srcSinogram2D, int indexSubset, int numSubsets):Sinogram2DinCylindrical3Dpet((Sinogram2DinCylindrical3Dpet*) srcSinogram2D, indexSubset, numSubsets)
{
  // La configuración del sinograma 3d la hace en el constructor de Sinogram2DinCylindrical3Dpet
  lengthFromBorderBlindArea_mm = 0;
  minDiffDetectors = srcSinogram2D->minDiffDetectors;
}

bool Sinogram2Din3DArPet::getPointsFromLor(int indexProj, int indexR, Point2D* p1, Point2D* p2)
{
  // A partir del ángulo, la distancia r, y las coordenadas de los anillos debo obtener ammbos puntos.
 // Voy viendo a que cabezal pertence. Para ello calculo la intersección con las seis rectas de los
 // detectores y me fijo si da en el segmento del detector.
 // La recta es r = x.cos(theta)+y.sin(theta)
 float x,y;
 int numPoints = 0;
 // Paso el ángulo de la proyección a radianes:
 float theta_rad = this->getAngValue(indexProj) * DEG_TO_RAD;
 // Corte en y de las rectas de los cabezales (o sea del hexágono). Sale de rotar el punto medio del
 // detector en 60º, que quedaría en (-distCenterToDetector_mm*cos60,distCenterToDetector_mm*sen60)
 // Reemplazano en y = tan30x +b, se obtiene la ordenada al origen:
 float yHexagonoCabezal_mm = (distCenterToDetector_mm+ depthDetector_mm/2) / sin(60*DEG_TO_RAD);
 // Límites de valores en y para los cabezales oblicuos (los obtengo rotando los puntos extremos del cabezal en x=dist):
 float limiteSuperiorY = -(-distCenterToDetector_mm)*sin(60*DEG_TO_RAD) + (lengthDetectorX_mm/2)*cos(60*DEG_TO_RAD) + depthDetector_mm/2*cos(30*DEG_TO_RAD);
 float limiteInferiorY = -(-distCenterToDetector_mm)*sin(60*DEG_TO_RAD) + (-lengthDetectorX_mm/2)*cos(60*DEG_TO_RAD) + depthDetector_mm/2*cos(30*DEG_TO_RAD);
 // Primer cabezal x = -Sinogram2Din3DArPet::distCenterToDetector_mm- depthDetector.
 x = -Sinogram2Din3DArPet::distCenterToDetector_mm - depthDetector_mm/2;
 y = (this->getRValue(indexR) - x*cos(theta_rad)) / sin(theta_rad);
 // Verifico que el y esté dentro de los valores del cabezal:
 if((y >= -Sinogram2Din3DArPet::lengthDetectorX_mm/2) && (y <= Sinogram2Din3DArPet::lengthDetectorX_mm/2))
 {
   // Es un punto en el detector. Todo está calculado sobre la superficie del mismo, si quisiera proyectar profundidad: + depthDetector_mm/2*cos(30*DEG_TO_RAD) para los detectores oblicuso.
   p1->X = x;
   p1->Y = y;
   numPoints++;
 }
 
 // Segundo cabezal:
 // La recta sería y=tan30*x + yHexagonoCabezal_mm
 x = (this->getRValue(indexR) - yHexagonoCabezal_mm*sin(theta_rad)) / (tan(30*DEG_TO_RAD)*sin(theta_rad)+cos(theta_rad));
 y = (this->getRValue(indexR) - x*cos(theta_rad)) / sin(theta_rad);
 // Verifico que el y esté dentro de los valores del cabezal:
 if((y <= limiteSuperiorY) && (y > limiteInferiorY))
 {
   // Es un punto en el detector:
   if(numPoints == 0)
   {
    p1->X = x;
    p1->Y = y;
   }
   else
   {
     p2->X = x;
     p2->Y = y;
   }
   numPoints++;
 }
 
 // Tercer cabezal:
 // La recta sería y=tan-30*x + yHexagonoCabezal_mm
 x = (this->getRValue(indexR) - yHexagonoCabezal_mm*sin(theta_rad)) / (tan(-30*DEG_TO_RAD)*sin(theta_rad)+cos(theta_rad));
 y = (this->getRValue(indexR) - x*cos(theta_rad)) / sin(theta_rad);
 // Verifico que el y esté dentro de los valores del cabezal:
 if((y <= limiteSuperiorY) && (y > limiteInferiorY))
 {
   // Es un punto en el detector:
   if(numPoints == 0)
   {
    p1->X = x;
    p1->Y = y;
   }
   else
   {
     p2->X = x;
     p2->Y = y;
   }
   numPoints++;
 }

  // Cuarto cabezal x = distCenterToDetector_mm depthDetector.
 x = Sinogram2Din3DArPet::distCenterToDetector_mm + depthDetector_mm/2;
 y = (this->getRValue(indexR) - x*cos(theta_rad)) / sin(theta_rad);
 // Verifico que el y esté dentro de los valores del cabezal:
 if((y > -Sinogram2Din3DArPet::lengthDetectorX_mm/2) && (y < Sinogram2Din3DArPet::lengthDetectorX_mm/2))
 {
   if(numPoints == 0)
   {
    p1->X = x;
    p1->Y = y;
   }
   else
   {
     p2->X = x;
     p2->Y = y;
   }
   numPoints++;
 }
 
 // La recta sería y=tan30*x - yHexagonoCabezal_mm
 x = (this->getRValue(indexR) + yHexagonoCabezal_mm*sin(theta_rad)) / (tan(30*DEG_TO_RAD)*sin(theta_rad)+cos(theta_rad));
 y = (this->getRValue(indexR) - x*cos(theta_rad)) / sin(theta_rad);
 // Verifico que el y esté dentro de los valores del cabezal:
 if((y >= -limiteSuperiorY) && (y <= -limiteInferiorY))
 {
   // Es un punto en el detector:
   if(numPoints == 0)
   {
    p1->X = x;
    p1->Y = y;
   }
   else
   {
     p2->X = x;
     p2->Y = y;
   }
   numPoints++;
 }
 
 // Sexto cabezal:
 // La recta sería y=tan-30*x - yHexagonoCabezal_mm
 x = (this->getRValue(indexR) + yHexagonoCabezal_mm*sin(theta_rad)) / (tan(-30*DEG_TO_RAD)*sin(theta_rad)+cos(theta_rad));
 y = (this->getRValue(indexR) - x*cos(theta_rad)) / sin(theta_rad);
 // Verifico que el y esté dentro de los valores del cabezal:
 if((y > -limiteSuperiorY) && (y < -limiteInferiorY))
 {
   // Es un punto en el detector:
   if(numPoints == 0)
   {
    p1->X = x;
    p1->Y = y;
   }
   else
   {
     p2->X = x;
     p2->Y = y;
   }
   numPoints++;
 }
 
 if(numPoints == 2)
   return true;
 else return false;
}

bool Sinogram2Din3DArPet::getPointsFromLor(int indexProj, int indexR, int indexRingConfig, Point3D* p1, Point3D* p2, float* geomFactor)
{
 // A partir del ángulo, la distancia r, y las coordenadas de los anillos debo obtener ammbos puntos.
 // Voy viendo a que cabezal pertence. Para ello calculo la intersección con las seis rectas de los
 // detectores y me fijo si da en el segmento del detector.
 // La recta es r = x.cos(theta)+y.sin(theta)
 float x,y;
 int numPoints = 0;
 // Paso el ángulo de la proyección a radianes:
 float theta_rad = this->getAngValue(indexProj) * DEG_TO_RAD;
 // También uso los indices de cabezal para determinar el peso, y también puedo definir de esa forma si no se utiliza
 // cabezales contiguos:
 int indexCabezal1 = 0, indexCabezal2 = 0;
 // Además tengo un peso por la probabilidad de detección, que por defecto es la de un rayo perpendicular añ detector:
 float lengthInDetector1_cm = depthDetector_mm/10;
 float lengthInDetector2_cm = depthDetector_mm/10;
 float probDetCabezal1;
 float probDetCabezal2;
 // Largo en z de la lor en el detector, no depende del cabezal ni nada solo de z1 y z2 (verificar).
 float lengthZinDetector1_cm, lengthZinDetector2_cm;
 // Utilizo un valor promedio para la profundidad del detector:
 float zDetector_mm = depthDetector_mm*2/3;
 // Corte en y de las rectas de los cabezales (o sea del hexágono). Sale de rotar el punto medio del
 // detector en 60º, que quedaría en (-distCenterToDetector_mm*cos60,distCenterToDetector_mm*sen60)
 // Antes lo hacia sobre el medio del cabezal depthDetector_mm, pero ahora lo hago sobre la cara superior e inferior
 // para tener la sensibilidad correcta en los bordes donde se afina el espesor del detector.
 // Reemplazano en y = tan30x +b, se obtiene la ordenada al origen:
 float yHexagonoCabezal_mm = (distCenterToDetector_mm + zDetector_mm) / sin(60*DEG_TO_RAD); // Antes (distCenterToDetector_mm+ depthDetector_mm/2)/ sin(60*DEG_TO_RAD)
 // Límites de valores en y para los cabezales oblicuos (los obtengo rotando los puntos extremos del cabezal en x=dist):
 float limiteSuperiorY = -(-distCenterToDetector_mm)*sin(60*DEG_TO_RAD) + (lengthDetectorX_mm/2-lengthFromBorderBlindArea_mm)*cos(60*DEG_TO_RAD) + zDetector_mm*cos(30*DEG_TO_RAD);
 float limiteInferiorY = -(-distCenterToDetector_mm)*sin(60*DEG_TO_RAD) + (-lengthDetectorX_mm/2+lengthFromBorderBlindArea_mm)*cos(60*DEG_TO_RAD) + zDetector_mm*cos(30*DEG_TO_RAD);
 
 //printf("LimInfY:%f\tLimSupY:%f\n", limiteInferiorY, limiteSuperiorY);
 // Primer cabezal x = -Sinogram2Din3DArPet::distCenterToDetector_mm- depthDetector.
 x = -Sinogram2Din3DArPet::distCenterToDetector_mm - zDetector_mm;
 y = (this->getRValue(indexR) - x*cos(theta_rad)) / sin(theta_rad);
 // Verifico que el y esté dentro de los valores del cabezal:
 if((y >= -Sinogram2Din3DArPet::lengthDetectorX_mm/2+lengthFromBorderBlindArea_mm) && (y <= Sinogram2Din3DArPet::lengthDetectorX_mm/2-lengthFromBorderBlindArea_mm))
 {
   // Es un punto en el detector. Todo está calculado sobre la superficie del mismo, si quisiera proyectar profundidad: + depthDetector_mm/2*cos(30*DEG_TO_RAD) para los detectores oblicuso.
   p1->X = x;
   p1->Y = y;
   p1->Z = this->getAxialValue1FromList(indexRingConfig);
   numPoints++;
   indexCabezal1 = 0;
   // El largo en el detector lo puedo obtener de dos formas, a partir del ángulo de incidencia:
   lengthInDetector1_cm =depthDetector_mm/10/cos(fabs(theta_rad-90*DEG_TO_RAD));
 }
 
 // Segundo cabezal:
 // La recta sería y=tan30*x + yHexagonoCabezal_mm
 x = (this->getRValue(indexR) - yHexagonoCabezal_mm*sin(theta_rad)) / (tan(30*DEG_TO_RAD)*sin(theta_rad)+cos(theta_rad));
 y = (this->getRValue(indexR) - x*cos(theta_rad)) / sin(theta_rad);
 // Verifico que el y esté dentro de los valores del cabezal:
 if((y <= limiteSuperiorY) && (y > limiteInferiorY))
 {
   // Es un punto en el detector:
   if(numPoints == 0)
   {
    p1->X = x;
    p1->Y = y;
    p1->Z = this->getAxialValue1FromList(indexRingConfig);
    indexCabezal1 = 1;
    lengthInDetector1_cm =depthDetector_mm/10/fabs(cos(30*DEG_TO_RAD-theta_rad));
    //lengthInDetector1_cm = sqrt((x_atras-x)*(x_atras-x)+(y_atras-y)*(y_atras-y))/10;
   }
   else
   {
     p2->X = x;
     p2->Y = y;
     p2->Z = this->getAxialValue2FromList(indexRingConfig);
     indexCabezal2 = 1;
     lengthInDetector2_cm =depthDetector_mm/10/fabs(cos(30*DEG_TO_RAD-theta_rad));
   }
   numPoints++;
 }
 
 // Tercer cabezal:
 // La recta sería y=tan-30*x + yHexagonoCabezal_mm
 x = (this->getRValue(indexR) - yHexagonoCabezal_mm*sin(theta_rad)) / (tan(-30*DEG_TO_RAD)*sin(theta_rad)+cos(theta_rad));
 y = (this->getRValue(indexR) - x*cos(theta_rad)) / sin(theta_rad);
 // Verifico que el y esté dentro de los valores del cabezal:
 if((y <= limiteSuperiorY) && (y > limiteInferiorY))
 {
   if(numPoints == 0)
   {
    p1->X = x;
    p1->Y = y;
    p1->Z = this->getAxialValue1FromList(indexRingConfig);
    indexCabezal1 = 2;
    lengthInDetector1_cm =depthDetector_mm/10/fabs(cos(30*DEG_TO_RAD+theta_rad));
    //lengthInDetector1_cm = sqrt((x_atras-x)*(x_atras-x)+(y_atras-y)*(y_atras-y))/10;
   }
   else
   {
     p2->X = x;
     p2->Y = y;
     p2->Z = this->getAxialValue2FromList(indexRingConfig);
     indexCabezal2 = 2;
     lengthInDetector2_cm =depthDetector_mm/10/fabs(cos(30*DEG_TO_RAD+theta_rad));
     //lengthInDetector2_cm = sqrt((x_atras-x)*(x_atras-x)+(y_atras-y)*(y_atras-y))/10;
   }
   numPoints++;
 }

  // Cuarto cabezal x = distCenterToDetector_mm depthDetector.
 x = Sinogram2Din3DArPet::distCenterToDetector_mm + depthDetector_mm*2/3;
 y = (this->getRValue(indexR) - x*cos(theta_rad)) / sin(theta_rad);
 // Verifico que el y esté dentro de los valores del cabezal:
 if((y > -Sinogram2Din3DArPet::lengthDetectorX_mm/2+lengthFromBorderBlindArea_mm) && (y < Sinogram2Din3DArPet::lengthDetectorX_mm/2-lengthFromBorderBlindArea_mm))
 {
   if(numPoints == 0)
   {
    p1->X = x;
    p1->Y = y;
    p1->Z = this->getAxialValue1FromList(indexRingConfig);
    indexCabezal1 = 3;
    lengthInDetector1_cm =depthDetector_mm/10/fabs(sin(theta_rad));
  }
   else
   {
     p2->X = x;
     p2->Y = y;
     p2->Z = this->getAxialValue2FromList(indexRingConfig);
     indexCabezal2 = 3;
     lengthInDetector2_cm =depthDetector_mm/10/fabs(sin(theta_rad));
   }
   numPoints++;
 }
 
 // La recta sería y=tan30*x - yHexagonoCabezal_mm
 x = (this->getRValue(indexR) + yHexagonoCabezal_mm*sin(theta_rad)) / (tan(30*DEG_TO_RAD)*sin(theta_rad)+cos(theta_rad));
 y = (this->getRValue(indexR) - x*cos(theta_rad)) / sin(theta_rad);
 // Verifico que el y esté dentro de los valores del cabezal:
 if((y >= -limiteSuperiorY) && (y <= -limiteInferiorY))
 {
   if(numPoints == 0)
   {
    p1->X = x;
    p1->Y = y;
    p1->Z = this->getAxialValue1FromList(indexRingConfig);
    indexCabezal1 = 4;
    lengthInDetector1_cm =depthDetector_mm/10/fabs(cos(30*DEG_TO_RAD-theta_rad));
    //lengthInDetector1_cm = sqrt((x_atras-x)*(x_atras-x)+(y_atras-y)*(y_atras-y))/10;
   }
   else
   {
     p2->X = x;
     p2->Y = y;
     p2->Z = this->getAxialValue2FromList(indexRingConfig);
     indexCabezal2 = 4;
     lengthInDetector2_cm =depthDetector_mm/10/fabs(cos(30*DEG_TO_RAD-theta_rad));
     //lengthInDetector2_cm = sqrt((x_atras-x)*(x_atras-x)+(y_atras-y)*(y_atras-y))/10;
   }
   numPoints++;
 }
 
 // Sexto cabezal:
 // La recta sería y=tan-30*x - yHexagonoCabezal_mm
 x = (this->getRValue(indexR) + yHexagonoCabezal_mm*sin(theta_rad)) / (tan(-30*DEG_TO_RAD)*sin(theta_rad)+cos(theta_rad));
 y = (this->getRValue(indexR) - x*cos(theta_rad)) / sin(theta_rad);
 // Verifico que el y esté dentro de los valores del cabezal:
 if((y > -limiteSuperiorY) && (y < -limiteInferiorY))
 {
   // Es un punto en el detector:
   if(numPoints == 0)
   {
    p1->X = x;
    p1->Y = y;
    p1->Z = this->getAxialValue1FromList(indexRingConfig);
    indexCabezal1 = 5;
    lengthInDetector1_cm =depthDetector_mm/10/fabs(cos(30*DEG_TO_RAD+theta_rad));
   }
   else
   {
     p2->X = x;
     p2->Y = y;
     p2->Z = this->getAxialValue2FromList(indexRingConfig);
     indexCabezal2 = 5;
     lengthInDetector2_cm =depthDetector_mm/10/fabs(cos(30*DEG_TO_RAD+theta_rad));
   }
   numPoints++;
 }

 // Si la cantidad de detectores intersectados es distinto de 2, retorno false:
 if(numPoints != 2)
   return false;

 // Los length en z dependen de la distancia entre p1 y p2 en el plano XY y de los respectivos largos en el detector:
 float distXY = sqrt((p2->X-p1->X)*(p2->X-p1->X)+(p2->Y-p1->Y)*(p2->Y-p1->Y));
 lengthZinDetector1_cm = fabs(p2->Z-p1->Z)/distXY*lengthInDetector1_cm;
 lengthZinDetector2_cm = fabs(p2->Z-p1->Z)/distXY*lengthInDetector2_cm;
 // Obtengo los largos totales en el detector y luego las probabilidadesd de detección:
 lengthInDetector1_cm = sqrt(lengthInDetector1_cm*lengthInDetector1_cm+lengthZinDetector1_cm*lengthZinDetector1_cm);
 lengthInDetector2_cm = sqrt(lengthInDetector2_cm*lengthInDetector2_cm+lengthZinDetector2_cm*lengthZinDetector2_cm);
 // Probabilidad de detección según largo de detección:
 probDetCabezal1 = 1-exp(-lengthInDetector1_cm*linearAttenuationCoef_1_cm);
 probDetCabezal2 = 1-exp(-lengthInDetector2_cm*linearAttenuationCoef_1_cm);
 //printf("theta:%f\tcabezal1:%d\tcabezal1:%d\t largoZ:%f\tlargo1:%f\tlargo2:%f\tprobDet1:%f\tprobDet2:%f\n", theta_rad/DEG_TO_RAD,indexCabezal1, indexCabezal2, lengthZinDetector_cm, lengthInDetector2_cm, probDetCabezal1, probDetCabezal2);
 // Peso geométrico:
 // *geomFactor = (sensCabezal1+sensCabezal2)/2; // Lors a ángulo inclinado, el ancho de la lor es el seno del ángulo que se forma.
 // Lo de la sensibilidad sobre el detector al final no va
 *geomFactor = probDetCabezal1*probDetCabezal2;//(sensCabezal1+sensCabezal2)/2 * probDetCabezal1*probDetCabezal2; // Lors a ángulo inclinado, el ancho de la lor es el seno del ángulo que se forma.
 //*geomFactor = 1; // Esto sería sin arc correction.
 // Los cabezales contiguos pueden estar configurados para no adquirir coincidencias. Esto se hace con el parámetro
 // minDiffDetectors. En caso de que se esté en esos casos, fuerzo a cero el factor geométrico:
 if((abs(indexCabezal2-indexCabezal1) < minDiffDetectors) || (abs(indexCabezal2-indexCabezal1) > (this->numCabezales-minDiffDetectors))) // La segunda condición es para los cabezales 5 y 0, que estánc contiguos pero la diferencia es 5. 
   *geomFactor = 0;
 return true;
	
}

bool Sinogram2Din3DArPet::getPointsFromLor (int indexProj, int indexR, Point2D* p1, Point2D* p2, float* geomFactor)
{
    // A partir del ángulo, la distancia r, y las coordenadas de los anillos debo obtener ammbos puntos.
 // Voy viendo a que cabezal pertence. Para ello calculo la intersección con las seis rectas de los
 // detectores y me fijo si da en el segmento del detector.
 // La recta es r = x.cos(theta)+y.sin(theta)
 float x, y;
 int numPoints = 0;
 // Paso el ángulo de la proyección a radianes:
 float theta_rad = this->getAngValue(indexProj) * DEG_TO_RAD;
 // También uso los indices de cabezal para determinar el peso, y también puedo definir de esa forma si no se utiliza
 // cabezales contiguos:
 int indexCabezal1 = 0, indexCabezal2 = 0;
  // Además tengo un peso por la probabilidad de detección, que por defecto es la de un rayo perpendicular añ detector:
 float lengthInDetector1_cm = depthDetector_mm/10;
 float lengthInDetector2_cm = depthDetector_mm/10;
 float probDetCabezal1;
 float probDetCabezal2;
 // Utilizo un valor promedio para la profundidad del detector:
 float zDetector_mm = depthDetector_mm*2/3;
 // Corte en y de las rectas de los cabezales (o sea del hexágono). Sale de rotar el punto medio del
 // detector en 60º, que quedaría en (-distCenterToDetector_mm*cos60,distCenterToDetector_mm*sen60)
 // Reemplazano en y = tan30x +b, se obtiene la ordenada al origen:
 float yHexagonoCabezal_mm = (distCenterToDetector_mm+ zDetector_mm) / sin(60*DEG_TO_RAD);
 // Límites de valores en y para los cabezales oblicuos (los obtengo rotando los puntos extremos del cabezal en x=dist):
 float limiteSuperiorY = -(-distCenterToDetector_mm)*sin(60*DEG_TO_RAD) + (lengthDetectorX_mm/2-lengthFromBorderBlindArea_mm)*cos(60*DEG_TO_RAD) + zDetector_mm*cos(30*DEG_TO_RAD);
 float limiteInferiorY = -(-distCenterToDetector_mm)*sin(60*DEG_TO_RAD) + (-(lengthDetectorX_mm/2-lengthFromBorderBlindArea_mm))*cos(60*DEG_TO_RAD) + zDetector_mm*cos(30*DEG_TO_RAD);
 // Primer cabezal x = -Sinogram2Din3DArPet::distCenterToDetector_mm- depthDetector.
 x = -Sinogram2Din3DArPet::distCenterToDetector_mm - zDetector_mm;
 y = (this->getRValue(indexR) - x*cos(theta_rad)) / sin(theta_rad);
 // Verifico que el y esté dentro de los valores del cabezal:
 if((y >= -Sinogram2Din3DArPet::lengthDetectorX_mm/2+lengthFromBorderBlindArea_mm) && (y <= Sinogram2Din3DArPet::lengthDetectorX_mm/2-lengthFromBorderBlindArea_mm))
 {
   // Es un punto en el detector. Todo está calculado sobre la superficie del mismo, si quisiera proyectar profundidad: + depthDetector_mm/2*cos(30*DEG_TO_RAD) para los detectores oblicuso.
   p1->X = x;
   p1->Y = y;
   numPoints++;
   indexCabezal1 = 0;
   // El largo en el detector lo puedo obtener de dos formas, a partir del ángulo de incidencia:
   lengthInDetector1_cm =depthDetector_mm/10/cos(fabs(theta_rad-90*DEG_TO_RAD));
 }
 
 // Segundo cabezal:
 // La recta sería y=tan30*x + yHexagonoCabezal_mm
 x = (this->getRValue(indexR) - yHexagonoCabezal_mm*sin(theta_rad)) / (tan(30*DEG_TO_RAD)*sin(theta_rad)+cos(theta_rad));
 y = (this->getRValue(indexR) - x*cos(theta_rad)) / sin(theta_rad);
 // Verifico que el y esté dentro de los valores del cabezal:
 if((y <= limiteSuperiorY) && (y > limiteInferiorY))
 {
   // Es un punto en el detector:
   if(numPoints == 0)
   {
    p1->X = x;
    p1->Y = y;
    indexCabezal1 = 1;
    // El largo en el detector lo puedo obtener de dos formas, a partir del ángulo de incidencia:
    lengthInDetector1_cm =depthDetector_mm/10/fabs(cos(30*DEG_TO_RAD-theta_rad));
   }
   else
   {
     p2->X = x;
     p2->Y = y;
     indexCabezal2 = 1;
     lengthInDetector2_cm =depthDetector_mm/10/fabs(cos(30*DEG_TO_RAD-theta_rad));
   }
   numPoints++;
 }
 
 // Tercer cabezal:
 // La recta sería y=tan-30*x + yHexagonoCabezal_mm
 x = (this->getRValue(indexR) - yHexagonoCabezal_mm*sin(theta_rad)) / (tan(-30*DEG_TO_RAD)*sin(theta_rad)+cos(theta_rad));
 y = (this->getRValue(indexR) - x*cos(theta_rad)) / sin(theta_rad);
 // Verifico que el y esté dentro de los valores del cabezal:
 if((y <= limiteSuperiorY) && (y > limiteInferiorY))
 {
   // Es un punto en el detector:
   if(numPoints == 0)
   {
    p1->X = x;
    p1->Y = y;
    indexCabezal1 = 2;
    lengthInDetector1_cm =depthDetector_mm/10/fabs(cos(30*DEG_TO_RAD+theta_rad));
   }
   else
   {
     p2->X = x;
     p2->Y = y;
     indexCabezal2 = 2;
     lengthInDetector2_cm =depthDetector_mm/10/fabs(cos(30*DEG_TO_RAD+theta_rad));
   }
   numPoints++;
 }

  // Cuarto cabezal x = distCenterToDetector_mm depthDetector.
 x = Sinogram2Din3DArPet::distCenterToDetector_mm + depthDetector_mm/2;
 y = (this->getRValue(indexR) - x*cos(theta_rad)) / sin(theta_rad);
 // Verifico que el y esté dentro de los valores del cabezal:
 if((y > -Sinogram2Din3DArPet::lengthDetectorX_mm/2+lengthFromBorderBlindArea_mm) && (y < Sinogram2Din3DArPet::lengthDetectorX_mm/2-lengthFromBorderBlindArea_mm))
 {
   if(numPoints == 0)
   {
    p1->X = x;
    p1->Y = y;
    indexCabezal1 = 3;
    lengthInDetector1_cm =depthDetector_mm/10/fabs(sin(theta_rad));    
  }
   else
   {
     p2->X = x;
     p2->Y = y;
     indexCabezal2 = 3;
     lengthInDetector2_cm =depthDetector_mm/10/fabs(sin(theta_rad));
   }
   numPoints++;
 }
 
 // La recta sería y=tan30*x - yHexagonoCabezal_mm
 x = (this->getRValue(indexR) + yHexagonoCabezal_mm*sin(theta_rad)) / (tan(30*DEG_TO_RAD)*sin(theta_rad)+cos(theta_rad));
 y = (this->getRValue(indexR) - x*cos(theta_rad)) / sin(theta_rad);
 // Verifico que el y esté dentro de los valores del cabezal:
 if((y >= -limiteSuperiorY) && (y <= -limiteInferiorY))
 {
   // Es un punto en el detector:
   if(numPoints == 0)
   {
    p1->X = x;
    p1->Y = y;
    indexCabezal1 = 4;
    lengthInDetector1_cm =depthDetector_mm/10/fabs(cos(30*DEG_TO_RAD-theta_rad));
   }
   else
   {
     p2->X = x;
     p2->Y = y;
     indexCabezal2 = 4;
     lengthInDetector2_cm =depthDetector_mm/10/fabs(cos(30*DEG_TO_RAD-theta_rad));
   }
   numPoints++;
 }
 
 // Sexto cabezal:
 // La recta sería y=tan-30*x - yHexagonoCabezal_mm
 x = (this->getRValue(indexR) + yHexagonoCabezal_mm*sin(theta_rad)) / (tan(-30*DEG_TO_RAD)*sin(theta_rad)+cos(theta_rad));
 y = (this->getRValue(indexR) - x*cos(theta_rad)) / sin(theta_rad);
 // Verifico que el y esté dentro de los valores del cabezal:
 if((y > -limiteSuperiorY) && (y < -limiteInferiorY))
 {
   // Es un punto en el detector:
   if(numPoints == 0)
   {
    p1->X = x;
    p1->Y = y;
    indexCabezal1 = 5;
    lengthInDetector1_cm =depthDetector_mm/10/fabs(cos(30*DEG_TO_RAD+theta_rad));
   }
   else
   {
     p2->X = x;
     p2->Y = y;
     indexCabezal2 = 5;
     lengthInDetector2_cm =depthDetector_mm/10/fabs(cos(30*DEG_TO_RAD+theta_rad));
   }
   numPoints++;
 }
  if(numPoints != 2)
   return false;
 // En el caso 3d calculaba el largo de la LOR axial, ahora no es necesario porque es 2d.
 
 // Probabilidad de detección según largo de detección:
 probDetCabezal1 = 1-exp(-lengthInDetector1_cm*linearAttenuationCoef_1_cm);
 probDetCabezal2 = 1-exp(-lengthInDetector2_cm*linearAttenuationCoef_1_cm);
 //printf("theta:%f\tcabezal1:%d\tcabezal1:%d\t largoZ:%f\tlargo1:%f\tlargo2:%f\tprobDet1:%f\tprobDet2:%f\n", theta_rad/DEG_TO_RAD,indexCabezal1, indexCabezal2, lengthZinDetector_cm, lengthInDetector2_cm, probDetCabezal1, probDetCabezal2);
 // Peso geométrico:
 // *geomFactor = (sensCabezal1+sensCabezal2)/2; // Lors a ángulo inclinado, el ancho de la lor es el seno del ángulo que se forma.
 // Lo de la sensibilidad sobre el detector al final no va
 *geomFactor = probDetCabezal1*probDetCabezal2;//(sensCabezal1+sensCabezal2)/2 * probDetCabezal1*probDetCabezal2; // Lors a ángulo inclinado, el ancho de la lor es el seno del ángulo que se forma.
 //*geomFactor = 1; // Esto sería sin arc correction.
 // Los cabezales contiguos pueden estar configurados para no adquirir coincidencias. Esto se hace con el parámetro
 // minDiffDetectors. En caso de que se esté en esos casos, fuerzo a cero el factor geométrico:
 if((abs(indexCabezal2-indexCabezal1) < minDiffDetectors) || (abs(indexCabezal2-indexCabezal1) > (this->numCabezales-minDiffDetectors))) // La segunda condición es para los cabezales 5 y 0, que estánc contiguos pero la diferencia es 5. 
   *geomFactor = 0;
 return true;

}