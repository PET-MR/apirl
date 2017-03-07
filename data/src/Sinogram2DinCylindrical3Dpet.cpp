/**
	\file Sinogram2DinCylindrical3Dpet.cpp
	\brief Archivo que contiene la implementación de la clase Sinogram2D.

	Este archivo define la clase Sinogram2DinCylindrical3Dpet. La misma define un sinograma de dos dimensiones
	que se encuentra en un scanner cilíndrico con adquisición 3D, por lo que tiene información sobre
	la combinación de anillos a la que pertenece y si son más de una porque hay polar mashing.
	\todo Implementar proyectores.
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.10.01
	\version 1.0.0
*/

#include <Sinogram2DinCylindrical3Dpet.h>

Sinogram2DinCylindrical3Dpet::Sinogram2DinCylindrical3Dpet(char* fileHeaderPath, float rFov_mm, float rScanner_mm):Sinogram2D(rFov_mm)
{
  radioScanner_mm = rScanner_mm;
  numZ = 0;
  ptrListRing1 = NULL;
  ptrListRing2 = NULL;
  ptrListZ1_mm = NULL;
  ptrListZ2_mm = NULL;
  this->readFromInterfile(fileHeaderPath);
}

Sinogram2DinCylindrical3Dpet::Sinogram2DinCylindrical3Dpet(unsigned int nProj, unsigned int nR, float rFov_mm, float rScanner_mm):Sinogram2D(rFov_mm)
{
  radioFov_mm = rFov_mm;
  numR = nR;
  numProj = nProj;
  numZ = 0;
  radioScanner_mm = rScanner_mm;
  ptrListRing1 = NULL;
  ptrListRing2 = NULL;
  ptrListZ1_mm = NULL;
  ptrListZ2_mm = NULL;
  // Allocates Memory for th Sinogram
  ptrSinogram = (float*) realloc(ptrSinogram, numProj*numR*sizeof(float));
  // Allocates Memory for the value's vectors
  ptrAngValues_deg = (float*) realloc(ptrAngValues_deg, numProj*sizeof(float));
  ptrRvalues_mm = (float*) realloc(ptrRvalues_mm, numR*sizeof(float));
  // Initialization
  float RIncrement = (2 * radioFov_mm) / numR;
  float PhiIncrement = (float)maxAng_deg / numProj;
  for(int i = 0; i < numProj; i ++)
  {
    // Initialization of Phi Values
    ptrAngValues_deg[i] = i * PhiIncrement;	// Modification now goes from 0, phiincrement, ...180-phiincrement.
    //ptrAngValues_deg[i] = PhiIncrement/2 + i * PhiIncrement;
    for(int j = 0; j < numR; j++)
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

Sinogram2DinCylindrical3Dpet::~Sinogram2DinCylindrical3Dpet()
{
  /// Limpio la memoria
  if(ptrListRing1!=NULL)
    free(ptrListRing1);
  if(ptrListRing2!=NULL)
    free(ptrListRing2);
  if(ptrListZ1_mm!=NULL)
    free(ptrListZ1_mm);
  if(ptrListZ2_mm!=NULL)
    free(ptrListZ2_mm);
}

/// Constructor de Copia
Sinogram2DinCylindrical3Dpet::Sinogram2DinCylindrical3Dpet(const Sinogram2DinCylindrical3Dpet* srcSinogram2D):Sinogram2D((Sinogram2D*) srcSinogram2D)
{
  ptrListRing1 = NULL;
  ptrListRing2 = NULL;
  ptrListZ1_mm = NULL;
  ptrListZ2_mm = NULL;
  // Ahora las variables propias de esta clase derivada:
  radioScanner_mm = srcSinogram2D->radioScanner_mm;
  numZ = srcSinogram2D->numZ;
  ptrListRing1 = (int*) malloc(sizeof(int)*numZ);
  memcpy(ptrListRing1, srcSinogram2D->ptrListRing1, sizeof(int)*numZ);
  ptrListRing2 = (int*) malloc(sizeof(int)*numZ);
  memcpy(ptrListRing2, srcSinogram2D->ptrListRing2, sizeof(int)*numZ);
  ptrListZ1_mm = (float*) malloc(sizeof(float)*numZ);
  memcpy(ptrListZ1_mm, srcSinogram2D->ptrListZ1_mm, sizeof(float)*numZ);
  ptrListZ2_mm = (float*) malloc(sizeof(float)*numZ);
  memcpy(ptrListZ2_mm, srcSinogram2D->ptrListZ2_mm, sizeof(float)*numZ);
  
  /*radioFov_mm = srcSinogram2D->radioFov_mm;
  numR = srcSinogram2D->numR;
  numProj = srcSinogram2D->numProj;
  radioScanner_mm = srcSinogram2D->radioScanner_mm;
  // Allocates Memory for th Sinogram and copy data
  ptrSinogram = (float*) malloc(numProj*numR*sizeof(float));
  memcpy(ptrSinogram, srcSinogram2D->ptrSinogram, numProj*numR*sizeof(float));
  // Allocates cand copy Memory for the value's vectors
  ptrAngValues_deg = (float*) malloc(numProj*sizeof(float));
  memcpy(ptrAngValues_deg, srcSinogram2D->ptrAngValues_deg, numProj*sizeof(float));
  ptrRvalues_mm = (float*) malloc(numR*sizeof(float));
  memcpy(ptrRvalues_mm, srcSinogram2D->ptrRvalues_mm, numR*sizeof(float));
  // Ahora las variables para el ring3d
  numZ = srcSinogram2D->numZ;
  ptrListRing1 = (int*) malloc(sizeof(int)*numZ);
  memcpy(ptrListRing2, srcSinogram2D->ptrListRing1, sizeof(int)*numZ);
  ptrListRing2 = (int*) malloc(sizeof(int)*numZ);
  memcpy(ptrListRing2, srcSinogram2D->ptrListRing2, sizeof(int)*numZ);*/
}

Sinogram2DinCylindrical3Dpet::Sinogram2DinCylindrical3Dpet(const Sinogram2DinCylindrical3Dpet* srcSinogram2D, int indexSubset, int numSubsets):Sinogram2D((Sinogram2D*) srcSinogram2D, indexSubset, numSubsets)
{
  ptrListRing1 = NULL;
  ptrListRing2 = NULL;
  ptrListZ1_mm = NULL;
  ptrListZ2_mm = NULL;
  // Ahora las variables propias de esta clase derivada:
  radioScanner_mm = srcSinogram2D->radioScanner_mm;
  numZ = srcSinogram2D->numZ;
  ptrListRing1 = (int*) malloc(sizeof(int)*numZ);
  memcpy(ptrListRing1, srcSinogram2D->ptrListRing1, sizeof(int)*numZ);
  ptrListRing2 = (int*) malloc(sizeof(int)*numZ);
  memcpy(ptrListRing2, srcSinogram2D->ptrListRing2, sizeof(int)*numZ);
  ptrListZ1_mm = (float*) malloc(sizeof(float)*numZ);
  memcpy(ptrListZ1_mm, srcSinogram2D->ptrListZ1_mm, sizeof(float)*numZ);
  ptrListZ2_mm = (float*) malloc(sizeof(float)*numZ);
  memcpy(ptrListZ2_mm, srcSinogram2D->ptrListZ2_mm, sizeof(float)*numZ);
  /*radioFov_mm = srcSinogram2D->radioFov_mm;
  numR = srcSinogram2D->numR;
  numProj = srcSinogram2D->numProj;
  radioScanner_mm = srcSinogram2D->radioScanner_mm;
  // Allocates Memory for th Sinogram and copy data
  ptrSinogram = (float*) malloc(numProj*numR*sizeof(float));
  memcpy(ptrSinogram, srcSinogram2D->ptrSinogram, numProj*numR*sizeof(float));
  // Allocates cand copy Memory for the value's vectors
  ptrAngValues_deg = (float*) malloc(numProj*sizeof(float));
  memcpy(ptrAngValues_deg, srcSinogram2D->ptrAngValues_deg, numProj*sizeof(float));
  ptrRvalues_mm = (float*) malloc(numR*sizeof(float));
  memcpy(ptrRvalues_mm, srcSinogram2D->ptrRvalues_mm, numR*sizeof(float));
  // Ahora las variables para el ring3d
  numZ = srcSinogram2D->numZ;
  ptrListRing1 = (int*) malloc(sizeof(int)*numZ);
  memcpy(ptrListRing2, srcSinogram2D->ptrListRing1, sizeof(int)*numZ);
  ptrListRing2 = (int*) malloc(sizeof(int)*numZ);
  memcpy(ptrListRing2, srcSinogram2D->ptrListRing2, sizeof(int)*numZ);*/
}


void Sinogram2DinCylindrical3Dpet::setMultipleRingConfig(int nZ, int* listRing1, int* listRing2, float* listZ1, float* listZ2)
{
  numZ = nZ;
  /* La lista de anillos la hago a través de una copia de vectores. */  
  ptrListRing1 = (int*)malloc(sizeof(int) * numZ);
  ptrListRing2 = (int*)malloc(sizeof(int) * numZ);
  ptrListZ1_mm = (float*)malloc(sizeof(float) * numZ);
  ptrListZ2_mm = (float*)malloc(sizeof(float) * numZ);
  memcpy(ptrListRing1, listRing1, sizeof(float) * numZ);
  memcpy(ptrListRing2, listRing2, sizeof(float) * numZ);
  memcpy(ptrListZ1_mm, listZ1, sizeof(float) * numZ);
  memcpy(ptrListZ2_mm, listZ2, sizeof(float) * numZ);
}

void Sinogram2DinCylindrical3Dpet::copyMultipleRingConfig(Sinogram2DinCylindrical3Dpet* srcSinogram2D)
{
  numZ = srcSinogram2D->numZ;
  /* La lista de anillos la hago a través de una copia de vectores. */  
  ptrListRing1 = (int*)realloc(ptrListRing1, sizeof(int) * numZ);
  ptrListRing2 = (int*)realloc(ptrListRing2, sizeof(int) * numZ);
  ptrListZ1_mm = (float*)realloc(ptrListZ1_mm, sizeof(float) * numZ);
  ptrListZ2_mm = (float*)realloc(ptrListZ2_mm, sizeof(float) * numZ);
  memcpy(ptrListRing1, srcSinogram2D->ptrListRing1, sizeof(float) * numZ);
  memcpy(ptrListRing2, srcSinogram2D->ptrListRing2, sizeof(float) * numZ);
  memcpy(ptrListZ1_mm, srcSinogram2D->ptrListZ1_mm, sizeof(float) * numZ);
  memcpy(ptrListZ2_mm, srcSinogram2D->ptrListZ2_mm, sizeof(float) * numZ);
}

bool Sinogram2DinCylindrical3Dpet::getFovLimits(Line2D lor, Point2D* limitPoint1, Point2D* limitPoint2)
{
  /// El FoV de un sinograma2D genérico es un círculo de radio radioFov_mm. Para obtener la intersección
  /// de una recta con el Fov debo resolver la siguiente ecuación:
  /// (X0+alpha*Vx).^2+(Y0+alpha*Vy).^2=RFOV.^2
  /// alpha = (-2*(Vx+Vy)  +  sqrt(4*Vx^2*(1-c)+4*Vy^2*(1-c) + 8(Vx+Vy)))/(2*(Vx^2+Vy^2))
  //float c = LOR.P0.X*LOR.P0.X + LOR.P0.Y*LOR.P0.Y - rFov*rFov;
  float SegundoTermino = sqrt( 4 * (lor.Vx*lor.Vx*(this->radioFov_mm * this->radioFov_mm - lor.P0.Y*lor.P0.Y)
	  + lor.Vy*lor.Vy * (this->radioFov_mm*this->radioFov_mm-lor.P0.X*lor.P0.X)) + 8*lor.Vx*lor.P0.X*lor.Vy*lor.P0.Y);
  if (SegundoTermino == 0)
  {
	// Raíz doble. Toca solo en un punto al FoV, o sea es tangente. No me interesa. Salgo
	return false;
  }
  else if (SegundoTermino != SegundoTermino)
  {
	// Esta condición solo se cumple para NaN, osea cuando para la raíz de un número negativo.
	// Por lo tanto no corta el Fov.
	return false;
  }
  /// Obtengo los valores de alpha donde se intersecciona la recta con la circunferencia.
  /// Como la debería cruzar en dos puntos hay dos soluciones.
  float alpha_xy_1 = (-2*(lor.Vx*lor.P0.X+lor.Vy*lor.P0.Y) + SegundoTermino)/(2*(lor.Vx*lor.Vx+lor.Vy*lor.Vy));
  float alpha_xy_2 = (-2*(lor.Vx*lor.P0.X+lor.Vy*lor.P0.Y) - SegundoTermino)/(2*(lor.Vx*lor.Vx+lor.Vy*lor.Vy));
  /// Ahora calculo los dos puntos (X,Y)
  limitPoint1->X = lor.P0.X + alpha_xy_1*lor.Vx;
  limitPoint1->Y = lor.P0.Y + alpha_xy_1*lor.Vy;
  limitPoint2->X = lor.P0.X + alpha_xy_2*lor.Vx;
  limitPoint1->Y = lor.P0.Y + alpha_xy_2*lor.Vy;
  return true;
}

bool Sinogram2DinCylindrical3Dpet::getPointsFromLor(int indexProj, int indexR, int indexRingConfig, Point3D* p1, Point3D* p2, float* geomFactor)
{
  float r = this->getRValue(indexR);
  float rad_PhiAngle = this->getAngValue(indexProj) * DEG_TO_RAD;
  float auxValue = sqrt(this->radioScanner_mm * this->radioScanner_mm - r * r);
  *geomFactor = 1;
  p1->X = r * cos(rad_PhiAngle) + sin(rad_PhiAngle) * auxValue;
  p1->Y = r * sin(rad_PhiAngle) - cos(rad_PhiAngle) * auxValue;
  p2->X = r * cos(rad_PhiAngle) - sin(rad_PhiAngle) * auxValue;
  p2->Y = r * sin(rad_PhiAngle) + cos(rad_PhiAngle) * auxValue;
  p1->Z = ptrListZ1_mm[indexRingConfig];
  p2->Z = ptrListZ2_mm[indexRingConfig];
  return true;
}
  
  
bool Sinogram2DinCylindrical3Dpet::getPointsFromLor (int indexAng, int indexR, Point2D* p1, Point2D* p2)
{
  float r = this->getRValue(indexR);
  float rad_PhiAngle = this->getAngValue(indexAng) * DEG_TO_RAD;
  float auxValue = sqrt(this->radioScanner_mm * this->radioScanner_mm - r * r);
  if(r > radioFov_mm)
  {
    // El r no puede ser mayor que el rfov:
    return false;
  }
  p1->X = r * cos(rad_PhiAngle) + sin(rad_PhiAngle) * auxValue;
  p1->Y = r * sin(rad_PhiAngle) - cos(rad_PhiAngle) * auxValue;
  p2->X = r * cos(rad_PhiAngle) - sin(rad_PhiAngle) * auxValue;
  p2->Y = r * sin(rad_PhiAngle) + cos(rad_PhiAngle) * auxValue;
  return true;
}

bool Sinogram2DinCylindrical3Dpet::getPointsFromLor (int indexAng, int indexR, Point2D* p1, Point2D* p2, float* geom)
{
  float r = this->getRValue(indexR);
  float rad_PhiAngle = this->getAngValue(indexAng) * DEG_TO_RAD;
  // RadioFov se usa si no se tiene geometrías, en el cylindricalpet se debe utilizar el radioscanner:
  float auxValue = sqrt(this->radioScanner_mm * this->radioScanner_mm - r * r);
  *geom = 1;
  if(r > radioFov_mm)
  {
    // El r no puede ser mayor que el rfov:
    return false;
  }
  p1->X = r * cos(rad_PhiAngle) + sin(rad_PhiAngle) * auxValue;
  p1->Y = r * sin(rad_PhiAngle) - cos(rad_PhiAngle) * auxValue;
  p2->X = r * cos(rad_PhiAngle) - sin(rad_PhiAngle) * auxValue;
  p2->Y = r * sin(rad_PhiAngle) + cos(rad_PhiAngle) * auxValue;
  return true;
}

bool Sinogram2DinCylindrical3Dpet::getPointsFromOverSampledLor (int indexAng, int indexR, int indexSubsample, int numSubsamples, Point2D* p1, Point2D* p2, float* geomFactor)
{
  float r = this->getRValue(indexR);
  float rad_PhiAngle = this->getAngValue(indexAng) * DEG_TO_RAD;
  float incrementR = this->getDeltaR(indexAng, indexR);
  r = r - (incrementR/2) + incrementR/numSubsamples * indexSubsample + (incrementR/(2*numSubsamples));
  float auxValue = sqrt(this->radioScanner_mm * this->radioScanner_mm - r * r);
  *geomFactor = 1;
  p1->X = r * cos(rad_PhiAngle) + sin(rad_PhiAngle) * auxValue;
  p1->Y = r * sin(rad_PhiAngle) - cos(rad_PhiAngle) * auxValue;
  p2->X = r * cos(rad_PhiAngle) - sin(rad_PhiAngle) * auxValue;
  p2->Y = r * sin(rad_PhiAngle) + cos(rad_PhiAngle) * auxValue;
  return true;
}

bool Sinogram2DinCylindrical3Dpet::getPointsFromOverSampledLor (int indexAng, int indexR, int indexSubsample, int numSubsamples, int indexRingConfig, float ringWidth_mm, int indexAxialSubsample, int numAxialSubsamples, Point3D* p1, Point3D* p2, float* geomFactor)
{
  float r = this->getRValue(indexR);
  float rad_PhiAngle = this->getAngValue(indexAng) * DEG_TO_RAD;
  float incrementR = this->getDeltaR(indexAng, indexR);
  float deltaZ = ringWidth_mm / numAxialSubsamples; // delta used for sampling the oversampled ring;
  r = r - (incrementR/2) + incrementR/numSubsamples * indexSubsample + (incrementR/(2*numSubsamples));
  float auxValue = sqrt(this->radioScanner_mm * this->radioScanner_mm - r * r);
  *geomFactor = 1;
  p1->X = r * cos(rad_PhiAngle) + sin(rad_PhiAngle) * auxValue;
  p1->Y = r * sin(rad_PhiAngle) - cos(rad_PhiAngle) * auxValue;
  p2->X = r * cos(rad_PhiAngle) - sin(rad_PhiAngle) * auxValue;
  p2->Y = r * sin(rad_PhiAngle) + cos(rad_PhiAngle) * auxValue;
  p1->Z = ptrListZ1_mm[indexRingConfig]-ringWidth_mm/2 +deltaZ/2+indexAxialSubsample*deltaZ;  // ptrListZ1_mm[indexRingConfig] is the coordinate for this ring. For the voersampled version I need samples between: ptrListZ1_mm[indexRingConfig]-ringWidth_mm/2 and ptrListZ1_mm[indexRingConfig]+ringWidth_mm/2
  p2->Z = ptrListZ2_mm[indexRingConfig]-ringWidth_mm/2 +deltaZ/2+indexAxialSubsample*deltaZ;
  return true;
}