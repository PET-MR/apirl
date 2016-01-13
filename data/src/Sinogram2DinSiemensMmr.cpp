/**
	\file Sinogram2DinSiemensMmr.cpp
	\brief Archivo que contiene la implementación de la clase Sinogram2DinSiemensMmr.

	Este archivo define la clase Sinogram2DinSiemensMmr. Básicamente fija las dimensiones del fov y del scanner, y sobrescribe
	las funciones para obtener las coordenadas de cada LOR respecto del sinograma cilíndrico general.
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2015.02.10
	\version 1.0.0
*/

#include <Sinogram2DinSiemensMmr.h>

/// Size of each crystal element.
const float Sinogram2DinSiemensMmr::crystalElementSize_mm = 4.0891f;
/// Size of each sinogram's bin.
const float Sinogram2DinSiemensMmr::binSize_mm = 4.0891f/2.0f;
/// Depth or length og each crystal.
const float Sinogram2DinSiemensMmr::crystalElementLength_mm = 20;
/// Mean depth of interaction:
const float Sinogram2DinSiemensMmr::meanDOI_mm = 9.6;

Sinogram2DinSiemensMmr::Sinogram2DinSiemensMmr(char* fileHeaderPath): Sinogram2DinCylindrical3Dpet(fileHeaderPath, 297, 328)
{
  radioScanner_mm = 328;
  radioFov_mm = 297;
  float lr;
  // The r value is non linear in the sinogram, because each bin represent one detector element and
  // with the curve of the cylindrical scanner the distance r to the center axis increases with the cos of the bin.
  for(int j = 0; j < numR; j++)
  {
    // ptrRvalues initialization is necesary just one time
    // 1) Get the length on the cylindrical surface for each bin (from x=0 to the center of the crystal element):
    lr = (binSize_mm/2 + binSize_mm*(j-(float)(numR/2)));
    // 2) Now I get the x coordinate for that r.
    ptrRvalues_mm[j] = (radioScanner_mm + meanDOI_mm* cos(lr/radioScanner_mm)) * sin(lr/radioScanner_mm);
  }
}

Sinogram2DinSiemensMmr::Sinogram2DinSiemensMmr(unsigned int nProj, unsigned int nR):Sinogram2DinCylindrical3Dpet(nProj, nR, 297, 328)
{
  float lr;
  radioScanner_mm = 328;
  radioFov_mm = 297;
  numR = nR;
  numProj = nProj;

  // Initialization
  float PhiIncrement = (float)maxAng_deg / numProj;
  // The r value is non linear in the sinogram, because each bin represent one detector element and
  // with the curve of the cylindrical scanner the distance r to the center axis increases with the cos of the bin.
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
	// 1) Get the length on the cylindrical surface for each bin (from x=0 to the center of the crystal element):
	lr = (binSize_mm/2 + binSize_mm*(j-(float)(numR/2)));
	//lr = (binSize_mm/4 + binSize_mm*(j-(float)(numR/2))); // PAreciera andar un poco mejor esta.
	// 2) Now I get the x coordinate for that r.
	ptrRvalues_mm[j] = (radioScanner_mm + meanDOI_mm* cos(lr/radioScanner_mm)) * sin(lr/(radioScanner_mm));
// 	#ifdef __DEBUG__
// 	  printf("\t%f", ptrRvalues_mm[j]);
// 	#endif
      }
      ptrSinogram[i * numR + j] = 0;
    }
  }
}

Sinogram2DinSiemensMmr::~Sinogram2DinSiemensMmr()
{
  /// Limpio la memoria
  
}

/// Constructor de Copia
Sinogram2DinSiemensMmr::Sinogram2DinSiemensMmr(const Sinogram2DinSiemensMmr* srcSinogram2D):Sinogram2DinCylindrical3Dpet((Sinogram2DinCylindrical3Dpet*) srcSinogram2D)
{
  
}

Sinogram2DinSiemensMmr::Sinogram2DinSiemensMmr(const Sinogram2DinSiemensMmr* srcSinogram2D, int indexSubset, int numSubsets):Sinogram2DinCylindrical3Dpet((Sinogram2DinCylindrical3Dpet*) srcSinogram2D, indexSubset, numSubsets)
{
  float lr; 
  radioScanner_mm = 328;
  radioFov_mm = 297;
  // Initialization of the values, that differ slightly from the cylindrical pet:
  // This is done in the sinogram2d and takes into account the subsets.
//   float PhiIncrement = (float)maxAng_deg / numProj;
//   for(int i = 0; i < numProj; i ++)
//   {
//     // Initialization of Phi Values
//     ptrAngValues_deg[i] = i * PhiIncrement;	// Modification now goes from 0, phiincrement, ...180-phiincrement.
//     //ptrAngValues_deg[i] = PhiIncrement/2 + i * PhiIncrement;
//   }
    // The r value is non linear in the sinogram, because each bin represent one detector element and
  // with the curve of the cylindrical scanner the distance r to the center axis increases with the cos of the bin.
  for(int j = 0; j < numR; j++)
  {
    // ptrRvalues initialization is necesary just one time
    // 1) Get the length on the cylindrical surface for each bin (from x=0 to the center of the crystal element):
    lr = (binSize_mm/2 + binSize_mm*(j-(float)(numR/2)));
    // 2) Now I get the x coordinate for that r.
    ptrRvalues_mm[j] = (radioScanner_mm + meanDOI_mm* cos(lr/radioScanner_mm)) * sin(lr/radioScanner_mm);
  }
}

bool Sinogram2DinSiemensMmr::getFovLimits(Line2D lor, Point2D* limitPoint1, Point2D* limitPoint2)
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

bool Sinogram2DinSiemensMmr::getPointsFromLor(int indexProj, int indexR, int indexRingConfig, Point3D* p1, Point3D* p2, float* geomFactor)
{
  float r = this->getRValue(indexR);
  float rad_PhiAngle = this->getAngValue(indexProj) * DEG_TO_RAD;
  float lr = (binSize_mm/2 + binSize_mm*(indexR-(float)(numR/2)));
  float effRadioScanner_mm = (radioScanner_mm + meanDOI_mm* cos(lr/radioScanner_mm));
  float auxValue = sqrt(effRadioScanner_mm * effRadioScanner_mm - r * r);
  *geomFactor = 1;
  p1->X = r * cos(rad_PhiAngle) + sin(rad_PhiAngle) * auxValue;
  p1->Y = r * sin(rad_PhiAngle) - cos(rad_PhiAngle) * auxValue;
  p2->X = r * cos(rad_PhiAngle) - sin(rad_PhiAngle) * auxValue;
  p2->Y = r * sin(rad_PhiAngle) + cos(rad_PhiAngle) * auxValue;
  p1->Z = ptrListZ1_mm[indexRingConfig] - meanDOI_mm/(effRadioScanner_mm) * (ptrListZ2_mm[indexRingConfig]-ptrListZ1_mm[indexRingConfig]);
  p2->Z = ptrListZ2_mm[indexRingConfig] - meanDOI_mm/(effRadioScanner_mm) * (ptrListZ1_mm[indexRingConfig]-ptrListZ2_mm[indexRingConfig]);
  return true;
}
  
  
bool Sinogram2DinSiemensMmr::getPointsFromLor (int indexAng, int indexR, Point2D* p1, Point2D* p2)
{
  float r = this->getRValue(indexR);
  float rad_PhiAngle = this->getAngValue(indexAng) * DEG_TO_RAD;
  float lr = (binSize_mm/2 + binSize_mm*(indexR-(float)(numR/2)));
  float effRadioScanner_mm = (radioScanner_mm + meanDOI_mm* cos(lr/radioScanner_mm));
  float auxValue = sqrt(effRadioScanner_mm * effRadioScanner_mm - r * r);
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

bool Sinogram2DinSiemensMmr::getPointsFromLor (int indexAng, int indexR, Point2D* p1, Point2D* p2, float* geom)
{
  float r = this->getRValue(indexR);
  float rad_PhiAngle = this->getAngValue(indexAng) * DEG_TO_RAD;
  // RadioFov se usa si no se tiene geometrías, en el cylindricalpet se debe utilizar el radioscanner:
  float auxValue = sqrt(radioScanner_mm * radioScanner_mm - r * r);
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