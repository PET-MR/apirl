#include <Geometry.h>


float distBetweenPoints(Point2D point1, Point2D point2)
{
  return sqrtf( (point2.X-point1.X)*(point2.X-point1.X) + (point2.Y-point1.Y)*(point2.Y-point1.Y) );
}

void IntersectionLinePlane(Line3D myLine, Plane myPlane, Point3D* IntersectionPoint)
{
	// Plane A*x+B*y+C*z+D =0
	// Line a = (X-X0)/Vx=(Y-Y0)/Vy=(Z-Z0)/Vz
	// Intersection will be: A*(a*Vx+X0)+B*(a*Vy+Y0)+C*(a*Vz+Z0)+D=0
	// So first I need to calculate a:
	// a = -(A*X0+B*Y0+C*Z0+D)/(A*Vx+B*Vy+C*Vz)
	float a = -(myPlane.A*myLine.P0.X + myPlane.B*myLine.P0.Y + myPlane.C*myLine.P0.Z + myPlane.D) /
		(myPlane.A*myLine.Vx + myPlane.B*myLine.Vy + myPlane.C*myLine.Vz);
	// Then I get the point coordinates from the line equation:
	// (X1,Y1,Z1) = (X0,Y0,Z0) + a * (Vx,Vy,Vz)
	IntersectionPoint->X = myLine.P0.X + a * myLine.Vx;
	IntersectionPoint->Y = myLine.P0.Y + a * myLine.Vy;
	IntersectionPoint->Z = myLine.P0.Z + a * myLine.Vz;
}

// This function calculates the parameters of the planes that limits
// the voxels of a volume in a field of view. As its a volume we have
// planes parallels to X, Y and Z axis. The center of Coordinate System
// is the center of the Field of View. The length of the arrays will be
// Pixels + 1 in each axis. The memory for the arrays must be allocated
// before calling this function.
void CalculatePixelsPlanes(Plane* PlanesX, Plane* PlanesY,Plane* PlanesZ, int SizeX,
					 int SizeY, int SizeZ, float RFOV, float ZFOV)
{
	float stepX = 2 * RFOV / SizeX;
	float stepY = 2 * RFOV / SizeY;
	float stepZ = ZFOV / SizeZ;
	for(int i = 0; i < SizeX + 1; i ++)
	{
		// The planes that define the xlimits of the voxels are X = X0
		PlanesX[i].A = 1;
		PlanesX[i].B = 0;
		PlanesX[i].C = 0;
		PlanesX[i].D = (-RFOV/2) + stepX * i;
	}
	
	for(int i = 0; i < SizeY + 1; i ++)
	{
		// The planes that define the xlimits of the voxels are X = X0
		PlanesY[i].A = 0;
		PlanesY[i].B = 1;
		PlanesY[i].C = 0;
		PlanesY[i].D = (-RFOV/2) + stepY * i;
	}
	
	for(int i = 0; i < SizeZ + 1; i ++)
	{
		// The planes that define the xlimits of the voxels are X = X0
		PlanesX[i].A = 0;
		PlanesX[i].B = 0;
		PlanesX[i].C = 1;
		PlanesX[i].D = (-ZFOV/2) + stepZ * i;
	}
}



void GetPointsFromLOR (float PhiAngle, float r, float Z1, float Z2, float Rscanner, Point3D* P1, Point3D* P2)
{
	float auxValue = sqrt(Rscanner * Rscanner - r * r);
	float rad_PhiAngle = PhiAngle * DEG_TO_RAD;
	P1->X = r * cos(rad_PhiAngle) + sin(rad_PhiAngle) * auxValue;
	P1->Y = r * sin(rad_PhiAngle) - cos(rad_PhiAngle) * auxValue;
	P1->Z = Z1;
	P2->X = r * cos(rad_PhiAngle) - sin(rad_PhiAngle) * auxValue;
	P2->Y = r * sin(rad_PhiAngle) + cos(rad_PhiAngle) * auxValue;
	P2->Z = Z2;
}
/*
void GetPointsFromLOR2 (double PhiAngle, double r, double Z1, double Z2, double Rscanner, Point3D* P1, Point3D* P2)
{
	double auxValue = sqrt(Rscanner * Rscanner - r * r);
	double rad_PhiAngle = PhiAngle * DEG_TO_RAD;
	P1->X = r * cos(rad_PhiAngle) - sin(rad_PhiAngle) * auxValue;
	P1->Y = r * sin(rad_PhiAngle) + cos(rad_PhiAngle) * auxValue;
	P1->Z = Z1;
	P2->X = r * cos(rad_PhiAngle) + sin(rad_PhiAngle) * auxValue;
	P2->Y = r * sin(rad_PhiAngle) - cos(rad_PhiAngle) * auxValue;
	P2->Z = Z2;
}

// Versi�n 2D
void GetPointsFromLOR(double PhiAngle, double r, double Rscanner, Point2D* P1, Point2D* P2)
{
	double auxValue = sqrt(Rscanner * Rscanner - r * r);
	double rad_PhiAngle = PhiAngle * DEG_TO_RAD;
	P1->X = r * cos(rad_PhiAngle) + sin(rad_PhiAngle) * auxValue;
	P1->Y = r * sin(rad_PhiAngle) - cos(rad_PhiAngle) * auxValue;
	P2->X = r * cos(rad_PhiAngle) - sin(rad_PhiAngle) * auxValue;
	P2->Y = r * sin(rad_PhiAngle) + cos(rad_PhiAngle) * auxValue;
}
*/
/// Coordenads para el TGS. Me devuelve el punto medio del colimador.
/// distCentroFrenteDetector: distancia del centro del fov al distCentroFrenteDetector (sería 400).
/// largoCol: largo del colimador.
void GetPointsFromTgsLor (float PhiAngle, float r,  float distCentroFrenteDetector, float largoCol, Point2D* P1, Point2D* P2)
{
	float rad_PhiAngle = PhiAngle * DEG_TO_RAD;
	float X0 = r;	/// Centro en X del colimador.
	float Y0 = -(distCentroFrenteDetector - largoCol/2);	/// Centro en Y del colimador.
    
	P1->X = X0 * cos(rad_PhiAngle) + Y0 * sin(rad_PhiAngle);
	P1->Y = -X0 * sin(rad_PhiAngle) + Y0 * cos(rad_PhiAngle);
	P2->X = X0 * cos(rad_PhiAngle) - Y0 * sin(rad_PhiAngle);
	P2->Y = -X0 * sin(rad_PhiAngle) - Y0 * cos(rad_PhiAngle);
}

/// Coordenadas de una lor del TGS, que tiene en cuenta que para cada colimador puede haber LORs
/// oblicuas. Debe pasarsele como dato la distancia del centro del detector al punto sobre la superficie
/// del detector que toca la lor; y la misma distancia pero sobre la cara exterior del agujero del colimador.
/// Esas dos distancias permiten obtener la inclinación de la LOR.
void GetPointsFromTgsLor (float PhiAngle, float r,  float distCentroFrenteDetector, float largoCol, float offsetDetector, float offsetCaraColimador, Point2D* P1, Point2D* P2)
{
	float rad_PhiAngle = PhiAngle * DEG_TO_RAD;
	/// Para lor oblicua debo obtener un punto sobre el detector, y un punto en el extremo opuesto.
	/// Punto sobre el detector:
	float X0 = r + offsetDetector;	/// Posición en X del punto sobre el detector: r + offsetDetector.
	float Y0 = distCentroFrenteDetector;	/// Coordenada Y sobre el detector.
    /// Punto en cara opuesta. La coordenada Y es la misma pero con signo opuesto, mientras que para la X
	/// la debo proyectar en base a la recta que forma entre (r+OffsetDetector) y (r+OffsetCaraColimador):
	float Y1 = -distCentroFrenteDetector;
	float X1 = X0 + (offsetCaraColimador-offsetDetector) / largoCol * distCentroFrenteDetector * 2;
	
	
	P1->X = X0 * cos(rad_PhiAngle) + Y0 * sin(rad_PhiAngle);
	P1->Y = -X0 * sin(rad_PhiAngle) + Y0 * cos(rad_PhiAngle);
	P2->X = X1 * cos(rad_PhiAngle) + Y1 * sin(rad_PhiAngle);
	P2->Y = -X1 * sin(rad_PhiAngle) + Y1 * cos(rad_PhiAngle);
}

// This function generates a 3d projection of a volume, and stores it in
// a Michelogram passed as a parameter. The Michelogram must be filled with
// zeros before calling this function.
// Image ya tiene que haber sido generada previamente con su tama�o correspondiente.
/*void Geometric3DProjection (Image* image, Michelogram* MyMichelogram, double Rscanner)
{
	/// Obtengo la estructura con el tama�o de la imagen.
	SizeImage sizeImage = image->getSize();
	const unsigned int nPixelsX = sizeImage.nPixelsX;
	const unsigned int nPixelsY = sizeImage.nPixelsY;
	const unsigned int nPixelsZ = sizeImage.nPixelsZ;
	const double stepX = sizeImage.sizePixelX_mm;
	const double stepY = sizeImage.sizePixelY_mm;
	const double stepZ = sizeImage.sizePixelZ_mm;
	const double minValueY = -Rscanner;
	const double stepZMichelogram = MyMichelogram->ZValues[1] - MyMichelogram->ZValues[0];
	const long int pixelsSlice = sizeImage.nPixelsX * sizeImage.nPixelsY;
	float* ptrPixels = image->getPixelsPtr();
	for(unsigned int i = 0; i < nPixelsZ; i++)
	{
		for(unsigned int j = 0; j < nPixelsY; j++)
		{
			for(unsigned int k = 0; k < nPixelsX; k++)
			{
				if(ptrPixels[i * pixelsSlice + j * nPixelsX + k]!=0)
				{
					for(unsigned int l = 0; l < MyMichelogram->NZ; l++)
					{
						int Z1 = l;
						// Nw I calculate Z values projecting the LOr in the ZY axis
						double ProyY = j * stepY  - MyMichelogram->RFOV + stepY/2;
						double Xvalue = k * stepX  - MyMichelogram->RFOV + stepX/2;
						double Zvalue = i * stepZ + stepZ/2; 
						// I have the line that goes from z1 to z2 in z, and from
						// 0 to 2sqrt(R2 - r2). And passes ofr the point (ProyY,Zvol)
						double Z2_double = MyMichelogram->ZValues[Z1] + 2 * Rscanner*(Zvalue-MyMichelogram->ZValues[Z1])/(ProyY-minValueY);
						int Z2 = SearchBin(MyMichelogram->ZValues, MyMichelogram->NZ, Z2_double);
						if(Z2 != -1)
						{
							// Valid Z value
							for(unsigned int m = 0; m < MyMichelogram->NProj; m++)
							{
								double Phi = MyMichelogram->Sinograms2D[0]->getAngValue(m) * DEG_TO_RAD;
								double r = cos(Phi)*Xvalue + sin(Phi)*ProyY;
								unsigned int n = SearchBin(MyMichelogram->Sinograms2D[0]->RValues, MyMichelogram->NR, r);
								if( n != -1)
								{
									// Valid r value
									// Linear interpolation to fill the michelogram
									unsigned int indiceSino1, indiceSino2; //Dos indices para la interoplacion en los Sinos
									if(Z2_double < MyMichelogram->ZValues[Z2])
									{
										indiceSino1 = Z1 + MyMichelogram->NZ * Z2;
										MyMichelogram->Sinograms2D[indiceSino1]->Sinogram[m * MyMichelogram->NR + n] +=
												(1 - (MyMichelogram->ZValues[Z2] - Z2_double) / stepZMichelogram) * ptrPixels[i * pixelsSlice + j * nPixelsX + k];
										if(Z2 != 0)
										{
											indiceSino2 = Z1 + MyMichelogram->NZ * (Z2-1);
											MyMichelogram->Sinograms2D[indiceSino2]->Sinogram[m * MyMichelogram->NR + n] +=
											(1 - (Z2_double - MyMichelogram->ZValues[Z2 - 1]) / stepZMichelogram) * ptrPixels[i * pixelsSlice + j * nPixelsX + k];
										}
										else
										{
											indiceSino2 = 0;
											//MyMichelogram->Sinograms2D[indiceSino2]->Sinogram[m * MyMichelogram->NR + n] +=
											//	(1 - (Z2_double - MyMichelogram->ZValues[Z2]) / stepZMichelogram) * MyVolume->Images2D[i]->Pixels[j * MyVolume->SizeX + k];
									
										}
									}
									else
									{
										if(Z2>=(MyMichelogram->NZ-1))
										{
											indiceSino1 = Z1 + MyMichelogram->NZ * (Z2);
											//MyMichelogram->Sinograms2D[indiceSino1]->Sinogram[m * MyMichelogram->NR + n] +=
											//	MyVolume->Images2D[i]->Pixels[j * MyVolume->SizeX + k];	
											
											
										}
										else
										{
											indiceSino1 = Z1 + MyMichelogram->NZ * (Z2+1);
											MyMichelogram->Sinograms2D[indiceSino1]->Sinogram[m * MyMichelogram->NR + n] +=
												(1 - (MyMichelogram->ZValues[Z2+1] - Z2_double) / stepZMichelogram) * ptrPixels[i * pixelsSlice + j * nPixelsX + k];	
										}
										indiceSino2 = Z1 + MyMichelogram->NZ * Z2;
										MyMichelogram->Sinograms2D[indiceSino2]->Sinogram[m * MyMichelogram->NR + n] +=
											(1 - (Z2_double - MyMichelogram->ZValues[Z2]) / stepZMichelogram) * ptrPixels[i * pixelsSlice + j * nPixelsX + k];						
									}
									if(r < MyMichelogram->Sinograms2D[0]->RValues[n])
									{
											MyMichelogram->Sinograms2D[indiceSino1]->Sinogram[m * MyMichelogram->NR + n] +=
												(1 - (MyMichelogram->Sinograms2D[0]->RValues[n] - r) / stepZMichelogram) * ptrPixels[i * pixelsSlice + j * nPixelsX + k];
											MyMichelogram->Sinograms2D[indiceSino2]->Sinogram[m * MyMichelogram->NR + n] +=
												(1 - (MyMichelogram->Sinograms2D[0]->RValues[n] - r) / stepZMichelogram) * ptrPixels[i * pixelsSlice + j * nPixelsX + k];
											if(n != 0)
											{
												MyMichelogram->Sinograms2D[indiceSino1]->Sinogram[m * MyMichelogram->NR + (n-1)] +=
													(1 - (r - MyMichelogram->Sinograms2D[0]->RValues[n-1]) / stepZMichelogram) * ptrPixels[i * pixelsSlice + j * nPixelsX + k];
												MyMichelogram->Sinograms2D[indiceSino2]->Sinogram[m * MyMichelogram->NR + (n-1)] +=
													(1 - (r - MyMichelogram->Sinograms2D[0]->RValues[n-1]) / stepZMichelogram) * ptrPixels[i * pixelsSlice + j * nPixelsX + k];
											}
									}
									else
									{
										MyMichelogram->Sinograms2D[indiceSino1]->Sinogram[m * MyMichelogram->NR + n] +=
											(1 - (r - MyMichelogram->Sinograms2D[0]->RValues[n]) / stepZMichelogram) * ptrPixels[i * pixelsSlice + j * nPixelsX + k];
										MyMichelogram->Sinograms2D[indiceSino2]->Sinogram[m * MyMichelogram->NR + n] +=
											(1 - (r - MyMichelogram->Sinograms2D[0]->RValues[n]) / stepZMichelogram) * ptrPixels[i * pixelsSlice + j * nPixelsX + k];
										if(n<(MyMichelogram->NR-1))
										{
											MyMichelogram->Sinograms2D[indiceSino1]->Sinogram[m * MyMichelogram->NR + (n + 1)] +=
												(1 - (MyMichelogram->Sinograms2D[0]->RValues[(n + 1)] - r) / stepZMichelogram) * ptrPixels[i * pixelsSlice + j * nPixelsX + k];
											MyMichelogram->Sinograms2D[indiceSino2]->Sinogram[m * MyMichelogram->NR + (n + 1)] +=
												(1 - (MyMichelogram->Sinograms2D[0]->RValues[(n + 1)] - r) / stepZMichelogram) * ptrPixels[i * pixelsSlice + j * nPixelsX + k];
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
}


void Geometric3DProjectionV2 (Image* image, Michelogram* MyMichelogram, double Rscanner)
{
	/// Obtengo la estructura con el tama�o de la imagen.
	SizeImage sizeImage = image->getSize();
	const unsigned int nPixelsX = sizeImage.nPixelsX;
	const unsigned int nPixelsY = sizeImage.nPixelsY;
	const unsigned int nPixelsZ = sizeImage.nPixelsZ;
	const double stepX = sizeImage.sizePixelX_mm;
	const double stepY = sizeImage.sizePixelY_mm;
	const double stepZ = sizeImage.sizePixelZ_mm;
	const double minValueY = -Rscanner;
	const double stepZMichelogram = MyMichelogram->ZValues[1] - MyMichelogram->ZValues[0];
	const long int pixelsSlice = sizeImage.nPixelsX * sizeImage.nPixelsY;
	float* ptrPixels = image->getPixelsPtr();

	// I divided the different axis in mucho more bins, to avoid sampling problems
	for(unsigned int i = 0; i < nPixelsZ; i++)
	{
		for(unsigned int j = 0; j < nPixelsY; j++)
		{
			for(unsigned int k = 0; k < nPixelsX; k++)
			{
				if(ptrPixels[i * pixelsSlice + j * nPixelsX + k]!=0)
				{
					for(unsigned int l = 0; l < (MyMichelogram->NZ * 4); l++)
					{
						int Z1 = floor((double)l/4);
						double Z1_double = l * stepZMichelogram/4; //MyMichelogram->ZValues[Z1]
						// Nw I calculate Z values projecting the LOr in the ZY axis
						double ProyY = j * stepY  - MyMichelogram->RFOV + stepY/2;
						double Xvalue = k * stepX  - MyMichelogram->RFOV + stepX/2;
						double Zvalue = i * stepZ + stepZ/2; 
						// I have the line that goes from z1 to z2 in z, and from
						// 0 to 2sqrt(R2 - r2). And passes ofr the point (ProyY,Zvol)
						double Z2_double = MyMichelogram->ZValues[Z1] + 2 * Rscanner*(Zvalue-Z1_double)/(ProyY-minValueY);
						int Z2 = SearchBin(MyMichelogram->ZValues, MyMichelogram->NZ, Z2_double);
						if(Z2 != -1)
						{
							// Valid Z value
							for(unsigned int m = 0; m < MyMichelogram->NProj; m++)
							{
								//unsigned int indicePhi = floor((double)m/4);
								double Phi = (MyMichelogram->Sinograms2D[0]->PhiValues[m] ) * DEG_TO_RAD;
								double r = cos(Phi)*Xvalue + sin(Phi)*ProyY;
								unsigned int n = SearchBin(MyMichelogram->Sinograms2D[0]->RValues, MyMichelogram->NR, r);
								if( n != -1)
								{
									MyMichelogram->Sinograms2D[Z1 + MyMichelogram->NZ * Z2]->Sinogram[m * MyMichelogram->NR + n] +=
										ptrPixels[i * pixelsSlice + j * nPixelsX + k];
										
								}
							}
						}
					}
				}
			}
		}
	}
}
*/