#ifndef _READPARAMETERS_H
#define	_READPARAMETERS_H

#include <stdio.h>
#include <iostream>
#include <string.h>
#include <Michelogram.h>
#include <Mlem.h>
#include <Mlem2d.h>
#include <Mlem2dMultiple.h>
#include <Mlem2dTgs.h>
#include <MlemSinogram3d.h>
#include <Sinograms2DinCylindrical3Dpet.h>
#include <ParametersFile.h>
#include <Projector.h>
#include <SiddonProjector.h>
#include <RotationBasedProjector.h>
#include <ConeOfResponseProjector.h>
#include <ConeOfResponseWithPenetrationProjector.h>
#include <Images.h>
#include <Geometry.h>
#include <string>

#define FIXED_KEYS 5

/* Encabezados de Funciones relacioandas con la carga de parámetros del Mlem */
int getSaveIntermidiateIntervals (string mlemFilename, string cmd, int* saveIterationInterval, bool* saveIntermediateData);
int getSensitivityFromFile (string mlemFilename, string cmd, bool* bSensitivityFromFile, string* sensitivityFilename);
int getProjectorBackprojectorNames(string mlemFilename, string cmd, string* strForwardprojector, string* strBackprojector);
int getRotationBasedProjectorParameters(string mlemFilename, string cmd, RotationBasedProjector::InterpolationMethods *interpMethod);
int getCylindricalScannerParameters(string mlemFilename, string cmd, float* radiusFov_mm, float* zFov_mm, float* radiusScanner_mm);
int getNumberOfSubsets(string mlemFilename, string cmd, float* numberOfSubsets);
int getArPetParameters(string mlemFilename, string cmd, float* radiusFov_mm, float* zFov_mm, float* blindArea_mm, int* minDiffDetectors);
int getCorrectionSinogramNames(string mlemFilename, string cmd, string* acfFilename, string* estimatedRandomsFilename, string* estimatedScatterFilename);
int getNormalizationSinogramName(string mlemFilename, string cmd, string* normFilename);
#endif
