/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * filename: m-intf.h                                                      *
 *                                                                         *
 * UTIL C-source: Medical Image Conversion Utility                         *
 *                                                                         *
 * purpose      : m-intf.c header file                                     *
 *                                                                         *
 * project      : (X)MedCon by Erik Nolf                                   *
 *                                                                         *
 * Modificado por Martin Belzunce para exportar las funciones.
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* $Id: m-intf.h,v 1.28 2010/08/28 23:44:23 enlf Exp $
 */

/*
   Copyright (C) 1997-2010 by Erik Nolf

   This program is free software; you can redistribute it and/or modify it
   under the terms of the GNU General Public License as published by the
   Free Software Foundation; either version 2, or (at your option) any later
   version.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
   Public License for more details.

   You should have received a copy of the GNU General Public License along
   with this program; if not, write to the Free Software Foundation, Inc.,
   59 Place - Suite 330, Boston, MA 02111-1307, USA.  */

#pragma once

/****************************************************************************
                              D E F I N E S 
****************************************************************************/

#define MDC_INTF_SIG    "interfile"

#define MDC_INTF_SUPP_VERS   "3.3"
#define MDC_INTF_SUPP_DATE   "1996:09:24"

#define MDC_INTF_MAXKEYCHARS 256

#define MDC_INTF_UNKNOWN 0

#define MDC_CNTRL_Z 0x0a1a

/* the data types */
#define MDC_INTF_STATIC      1
#define MDC_INTF_DYNAMIC     2
#define MDC_INTF_GATED       3
#define MDC_INTF_TOMOGRAPH   4
#define MDC_INTF_CURVE       5
#define MDC_INTF_ROI         6
#define MDC_INTF_GSPECT      7
#define MDC_INTF_PET      8

#define MDC_INTF_DIALECT_PET 10

/* the process status */
#define MDC_INTF_ACQUIRED      1
#define MDC_INTF_RECONSTRUCTED 2
// DLL export/import declaration: visibility of objects
#ifndef LINK_STATIC
	#ifdef WIN32               // Win32 build
		#ifdef DLL_BUILD    // this applies to DLL building
			#define DLLEXPORT __declspec(dllexport)
		#else                   // this applies to DLL clients/users
			#define DLLEXPORT __declspec(dllimport)
		#endif
		#define DLLLOCAL        // not explicitly export-marked objects are local by default on Win32
	#else
		#ifdef HAVE_GCCVISIBILITYPATCH   // GCC 4.x and patched GCC 3.4 under Linux
			#define DLLEXPORT __attribute__ ((visibility("default")))
			#define DLLLOCAL __attribute__ ((visibility("hidden")))
		#else
			#define DLLEXPORT
			#define DLLLOCAL
		#endif
	#endif
#else                         // static linking
	#define DLLEXPORT
	#define DLLLOCAL
#endif
#ifdef __cplusplus
//extern "C"
#endif

DLLEXPORT typedef struct MdcInterFile_t {

  Int8 DIALECT; 
  int dim_num, dim_found;                           /* for handling dialect */
  int data_type, process_status, pixel_type;
  Uint32 width, height, images_per_dimension, time_slots;
  Uint32 data_offset, data_blocks, imagesize, number_images;
  Uint32 energy_windows, frame_groups, time_windows, detector_heads;
  float pixel_xsize, pixel_ysize;
  float slice_thickness, centre_centre_separation;  /* in [pixels] official */
  float slice_thickness_mm;                         /* in [mm]     dialect  */ 
  float study_duration, image_duration, image_pause, group_pause, ext_rot;
  float procent_cycles_acquired;
  float rescale_slope, rescale_intercept;
  Int8 patient_rot, patient_orient, slice_orient;

} MDC_INTERFILE;

/****************************************************************************
                            F U N C T I O N S
****************************************************************************/

DLLEXPORT int MdcCheckINTF(FILEINFO *fi);
int MdcGetIntfKey(FILE *fp);
void MdcInitIntf(MDC_INTERFILE *intf);
int MdcIsEmptyKeyValue(void);
int MdcIntfIsString(char *string, int key);
int MdcIsArrayKey(void);
int MdcGetMaxIntArrayKey(void);
int MdcGetIntKey(void);
int MdcGetYesNoKey(void);
float MdcGetFloatKey(void);
void MdcGetStrKey(char *str);
void MdcGetSubStrKey(char *str, int n);
void MdcGetDateKey(char *str);
void MdcGetSplitDateKey(Int16 *year, Int16 *month, Int16 *day);
void MdcGetSplitTimeKey(Int16 *hour, Int16 *minute, Int16 *second);
int MdcGetDataType(void);
int MdcGetProcessStatus(void);
int MdcGetPatRotation(void);
int MdcGetPatOrientation(void);
int MdcGetSliceOrient(void);
int MdcGetPatSlOrient(MDC_INTERFILE *intf);
int MdcGetPixelType(void);
int MdcGetRotation(void);
int MdcGetMotion(void);
int MdcGetGSpectNesting(void);
int MdcSpecifyPixelType(MDC_INTERFILE *intf);
char *MdcHandleIntfDialect(FILEINFO *fi, MDC_INTERFILE *intf);
char *MdcReadIntfHeader(FILEINFO *fi, MDC_INTERFILE *intf);
char *MdcReadIntfImages(FILEINFO *fi, MDC_INTERFILE *intf);
const char *MdcReadINTF(FILEINFO *fi);
char *MdcType2Intf(int type);
char *MdcGetProgramDate(void);
char *MdcSetPatRotation(int patient_slice_orient);
char *MdcSetPatOrientation(int patient_slice_orient);
char *MdcCheckIntfDim(FILEINFO *fi);
char *MdcWriteGenImgData(FILEINFO *fi);
char *MdcWriteWindows(FILEINFO *fi);
char *MdcWriteMatrixInfo(FILEINFO *fi, Uint32 img);
char *MdcWriteIntfStatic(FILEINFO *fi);
char *MdcWriteIntfDynamic(FILEINFO *fi);
char *MdcWriteIntfTomo(FILEINFO *fi);
char *MdcWriteIntfGated(FILEINFO *fi);
char *MdcWriteIntfGSPECT(FILEINFO *fi);
/// Agregado por Martin Belzunce:
char *MdcWriteIntfPET(FILEINFO *fi);
char *MdcWriteIntfHeader(FILEINFO *fi);
char *MdcWriteIntfImages(FILEINFO *fi);
const char *MdcWriteINTF(FILEINFO *fi);
